# -*- coding: utf-8 -*-
"""
DeFiLlama TVL MCP (FastMCP) — efficient & incremental

Tools:
  - resolve_protocol(project_hint: str, ttl_hours: int = 24) -> dict
  - refresh_protocol(slug: str) -> dict
  - event_window(slug: str, event_time_utc: str, pre_days: int = 7, post_days: int = 7) -> dict
  - health() -> dict

Design goals:
  * Minimize API calls with:
      - Metadata TTL in SQLite (skip refresh if fresh)
      - Parquet TVL series incremental merge (skip identical writes)
      - Shared HTTP session with retries/backoff
  * Keep pure helpers for easy testing.
"""

from __future__ import annotations
import os
import re
import json
import time
import math
import sqlite3
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd

try:
    from fastmcp import FastMCP
except Exception:
    from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
LLAMA_BASE = "https://api.llama.fi"
DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
TVL_DIR = DATA_DIR / "tvl"
DB_PATH = DATA_DIR / "defillama_cache.sqlite"

DATA_DIR.mkdir(parents=True, exist_ok=True)
TVL_DIR.mkdir(parents=True, exist_ok=True)

HTTP_TIMEOUT = int(os.getenv("LLAMA_HTTP_TIMEOUT", "45"))
RETRY_TOTAL = int(os.getenv("LLAMA_HTTP_RETRIES", "5"))
BASE_SLEEP = float(os.getenv("LLAMA_BASE_SLEEP", "0.35"))
UA = os.getenv("LLAMA_UA", "defillama-mcp/1.2")

log = logging.getLogger("defillama_mcp")
logging.basicConfig(level=logging.INFO)

mcp = FastMCP("defillama_tvl_mcp")

UTC = timezone.utc

# ---------------------------------------------------------------------
# HTTP session with retries/backoff
# ---------------------------------------------------------------------
def _session() -> requests.Session:
    from urllib3.util.retry import Retry
    from requests.adapters import HTTPAdapter

    s = requests.Session()
    s.headers.update({"User-Agent": UA})
    retry = Retry(
        total=RETRY_TOTAL,
        read=RETRY_TOTAL,
        connect=RETRY_TOTAL,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s

HTTP = _session()

def _sleep():
    # Small polite delay to avoid hammering
    time.sleep(BASE_SLEEP)

# ---------------------------------------------------------------------
# SQLite (protocol metadata + processed events)
# ---------------------------------------------------------------------
def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS protocols (
        slug TEXT PRIMARY KEY,
        name TEXT,
        symbol TEXT,
        cmc_id TEXT,
        gecko_id TEXT,
        categories TEXT,
        chains TEXT,
        governance_hint TEXT,
        last_refreshed_at INTEGER
    );
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS processed_events (
        protocol_slug TEXT,
        event_id TEXT,
        event_time_utc TEXT,
        pre_days INTEGER,
        post_days INTEGER,
        PRIMARY KEY (protocol_slug, event_id, pre_days, post_days)
    );
    """)
    conn.commit()
    return conn

def _now_ts() -> int:
    return int(time.time())

# ---------------------------------------------------------------------
# Llama API helpers
# ---------------------------------------------------------------------
def _get_json(url: str, params: Optional[dict] = None) -> Any:
    _sleep()
    r = HTTP.get(url, params=params or {}, timeout=HTTP_TIMEOUT)
    if r.status_code != 200:
        # Surface concise error (avoid uncontrolled loops)
        raise RuntimeError(f"GET {url} -> {r.status_code}: {r.text[:200]}")
    return r.json()

def fetch_protocols_list() -> pd.DataFrame:
    """Fetch the full protocols list; normalize into a compact frame."""
    data = _get_json(f"{LLAMA_BASE}/protocols")
    rows: List[Dict[str, Any]] = []
    for d in data:
        rows.append({
            "slug": d.get("slug") or (d.get("name", "").lower().replace(" ", "-") if d.get("name") else None),
            "name": d.get("name"),
            "symbol": d.get("symbol"),
            "cmc_id": str(d.get("cmcId")) if d.get("cmcId") not in (None, "") else None,
            "gecko_id": d.get("gecko_id"),
            "categories": ",".join(d.get("category", [])) if isinstance(d.get("category"), list) else d.get("category"),
            "chains": ",".join(d.get("chains", [])) if isinstance(d.get("chains"), list) else None,
            "governance_hint": json.dumps({"token_symbol": d.get("symbol")}),
        })
    return pd.DataFrame([r for r in rows if r["slug"]])

def refresh_protocols_cache(ttl_hours: int = 24) -> int:
    """
    Refresh metadata cache if expired. Returns number of rows upserted.
    TTL prevents unnecessary re-downloads within the TTL window.
    """
    with _db() as conn:
        cur = conn.execute("SELECT MAX(last_refreshed_at) FROM protocols;")
        row = cur.fetchone()
        last = int(row[0]) if row and row[0] is not None else 0
        if last and (_now_ts() - last) < ttl_hours * 3600:
            # Still fresh; skip network
            return 0

    df = fetch_protocols_list()
    now_ts = _now_ts()
    with _db() as conn:
        for _, r in df.iterrows():
            conn.execute("""
                INSERT INTO protocols(slug,name,symbol,cmc_id,gecko_id,categories,chains,governance_hint,last_refreshed_at)
                VALUES(?,?,?,?,?,?,?,?,?)
                ON CONFLICT(slug) DO UPDATE SET
                  name=excluded.name,
                  symbol=excluded.symbol,
                  cmc_id=excluded.cmc_id,
                  gecko_id=excluded.gecko_id,
                  categories=excluded.categories,
                  chains=excluded.chains,
                  governance_hint=excluded.governance_hint,
                  last_refreshed_at=excluded.last_refreshed_at;
            """, (r["slug"], r["name"], r["symbol"], r["cmc_id"], r["gecko_id"],
                  r["categories"], r["chains"], r["governance_hint"], now_ts))
        conn.commit()
    return len(df)

def _slug_exists(slug: str) -> bool:
    with _db() as conn:
        cur = conn.execute("SELECT 1 FROM protocols WHERE slug=? LIMIT 1;", (slug,))
        return cur.fetchone() is not None

def guess_slug_by_space(space: str) -> Optional[str]:
    """
    Heuristic match for Snapshot spaces:
      1) '<slug>-snapshot.eth' -> '<slug>' if exists.
      2) Fuzzy containment over slug/name/symbol.
    """
    s = space.strip().lower()
    m = re.match(r"^([a-z0-9-]+)-snapshot\.eth$", s)
    if m:
        base = m.group(1)
        if _slug_exists(base):
            return base

    with _db() as conn:
        cur = conn.execute("SELECT slug,name,symbol FROM protocols;")
        best = None
        s_norm = s.replace(".eth", "").replace(" ", "").replace("-", "")
        for slug, name, symbol in cur.fetchall():
            cands = [slug or "", name or "", symbol or ""]
            score = 0
            for c in cands:
                c0 = c.lower().replace(" ", "").replace("-", "")
                if s_norm in c0 or c0 in s_norm:
                    score += 1
            if score and (best is None or score > best[0]):
                best = (score, slug)
        return best[1] if best else None

# ---------------------------------------------------------------------
# TVL series helpers (incremental Parquet)
# ---------------------------------------------------------------------
def fetch_protocol_detail(slug: str) -> Dict[str, Any]:
    """Fetch full protocol detail JSON (contains 'tvl' or 'chainTvls' trees)."""
    return _get_json(f"{LLAMA_BASE}/protocol/{slug}")

def _pairs_from_detail(detail: Dict[str, Any]) -> List[Tuple[int, float]]:
    """Extract (unix_ts, total_tvl_usd) pairs from detail JSON."""
    def _extract_pairs(arr):
        out: List[Tuple[int, float]] = []
        for x in arr:
            if isinstance(x, dict) and "date" in x and "totalLiquidityUSD" in x:
                try:
                    out.append((int(x["date"]), float(x["totalLiquidityUSD"])))
                except Exception:
                    pass
        return out

    if "tvl" in detail and isinstance(detail["tvl"], list):
        pairs = _extract_pairs(detail["tvl"])
    else:
        pairs_map: Dict[int, float] = {}
        chain_tvls = (detail.get("chainTvls") or detail.get("chainTvlsCharts") or {})
        for _, chain_obj in (chain_tvls or {}).items():
            for x in chain_obj.get("tvl", []):
                if isinstance(x, dict) and "date" in x and "totalLiquidityUSD" in x:
                    try:
                        ts = int(x["date"])
                        pairs_map[ts] = pairs_map.get(ts, 0.0) + float(x["totalLiquidityUSD"])
                    except Exception:
                        continue
        pairs = sorted(pairs_map.items(), key=lambda t: t[0])
    return pairs

def tvl_to_df(pairs: List[Tuple[int, float]]) -> pd.DataFrame:
    if not pairs:
        return pd.DataFrame(columns=["date", "tvl"])
    df = pd.DataFrame(pairs, columns=["unix", "tvl"])
    df["date"] = pd.to_datetime(df["unix"], unit="s", utc=True).tz_convert(None)
    return df.drop(columns=["unix"]).sort_values("date")[["date", "tvl"]]

def _tvl_path(slug: str) -> Path:
    return TVL_DIR / f"{slug}.parquet"

def refresh_tvl_cache(slug: str) -> Dict[str, Any]:
    """
    Incrementally refresh TVL parquet for the slug, returning counts.
    If the merged data is identical to the existing file, skip writing (I/O save).
    """
    detail = fetch_protocol_detail(slug)  # full JSON (Llama lacks partial range API)
    new_df = tvl_to_df(_pairs_from_detail(detail))
    if new_df.empty:
        return {"slug": slug, "rows_before": 0 if not _tvl_path(slug).exists() else len(pd.read_parquet(_tvl_path(slug))),
                "rows_added": 0, "rows_after": 0, "written": False}

    fpath = _tvl_path(slug)
    if fpath.exists():
        old = pd.read_parquet(fpath)
        merged = (pd.concat([old, new_df], ignore_index=True)
                  .drop_duplicates(subset=["date"])
                  .sort_values("date"))
        # If nothing changed, don't rewrite the file
        if len(merged) == len(old) and old["date"].iloc[-1] == merged["date"].iloc[-1] and math.isclose(
            float(old["tvl"].iloc[-1]), float(merged["tvl"].iloc[-1]), rel_tol=1e-12, abs_tol=1e-9
        ):
            return {"slug": slug, "rows_before": len(old), "rows_added": 0, "rows_after": len(old), "written": False}
    else:
        merged = new_df

    merged.to_parquet(fpath, index=False)
    rows_before = 0 if not fpath.exists() else len(pd.read_parquet(fpath))
    return {"slug": slug, "rows_before": rows_before, "rows_added": len(merged) - rows_before,
            "rows_after": len(merged), "written": True}

def load_tvl(slug: str) -> pd.DataFrame:
    fpath = _tvl_path(slug)
    if not fpath.exists():
        raise FileNotFoundError(f"TVL cache not found for {slug}. Call refresh first.")
    df = pd.read_parquet(fpath)
    return df.sort_values("date").dropna()

# ---------------------------------------------------------------------
# Event window stats (pure)
# ---------------------------------------------------------------------
@dataclass
class WindowResult:
    n_days: int
    tvl_total: float
    tvl_avg: float

def _window_slice(df: pd.DataFrame, t0: pd.Timestamp, pre_days: int, post_days: int) -> Dict[str, pd.DataFrame]:
    df = df.copy().sort_values("date")
    t0 = pd.to_datetime(t0, utc=True).tz_convert(None)
    pre_from = t0 - timedelta(days=pre_days)
    pre_to = t0 - timedelta(days=1)
    post_from = t0 + timedelta(days=1)
    post_to = t0 + timedelta(days=post_days)
    return {
        "pre": df[(df["date"] >= pre_from) & (df["date"] <= pre_to)],
        "at": df[(df["date"] >= t0) & (df["date"] <= t0)],
        "post": df[(df["date"] >= post_from) & (df["date"] <= post_to)],
    }

def _win_stats(seg: pd.DataFrame) -> WindowResult:
    if seg.empty:
        return WindowResult(n_days=0, tvl_total=0.0, tvl_avg=0.0)
    n = int(seg["date"].nunique())
    tot = float(seg["tvl"].sum())
    avg = float(tot / max(1, n))
    return WindowResult(n_days=n, tvl_total=tot, tvl_avg=avg)

def event_stats_tvl(df: pd.DataFrame, event_time_utc: str, pre_days=7, post_days=7) -> Dict[str, Any]:
    t0 = pd.to_datetime(event_time_utc, utc=True)
    segs = _window_slice(df, t0, pre_days, post_days)
    pre = _win_stats(segs["pre"])
    post = _win_stats(segs["post"])
    # simple abnormal = (post_avg / pre_avg) - 1
    abn = None if pre.tvl_avg == 0 else (post.tvl_avg / pre.tvl_avg - 1.0)
    return {
        "event_time_utc": pd.Timestamp(t0).tz_convert(None).isoformat(),
        "pre": {"n_days": pre.n_days, "tvl_total": pre.tvl_total, "tvl_avg": pre.tvl_avg},
        "post": {"n_days": post.n_days, "tvl_total": post.tvl_total, "tvl_avg": post.tvl_avg},
        "abnormal_change": None if abn is None else float(round(abn, 6)),
    }

# ---------------------------------------------------------------------
# Tool impl (plain)
# ---------------------------------------------------------------------
def resolve_protocol_impl(project_hint: str, ttl_hours: int = 24) -> Dict[str, Any]:
    """
    Resolve a protocol slug given a hint (Snapshot space/name/symbol).
    Performs metadata refresh only when TTL expired.
    """
    # Refresh metadata cache if needed (TTL)
    _ = refresh_protocols_cache(ttl_hours=ttl_hours)  # may be 0 if fresh  :contentReference[oaicite:4]{index=4}

    # Heuristic resolution
    slug = guess_slug_by_space(project_hint)
    return {"project_hint": project_hint, "slug": slug, "refreshed": bool(_)}

def refresh_protocol_impl(slug: str) -> Dict[str, Any]:
    """
    Refresh TVL series for a slug with incremental parquet merge.
    If there's no change, it avoids rewriting the file.
    """
    if not slug or not isinstance(slug, str):
        raise ValueError("slug must be a non-empty string")
    meta_ok = _slug_exists(slug)
    if not meta_ok:
        # try warm metadata (avoid hard failure on first call)
        refresh_protocols_cache(ttl_hours=0)
        if not _slug_exists(slug):
            raise ValueError(f"Unknown slug '{slug}'. Try resolve_protocol first.")
    stats = refresh_tvl_cache(slug)  # incremental merge  :contentReference[oaicite:5]{index=5}
    return stats | {"slug": slug}

def event_window_impl(slug: str, event_time_utc: str, pre_days: int = 7, post_days: int = 7) -> Dict[str, Any]:
    """
    Compute pre/post TVL window stats around a UTC timestamp (ISO string).
    Requires that the TVL parquet exists (run refresh first).
    """
    df = load_tvl(slug)
    out = event_stats_tvl(df, event_time_utc, pre_days=pre_days, post_days=post_days)
    return {"slug": slug, "window": {"pre_days": pre_days, "post_days": post_days}, "stats": out}

# ---------------------------------------------------------------------
# FastMCP tools
# ---------------------------------------------------------------------
@mcp.tool()
def resolve_protocol(project_hint: str, ttl_hours: int = 24) -> Dict[str, Any]:
    """Resolve a protocol slug from a project hint (uses TTL to avoid excess refresh)."""
    return resolve_protocol_impl(project_hint, ttl_hours)

@mcp.tool()
def refresh_protocol(slug: str) -> Dict[str, Any]:
    """Refresh TVL series for the given slug (incremental merge)."""
    return refresh_protocol_impl(slug)

@mcp.tool()
def event_window(slug: str, event_time_utc: str, pre_days: int = 7, post_days: int = 7) -> Dict[str, Any]:
    """Compute pre/post TVL stats for an event time (requires refreshed TVL cache)."""
    return event_window_impl(slug, event_time_utc, pre_days, post_days)

@mcp.tool()
def health() -> Dict[str, Any]:
    return {"ok": True, "service": "defillama_tvl_mcp", "db": str(DB_PATH), "dir": str(TVL_DIR)}
