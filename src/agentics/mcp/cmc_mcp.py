# -*- coding: utf-8 -*-
"""
CMC Price MCP (FastMCP) — improved for API efficiency

Tools:
  - resolve_tokens(project_hint: str, prefer_governance: bool = True) -> dict
  - refresh_price(token: str|int, interval: "1h"|"1d", start_date: "YYYY-MM-DD", end_date: "YYYY-MM-DD") -> dict
  - price_window(token: str|int, interval: "1h"|"1d", start_date: "YYYY-MM-DD", end_date: "YYYY-MM-DD") -> dict

Key behaviors (cache-first & incremental):
  * Read existing rows from local DuckDB for the requested date window.
  * Compute missing date ranges (gaps) and fetch ONLY those from API.
  * Upsert (INSERT OR REPLACE) by (date, ucid) → idempotent incremental updates.
  * If everything is already cached, no network calls are made.

Other improvements:
  * Shared HTTP session with retries (429/5xx), stable UA, per-request request-id
  * Prefer official CMC Pro endpoints when API key is present; otherwise use web JSON
  * Safer token resolution: metadata.csv -> Pro /map -> web search (last resort)
  * Robust header handling and stricter schema normalization
"""

from __future__ import annotations
import os
import re
import uuid
import json
import math
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone

import duckdb
import pandas as pd
import requests

try:
    from fastmcp import FastMCP
except Exception:
    from mcp.server.fastmcp import FastMCP

# ------------------------------------------------------------------------------
# Config & Paths
# ------------------------------------------------------------------------------
DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DUCK_PATH = Path(os.environ.get("CMC_DUCKDB", DATA_DIR / "cmc.duckdb"))

# Web JSON (unofficial) endpoints — used when Pro key is absent
CMC_HIST_URL = "https://api.coinmarketcap.com/data-api/v3.1/cryptocurrency/historical"
CMC_SEARCH_URL = "https://api.coinmarketcap.com/data-api/v3/search"

# Official Pro API (used when key present)
CMC_PRO_KEY = os.getenv("CMC_PRO_API_KEY")
CMC_PRO_BASE = "https://pro-api.coinmarketcap.com"
CMC_PRO_MAP = f"{CMC_PRO_BASE}/v1/cryptocurrency/map"
CMC_PRO_INFO = f"{CMC_PRO_BASE}/v2/cryptocurrency/info"
CMC_PRO_OHLCV = f"{CMC_PRO_BASE}/v2/cryptocurrency/ohlcv/historical"

# Common headers for web JSON
CMC_WEB_HEADERS_BASE = {
    "accept": "application/json, text/plain, */*",
    "accept-language": "en-US,en;q=0.9,ko-KR;q=0.8,ko;q=0.7",
    "cache-control": "no-cache",
    "origin": "https://coinmarketcap.com",
    "platform": "web",
    "priority": "u=1, i",
    "referer": "https://coinmarketcap.com/",
    "sec-ch-ua": "\"Google Chrome\";v=\"135\", \"Not-A.Brand\";v=\"8\", \"Chromium\";v=\"135\"",
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "\"Windows\"",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-site",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
}

# Optional metadata.csv locations
_RAW_ENV = os.environ.get("CMC_METADATA", "").strip()
_METADATA_CANDIDATES: List[Path] = []
if _RAW_ENV:
    p = Path(_RAW_ENV)
    if p.is_file():
        _METADATA_CANDIDATES.append(p)
_METADATA_CANDIDATES += [
    Path("resource/metadata.csv"),
    Path("metadata.csv"),
    DATA_DIR / "metadata.csv",
]

# HTTP / Retry config
HTTP_TIMEOUT = int(os.getenv("CMC_HTTP_TIMEOUT", "45"))
RETRY_TOTAL = int(os.getenv("CMC_HTTP_RETRIES", "5"))

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("cmc_price_mcp")

# ------------------------------------------------------------------------------
# FastMCP app
# ------------------------------------------------------------------------------
mcp = FastMCP("cmc_price_mcp")

# ------------------------------------------------------------------------------
# HTTP session with retries
# ------------------------------------------------------------------------------
def _session() -> requests.Session:
    from urllib3.util.retry import Retry
    from requests.adapters import HTTPAdapter

    s = requests.Session()
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

def _headers_with_reqid(base: Dict[str, str]) -> Dict[str, str]:
    """Clone headers and attach a per-request id to reduce upstream cache collisions."""
    h = dict(base)
    h["x-request-id"] = uuid.uuid4().hex
    return h

# ------------------------------------------------------------------------------
# DuckDB helpers
# ------------------------------------------------------------------------------
def _connect_duck(path: Path = DUCK_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path))

def _ensure_ohlcv_table(conn, table: str):
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            date DATE,                 -- primary key per (date, ucid)
            ucid BIGINT,               -- CMC id
            price_close DOUBLE,        -- USD close
            price_open DOUBLE,
            price_high DOUBLE,
            price_low DOUBLE,
            volume DOUBLE,
            market_cap DOUBLE
        );
    """)
    conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{table}_date_uc ON {table} (date, ucid);")

def _interval_table(interval: str) -> str:
    return "ohlcv_1h" if interval.lower() in ("1h", "h1") else "ohlcv_1d"

# ------------------------------------------------------------------------------
# Metadata loading and token resolution
# ------------------------------------------------------------------------------
def _load_metadata_csv() -> pd.DataFrame:
    """Load metadata.csv if present; otherwise return empty frame."""
    for p in _METADATA_CANDIDATES:
        try:
            if p.is_file():
                df = pd.read_csv(p, encoding_errors="ignore")
                for c in ("id", "symbol", "name", "slug", "rank"):
                    if c not in df.columns:
                        df[c] = None
                return df
        except Exception as e:
            log.warning("Failed to read metadata CSV at %s: %s", p, e)
    return pd.DataFrame(columns=["id", "symbol", "name", "slug", "rank"])

def _resolve_by_metadata(token: str) -> Optional[int]:
    df = _load_metadata_csv()
    if df.empty:
        return None
    s = token.strip().lower()
    q = df[
        (df["symbol"].astype(str).str.lower() == s)
        | (df["slug"].astype(str).str.lower() == s)
        | (df["name"].astype(str).str.lower() == s)
    ]
    if not q.empty and pd.notna(q.iloc[0].get("id")):
        return int(q.iloc[0]["id"])
    return None

def _resolve_by_pro_map(token: str) -> Optional[int]:
    if not CMC_PRO_KEY:
        return None
    try:
        r = HTTP.get(CMC_PRO_MAP, headers={"X-CMC_PRO_API_KEY": CMC_PRO_KEY, "Accept": "application/json"},
                     params={"listing_status": "active,untracked,inactive", "limit": 5000}, timeout=HTTP_TIMEOUT)
        if r.status_code != 200:
            return None
        data = (r.json() or {}).get("data", [])
        s = token.strip().lower()
        best = None
        for item in data:
            score = 0
            for v in (item.get("symbol"), item.get("slug"), item.get("name")):
                if not v:
                    continue
                v0 = str(v).lower()
                if v0 == s:
                    score += 2
                if s in v0 or v0 in s:
                    score += 1
            if score and (best is None or score > best[0]):
                best = (score, item.get("id"))
        return int(best[1]) if best else None
    except Exception:
        return None

def _resolve_by_web_search(token: str) -> Optional[int]:
    try:
        r = HTTP.get(CMC_SEARCH_URL, params={"q": token, "query": token}, timeout=HTTP_TIMEOUT)
        if r.status_code != 200:
            return None
        data = r.json().get("data") or {}
        pools = []
        for key in ("crypto", "currencies", "cryptocurrency", "trending", "topSearches"):
            node = data.get(key)
            if isinstance(node, list):
                pools.extend(node)
        best = None
        s_norm = token.strip().lower()
        for item in pools:
            cid = item.get("id") or item.get("cid")
            sym = item.get("symbol") or item.get("code")
            name = item.get("name") or item.get("title")
            slug = item.get("slug")
            score = 0
            for v in (sym, slug, name):
                if not v:
                    continue
                v0 = str(v).lower()
                if s_norm == v0:
                    score += 2
                if s_norm in v0 or v0 in s_norm:
                    score += 1
            if cid and (best is None or score > best[0]):
                best = (score, cid)
        return int(best[1]) if best else None
    except Exception:
        return None

def _resolve_token_to_id(token: str | int) -> int:
    """Robust token → CMC id resolution with stable priority."""
    if isinstance(token, int):
        return token
    s = str(token).strip()
    digits = re.sub(r"[^0-9]", "", s)
    if digits:
        return int(digits)
    for fn in (_resolve_by_metadata, _resolve_by_pro_map, _resolve_by_web_search):
        cid = fn(s)
        if cid:
            return cid
    raise ValueError(f"Cannot resolve token '{token}' to CMC id")

# ------------------------------------------------------------------------------
# Fetchers (Web JSON / Pro)
# ------------------------------------------------------------------------------
def _fetch_hist_web_json(cmc_id: int, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
    """
    Use the web JSON endpoint (no API key). Interval: '1d' or '1h'.
    Returns a DataFrame with columns: date, price_close, price_open/high/low, volume, market_cap
    """
    headers = _headers_with_reqid(CMC_WEB_HEADERS_BASE)
    params = {
        "id": cmc_id,
        "convert": "USD",
        "timeStart": start_date,
        "timeEnd": end_date,
        "interval": "1d" if interval == "1d" else "1h",
    }
    r = HTTP.get(CMC_HIST_URL, headers=headers, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    js = r.json()
    quotes = (((js.get("data") or {}).get("quotes") or {}).get("USD") or [])
    rows = []
    for q in quotes:
        # Normalize time fields defensively (web JSON uses timeOpen/timeClose)
        t_open = q.get("timeOpen") or q.get("time_open") or q.get("t")
        t_close = q.get("timeClose") or q.get("time_close") or q.get("t")
        d = pd.to_datetime(t_close or t_open, utc=True).date()
        rows.append({
            "date": d,
            "price_close": float(q.get("quote", {}).get("close") if "quote" in q else q.get("close")),
            "price_open": float(q.get("quote", {}).get("open")  if "quote" in q else q.get("open")),
            "price_high": float(q.get("quote", {}).get("high")  if "quote" in q else q.get("high")),
            "price_low":  float(q.get("quote", {}).get("low")   if "quote" in q else q.get("low")),
            "volume":     float(q.get("quote", {}).get("volume") if "quote" in q else q.get("volume", 0.0)),
            "market_cap": float(q.get("quote", {}).get("marketCap") if "quote" in q else q.get("marketCap", 0.0)),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return (df.drop_duplicates("date").sort_values("date"))

def _fetch_hist_pro(cmc_id: int, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
    """Use CMC Pro API if key is present. Interval: '1d' only (Pro hourly varies by plan)."""
    if not CMC_PRO_KEY:
        raise RuntimeError("CMC_PRO_API_KEY not set")
    params = {
        "id": cmc_id,
        "time_start": start_date,
        "time_end": end_date,
        "interval": "daily" if interval == "1d" else "hourly",
        "convert": "USD",
    }
    headers = {"X-CMC_PRO_API_KEY": CMC_PRO_KEY, "Accept": "application/json"}
    r = HTTP.get(CMC_PRO_OHLCV, headers=headers, params=params, timeout=HTTP_TIMEOUT)
    if r.status_code != 200 and interval != "1d":
        # Some plans don't support hourly on v2; try v3 fallback if available
        alt = f"{CMC_PRO_BASE}/v3/cryptocurrency/ohlcv/historical"
        r = HTTP.get(alt, headers=headers, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    js = r.json()
    quotes = (js.get("data") or {}).get("quotes") or (js.get("data") or {}).get("ohlcv") or []
    rows = []
    for q in quotes:
        t = q.get("time_open") or q.get("time_period_start") or q.get("t")
        d = pd.to_datetime(t, utc=True).date()
        usd = q.get("quote", {}).get("USD", {})
        rows.append({
            "date": d,
            "price_close": float(usd.get("close")),
            "price_open": float(usd.get("open")),
            "price_high": float(usd.get("high")),
            "price_low":  float(usd.get("low")),
            "volume":     float(usd.get("volume", 0.0)),
            "market_cap": float(usd.get("market_cap", 0.0)),
        })
    df = pd.DataFrame(rows)
    return (df.drop_duplicates("date").sort_values("date"))

# ------------------------------------------------------------------------------
# Incremental upsert logic
# ------------------------------------------------------------------------------
def _read_existing(conn, table: str, ucid: int, start_date: str, end_date: str) -> pd.DataFrame:
    sql = f"""
        SELECT date, ucid, price_close, price_open, price_high, price_low, volume, market_cap
        FROM {table}
        WHERE ucid = ? AND date BETWEEN ? AND ?
        ORDER BY date;
    """
    return conn.execute(sql, [ucid, start_date, end_date]).df()

def _missing_date_ranges(existing: pd.DataFrame, start_date: str, end_date: str) -> List[Tuple[str, str]]:
    """
    Given an existing date series [start_date, end_date], return missing contiguous gaps
    that should be fetched from the API, as (gap_start, gap_end) pairs.
    """
    s = pd.to_datetime(start_date).date()
    e = pd.to_datetime(end_date).date()
    if existing.empty:
        return [(s.isoformat(), e.isoformat())]
    have = set(pd.to_datetime(existing["date"]).dt.date.tolist())
    all_days = pd.date_range(s, e, freq="D").date
    gaps = []
    run_start = None
    for d in all_days:
        if d not in have:
            if run_start is None:
                run_start = d
        else:
            if run_start is not None:
                gaps.append((run_start.isoformat(), (d - timedelta(days=1)).isoformat()))
                run_start = None
    if run_start is not None:
        gaps.append((run_start.isoformat(), e.isoformat()))
    return gaps

def _upsert_rows(conn, table: str, ucid: int, df: pd.DataFrame):
    if df is None or df.empty:
        return 0
    tmp = df.copy()
    tmp["ucid"] = ucid
    # Use INSERT OR REPLACE semantics on the unique index (date, ucid)
    conn.register("tmp_upsert", tmp)
    conn.execute(f"""
        INSERT OR REPLACE INTO {table}
        SELECT date, ucid, price_close, price_open, price_high, price_low, volume, market_cap
        FROM tmp_upsert;
    """)
    conn.unregister("tmp_upsert")
    return len(tmp)

# ------------------------------------------------------------------------------
# Plain impl functions
# ------------------------------------------------------------------------------
def resolve_tokens_impl(project_hint: str, prefer_governance: bool = True) -> Dict[str, Any]:
    """
    Suggest candidate tokens from metadata.csv; no network side effects here.
    """
    meta = _load_metadata_csv()
    if meta.empty:
        return {"project_hint": project_hint, "candidates": [], "governance_token": None, "native_token": None}

    h = project_hint.lower().replace(".eth", "").replace("-", " ").replace("_", " ").strip()
    scored = []
    for _, r in meta.iterrows():
        fields = [str(r.get("name") or ""), str(r.get("slug") or ""), str(r.get("symbol") or "")]
        score = 0
        for f in fields:
            f0 = f.lower().replace("-", " ").replace("_", " ")
            if h in f0 or f0 in h:
                score += 1
        if score:
            scored.append((score, r))
    scored.sort(key=lambda x: (-x[0], int(x[1].get("rank") or 1e9)))
    cands = []
    for sc, r in scored[:8]:
        cands.append({
            "id": int(r.get("id")) if pd.notna(r.get("id")) else None,
            "symbol": r.get("symbol"),
            "name": r.get("name"),
            "slug": r.get("slug"),
            "score": sc,
            "is_governance_hint": any(k in f"{r.get('name','')} {r.get('slug','')}".lower()
                                      for k in ["dao","governance","vote","gov"])
        })
    def _pick(governance: bool) -> Optional[Dict[str, Any]]:
        if not cands: return None
        if governance:
            govs = [c for c in cands if c.get("is_governance_hint")]
            if govs: return govs[0]
        return cands[0]

    return {
        "project_hint": project_hint,
        "governance_token": _pick(True),
        "native_token": _pick(False),
        "candidates": cands
    }

def refresh_price_impl(
    token: str | int,
    interval: str = "1d",
    start_date: str = "2018-01-01",
    end_date: str = "2025-12-31",
    duck_path: Path = DUCK_PATH,
) -> Dict[str, Any]:
    """
    Incrementally refresh OHLCV into DuckDB for [start_date, end_date].
    Only downloads missing date ranges to minimize API calls.
    """
    ucid = _resolve_token_to_id(token)
    table = _interval_table(interval)
    conn = _connect_duck(duck_path)
    _ensure_ohlcv_table(conn, table)

    # 1) Read existing rows
    existing = _read_existing(conn, table, ucid, start_date, end_date)

    # 2) Detect gaps
    gaps = _missing_date_ranges(existing, start_date, end_date)
    fetched = 0
    used = "pro" if CMC_PRO_KEY else "web"

    # 3) Fetch only for gaps, then upsert
    for (gs, ge) in gaps:
        if CMC_PRO_KEY:
            df_gap = _fetch_hist_pro(ucid, gs, ge, interval)
        else:
            df_gap = _fetch_hist_web_json(ucid, gs, ge, interval)
        fetched += _upsert_rows(conn, table, ucid, df_gap)

    # 4) Return final count and summary
    final = _read_existing(conn, table, ucid, start_date, end_date)
    return {
        "ucid": ucid,
        "interval": interval,
        "range": {"start": start_date, "end": end_date},
        "source": used,
        "rows_before": int(len(existing)),
        "rows_added": int(fetched),
        "rows_after": int(len(final)),
    }

def price_window_impl(
    token: str | int,
    interval: str,
    start_date: str,
    end_date: str,
    duck_path: Path = DUCK_PATH,
) -> Dict[str, Any]:
    """Read a price window from DuckDB (no network)."""
    ucid = _resolve_token_to_id(token)
    table = _interval_table(interval)
    conn = _connect_duck(duck_path)
    _ensure_ohlcv_table(conn, table)
    df = _read_existing(conn, table, ucid, start_date, end_date)
    out = df.sort_values("date").to_dict(orient="records")
    return {
        "ucid": ucid,
        "interval": interval,
        "range": {"start": start_date, "end": end_date},
        "count": len(out),
        "series": out,
    }

# ------------------------------------------------------------------------------
# FastMCP tools
# ------------------------------------------------------------------------------
@mcp.tool()
def resolve_tokens(project_hint: str, prefer_governance: bool = True) -> Dict[str, Any]:
    """Suggest governance/native token candidates for a project hint."""
    return resolve_tokens_impl(project_hint, prefer_governance)

@mcp.tool()
def refresh_price(token: str, interval: str = "1d", start_date: str = "2018-01-01", end_date: str = "2025-12-31") -> Dict[str, Any]:
    """
    Incrementally refresh OHLCV to DuckDB.
    - interval: "1d" (preferred) or "1h" (web JSON or Pro hourly subject to plan)
    """
    return refresh_price_impl(token, interval, start_date, end_date, DUCK_PATH)

@mcp.tool()
def price_window(token: str, interval: str, start_date: str, end_date: str) -> Dict[str, Any]:
    """Return a price window (no network)."""
    return price_window_impl(token, interval, start_date, end_date, DUCK_PATH)

@mcp.tool()
def health() -> Dict[str, Any]:
    return {"ok": True, "service": "cmc_price_mcp", "duckdb": str(DUCK_PATH)}
