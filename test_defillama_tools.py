# -*- coding: utf-8 -*-
"""
DeFiLlama TVL MCP (FastMCP)
- Exposes three tools: resolve_protocol, refresh_protocol, event_window
- Caching: protocol meta -> SQLite, per-protocol TVL timeseries -> Parquet
- Testing-friendly: each tool has an *_impl() function that can be called directly
"""

from __future__ import annotations
import os, json, time, math, sqlite3, logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests
import pandas as pd

# Import FastMCP (choose the one that matches your environment)
try:
    from fastmcp import FastMCP           # pip install fastmcp
except Exception:
    # Fallback if you use the official MCP server SDK
    from mcp.server.fastmcp import FastMCP  # pip install "modelcontextprotocol[server]"

# Optional utilities you shared; code works even if missing
try:
    import util  # noqa: F401
except Exception:
    util = None

LLAMA_BASE = "https://api.llama.fi"
DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
TVL_DIR = DATA_DIR / "tvl"
DB_PATH = DATA_DIR / "defillama_cache.sqlite"

TVL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("defillama_mcp")

mcp = FastMCP("defillama_tvl_mcp")


# ----------------------------- SQLite (protocol meta) ------------------------

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


# --------------------------- HTTP with retries --------------------------------

def _get_json(url: str, max_retries=4, timeout=30) -> Any:
    last_exc = None
    for i in range(max_retries):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            else:
                log.warning("GET %s -> %s %s", url, r.status_code, r.text[:200])
        except Exception as e:
            last_exc = e
            log.warning("GET %s error (%s/%s): %s", url, i+1, max_retries, e)
        time.sleep(1.2 * (i + 1))
    if last_exc:
        raise last_exc
    raise RuntimeError(f"Failed to GET {url}")


# --------------------------- DeFiLlama client --------------------------------

def fetch_protocols_list() -> pd.DataFrame:
    url = f"{LLAMA_BASE}/protocols"
    data = _get_json(url)
    rows = []
    for d in data:
        rows.append({
            "slug": d.get("slug") or (d.get("name","").lower().replace(" ","-") if d.get("name") else None),
            "name": d.get("name"),
            "symbol": d.get("symbol"),
            "cmc_id": str(d.get("cmcId")) if d.get("cmcId") not in (None, "") else None,
            "gecko_id": d.get("gecko_id"),
            "categories": ",".join(d.get("category", [])) if isinstance(d.get("category"), list) else d.get("category"),
            "chains": ",".join(d.get("chains", [])) if isinstance(d.get("chains"), list) else None,
            "governance_hint": json.dumps({"token_symbol": d.get("symbol")}),
        })
    return pd.DataFrame([r for r in rows if r["slug"]])


def refresh_protocols_cache() -> int:
    df = fetch_protocols_list()
    now_ts = int(time.time())
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
            """, (
                r["slug"], r["name"], r["symbol"], r["cmc_id"], r["gecko_id"],
                r["categories"], r["chains"], r["governance_hint"], now_ts
            ))
        conn.commit()
    return len(df)


def fetch_protocol_detail(slug: str) -> Dict[str, Any]:
    url = f"{LLAMA_BASE}/protocol/{slug}"
    return _get_json(url)


def tvl_timeseries_from_detail(detail_json: Dict[str, Any]) -> pd.DataFrame:
    """Prefer detail_json['tvl']; otherwise sum chainTvls[*]['tvl']."""
    def _extract_pairs(arr):
        out = []
        for x in arr:
            if isinstance(x, dict) and "date" in x and "totalLiquidityUSD" in x:
                out.append((int(x["date"]), float(x["totalLiquidityUSD"])))
        return out

    if "tvl" in detail_json and isinstance(detail_json["tvl"], list):
        pairs = _extract_pairs(detail_json["tvl"])
    else:
        pairs_map = {}
        chain_tvls = (detail_json.get("chainTvls") or detail_json.get("chainTvlsCharts") or {})
        for _, chain_obj in (chain_tvls or {}).items():
            for x in chain_obj.get("tvl", []):
                if isinstance(x, dict) and "date" in x and "totalLiquidityUSD" in x:
                    ts = int(x["date"])
                    pairs_map[ts] = pairs_map.get(ts, 0.0) + float(x["totalLiquidityUSD"])
        pairs = sorted(pairs_map.items(), key=lambda t: t[0])

    if not pairs:
        return pd.DataFrame(columns=["date", "tvl"])

    df = pd.DataFrame(pairs, columns=["unix", "tvl"])
    df["date"] = pd.to_datetime(df["unix"], unit="s", utc=True).dt.tz_convert(None)
    df = df.drop(columns=["unix"]).sort_values("date")
    return df[["date", "tvl"]]


def refresh_tvl_cache(slug: str) -> int:
    detail = fetch_protocol_detail(slug)
    df = tvl_timeseries_from_detail(detail)
    if df.empty:
        return 0
    fpath = TVL_DIR / f"{slug}.parquet"
    if fpath.exists():
        old = pd.read_parquet(fpath)
        merged = (
            pd.concat([old, df], ignore_index=True)
            .drop_duplicates(subset=["date"])
            .sort_values("date")
        )
    else:
        merged = df
    merged.to_parquet(fpath, index=False)
    return len(merged)


def load_tvl(slug: str) -> pd.DataFrame:
    fpath = TVL_DIR / f"{slug}.parquet"
    if not fpath.exists():
        raise FileNotFoundError(f"TVL cache not found for {slug}. Call refresh first.")
    df = pd.read_parquet(fpath)
    return df.sort_values("date").dropna()


# ------------------------------ Matching utils --------------------------------

def guess_slug_by_space(space: str) -> Optional[str]:
    """
    Heuristic match using cached slug/name/symbol.
    Snapshot spaces like 'lido-snapshot.eth' often align with 'lido'.
    """
    with _db() as conn:
        cur = conn.execute("SELECT slug,name,symbol FROM protocols;")
        best = None
        s_norm = space.lower().replace(".eth", "").replace(" ", "").replace("-", "")
        for slug, name, symbol in cur.fetchall():
            cands = [slug, (name or ""), (symbol or "")]
            score = 0
            for c in cands:
                c0 = c.lower().replace(" ", "").replace("-", "")
                if s_norm in c0 or c0 in s_norm:
                    score += 1
            if score and (best is None or score > best[0]):
                best = (score, slug)
        return best[1] if best else None


# ------------------------------ Event windows ---------------------------------

def _window_idx(df: pd.DataFrame, t0: pd.Timestamp, pre_days: int, post_days: int) -> Tuple[int, int, int]:
    idx0 = int((df["date"] - t0).abs().argmin())
    start = int(max(0, idx0 - pre_days))
    end = int(min(len(df) - 1, idx0 + post_days))
    return start, idx0, end


def event_stats_tvl(df: pd.DataFrame, t0: pd.Timestamp, pre_days=7, post_days=7) -> Dict[str, Any]:
    df = df.copy().sort_values("date")
    df["log_tvl"] = df["tvl"].clip(lower=1e-9).apply(math.log)

    s, i0, e = _window_idx(df, t0, pre_days, post_days)
    win = df.iloc[s:e+1].reset_index(drop=True)

    pre = df.iloc[max(0, i0-pre_days):i0]
    post = df.iloc[i0+1:min(len(df), i0+1+post_days)]

    res: Dict[str, Any] = {
        "t0_aligned": df.iloc[i0]["date"].isoformat(),
        "pre_days": int(len(pre)),
        "post_days": int(len(post)),
        "tvl_t0": float(df.iloc[i0]["tvl"]),
        "tvl_pre_mean": float(pre["tvl"].mean()) if len(pre) else None,
        "tvl_post_mean": float(post["tvl"].mean()) if len(post) else None,
        "log_pre_mean": float(pre["log_tvl"].mean()) if len(pre) else None,
        "log_post_mean": float(post["log_tvl"].mean()) if len(post) else None,
        "delta_pct_0_to_postK": (float(post["tvl"].iloc[-1]/df.iloc[i0]["tvl"] - 1.0) if len(post) else None),
        "delta_pct_preK_to_0": (float(df.iloc[i0]["tvl"]/pre["tvl"].iloc[0] - 1.0) if len(pre) else None),
        "window": {
            "start": df.iloc[s]["date"].isoformat(),
            "end": df.iloc[e]["date"].isoformat(),
        },
        "window_series": win[["date","tvl"]].assign(date=lambda x: x["date"].astype(str)).to_dict(orient="records")
    }

    if len(pre) > 2 and len(post) > 2:
        m1, v1, n1 = pre["log_tvl"].mean(), pre["log_tvl"].var(ddof=1), len(pre)
        m2, v2, n2 = post["log_tvl"].mean(), post["log_tvl"].var(ddof=1), len(post)
        denom = math.sqrt(v1/n1 + v2/n2) if (v1>0 and v2>0) else None
        if denom and denom > 0:
            res["welch_z_on_log_tvl"] = float((m2 - m1) / denom)
    return res


# ------------------------------ Impl functions --------------------------------
# (Plain callables for tests and internal use. MCP tool wrappers call these.)

def resolve_protocol_impl(project_hint: str) -> Dict[str, Any]:
    with _db() as conn:
        cur = conn.execute("SELECT slug FROM protocols WHERE slug=?;", (project_hint,))
        hit = cur.fetchone()
    if hit:
        return {"protocol_slug": project_hint, "matched_from": "slug"}

    with _db() as conn:
        n = conn.execute("SELECT COUNT(*) FROM protocols;").fetchone()[0]
    if n == 0:
        refresh_protocols_cache()

    guessed = guess_slug_by_space(project_hint)
    return {"protocol_slug": guessed, "matched_from": ("guess" if guessed else None)}


def refresh_protocol_impl(protocol_slug: str) -> Dict[str, Any]:
    rows = refresh_tvl_cache(protocol_slug)
    return {"rows": int(rows), "protocol_slug": protocol_slug}


def event_window_impl(protocol_or_space: str, event_time_utc: str,
                      pre_days: int = 7, post_days: int = 7, event_id: Optional[str] = None) -> Dict[str, Any]:
    with _db() as conn:
        cur = conn.execute("SELECT slug FROM protocols WHERE slug=?;", (protocol_or_space,))
        hit = cur.fetchone()
    slug = protocol_or_space if hit else guess_slug_by_space(protocol_or_space)
    if not slug:
        refresh_protocols_cache()
        slug = guess_slug_by_space(protocol_or_space)
    if not slug:
        return {"error": f"Cannot resolve protocol from '{protocol_or_space}'"}

    try:
        df = load_tvl(slug)
    except FileNotFoundError:
        refresh_tvl_cache(slug)
        df = load_tvl(slug)

    t0 = pd.to_datetime(event_time_utc, utc=True).tz_localize(None)
    stats = event_stats_tvl(df, t0, pre_days, post_days)

    if event_id:
        with _db() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO processed_events(protocol_slug,event_id,event_time_utc,pre_days,post_days)
                VALUES(?,?,?,?,?);
            """, (slug, event_id, event_time_utc, int(pre_days), int(post_days)))
            conn.commit()

    return {"protocol_slug": slug, "stats": stats}


# --------------------------------- MCP Tools ----------------------------------

@mcp.tool()
def resolve_protocol(project_hint: str) -> Dict[str, Any]:
    """Resolve a protocol slug from a Snapshot space or direct slug."""
    return resolve_protocol_impl(project_hint)

@mcp.tool()
def refresh_protocol(protocol_slug: str) -> Dict[str, Any]:
    """Refresh TVL cache (stores/merges a Parquet with daily TVL)."""
    return refresh_protocol_impl(protocol_slug)

@mcp.tool()
def event_window(protocol_or_space: str, event_time_utc: str,
                 pre_days: int = 7, post_days: int = 7, event_id: Optional[str] = None) -> Dict[str, Any]:
    """Compute TVL event-window stats around the given ISO8601 timestamp."""
    return event_window_impl(protocol_or_space, event_time_utc, pre_days, post_days, event_id)


# ------------------------------ Manual bootstrap ------------------------------

def _bootstrap_if_needed() -> None:
    """Ensure protocol cache exists once when running as a standalone server."""
    with _db() as conn:
        n = conn.execute("SELECT COUNT(*) FROM protocols;").fetchone()[0]
    if n == 0:
        log.info("Bootstrapping DeFiLlama protocols cache...")
        cnt = refresh_protocols_cache()
        log.info("Protocols cached: %s", cnt)


if __name__ == "__main__":
    # Run FastMCP server over stdio
    _bootstrap_if_needed()
    mcp.run()
