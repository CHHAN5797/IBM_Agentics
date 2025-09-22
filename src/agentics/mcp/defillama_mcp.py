# -*- coding: utf-8 -*-
"""
DeFiLlama TVL MCP (slug-only, minimal)
- Only works with a provided DeFiLlama protocol slug (e.g., 'aave').
- No link parsing, no resolve by space, no normalization heuristics.
- If parquet is missing, it auto-fetches and caches TVL for the slug.
"""

from __future__ import annotations
import os, json, time, logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd

try:
    from fastmcp import FastMCP
except Exception:
    from mcp.server.fastmcp import FastMCP

LLAMA_BASE = "https://api.llama.fi"
DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
TVL_DIR = DATA_DIR / "tvl"
DATA_DIR.mkdir(parents=True, exist_ok=True)
TVL_DIR.mkdir(parents=True, exist_ok=True)

HTTP_TIMEOUT = int(os.getenv("LLAMA_HTTP_TIMEOUT", "45"))
RETRY_TOTAL  = int(os.getenv("LLAMA_HTTP_RETRIES", "5"))
BASE_SLEEP   = float(os.getenv("LLAMA_BASE_SLEEP", "0.35"))
UA           = os.getenv("LLAMA_UA", "defillama-mcp/slug-only-1.0")

log = logging.getLogger("defillama_mcp")
logging.basicConfig(level=logging.INFO)
mcp = FastMCP("defillama_tvl_mcp")
UTC = timezone.utc

def _session() -> requests.Session:
    from urllib3.util.retry import Retry
    from requests.adapters import HTTPAdapter
    s = requests.Session()
    s.headers.update({"User-Agent": UA})
    retry = Retry(
        total=RETRY_TOTAL, read=RETRY_TOTAL, connect=RETRY_TOTAL,
        backoff_factor=0.6, status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"], raise_on_status=False
    )
    ad = HTTPAdapter(max_retries=retry)
    s.mount("http://", ad); s.mount("https://", ad)
    return s

HTTP = _session()
def _sleep(): time.sleep(BASE_SLEEP)

def _get_json(url: str, params: Optional[dict] = None) -> Any:
    _sleep()
    r = HTTP.get(url, params=params or {}, timeout=HTTP_TIMEOUT)
    if r.status_code != 200:
        raise RuntimeError(f"GET {url} -> {r.status_code}: {r.text[:200]}")
    return r.json()

def fetch_protocol_detail(slug: str) -> Dict[str, Any]:
    if not slug or not isinstance(slug, str):
        raise ValueError("slug must be a non-empty string")
    return _get_json(f"{LLAMA_BASE}/protocol/{slug}")

def _pairs_from_detail(detail: Dict[str, Any]) -> List[Tuple[int, float]]:
    """
    Normalize DeFiLlama detail payload to [(unix_ts, tvl_usd), ...].
    Prefer 'tvl' array if present; otherwise aggregate per-chain series.
    """
    def _extract(arr):
        out=[]
        for x in arr or []:
            if isinstance(x, dict) and "date" in x and "totalLiquidityUSD" in x:
                try:
                    out.append((int(x["date"]), float(x["totalLiquidityUSD"])))
                except Exception:
                    pass
        return out

    if isinstance(detail.get("tvl"), list):
        pairs = _extract(detail["tvl"])
    else:
        pairs_map = {}
        chain_tvls = (detail.get("chainTvls") or detail.get("chainTvlsCharts") or {})
        for _, obj in (chain_tvls or {}).items():
            for x in obj.get("tvl", []):
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
    detail = fetch_protocol_detail(slug)
    new_df = tvl_to_df(_pairs_from_detail(detail))
    fpath = _tvl_path(slug)
    if new_df.empty:
        before = 0 if not fpath.exists() else len(pd.read_parquet(fpath))
        return {"slug": slug, "rows_before": before, "rows_added": 0, "rows_after": before, "written": False}

    if fpath.exists():
        old = pd.read_parquet(fpath)
        merged = (pd.concat([old, new_df], ignore_index=True)
                  .drop_duplicates(subset=["date"]).sort_values("date"))
    else:
        merged = new_df

    merged.to_parquet(fpath, index=False)
    return {
        "slug": slug,
        "rows_before": len(merged),
        "rows_added": 0,      # deduped union; we do not track delta precisely here
        "rows_after": len(merged),
        "written": True
    }

def load_tvl(slug: str) -> pd.DataFrame:
    f = _tvl_path(slug)
    if not f.exists():
        raise FileNotFoundError(f"TVL cache not found for slug={slug!r}")
    df = pd.read_parquet(f)
    return df.sort_values("date").dropna()

@dataclass
class WindowResult:
    n_days: int
    tvl_total: float
    tvl_avg: float

def _window_slice(df: pd.DataFrame, t0: pd.Timestamp, pre_days: int, post_days: int) -> Dict[str, pd.DataFrame]:
    df = df.copy().sort_values("date")
    t0 = pd.to_datetime(t0, utc=True).tz_convert(None)
    pre_from = t0 - timedelta(days=pre_days); pre_to = t0 - timedelta(days=1)
    post_from = t0 + timedelta(days=1); post_to = t0 + timedelta(days=post_days)
    return {
        "pre":  df[(df["date"] >= pre_from) & (df["date"] <= pre_to)],
        "at":   df[(df["date"] >= t0)       & (df["date"] <= t0)],
        "post": df[(df["date"] >= post_from)& (df["date"] <= post_to)],
    }

def _win_stats(seg: pd.DataFrame) -> WindowResult:
    if seg.empty:
        return WindowResult(0, 0.0, 0.0)
    n = int(seg["date"].nunique())
    tot = float(seg["tvl"].sum())
    return WindowResult(n_days=n, tvl_total=tot, tvl_avg=float(tot / max(1, n)))

def event_stats_tvl(df: pd.DataFrame, event_time_utc: str, pre_days=7, post_days=7) -> Dict[str, Any]:
    t0 = pd.to_datetime(event_time_utc, utc=True)
    segs = _window_slice(df, t0, pre_days, post_days)
    pre = _win_stats(segs["pre"]); post = _win_stats(segs["post"])
    abn = None if pre.tvl_avg == 0 else (post.tvl_avg / pre.tvl_avg - 1.0)
    return {
        "event_time_utc": pd.Timestamp(t0).tz_convert(None).isoformat(),
        "pre":  {"n_days": pre.n_days,  "tvl_total": pre.tvl_total,  "tvl_avg": pre.tvl_avg},
        "post": {"n_days": post.n_days, "tvl_total": post.tvl_total, "tvl_avg": post.tvl_avg},
        "abnormal_change": None if abn is None else float(round(abn, 6)),
    }

# ----------------- MCP Tools (slug-only) -----------------

@mcp.tool()
def refresh_protocol(slug: str) -> Dict[str, Any]:
    """
    Ensure local TVL parquet exists/updated for a given slug.
    """
    if not slug or not isinstance(slug, str):
        raise ValueError("slug must be a non-empty string")
    return refresh_tvl_cache(slug)

@mcp.tool()
def event_window(slug: str, event_time_utc: str, pre_days: int = 7, post_days: int = 7) -> Dict[str, Any]:
    """
    Compute TVL abnormal change around `event_time_utc` using the same window
    scheme as token price (pre/post days default to 7).
    """
    if not slug or not isinstance(slug, str):
        raise ValueError("slug must be a non-empty string")
    try:
        df = load_tvl(slug)
    except FileNotFoundError:
        refresh_tvl_cache(slug)
        df = load_tvl(slug)
    stats = event_stats_tvl(df, event_time_utc, pre_days=pre_days, post_days=post_days)
    return {"slug": slug, "window": {"pre_days": pre_days, "post_days": post_days}, "stats": stats}

@mcp.tool()
def health() -> Dict[str, Any]:
    return {"ok": True, "service": "defillama_tvl_mcp", "dir": str(TVL_DIR)}

if __name__ == "__main__":
    mcp.run(transport="stdio")
