# -*- coding: utf-8 -*-
"""Shared utilities for DeFiLlama TVL analysis."""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

from .defillama_protocol_cache import (
    protocol_count,
    rank_protocol_candidates,
    refresh_protocols_cache,
    slug_exists,
)

LLAMA_BASE = "https://api.llama.fi"
DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
TVL_DIR = DATA_DIR / "tvl"
DB_PATH = DATA_DIR / "defillama_cache.sqlite"
DATA_DIR.mkdir(parents=True, exist_ok=True)
TVL_DIR.mkdir(parents=True, exist_ok=True)

HTTP_TIMEOUT = int(os.getenv("LLAMA_HTTP_TIMEOUT", "45"))
RETRY_TOTAL = int(os.getenv("LLAMA_HTTP_RETRIES", "5"))
BASE_SLEEP = float(os.getenv("LLAMA_BASE_SLEEP", "0.35"))
UA = os.getenv("LLAMA_UA", "defillama-mcp/utils-1.0")

log = logging.getLogger("defillama_utils")
UTC = timezone.utc


def _session() -> requests.Session:
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

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
    ad = HTTPAdapter(max_retries=retry)
    s.mount("http://", ad)
    s.mount("https://", ad)
    return s


HTTP = _session()


def _sleep():
    time.sleep(BASE_SLEEP)


def _get_json(url: str, params: Optional[dict] = None) -> Any:
    _sleep()
    r = HTTP.get(url, params=params or {}, timeout=HTTP_TIMEOUT)
    if r.status_code != 200:
        raise RuntimeError(f"GET {url} -> {r.status_code}: {r.text[:200]}")
    return r.json()


def fetch_protocols_list() -> List[Dict[str, Optional[str]]]:
    data = _get_json(f"{LLAMA_BASE}/protocols")
    rows: List[Dict[str, Optional[str]]] = []
    for item in data:
        slug = (item.get("slug") or "").strip().lower()
        if not slug:
            name = (item.get("name") or "").strip().lower()
            slug = name.replace(" ", "-") if name else ""
        if slug:
            rows.append(
                {
                    "slug": slug,
                    "name": item.get("name") or None,
                    "symbol": item.get("symbol") or None,
                }
            )
    return rows


def _refresh_meta(ttl_hours: int = 24) -> int:
    return refresh_protocols_cache(DB_PATH, fetch_protocols_list, ttl_hours)


def fetch_protocol_detail(slug: str) -> Dict[str, Any]:
    if not slug or not isinstance(slug, str):
        raise ValueError("slug must be a non-empty string")
    return _get_json(f"{LLAMA_BASE}/protocol/{slug}")


def _pairs_from_detail(detail: Dict[str, Any]) -> List[Tuple[int, float]]:
    """Normalize detail payload to [(unix_ts, tvl_usd)], aggregating chains."""

    def _extract(arr):
        out = []
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
        chain_tvls = detail.get("chainTvls") or detail.get("chainTvlsCharts") or {}
        for _, obj in (chain_tvls or {}).items():
            for x in obj.get("tvl", []):
                if isinstance(x, dict) and "date" in x and "totalLiquidityUSD" in x:
                    try:
                        ts = int(x["date"])
                        pairs_map[ts] = pairs_map.get(ts, 0.0) + float(
                            x["totalLiquidityUSD"]
                        )
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
        return {
            "slug": slug,
            "rows_before": before,
            "rows_added": 0,
            "rows_after": before,
            "written": False,
        }

    if fpath.exists():
        old = pd.read_parquet(fpath)
        before = len(old)
        merged = (
            pd.concat([old, new_df], ignore_index=True)
            .drop_duplicates(subset=["date"])
            .sort_values("date")
        )
    else:
        before = 0
        merged = new_df

    rows_after = len(merged)
    rows_added = rows_after - before
    merged.to_parquet(fpath, index=False)
    return {
        "slug": slug,
        "rows_before": before,
        "rows_added": rows_added,
        "rows_after": rows_after,
        "written": True,
    }


def load_tvl(slug: str) -> pd.DataFrame:
    f = _tvl_path(slug)
    if not f.exists():
        raise FileNotFoundError(f"TVL cache not found for slug={slug!r}")
    df = pd.read_parquet(f)
    return df.sort_values("date").dropna()


def _snapshot_base(hint: str) -> Optional[str]:
    if not hint:
        return None
    hint = hint.strip().lower()
    if not hint:
        return None
    if hint.endswith(".eth"):
        base = hint[:-4]
        if base.endswith("-snapshot"):
            base = base[:-9]
        return base or None
    return None


def resolve_protocol_from_snapshot_space(space: str) -> Optional[str]:
    """Resolve DeFi protocol slug from Snapshot space name."""
    if not space:
        return None

    # Extract base name from .eth domains
    base = _snapshot_base(space)
    hints = [space.lower()]
    if base:
        hints.append(base)

    # Common mappings
    space_lower = space.lower()
    if space_lower in ("aavedao.eth", "aave.eth"):
        hints.append("aave")
    elif space_lower in ("uniswap", "uniswapgovernance.eth"):
        hints.append("uniswap")
    elif space_lower in ("compound-community.eth", "comp-vote.eth"):
        hints.append("compound")
    elif space_lower in ("makerdao.eth", "maker"):
        hints.append("maker")

    # Try to resolve using existing protocol resolution logic
    count = protocol_count(DB_PATH)
    if count == 0:
        _refresh_meta(ttl_hours=0)

    candidates = rank_protocol_candidates(DB_PATH, hints, limit=1)
    if candidates and len(candidates) > 0:
        return candidates[0].get("slug") if isinstance(candidates[0], dict) else candidates[0]
    return None


def event_stats_tvl(
    df: pd.DataFrame, event_time_utc: str, pre_days=7, post_days=7
) -> Dict[str, Any]:
    """Compute TVL statistics around an event time."""
    from dataclasses import dataclass

    @dataclass
    class WindowResult:
        n_days: int
        tvl_total: float
        tvl_avg: float

    def _window_slice(
        df: pd.DataFrame, t0: pd.Timestamp, pre_days: int, post_days: int
    ) -> Dict[str, pd.DataFrame]:
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
            return WindowResult(0, 0.0, 0.0)
        n = int(seg["date"].nunique())
        tot = float(seg["tvl"].sum())
        return WindowResult(n_days=n, tvl_total=tot, tvl_avg=float(tot / max(1, n)))

    t0 = pd.to_datetime(event_time_utc, utc=True)
    segs = _window_slice(df, t0, pre_days, post_days)
    pre = _win_stats(segs["pre"])
    post = _win_stats(segs["post"])
    abn = None if pre.tvl_avg == 0 else (post.tvl_avg / pre.tvl_avg - 1.0)
    return {
        "event_time_utc": pd.Timestamp(t0).tz_convert(None).isoformat(),
        "pre": {
            "n_days": pre.n_days,
            "tvl_total": pre.tvl_total,
            "tvl_avg": pre.tvl_avg,
        },
        "post": {
            "n_days": post.n_days,
            "tvl_total": post.tvl_total,
            "tvl_avg": post.tvl_avg,
        },
        "abnormal_change": None if abn is None else float(round(abn, 6)),
    }


def get_tvl_impact_for_proposal(
    space: str, proposal_end_utc: str, pre_days: int = 7, post_days: int = 7
) -> Dict[str, Any]:
    """Get TVL impact analysis for a proposal."""
    try:
        # Resolve protocol slug from space
        protocol_slug = resolve_protocol_from_snapshot_space(space)
        if not protocol_slug:
            return {
                "protocol_slug": None,
                "status": "no_protocol_mapping",
                "error": f"Could not map space '{space}' to DeFi protocol"
            }

        # Check if protocol exists in cache
        if not slug_exists(DB_PATH, protocol_slug):
            _refresh_meta(ttl_hours=0)
            if not slug_exists(DB_PATH, protocol_slug):
                return {
                    "protocol_slug": protocol_slug,
                    "status": "protocol_not_found",
                    "error": f"Protocol '{protocol_slug}' not found in DeFiLlama"
                }

        # Try to load TVL data
        try:
            df = load_tvl(protocol_slug)
        except FileNotFoundError:
            # Try to refresh TVL cache
            try:
                refresh_tvl_cache(protocol_slug)
                df = load_tvl(protocol_slug)
            except Exception as e:
                return {
                    "protocol_slug": protocol_slug,
                    "status": "tvl_data_unavailable",
                    "error": f"Could not fetch TVL data: {str(e)}"
                }

        # Compute TVL impact
        stats = event_stats_tvl(df, proposal_end_utc, pre_days, post_days)

        return {
            "protocol_slug": protocol_slug,
            "status": "success",
            "event_time_utc": stats["event_time_utc"],
            "pre_tvl_avg": stats["pre"]["tvl_avg"],
            "post_tvl_avg": stats["post"]["tvl_avg"],
            "abnormal_change": stats["abnormal_change"],
            "pre_days": pre_days,
            "post_days": post_days
        }

    except Exception as e:
        return {
            "protocol_slug": None,
            "status": "error",
            "error": f"Unexpected error: {str(e)}"
        }