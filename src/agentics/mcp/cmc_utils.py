# -*- coding: utf-8 -*-
"""Shared utilities for CMC price impact analysis."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, List, Optional

import pandas as pd


CSV_PATH = os.getenv(
    "DAO_REGISTRY_CSV",
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "assets_registry",
            "dao_registry.csv"
        )
    ),
)


def _norm(s: Optional[str]) -> str:
    """Normalize string by stripping whitespace."""
    return (s or "").strip()


def _normalize_row(r: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a single CSV row."""
    low = {
        (k or "").strip().lower(): (v if v is not None else "")
        for k, v in r.items()
    }
    return {
        "space": _norm(low.get("space")),
        "cmc_ucid": _norm(
            low.get("cmc_ucid") or low.get("ucid") or low.get("cmc id") or ""
        ),
        "cmc_ticker": _norm(
            low.get("cmc_ticker") or low.get("ticker") or ""
        ),
        "defillama_name": _norm(
            low.get("name_defillama")
            or low.get("defillama_name")
            or low.get("name")
            or ""
        ),
        "contracts": _norm(
            low.get("contracts") or low.get("contract") or ""
        ),
    }


def _load_registry_rows() -> List[Dict[str, Any]]:
    """Load DAO registry CSV and normalize columns."""
    if not os.path.exists(CSV_PATH):
        raise RuntimeError(f"DAO registry CSV not found: {CSV_PATH}")
    with open(CSV_PATH, "r", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        return [_normalize_row(r) for r in reader]


def resolve_token_from_snapshot_space(
    space: str
) -> Optional[Dict[str, Any]]:
    """
    Resolve CMC token from Snapshot space.

    Returns dict with cmc_ucid and ticker, or None.
    """
    if not space:
        return None
    try:
        rows = _load_registry_rows()
    except Exception:
        return None
    space_lower = space.strip().lower()
    for row in rows:
        if (row["space"] or "").strip().lower() == space_lower:
            if row["cmc_ucid"] and row["cmc_ticker"]:
                return {
                    "cmc_ucid": row["cmc_ucid"],
                    "ticker": row["cmc_ticker"]
                }
    return None


@dataclass
class PriceWindowResult:
    """Price statistics for a time window."""
    n_days: int
    price_total: float
    price_avg: float


def _window_slice(
    df: pd.DataFrame, t0: pd.Timestamp, pre_days: int, post_days: int
) -> Dict[str, pd.DataFrame]:
    """Slice dataframe into pre/at/post windows."""
    df = df.copy().sort_values("date")
    t0 = pd.to_datetime(t0, utc=True).tz_localize(None)
    pre_from = t0 - timedelta(days=pre_days)
    pre_to = t0 - timedelta(days=1)
    post_from = t0 + timedelta(days=1)
    post_to = t0 + timedelta(days=post_days)
    return {
        "pre": df[(df["date"] >= pre_from) & (df["date"] <= pre_to)],
        "at": df[(df["date"] >= t0) & (df["date"] <= t0)],
        "post": df[(df["date"] >= post_from) & (df["date"] <= post_to)],
    }


def _win_stats_price(seg: pd.DataFrame) -> PriceWindowResult:
    """Compute price statistics for a window segment."""
    if seg.empty:
        return PriceWindowResult(0, 0.0, 0.0)
    n = int(seg["date"].nunique())
    tot = float(seg["price_usd"].sum())
    avg = float(tot / max(1, n))
    return PriceWindowResult(n_days=n, price_total=tot, price_avg=avg)


def event_stats_price(
    df: pd.DataFrame, event_time_utc: str, pre_days: int = 7,
    post_days: int = 7
) -> Dict[str, Any]:
    """Compute price statistics around an event time."""
    t0 = pd.to_datetime(event_time_utc, utc=True)
    segs = _window_slice(df, t0, pre_days, post_days)
    pre = _win_stats_price(segs["pre"])
    post = _win_stats_price(segs["post"])
    abn = (
        None if pre.price_avg == 0
        else (post.price_avg / pre.price_avg - 1.0)
    )
    return {
        "event_time_utc": pd.Timestamp(t0).tz_localize(None).isoformat(),
        "pre": {
            "n_days": pre.n_days,
            "price_total": pre.price_total,
            "price_avg": pre.price_avg,
        },
        "post": {
            "n_days": post.n_days,
            "price_total": post.price_total,
            "price_avg": post.price_avg,
        },
        "abnormal_change": (
            None if abn is None else float(round(abn, 6))
        ),
    }


def _error_resp(
    token_id: Optional[str], ticker: Optional[str], status: str,
    error: str
) -> Dict[str, Any]:
    """Build error response dict."""
    return {
        "token_id": token_id,
        "ticker": ticker,
        "status": status,
        "error": error,
    }


def _fetch_price_data(
    token_id: str, start_date: str, end_date: str
) -> Dict[str, Any]:
    """Fetch price data from CMC offline module."""
    from agentics.mcp.cmc_offline_mcp import price_window_impl
    return price_window_impl(
        token=token_id, start_date=start_date, end_date=end_date
    )


def _build_success_resp(
    token_id: str, ticker: str, stats: Dict[str, Any], pre_days: int,
    post_days: int
) -> Dict[str, Any]:
    """Build success response dict."""
    return {
        "token_id": token_id,
        "ticker": ticker,
        "status": "success",
        "event_time_utc": stats["event_time_utc"],
        "pre_price_avg": stats["pre"]["price_avg"],
        "post_price_avg": stats["post"]["price_avg"],
        "abnormal_change": stats["abnormal_change"],
        "pre_days": pre_days,
        "post_days": post_days,
    }


def _prepare_price_dataframe(
    token_id: str, ticker: str, proposal_end_utc: str, pre_days: int,
    post_days: int
) -> tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    """Fetch and prepare price dataframe, return (df, error_response)."""
    try:
        event_dt = pd.to_datetime(proposal_end_utc, utc=True)
        start_date = (
            event_dt - timedelta(days=pre_days)
        ).strftime("%Y-%m-%d")
        end_date = (event_dt + timedelta(days=post_days)).strftime("%Y-%m-%d")
        price_data = _fetch_price_data(token_id, start_date, end_date)
    except ImportError:
        return None, _error_resp(
            token_id, ticker, "error", "CMC offline module not available"
        )
    except Exception as e:
        return None, _error_resp(
            token_id, ticker, "data_unavailable",
            f"Could not fetch price data: {str(e)}"
        )
    series = price_data.get("series") or []
    if not series:
        return None, _error_resp(
            token_id, ticker, "data_unavailable",
            "No price data available for date range"
        )
    df = pd.DataFrame(series)
    df["date"] = pd.to_datetime(df["date"])
    return df, None


def get_price_impact_for_proposal(
    space: str, proposal_end_utc: str, pre_days: int = 7,
    post_days: int = 7
) -> Dict[str, Any]:
    """Get price impact analysis for a proposal."""
    try:
        token_info = resolve_token_from_snapshot_space(space)
        if not token_info:
            return _error_resp(
                None, None, "no_token_mapping",
                f"Could not map space '{space}' to CMC token"
            )
        token_id = token_info["cmc_ucid"]
        ticker = token_info["ticker"]
        df, error = _prepare_price_dataframe(
            token_id, ticker, proposal_end_utc, pre_days, post_days
        )
        if error:
            return error
        stats = event_stats_price(df, proposal_end_utc, pre_days, post_days)
        return _build_success_resp(
            token_id, ticker, stats, pre_days, post_days
        )
    except Exception as e:
        return _error_resp(
            None, None, "error", f"Unexpected error: {str(e)}"
        )