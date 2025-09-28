"""Offline CMC MCP reading cmc_historical_daily_2013_2025.parquet.

Tools: resolve_tokens, price_window (1d only), healthcheck.
Set CMC_OFFLINE_PARQUET if the snapshot lives elsewhere.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Annotated

import duckdb
import pandas as pd
from pydantic import Field

try:
    from fastmcp import FastMCP
except Exception:  # pragma: no cover - fallback for packaged runtime
    from mcp.server.fastmcp import FastMCP


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MCP_NAME = "cmc_price_offline"
DATA_FILE_NAME = "cmc_historical_daily_2013_2025.parquet"

mcp = FastMCP(MCP_NAME)


def _pick_parquet() -> Path:
    """Locate the parquet snapshot respecting env override."""
    env_path = os.environ.get("CMC_OFFLINE_PARQUET", "").strip()
    candidates = ([Path(env_path).expanduser()] if env_path else []) + [
        PROJECT_ROOT / DATA_FILE_NAME,
        PROJECT_ROOT / "data" / DATA_FILE_NAME,
    ]
    for cand in candidates:
        if cand and cand.exists():
            return cand.resolve()
    raise FileNotFoundError("cmc offline parquet missing; set CMC_OFFLINE_PARQUET")


@lru_cache(maxsize=1)
def parquet_path() -> Path:
    return _pick_parquet()


def _duck() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=2")
    return con


@dataclass(frozen=True)
class TokenInfo:
    ucid: int
    symbol: Optional[str]
    slug: Optional[str]
    name: Optional[str]
    first_date: date
    last_date: date

    @property
    def payload(self) -> Dict[str, Any]:
        return {
            "id": self.ucid,
            "symbol": self.symbol,
            "slug": self.slug,
            "name": self.name,
            "first_date": self.first_date.isoformat(),
            "last_date": self.last_date.isoformat(),
        }


class TokenIndex:
    """Lazy index for token metadata extracted from the parquet snapshot."""

    def __init__(self, rows: Iterable[TokenInfo]):
        rows = list(rows)
        self.by_ucid: Dict[int, TokenInfo] = {info.ucid: info for info in rows}
        self.by_symbol: Dict[str, List[TokenInfo]] = {}
        self.by_slug: Dict[str, TokenInfo] = {}
        self.name_bank: List[tuple[str, TokenInfo]] = []
        for info in rows:
            if info.symbol:
                self.by_symbol.setdefault(info.symbol.lower(), []).append(info)
            if info.slug:
                self.by_slug.setdefault(info.slug.lower(), info)
            if info.name:
                self.name_bank.append((info.name.lower(), info))
        self.name_bank.sort(key=lambda t: t[0])
        self.min_date = min(info.first_date for info in rows)
        self.max_date = max(info.last_date for info in rows)

    @property
    def date_bounds(self) -> Dict[str, str]:
        return {"start": self.min_date.isoformat(), "end": self.max_date.isoformat()}

    def resolve_identifier(self, token: str | int) -> TokenInfo:
        needle = str(token).strip()
        if not needle:
            raise ValueError("token identifier must be non-empty")
        if needle.isdigit():
            ucid = int(needle)
            info = self.by_ucid.get(ucid)
            if info:
                return info
        lowered = needle.lower()
        if lowered in self.by_symbol:
            return self.by_symbol[lowered][0]
        if lowered in self.by_slug:
            return self.by_slug[lowered]
        for name, info in self.name_bank:
            if lowered in name:
                return info
        raise ValueError(f"unrecognized token identifier: {token}")

    def search(self, needle: str, limit: int = 5) -> List[TokenInfo]:
        lowered = needle.lower().strip()
        out: List[TokenInfo] = []
        seen: set[int] = set()

        def push(candidate: Optional[TokenInfo]) -> None:
            if candidate and candidate.ucid not in seen:
                out.append(candidate)
                seen.add(candidate.ucid)

        if lowered.isdigit():
            push(self.by_ucid.get(int(lowered)))
        push(self.by_slug.get(lowered))
        for info in self.by_symbol.get(lowered, []):
            push(info)
        if lowered and len(out) < limit:
            for name, info in self.name_bank:
                if lowered in name:
                    push(info)
                if len(out) >= limit:
                    break
        return out[:limit]


@lru_cache(maxsize=1)
def token_index() -> TokenIndex:
    sql = """
        SELECT
            ucid::INT AS ucid,
            arg_max(name, date) AS name,
            arg_max(symbol, date) AS symbol,
            arg_max(slug, date) AS slug,
            min(date)::DATE AS first_date,
            max(date)::DATE AS last_date
        FROM read_parquet(?)
        GROUP BY ucid
    """
    con = _duck()
    try:
        df: pd.DataFrame = con.execute(sql, [str(parquet_path())]).fetch_df()
    finally:
        con.close()
    clean = lambda v: v if pd.notnull(v) else None  # noqa: E731 - tiny helper
    infos = [
        TokenInfo(
            int(row.ucid),
            clean(row.symbol),
            clean(row.slug),
            clean(row.name),
            pd.to_datetime(row.first_date).date(),
            pd.to_datetime(row.last_date).date(),
        )
        for row in df.itertuples(index=False)
    ]
    return TokenIndex(infos)


DATE_FMT = "%Y-%m-%d"


def _parse_date(value: str, *, label: str) -> date:
    try:
        return datetime.strptime(value, DATE_FMT).date()
    except ValueError as exc:
        raise ValueError(f"{label} must be YYYY-MM-DD") from exc


def _validate_range(start: date, end: date) -> None:
    if start > end:
        raise ValueError("start_date must be on or before end_date")


def price_window_impl(token: str | int, start_date: str, end_date: str) -> Dict[str, Any]:
    idx = token_index()
    info = idx.resolve_identifier(token)
    start = _parse_date(start_date, label="start_date")
    end = _parse_date(end_date, label="end_date")
    _validate_range(start, end)
    bounds = idx.date_bounds
    if start.isoformat() < bounds["start"] or end.isoformat() > bounds["end"]:
        raise ValueError(
            f"requested range {start}..{end} outside dataset bounds {bounds['start']}..{bounds['end']}"
        )
    sql = """
        SELECT
            date::DATE AS date,
            price_USD AS price_usd,
            volume24h_USD AS volume_usd,
            marketCap_USD AS marketcap_usd,
            percentChange24h_USD AS pct_change_24h,
            percentChange7d_USD AS pct_change_7d
        FROM read_parquet(?)
        WHERE ucid = ? AND date BETWEEN ? AND ?
        ORDER BY date
    """
    con = _duck()
    try:
        df = con.execute(
            sql,
            [
                str(parquet_path()),
                info.ucid,
                start.isoformat(),
                end.isoformat(),
            ],
        ).fetch_df()
    finally:
        con.close()
    if df.empty:
        series: List[Dict[str, Any]] = []
    else:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime(DATE_FMT)
        series = df.to_dict(orient="records")
    return {
        "token": info.payload,
        "interval": "1d",
        "range": {"start": start.isoformat(), "end": end.isoformat()},
        "count": len(series),
        "series": series,
    }


def resolve_tokens_impl(project_hint: str, prefer_governance: bool = True) -> Dict[str, Any]:
    hint = (project_hint or "").strip()
    if not hint:
        raise ValueError("project_hint must be provided")
    idx = token_index()
    matches = idx.search(hint, limit=6)
    candidates = [info.payload for info in matches]
    primary = candidates[0] if candidates else None
    # Offline dataset cannot disambiguate governance vs native; expose best match as both.
    return {
        "project_hint": hint,
        "governance_token": primary,
        "native_token": primary,
        "candidates": candidates,
        "notes": "offline-cmc: governance/native identical due to dataset limits",
    }


def healthcheck_impl() -> Dict[str, Any]:
    path = parquet_path()
    idx = token_index()
    return {
        "ok": True,
        "service": MCP_NAME,
        "parquet": str(path),
        "tokens": len(idx.by_ucid),
        "date_bounds": idx.date_bounds,
    }


@mcp.tool(
    name="resolve_tokens",
    title="Resolve Token Candidates (Offline)",
    description="Find and rank cryptocurrency token candidates from offline CMC data based on a project hint. Prioritizes governance tokens when available. Use this for offline token discovery without API rate limits.",
    annotations={
        "readOnlyHint": True,
        "openWorldHint": False,
        "idempotentHint": True
    }
)
def resolve_tokens(
    project_hint: Annotated[str, Field(
        description="Project name or hint to search for tokens (e.g., 'Aave', 'uniswap', 'compound')",
        min_length=1,
        max_length=100
    )],
    prefer_governance: Annotated[bool, Field(
        description="Whether to prioritize governance tokens over native tokens"
    )] = True
) -> Dict[str, Any]:
    return resolve_tokens_impl(project_hint, prefer_governance)


@mcp.tool(
    name="price_window",
    title="Get Price Data Window (Offline)",
    description="Retrieve historical price data for a token within a date range from offline CMC parquet data. Only supports daily (1d) intervals. Use this for historical price analysis without API costs.",
    annotations={
        "readOnlyHint": True,
        "openWorldHint": False,
        "idempotentHint": True
    }
)
def price_window(
    token: Annotated[str, Field(
        description="Token symbol or CMC ID (e.g., 'BTC', 'ETH', '1027')",
        min_length=1,
        max_length=50
    )],
    start_date: Annotated[str, Field(
        description="Start date in YYYY-MM-DD format",
        pattern=r"^\d{4}-\d{2}-\d{2}$"
    )],
    end_date: Annotated[str, Field(
        description="End date in YYYY-MM-DD format",
        pattern=r"^\d{4}-\d{2}-\d{2}$"
    )],
    interval: Annotated[str, Field(
        description="Price data interval (only '1d' supported for offline data)"
    )] = "1d"
) -> Dict[str, Any]:
    if interval.lower() != "1d":
        raise ValueError("offline cmc supports interval='1d' only")
    return price_window_impl(token, start_date, end_date)


@mcp.tool(
    name="healthcheck",
    title="CMC Offline Service Health Check",
    description="Check the health status of the CMC Offline MCP service. Verifies parquet data availability and service readiness.",
    annotations={
        "readOnlyHint": True,
        "openWorldHint": False,
        "idempotentHint": True
    }
)
def healthcheck() -> Dict[str, Any]:
    return healthcheck_impl()


if __name__ == "__main__":
    mcp.run(transport="stdio")
