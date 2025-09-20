"""Smoke tests for the offline CMC MCP module."""

from agentics.mcp import cmc_offline_mcp as cmc


def test_resolve_tokens_prefers_slug_match():
    result = cmc.resolve_tokens_impl("ethereum")
    primary = result["governance_token"]
    assert primary is not None
    assert primary["slug"] == "ethereum"
    assert primary["symbol"] == "ETH"


def test_price_window_returns_daily_series():
    window = cmc.price_window_impl("BTC", "2020-01-01", "2020-01-03")
    assert window["count"] == 3
    dates = [row["date"] for row in window["series"]]
    assert dates == ["2020-01-01", "2020-01-02", "2020-01-03"]
    assert window["series"][0]["price_usd"] > 0


def test_healthcheck_reports_bounds_and_count():
    health = cmc.healthcheck_impl()
    assert health["ok"] is True
    assert health["tokens"] > 1000
    bounds = health["date_bounds"]
    assert bounds["start"] <= "2013-04-28"
    assert bounds["end"] >= "2025-09-01"
