import importlib
from pathlib import Path

import pandas as pd
import pytest

m = importlib.import_module("agentics.mcp.defillama_mcp")
cache = importlib.import_module("agentics.mcp.defillama_protocol_cache")


def _seed_protocols(db_path: Path) -> None:
    records = [
        {"slug": "lido", "name": "Lido DAO", "symbol": "LDO"},
        {"slug": "lido-xyz", "name": "Lido XYZ", "symbol": "LDX"},
        {"slug": "compound", "name": "Compound", "symbol": "COMP"},
    ]
    cache.refresh_protocols_cache(db_path, lambda: records, ttl_hours=0)


def test_resolve_protocol_candidates(monkeypatch, tmp_path):
    db_path = tmp_path / "cache.sqlite"
    _seed_protocols(db_path)

    monkeypatch.setattr(m, "DB_PATH", db_path)
    monkeypatch.setattr(m, "_refresh_meta", lambda ttl_hours=24: 0)

    result = m.resolve_protocol_impl("lido-snapshot.eth", top_n=3)
    candidates = result.get("candidates") or []

    assert len(candidates) == 3
    assert [c["slug"] for c in candidates[:2]] == ["lido", "lido-xyz"]
    assert candidates[0]["matches"][0]["kind"] == "exact"


def test_event_stats_window():
    dates = pd.date_range("2024-01-01", periods=7, freq="D", tz="UTC").tz_convert(None)
    tvl = [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0]
    df = pd.DataFrame({"date": dates, "tvl": tvl})

    stats = m.event_stats_tvl(df, "2024-01-04T00:00:00Z", pre_days=2, post_days=2)

    assert stats["pre"]["n_days"] == 2
    assert stats["pre"]["tvl_total"] == pytest.approx(230.0)
    assert stats["post"]["tvl_avg"] == pytest.approx(145.0)
    assert stats["abnormal_change"] == pytest.approx(0.26087, rel=1e-5)
