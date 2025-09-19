import importlib
import pandas as pd

m = importlib.import_module("agentics.mcp.defillama_mcp")

def test_bootstrap_and_guess():
    # Bootstrap protocol meta
    count = m.refresh_protocols_cache()
    assert count >= 1

    # Resolve from Snapshot space
    out = m.resolve_protocol_impl("lido-snapshot.eth")
    assert out["protocol_slug"] is not None

def test_refresh_and_event():
    slug = "lido"
    rows = m.refresh_tvl_cache(slug)
    assert rows >= 1

    df = m.load_tvl(slug)
    assert isinstance(df, pd.DataFrame)
    assert {"date", "tvl"}.issubset(df.columns)

    stats = m.event_stats_tvl(df, pd.Timestamp("2021-04-01T00:00:00Z").tz_convert(None), 7, 7)
    assert "t0_aligned" in stats
    assert isinstance(stats["window_series"], list) and len(stats["window_series"]) >= 5
