# feat/agent Branch – What's New

## Things We Just Added
- Offline CoinMarketCap helper (`cmc_offline_mcp`) lets the tools read prices from a local Parquet file, so no API key or internet calls are needed.
- A new example script (`examples/agentics_proposal_decision.py`) walks through gathering data and drafting a yes/no voting note for a Snapshot proposal.
- Lightweight smoke tests (`tests/test_cmc_offline_mcp.py`) double-check that the offline price data answers basic questions correctly.
- README now points out where to put the price history file and how to launch the offline price server.

## Things That Changed or Disappeared
- The older orchestra scripts were removed; tool loading now happens inside the new example with a simple helper.
- Several MCP modules now run directly with `python -m ...`, making it clearer how to start each tool on its own.

## Helpful Notes Before You Try It
- Keep a copy of `cmc_historical_daily_2013_2025.parquet` in the project root (or set `CMC_OFFLINE_PARQUET` to wherever you store it).
- Make sure `duckdb` is installed (added to `pyproject.toml`) so the offline reader works.
- To launch the offline price server, run `uv run python -m agentics.mcp.cmc_offline_mcp` from the project root.
- The voting example expects your `.env` to point at an LLM provider and will ask for a Snapshot proposal link when you run it.
