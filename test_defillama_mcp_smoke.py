# tests/test_defillama_mcp_smoke.py
from __future__ import annotations
import json
from datetime import datetime, timezone

# Direct import (no MCP server spin-up)
from agentics.mcp.defillama_mcp import (
    refresh_protocol,   # <- slug-only
    event_window,       # <- slug-only
    _tvl_path,          # for local cache check
)

def jprint(obj, maxlen=2000):
    s = json.dumps(obj, indent=2, default=str)
    print(s if len(s) <= maxlen else s[:maxlen] + "\n... (truncated) ...")

def main():
    """
    Smoke test for the slug-only DeFiLlama MCP.
    - Uses the 'aave' slug directly (no link parsing).
    - Refreshes local parquet cache (if missing, downloads & writes).
    - Computes a 3/3 pre/post TVL window around 'now' in UTC.
    """
    slug = "aave"
    print(f"[1] slug={slug}")

    # 2) refresh: ensure parquet exists / updated for this slug
    print("[2] refresh_protocol ...")
    ref = refresh_protocol(slug=slug)
    jprint(ref)

    # Check parquet existence
    fpath = _tvl_path(slug)
    print(f"[check] parquet exists: {fpath} -> {fpath.exists()}")

    # 3) event window: use 'now' as the event time (UTC), 3 days pre/post
    event_time_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[3] event_window at {event_time_utc} (pre=3, post=3) ...")
    win = event_window(
        slug=slug,
        event_time_utc=event_time_utc,
        pre_days=3,
        post_days=3,
    )
    jprint(win)

    # 4) Key metric: abnormal_change (ratio). Multiply by 100 to get %.
    stats = (win or {}).get("stats", {})
    abn = stats.get("abnormal_change")
    print(f"[result] abnormal_change={abn}  (×100 => %)")
    if abn is not None:
        pct = round(float(abn) * 100.0, 4)
        print(f"[result] TVL impact ≈ {pct}%")

if __name__ == "__main__":
    main()
