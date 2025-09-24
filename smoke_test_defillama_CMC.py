#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end smoke test for:
 - DeFiLlama TVL MCP (resolve -> refresh -> event window)
 - CMC Price MCP (resolve tokens -> refresh -> price window)

It prints a short summary and writes two CSVs with the event-window series.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import pandas as pd


# ---------- Pretty helpers ----------
def jprint(x, head=2000):
    print(json.dumps(x, indent=2, default=str)[:head])


def pick_token_id(resolved: dict) -> tuple[int | None, str | None]:
    """Choose governance token id if available, otherwise native; return (id, symbol)."""
    gov = (resolved or {}).get("governance_token") or {}
    nat = (resolved or {}).get("native_token") or {}
    for cand in (gov, nat):
        if cand and cand.get("id") is not None:
            return int(cand["id"]), cand.get("symbol")
    return None, None


# ---------- Main test ----------
def main():
    parser = argparse.ArgumentParser(description="Smoke test for TVL + CMC MCP")
    parser.add_argument(
        "--project_hint",
        type=str,
        default="lido-snapshot.eth",
        help="Protocol/DAO hint (slug/name/snapshot space), e.g., 'lido-snapshot.eth'",
    )
    parser.add_argument(
        "--event_time_utc",
        type=str,
        default="2021-04-01T00:00:00Z",
        help="Event timestamp in ISO8601 (UTC)",
    )
    parser.add_argument("--pre_days", type=int, default=7)
    parser.add_argument("--post_days", type=int, default=7)
    parser.add_argument(
        "--price_interval", type=str, default="1d", choices=["1d", "1h"]
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default="2020-01-01",
        help="(price) fetch start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default="2025-12-31",
        help="(price) fetch end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="data/exports",
        help="Directory to write CSV outputs",
    )
    args = parser.parse_args()

    # Ensure PYTHONPATH=src set by user before running.
    try:
        tvl = importlib.import_module("agentics.mcp.defillama_mcp")
        cmc = importlib.import_module("agentics.mcp.cmc_mcp")
    except Exception as e:
        print(
            "Failed to import MCP modules. Make sure PYTHONPATH=src is set.",
            file=sys.stderr,
        )
        raise

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("\n========== 1) DeFiLlama TVL: resolve protocol ==========")
    # Bootstrap meta (safe to call repeatedly)
    tvl.refresh_protocols_cache()
    res_proto = tvl.resolve_protocol_impl(args.project_hint)
    jprint(res_proto)
    slug = next(
        (c.get("slug") for c in (res_proto.get("candidates") or []) if c.get("slug")),
        None,
    )
    if not slug:
        # One more attempt after cache refresh (defensive)
        tvl.refresh_protocols_cache()
        res_proto = tvl.resolve_protocol_impl(args.project_hint)
        slug = next(
            (
                c.get("slug")
                for c in (res_proto.get("candidates") or [])
                if c.get("slug")
            ),
            None,
        )
    if not slug:
        print(
            f"[TVL] Failed to resolve protocol from hint: {args.project_hint}",
            file=sys.stderr,
        )
        sys.exit(2)

    print("\n========== 2) DeFiLlama TVL: refresh + event window ==========")
    ref_tvl = tvl.refresh_tvl_cache(slug)
    print(f"TVL rows cached for '{slug}':", ref_tvl)
    evt_tvl = tvl.event_window_impl(
        protocol_or_space=args.project_hint,
        event_time_utc=args.event_time_utc,
        pre_days=args.pre_days,
        post_days=args.post_days,
        event_id="smoke-tvl-event",
    )
    jprint(
        {
            "protocol_slug": evt_tvl.get("protocol_slug"),
            "stats_keys": list((evt_tvl.get("stats") or {}).keys()),
        }
    )
    # write window series to CSV
    tvl_series = (evt_tvl.get("stats") or {}).get("window_series") or []
    tvl_csv = outdir / f"tvl_window_{slug}.csv"
    pd.DataFrame(tvl_series).to_csv(tvl_csv, index=False)
    print(f"[OK] TVL window series -> {tvl_csv}")

    print("\n========== 3) CMC Price: resolve tokens ==========")
    res_tok = cmc.resolve_tokens_impl(args.project_hint, prefer_governance=True)
    jprint(
        {
            "governance_token": res_tok.get("governance_token"),
            "native_token": res_tok.get("native_token"),
            "num_candidates": len(res_tok.get("candidates") or []),
        }
    )
    tok_id, tok_sym = pick_token_id(res_tok)
    tok_label = tok_sym or str(tok_id)
    if tok_id is None:
        # try using symbol 'LDO' for lido as last resort (demo)
        fallback = "LDO" if "lido" in args.project_hint.lower() else None
        if fallback is None:
            print(
                f"[CMC] Cannot resolve token id from hint '{args.project_hint}'",
                file=sys.stderr,
            )
            sys.exit(3)
        print(f"[CMC] Falling back to symbol: {fallback}")
        tok_id = fallback
        tok_label = fallback

    print("\n========== 4) CMC Price: refresh + price window ==========")
    ref_px = cmc.refresh_price_impl(
        token=tok_id,
        interval=args.price_interval,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    jprint(
        {
            "token_id": ref_px.get("token_id"),
            "interval": ref_px.get("interval"),
            "inserted_rows": ref_px.get("rows"),
        }
    )

    evt_px = cmc.price_window_impl(
        token=tok_id,
        event_time_utc=args.event_time_utc,
        pre_days=args.pre_days,
        post_days=args.post_days,
        interval=args.price_interval,
    )
    px_stats = evt_px.get("stats") or {}
    jprint({"token_id": evt_px.get("token_id"), "stats_keys": list(px_stats.keys())})
    # write window series to CSV
    px_series = px_stats.get("window_series") or []
    px_csv = outdir / f"price_window_{tok_label}.csv"
    pd.DataFrame(px_series).to_csv(px_csv, index=False)
    print(f"[OK] Price window series -> {px_csv}")

    # ---------- tiny combined summary ----------
    print("\n========== ✅ Combined summary ==========")
    summary = {
        "project_hint": args.project_hint,
        "protocol_slug": slug,
        "event_time_utc": args.event_time_utc,
        "pre_days": args.pre_days,
        "post_days": args.post_days,
        "tvl_t0": (evt_tvl.get("stats") or {}).get("tvl_t0"),
        "tvl_delta_0_to_postK": (evt_tvl.get("stats") or {}).get(
            "delta_pct_0_to_postK"
        ),
        "price_t0": px_stats.get("px_t0"),
        "welch_z_on_log_ret": px_stats.get("welch_z_on_log_ret"),
        "files": {"tvl_csv": str(tvl_csv), "price_csv": str(px_csv)},
    }
    jprint(summary, head=4000)
    print("\nAll good. 🎉")


if __name__ == "__main__":
    # Tip: ensure PYTHONPATH=src
    # Example:
    #   export PYTHONPATH=src
    #   python smoke_test_all.py --project_hint lido-snapshot.eth --event_time_utc 2021-04-01T00:00:00Z
    main()
