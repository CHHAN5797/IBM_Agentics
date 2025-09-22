#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smoke test for snapshot_api.py — verifies get_votes_all works end-to-end.

Usage:
  python smoketest.py --url <snapshot_proposal_url>
  or
  python smoketest.py --id <proposal_id>

Notes:
- Requires snapshot_api.py in the same folder (or /mnt/data if you run inside container).
- You can set SNAPSHOT_API env var if you want to use a different GraphQL endpoint.
"""

import argparse
import importlib.util
import json
import pathlib
import sys

# --- load snapshot_api.py dynamically ---
SNAPSHOT_API_PATH = pathlib.Path(__file__).parent / "snapshot_api.py"
if not SNAPSHOT_API_PATH.exists():
    # fallback: maybe user has it in /mnt/data
    SNAPSHOT_API_PATH = pathlib.Path("/mnt/data/snapshot_api.py")

spec = importlib.util.spec_from_file_location("snapshot_api", SNAPSHOT_API_PATH)
snap = importlib.util.module_from_spec(spec)
sys.modules["snapshot_api"] = snap
spec.loader.exec_module(snap)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", help="Proposal ID (0x...64 hex)")
    parser.add_argument("--url", help="Snapshot proposal URL")
    parser.add_argument("--limit", type=int, default=1000, help="Page size for get_votes_all")
    args = parser.parse_args()

    if args.url:
        pid = snap.resolve_proposal_id_from_url(args.url)
        if not pid:
            print(f"Could not extract proposal id from URL: {args.url}")
            sys.exit(1)
    elif args.id:
        pid = args.id.strip()
    else:
        print("You must provide either --url or --id")
        sys.exit(1)

    print(f"[1] Proposal id = {pid}")

    meta = snap.get_proposal_by_id(pid)
    print("\n=== Proposal Meta ===")
    print(json.dumps(meta, indent=2, ensure_ascii=False))

    res = snap.get_votes_all(pid, page_size=args.limit, normalize_for_timeline=True)
    votes = res.get("votes") or []
    print("\n=== Votes Result ===")
    print(f"Total votes: {res.get('count')} (normalized_note={res.get('normalized_note')})")
    print("First 3 votes sample:")
    print(json.dumps(votes[:3], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
