#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CMC diagnostics: determine whether failures are due to missing CMC Pro API key
and/or missing metadata.csv, and verify that web search fallback works.

Usage:
  export PYTHONPATH=src
  python cmc_diag.py --term LDO
"""

from __future__ import annotations
import os, json, sys, argparse, time
import requests
import pandas as pd

def mask(s: str, keep: int = 4) -> str:
    if not s: return "None"
    return ("*" * max(0, len(s) - keep)) + s[-keep:]

def try_read_csv(path: str) -> tuple[bool, int, str]:
    try:
        if not os.path.isfile(path):
            return False, 0, "not_a_file"
        df = pd.read_csv(path, nrows=5, encoding_errors="ignore")
        return True, len(df.index), "ok"
    except Exception as e:
        return False, 0, f"error:{e}"

def try_cmc_pro_map(api_key: str | None) -> tuple[bool, str]:
    if not api_key:
        return False, "no_api_key"
    try:
        r = requests.get(
            "https://pro-api.coinmarketcap.com/v1/cryptocurrency/map",
            headers={"X-CMC_PRO_API_KEY": api_key, "Accept": "application/json"},
            params={"listing_status": "active,untracked,inactive", "limit": 5},
            timeout=20,
        )
        if r.status_code == 200:
            data = r.json()
            ok = isinstance(data.get("data"), list)
            return ok, ("ok" if ok else f"bad_payload:{data}")
        return False, f"http_{r.status_code}:{r.text[:200]}"
    except Exception as e:
        return False, f"error:{e}"

def deep_iter(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            for z in deep_iter(v): yield z
    elif isinstance(obj, list):
        for it in obj:
            for z in deep_iter(it): yield z

def search_cmc_web(term: str) -> tuple[bool, int | None, str]:
    try:
        r = requests.get("https://api.coinmarketcap.com/data-api/v3/search",
                         params={"q": term}, timeout=20)
        if r.status_code != 200:
            return False, None, f"http_{r.status_code}"
        data = r.json()
        best = None
        s_norm = term.lower()
        for node in deep_iter(data.get("data", {})):
            if not isinstance(node, dict): continue
            cid = node.get("id") or node.get("cid")
            sym = node.get("symbol") or node.get("code")
            name = node.get("name") or node.get("title")
            slug = node.get("slug")
            if not cid: continue
            score = 0
            for v in (sym, slug, name):
                if not v: continue
                v0 = str(v).lower()
                if v0 == s_norm: score += 3
                elif s_norm in v0 or v0 in s_norm: score += 1
            if best is None or score > best[0]:
                best = (score, cid, sym, slug, name)
        if best:
            return True, int(best[1]), f"match_score={best[0]} sym={best[2]} slug={best[3]} name={best[4]}"
        return False, None, "no_match"
    except Exception as e:
        return False, None, f"error:{e}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--term", type=str, default="LDO", help="symbol/slug/name to resolve (e.g., LDO, lido)")
    parser.add_argument("--metadata", type=str, nargs="*", default=[
        os.environ.get("CMC_METADATA", "").strip(),
        "/mnt/data/metadata.csv",
        "resource/metadata.csv",
        "metadata.csv",
        os.path.join(os.environ.get("DATA_DIR", "data"), "metadata.csv"),
    ], help="metadata.csv candidate paths to check")
    args = parser.parse_args()

    print("=== CMC Diagnostics ===")
    api_key = os.environ.get("CMC_PRO_API_KEY", "")
    print(f"CMC_PRO_API_KEY present: {'YES' if api_key else 'NO'} (masked: {mask(api_key)})")

    # 1) metadata.csv check
    any_meta = False
    for p in [x for x in args.metadata if x]:
        ok, nrows, msg = try_read_csv(p)
        print(f"metadata check: {p} -> exists_file={ok}, sample_rows={nrows}, note={msg}")
        any_meta = any_meta or ok

    # 2) Pro map
    ok_map, note_map = try_cmc_pro_map(api_key if api_key else None)
    print(f"CMC Pro /map: ok={ok_map}, note={note_map}")

    # 3) web search fallback
    ok_search, cid, note_search = search_cmc_web(args.term)
    print(f"CMC web search for '{args.term}': ok={ok_search}, cid={cid}, note={note_search}")

    # 4) final diagnosis
    print("\n--- Diagnosis ---")
    if any_meta or ok_map:
        print("Likely NOT due to missing API key (metadata available via file or Pro API).")
    else:
        if ok_search and cid:
            print("No metadata/Pro API, but web search fallback WORKS. Failures (if any) are likely elsewhere.")
        else:
            print("No metadata.csv and Pro API unavailable, and web search failed →")
            print("=> Very likely the error is due to missing CMC metadata sources (API key and file).")
            print("   Fix: provide CMC_PRO_API_KEY or point CMC_METADATA to a valid metadata.csv")

if __name__ == "__main__":
    main()
