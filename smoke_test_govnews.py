#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smoke test (latest mode): fetch the 5 most recent items from each RSS feed
(CoinDesk, The Block, Decrypt), regardless of date window or keywords.

- No dependency on search windows or project hints.
- Does not require any API keys.
"""

from __future__ import annotations
import importlib
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import json

UA = {"User-Agent": "Mozilla/5.0 (govnews_smoke_latest)"}

def jprint(obj, maxlen=2000):
    s = json.dumps(obj, indent=2, default=str)
    print(s if len(s) <= maxlen else s[:maxlen] + "\n... (truncated) ...")

def fetch_latest(feed_url: str, per_feed: int = 5):
    """Return a list of the latest N items from the RSS feed."""
    try:
        r = requests.get(feed_url, headers=UA, timeout=20)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        items = root.findall(".//item")
        out = []
        for it in items[:per_feed]:
            title = it.findtext("title") or ""
            link = it.findtext("link") or ""
            desc = it.findtext("description") or ""
            pub = it.findtext("pubDate") or ""
            try:
                ts = pd.to_datetime(pub, utc=True).tz_localize(None).isoformat()
            except Exception:
                ts = None
            out.append({
                "title": title,
                "url": link,
                "published_at": ts,
                "snippet": desc,
            })
        return out
    except Exception as e:
        return [{"error": f"fetch_failed: {e}"}]

def main():
    # Import the MCP module just to reuse the RSS_FEEDS mapping
    m = importlib.import_module("agentics.mcp.govnews_mcp")
    feeds = getattr(m, "RSS_FEEDS", {
        "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "TheBlock": "https://www.theblock.co/rss",
        "Decrypt":  "https://decrypt.co/feed",
    })

    print("=== Latest headlines (5 per feed) ===")
    for name, url in feeds.items():
        print(f"\n--- {name} ---")
        rows = fetch_latest(url, per_feed=5)
        jprint(rows)

    print("\n✅ latest mode smoke ok")

if __name__ == "__main__":
    main()
