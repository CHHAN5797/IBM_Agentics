# -*- coding: utf-8 -*-
"""
Governance News MCP (FastMCP) — efficient RSS with conditional GET & dedupe

Tools:
  1) search_governance_news(project_hint, start_date, end_date, lang="en", max_records=100, ttl_minutes=30)
  2) proposal_news_window(project_hint, proposal_title_or_id, event_time_utc, pre_days=7, post_days=7,
                          lang="en", max_records=100, ttl_minutes=30)
  3) health()

Design goals:
  * Minimize repeated network calls:
      - Per-feed TTL
      - Conditional GET (ETag / Last-Modified) persisted in SQLite
  * Avoid duplicates:
      - URL as primary key in SQLite
      - Secondary (title+snippet) content hash
  * Keep helpers pure and testable.
"""

from __future__ import annotations
import os
import re
import json
import time
import sqlite3
import logging
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Annotated
from datetime import datetime, timedelta

import requests
import pandas as pd
import xml.etree.ElementTree as ET

try:
    from fastmcp import FastMCP
except Exception:
    from mcp.server.fastmcp import FastMCP

from pydantic import Field

# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------
DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "govnews.sqlite"

# Public RSS feeds (can be extended)
RSS_FEEDS = {
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "TheBlock": "https://www.theblock.co/rss",
    "Decrypt":  "https://decrypt.co/feed",
}

UA = {"User-Agent": "Mozilla/5.0 (govnews_mcp/1.2)"}
HTTP_TIMEOUT = int(os.getenv("GOVNEWS_HTTP_TIMEOUT", "20"))
RETRY_TOTAL = int(os.getenv("GOVNEWS_HTTP_RETRIES", "4"))
BASE_SLEEP = float(os.getenv("GOVNEWS_BASE_SLEEP", "0.25"))
LOG_LEVEL = os.getenv("GOVNEWS_LOG_LEVEL", "INFO").upper()

GOV_KEYWORDS = [
    "dao", "governance", "proposal", "vote", "snapshot", "on-chain",
    "onchain", "quorum", "delegate"
]

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
log = logging.getLogger("govnews_mcp")

mcp = FastMCP("govnews_mcp")

# ------------------------------------------------------------------------------
# DB
# ------------------------------------------------------------------------------
def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    # Articles table (URL PK; content_hash helps dedupe on content variations)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS articles (
        url TEXT PRIMARY KEY,
        title TEXT,
        source TEXT,
        lang TEXT,
        published_at TEXT,
        snippet TEXT,
        queries TEXT,
        fetched_at INTEGER,
        content_hash TEXT
    );""")
    # Feeds table for conditional GET + TTL
    conn.execute("""
    CREATE TABLE IF NOT EXISTS feeds (
        source TEXT PRIMARY KEY,
        url TEXT NOT NULL,
        etag TEXT,
        last_modified TEXT,
        last_fetch_ts INTEGER
    );""")
    conn.commit()
    return conn

def _now_ts() -> int:
    return int(time.time())

def _content_hash(title: str, snippet: str) -> str:
    h = hashlib.sha1()
    h.update((title or "").encode("utf-8", "ignore"))
    h.update(b"\x00")
    h.update((snippet or "").encode("utf-8", "ignore"))
    return h.hexdigest()

# ------------------------------------------------------------------------------
# HTTP with retries/backoff (simple)
# ------------------------------------------------------------------------------
session = requests.Session()
session.headers.update(UA)

def _sleep(i: int) -> None:
    # Linear backoff with a small base sleep to avoid hammering
    time.sleep(BASE_SLEEP * (1 + i))

def _conditional_headers(source: str) -> Dict[str, str]:
    """Build conditional GET headers from feed cache."""
    with _db() as conn:
        cur = conn.execute("SELECT etag, last_modified FROM feeds WHERE source=?;", (source,))
        row = cur.fetchone()
    h = {}
    if row:
        etag, lm = row
        if etag:
            h["If-None-Match"] = etag
        if lm:
            h["If-Modified-Since"] = lm
    return h

def _store_feed_headers(source: str, url: str, etag: Optional[str], last_modified: Optional[str]) -> None:
    with _db() as conn:
        conn.execute("""
        INSERT INTO feeds(source, url, etag, last_modified, last_fetch_ts)
        VALUES(?,?,?,?,?)
        ON CONFLICT(source) DO UPDATE SET
          url=excluded.url,
          etag=excluded.etag,
          last_modified=excluded.last_modified,
          last_fetch_ts=excluded.last_fetch_ts;
        """, (source, url, etag, last_modified, _now_ts()))
        conn.commit()

def _respect_ttl(source: str, ttl_minutes: int) -> bool:
    """Return True if feed was fetched within TTL and can be skipped."""
    if ttl_minutes <= 0:
        return False
    with _db() as conn:
        cur = conn.execute("SELECT last_fetch_ts FROM feeds WHERE source=?;", (source,))
        row = cur.fetchone()
    if not row or not row[0]:
        return False
    return (_now_ts() - int(row[0])) < ttl_minutes * 60

# ------------------------------------------------------------------------------
# RSS parsing
# ------------------------------------------------------------------------------
def _parse_rss(xml_text: str, source: str, query_label: str,
               start: datetime, end: datetime, lang: str, per_source_max: int) -> List[Dict[str, Any]]:
    """
    Very lightweight RSS parser (ElementTree). Returns normalized article dicts.
    """
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return []

    items = root.findall(".//item")
    out: List[Dict[str, Any]] = []

    for it in items:
        title = (it.findtext("title") or "").strip()
        link = (it.findtext("link") or "").strip()
        desc = (it.findtext("description") or "").strip()
        pub = it.findtext("pubDate") or ""
        # Loose date parsing with fallback to now (UTC-naive)
        try:
            ts = pd.to_datetime(pub, utc=True).tz_localize(None)
        except Exception:
            ts = pd.Timestamp.utcnow().tz_localize(None)

        # Time window filter
        if not (start <= ts.to_pydatetime() <= end):
            continue

        # Governance relevance filter (cheap keyword check)
        bucket = f"{title} {desc}".lower()
        if not any(k in bucket for k in GOV_KEYWORDS):
            continue

        rec = {
            "url": link,
            "title": title,
            "source": source,
            "lang": lang,
            "published_at": ts.isoformat(),
            "snippet": desc,
            "queries": json.dumps({"provider": "rss", "q": query_label}),
            "content_hash": _content_hash(title, desc),
        }
        out.append(rec)

        if len(out) >= per_source_max:
            break

    return out

# ------------------------------------------------------------------------------
# Fetch one feed with TTL + conditional GET
# ------------------------------------------------------------------------------
def fetch_feed_once(source: str, url: str, query_label: str,
                    start: datetime, end: datetime,
                    lang: str, per_source_max: int,
                    ttl_minutes: int = 30) -> List[Dict[str, Any]]:
    """
    Returns new (deduped) articles inserted into DB; respects TTL and conditional GET.
    """
    # Skip if fetched recently
    if _respect_ttl(source, ttl_minutes):
        log.debug("TTL skip for %s", source)
        return []

    # Conditional GET
    headers = {**UA, **_conditional_headers(source)}
    last_exc = None
    text = None
    for i in range(RETRY_TOTAL):
        try:
            r = session.get(url, headers=headers, timeout=HTTP_TIMEOUT)
            if r.status_code == 304:
                # Not modified; record fetch time & return empty
                _store_feed_headers(source, url, r.headers.get("ETag"), r.headers.get("Last-Modified"))
                return []
            if r.status_code == 200:
                text = r.text
                _store_feed_headers(source, url, r.headers.get("ETag"), r.headers.get("Last-Modified"))
                break
            log.warning("GET %s -> %s", url, r.status_code)
        except Exception as e:
            last_exc = e
            log.warning("GET %s error (%d/%d): %s", url, i+1, RETRY_TOTAL, e)
        _sleep(i)
    if text is None and last_exc:
        raise last_exc
    if text is None:
        return []

    # Parse & filter
    recs = _parse_rss(text, source, query_label, start, end, lang, per_source_max)

    # Upsert into DB (URL PK; check also content_hash to avoid noisy near-duplicates)
    new_recs: List[Dict[str, Any]] = []
    with _db() as conn:
        for r in recs:
            if not r.get("url"):
                continue
            # Check duplicate by URL first
            cur = conn.execute("SELECT content_hash FROM articles WHERE url=?;", (r["url"],))
            row = cur.fetchone()
            if row:
                continue
            # Check duplicate by content hash (near-duplicate with different URL is rare but possible)
            cur = conn.execute("SELECT url FROM articles WHERE content_hash=?;", (r["content_hash"],))
            if cur.fetchone():
                continue
            conn.execute("""
            INSERT INTO articles(url,title,source,lang,published_at,snippet,queries,fetched_at,content_hash)
            VALUES(?,?,?,?,?,?,?,?,?)
            """, (
                r["url"], r["title"], r["source"], r["lang"], r["published_at"], r["snippet"],
                r["queries"], _now_ts(), r["content_hash"]
            ))
            new_recs.append(r)
        conn.commit()

    return new_recs

# ------------------------------------------------------------------------------
# Query helpers
# ------------------------------------------------------------------------------
def _window_dates(start_date: str, end_date: str) -> Tuple[datetime, datetime]:
    start = pd.to_datetime(start_date, utc=True).tz_localize(None).to_pydatetime()
    end = pd.to_datetime(end_date, utc=True).tz_localize(None).to_pydatetime()
    return start, end

def _query_articles_between(start: datetime, end: datetime, like: Optional[str] = None) -> List[Dict[str, Any]]:
    with _db() as conn:
        if like:
            cur = conn.execute("""
                SELECT url, title, source, lang, published_at, snippet, queries, fetched_at, content_hash
                FROM articles
                WHERE published_at BETWEEN ? AND ? AND (title LIKE ? OR snippet LIKE ?)
                ORDER BY published_at DESC
            """, (start.isoformat(), end.isoformat(), f"%{like}%", f"%{like}%"))
        else:
            cur = conn.execute("""
                SELECT url, title, source, lang, published_at, snippet, queries, fetched_at, content_hash
                FROM articles
                WHERE published_at BETWEEN ? AND ?
                ORDER BY published_at DESC
            """, (start.isoformat(), end.isoformat()))
        rows = cur.fetchall()
    return [{
        "url": r[0], "title": r[1], "source": r[2], "lang": r[3],
        "published_at": r[4], "snippet": r[5], "queries": r[6],
        "fetched_at": r[7], "content_hash": r[8]
    } for r in rows]

# ------------------------------------------------------------------------------
# Plain impl
# ------------------------------------------------------------------------------
def search_governance_news_impl(project_hint: str, start_date: str, end_date: str,
                                lang: str = "en", max_records: int = 100, ttl_minutes: int = 30) -> Dict[str, Any]:
    start, end = _window_dates(start_date, end_date)
    per_source_max = max(1, max_records // max(1, len(RSS_FEEDS)))

    # 1) Fetch each feed once (respect TTL & conditional GET)
    query_label = f"{project_hint} governance"
    for src, url in RSS_FEEDS.items():
        try:
            fetch_feed_once(src, url, query_label, start, end, lang, per_source_max, ttl_minutes=ttl_minutes)
        except Exception as e:
            log.warning("feed fetch failed: %s -> %s", src, e)

    # 2) Read from DB and filter by project hint (title or snippet LIKE)
    like = project_hint.strip()
    all_rows = _query_articles_between(start, end, like=like)
    df = pd.DataFrame(all_rows)
    if df.empty:
        return {"count": 0, "articles": [], "project_hint": project_hint,
                "window": {"start": start_date, "end": end_date}}

    out = (df.drop_duplicates(subset=["url"])
             .sort_values("published_at", ascending=False)
             .head(max_records)
             .to_dict(orient="records"))
    return {"count": len(out), "articles": out, "project_hint": project_hint,
            "window": {"start": start_date, "end": end_date}}

def proposal_news_window_impl(project_hint: str, proposal_title_or_id: str, event_time_utc: str,
                              pre_days: int = 7, post_days: int = 7,
                              lang: str = "en", max_records: int = 100, ttl_minutes: int = 30) -> Dict[str, Any]:
    # Build an absolute window around the event
    t0 = pd.to_datetime(event_time_utc, utc=True).tz_localize(None)
    start = (t0 - timedelta(days=pre_days)).to_pydatetime()
    end = (t0 + timedelta(days=post_days)).to_pydatetime()

    query_label = f"{project_hint} {proposal_title_or_id} governance"
    per_source_max = max(1, max_records // max(1, len(RSS_FEEDS)))

    # Fetch feeds (conditional + TTL)
    for src, url in RSS_FEEDS.items():
        try:
            fetch_feed_once(src, url, query_label, start, end, lang, per_source_max, ttl_minutes=ttl_minutes)
        except Exception as e:
            log.warning("feed fetch failed: %s -> %s", src, e)

    # Query DB for the precise window with fuzzy match
    like = f"{project_hint} {proposal_title_or_id}".strip()
    all_rows = _query_articles_between(start, end, like=like)
    df = pd.DataFrame(all_rows)
    if df.empty:
        return {"count": 0, "articles": [], "project_hint": project_hint,
                "proposal": proposal_title_or_id,
                "window": {"start": start.isoformat(), "end": end.isoformat()}}

    out = (df.drop_duplicates(subset=["url"])
             .sort_values("published_at", ascending=False)
             .head(max_records)
             .to_dict(orient="records"))
    return {"count": len(out), "articles": out, "project_hint": project_hint,
            "proposal": proposal_title_or_id,
            "window": {"start": start.isoformat(), "end": end.isoformat()}}

# ------------------------------------------------------------------------------
# Tools
# ------------------------------------------------------------------------------
@mcp.tool(
    name="search_governance_news",
    title="Search Governance News",
    description="Search for governance-related news within a date range using RSS feeds with cache optimization. Finds news articles related to DeFi, DAO governance, and protocol updates with intelligent deduplication.",
    annotations={
        "readOnlyHint": True,
        "openWorldHint": True,
        "idempotentHint": True
    }
)
def search_governance_news(
    project_hint: Annotated[str, Field(
        description="Project name or hint to search for (e.g., 'Aave', 'Uniswap', 'MakerDAO')",
        min_length=1,
        max_length=100
    )],
    start_date: Annotated[str, Field(
        description="Start date in YYYY-MM-DD format",
        pattern=r"^\d{4}-\d{2}-\d{2}$"
    )],
    end_date: Annotated[str, Field(
        description="End date in YYYY-MM-DD format",
        pattern=r"^\d{4}-\d{2}-\d{2}$"
    )],
    lang: Annotated[str, Field(
        description="Language code for news search (e.g., 'en', 'es')",
        max_length=5
    )] = "en",
    max_records: Annotated[int, Field(
        description="Maximum number of news records to return",
        ge=1,
        le=500
    )] = 100,
    ttl_minutes: Annotated[int, Field(
        description="Cache TTL in minutes for RSS feeds",
        ge=1,
        le=1440
    )] = 30
) -> Dict[str, Any]:
    """Search governance-related news within a date range (cache-first, conditional GET)."""
    return search_governance_news_impl(project_hint, start_date, end_date, lang, max_records, ttl_minutes)

@mcp.tool(
    name="proposal_news_window",
    title="Search News Around Proposal Event",
    description="Search governance-related news around a specific proposal event time window. Useful for analyzing news sentiment and coverage before and after governance proposals with temporal context.",
    annotations={
        "readOnlyHint": True,
        "openWorldHint": True,
        "idempotentHint": True
    }
)
def proposal_news_window(
    project_hint: Annotated[str, Field(
        description="Project name or hint to search for news about",
        min_length=1,
        max_length=100
    )],
    proposal_title_or_id: Annotated[str, Field(
        description="Proposal title or identifier for context",
        min_length=1,
        max_length=200
    )],
    event_time_utc: Annotated[str, Field(
        description="Event timestamp in UTC (ISO 8601 format)",
        pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z?$"
    )],
    pre_days: Annotated[int, Field(
        description="Number of days before event to search",
        ge=0,
        le=30
    )] = 7,
    post_days: Annotated[int, Field(
        description="Number of days after event to search",
        ge=0,
        le=30
    )] = 7,
    lang: Annotated[str, Field(
        description="Language code for news search",
        max_length=5
    )] = "en",
    max_records: Annotated[int, Field(
        description="Maximum number of news records to return",
        ge=1,
        le=500
    )] = 100,
    ttl_minutes: Annotated[int, Field(
        description="Cache TTL in minutes for RSS feeds",
        ge=1,
        le=1440
    )] = 30
) -> Dict[str, Any]:
    """Search governance-related news around an event window (pre/post days)."""
    return proposal_news_window_impl(project_hint, proposal_title_or_id, event_time_utc,
                                     pre_days, post_days, lang, max_records, ttl_minutes)

@mcp.tool(
    name="health",
    title="Governance News Service Health Check",
    description="Check the health status of the Governance News MCP service. Returns service status and database path information.",
    annotations={
        "readOnlyHint": True,
        "openWorldHint": False,
        "idempotentHint": True
    }
)
def health() -> Dict[str, Any]:
    return {"ok": True, "service": "govnews_mcp", "db": str(DB_PATH)}


if __name__ == "__main__":
    mcp.run(transport="stdio")
