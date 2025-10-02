# --- PATCH START: forums_mcp.py ---

from __future__ import annotations
import time, random
from typing import Dict, Any, List, Optional, Tuple, Annotated
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
import requests
from mcp.server.fastmcp import FastMCP
from pydantic import Field

from agentics.mcp.sentiment_utils import score_text

TIMEOUT = 30
BASE_SLEEP = 0.6
JITTER = (0.1, 0.35)

FORUM_HOSTS = {
    "governance.aave.com",
    "forum.arbitrum.foundation",
    "forum.decentraland.org",
    "forum.balancer.fi",
    "gov.curve.finance",
    "gov.1inch.io",
    "forum.aura.finance",
    "research.lido.fi",
    "gov.uniswap.org",
    "forum.ceg.vote",
}

_last_call_at: Dict[str, float] = {}
MIN_GAP_PER_HOST = 0.7  

session = requests.Session()
session.headers.update({"User-Agent": "mcp-forums/1.1"})


_conditional_cache: Dict[str, Dict[str, str]] = {}  # url -> {"ETag": "...", "Last-Modified": "..."}

def _sleep_global():
    time.sleep(BASE_SLEEP + random.uniform(*JITTER))

def _host(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""

def _respect_host_rate(host: str):
    now = time.time()
    last = _last_call_at.get(host, 0.0)
    wait = MIN_GAP_PER_HOST - (now - last)
    if wait > 0:
        time.sleep(wait)
    _last_call_at[host] = time.time()

def _is_discourse(url: str) -> bool:
    h = _host(url)
    path = urlparse(url).path.lower()
    return h in FORUM_HOSTS and "/t/" in path

def _with_page_json(u: str, page: int) -> str:
    pu = urlparse(u if u.endswith(".json") else (u.rstrip("/") + ".json"))
    q = dict(parse_qsl(pu.query))
    q["include_raw"] = "1"
    if page > 1:
        q["page"] = str(page)
    else:
        q.pop("page", None)
    pu2 = pu._replace(query=urlencode(q))
    return urlunparse(pu2)


def _get(url: str) -> requests.Response:

    h = _host(url)
    _respect_host_rate(h)
    _sleep_global()


    hdrs = {}
    meta = _conditional_cache.get(url) or {}
    if "ETag" in meta:
        hdrs["If-None-Match"] = meta["ETag"]
    if "Last-Modified" in meta:
        hdrs["If-Modified-Since"] = meta["Last-Modified"]

    rr = session.get(url, timeout=TIMEOUT, headers=hdrs)

    if rr.status_code == 304:
        
        return rr

    if rr.status_code == 200:
        et = rr.headers.get("ETag")
        lm = rr.headers.get("Last-Modified")
        if et or lm:
            _conditional_cache[url] = {k: v for k, v in (("ETag", et), ("Last-Modified", lm)) if v}

    return rr

mcp = FastMCP("ForumsMCP")

@mcp.tool(
    name="fetch_discussion",
    title="Fetch Forum Discussion with Sentiment",
    description="Fetch a Discourse forum thread with pagination support and automatic sentiment analysis. Extracts structured discussion data including posts, authors, timestamps, and sentiment scores (Positive/Negative/Neutral) for each post. Returns aggregate sentiment summary. Use this for governance forum analysis and community sentiment tracking.",
    annotations={
        "readOnlyHint": True,
        "openWorldHint": True,
        "idempotentHint": True
    }
)
def fetch_discussion(
    url: Annotated[str, Field(
        description="Forum discussion URL (preferably Discourse-based)",
        min_length=1,
        max_length=500
    )],
    max_pages: Annotated[int, Field(
        description="Maximum number of pages to fetch for pagination",
        ge=1,
        le=20
    )] = 5
) -> Dict[str, Any]:
    """
    Fetch a Discourse thread with pagination and sentiment analysis.

    Extracts posts with sentiment scores and returns normalized structure:
    - Tries <url>.json endpoint (whitelist-agnostic)
    - Enriches each post with sentiment analysis
    - Returns aggregate sentiment_summary

    Returns:
        Dict with 'posts' (includes sentiment fields) and 'sentiment_summary'
    """
    def is_discourse_like(resp_json: dict) -> bool:
        return isinstance(resp_json, dict) and ("post_stream" in resp_json or "posts_count" in resp_json)

    posts: List[Dict[str, Any]] = []
    header: Dict[str, Any] = {}
    posts_count_expected: Optional[int] = None
    seen_ids: set[int] = set()

    tried_whitelist = _is_discourse(url)
    for page in range(1, (max_pages if max_pages > 0 else 9999) + 1):
        # Try Discourse JSON endpoint regardless of whitelist
        u = _with_page_json(url, page)
        rr = _get(u)
        if rr.status_code == 304:
            break
        if rr.status_code != 200:
            # If first page fails and not whitelisted → 멈추기
            if page == 1 and not tried_whitelist:
                return {"type": "generic", "note": "Not a Discourse JSON page", "url": url, "status": rr.status_code}
            break

        j = rr.json()
        if page == 1 and not (tried_whitelist or is_discourse_like(j)):
            return {"type": "generic", "note": "JSON fetched but not Discourse-like", "url": url}

        if page == 1:
            posts_count_expected = j.get("posts_count")
            header = {
                "title": j.get("title"),
                "slug": j.get("slug"),
                "created_at": j.get("created_at"),
                "posts_count": posts_count_expected,
                "tags": j.get("tags"),
                "url": url,
            }

        chunk = (j.get("post_stream") or {}).get("posts", []) or []
        if not chunk:
            break

        new_items = []
        for p in chunk:
            pid = p.get("id")
            if isinstance(pid, int) and pid in seen_ids:
                continue
            seen_ids.add(pid)
            new_items.append({
                "id": pid,
                "username": p.get("username"),
                "user_id": p.get("user_id"),
                "created_at": p.get("created_at"),
                "updated_at": p.get("updated_at"),
                "raw": p.get("raw"),
                "cooked": p.get("cooked"),
                "post_number": p.get("post_number"),
                "reply_to_post_number": p.get("reply_to_post_number"),
            })
        posts.extend(new_items)

        if isinstance(posts_count_expected, int) and len(posts) >= posts_count_expected:
            break

    # Enrich posts with sentiment analysis
    sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for post in posts:
        try:
            text = post.get("raw") or post.get("cooked") or ""
            score, label, method = score_text(text)
            post["sentiment"] = label
            post["sentiment_score"] = score
            post["sentiment_method"] = method
            sentiment_counts[label] += 1
        except Exception:
            # Graceful degradation on error
            post["sentiment"] = "Neutral"
            post["sentiment_score"] = 0.0
            post["sentiment_method"] = "error"
            sentiment_counts["Neutral"] += 1

    return {
        "type": "discourse",
        "thread": header,
        "posts": posts,
        "posts_returned": len(posts),
        "complete": (isinstance(posts_count_expected, int) and len(posts) >= posts_count_expected),
        "sentiment_summary": sentiment_counts
    }

@mcp.tool(
    name="fetch_page",
    title="Fetch Generic Web Page",
    description="Fallback tool to fetch raw HTML/text content from non-Discourse pages with conditional GET caching. Use this for general web content retrieval when forum-specific tools are not applicable.",
    annotations={
        "readOnlyHint": True,
        "openWorldHint": True,
        "idempotentHint": True
    }
)
def fetch_page(
    url: Annotated[str, Field(
        description="Web page URL to fetch",
        min_length=1,
        max_length=500
    )],
    max_bytes: Annotated[int, Field(
        description="Maximum bytes to read from the response content",
        ge=1000,
        le=1_000_000
    )] = 400_000
) -> Dict[str, Any]:
    """Fallback: fetch raw HTML/text for non-Discourse pages with conditional GET."""
    rr = _get(url)
    text = rr.text if rr.status_code == 200 else ""
    if len(text) > max_bytes:
        text = text[:max_bytes]
    return {"type": "generic", "url": url, "status": rr.status_code, "content": text}

@mcp.tool(
    name="health",
    title="Forums Service Health Check",
    description="Check the health status of the Forums MCP service. Returns service status and identification information.",
    annotations={
        "readOnlyHint": True,
        "openWorldHint": False,
        "idempotentHint": True
    }
)
def health() -> Dict[str, Any]:
    return {"ok": True, "service": "ForumsMCP"}


if __name__ == "__main__":
    mcp.run(transport="stdio")
# --- PATCH END ---
