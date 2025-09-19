# --- PATCH START: forums_mcp.py ---

from __future__ import annotations
import time, random
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
import requests
from mcp.server.fastmcp import FastMCP

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

@mcp.tool()
def fetch_discussion(url: str, max_pages: int = 5) -> Dict[str, Any]:
    """
    Fetch a Discourse thread as JSON with pagination; return normalized structure.
    """
    if not _is_discourse(url):
        return {"type": "generic", "note": "Not a known Discourse host/path", "url": url}

    posts: List[Dict[str, Any]] = []
    header: Dict[str, Any] = {}
    posts_count_expected: Optional[int] = None
    seen_ids: set[int] = set()

    for page in range(1, (max_pages if max_pages > 0 else 9999) + 1):
        u = _with_page_json(url, page)
        rr = _get(u)
        if rr.status_code == 304:
            
            break
        if rr.status_code != 200:
            break

        j = rr.json()
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

    return {
        "type": "discourse",
        "thread": header,
        "posts": posts,
        "posts_returned": len(posts),
        "complete": (isinstance(posts_count_expected, int) and len(posts) >= posts_count_expected)
    }

@mcp.tool()
def fetch_page(url: str, max_bytes: int = 400_000) -> Dict[str, Any]:
    """Fallback: fetch raw HTML/text for non-Discourse pages with conditional GET."""
    rr = _get(url)
    text = rr.text if rr.status_code == 200 else ""
    if len(text) > max_bytes:
        text = text[:max_bytes]
    return {"type": "generic", "url": url, "status": rr.status_code, "content": text}

@mcp.tool()
def health() -> Dict[str, Any]:
    return {"ok": True, "service": "ForumsMCP"}
# --- PATCH END ---
