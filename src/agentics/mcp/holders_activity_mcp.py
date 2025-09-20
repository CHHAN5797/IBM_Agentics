# -*- coding: utf-8 -*-
"""
Etherscan-based Onchain Holders MCP (Improved)
- Fetch ERC-20 holder count & distribution (Etherscan)
- Fetch Snapshot proposal votes (via shared snapshot_api if available)
- Compute concentration metrics (Gini, Top-N share) + simple participation proxy
- Reduce API waste with per-endpoint TTL cache, session retries, and polite rate limiting

Tools (FastMCP):
  - analyze_holders(AnalyzeArgs) -> dict
  - health() -> dict
"""

from __future__ import annotations
import os
import json
import time
import math
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
from pydantic import BaseModel, Field
from dateutil import parser as dtp

try:
    # Prefer the canonical FastMCP import if present
    from fastmcp import FastMCP
except Exception:
    from mcp.server.fastmcp import FastMCP

# ------------------------------- Config ---------------------------------
UTC = timezone.utc

ETHERSCAN_BASE = os.getenv("ETHERSCAN_BASE", "https://api.etherscan.io/v2/api")
ETHERSCAN_KEY  = os.getenv("ETHERSCAN_API_KEY")
SNAPSHOT_GQL   = os.getenv("SNAPSHOT_GQL", "https://hub.snapshot.org/graphql")

HTTP_TIMEOUT   = int(os.getenv("HOLDERS_HTTP_TIMEOUT", "60"))
BASE_SLEEP     = float(os.getenv("HOLDERS_BASE_SLEEP", "0.35"))
RETRY_MAX      = int(os.getenv("HOLDERS_MAX_RETRIES", "5"))
BACKOFF_BASE   = float(os.getenv("HOLDERS_BACKOFF_BASE", "1.7"))

# Default TTLs per endpoint (seconds)
TTL_HOLDERCOUNT = int(os.getenv("TTL_HOLDERCOUNT_SEC", str(6 * 3600)))    # 6h, can be 24h for slow-moving
TTL_TOPHOLDERS  = int(os.getenv("TTL_TOPHOLDERS_SEC",  str(2 * 3600)))    # 2h
TTL_HOLDERLIST  = int(os.getenv("TTL_HOLDERLIST_SEC",  str(30 * 60)))     # 30m
TTL_TOKENINFO   = int(os.getenv("TTL_TOKENINFO_SEC",   str(24 * 3600)))   # 24h

UA = os.getenv("HOLDERS_UA", "holders-activity-mcp/1.1")

# ------------------------------- Utilities ------------------------------

def dt_utc(x) -> datetime:
    """Parse to timezone-aware UTC datetime."""
    return x if isinstance(x, datetime) else dtp.parse(str(x)).astimezone(UTC)

def sha1_key(*parts: str) -> str:
    """Stable short cache key from parts."""
    s = "|".join(parts)
    return hashlib.sha1(s.encode()).hexdigest()

# ----------------------- Concentration metrics (pure) -------------------

def gini_from_weights(w: np.ndarray) -> float:
    """Gini coefficient for nonnegative weights."""
    if w.size == 0:
        return 0.0
    x = np.sort(np.abs(w))
    s = x.sum()
    if s == 0:
        return 0.0
    # Gini via mean absolute difference shortcut
    cumx = np.cumsum(x)
    n = x.size
    return float((n + 1 - 2 * (cumx / s).sum() / n))

def top_share(values: np.ndarray, k: int) -> Optional[float]:
    """Share held by the largest k balances."""
    if values.size == 0 or values.sum() == 0 or k <= 0:
        return None
    idx = np.argsort(values)[::-1][:k]
    return float(values[idx].sum() / values.sum())

def parse_balances(rows: List[Dict[str, Any]]) -> np.ndarray:
    """Convert Etherscan holder rows to float balances (raw units)."""
    if not rows:
        return np.array([], dtype=float)
    vals = []
    for r in rows:
        # Etherscan 'balance'/'Balance' fields vary by endpoint
        v = r.get("balance") or r.get("Balance") or r.get("value")
        try:
            vals.append(float(v))
        except Exception:
            vals.append(0.0)
    return np.array(vals, dtype=float)

# ------------------------------- File cache -----------------------------

class FileCache:
    """
    Tiny JSON file cache with TTL.
    - A single JSON file per key: {"_ts": epoch_sec, "data": <payload>}
    """
    def __init__(self, root: str = ".cache/etherscan"):
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str, ttl_sec: Optional[int] = None) -> Optional[dict]:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            with p.open("r") as f:
                obj = json.load(f)
            ts = int(obj.get("_ts", 0))
            if ttl_sec is not None and (time.time() - ts) > max(0, ttl_sec):
                return None
            return obj.get("data")
        except Exception:
            return None

    def set(self, key: str, obj: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        p = self._path(key)
        with p.open("w") as f:
            json.dump({"_ts": int(time.time()), "data": obj}, f)

# --------------------------- HTTP session (shared) -----------------------

def build_session() -> requests.Session:
    """
    Build a shared HTTP session with retries for 429/5xx, backoff, and UA.
    """
    from urllib3.util.retry import Retry
    from requests.adapters import HTTPAdapter

    sess = requests.Session()
    sess.headers.update({"User-Agent": UA})

    retry = Retry(
        total=RETRY_MAX,
        read=RETRY_MAX,
        connect=RETRY_MAX,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    return sess

HTTP = build_session()

def _sleep():
    """Polite base sleep with small jitter."""
    j = 0.10 + 0.25 * (hash(time.time()) % 100) / 100.0
    time.sleep(BASE_SLEEP + j)

# ----------------------------- Etherscan client -------------------------

class Etherscan:
    def __init__(self, base=ETHERSCAN_BASE, api_key=ETHERSCAN_KEY, chainid: int = 1, cache: Optional[FileCache] = None):
        if not api_key:
            raise RuntimeError("Set ETHERSCAN_API_KEY in environment.")
        self.base = base
        self.key  = api_key
        self.chainid = chainid
        self.s = HTTP
        self.cache = cache or FileCache()

    def _get(self, params: Dict[str, Any], ttl: int, cache_key: Optional[str] = None) -> dict:
        """
        GET with TTL cache.
        - Etherscan v2 requires 'chainid', and API key via 'apikey'
        """
        key = cache_key or sha1_key(*(f"{k}={v}" for k, v in sorted(params.items())))
        cached = self.cache.get(key, ttl_sec=ttl)
        if cached is not None:
            return cached

        _sleep()
        q = {"chainid": self.chainid, **params, "apikey": self.key}
        r = self.s.get(self.base, params=q, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        js = r.json()
        self.cache.set(key, js)
        return js

    def holder_count(self, token_address: str) -> int:
        """tokenholdercount → integer count of holders (long TTL)."""
        js = self._get({
            "module": "token",
            "action": "tokenholdercount",
            "contractaddress": token_address
        }, ttl=TTL_HOLDERCOUNT, cache_key=sha1_key("holdercount", str(self.chainid), token_address))
        if js.get("status") == "1":
            return int(js["result"])
        raise RuntimeError(f"Etherscan tokenholdercount error: {js}")

    def top_holders(self, token_address: str, offset: int = 100) -> List[Dict[str, Any]]:
        """topholders → list of top holders (up to 1000) (mid TTL)."""
        js = self._get({
            "module": "token",
            "action": "topholders",
            "contractaddress": token_address,
            "offset": offset
        }, ttl=TTL_TOPHOLDERS, cache_key=sha1_key("topholders", str(self.chainid), token_address, str(offset)))
        if js.get("status") in ("1", "Ok"):
            return js.get("result", [])
        raise RuntimeError(f"Etherscan topholders error: {js}")

    def token_holder_list(self, token_address: str, page: int = 1, offset: int = 100) -> List[Dict[str, Any]]:
        """
        tokenholderlist → current holders and balances (paginated; short TTL).
        Use carefully; it's heavier than 'topholders'.
        """
        js = self._get({
            "module": "token",
            "action": "tokenholderlist",
            "contractaddress": token_address,
            "page": page,
            "offset": offset
        }, ttl=TTL_HOLDERLIST, cache_key=sha1_key("holderlist", str(self.chainid), token_address, str(page), str(offset)))
        if js.get("status") == "1":
            return js.get("result", [])
        raise RuntimeError(f"Etherscan tokenholderlist error: {js}")

    def token_info(self, token_address: str) -> Dict[str, Any]:
        """tokeninfo → basic project info (symbol/decimals may be included)."""
        js = self._get({
            "module": "token",
            "action": "tokeninfo",
            "contractaddress": token_address
        }, ttl=TTL_TOKENINFO, cache_key=sha1_key("tokeninfo", str(self.chainid), token_address))
        if js.get("status") == "1":
            res = js.get("result") or []
            return res[0] if res else {}
        # Non-fatal: return empty dict
        return {}

# ----------------------- Snapshot client (prefer shared) -----------------

class _SnapshotLocal:
    """
    Prefer reusing the shared snapshot_api helper functions if importable:
      - _fetch_proposal_by_id
      - _fetch_votes_all
    Fallback to local HTTP with the same retry/backoff semantics.
    """
    def __init__(self, endpoint=SNAPSHOT_GQL):
        self.endpoint = endpoint
        self.s = HTTP

        # Try to import shared helpers (agentics.mcp.snapshot_api)
        self._use_local_helpers = False
        try:
            # Delayed import so this file remains standalone if path differs
            from agentics.mcp.snapshot_api import _fetch_proposal_by_id, _fetch_votes_all  # type: ignore
            self._fetch_proposal_by_id = _fetch_proposal_by_id
            self._fetch_votes_all = _fetch_votes_all
            self._use_local_helpers = True
        except Exception:
            self._use_local_helpers = False

    # -------- shared helpers path --------
    def proposal_meta(self, proposal_id: str) -> dict:
        if self._use_local_helpers:
            return self._fetch_proposal_by_id(proposal_id)  # type: ignore
        # Fallback: minimal GraphQL call
        q = """
        query($id:String!){
          proposal(id:$id){
            id title body start end created choices scores scores_total space{ id name }
          }
        }"""
        return self._gql(q, {"id": proposal_id}).get("proposal") or {}

    def proposal_votes(self, proposal_id: str) -> List[dict]:
        if self._use_local_helpers:
            return self._fetch_votes_all(proposal_id)  # type: ignore
        # Fallback: page through votes
        q = """
        query($id:String!, $first:Int!, $skip:Int!){
          votes(first:$first, skip:$skip, where:{proposal:$id}, orderBy:"vp", orderDirection: desc){
            voter vp created
          }
        }"""
        out, skip, first = [], 0, 1000
        while True:
            chunk = self._gql(q, {"id": proposal_id, "first": first, "skip": skip})["votes"]
            out.extend(chunk)
            if len(chunk) < first:
                break
            skip += first
        return out

    # -------- HTTP fallback core --------
    def _gql(self, query: str, variables: dict) -> dict:
        retries = 0
        while True:
            _sleep()
            try:
                r = self.s.post(self.endpoint, json={"query": query, "variables": variables}, timeout=HTTP_TIMEOUT)
            except requests.RequestException:
                if retries < RETRY_MAX:
                    time.sleep((BACKOFF_BASE ** retries) + 0.1)
                    retries += 1
                    continue
                raise
            if r.status_code == 200:
                js = r.json()
                if "errors" in js:
                    raise RuntimeError(js["errors"])
                return js["data"]
            if r.status_code in (429, 502, 503, 504) and retries < RETRY_MAX:
                ra = r.headers.get("Retry-After")
                delay = float(ra) if (ra and ra.isdigit()) else (BACKOFF_BASE ** retries)
                time.sleep(delay + 0.1)
                retries += 1
                continue
            r.raise_for_status()

# ------------------------------- Schemas --------------------------------

class AnalyzeArgs(BaseModel):
    space: Optional[str] = None
    proposal_id: str
    token_address: str
    chainid: int = 1
    # distribution source: "topholders" (fast) or "holderlist" (heavy)
    distribution_source: str = Field(default="topholders", pattern="^(topholders|holderlist)$")
    top_n: int = 100  # for shares/Gini calculations; limited by available list length
    holderlist_page_limit: int = 0  # when using holderlist, how many pages to pull (0=only first page)
    cache_dir: str = ".cache/etherscan"

# ------------------------------- Core -----------------------------------

def _participation_ratio(votes_count: int, holder_count: Optional[int]) -> Optional[float]:
    """Simple participation proxy: votes_count / holder_count (None-safe)."""
    if not holder_count or holder_count <= 0:
        return None
    return round(float(votes_count) / float(holder_count), 6)

def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default

def _extract_decimals(token_info: Dict[str, Any]) -> Optional[int]:
    """Try to read decimals from tokeninfo result; various keys may exist."""
    for k in ("decimals", "tokenDecimal", "TokenDecimal"):
        if k in token_info:
            try:
                return int(token_info[k])
            except Exception:
                pass
    return None

# ------------------------------- MCP app --------------------------------

mcp = FastMCP("HoldersMCP")

@mcp.tool()
def analyze_holders(args: AnalyzeArgs) -> Dict[str, Any]:
    """
    Compute holder distribution & participation proxies for a Snapshot proposal.
    - Minimizes API calls via TTL cache and shared snapshot_api reuse.
    """
    es = Etherscan(chainid=args.chainid, cache=FileCache(args.cache_dir))
    snap = _SnapshotLocal()

    # 1) Snapshot meta & votes (shared helper → HTTP fallback)
    meta = snap.proposal_meta(args.proposal_id) or {}
    votes = snap.proposal_votes(args.proposal_id) or []
    votes_sorted = sorted(votes, key=lambda v: _safe_int(v.get("created"), 0))
    votes_count = len(votes_sorted)

    # 2) Holder count (long TTL; slow-moving)
    holder_cnt = es.holder_count(args.token_address)

    # 3) Distribution (choose endpoint)
    balances = np.array([], dtype=float)
    source = args.distribution_source
    if source == "topholders":
        rows = es.top_holders(args.token_address, offset=min(1000, max(10, args.top_n)))
        balances = parse_balances(rows)
    else:
        # holderlist: optionally limited pages (heavy)
        page, rows_all = 1, []
        while True:
            rows = es.token_holder_list(args.token_address, page=page, offset=100)
            if not rows:
                break
            rows_all.extend(rows)
            if args.holderlist_page_limit and page >= args.holderlist_page_limit:
                break
            page += 1
        balances = parse_balances(rows_all)

    # 4) Metrics
    k = max(1, int(args.top_n))
    gini = gini_from_weights(balances)
    topk_share = top_share(balances, k)
    part_ratio = _participation_ratio(votes_count, holder_cnt)

    # 5) Token decimals (for optional normalization)
    token_meta = es.token_info(args.token_address)
    decimals = _extract_decimals(token_meta)

    return {
        "space": args.space,
        "proposal": {
            "id": meta.get("id"),
            "title": meta.get("title"),
            "start": meta.get("start"),
            "end": meta.get("end"),
            "state": meta.get("state"),
            "choices": meta.get("choices"),
            "scores": meta.get("scores"),
            "scores_total": meta.get("scores_total"),
            "snapshot_space": (meta.get("space") or {}).get("id"),
        },
        "holders": {
            "chainid": args.chainid,
            "token_address": args.token_address,
            "holder_count": holder_cnt,
            "distribution_source": source,
            "decimals": decimals,
            "top_n": k,
        },
        "metrics": {
            "gini": round(float(gini), 6) if gini is not None else None,
            "topk_share": round(float(topk_share), 6) if topk_share is not None else None,
            "votes_count": votes_count,
            "participation_ratio": part_ratio,
        },
        "notes": {
            "cache_dir": args.cache_dir,
            "ttl": {
                "holdercount_sec": TTL_HOLDERCOUNT,
                "topholders_sec": TTL_TOPHOLDERS,
                "holderlist_sec": TTL_HOLDERLIST,
                "tokeninfo_sec": TTL_TOKENINFO,
            },
            "snapshot_via_shared_helpers": True if getattr(snap, "_use_local_helpers", False) else False
        }
    }

@mcp.tool()
def health() -> Dict[str, Any]:
    return {"ok": True, "service": "HoldersMCP"}

# ------------------------------- End ------------------------------------


if __name__ == "__main__":
    mcp.run(transport="stdio")
