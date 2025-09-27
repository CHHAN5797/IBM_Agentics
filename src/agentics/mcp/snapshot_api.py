# src/agentics/mcp/snapshot_api.py
from __future__ import annotations

"""
Snapshot GraphQL -> FastMCP tools (stdio server)

This module exposes Snapshot read-only operations as MCP tools:
- list_proposals
- list_finished_proposals
- get_proposal_by_id
- get_proposal_result_by_id
- get_votes_page
- get_votes_all
- resolve_proposal_id_from_url
- health

Run as a stdio MCP server:
    export PYTHONPATH=src
    python src/agentics/mcp/snapshot_api.py
"""

import os
import re
import time
import random
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from agentics.utils.similarity import tokens, text_similarity

# -----------------------------
# Config
# -----------------------------
SNAPSHOT_API = os.getenv("SNAPSHOT_API", "https://hub.snapshot.org/graphql")
TIMEOUT = int(os.getenv("SNAPSHOT_TIMEOUT", "30"))
BASE_SLEEP = float(os.getenv("SNAPSHOT_BASE_SLEEP", "0.6"))
MAX_RETRIES = int(os.getenv("SNAPSHOT_MAX_RETRIES", "5"))
BACKOFF_BASE = float(os.getenv("SNAPSHOT_BACKOFF_BASE", "1.7"))
JITTER_MIN = float(os.getenv("SNAPSHOT_JITTER_MIN", "0.10"))
JITTER_MAX = float(os.getenv("SNAPSHOT_JITTER_MAX", "0.35"))

# Shared HTTP session with a stable UA to avoid being rate-limited too aggressively.
_session = requests.Session()
_session.headers.update({"User-Agent": "mcp-snapshot-api/1.0"})

def _sleep() -> None:
    """Polite sleep with jitter between requests to avoid rate limiting."""
    time.sleep(BASE_SLEEP + random.uniform(JITTER_MIN, JITTER_MAX))

# -----------------------------
# GraphQL helpers
# -----------------------------
def gql(query: str, variables: Optional[dict] = None) -> dict:
    """Call Snapshot GraphQL endpoint with retry/backoff."""
    retries = 0
    while True:
        _sleep()
        try:
            r = _session.post(
                SNAPSHOT_API,
                json={"query": query, "variables": variables or {}},
                timeout=TIMEOUT,
            )
        except requests.RequestException:
            if retries < MAX_RETRIES:
                time.sleep((BACKOFF_BASE ** retries) + random.uniform(JITTER_MIN, JITTER_MAX))
                retries += 1
                continue
            raise
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 502, 503, 504) and retries < MAX_RETRIES:
            ra = r.headers.get("Retry-After")
            delay = float(ra) if (ra and ra.isdigit()) else (BACKOFF_BASE ** retries)
            time.sleep(delay + random.uniform(JITTER_MIN, JITTER_MAX))
            retries += 1
            continue
        r.raise_for_status()

# -----------------------------
# GraphQL queries
# -----------------------------
PROPOSALS_Q = """
query($space: String!, $first: Int!, $skip: Int!) {
  proposals(
    first: $first
    skip: $skip
    where: { space_in: [$space] }
    orderBy: "created"
    orderDirection: desc
  ) {
    id
    title
    author
    body
    discussion
    start
    end
    state
  }
}
"""

PROPOSAL_BY_ID_Q = """
query($id: String!) {
  proposal(id: $id) {
    id
    title
    body
    author
    choices
    start
    end
    discussion
    state
  }
}
"""

PROPOSAL_RESULT_Q = """
query($id: String!) {
  proposal(id: $id) {
    id
    choices
    scores
    scores_total
    state
  }
}
"""

VOTES_BY_PROPOSAL_Q = """
query($proposal: String!, $first: Int!, $skip: Int!) {
  votes(
    first: $first
    skip: $skip
    where: { proposal: $proposal }
    orderBy: "created"
    orderDirection: asc
  ) {
    id
    voter
    created
    choice
    vp
    reason
  }
}
"""

# -----------------------------
# Pydantic I/O models (optional but helpful for schema clarity)
# -----------------------------
class ProposalsIn(BaseModel):
    space: str = Field(..., description="Snapshot space, e.g., 'aavedao.eth'")
    limit: int = Field(200, ge=1, le=1000, description="Max proposals to return (client-side trim)")

class VotesPageIn(BaseModel):
    proposal_id: str = Field(..., description="Snapshot proposal id")
    first: int = Field(500, ge=1, le=1000, description="Page size")
    skip: int = Field(0, ge=0, description="Offset for pagination")

class SimilarProposalsIn(BaseModel):
    proposal_id: str = Field(..., description="Reference proposal ID")
    space: str = Field(..., description="Snapshot space, e.g., 'aavedao.eth'")
    max_days: int = Field(60, ge=1, le=365, description="Search within N days")
    max_n: int = Field(10, ge=1, le=50, description="Max similar proposals")

# -----------------------------
# FastMCP app
# -----------------------------
mcp = FastMCP("SnapshotAPI")

# -----------------------------
# Core helpers (reused by tools)
# -----------------------------
def _fetch_all_proposals(space: str, batch: int = 100) -> List[dict]:
    """Fetch all proposals for a space using paged GraphQL queries."""
    out: List[dict] = []
    skip = 0
    while True:
        data = gql(PROPOSALS_Q, {"space": space, "first": batch, "skip": skip})
        chunk = (data.get("data") or {}).get("proposals") or []
        if not chunk:
            break
        out.extend(chunk)
        if len(chunk) < batch:
            break
        skip += batch
    return out

def _finished_only(proposals: List[dict]) -> List[dict]:
    """Filter proposals to those in 'closed' state whose end <= now."""
    now_ts = int(datetime.now(timezone.utc).timestamp())
    return [p for p in proposals if p.get("state") == "closed" and int(p.get("end") or 0) <= now_ts]

def _find_similar_proposals_logic(
    reference_proposal: dict,
    all_proposals: List[dict],
    max_days: int,
    max_n: int,
    jaccard_min: float = 0.30
) -> List[dict]:
    """
    Find similar proposals based on time proximity and content similarity.

    Args:
        reference_proposal: The proposal to find similar ones for
        all_proposals: All proposals in the space
        max_days: Maximum days to look back
        max_n: Maximum number of results
        jaccard_min: Minimum Jaccard similarity threshold

    Returns:
        List of similar proposals with similarity scores
    """
    ref_start = int(reference_proposal.get("start") or 0)
    ref_title = reference_proposal.get("title") or ""
    ref_body = reference_proposal.get("body") or ""
    ref_author = reference_proposal.get("author") or ""

    # Filter: closed proposals that ended before reference started
    closed_before = [
        p for p in all_proposals
        if (
            p.get("state") == "closed"
            and int(p.get("end") or 0) < ref_start
        )
    ]

    # Sort by end time (most recent first)
    closed_before.sort(key=lambda p: int(p.get("end") or 0), reverse=True)

    # Filter by time range (within max_days)
    day_sec = 86400
    within_days = []
    for p in closed_before:
        days_ago = (ref_start - int(p.get("end") or 0)) / day_sec
        if days_ago <= max_days:
            within_days.append(p)
        else:
            break  # Since sorted by end time desc, we can break early

    # Calculate similarity and filter
    ref_tokens = tokens(ref_title + " " + ref_body)

    def _match_topic_or_author(p: dict) -> tuple[bool, float]:
        # Same author match
        if (
            ref_author
            and p.get("author")
            and str(p["author"]).lower() == str(ref_author).lower()
        ):
            return True, 1.0  # Perfect match for same author

        # Content similarity
        p_title = p.get("title") or ""
        p_body = p.get("body") or ""
        p_tokens = tokens(p_title + " " + p_body)

        sim = text_similarity(ref_title + " " + ref_body, p_title + " " + p_body)
        return sim >= jaccard_min, sim

    # Build results with similarity scores
    results = []
    for p in within_days:
        matches, similarity = _match_topic_or_author(p)
        if matches:
            results.append({
                **p,
                "similarity_score": round(similarity, 4)
            })

    # Sort by similarity (highest first) and limit
    results.sort(key=lambda x: x["similarity_score"], reverse=True)
    return results[:max_n]

def _fetch_proposal_by_id(pid: str) -> dict:
    data = gql(PROPOSAL_BY_ID_Q, {"id": pid})
    return (data.get("data") or {}).get("proposal") or {}

def _fetch_proposal_result_by_id(pid: str) -> dict:
    data = gql(PROPOSAL_RESULT_Q, {"id": pid})
    return (data.get("data") or {}).get("proposal") or {}

def _fetch_votes_page(pid: str, first: int, skip: int) -> List[dict]:
    data = gql(VOTES_BY_PROPOSAL_Q, {"proposal": pid, "first": first, "skip": skip})
    return ( (data.get("data") or {}).get("votes") ) or []

def _fetch_votes_all(pid: str, batch: int = 500) -> List[dict]:
    """Fetch all votes for a proposal with paging (ascending by created)."""
    out: List[dict] = []
    skip = 0
    while True:
        chunk = _fetch_votes_page(pid, batch, skip)
        if not chunk:
            break
        out.extend(chunk)
        if len(chunk) < batch:
            break
        skip += batch
    return out

# -----------------------------
# Tools
# -----------------------------
@mcp.tool()
def list_proposals(args: ProposalsIn) -> List[dict]:
    """
    List proposals for a given Snapshot space (most recent first).
    This returns up to 'limit' proposals (client-side truncated).
    """
    all_props = _fetch_all_proposals(args.space)
    return all_props[: max(1, min(args.limit, 1000))]

@mcp.tool()
def list_finished_proposals(args: ProposalsIn) -> List[dict]:
    """
    List finished proposals (state='closed' and end <= now) for a space.
    This returns up to 'limit' proposals (client-side truncated).
    """
    all_props = _fetch_all_proposals(args.space)
    fins = _finished_only(all_props)
    return fins[: max(1, min(args.limit, 1000))]

@mcp.tool()
def get_proposal_by_id(proposal_id: str) -> dict:
    """Get a single proposal metadata record by id."""
    return _fetch_proposal_by_id(proposal_id)

@mcp.tool()
def get_proposal_result_by_id(proposal_id: str) -> dict:
    """Get result (choices/scores/scores_total/state) for a proposal id."""
    return _fetch_proposal_result_by_id(proposal_id)

@mcp.tool()
def get_votes_page(proposal_id: str, first: int = 500, skip: int = 0) -> List[dict]:
    """Get one page of votes for a proposal (ascending by created)."""
    # Defensive bounds
    first = max(1, min(int(first or 500), 1000))
    skip = max(0, int(skip or 0))
    return _fetch_votes_page(proposal_id, first, skip)

@mcp.tool()
def get_votes_all(proposal_id: str, batch: int = 500) -> List[dict]:
    """Get all votes for a proposal using pagination."""
    batch = max(1, min(int(batch or 500), 1000))
    return _fetch_votes_all(proposal_id, batch=batch)

@mcp.tool()
def resolve_proposal_id_from_url(snapshot_url: str) -> Optional[str]:
    """
    Resolve proposal id from a full Snapshot URL.
    Example: https://snapshot.org/#/aavedao.eth/proposal/0xABC... -> 0xABC...
    """
    m = re.search(r"/proposal/([0-9a-zA-Z]+)", snapshot_url)
    return m.group(1) if m else None

@mcp.tool()
def find_similar_proposals(args: SimilarProposalsIn) -> List[dict]:
    """
    Find proposals similar to a reference proposal based on content and
    author similarity.

    Returns proposals that:
    - Are in 'closed' state and ended before the reference proposal started
    - Are within the specified time range (max_days)
    - Have similar content (Jaccard similarity >= 0.30) or same author
    - Include similarity scores and basic vote results if available

    Each result includes similarity_score field (0.0 to 1.0).
    """
    try:
        # Fetch reference proposal
        reference = _fetch_proposal_by_id(args.proposal_id)
        if not reference:
            raise ValueError(f"Proposal not found: {args.proposal_id}")

        # Fetch all proposals in the space
        all_proposals = _fetch_all_proposals(args.space)
        if not all_proposals:
            return []

        # Find similar proposals
        similar = _find_similar_proposals_logic(
            reference_proposal=reference,
            all_proposals=all_proposals,
            max_days=args.max_days,
            max_n=args.max_n
        )

        # Enrich with vote results if available
        enriched = []
        for proposal in similar:
            enriched_proposal = {
                "id": proposal.get("id"),
                "title": proposal.get("title"),
                "author": proposal.get("author"),
                "body": proposal.get("body"),
                "end_utc": None,
                "similarity_score": proposal.get("similarity_score"),
                "vote_result": None
            }

            # Convert end timestamp to UTC string
            end_ts = proposal.get("end")
            if end_ts:
                try:
                    end_dt = datetime.fromtimestamp(int(end_ts), tz=timezone.utc)
                    enriched_proposal["end_utc"] = end_dt.strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    )
                except (ValueError, TypeError):
                    pass

            # Try to get vote result
            try:
                vote_result = _fetch_proposal_result_by_id(proposal.get("id", ""))
                if vote_result and vote_result.get("choices"):
                    enriched_proposal["vote_result"] = {
                        "choices": vote_result.get("choices"),
                        "scores": vote_result.get("scores"),
                        "scores_total": vote_result.get("scores_total"),
                        "state": vote_result.get("state")
                    }
            except Exception:
                # Graceful degradation if vote result fetch fails
                pass

            enriched.append(enriched_proposal)

        return enriched

    except Exception as e:
        # Return structured error for MCP client
        raise ValueError(f"Error finding similar proposals: {str(e)}")

@mcp.tool()
def health() -> Dict[str, Any]:
    """Simple health check tool."""
    return {
        "ok": True,
        "service": "SnapshotAPI",
        "api": SNAPSHOT_API,
        "timeout": TIMEOUT,
        "retries": MAX_RETRIES,
    }

# -----------------------------
# MCP stdio launcher
# -----------------------------
if __name__ == "__main__":
    # Run as a stdio MCP server so orchestrators/IDEs can attach as a tool.
    mcp.run(transport="stdio")
