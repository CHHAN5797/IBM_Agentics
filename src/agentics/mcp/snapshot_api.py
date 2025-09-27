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
from typing import Any, Dict, List, Optional, Annotated

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
# Pydantic I/O models with enhanced validation
# -----------------------------
class ProposalsIn(BaseModel):
    space: str = Field(
        ...,
        description="Snapshot space identifier (e.g., 'aavedao.eth', 'uniswap')",
        min_length=1,
        max_length=100
    )
    limit: int = Field(
        200,
        ge=1,
        le=1000,
        description="Maximum number of proposals to return (client-side limit)"
    )

class VotesPageIn(BaseModel):
    proposal_id: str = Field(
        ...,
        description="Snapshot proposal ID (hexadecimal string)",
        min_length=1,
        max_length=100
    )
    first: int = Field(
        500,
        ge=1,
        le=1000,
        description="Number of votes per page for pagination"
    )
    skip: int = Field(
        0,
        ge=0,
        description="Number of votes to skip (pagination offset)"
    )

class SimilarProposalsIn(BaseModel):
    proposal_id: str = Field(
        ...,
        description="Reference proposal ID to find similar proposals for",
        min_length=1,
        max_length=100
    )
    space: str = Field(
        ...,
        description="Snapshot space to search within (e.g., 'aavedao.eth')",
        min_length=1,
        max_length=100
    )
    max_days: int = Field(
        60,
        ge=1,
        le=365,
        description="Maximum days to look back from reference proposal start date"
    )
    max_n: int = Field(
        10,
        ge=1,
        le=50,
        description="Maximum number of similar proposals to return"
    )

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
@mcp.tool(
    name="list_proposals",
    title="List Governance Proposals",
    description="List proposals for a Snapshot space ordered by creation date (most recent first). Use this to browse governance proposals or get an overview of recent activity in a DAO space.",
    annotations={
        "readOnlyHint": True,
        "openWorldHint": True,
        "idempotentHint": True
    }
)
def list_proposals(
    space: Annotated[str, Field(
        description="Snapshot space identifier (e.g., 'aavedao.eth', 'uniswap')",
        min_length=1,
        max_length=100
    )],
    limit: Annotated[int, Field(
        description="Maximum number of proposals to return (client-side limit)",
        ge=1,
        le=1000
    )] = 200
) -> List[dict]:
    """
    List proposals for a given Snapshot space (most recent first).
    This returns up to 'limit' proposals (client-side truncated).
    """
    all_props = _fetch_all_proposals(space)
    return all_props[: max(1, min(limit, 1000))]

@mcp.tool(
    name="list_finished_proposals",
    title="List Finished Proposals",
    description="List completed governance proposals (closed state with ended voting period). Use this to analyze historical governance decisions and voting outcomes in a DAO space.",
    annotations={
        "readOnlyHint": True,
        "openWorldHint": True,
        "idempotentHint": True
    }
)
def list_finished_proposals(
    space: Annotated[str, Field(
        description="Snapshot space identifier (e.g., 'aavedao.eth', 'uniswap')",
        min_length=1,
        max_length=100
    )],
    limit: Annotated[int, Field(
        description="Maximum number of proposals to return (client-side limit)",
        ge=1,
        le=1000
    )] = 200
) -> List[dict]:
    """
    List finished proposals (state='closed' and end <= now) for a space.
    This returns up to 'limit' proposals (client-side truncated).
    """
    all_props = _fetch_all_proposals(space)
    fins = _finished_only(all_props)
    return fins[: max(1, min(limit, 1000))]

@mcp.tool(
    name="get_proposal_by_id",
    title="Get Proposal Details",
    description="Retrieve detailed metadata for a specific proposal by its ID. Use this to get comprehensive information about a single governance proposal including title, body, choices, and voting parameters.",
    annotations={
        "readOnlyHint": True,
        "openWorldHint": True,
        "idempotentHint": True
    }
)
def get_proposal_by_id(
    proposal_id: Annotated[str, Field(
        description="Snapshot proposal ID (hexadecimal string starting with 0x)",
        min_length=1,
        max_length=100
    )]
) -> dict:
    """Get a single proposal metadata record by id."""
    return _fetch_proposal_by_id(proposal_id)

@mcp.tool(
    name="get_proposal_result_by_id",
    title="Get Voting Results",
    description="Get voting results for a proposal including vote tallies, choice options, and total scores. Use this to analyze voting outcomes and see which option won.",
    annotations={
        "readOnlyHint": True,
        "openWorldHint": True,
        "idempotentHint": True
    }
)
def get_proposal_result_by_id(
    proposal_id: Annotated[str, Field(
        description="Snapshot proposal ID (hexadecimal string starting with 0x)",
        min_length=1,
        max_length=100
    )]
) -> dict:
    """Get result (choices/scores/scores_total/state) for a proposal id."""
    return _fetch_proposal_result_by_id(proposal_id)

@mcp.tool(
    name="get_votes_page",
    title="Get Paginated Votes",
    description="Get a paginated list of individual votes for a proposal. Use this to examine voting patterns, individual voter choices, and voting power distribution with pagination control.",
    annotations={
        "readOnlyHint": True,
        "openWorldHint": True,
        "idempotentHint": True
    }
)
def get_votes_page(
    proposal_id: Annotated[str, Field(
        description="Snapshot proposal ID (hexadecimal string starting with 0x)",
        min_length=1,
        max_length=100
    )],
    first: Annotated[int, Field(
        description="Number of votes per page for pagination",
        ge=1,
        le=1000
    )] = 500,
    skip: Annotated[int, Field(
        description="Number of votes to skip (pagination offset)",
        ge=0
    )] = 0
) -> List[dict]:
    """Get one page of votes for a proposal (ascending by created)."""
    # Defensive bounds
    first = max(1, min(int(first or 500), 1000))
    skip = max(0, int(skip or 0))
    return _fetch_votes_page(proposal_id, first, skip)

@mcp.tool(
    name="get_votes_all",
    title="Get All Votes",
    description="Retrieve all votes for a proposal using automatic pagination. Use this for comprehensive vote analysis when you need the complete voting record.",
    annotations={
        "readOnlyHint": True,
        "openWorldHint": True,
        "idempotentHint": True
    }
)
def get_votes_all(
    proposal_id: Annotated[str, Field(
        description="Snapshot proposal ID (hexadecimal string starting with 0x)",
        min_length=1,
        max_length=100
    )],
    batch: Annotated[int, Field(
        description="Batch size for pagination (internal use)",
        ge=1,
        le=1000
    )] = 500
) -> List[dict]:
    """Get all votes for a proposal using pagination."""
    batch = max(1, min(int(batch or 500), 1000))
    return _fetch_votes_all(proposal_id, batch=batch)

@mcp.tool(
    name="resolve_proposal_id_from_url",
    title="Extract Proposal ID from URL",
    description="Extract proposal ID from a Snapshot URL. Use this to convert web URLs into proposal IDs for use with other tools.",
    annotations={
        "readOnlyHint": True,
        "openWorldHint": False,
        "idempotentHint": True
    }
)
def resolve_proposal_id_from_url(
    snapshot_url: Annotated[str, Field(
        description="Full Snapshot URL (e.g., 'https://snapshot.org/#/aavedao.eth/proposal/0x123...')",
        min_length=1
    )]
) -> Optional[str]:
    """
    Resolve proposal id from a full Snapshot URL.
    Example: https://snapshot.org/#/aavedao.eth/proposal/0xABC... -> 0xABC...
    """
    m = re.search(r"/proposal/([0-9a-zA-Z]+)", snapshot_url)
    return m.group(1) if m else None

@mcp.tool(
    name="find_similar_proposals",
    title="Find Similar Proposals",
    description="Find proposals similar to a reference proposal using content similarity and author matching. Use this to discover historical precedents, related governance decisions, or recurring proposal patterns.",
    annotations={
        "readOnlyHint": True,
        "openWorldHint": True,
        "idempotentHint": True
    }
)
def find_similar_proposals(
    proposal_id: Annotated[str, Field(
        description="Reference proposal ID to find similar proposals for",
        min_length=1,
        max_length=100
    )],
    space: Annotated[str, Field(
        description="Snapshot space to search within (e.g., 'aavedao.eth')",
        min_length=1,
        max_length=100
    )],
    max_days: Annotated[int, Field(
        description="Maximum days to look back from reference proposal start date",
        ge=1,
        le=365
    )] = 60,
    max_n: Annotated[int, Field(
        description="Maximum number of similar proposals to return",
        ge=1,
        le=50
    )] = 10
) -> List[dict]:
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
        reference = _fetch_proposal_by_id(proposal_id)
        if not reference:
            raise ValueError(f"Proposal not found: {proposal_id}")

        # Fetch all proposals in the space
        all_proposals = _fetch_all_proposals(space)
        if not all_proposals:
            return []

        # Find similar proposals
        similar = _find_similar_proposals_logic(
            reference_proposal=reference,
            all_proposals=all_proposals,
            max_days=max_days,
            max_n=max_n
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

@mcp.tool(
    name="health",
    title="Health Check",
    description="Check the health and configuration of the Snapshot API service. Use this to verify connectivity and service status.",
    annotations={
        "readOnlyHint": True,
        "openWorldHint": False,
        "idempotentHint": True
    }
)
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
