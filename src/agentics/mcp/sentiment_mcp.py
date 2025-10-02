#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sentiment MCP (FastMCP, stdio)
- Deterministic sentiment for forum comments (Positive/Negative/Neutral)
- Uses NLTK VADER if available; otherwise falls back to a domain-tuned
  lexicon with simple stemming.
- Exposes:
    - classify_texts(texts: List[str])
    - classify_forum_comments(comments: List[Dict])
"""

from __future__ import annotations
from typing import Any, Dict, List, Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from agentics.mcp.sentiment_utils import score_text

# --------------------------
# FastMCP tools
# --------------------------
mcp = FastMCP("SentimentMCP", "1.0")

@mcp.tool(
    name="classify_texts",
    title="Classify Text Sentiment",
    description="Classify sentiment of a list of texts using VADER or fallback lexicon-based analysis. Returns sentiment scores and labels (Positive/Negative/Neutral) for each text with analysis method used.",
    annotations={
        "readOnlyHint": True,
        "openWorldHint": False,
        "idempotentHint": True
    }
)
def classify_texts(
    texts: Annotated[List[str], Field(
        description="List of text strings to analyze for sentiment",
        min_length=1,
        max_length=100
    )]
) -> List[Dict[str, Any]]:
    """
    Classify a list of texts.

    Returns:
        List of dicts with 'text', 'score', 'label', 'method' fields
    """
    out: List[Dict[str, Any]] = []
    for x in texts or []:
        s, lab, m = score_text(x or "")
        out.append({"text": x, "score": s, "label": lab, "method": m})
    return out

@mcp.tool(
    name="classify_forum_comments",
    title="Classify Forum Comment Sentiment",
    description="Classify sentiment of forum comments with structured input. Analyzes comment body text and returns per-comment sentiment analysis plus aggregate counts. Use this for governance forum sentiment analysis.",
    annotations={
        "readOnlyHint": True,
        "openWorldHint": False,
        "idempotentHint": True
    }
)
def classify_forum_comments(
    comments: Annotated[List[Dict[str, Any]], Field(
        description="List of comment objects with 'body', 'author', 'created' fields",
        min_length=1,
        max_length=500
    )]
) -> Dict[str, Any]:
    """
    Classify sentiment of forum comments.

    Args:
        comments: List of comment dicts with 'body' field

    Returns:
        Dict with 'per_comment' and 'counts' keys
    """
    per: List[Dict[str, Any]] = []
    counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for c in comments or []:
        body = (c or {}).get("body") or ""
        s, lab, m = score_text(body)
        item = dict(c)
        item.update({
            "sentiment": lab,
            "sentiment_score": s,
            "sentiment_method": m
        })
        per.append(item)
        counts[lab] = counts.get(lab, 0) + 1
    return {"per_comment": per, "counts": counts}

if __name__ == "__main__":
    import sys, traceback

    def _run_stdio() -> None:
        """Launch FastMCP over stdio across library versions."""
        if hasattr(mcp, "run_stdio"):
            mcp.run_stdio()
        else:
            mcp.run(transport="stdio")

    try:
        print("[SentimentMCP] starting (stdio)...", flush=True)
        _run_stdio()
    except Exception as e:
        print("[SentimentMCP] stdio run failed:", repr(e), file=sys.stderr, flush=True)
        traceback.print_exc()
        try:
            print("[SentimentMCP] falling back to run()...", flush=True)
            mcp.run()
        except Exception as e2:
            print("[SentimentMCP] run() also failed:", repr(e2), file=sys.stderr, flush=True)
            traceback.print_exc()
            raise
