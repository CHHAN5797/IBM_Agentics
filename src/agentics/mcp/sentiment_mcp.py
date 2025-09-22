#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sentiment MCP (FastMCP, stdio)
- Deterministic sentiment for forum comments (Positive/Negative/Neutral)
- Uses NLTK VADER if available; otherwise falls back to a domain-tuned lexicon with simple stemming.
- Exposes:
    - classify_texts(texts: List[str]) -> List[{"text","score","label","method"}]
    - classify_forum_comments(comments: List[Dict]) -> {"per_comment":[...], "counts": {...}}
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple

from mcp.server.fastmcp import FastMCP

# --------------------------
# Optional: VADER (nltk)
# --------------------------
_SIA = None
try:
    from nltk.sentiment import SentimentIntensityAnalyzer  # type: ignore
    try:
        _SIA = SentimentIntensityAnalyzer()  # may raise LookupError if vader_lexicon missing
    except LookupError:
        _SIA = None
except Exception:
    _SIA = None

# --------------------------
# Fallback lexicon (DAO-tuned)
# --------------------------
import re
_WORD = re.compile(r"[a-zA-Z][a-zA-Z0-9_\-']+")

# stems (simple suffix strip will map supports/supported/supporting -> support)
_POS_STEMS = {
    "support", "agree", "benefit", "approve", "favor", "lgtm", "ack", "yes", "endorse",
}
_NEG_STEMS = {
    "against", "disagree", "risk", "reject", "oppose", "concern", "nay", "nack", "no",
}
# phrases often used on DAO forums
_POS_PHRASES = {"strongly support", "in favor", "+1", "looks good", "ship it"}
_NEG_PHRASES = {"strongly oppose", "not in favor", "-1", "block", "veto"}

def _norm_token(w: str) -> str:
    w = w.lower()
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > 4 and w.endswith(suf):
            return w[: -len(suf)]
    return w

def _lexicon_label(text: str) -> Tuple[float, str]:
    t = (text or "").lower()
    for ph in _POS_PHRASES:
        if ph in t:
            return (1.0, "Positive")
    for ph in _NEG_PHRASES:
        if ph in t:
            return (-1.0, "Negative")
    toks = {_norm_token(w) for w in _WORD.findall(t)}
    pos_hits = len([w for w in toks if w in _POS_STEMS])
    neg_hits = len([w for w in toks if w in _NEG_STEMS])
    if pos_hits > neg_hits:
        return ((pos_hits - neg_hits) / max(1, pos_hits + neg_hits), "Positive")
    if neg_hits > pos_hits:
        return ((pos_hits - neg_hits) / max(1, pos_hits + neg_hits), "Negative")
    return (0.0, "Neutral")

def _score_text(text: str) -> Tuple[float, str, str]:
    """
    Returns (score in [-1,1], label ∈ {Positive,Negative,Neutral}, method ∈ {"vader","lexicon","empty"})
    """
    t = (text or "").strip()
    if not t:
        return 0.0, "Neutral", "empty"
    if _SIA is not None:
        s = _SIA.polarity_scores(t)["compound"]
        if s >= 0.05:  return s, "Positive", "vader"
        if s <= -0.05: return s, "Negative", "vader"
        return s, "Neutral", "vader"
    s, lab = _lexicon_label(t)
    return s, lab, "lexicon"

# --------------------------
# FastMCP tools
# --------------------------
mcp = FastMCP("SentimentMCP", "1.0")

@mcp.tool()
def classify_texts(texts: List[str]) -> List[Dict[str, Any]]:
    """
    Classify a list of texts. Returns list of {'text', 'score', 'label', 'method'}.
    """
    out: List[Dict[str, Any]] = []
    for x in texts or []:
        s, lab, m = _score_text(x or "")
        out.append({"text": x, "score": s, "label": lab, "method": m})
    return out

@mcp.tool()
def classify_forum_comments(comments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Input comments: [{'author':..., 'created':..., 'body':...}, ...]
    Returns: {'per_comment': [...], 'counts': {'Positive':x,'Negative':y,'Neutral':z}}
    """
    per: List[Dict[str, Any]] = []
    counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for c in comments or []:
        body = (c or {}).get("body") or ""
        s, lab, m = _score_text(body)
        item = dict(c)
        item.update({"sentiment": lab, "sentiment_score": s, "sentiment_method": m})
        per.append(item)
        counts[lab] = counts.get(lab, 0) + 1
    return {"per_comment": per, "counts": counts}

if __name__ == "__main__":
    import sys, traceback
    try:
        print("[SentimentMCP] starting (stdio)...", flush=True)
        mcp.run_stdio()
    except Exception as e:
        print("[SentimentMCP] run_stdio failed:", repr(e), file=sys.stderr, flush=True)
        traceback.print_exc()
        try:
            print("[SentimentMCP] falling back to run()...", flush=True)
            mcp.run()
        except Exception as e2:
            print("[SentimentMCP] run() also failed:", repr(e2), file=sys.stderr, flush=True)
            traceback.print_exc()
            raise
