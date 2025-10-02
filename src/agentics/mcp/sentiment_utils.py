#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sentiment Analysis Utilities

Provides deterministic sentiment analysis for text content using
VADER (NLTK) when available, with fallback to a domain-tuned
lexicon-based approach optimized for DAO/governance forum content.

Public API:
    score_text(text: str) -> Tuple[float, str, str]
"""

from __future__ import annotations
import re
from typing import Tuple


# VADER sentiment analyzer (optional dependency)
_SIA = None
try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    try:
        _SIA = SentimentIntensityAnalyzer()
    except LookupError:
        _SIA = None
except Exception:
    _SIA = None


# Lexicon patterns for DAO/governance content
_WORD = re.compile(r"[a-zA-Z][a-zA-Z0-9_\-']+")

_POS_STEMS = {
    "support", "agree", "benefit", "approve", "favor",
    "lgtm", "ack", "yes", "endorse",
}
_NEG_STEMS = {
    "against", "disagree", "risk", "reject", "oppose",
    "concern", "nay", "nack", "no",
}
_POS_PHRASES = {
    "strongly support", "in favor", "+1", "looks good", "ship it"
}
_NEG_PHRASES = {
    "strongly oppose", "not in favor", "-1", "block", "veto"
}


def _norm_token(w: str) -> str:
    """
    Normalize token by simple stemming (suffix stripping).

    Args:
        w: Token to normalize

    Returns:
        Normalized token (lowercase, common suffixes removed)
    """
    w = w.lower()
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > 4 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def _lexicon_label(text: str) -> Tuple[float, str]:
    """
    Classify sentiment using lexicon-based approach.

    Args:
        text: Text to analyze

    Returns:
        Tuple of (score in [-1,1], label ∈ {Positive,Negative,Neutral})
    """
    t = (text or "").lower()

    # Check for strong phrase matches first
    for ph in _POS_PHRASES:
        if ph in t:
            return (1.0, "Positive")
    for ph in _NEG_PHRASES:
        if ph in t:
            return (-1.0, "Negative")

    # Token-based scoring
    toks = {_norm_token(w) for w in _WORD.findall(t)}
    pos_hits = len([w for w in toks if w in _POS_STEMS])
    neg_hits = len([w for w in toks if w in _NEG_STEMS])

    if pos_hits > neg_hits:
        score = (pos_hits - neg_hits) / max(1, pos_hits + neg_hits)
        return (score, "Positive")
    if neg_hits > pos_hits:
        score = (pos_hits - neg_hits) / max(1, pos_hits + neg_hits)
        return (score, "Negative")

    return (0.0, "Neutral")


def score_text(text: str) -> Tuple[float, str, str]:
    """
    Analyze sentiment of text using VADER or lexicon fallback.

    Args:
        text: Text content to analyze

    Returns:
        Tuple of (score, label, method) where:
        - score: float in [-1, 1] range
        - label: "Positive", "Negative", or "Neutral"
        - method: "vader", "lexicon", or "empty"
    """
    t = (text or "").strip()
    if not t:
        return 0.0, "Neutral", "empty"

    # Try VADER first if available
    if _SIA is not None:
        s = _SIA.polarity_scores(t)["compound"]
        if s >= 0.05:
            return s, "Positive", "vader"
        if s <= -0.05:
            return s, "Negative", "vader"
        return s, "Neutral", "vader"

    # Fallback to lexicon-based analysis
    s, lab = _lexicon_label(t)
    return s, lab, "lexicon"
