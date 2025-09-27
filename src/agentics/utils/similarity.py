# src/agentics/utils/similarity.py
from __future__ import annotations

"""
Text similarity utilities for proposal comparison.

This module provides functions for tokenization and Jaccard similarity
calculation, extracted from the governance decision pipeline for reuse
in MCP tools and other components.
"""

import re
from typing import Optional, Set


# Common English stopwords for filtering
_STOP = set(
    "the and for with from into that this have has are were was will shall "
    "of in on to by at as is it be or an a we you they our their".split()
)


def tokens(text: Optional[str]) -> Set[str]:
    """
    Extract meaningful tokens from text for similarity comparison.

    Args:
        text: Input text to tokenize

    Returns:
        Set of lowercase tokens (length >= 3, excluding stopwords)
    """
    if not text:
        return set()

    # Extract alphanumeric tokens (with hyphens)
    toks = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]+", text.lower())

    # Filter by length and stopwords
    return set(t for t in toks if len(t) >= 3 and t not in _STOP)


def jaccard_similarity(a: Set[str], b: Set[str]) -> float:
    """
    Calculate Jaccard similarity between two token sets.

    Args:
        a: First token set
        b: Second token set

    Returns:
        Jaccard similarity score (0.0 to 1.0)
    """
    if not a or not b:
        return 0.0

    intersection = len(a & b)
    union = len(a | b)

    return float(intersection) / float(union) if union else 0.0


def text_similarity(text1: Optional[str], text2: Optional[str]) -> float:
    """
    Calculate text similarity using Jaccard similarity of tokens.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Similarity score (0.0 to 1.0)
    """
    tokens1 = tokens(text1)
    tokens2 = tokens(text2)

    return jaccard_similarity(tokens1, tokens2)