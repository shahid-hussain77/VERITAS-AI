"""
N-gram utilities for exact/near-exact copy detection.
"""
from __future__ import annotations

import re
from typing import Iterator


_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens."""
    if not text:
        return []
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """Return all n-grams as tuples."""
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def ngram_set(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    """Return set of n-grams (unique)."""
    return set(ngrams(tokens, n))


def ngram_jaccard(
    tokens_a: list[str],
    tokens_b: list[str],
    n: int = 5,
) -> float:
    """
    Jaccard similarity between n-gram sets.

    Returns:
        0.0 to 1.0
    """
    set_a = ngram_set(tokens_a, n)
    set_b = ngram_set(tokens_b, n)
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0


def ngram_containment(
    tokens_a: list[str],
    tokens_b: list[str],
    n: int = 5,
) -> float:
    """
    Containment: how much of A's n-grams appear in B.

    Better than Jaccard when B is much larger than A.
    """
    set_a = ngram_set(tokens_a, n)
    set_b = ngram_set(tokens_b, n)
    if not set_a:
        return 0.0
    return len(set_a & set_b) / len(set_a)


def longest_common_substring(
    tokens_a: list[str],
    tokens_b: list[str],
    min_length: int = 5,
) -> list[tuple[int, int, int]]:
    """
    Find all maximal common substrings (consecutive token runs).

    Returns list of (start_a, start_b, length) sorted by length desc.
    """
    if not tokens_a or not tokens_b:
        return []

    # DP table
    n, m = len(tokens_a), len(tokens_b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    matches = []

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if tokens_a[i - 1] == tokens_b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1

    # Collect maximal runs
    used = set()
    for i in range(n, 0, -1):
        for j in range(m, 0, -1):
            length = dp[i][j]
            if length < min_length:
                continue
            start_a = i - length
            start_b = j - length
            key = (start_a, start_b, length)
            if key in used:
                continue
            # Check this is maximal (can't extend)
            if i < n and j < m and dp[i + 1][j + 1] > 0:
                continue
            if start_a > 0 and start_b > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
                pass
            used.add(key)
            matches.append(key)

    matches.sort(key=lambda x: x[2], reverse=True)
    return matches


def find_matching_spans(
    text_a: str,
    text_b: str,
    min_words: int = 8,
) -> list[dict]:
    """
    Find all matching spans between two texts.

    Returns list of dicts:
        {
            "submitted_text": str,
            "source_text": str,
            "start_a": int,
            "start_b": int,
            "length_words": int,
            "similarity": float,
        }
    """
    tokens_a = tokenize(text_a)
    tokens_b = tokenize(text_b)

    if not tokens_a or not tokens_b:
        return []

    matches = longest_common_substring(tokens_a, tokens_b, min_length=min_words)

    results = []
    for start_a, start_b, length in matches:
        span_a = tokens_a[start_a : start_a + length]
        span_b = tokens_b[start_b : start_b + length]

        # Reconstruct with original casing — approximate
        sub_text = " ".join(span_a)
        src_text = " ".join(span_b)

        results.append({
            "submitted_text": sub_text,
            "source_text": src_text,
            "start_a": start_a,
            "start_b": start_b,
            "length_words": length,
            "similarity": 1.0,  # exact match
        })

    return results


__all__ = [
    "tokenize",
    "ngrams",
    "ngram_set",
    "ngram_jaccard",
    "ngram_containment",
    "longest_common_substring",
    "find_matching_spans",
]