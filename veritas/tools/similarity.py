"""
Similarity algorithms:
- MinHash (approximate Jaccard)
- SimHash (locality-sensitive hashing)
- TF-IDF cosine
- Levenshtein (edit distance)
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Optional

import numpy as np

from veritas.tools.ngram import tokenize, ngrams


# ==================== MinHash ====================
class MinHash:
    """
    MinHash for approximate Jaccard similarity.

    Usage:
        mh = MinHash(num_perm=128)
        sig_a = mh.signature(text_a)
        sig_b = mh.signature(text_b)
        sim = mh.jaccard(sig_a, sig_b)
    """

    def __init__(self, num_perm: int = 128, ngram: int = 5, seed: int = 42):
        self.num_perm = num_perm
        self.ngram = ngram
        self.seed = seed
        # Generate random hash parameters
        rng = np.random.default_rng(seed)
        self._a = rng.integers(1, 2**31 - 1, size=num_perm, dtype=np.int64)
        self._b = rng.integers(0, 2**31 - 1, size=num_perm, dtype=np.int64)
        self._p = np.int64(2**31 - 1)  # Mersenne prime

    def _hash(self, s: str) -> int:
        h = hashlib.md5(s.encode("utf-8"), usedforsecurity=False).digest()
        return int.from_bytes(h[:8], "little") & 0x7FFFFFFFFFFFFFFF

    def signature(self, text: str) -> np.ndarray:
        """Compute MinHash signature for text."""
        tokens = tokenize(text)
        grams = set(ngrams(tokens, self.ngram))
        if not grams:
            return np.full(self.num_perm, np.iinfo(np.int64).max, dtype=np.int64)

        hashes = np.array([self._hash(" ".join(g)) for g in grams], dtype=np.int64)
        # h_min = (a * h + b) % p
        all_hashes = (self._a[:, None] * hashes[None, :] + self._b[:, None]) % self._p
        return all_hashes.min(axis=1)

    @staticmethod
    def jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
        """Estimate Jaccard similarity from signatures."""
        if sig_a.size == 0 or sig_b.size == 0:
            return 0.0
        return float(np.mean(sig_a == sig_b))


# ==================== SimHash ====================
class SimHash:
    """
    SimHash for near-duplicate detection.

    Hash is 64-bit. Hamming distance between hashes
    approximates similarity.
    """

    def __init__(self, bits: int = 64, ngram: int = 3):
        self.bits = bits
        self.ngram = ngram

    def _hash64(self, s: str) -> int:
        h = hashlib.md5(s.encode("utf-8"), usedforsecurity=False).digest()
        return int.from_bytes(h[:8], "little")

    def compute(self, text: str) -> int:
        """Compute SimHash as 64-bit integer."""
        tokens = tokenize(text)
        grams = ngrams(tokens, self.ngram)
        if not grams:
            return 0

        # Count n-grams
        counter = Counter(" ".join(g) for g in grams)

        # Weighted vote per bit
        v = [0] * self.bits
        for gram, weight in counter.items():
            h = self._hash64(gram)
            for i in range(self.bits):
                bit = (h >> i) & 1
                v[i] += weight if bit else -weight

        # Build final hash
        out = 0
        for i in range(self.bits):
            if v[i] > 0:
                out |= 1 << i
        return out

    @staticmethod
    def hamming(h1: int, h2: int, bits: int = 64) -> int:
        x = h1 ^ h2
        return bin(x).count("1")

    @classmethod
    def similarity(cls, h1: int, h2: int, bits: int = 64) -> float:
        """1.0 = identical, 0.0 = completely different."""
        dist = cls.hamming(h1, h2, bits)
        return 1.0 - dist / bits


# ==================== TF-IDF ====================
class TFIDFSimilarity:
    """TF-IDF based cosine similarity."""

    def __init__(self):
        self._vectorizer = None
        self._matrix = None

    def fit(self, texts: list[str]) -> None:
        """Fit vectorizer on corpus."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            max_features=10000,
        )
        self._matrix = self._vectorizer.fit_transform(texts)

    def similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity between two texts."""
        from sklearn.metrics.pairwise import cosine_similarity
        if self._vectorizer is None:
            self.fit([text_a, text_b])
        va = self._vectorizer.transform([text_a])
        vb = self._vectorizer.transform([text_b])
        return float(cosine_similarity(va, vb)[0, 0])


def tfidf_cosine(texts_a: list[str], texts_b: list[str]) -> np.ndarray:
    """
    TF-IDF cosine similarity matrix between two lists.

    Returns (len_a, len_b) matrix.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    corpus = list(texts_a) + list(texts_b)
    vec = TfidfVectorizer(
        lowercase=True, ngram_range=(1, 2), min_df=1, max_features=20000,
    )
    matrix = vec.fit_transform(corpus)
    m_a = matrix[: len(texts_a)]
    m_b = matrix[len(texts_a):]
    return cosine_similarity(m_a, m_b)


# ==================== Levenshtein ====================
def levenshtein(s1: str, s2: str) -> int:
    """Levenshtein edit distance."""
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)

    # Use two-row optimization
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            insert = curr[j] + 1
            delete = prev[j + 1] + 1
            replace = prev[j] + (c1 != c2)
            curr.append(min(insert, delete, replace))
        prev = curr
    return prev[-1]


def levenshtein_ratio(s1: str, s2: str) -> float:
    """Normalized Levenshtein similarity (0 to 1)."""
    if not s1 and not s2:
        return 1.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    return 1.0 - levenshtein(s1, s2) / max_len


def jaro_winkler(s1: str, s2: str, p: float = 0.1) -> float:
    """Jaro-Winkler similarity (0 to 1). Good for short strings."""
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    len1, len2 = len(s1), len(s2)
    match_dist = max(len1, len2) // 2 - 1

    s1_matches = [False] * len1
    s2_matches = [False] * len2

    matches = 0
    for i in range(len1):
        start = max(0, i - match_dist)
        end = min(i + match_dist + 1, len2)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    # Transpositions
    k = 0
    transpositions = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
    transpositions //= 2

    jaro = (
        matches / len1
        + matches / len2
        + (matches - transpositions) / matches
    ) / 3.0

    # Winkler prefix bonus
    prefix = 0
    for i in range(min(len1, len2, 4)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break

    return jaro + prefix * p * (1 - jaro)


# ==================== Sentence matching ====================
def best_sentence_match(
    sentence: str,
    candidates: list[str],
    threshold: float = 0.6,
) -> Optional[tuple[int, float]]:
    """
    Find best matching sentence from candidates.

    Uses Jaro-Winkler + token overlap.

    Returns (index, score) or None.
    """
    if not sentence or not candidates:
        return None

    best_idx = -1
    best_score = 0.0

    s_tokens = set(tokenize(sentence))

    for i, cand in enumerate(candidates):
        if not cand:
            continue
        c_tokens = set(tokenize(cand))

        if not s_tokens or not c_tokens:
            continue

        # Token overlap (Jaccard)
        overlap = len(s_tokens & c_tokens) / len(s_tokens | c_tokens)

        # Jaro-Winkler on raw strings (fast pre-check)
        jw = jaro_winkler(sentence[:200], cand[:200])

        # Weighted score
        score = 0.6 * overlap + 0.4 * jw

        if score > best_score:
            best_score = score
            best_idx = i

    if best_score >= threshold:
        return (best_idx, best_score)
    return None


__all__ = [
    "MinHash",
    "SimHash",
    "TFIDFSimilarity",
    "tfidf_cosine",
    "levenshtein",
    "levenshtein_ratio",
    "jaro_winkler",
    "best_sentence_match",
]