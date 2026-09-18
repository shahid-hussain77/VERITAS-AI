"""
Stylometry — writing style fingerprint.

Features:
- Avg sentence length
- Vocabulary richness (TTR)
- Punctuation density
- Function word usage
- Readability (Flesch-like)
- Sentence complexity
- Common phrases
"""
from __future__ import annotations
import math
import re
from collections import Counter
from dataclasses import dataclass, field

from veritas.tools.ngram import tokenize


FUNCTION_WORDS = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in",
    "on", "at", "by", "for", "with", "from", "as", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "should", "could", "can",
    "may", "might", "must", "this", "that", "these", "those",
    "i", "you", "he", "she", "it", "we", "they", "them", "their",
    "our", "your", "his", "her", "its", "my", "me", "us",
}


@dataclass
class StyleFingerprint:
    avg_sentence_length: float = 0.0
    sentence_length_std: float = 0.0
    vocabulary_richness: float = 0.0
    punctuation_density: float = 0.0
    function_word_ratio: float = 0.0
    avg_word_length: float = 0.0
    readability_score: float = 0.0
    sentence_complexity: float = 0.0
    common_phrases: list = field(default_factory=list)
    word_count: int = 0
    sentence_count: int = 0

    def to_dict(self):
        return {
            "avg_sentence_length": round(self.avg_sentence_length, 3),
            "sentence_length_std": round(self.sentence_length_std, 3),
            "vocabulary_richness": round(self.vocabulary_richness, 3),
            "punctuation_density": round(self.punctuation_density, 3),
            "function_word_ratio": round(self.function_word_ratio, 3),
            "avg_word_length": round(self.avg_word_length, 3),
            "readability_score": round(self.readability_score, 3),
            "sentence_complexity": round(self.sentence_complexity, 3),
            "common_phrases": self.common_phrases[:5],
            "word_count": self.word_count,
            "sentence_count": self.sentence_count,
        }

    def to_vector(self) -> list:
        """Numeric vector for comparison."""
        return [
            self.avg_sentence_length / 30.0,
            self.sentence_length_std / 15.0,
            self.vocabulary_richness,
            self.punctuation_density * 20,
            self.function_word_ratio,
            self.avg_word_length / 10.0,
            self.readability_score,
            self.sentence_complexity,
        ]


def _sentences(text):
    text = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [p.strip() for p in parts if len(p.strip()) > 5]


def compute_fingerprint(text: str) -> StyleFingerprint:
    """Extract stylometric fingerprint."""
    if not text or len(text.split()) < 30:
        return StyleFingerprint()

    sents = _sentences(text)
    words = tokenize(text)
    fp = StyleFingerprint()
    fp.word_count = len(words)
    fp.sentence_count = len(sents)

    # Sentence length
    if sents:
        lens = [len(s.split()) for s in sents]
        fp.avg_sentence_length = sum(lens) / len(lens)
        mean = fp.avg_sentence_length
        fp.sentence_length_std = math.sqrt(
            sum((x - mean) ** 2 for x in lens) / len(lens)
        )

    # Vocabulary
    if words:
        fp.vocabulary_richness = len(set(words)) / len(words)
        fp.avg_word_length = sum(len(w) for w in words) / len(words)

    # Punctuation
    puncts = sum(1 for c in text if c in ".,;:!?—-")
    fp.punctuation_density = puncts / max(len(text), 1)

    # Function words
    if words:
        fw = sum(1 for w in words if w in FUNCTION_WORDS)
        fp.function_word_ratio = fw / len(words)

    # Complexity (subordinate clauses per sentence)
    if sents:
        sub_ords = ["because", "although", "while", "since", "if", "when",
                    "whereas", "unless", "until", "though", "that", "which", "who"]
        total = sum(1 for s in sents for so in sub_ords
                    if re.search(rf"\b{so}\b", s.lower()))
        fp.sentence_complexity = total / len(sents)

    # Readability (Flesch-like)
    syllables = sum(_count_syllables(w) for w in words)
    if fp.sentence_count > 0 and fp.word_count > 0:
        asl = fp.word_count / fp.sentence_count
        asw = syllables / fp.word_count
        fp.readability_score = max(0.0, min(1.0,
            (206.835 - 1.015 * asl - 84.6 * asw) / 100.0
        ))

    # Common phrases (3-grams)
    if len(words) >= 10:
        trigrams = Counter(zip(words, words[1:], words[2:]))
        fp.common_phrases = [
            " ".join(t) for t, c in trigrams.most_common(10)
            if c >= 2
        ][:5]

    return fp


def _count_syllables(word: str) -> int:
    word = word.lower()
    vowels = "aeiouy"
    count = 0
    prev = False
    for ch in word:
        is_v = ch in vowels
        if is_v and not prev:
            count += 1
        prev = is_v
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def style_deviation(fp_a: StyleFingerprint, fp_b: StyleFingerprint) -> dict:
    """
    Compute deviation between two fingerprints.

    Returns dict with normalized diff per feature + overall.
    """
    va = fp_a.to_vector()
    vb = fp_b.to_vector()
    if not va or not vb:
        return {"overall": 0.0, "per_feature": {}}

    names = ["avg_sent_len", "sent_len_std", "vocab_richness",
             "punct_density", "func_word_ratio", "avg_word_len",
             "readability", "complexity"]

    diffs = {}
    for n, a, b in zip(names, va, vb):
        diffs[n] = round(abs(a - b), 4)

    overall = sum(diffs.values()) / len(diffs) if diffs else 0.0
    return {"overall": round(overall, 4), "per_feature": diffs}


__all__ = ["StyleFingerprint", "compute_fingerprint", "style_deviation"]