"""
AI-writing signal detection.

Multiple statistical signals — no single "AI %".
"""
from __future__ import annotations
import math
import re
from collections import Counter
from dataclasses import dataclass, field

from veritas.tools.ngram import tokenize


AI_PHRASES = [
    "furthermore", "moreover", "in conclusion", "it is important to note",
    "it is worth noting", "in summary", "additionally",
    "in today's world", "in the realm of", "when it comes to",
    "plays a crucial role", "plays a vital role", "delve into",
    "comprehensive understanding", "multifaceted",
    "in the ever-evolving", "cutting-edge", "groundbreaking",
    "paradigm shift", "leverage", "robust", "seamless",
    "navigate the complexities", "tapestry", "testament to",
]


@dataclass
class AIWritingSignal:
    name: str
    score: float          # 0-1
    level: str            # low | medium | high
    detail: str = ""


@dataclass
class AIWritingAnalysis:
    signals: list = field(default_factory=list)
    overall_score: float = 0.0
    overall_level: str = "INCONCLUSIVE"
    metrics: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "signals": [{"name": s.name, "score": round(s.score, 4),
                         "level": s.level, "detail": s.detail}
                        for s in self.signals],
            "overall_score": round(self.overall_score, 4),
            "overall_level": self.overall_level,
            "metrics": self.metrics,
        }


def _sentences(text):
    text = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [p.strip() for p in parts if len(p.strip()) > 5]


def _words(text):
    return tokenize(text)


def _signal_perplexity(text):
    """Low perplexity (highly predictable) → AI-like."""
    words = _words(text)
    if len(words) < 20:
        return AIWritingSignal("perplexity", 0.0, "low", "Too short")
    bigrams = Counter(zip(words, words[1:]))
    total = len(words) - 1
    entropy = 0.0
    for count in bigrams.values():
        p = count / total
        entropy -= p * math.log2(p)
    max_entropy = math.log2(max(total, 2))
    normalized = entropy / max_entropy if max_entropy > 0 else 0

    # AI text: normalized entropy LOW (predictable)
    # Human text: HIGH entropy
    score = max(0.0, min(1.0, 1.0 - normalized * 2))
    level = "high" if score >= 0.65 else "medium" if score >= 0.40 else "low"
    return AIWritingSignal("perplexity", score, level,
                           f"Normalized entropy: {normalized:.3f}")


def _signal_burstiness(text):
    """Sentence length variation. AI: low variation."""
    sents = _sentences(text)
    if len(sents) < 5:
        return AIWritingSignal("burstiness", 0.0, "low", "Few sentences")
    lens = [len(s.split()) for s in sents]
    mean = sum(lens) / len(lens)
    variance = sum((x - mean) ** 2 for x in lens) / len(lens)
    std = math.sqrt(variance)
    cv = std / mean if mean > 0 else 0
    # High CV = human; low CV = AI
    # Typical AI CV ~0.3-0.4, human ~0.5-0.7
    score = max(0.0, min(1.0, 1.0 - cv * 1.8))
    level = "high" if score >= 0.6 else "medium" if score >= 0.35 else "low"
    return AIWritingSignal("burstiness", score, level,
                           f"CV: {cv:.3f}, mean len: {mean:.1f}")


def _signal_vocabulary(text):
    """Vocabulary richness. AI tends to be repetitive."""
    words = _words(text)
    if len(words) < 30:
        return AIWritingSignal("vocabulary", 0.0, "low", "Too short")
    unique = len(set(words))
    ttr = unique / len(words)
    # Human TTR ~0.55-0.75, AI ~0.40-0.55
    score = max(0.0, min(1.0, 1.0 - (ttr - 0.35) * 2.5))
    level = "high" if score >= 0.6 else "medium" if score >= 0.35 else "low"
    return AIWritingSignal("vocabulary", score, level,
                           f"TTR: {ttr:.3f}")


def _signal_sentence_uniformity(text):
    """Sentence structure uniformity."""
    sents = _sentences(text)
    if len(sents) < 5:
        return AIWritingSignal("sentence_uniformity", 0.0, "low", "Few sentences")
    starts = []
    for s in sents:
        w = s.split()
        if w:
            starts.append(" ".join(w[:2]).lower())
    counter = Counter(starts)
    common_ratio = counter.most_common(1)[0][1] / len(starts) if starts else 0
    score = max(0.0, min(1.0, common_ratio * 2 - 0.3))
    level = "high" if score >= 0.6 else "medium" if score >= 0.35 else "low"
    return AIWritingSignal("sentence_uniformity", score, level,
                           f"Top start phrase repeats {counter.most_common(1)[0][1]}x")


def _signal_ai_phrases(text):
    """Detect common AI-generated phrases."""
    lower = text.lower()
    words = len(lower.split())
    if words < 20:
        return AIWritingSignal("ai_phrases", 0.0, "low", "Too short")
    hits = [p for p in AI_PHRASES if p in lower]
    density = len(hits) / max(words / 100, 1)
    score = max(0.0, min(1.0, density / 4))
    level = "high" if score >= 0.6 else "medium" if score >= 0.3 else "low"
    detail = f"{len(hits)} phrases: {hits[:3]}" if hits else "No AI phrases"
    return AIWritingSignal("ai_phrases", score, level, detail)


def _signal_punctuation(text):
    """Punctuation diversity. AI uses limited variety."""
    if len(text) < 100:
        return AIWritingSignal("punctuation", 0.0, "low", "Too short")
    marks = Counter(c for c in text if c in ".,;:!?—-\"'()")
    if not marks:
        return AIWritingSignal("punctuation", 0.5, "medium", "No punctuation")
    variety = len(marks)
    score = max(0.0, min(1.0, 1.0 - variety / 8))
    level = "high" if score >= 0.6 else "medium" if score >= 0.35 else "low"
    return AIWritingSignal("punctuation", score, level,
                           f"{variety} distinct marks used")


def analyze_ai_writing(text: str) -> AIWritingAnalysis:
    """Run all AI-writing signals."""
    if not text or len(text.split()) < 30:
        return AIWritingAnalysis(metrics={"word_count": len(text.split())})

    signals = [
        _signal_perplexity(text),
        _signal_burstiness(text),
        _signal_vocabulary(text),
        _signal_sentence_uniformity(text),
        _signal_ai_phrases(text),
        _signal_punctuation(text),
    ]

    # Weighted overall
    weights = [0.25, 0.20, 0.15, 0.15, 0.20, 0.05]
    overall = sum(s.score * w for s, w in zip(signals, weights))

    # Level
    high_count = sum(1 for s in signals if s.level == "high")
    low_count = sum(1 for s in signals if s.level == "low")

    if overall >= 0.65 and high_count >= 4:
        level = "HIGH"
    elif overall >= 0.50 and high_count >= 3:
        level = "MEDIUM"
    elif overall >= 0.35:
        level = "LOW"
    else:
        level = "INCONCLUSIVE"

    # If signals contradict strongly, INCONCLUSIVE
    if high_count >= 2 and low_count >= 2:
        level = "INCONCLUSIVE"

    return AIWritingAnalysis(
        signals=signals,
        overall_score=overall,
        overall_level=level,
        metrics={"word_count": len(text.split()),
                 "sentence_count": len(_sentences(text))},
    )


__all__ = ["AIWritingSignal", "AIWritingAnalysis", "analyze_ai_writing"]