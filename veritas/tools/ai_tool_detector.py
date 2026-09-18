"""
Specific AI tool detection — Gemini, ChatGPT, Claude patterns.

Looks for known stylistic signatures and common phrases.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field


# Known markers by tool
GEMINI_MARKERS = [
    r"\bas an ai\b",
    r"\bi cannot\b.*\bprovide\b",
    r"\bi'm unable to\b",
    r"\bhowever, it's important to note\b",
    r"\bin today's (?:world|digital age|fast-paced)\b",
    r"\bdelve into\b",
    r"\bembark on\b",
    r"\btapestry of\b",
]

CHATGPT_MARKERS = [
    r"\bcertainly!?\b",
    r"\bof course!?\b",
    r"\bit'?s worth noting that\b",
    r"\bin conclusion,? it is\b",
    r"\bfurthermore,? it is\b",
    r"\bmoreover,? it is\b",
    r"\bnavigate the complexities\b",
    r"\bplays a (?:crucial|vital|pivotal) role\b",
    r"\bmultifaceted\b",
    r"\bcomprehensive (?:understanding|overview|analysis)\b",
]

CLAUDE_MARKERS = [
    r"\bi appreciate\b",
    r"\bi hope this helps\b",
    r"\bwould you like me to\b",
    r"\bhere'?s a (?:breakdown|summary|list)\b",
    r"\bi'd be happy to\b",
    r"\blet me know if\b",
]


AI_STYLE_PATTERNS = [
    # Overly structured
    (r"(?:^|\n)\s*#{1,4}\s+", "markdown_headers_in_prose"),
    # Numbered lists in flow
    (r"\n\s*1\.\s+.{20,}\n\s*2\.\s+", "structured_numbering"),
    (r"\n\s*[-•]\s+.{20,}\n\s*[-•]\s+", "structured_bullets"),
    # Empty transitions
    (r"\bIn conclusion\b", "explicit_conclusion"),
    (r"\bTo summarize\b", "explicit_summary"),
    (r"\bIn summary\b", "explicit_summary"),
    # Repetitive transitions
    (r"\bMoreover\b.{10,}\bFurthermore\b", "stacked_transitions"),
    # Vague intensifiers
    (r"\btruly\b.{1,50}\btruly\b", "repeated_intensifier"),
    (r"\bvery\b.{1,50}\bvery\b.{1,50}\bvery\b", "triple_very"),
]


@dataclass
class AIToolSignal:
    tool: str
    strength: str        # strong | medium | weak
    matched_patterns: list = field(default_factory=list)
    score: float = 0.0


@dataclass
class AIToolAnalysis:
    signals: list = field(default_factory=list)
    top_tool: str = "unknown"
    confidence: float = 0.0


def _count_patterns(text: str, patterns: list) -> list:
    lower = text.lower()
    hits = []
    for pat in patterns:
        matches = re.findall(pat, lower, re.IGNORECASE)
        if matches:
            hits.append({
                "pattern": pat,
                "count": len(matches),
                "example": matches[0][:80] if matches else "",
            })
    return hits


def analyze_ai_tool(text: str) -> AIToolAnalysis:
    """Detect which AI tool likely produced this text."""
    if not text or len(text.split()) < 30:
        return AIToolAnalysis()

    signals = []

    for tool, patterns in [
        ("gemini", GEMINI_MARKERS),
        ("chatgpt", CHATGPT_MARKERS),
        ("claude", CLAUDE_MARKERS),
    ]:
        hits = _count_patterns(text, patterns)
        if not hits:
            continue

        total = sum(h["count"] for h in hits)
        score = min(1.0, total / 5)

        strength = (
            "strong" if total >= 4 else
            "medium" if total >= 2 else
            "weak"
        )
        signals.append(AIToolSignal(
            tool=tool,
            strength=strength,
            matched_patterns=hits,
            score=score,
        ))

    # Style signals (tool-agnostic but AI-like)
    style_hits = _count_patterns(text, [p for p, _ in AI_STYLE_PATTERNS])
    if style_hits:
        signals.append(AIToolSignal(
            tool="ai_generic_style",
            strength="medium" if len(style_hits) >= 2 else "weak",
            matched_patterns=style_hits,
            score=min(1.0, len(style_hits) / 4),
        ))

    if not signals:
        return AIToolAnalysis()

    signals.sort(key=lambda s: s.score, reverse=True)
    top = signals[0]

    return AIToolAnalysis(
        signals=signals,
        top_tool=top.tool,
        confidence=top.score,
    )


__all__ = ["AIToolSignal", "AIToolAnalysis", "analyze_ai_tool"]