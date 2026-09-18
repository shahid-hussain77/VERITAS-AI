"""
Finding data models.

A Finding is what an agent produces when it detects something.
Evidence supports a Finding with concrete proof.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import hashlib


class Severity(str, Enum):
    """Severity levels for findings."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FindingType(str, Enum):
    """Categories of findings."""
    EXACT_COPY = "exact_copy"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    PARAPHRASE = "paraphrase"
    CROSS_LANGUAGE = "cross_language"
    SELF_PLAGIARISM = "self_plagiarism"
    CITATION_ISSUE = "citation_issue"
    FAKE_CITATION = "fake_citation"
    AI_WRITING = "ai_writing"
    STYLE_DEVIATION = "style_deviation"
    COLLUSION = "collusion"
    SOURCE_MATCH = "source_match"
    RESEARCH_INTEGRITY = "research_integrity"


@dataclass
class Evidence:
    """Concrete evidence supporting a finding."""
    submitted_text: str = ""
    source_text: str = ""
    source_document: str = ""
    similarity: float = 0.0
    method: str = ""
    page: int = 0
    paragraph_index: int = 0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "submitted_text": self.submitted_text[:300],
            "source_text": self.source_text[:300],
            "source_document": self.source_document,
            "similarity": round(self.similarity, 4),
            "method": self.method,
            "page": self.page,
            "paragraph_index": self.paragraph_index,
            "metadata": self.metadata,
        }


@dataclass
class Finding:
    """
    A single finding from an agent.

    Findings are evidence-based. They should not say
    "student cheated" but "these passages are similar".
    """
    id: str = ""
    type: FindingType = FindingType.EXACT_COPY
    severity: Severity = Severity.MEDIUM
    title: str = ""
    description: str = ""
    confidence: float = 0.0
    evidence: list[Evidence] = field(default_factory=list)
    explanation: list[str] = field(default_factory=list)
    agent: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            raw = f"{self.type}:{self.title}:{self.description}"
            self.id = hashlib.sha256(raw.encode()).hexdigest()[:12]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "confidence": round(self.confidence, 4),
            "evidence": [e.to_dict() for e in self.evidence],
            "explanation": self.explanation,
            "agent": self.agent,
            "metadata": self.metadata,
        }


@dataclass
class AgentResult:
    """Standard result returned by every agent."""
    agent: str = ""
    status: str = "ok"  # ok | failed | skipped
    findings: list[Finding] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    notes: str = ""
    error: str = ""
    duration_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict:
        return {
            "agent": self.agent,
            "status": self.status,
            "findings_count": len(self.findings),
            "findings": [f.to_dict() for f in self.findings],
            "metrics": self.metrics,
            "notes": self.notes,
            "error": self.error,
            "duration_seconds": round(self.duration_seconds, 3),
        }

    @classmethod
    def failed(cls, agent: str, error: str) -> "AgentResult":
        return cls(agent=agent, status="failed", error=error)

    @classmethod
    def skipped(cls, agent: str, reason: str) -> "AgentResult":
        return cls(agent=agent, status="skipped", notes=reason)


__all__ = [
    "Severity", "FindingType", "Evidence", "Finding", "AgentResult",
]