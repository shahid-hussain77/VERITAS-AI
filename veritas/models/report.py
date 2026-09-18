"""
Report data models.

The final output of VERITAS-AI.
Evidence-based. No single "plagiarism %" score.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from veritas.models.finding import Finding, AgentResult


@dataclass
class IntegrityScore:
    """
    Multi-dimensional integrity score.

    NOT a single "plagiarism %". Instead, separate signals.
    """
    exact_overlap: float = 0.0
    semantic_overlap: float = 0.0
    paraphrase_overlap: float = 0.0
    cross_language_overlap: float = 0.0
    self_overlap: float = 0.0
    source_matches: int = 0
    citation_issues: int = 0
    reference_issues: int = 0
    ai_writing_signal: str = "INCONCLUSIVE"  # LOW | MEDIUM | HIGH | INCONCLUSIVE
    style_deviation: float = 0.0
    collusion_clusters: int = 0

    @property
    def overall_confidence(self) -> str:
        """Overall evidence confidence level."""
        signals = [
            self.exact_overlap,
            self.semantic_overlap,
            self.paraphrase_overlap,
            self.self_overlap,
        ]
        avg = sum(signals) / len(signals) if signals else 0.0
        if avg >= 0.6:
            return "HIGH"
        elif avg >= 0.3:
            return "MEDIUM"
        else:
            return "LOW"

    def to_dict(self) -> dict:
        return {
            "exact_overlap": round(self.exact_overlap, 4),
            "semantic_overlap": round(self.semantic_overlap, 4),
            "paraphrase_overlap": round(self.paraphrase_overlap, 4),
            "cross_language_overlap": round(self.cross_language_overlap, 4),
            "self_overlap": round(self.self_overlap, 4),
            "source_matches": self.source_matches,
            "citation_issues": self.citation_issues,
            "reference_issues": self.reference_issues,
            "ai_writing_signal": self.ai_writing_signal,
            "style_deviation": round(self.style_deviation, 4),
            "collusion_clusters": self.collusion_clusters,
            "overall_confidence": self.overall_confidence,
        }


@dataclass
class Report:
    """The final integrity report."""
    document_id: str = ""
    document_name: str = ""
    document_words: int = 0
    document_pages: int = 0

    score: IntegrityScore = field(default_factory=IntegrityScore)
    findings: list[Finding] = field(default_factory=list)
    agent_results: list[AgentResult] = field(default_factory=list)

    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    duration_seconds: float = 0.0
    human_review_recommended: bool = False
    notes: str = ""

    def add_findings(self, findings: list[Finding]) -> None:
        self.findings.extend(findings)

    def add_agent_result(self, result: AgentResult) -> None:
        self.agent_results.append(result)
        self.add_findings(result.findings)

    @property
    def findings_by_type(self) -> dict:
        """Group findings by type."""
        out: dict = {}
        for f in self.findings:
            key = f.type.value
            out.setdefault(key, []).append(f)
        return out

    @property
    def high_severity_count(self) -> int:
        from veritas.models.finding import Severity
        return sum(
            1 for f in self.findings
            if f.severity in (Severity.HIGH, Severity.CRITICAL)
        )

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "document_name": self.document_name,
            "document_words": self.document_words,
            "document_pages": self.document_pages,
            "score": self.score.to_dict(),
            "findings_count": len(self.findings),
            "findings": [f.to_dict() for f in self.findings],
            "agent_results": [r.to_dict() for r in self.agent_results],
            "created_at": self.created_at,
            "duration_seconds": round(self.duration_seconds, 3),
            "human_review_recommended": self.human_review_recommended,
            "notes": self.notes,
        }

    def summary(self) -> str:
        """Human-readable summary."""
        s = self.score
        lines = [
            "=" * 60,
            "  VERITAS-AI — ACADEMIC INTEGRITY REPORT",
            "=" * 60,
            f"  Document:        {self.document_name}",
            f"  Words:           {self.document_words:,}",
            f"  Pages:           {self.document_pages}",
            "",
            "-" * 60,
            "  SIGNALS",
            "-" * 60,
            f"  Exact overlap          {s.exact_overlap * 100:.1f}%",
            f"  Semantic overlap       {s.semantic_overlap * 100:.1f}%",
            f"  Paraphrase overlap     {s.paraphrase_overlap * 100:.1f}%",
            f"  Cross-language         {s.cross_language_overlap * 100:.1f}%",
            f"  Self-overlap           {s.self_overlap * 100:.1f}%",
            f"  Source matches         {s.source_matches}",
            f"  Citation issues        {s.citation_issues}",
            f"  Reference issues       {s.reference_issues}",
            f"  Collusion clusters     {s.collusion_clusters}",
            "",
            f"  AI-writing signal      {s.ai_writing_signal}",
            f"  Overall confidence     {s.overall_confidence}",
            "",
            f"  Findings (total)       {len(self.findings)}",
            f"  High severity          {self.high_severity_count}",
            "",
            f"  Human review needed:   {'YES' if self.human_review_recommended else 'NO'}",
            "=" * 60,
        ]
        return "\n".join(lines)


__all__ = ["IntegrityScore", "Report"]