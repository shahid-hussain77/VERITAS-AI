"""
Fake Citation Agent.

Analyzes references for fabricated patterns.

Output:
    Findings listing suspicious references with evidence
"""
from __future__ import annotations

from typing import Optional

from veritas.agents.base import BaseAgent, AgentContext
from veritas.models.finding import (
    AgentResult, Evidence, Finding, FindingType, Severity,
)
from veritas.tools.fake_citation_detector import analyze_references
from veritas.tools.text_utils import clean_text


class FakeCitationAgent(BaseAgent):
    """Detect fabricated / suspicious references."""

    name = "fake_citation_agent"
    description = "Detects potentially fabricated references."

    def __init__(self, run_id: str = "", verbose: bool = True):
        super().__init__(run_id=run_id, verbose=verbose)

    def run(self, task: str, context: Optional[AgentContext] = None) -> AgentResult:
        start = self._time_start()

        if context is None or not context.document_text:
            return AgentResult.failed(self.name, "No document text")

        text = clean_text(context.document_text)

        ref_lines = context.extras.get("references") or []
        if not ref_lines:
            from veritas.tools.text_utils import detect_references
            ref_lines, _ = detect_references(text)

        if not ref_lines:
            return AgentResult(
                agent=self.name,
                status="ok",
                findings=[],
                metrics={"references_analyzed": 0},
                notes="No references found",
                duration_seconds=self._time_end(start),
            )

        self.log_info(f"Analyzing {len(ref_lines)} reference(s)")

        analyses = analyze_references(ref_lines)

        # Filter suspicious
        suspicious = [
            a for a in analyses
            if a.risk_level in ("medium", "high", "critical")
        ]

        metrics = {
            "references_analyzed": len(analyses),
            "suspicious": len(suspicious),
            "high_risk": sum(1 for a in analyses if a.risk_level == "high"),
            "critical_risk": sum(1 for a in analyses if a.risk_level == "critical"),
        }

        findings: list[Finding] = []

        if suspicious:
            # Sort by risk
            suspicious.sort(key=lambda a: a.fake_probability, reverse=True)

            # Evidence
            evidence_list = []
            for a in suspicious[:8]:
                # Build signal summary
                signal_summary = "; ".join(
                    f"{s.name}: {s.detail}"
                    for s in a.signals
                    if s.status in ("warn", "fail")
                )

                evidence_list.append(Evidence(
                    submitted_text=a.reference_raw[:300],
                    source_text=f"(risk: {a.risk_level}, score: {a.fake_probability:.2f})",
                    similarity=a.fake_probability,
                    method="fake_citation_analysis",
                    metadata={
                        "risk_level": a.risk_level,
                        "fake_probability": round(a.fake_probability, 4),
                        "signals": signal_summary,
                        "reference_index": a.reference_index,
                    },
                ))

            avg_prob = sum(a.fake_probability for a in suspicious) / len(suspicious)

            if avg_prob >= 0.60:
                severity = Severity.HIGH
            elif avg_prob >= 0.40:
                severity = Severity.MEDIUM
            else:
                severity = Severity.LOW

            explanation = [
                f"Analyzed {len(analyses)} references",
                f"Suspicious: {len(suspicious)}",
                f"High-risk: {metrics['high_risk']}",
                f"Critical: {metrics['critical_risk']}",
                "",
                "Signals used:",
                "  • DOI presence and format",
                "  • Journal name in known list",
                "  • Publication year plausibility",
                "  • Author name patterns",
                "  • AI-template patterns",
                "",
                "⚠️ Offline analysis only. Enable online mode "
                "(Crossref/OpenAlex) for full verification.",
            ]

            finding = self.make_finding(
                finding_type=FindingType.FAKE_CITATION,
                title=f"Suspicious references: {len(suspicious)}",
                description=(
                    f"{len(suspicious)} reference(s) show patterns "
                    f"consistent with fabrication."
                ),
                confidence=min(0.9, avg_prob + 0.2),
                evidence=evidence_list,
                explanation=explanation,
                severity=severity,
                metadata={
                    "suspicious_count": len(suspicious),
                    "avg_probability": round(avg_prob, 4),
                    "details": [a.to_dict() for a in suspicious[:5]],
                },
            )
            findings.append(finding)

        duration = self._time_end(start)
        self.log_success(
            f"Found {len(suspicious)} suspicious reference(s) "
            f"in {duration:.2f}s"
        )

        return AgentResult(
            agent=self.name,
            status="ok",
            findings=findings,
            metrics=metrics,
            notes=f"Analyzed {len(analyses)} references",
            duration_seconds=duration,
        )


__all__ = ["FakeCitationAgent"]