"""AI-Writing Signals Agent."""
from __future__ import annotations
from typing import Optional

from veritas.agents.base import BaseAgent, AgentContext
from veritas.models.finding import (
    AgentResult, Evidence, Finding, FindingType, Severity,
)
from veritas.tools.ai_writing_detector import analyze_ai_writing
from veritas.tools.text_utils import clean_text


class AIWritingAgent(BaseAgent):
    name = "ai_writing_agent"
    description = "Detects AI-writing statistical signals."

    def run(self, task: str, context: Optional[AgentContext] = None) -> AgentResult:
        start = self._time_start()
        if context is None or not context.document_text:
            return AgentResult.failed(self.name, "No document text")

        text = clean_text(context.document_text)
        if len(text.split()) < 50:
            return AgentResult(agent=self.name, status="ok", findings=[],
                               metrics={"word_count": len(text.split())},
                               notes="Document too short",
                               duration_seconds=self._time_end(start))

        self.log_info("Analyzing AI-writing signals")
        analysis = analyze_ai_writing(text)

        metrics = {
            "overall_score": round(analysis.overall_score, 4),
            "overall_level": analysis.overall_level,
            **analysis.metrics,
        }

        findings = []

        # Only report if level is MEDIUM or HIGH
        if analysis.overall_level in ("MEDIUM", "HIGH"):
            evidence_list = [
                Evidence(
                    submitted_text=f"Signal: {s.name}",
                    source_text=f"Level: {s.level}, Score: {s.score:.3f}",
                    similarity=s.score,
                    method="ai_writing_signal",
                    metadata={"name": s.name, "level": s.level,
                              "score": round(s.score, 4), "detail": s.detail},
                )
                for s in analysis.signals
            ]

            if analysis.overall_level == "HIGH":
                severity = Severity.HIGH
                conf = 0.75
            else:
                severity = Severity.MEDIUM
                conf = 0.55

            explanation = [
                f"Overall AI-writing score: {analysis.overall_score:.2%}",
                f"Classification: {analysis.overall_level}",
                "",
                "Signal breakdown:",
            ]
            for s in analysis.signals:
                explanation.append(f"  • {s.name}: {s.level.upper()} ({s.score:.2f})")
                explanation.append(f"    {s.detail}")

            explanation.extend([
                "",
                "⚠️ IMPORTANT: AI-detection is NOT definitive evidence.",
                "Multiple signals together suggest possible AI assistance.",
                "Human review STRONGLY recommended.",
                "This is not proof of misconduct.",
            ])

            findings.append(self.make_finding(
                finding_type=FindingType.AI_WRITING,
                title=f"AI-writing signals: {analysis.overall_level}",
                description=(
                    f"Statistical analysis shows {analysis.overall_level.lower()} "
                    f"AI-writing indicators."
                ),
                confidence=conf,
                evidence=evidence_list,
                explanation=explanation,
                severity=severity,
                metadata={"overall_score": round(analysis.overall_score, 4),
                          "overall_level": analysis.overall_level,
                          "signals": analysis.to_dict()["signals"]},
            ))
        else:
            # Low / INCONCLUSIVE — still report as INFO
            findings.append(self.make_finding(
                finding_type=FindingType.AI_WRITING,
                title=f"AI-writing signals: {analysis.overall_level}",
                description="No strong AI-writing signals detected.",
                confidence=0.5,
                evidence=[],
                explanation=[
                    f"Overall score: {analysis.overall_score:.2%}",
                    "Mixed or weak signals.",
                ],
                severity=Severity.INFO,
                metadata={"overall_score": round(analysis.overall_score, 4),
                          "overall_level": analysis.overall_level},
            ))

        duration = self._time_end(start)
        self.log_success(f"AI-writing level: {analysis.overall_level} ({duration:.2f}s)")

        return AgentResult(agent=self.name, status="ok", findings=findings,
                           metrics=metrics,
                           notes=f"AI-writing: {analysis.overall_level}",
                           duration_seconds=duration)


__all__ = ["AIWritingAgent"]