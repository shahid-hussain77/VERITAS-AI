"""Writing Style Agent — stylometric deviation analysis."""
from __future__ import annotations
from typing import Optional

from veritas.agents.base import BaseAgent, AgentContext
from veritas.models.finding import (
    AgentResult, Evidence, Finding, FindingType, Severity,
)
from veritas.models.document import Document
from veritas.tools.style_analyzer import compute_fingerprint, style_deviation
from veritas.tools.text_utils import clean_text


class StyleAgent(BaseAgent):
    name = "style_agent"
    description = "Analyzes writing style deviation from author history."

    def run(self, task: str, context: Optional[AgentContext] = None) -> AgentResult:
        start = self._time_start()
        if context is None or not context.document_text:
            return AgentResult.failed(self.name, "No document text")

        text = clean_text(context.document_text)
        if len(text.split()) < 100:
            return AgentResult(agent=self.name, status="ok", findings=[],
                               metrics={"word_count": len(text.split())},
                               notes="Text too short for stylometry",
                               duration_seconds=self._time_end(start))

        history = context.extras.get("author_history") or []
        if not history:
            return AgentResult(agent=self.name, status="ok", findings=[],
                               metrics={"history_size": 0},
                               notes="No author history for style comparison",
                               duration_seconds=self._time_end(start))

        self.log_info(f"Stylometry: current vs {len(history)} historical doc(s)")

        current_fp = compute_fingerprint(text)
        deviations = []
        for old_doc in history:
            if not isinstance(old_doc, Document):
                continue
            old_text = clean_text(old_doc.text)
            if len(old_text.split()) < 100:
                continue
            old_fp = compute_fingerprint(old_text)
            dev = style_deviation(old_fp, current_fp)
            deviations.append((old_doc, dev))

        if not deviations:
            return AgentResult(agent=self.name, status="ok", findings=[],
                               metrics={"history_size": len(history),
                                        "comparable": 0},
                               notes="No comparable history",
                               duration_seconds=self._time_end(start))

        avg_dev = sum(d["overall"] for _, d in deviations) / len(deviations)

        metrics = {
            "history_size": len(history),
            "comparable": len(deviations),
            "avg_deviation": round(avg_dev, 4),
            "current_fingerprint": current_fp.to_dict(),
        }

        findings = []

        # Only flag if deviation is significant
        if avg_dev >= 0.15:
            evidence_list = []
            for old_doc, dev in deviations[:3]:
                per_feature = "; ".join(
                    f"{k}={v:.3f}" for k, v in dev["per_feature"].items()
                )
                evidence_list.append(Evidence(
                    submitted_text=f"Historical doc: {old_doc.source_name}",
                    source_text=f"Deviation: {dev['overall']:.4f}",
                    source_document=old_doc.source_name,
                    similarity=dev["overall"],
                    method="style_deviation",
                    metadata={"per_feature": dev["per_feature"]},
                ))

            severity = Severity.HIGH if avg_dev >= 0.30 else Severity.MEDIUM

            findings.append(self.make_finding(
                finding_type=FindingType.STYLE_DEVIATION,
                title=f"Writing style deviation: {avg_dev:.2%}",
                description=(
                    f"Current submission's writing style deviates "
                    f"{avg_dev:.1%} from author's historical style."
                ),
                confidence=min(0.8, avg_dev * 2),
                evidence=evidence_list,
                explanation=[
                    f"Compared against {len(deviations)} historical document(s)",
                    f"Average deviation: {avg_dev:.2%}",
                    f"Current fingerprint:",
                    f"  • Avg sentence length: {current_fp.avg_sentence_length:.1f}",
                    f"  • Vocabulary richness: {current_fp.vocabulary_richness:.3f}",
                    f"  • Function word ratio: {current_fp.function_word_ratio:.3f}",
                    "⚠️ Style change may be legitimate (topic, mood, time).",
                    "This is only an evidence signal, NOT proof of misconduct.",
                ],
                severity=severity,
                metadata={"avg_deviation": round(avg_dev, 4),
                          "current_fingerprint": current_fp.to_dict(),
                          "per_document": [
                              {"source": d.source_name,
                               "deviation": dev["overall"]}
                              for d, dev in deviations
                          ]},
            ))

        duration = self._time_end(start)
        self.log_success(f"Style deviation: {avg_dev:.4f} ({duration:.2f}s)")

        return AgentResult(agent=self.name, status="ok", findings=findings,
                           metrics=metrics,
                           notes=f"Deviation: {avg_dev:.2%}",
                           duration_seconds=duration)


__all__ = ["StyleAgent"]