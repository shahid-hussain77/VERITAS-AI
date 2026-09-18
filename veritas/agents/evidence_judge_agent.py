"""
Evidence Judge Agent — fuses all agent results into a final report.

Does NOT detect anything itself.
Combines findings from all agents + produces IntegrityScore + Report.
"""
from __future__ import annotations
from typing import Optional

from veritas.agents.base import BaseAgent, AgentContext
from veritas.config import REPORT
from veritas.models.finding import AgentResult, Finding, Severity, FindingType
from veritas.models.report import Report, IntegrityScore


class EvidenceJudgeAgent(BaseAgent):
    name = "evidence_judge_agent"
    description = "Fuses all agent findings into final report."

    def run(self, task: str, context: Optional[AgentContext] = None) -> AgentResult:
        start = self._time_start()
        if context is None:
            return AgentResult.failed(self.name, "No context")

        # All agent results collected by orchestrator
        agent_results: list[AgentResult] = context.extras.get("agent_results", [])
        doc_name = context.metadata.get("document_name", "document")
        doc_id = context.document_id
        word_count = len(context.document_text.split()) if context.document_text else 0
        page_count = context.metadata.get("page_count", 0)

        if not agent_results:
            return AgentResult.failed(self.name, "No agent results to fuse")

        self.log_info(f"Fusing {len(agent_results)} agent result(s)")

        # Build score
        score = IntegrityScore()
        all_findings: list[Finding] = []

        for ar in agent_results:
            all_findings.extend(ar.findings)
            for f in ar.findings:
                ft = f.type
                if ft == FindingType.EXACT_COPY:
                    score.exact_overlap = max(score.exact_overlap, f.confidence)
                elif ft == FindingType.SEMANTIC_SIMILARITY:
                    score.semantic_overlap = max(score.semantic_overlap, f.confidence)
                elif ft == FindingType.PARAPHRASE:
                    score.paraphrase_overlap = max(score.paraphrase_overlap, f.confidence)
                elif ft == FindingType.CROSS_LANGUAGE:
                    score.cross_language_overlap = max(score.cross_language_overlap, f.confidence)
                elif ft == FindingType.SELF_PLAGIARISM:
                    score.self_overlap = max(score.self_overlap, f.confidence)
                elif ft == FindingType.SOURCE_MATCH:
                    score.source_matches = max(score.source_matches,
                                               len(f.evidence))
                elif ft == FindingType.CITATION_ISSUE:
                    if "missing" in f.title.lower():
                        score.citation_issues = max(score.citation_issues,
                                                    len(f.evidence) or 1)
                    else:
                        score.citation_issues += 1
                elif ft == FindingType.FAKE_CITATION:
                    score.reference_issues = len(f.evidence)
                elif ft == FindingType.AI_WRITING:
                    lvl = f.metadata.get("overall_level", "INCONCLUSIVE")
                    if lvl in ("HIGH", "MEDIUM", "LOW"):
                        score.ai_writing_signal = lvl
                elif ft == FindingType.STYLE_DEVIATION:
                    score.style_deviation = max(score.style_deviation, f.confidence)
                elif ft == FindingType.COLLUSION:
                    clusters = f.metadata.get("clusters") or []
                    score.collusion_clusters = max(score.collusion_clusters,
                                                   len(clusters))

        # Human review decision
        high_sev = [f for f in all_findings
                    if f.severity in (Severity.HIGH, Severity.CRITICAL)]
        human_review = len(high_sev) >= 1 or score.overall_confidence == "HIGH"

        # Build report
        report = Report(
            document_id=doc_id,
            document_name=doc_name,
            document_words=word_count,
            document_pages=page_count,
            score=score,
            findings=all_findings,
            agent_results=agent_results,
            human_review_recommended=human_review,
        )

        # Build a meta-finding summarizing the report
        summary_finding = self.make_finding(
            finding_type=FindingType.RESEARCH_INTEGRITY,
            title="Evidence Fusion Summary",
            description=(
                f"Fused {len(agent_results)} agents, {len(all_findings)} findings. "
                f"Confidence: {score.overall_confidence}, "
                f"Human review: {'YES' if human_review else 'NO'}"
            ),
            confidence=0.95,
            evidence=[],
            explanation=[
                f"Exact overlap: {score.exact_overlap:.2%}",
                f"Semantic overlap: {score.semantic_overlap:.2%}",
                f"Paraphrase overlap: {score.paraphrase_overlap:.2%}",
                f"Cross-language: {score.cross_language_overlap:.2%}",
                f"Self-overlap: {score.self_overlap:.2%}",
                f"Source matches: {score.source_matches}",
                f"Citation issues: {score.citation_issues}",
                f"Reference issues: {score.reference_issues}",
                f"AI-writing: {score.ai_writing_signal}",
                f"Style deviation: {score.style_deviation:.2%}",
                f"Collusion clusters: {score.collusion_clusters}",
                "",
                f"Overall confidence: {score.overall_confidence}",
                f"High severity findings: {len(high_sev)}",
            ],
            severity=Severity.INFO,
            metadata={"report": report.to_dict()},
        )

        duration = self._time_end(start)
        self.log_success(
            f"Report built: {len(all_findings)} findings, "
            f"confidence={score.overall_confidence} ({duration:.2f}s)"
        )

        return AgentResult(
            agent=self.name,
            status="ok",
            findings=[summary_finding],
            metrics={
                "total_findings": len(all_findings),
                "high_severity": len(high_sev),
                "overall_confidence": score.overall_confidence,
                "human_review": human_review,
                "report": report.to_dict(),
            },
            notes="Final report built",
            duration_seconds=duration,
        )


__all__ = ["EvidenceJudgeAgent"]