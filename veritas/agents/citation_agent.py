"""
Citation Integrity Agent.

Checks:
- Missing references for inline citations
- Unused references
- Excessive citation usage
- Uncited claim-like statements
- Citation coverage

Input:
    context.document_text  → full text
    context.extras["references"]  → OR extract from text

Output:
    Findings describing citation issues
"""
from __future__ import annotations

from typing import Optional

from veritas.agents.base import BaseAgent, AgentContext
from veritas.models.finding import (
    AgentResult, Evidence, Finding, FindingType, Severity,
)
from veritas.tools.citation_parser import (
    analyze_citations, find_uncited_claims,
)
from veritas.tools.text_utils import clean_text


class CitationAgent(BaseAgent):
    """
    Analyze citation integrity.

    Detects:
    - Missing references
    - Unused references
    - Excessive citations
    - Uncited claims
    """

    name = "citation_agent"
    description = "Checks citation and reference integrity."

    def __init__(self, run_id: str = "", verbose: bool = True):
        super().__init__(run_id=run_id, verbose=verbose)

    def run(self, task: str, context: Optional[AgentContext] = None) -> AgentResult:
        start = self._time_start()

        if context is None or not context.document_text:
            return AgentResult.failed(self.name, "No document text")

        text = clean_text(context.document_text)

        # References: from extras or extract from text
        ref_lines = context.extras.get("references") or []
        if not ref_lines:
            # Try to extract from text
            from veritas.tools.text_utils import detect_references
            ref_lines, _ = detect_references(text)

        if not text.strip():
            return AgentResult.failed(self.name, "Empty document")

        self.log_info(f"Analyzing citation integrity ({len(ref_lines)} refs)")

        # Analysis
        analysis = analyze_citations(text, ref_lines)
        uncited_claims = find_uncited_claims(text)

        metrics = {
            "total_citations": analysis["total_citations"],
            "numeric_citations": analysis["numeric_count"],
            "author_year_citations": analysis["author_year_count"],
            "total_references": analysis["total_references"],
            "missing_refs": len(analysis["missing_refs"]),
            "unused_refs": len(analysis["unused_refs"]),
            "duplicate_citations": len(analysis["duplicate_citations"]),
            "uncited_claims": len(uncited_claims),
            "coverage": analysis["coverage"],
        }

        findings: list[Finding] = []

        # Finding 1: Missing references
        if analysis["missing_refs"]:
            evidence_list = [
                Evidence(
                    submitted_text=f"Citation [{n}]",
                    source_text="(no matching reference)",
                    source_document=context.document_path or "",
                    similarity=1.0,
                    method="citation_check",
                    metadata={"missing_number": n},
                )
                for n in analysis["missing_refs"][:8]
            ]

            findings.append(self.make_finding(
                finding_type=FindingType.CITATION_ISSUE,
                title=f"Missing references for {len(analysis['missing_refs'])} citation(s)",
                description=(
                    f"Inline citations reference numbers that don't exist "
                    f"in the reference list: {analysis['missing_refs'][:5]}"
                ),
                confidence=0.95,
                evidence=evidence_list,
                explanation=[
                    f"Numbers cited in text: {analysis['unique_cited_numbers']}",
                    f"Numbers in reference list: {analysis['reference_numbers']}",
                    f"Missing: {analysis['missing_refs']}",
                    "This indicates either a citation typo or a "
                    "dangling reference.",
                ],
                severity=Severity.HIGH,
                metadata={"missing_refs": analysis["missing_refs"]},
            ))

        # Finding 2: Unused references
        if analysis["unused_refs"]:
            findings.append(self.make_finding(
                finding_type=FindingType.CITATION_ISSUE,
                title=f"Unused references: {len(analysis['unused_refs'])} entries",
                description=(
                    f"Reference list contains entries never cited in text: "
                    f"{analysis['unused_refs'][:5]}"
                ),
                confidence=0.9,
                explanation=[
                    f"Unused numbers: {analysis['unused_refs']}",
                    "May indicate padding or leftover references.",
                ],
                severity=Severity.LOW,
                metadata={"unused_refs": analysis["unused_refs"]},
            ))

        # Finding 3: Uncited claims
        if uncited_claims:
            evidence_list = [
                Evidence(
                    submitted_text=c["sentence"][:200],
                    source_text=f"(claim indicator: {c['matched']})",
                    similarity=0.0,
                    method="claim_detection",
                    metadata=c,
                )
                for c in uncited_claims[:5]
            ]

            findings.append(self.make_finding(
                finding_type=FindingType.CITATION_ISSUE,
                title=f"{len(uncited_claims)} claim(s) without citations",
                description=(
                    "Statements that look like factual claims "
                    "lack citations."
                ),
                confidence=0.65,
                evidence=evidence_list,
                explanation=[
                    "Sentence patterns that typically need citations:",
                    "  • 'studies show', 'research shows'",
                    "  • 'according to'",
                    "  • 'X%' percentages",
                    "  • 'evidence suggests'",
                    "These may need supporting references.",
                ],
                severity=Severity.MEDIUM,
                metadata={"uncited_claims": len(uncited_claims)},
            ))

        # Finding 4: Excessive citations
        if analysis["duplicate_citations"]:
            findings.append(self.make_finding(
                finding_type=FindingType.CITATION_ISSUE,
                title=f"Over-cited: {len(analysis['duplicate_citations'])} source(s)",
                description=(
                    f"Some citations are used 3+ times: "
                    f"{list(analysis['duplicate_citations'].items())[:5]}"
                ),
                confidence=0.55,
                explanation=[
                    "Frequent citation of the same source may indicate:",
                    "  • Over-reliance on a single source",
                    "  • Missing diverse sources",
                ],
                severity=Severity.INFO,
                metadata={"duplicates": analysis["duplicate_citations"]},
            ))

        findings.sort(key=lambda f: f.confidence, reverse=True)

        duration = self._time_end(start)
        self.log_success(
            f"Found {len(findings)} citation finding(s) in {duration:.2f}s"
        )

        return AgentResult(
            agent=self.name,
            status="ok",
            findings=findings,
            metrics=metrics,
            notes=(
                f"Citations: {analysis['total_citations']} "
                f"Refs: {analysis['total_references']} "
                f"Coverage: {analysis['coverage']:.0%}"
            ),
            duration_seconds=duration,
        )


__all__ = ["CitationAgent"]