"""Research Integrity Agent — combined checks for papers."""
from __future__ import annotations
import re
from collections import Counter
from typing import Optional

from veritas.agents.base import BaseAgent, AgentContext
from veritas.models.finding import (
    AgentResult, Evidence, Finding, FindingType, Severity,
)
from veritas.tools.text_utils import clean_text, split_sentences
from veritas.tools.ngram import tokenize
from veritas.tools.citation_parser import (
    extract_citations, analyze_citations,
)


class ResearchIntegrityAgent(BaseAgent):
    name = "research_integrity_agent"
    description = "Research-paper-specific integrity checks."

    def run(self, task: str, context: Optional[AgentContext] = None) -> AgentResult:
        start = self._time_start()
        if context is None or not context.document_text:
            return AgentResult.failed(self.name, "No document text")

        text = clean_text(context.document_text)
        ref_lines = context.extras.get("references") or []
        if not ref_lines:
            from veritas.tools.text_utils import detect_references
            ref_lines, _ = detect_references(text)

        findings = []
        metrics = {}

        # 1. Duplicated paragraphs
        dup = self._find_duplicate_paragraphs(text)
        metrics["duplicate_paragraphs"] = len(dup)

        if dup:
            findings.append(self.make_finding(
                finding_type=FindingType.RESEARCH_INTEGRITY,
                title=f"Duplicate paragraphs: {len(dup)}",
                description="Same paragraph appears multiple times.",
                confidence=0.95,
                evidence=[
                    Evidence(submitted_text=d["text"][:200],
                             source_text=f"Appears {d['count']} times",
                             similarity=1.0,
                             method="duplicate_detection")
                    for d in dup[:5]
                ],
                explanation=[f"{len(dup)} paragraph(s) appear 2+ times"],
                severity=Severity.MEDIUM,
            ))

        # 2. Repeated sections (headings)
        headings = self._find_repeated_headings(text)
        metrics["repeated_headings"] = len(headings)
        if headings:
            findings.append(self.make_finding(
                finding_type=FindingType.RESEARCH_INTEGRITY,
                title=f"Repeated section headings: {len(headings)}",
                description="Same heading appears multiple times.",
                confidence=0.85,
                evidence=[
                    Evidence(submitted_text=h["heading"],
                             similarity=1.0, method="heading_repeat")
                    for h in headings[:5]
                ],
                explanation=[f"{len(headings)} heading(s) repeated"],
                severity=Severity.LOW,
            ))

        # 3. Terminology inconsistency
        terms = self._find_terminology_variance(text)
        metrics["terminology_variance"] = len(terms)
        if terms:
            findings.append(self.make_finding(
                finding_type=FindingType.RESEARCH_INTEGRITY,
                title=f"Inconsistent terminology: {len(terms)} pairs",
                description="Same concept referred to with different terms.",
                confidence=0.55,
                evidence=[
                    Evidence(submitted_text=t["a"] + " ↔ " + t["b"],
                             similarity=0.5, method="term_variance")
                    for t in terms[:5]
                ],
                explanation=[f"{len(terms)} concept(s) use inconsistent terms"],
                severity=Severity.LOW,
            ))

        # 4. Citation-reference mismatch (via citation parser)
        analysis = analyze_citations(text, ref_lines)
        metrics["citation_issues"] = len(analysis["issues"])
        if analysis["missing_refs"] or analysis["unused_refs"]:
            findings.append(self.make_finding(
                finding_type=FindingType.RESEARCH_INTEGRITY,
                title="Citation/reference mismatches",
                description=(
                    f"Missing: {len(analysis['missing_refs'])}, "
                    f"Unused: {len(analysis['unused_refs'])}"
                ),
                confidence=0.9,
                evidence=[],
                explanation=[
                    f"Missing refs: {analysis['missing_refs'][:10]}",
                    f"Unused refs: {analysis['unused_refs'][:10]}",
                ],
                severity=Severity.MEDIUM,
            ))

        duration = self._time_end(start)
        self.log_success(f"Research integrity: {len(findings)} findings ({duration:.2f}s)")

        return AgentResult(agent=self.name, status="ok", findings=findings,
                           metrics=metrics,
                           notes=f"{len(findings)} integrity issues",
                           duration_seconds=duration)

    def _find_duplicate_paragraphs(self, text):
        paras = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 80]
        seen = Counter(paras)
        return [
            {"text": p, "count": c}
            for p, c in seen.items() if c >= 2
        ]

    def _find_repeated_headings(self, text):
        heads = []
        for line in text.split("\n"):
            line = line.strip()
            if 3 <= len(line.split()) <= 10 and line[0:1].isupper():
                if not line.endswith("."):
                    heads.append(line)
        counter = Counter(heads)
        return [{"heading": h, "count": c}
                for h, c in counter.items() if c >= 2]

    def _find_terminology_variance(self, text):
        # Simple: look for known pairs
        pairs = [
            ("artificial intelligence", "AI"),
            ("machine learning", "ML"),
            ("deep learning", "DL"),
            ("neural network", "NN"),
            ("natural language processing", "NLP"),
        ]
        lower = text.lower()
        out = []
        for full, abbr in pairs:
            if full in lower and abbr.lower() in lower:
                # Check if used inconsistently
                full_count = lower.count(full)
                abbr_count = lower.count(abbr.lower())
                if full_count >= 3 and abbr_count >= 3:
                    out.append({"a": full, "b": abbr,
                                "full_count": full_count,
                                "abbr_count": abbr_count})
        return out


__all__ = ["ResearchIntegrityAgent"]