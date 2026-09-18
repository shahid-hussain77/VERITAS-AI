"""
Paraphrase Attack Agent.

Unlike the Semantic Agent (which detects meaning similarity),
this agent detects SPECIFIC PARAPHRASE TECHNIQUES:

- Synonym replacement
- Voice change (active ↔ passive)
- Nominalization (verb → noun)
- Word order shuffling
- Structural changes

Output identifies WHICH techniques were used.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from veritas.agents.base import BaseAgent, AgentContext
from veritas.config import THRESHOLDS
from veritas.models.finding import (
    AgentResult, Evidence, Finding, FindingType, Severity,
)
from veritas.models.document import Document
from veritas.tools.embeddings import Embedder
from veritas.tools.text_utils import split_sentences, clean_text
from veritas.tools.paraphrase_detector import paraphrase_indicators


class ParaphraseAgent(BaseAgent):
    """
    Detect paraphrase attacks.

    Combines:
    - Semantic similarity (need high meaning match)
    - Paraphrase indicators (technique detection)
    """

    name = "paraphrase_agent"
    description = "Detects paraphrased passages with technique identification."

    def __init__(self, run_id: str = "", verbose: bool = True):
        super().__init__(run_id=run_id, verbose=verbose)
        self.embedder = Embedder.get(verbose=False)

    def run(self, task: str, context: Optional[AgentContext] = None) -> AgentResult:
        start = self._time_start()

        if context is None or not context.document_text:
            return AgentResult.failed(self.name, "No document text in context")

        submitted = clean_text(context.document_text)
        sources = self._collect_sources(context)

        if not sources:
            return AgentResult(
                agent=self.name,
                status="ok",
                findings=[],
                metrics={"sources_compared": 0},
                notes="No source documents",
                duration_seconds=self._time_end(start),
            )

        self.log_info(f"Paraphrase check: 1 vs {len(sources)} source(s)")

        sub_sentences = [
            s for s in split_sentences(submitted) if len(s.split()) >= 8
        ]

        if not sub_sentences:
            return AgentResult(
                agent=self.name,
                status="ok",
                findings=[],
                metrics={"sentence_count": 0},
                notes="Submitted text too short",
                duration_seconds=self._time_end(start),
            )

        all_findings: list[Finding] = []
        metrics = {
            "sources_compared": len(sources),
            "submitted_sentences": len(sub_sentences),
            "paraphrase_matches": 0,
        }

        for src_name, src_text in sources:
            findings, src_metrics = self._compare(
                sub_sentences, src_text, src_name,
            )
            all_findings.extend(findings)
            metrics["paraphrase_matches"] += src_metrics["matches"]

        all_findings.sort(key=lambda f: f.confidence, reverse=True)

        duration = self._time_end(start)
        self.log_success(
            f"Found {len(all_findings)} paraphrase finding(s) in {duration:.2f}s"
        )

        return AgentResult(
            agent=self.name,
            status="ok",
            findings=all_findings,
            metrics=metrics,
            notes=f"Paraphrase comparison with {len(sources)} source(s)",
            duration_seconds=duration,
        )

    def _collect_sources(self, context: AgentContext) -> list[tuple[str, str]]:
        sources = []
        for d in context.extras.get("source_documents", []) or []:
            if isinstance(d, Document):
                sources.append((d.source_name, d.text))
        for t in context.extras.get("source_texts", []) or []:
            if isinstance(t, str) and t.strip():
                sources.append(("source", t))
        for d in context.other_documents or []:
            if isinstance(d, Document):
                sources.append((d.source_name, d.text))
        return sources

    def _compare(
        self,
        submitted_sentences: list[str],
        source_text: str,
        source_name: str,
    ) -> tuple[list[Finding], dict]:
        findings: list[Finding] = []
        metrics = {"matches": 0}

        source_sentences = [
            s for s in split_sentences(source_text) if len(s.split()) >= 8
        ]

        if not source_sentences:
            return findings, metrics

        # Embed
        sub_vecs = self.embedder.embed_en(submitted_sentences)
        src_vecs = self.embedder.embed_en(source_sentences)
        sim_matrix = np.dot(sub_vecs, src_vecs.T)

        evidence_list: list[Evidence] = []
        technique_counts = {
            "synonym": 0,
            "voice": 0,
            "nominalization": 0,
            "word_order": 0,
        }

        for i, sub_sent in enumerate(submitted_sentences):
            row = sim_matrix[i]
            # Check top candidate
            best_j = int(np.argmax(row))
            best_sim = float(row[best_j])

            # Must have meaningful semantic similarity
            if best_sim < THRESHOLDS.paraphrase_min:
                continue

            src_sent = source_sentences[best_j]

            # Compute paraphrase indicators
            indicators = paraphrase_indicators(sub_sent, src_sent)

            # Must show at least one strong paraphrase signal
            has_synonym = indicators["synonym_density"] >= 0.30
            has_voice = indicators["voice_changed"]
            has_nom = indicators["nominalization"] >= 0.30
            has_order = indicators["word_order_change"] >= 0.30

            techniques_used = []
            if has_synonym:
                techniques_used.append("synonym_replacement")
                technique_counts["synonym"] += 1
            if has_voice:
                techniques_used.append("voice_change")
                technique_counts["voice"] += 1
            if has_nom:
                techniques_used.append("nominalization")
                technique_counts["nominalization"] += 1
            if has_order:
                techniques_used.append("word_order_change")
                technique_counts["word_order"] += 1

            # Require at least one technique
            if not techniques_used:
                continue

            # Combined score
            combined = (
                0.6 * best_sim + 0.4 * indicators["overall"]
            )

            evidence_list.append(Evidence(
                submitted_text=sub_sent,
                source_text=src_sent,
                source_document=source_name,
                similarity=combined,
                method="paraphrase_analysis",
                metadata={
                    "semantic_similarity": round(best_sim, 4),
                    "indicators": indicators,
                    "techniques": techniques_used,
                },
            ))

        if not evidence_list:
            return findings, metrics

        metrics["matches"] = len(evidence_list)

        # Sort by combined score
        evidence_list.sort(key=lambda e: e.similarity, reverse=True)
        top_evidence = evidence_list[:8]

        # Average
        avg_score = sum(e.similarity for e in top_evidence) / len(top_evidence)

        # Determine dominant techniques
        dominant = [k for k, v in technique_counts.items() if v >= 2]
        if not dominant:
            dominant = [k for k, v in technique_counts.items() if v >= 1]

        # Severity
        if avg_score >= 0.85:
            severity = Severity.CRITICAL
        elif avg_score >= 0.75:
            severity = Severity.HIGH
        elif avg_score >= 0.65:
            severity = Severity.MEDIUM
        else:
            severity = Severity.LOW

        explanation = [
            f"Detected {len(evidence_list)} sentence(s) with paraphrase patterns",
            f"Average combined score: {avg_score:.2%}",
            f"Source: {source_name}",
            "",
            "Paraphrase techniques detected:",
        ]
        for tech in dominant:
            explanation.append(
                f"  • {tech.replace('_', ' ')}: {technique_counts[tech]} time(s)"
            )

        finding = self.make_finding(
            finding_type=FindingType.PARAPHRASE,
            title=f"Paraphrase detected against {source_name}",
            description=(
                f"{len(evidence_list)} sentence(s) show paraphrase patterns. "
                f"Techniques: {', '.join(dominant)}."
            ),
            confidence=avg_score,
            evidence=top_evidence,
            explanation=explanation,
            severity=severity,
            metadata={
                "source": source_name,
                "matches": len(evidence_list),
                "avg_score": round(avg_score, 4),
                "technique_counts": technique_counts,
                "dominant_techniques": dominant,
            },
        )
        findings.append(finding)
        return findings, metrics


__all__ = ["ParaphraseAgent"]