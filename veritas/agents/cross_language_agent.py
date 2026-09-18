"""
Cross-Language Plagiarism Agent.

Detects plagiarism when:
- Source is in language A (e.g., English)
- Submission is in language B (e.g., Urdu)

Uses multilingual embeddings to compare across languages.
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
from veritas.tools.language_detect import (
    detect_language, language_name, is_same_language,
)
from veritas.tools.text_utils import split_sentences, clean_text


class CrossLanguageAgent(BaseAgent):
    """
    Detect cross-language plagiarism using multilingual embeddings.
    """

    name = "cross_language_agent"
    description = "Detects plagiarism across different languages."

    def __init__(self, run_id: str = "", verbose: bool = True):
        super().__init__(run_id=run_id, verbose=verbose)
        self.embedder = Embedder.get(verbose=False)

    def run(self, task: str, context: Optional[AgentContext] = None) -> AgentResult:
        start = self._time_start()

        if context is None or not context.document_text:
            return AgentResult.failed(self.name, "No document text in context")

        submitted = clean_text(context.document_text)
        submitted_lang = detect_language(submitted)

        sources = self._collect_sources(context)

        if not sources:
            return AgentResult(
                agent=self.name,
                status="ok",
                findings=[],
                metrics={
                    "submitted_language": submitted_lang,
                    "sources_compared": 0,
                },
                notes="No source documents",
                duration_seconds=self._time_end(start),
            )

        self.log_info(
            f"Submitted language: {language_name(submitted_lang)} "
            f"({submitted_lang})"
        )

        all_findings: list[Finding] = []
        metrics = {
            "submitted_language": submitted_lang,
            "sources_compared": len(sources),
            "cross_language_pairs": 0,
            "matches": 0,
        }

        for src_name, src_text in sources:
            src_lang = detect_language(src_text)

            # Only cross-language pairs
            if src_lang == submitted_lang:
                continue

            metrics["cross_language_pairs"] += 1
            self.log_info(
                f"  Cross-language: {src_lang} → {submitted_lang}"
            )

            findings, src_metrics = self._compare(
                submitted, submitted_lang,
                src_text, src_lang, src_name,
            )
            all_findings.extend(findings)
            metrics["matches"] += src_metrics["matches"]

        all_findings.sort(key=lambda f: f.confidence, reverse=True)

        duration = self._time_end(start)
        self.log_success(
            f"Found {len(all_findings)} cross-language finding(s) "
            f"in {duration:.2f}s"
        )

        return AgentResult(
            agent=self.name,
            status="ok",
            findings=all_findings,
            metrics=metrics,
            notes=f"Cross-language comparison: {metrics['cross_language_pairs']} pairs",
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
        submitted: str,
        sub_lang: str,
        source: str,
        src_lang: str,
        src_name: str,
    ) -> tuple[list[Finding], dict]:
        findings: list[Finding] = []
        metrics = {"matches": 0}

        sub_sentences = [
            s for s in split_sentences(submitted) if len(s.split()) >= 5
        ]
        src_sentences = [
            s for s in split_sentences(source) if len(s.split()) >= 5
        ]

        if not sub_sentences or not src_sentences:
            return findings, metrics

        # Use MULTILINGUAL embedder
        sub_vecs = self.embedder.embed_multi(sub_sentences)
        src_vecs = self.embedder.embed_multi(src_sentences)

        sim_matrix = np.dot(sub_vecs, src_vecs.T)

        evidence_list: list[Evidence] = []

        for i, sub_sent in enumerate(sub_sentences):
            row = sim_matrix[i]
            best_j = int(np.argmax(row))
            best_sim = float(row[best_j])

            if best_sim < THRESHOLDS.cross_language_min:
                continue

            src_sent = src_sentences[best_j]

            evidence_list.append(Evidence(
                submitted_text=sub_sent,
                source_text=src_sent,
                source_document=src_name,
                similarity=best_sim,
                method="multilingual_embedding",
                metadata={
                    "submitted_language": sub_lang,
                    "source_language": src_lang,
                    "submitted_idx": i,
                    "source_idx": best_j,
                },
            ))

        if not evidence_list:
            return findings, metrics

        metrics["matches"] = len(evidence_list)

        evidence_list.sort(key=lambda e: e.similarity, reverse=True)
        top_evidence = evidence_list[:8]
        avg_score = sum(e.similarity for e in top_evidence) / len(top_evidence)

        if avg_score >= 0.85:
            severity = Severity.CRITICAL
        elif avg_score >= 0.78:
            severity = Severity.HIGH
        elif avg_score >= 0.70:
            severity = Severity.MEDIUM
        else:
            severity = Severity.LOW

        sub_lang_name = language_name(sub_lang)
        src_lang_name = language_name(src_lang)

        finding = self.make_finding(
            finding_type=FindingType.CROSS_LANGUAGE,
            title=(
                f"Cross-language similarity: "
                f"{src_lang_name} → {sub_lang_name}"
            ),
            description=(
                f"{len(evidence_list)} sentence(s) in {sub_lang_name} "
                f"have high semantic similarity with {src_lang_name} source."
            ),
            confidence=avg_score,
            evidence=top_evidence,
            explanation=[
                f"Source language: {src_lang_name} ({src_lang})",
                f"Submitted language: {sub_lang_name} ({sub_lang})",
                f"Matches: {len(evidence_list)}",
                f"Average similarity: {avg_score:.2%}",
                f"Method: multilingual embeddings (paraphrase-multilingual-MiniLM)",
                f"Source document: {src_name}",
                "",
                "⚠️ Cross-language plagiarism is often missed by "
                "traditional checkers. Investigate manually.",
            ],
            severity=severity,
            metadata={
                "source": src_name,
                "source_language": src_lang,
                "submitted_language": sub_lang,
                "matches": len(evidence_list),
                "avg_score": round(avg_score, 4),
            },
        )
        findings.append(finding)
        return findings, metrics


__all__ = ["CrossLanguageAgent"]