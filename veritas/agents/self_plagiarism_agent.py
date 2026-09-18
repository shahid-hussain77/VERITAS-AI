"""
Self-Plagiarism Agent.

Compares a submission against the same author's previous work.

Use case:
- Teacher uploads Student_A's semester history
- New submission comes in
- Agent finds overlaps with previous submissions

This is different from regular plagiarism:
- SAME author
- DIFFERENT time
- Often unintentional or semi-intentional
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
from veritas.tools.ngram import tokenize, ngram_jaccard, find_matching_spans
from veritas.tools.similarity import MinHash, SimHash
from veritas.tools.text_utils import split_sentences, clean_text


class SelfPlagiarismAgent(BaseAgent):
    """
    Detect self-plagiarism (reuse of own previous work).

    Input:
        context.document_text                  → current submission
        context.extras["author_history"]       → list of Document
                                                (same author's past work)

    Output:
        Findings showing overlap with previous submissions
    """

    name = "self_plagiarism_agent"
    description = "Detects reuse of own previous submissions."

    def __init__(self, run_id: str = "", verbose: bool = True):
        super().__init__(run_id=run_id, verbose=verbose)
        self.embedder = Embedder.get(verbose=False)
        self._minhash = MinHash(num_perm=128, ngram=5)
        self._simhash = SimHash(bits=64, ngram=3)

    def run(self, task: str, context: Optional[AgentContext] = None) -> AgentResult:
        start = self._time_start()

        if context is None or not context.document_text:
            return AgentResult.failed(self.name, "No document text")

        submitted = clean_text(context.document_text)

        # Get author history
        history = context.extras.get("author_history", []) or []

        if not history:
            return AgentResult(
                agent=self.name,
                status="ok",
                findings=[],
                metrics={"history_size": 0},
                notes="No author history provided",
                duration_seconds=self._time_end(start),
            )

        self.log_info(f"Checking against {len(history)} previous submission(s)")

        all_findings: list[Finding] = []
        metrics = {
            "history_size": len(history),
            "documents_overlapped": 0,
            "highest_overlap": 0.0,
        }

        for old_doc in history:
            if isinstance(old_doc, Document):
                findings, doc_metrics = self._compare(submitted, old_doc)
                all_findings.extend(findings)
                if doc_metrics["overlap"] > 0:
                    metrics["documents_overlapped"] += 1
                metrics["highest_overlap"] = max(
                    metrics["highest_overlap"], doc_metrics["overlap"],
                )

        all_findings.sort(key=lambda f: f.confidence, reverse=True)

        duration = self._time_end(start)
        self.log_success(
            f"Found {len(all_findings)} self-plagiarism finding(s) "
            f"in {duration:.2f}s"
        )

        return AgentResult(
            agent=self.name,
            status="ok",
            findings=all_findings,
            metrics=metrics,
            notes=f"Compared against {len(history)} previous submission(s)",
            duration_seconds=duration,
        )

    def _compare(
        self,
        submitted: str,
        old_doc: Document,
    ) -> tuple[list[Finding], dict]:
        findings: list[Finding] = []
        metrics = {"overlap": 0.0}

        old_text = old_doc.text
        if not old_text.strip():
            return findings, metrics

        # Quick signals
        sub_tokens = tokenize(submitted)
        old_tokens = tokenize(old_text)

        if not sub_tokens or not old_tokens:
            return findings, metrics

        # Whole-document similarity
        try:
            minhash_sim = MinHash.jaccard(
                self._minhash.signature(submitted),
                self._minhash.signature(old_text),
            )
        except Exception:
            minhash_sim = 0.0

        try:
            simhash_sim = SimHash.similarity(
                self._simhash.compute(submitted),
                self._simhash.compute(old_text),
            )
        except Exception:
            simhash_sim = 0.0

        jaccard = ngram_jaccard(sub_tokens, old_tokens, n=5)

        # Sentence-level matching
        sub_sentences = [
            s for s in split_sentences(submitted) if len(s.split()) >= 8
        ]
        old_sentences = [
            s for s in split_sentences(old_text) if len(s.split()) >= 8
        ]

        evidence_list: list[Evidence] = []

        if sub_sentences and old_sentences:
            sub_vecs = self.embedder.embed_en(sub_sentences)
            old_vecs = self.embedder.embed_en(old_sentences)
            sim_matrix = np.dot(sub_vecs, old_vecs.T)

            for i, sub_sent in enumerate(sub_sentences):
                row = sim_matrix[i]
                best_j = int(np.argmax(row))
                best_sim = float(row[best_j])

                if best_sim < THRESHOLDS.self_plagiarism_min:
                    continue

                evidence_list.append(Evidence(
                    submitted_text=sub_sent,
                    source_text=old_sentences[best_j],
                    source_document=old_doc.source_name,
                    similarity=best_sim,
                    method="self_plagiarism",
                    metadata={
                        "submitted_idx": i,
                        "previous_idx": best_j,
                        "previous_doc_id": old_doc.id,
                        "previous_doc_name": old_doc.source_name,
                    },
                ))

        # Exact spans (strongest evidence)
        exact_spans = find_matching_spans(submitted, old_text, min_words=10)

        # Combined overlap score
        overall_overlap = max(
            minhash_sim, simhash_sim, jaccard,
            0.0 if not evidence_list else
            sum(e.similarity for e in evidence_list) / len(evidence_list),
        )

        metrics["overlap"] = overall_overlap

        # Only report if meaningful
        if overall_overlap < THRESHOLDS.self_plagiarism_min and not exact_spans:
            return findings, metrics

        # Build evidence
        final_evidence = []

        if exact_spans:
            for span in exact_spans[:3]:
                final_evidence.append(Evidence(
                    submitted_text=span["submitted_text"],
                    source_text=span["source_text"],
                    source_document=old_doc.source_name,
                    similarity=1.0,
                    method="exact_self_copy",
                    metadata={"length_words": span["length_words"]},
                ))

        final_evidence.extend(evidence_list[:5])
        final_evidence.sort(key=lambda e: e.similarity, reverse=True)

        if not final_evidence:
            return findings, metrics

        avg_score = sum(e.similarity for e in final_evidence[:5]) / \
                    min(5, len(final_evidence))

        # Severity
        if avg_score >= 0.85:
            severity = Severity.HIGH
        elif avg_score >= 0.72:
            severity = Severity.MEDIUM
        else:
            severity = Severity.LOW

        explanation = [
            f"Overlap with previous submission: {old_doc.source_name}",
            f"Overall overlap score: {overall_overlap:.2%}",
            f"MinHash: {minhash_sim:.2%}",
            f"SimHash: {simhash_sim:.2%}",
            f"n-gram Jaccard: {jaccard:.2%}",
            f"Sentence matches: {len(evidence_list)}",
            f"Exact phrase matches: {len(exact_spans)}",
            "",
            "⚠️ Self-plagiarism can be unintentional. "
            "Investigate context (was this submitted elsewhere?).",
        ]

        finding = self.make_finding(
            finding_type=FindingType.SELF_PLAGIARISM,
            title=f"Self-overlap with '{old_doc.source_name}'",
            description=(
                f"Current submission overlaps {overall_overlap:.1%} with "
                f"previous submission '{old_doc.source_name}'."
            ),
            confidence=avg_score,
            evidence=final_evidence[:8],
            explanation=explanation,
            severity=severity,
            metadata={
                "previous_doc_id": old_doc.id,
                "previous_doc_name": old_doc.source_name,
                "overall_overlap": round(overall_overlap, 4),
                "minhash": round(minhash_sim, 4),
                "simhash": round(simhash_sim, 4),
                "jaccard": round(jaccard, 4),
                "exact_spans": len(exact_spans),
                "sentence_matches": len(evidence_list),
            },
        )
        findings.append(finding)
        return findings, metrics


__all__ = ["SelfPlagiarismAgent"]