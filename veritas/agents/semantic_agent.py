"""
Semantic Plagiarism Agent.

Detects plagiarism where:
- Wording is different
- Meaning is the same

Techniques:
- Sentence embeddings (all-MiniLM-L6-v2)
- Cosine similarity
- Chunk-level comparison

This is VERITAS-AI's KEY differentiator.

Input:
    context.document_text      → submitted text
    context.extras["source_texts"]  → list of source texts

Output:
    Findings with semantic similarity evidence
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from veritas.agents.base import BaseAgent, AgentContext
from veritas.config import THRESHOLDS, PROCESSING
from veritas.models.finding import (
    AgentResult, Evidence, Finding, FindingType, Severity,
)
from veritas.models.document import Document
from veritas.tools.embeddings import Embedder
from veritas.tools.text_utils import split_sentences, clean_text


class SemanticAgent(BaseAgent):
    """
    Detect semantic (meaning-based) plagiarism.

    Uses sentence embeddings to compare:
    - Submitted sentences vs source sentences
    - Chunks vs chunks (for longer texts)
    """

    name = "semantic_agent"
    description = "Detects meaning-level plagiarism using embeddings."

    def __init__(self, run_id: str = "", verbose: bool = True):
        super().__init__(run_id=run_id, verbose=verbose)
        self.embedder = Embedder.get(verbose=False)  # quiet

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
                notes="No source documents provided",
                duration_seconds=self._time_end(start),
            )

        self.log_info(f"Semantic compare: 1 submitted vs {len(sources)} source(s)")

        # Sentence-level comparison
        sub_sentences = [
            s for s in split_sentences(submitted)
            if len(s.split()) >= 6
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

        self.log_info(f"Submitted: {len(sub_sentences)} sentences")

        all_findings: list[Finding] = []
        metrics = {
            "sources_compared": len(sources),
            "submitted_sentences": len(sub_sentences),
            "semantic_matches_total": 0,
            "highest_similarity": 0.0,
        }

        for src_name, src_text in sources:
            findings, src_metrics = self._compare_source(
                sub_sentences, src_text, src_name,
            )
            all_findings.extend(findings)
            metrics["semantic_matches_total"] += src_metrics["matches"]
            metrics["highest_similarity"] = max(
                metrics["highest_similarity"],
                src_metrics["highest"],
            )

        all_findings.sort(key=lambda f: f.confidence, reverse=True)

        duration = self._time_end(start)
        self.log_success(
            f"Found {len(all_findings)} finding(s) in {duration:.2f}s"
        )

        return AgentResult(
            agent=self.name,
            status="ok",
            findings=all_findings,
            metrics=metrics,
            notes=f"Semantic comparison with {len(sources)} source(s)",
            duration_seconds=duration,
        )

    # ---------- Source collection ----------
    def _collect_sources(self, context: AgentContext) -> list[tuple[str, str]]:
        sources: list[tuple[str, str]] = []

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

    # ---------- Compare one source ----------
    def _compare_source(
        self,
        submitted_sentences: list[str],
        source_text: str,
        source_name: str,
    ) -> tuple[list[Finding], dict]:
        findings: list[Finding] = []
        metrics = {"matches": 0, "highest": 0.0}

        source_sentences = [
            s for s in split_sentences(source_text)
            if len(s.split()) >= 6
        ]

        if not source_sentences:
            return findings, metrics

        # Embed both
        sub_vecs = self.embedder.embed_en(submitted_sentences)  # (N, 384)
        src_vecs = self.embedder.embed_en(source_sentences)     # (M, 384)

        # Cosine similarity matrix (both normalized → dot product)
        sim_matrix = np.dot(sub_vecs, src_vecs.T)  # (N, M)

        # For each submitted sentence, find best source match
        matches: list[dict] = []
        for i, sub_sent in enumerate(submitted_sentences):
            row = sim_matrix[i]
            best_j = int(np.argmax(row))
            best_score = float(row[best_j])

            if best_score < THRESHOLDS.semantic_min:
                continue

            # Check that we haven't already matched this source
            matches.append({
                "submitted": sub_sent,
                "source": source_sentences[best_j],
                "score": best_score,
                "submitted_idx": i,
                "source_idx": best_j,
            })

        if not matches:
            return findings, metrics

        # Deduplicate by source index (keep highest score)
        best_per_source: dict[int, dict] = {}
        for m in matches:
            src_idx = m["source_idx"]
            if src_idx not in best_per_source or \
               m["score"] > best_per_source[src_idx]["score"]:
                best_per_source[src_idx] = m

        unique_matches = sorted(
            best_per_source.values(),
            key=lambda x: x["score"],
            reverse=True,
        )

        metrics["matches"] = len(unique_matches)
        metrics["highest"] = unique_matches[0]["score"] if unique_matches else 0.0

        # Build evidence
        evidence = []
        for m in unique_matches[:8]:
            evidence.append(Evidence(
                submitted_text=m["submitted"],
                source_text=m["source"],
                source_document=source_name,
                similarity=m["score"],
                method="semantic_embedding",
                metadata={
                    "submitted_idx": m["submitted_idx"],
                    "source_idx": m["source_idx"],
                },
            ))

        # Compute average of top matches
        top_scores = [m["score"] for m in unique_matches[:5]]
        avg_score = float(np.mean(top_scores))

        # Severity
        if avg_score >= 0.90:
            severity = Severity.CRITICAL
        elif avg_score >= 0.80:
            severity = Severity.HIGH
        elif avg_score >= 0.70:
            severity = Severity.MEDIUM
        else:
            severity = Severity.LOW

        # Build finding
        finding = self.make_finding(
            finding_type=FindingType.SEMANTIC_SIMILARITY,
            title=f"Semantic similarity with {source_name}",
            description=(
                f"{len(unique_matches)} sentence(s) have high meaning "
                f"similarity with source. Top score: {unique_matches[0]['score']:.2%}"
            ),
            confidence=avg_score,
            evidence=evidence,
            explanation=[
                f"Compared {len(submitted_sentences)} submitted sentences "
                f"against {len(source_sentences)} source sentences",
                f"Method: sentence embeddings (all-MiniLM-L6-v2) + cosine similarity",
                f"Average top-5 similarity: {avg_score:.2%}",
                f"Highest single match: {unique_matches[0]['score']:.2%}",
                f"Wording may differ but meaning overlaps significantly",
            ],
            severity=severity,
            metadata={
                "source": source_name,
                "matches": len(unique_matches),
                "avg_score": round(avg_score, 4),
                "top_score": round(unique_matches[0]["score"], 4),
            },
        )
        findings.append(finding)
        return findings, metrics


__all__ = ["SemanticAgent"]