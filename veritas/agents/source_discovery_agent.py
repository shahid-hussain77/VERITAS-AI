"""Source Discovery Agent — find similar docs from local database."""
from __future__ import annotations
from typing import Optional

import numpy as np

from veritas.agents.base import BaseAgent, AgentContext
from veritas.models.finding import (
    AgentResult, Evidence, Finding, FindingType, Severity,
)
from veritas.models.document import Document
from veritas.tools.embeddings import Embedder
from veritas.tools.text_utils import clean_text


class SourceDiscoveryAgent(BaseAgent):
    name = "source_discovery_agent"
    description = "Discovers similar sources from local corpus."

    def __init__(self, run_id: str = "", verbose: bool = True):
        super().__init__(run_id=run_id, verbose=verbose)
        self.embedder = Embedder.get(verbose=False)

    def run(self, task: str, context: Optional[AgentContext] = None) -> AgentResult:
        start = self._time_start()
        if context is None or not context.document_text:
            return AgentResult.failed(self.name, "No document text")

        text = clean_text(context.document_text)
        corpus = context.extras.get("corpus_documents") or []
        corpus = [d for d in corpus if isinstance(d, Document)]

        if not corpus:
            return AgentResult(agent=self.name, status="ok", findings=[],
                               metrics={"corpus_size": 0},
                               notes="No corpus provided",
                               duration_seconds=self._time_end(start))

        self.log_info(f"Source discovery: query vs {len(corpus)} corpus docs")

        query_vec = self.embedder.embed_en(text)
        corpus_texts = [clean_text(d.text) for d in corpus]
        corpus_vecs = self.embedder.embed_en(corpus_texts)

        sims = np.dot(corpus_vecs, query_vec)
        order = np.argsort(-sims)

        threshold = 0.60
        top = [
            (corpus[i], float(sims[i]))
            for i in order[:10] if sims[i] >= threshold
        ]

        metrics = {
            "corpus_size": len(corpus),
            "matches": len(top),
            "threshold": threshold,
            "top_similarity": round(float(sims[order[0]]), 4) if len(order) else 0.0,
        }

        findings = []
        if top:
            evidence_list = [
                Evidence(
                    submitted_text=f"Query matches: {d.source_name}",
                    source_text=f"Similarity: {s:.4f}",
                    source_document=d.source_name,
                    similarity=s,
                    method="source_discovery",
                    metadata={"source_id": d.id, "similarity": round(s, 4)},
                )
                for d, s in top[:5]
            ]

            severity = Severity.HIGH if top[0][1] >= 0.85 else Severity.MEDIUM

            findings.append(self.make_finding(
                finding_type=FindingType.SOURCE_MATCH,
                title=f"Potential sources: {len(top)}",
                description=f"Found {len(top)} similar source(s) in local corpus.",
                confidence=min(0.9, top[0][1]),
                evidence=evidence_list,
                explanation=[
                    f"Corpus size: {len(corpus)}",
                    f"Threshold: {threshold}",
                    "",
                    "Top matches:",
                ] + [
                    f"  {i+1}. {d.source_name} → {s:.2%}"
                    for i, (d, s) in enumerate(top[:5])
                ],
                severity=severity,
                metadata={"matches": [
                    {"source": d.source_name, "similarity": round(s, 4)}
                    for d, s in top
                ]},
            ))

        duration = self._time_end(start)
        self.log_success(f"Discovered {len(top)} source(s) ({duration:.2f}s)")

        return AgentResult(agent=self.name, status="ok", findings=findings,
                           metrics=metrics,
                           notes=f"{len(top)} sources",
                           duration_seconds=duration)


__all__ = ["SourceDiscoveryAgent"]