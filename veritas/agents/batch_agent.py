"""Batch/Classroom Agent — compares many documents pairwise."""
from __future__ import annotations
import itertools
from typing import Optional

import numpy as np

from veritas.agents.base import BaseAgent, AgentContext
from veritas.config import PROCESSING
from veritas.models.finding import (
    AgentResult, Evidence, Finding, FindingType, Severity,
)
from veritas.models.document import Document
from veritas.tools.embeddings import Embedder
from veritas.tools.text_utils import clean_text, split_sentences


class BatchAgent(BaseAgent):
    """Compare N documents pairwise (classroom mode)."""
    name = "batch_agent"
    description = "Detects similarity across many submissions."

    def __init__(self, run_id: str = "", verbose: bool = True):
        super().__init__(run_id=run_id, verbose=verbose)
        self.embedder = Embedder.get(verbose=False)

    def run(self, task: str, context: Optional[AgentContext] = None) -> AgentResult:
        start = self._time_start()
        if context is None:
            return AgentResult.failed(self.name, "No context")

        docs = context.extras.get("classroom_documents") or []
        docs = [d for d in docs if isinstance(d, Document)]

        if len(docs) < 2:
            return AgentResult(agent=self.name, status="ok", findings=[],
                               metrics={"documents": len(docs)},
                               notes="Need at least 2 documents",
                               duration_seconds=self._time_end(start))

        self.log_info(f"Batch comparing {len(docs)} documents")

        # Compute per-document embeddings (whole doc)
        texts = [clean_text(d.text) for d in docs]
        vecs = self.embedder.embed_en(texts)

        # Similarity matrix
        sim_matrix = np.dot(vecs, vecs.T)

        # Collect pairwise matches above threshold
        threshold = 0.72
        pairs = []
        n = len(docs)
        for i, j in itertools.combinations(range(n), 2):
            s = float(sim_matrix[i, j])
            if s >= threshold:
                pairs.append((i, j, s))

        pairs.sort(key=lambda x: x[2], reverse=True)

        metrics = {
            "documents": n,
            "pairs_compared": n * (n - 1) // 2,
            "high_similarity_pairs": len(pairs),
            "threshold": threshold,
        }

        findings = []

        if pairs:
            # Group pairs into clusters (union-find)
            parent = list(range(n))

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(x, y):
                rx, ry = find(x), find(y)
                if rx != ry:
                    parent[rx] = ry

            for i, j, _ in pairs:
                union(i, j)

            clusters = {}
            for i in range(n):
                root = find(i)
                clusters.setdefault(root, []).append(i)

            # Only report clusters with 2+ docs
            real_clusters = {k: v for k, v in clusters.items() if len(v) >= 2}

            evidence_list = []
            for i, j, s in pairs[:10]:
                evidence_list.append(Evidence(
                    submitted_text=f"{docs[i].source_name}",
                    source_text=f"{docs[j].source_name}",
                    similarity=s,
                    method="batch_pairwise",
                    metadata={"doc_a": docs[i].source_name,
                              "doc_b": docs[j].source_name,
                              "similarity": round(s, 4)},
                ))

            if len(real_clusters) > 0 or pairs:
                severity = Severity.HIGH if pairs[0][2] >= 0.85 else Severity.MEDIUM

                findings.append(self.make_finding(
                    finding_type=FindingType.COLLUSION,
                    title=f"{len(pairs)} high-similarity pair(s) across {n} submissions",
                    description=(
                        f"Found {len(pairs)} document pairs with "
                        f"similarity ≥ {threshold}. "
                        f"Clusters: {len(real_clusters)}"
                    ),
                    confidence=min(0.9, 0.5 + 0.02 * len(pairs)),
                    evidence=evidence_list,
                    explanation=[
                        f"Documents analyzed: {n}",
                        f"Pairs compared: {n*(n-1)//2}",
                        f"Above threshold ({threshold}): {len(pairs)}",
                        f"Clusters: {len(real_clusters)}",
                        "",
                        "Top matches:",
                    ] + [
                        f"  • {docs[i].source_name} ↔ {docs[j].source_name} "
                        f"({s:.2%})"
                        for i, j, s in pairs[:5]
                    ],
                    severity=severity,
                    metadata={"pairs": len(pairs),
                              "clusters": [
                                  [docs[i].source_name for i in cluster]
                                  for cluster in real_clusters.values()
                              ]},
                ))

        duration = self._time_end(start)
        self.log_success(f"Batch: {len(pairs)} pairs ({duration:.2f}s)")

        return AgentResult(agent=self.name, status="ok", findings=findings,
                           metrics=metrics,
                           notes=f"{len(pairs)} high-sim pairs",
                           duration_seconds=duration)


__all__ = ["BatchAgent"]