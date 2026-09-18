"""Collusion Detection Agent — graph-based student similarity."""
from __future__ import annotations
import itertools
from typing import Optional

import numpy as np

from veritas.agents.base import BaseAgent, AgentContext
from veritas.models.finding import (
    AgentResult, Evidence, Finding, FindingType, Severity,
)
from veritas.models.document import Document
from veritas.tools.embeddings import Embedder
from veritas.tools.text_utils import clean_text, split_sentences


class CollusionAgent(BaseAgent):
    """Detect collusion patterns via document graph."""
    name = "collusion_agent"
    description = "Detects collusion clusters in submissions."

    def __init__(self, run_id: str = "", verbose: bool = True):
        super().__init__(run_id=run_id, verbose=verbose)
        self.embedder = Embedder.get(verbose=False)

    def run(self, task: str, context: Optional[AgentContext] = None) -> AgentResult:
        start = self._time_start()
        if context is None:
            return AgentResult.failed(self.name, "No context")

        docs = context.extras.get("classroom_documents") or []
        docs = [d for d in docs if isinstance(d, Document)]
        if len(docs) < 3:
            return AgentResult(agent=self.name, status="ok", findings=[],
                               metrics={"documents": len(docs)},
                               notes="Need 3+ docs for collusion graph",
                               duration_seconds=self._time_end(start))

        self.log_info(f"Collusion analysis: {len(docs)} docs")

        # Sentence-level embeddings
        all_sentences = []
        sentence_to_doc = []
        for idx, d in enumerate(docs):
            sents = [s for s in split_sentences(clean_text(d.text))
                     if len(s.split()) >= 8]
            for s in sents:
                all_sentences.append(s)
                sentence_to_doc.append(idx)

        if not all_sentences:
            return AgentResult(agent=self.name, status="ok", findings=[],
                               metrics={"sentences": 0},
                               notes="No sentences",
                               duration_seconds=self._time_end(start))

        vecs = self.embedder.embed_en(all_sentences)
        sim = np.dot(vecs, vecs.T)

        # Build edge list: doc pairs + shared sentence count
        edge_scores = {}
        threshold = 0.80
        n = len(docs)

        for i in range(len(all_sentences)):
            for j in range(i + 1, len(all_sentences)):
                di = sentence_to_doc[i]
                dj = sentence_to_doc[j]
                if di == dj:
                    continue
                s = float(sim[i, j])
                if s < threshold:
                    continue
                key = tuple(sorted((di, dj)))
                edge_scores.setdefault(key, []).append(s)

        if not edge_scores:
            return AgentResult(agent=self.name, status="ok", findings=[],
                               metrics={"edges": 0, "documents": n},
                               notes="No collusion edges found",
                               duration_seconds=self._time_end(start))

        # Aggregate per doc-pair
        pair_stats = []
        for (di, dj), scores in edge_scores.items():
            pair_stats.append({
                "doc_a": docs[di].source_name,
                "doc_b": docs[dj].source_name,
                "shared_sentences": len(scores),
                "avg_similarity": sum(scores) / len(scores),
                "i": di, "j": dj,
            })

        pair_stats.sort(key=lambda x: x["shared_sentences"], reverse=True)

        # Cluster via graph (networkx)
        import networkx as nx
        G = nx.Graph()
        for k in range(n):
            G.add_node(k, name=docs[k].source_name)
        for p in pair_stats:
            if p["shared_sentences"] >= 2:
                G.add_edge(p["i"], p["j"],
                           weight=p["shared_sentences"],
                           similarity=p["avg_similarity"])

        components = list(nx.connected_components(G))
        clusters = [c for c in components if len(c) >= 2]

        metrics = {
            "documents": n,
            "sentences": len(all_sentences),
            "edges": len(pair_stats),
            "clusters": len(clusters),
            "threshold": threshold,
        }

        findings = []
        if clusters:
            evidence_list = []
            for p in pair_stats[:10]:
                evidence_list.append(Evidence(
                    submitted_text=p["doc_a"],
                    source_text=p["doc_b"],
                    similarity=p["avg_similarity"],
                    method="collusion_graph",
                    metadata={
                        "shared_sentences": p["shared_sentences"],
                        "avg_similarity": round(p["avg_similarity"], 4),
                    },
                ))

            cluster_strs = [
                [docs[i].source_name for i in c] for c in clusters
            ]

            severity = Severity.HIGH if any(len(c) >= 3 for c in clusters) else Severity.MEDIUM

            findings.append(self.make_finding(
                finding_type=FindingType.COLLUSION,
                title=f"Collusion clusters detected: {len(clusters)}",
                description=(
                    f"Found {len(clusters)} cluster(s) of documents sharing "
                    f"multiple similar sentences (≥{threshold})."
                ),
                confidence=min(0.92, 0.6 + 0.05 * len(clusters)),
                evidence=evidence_list,
                explanation=[
                    f"Documents analyzed: {n}",
                    f"Similar sentence pairs: {len(pair_stats)}",
                    f"Clusters (2+ docs): {len(clusters)}",
                    "",
                    "Clusters:",
                ] + [
                    f"  Cluster {i+1}: {', '.join(c)}"
                    for i, c in enumerate(cluster_strs)
                ],
                severity=severity,
                metadata={"clusters": cluster_strs,
                          "pair_stats": pair_stats[:10]},
            ))

        duration = self._time_end(start)
        self.log_success(f"Collusion: {len(clusters)} clusters ({duration:.2f}s)")

        return AgentResult(agent=self.name, status="ok", findings=findings,
                           metrics=metrics,
                           notes=f"{len(clusters)} clusters",
                           duration_seconds=duration)


__all__ = ["CollusionAgent"]