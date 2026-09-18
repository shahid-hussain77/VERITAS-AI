"""Web Plagiarism Agent — online source matching."""
from __future__ import annotations
from typing import Optional

from veritas.agents.base import BaseAgent, AgentContext
from veritas.models.finding import (
    AgentResult, Evidence, Finding, FindingType, Severity,
)
from veritas.tools.embeddings import Embedder
from veritas.tools.text_utils import clean_text, split_sentences
from veritas.tools.web_search import (
    is_available, web_search, fetch_pages, build_queries,
)
from veritas.tools.similarity import jaro_winkler


class WebPlagiarismAgent(BaseAgent):
    name = "web_plagiarism_agent"
    description = "Searches the web for matching sources."

    def __init__(self, run_id: str = "", verbose: bool = True):
        super().__init__(run_id=run_id, verbose=verbose)
        self.embedder = Embedder.get(verbose=False)

    def run(self, task: str, context: Optional[AgentContext] = None) -> AgentResult:
        start = self._time_start()
        if context is None or not context.document_text:
            return AgentResult.failed(self.name, "No document text")

        if not is_available():
            return AgentResult(
                agent=self.name, status="skipped",
                notes="Online tools not installed (pip install duckduckgo-search trafilatura)",
                duration_seconds=self._time_end(start),
            )

        text = clean_text(context.document_text)
        if len(text.split()) < 30:
            return AgentResult(
                agent=self.name, status="ok", findings=[],
                notes="Text too short",
                duration_seconds=self._time_end(start),
            )

        self.log_info("Building web search queries...")
        queries = build_queries(text, num_queries=3)
        self.log_info(f"Queries: {len(queries)}")

        all_web_results = []
        for q in queries:
            self.log_info(f"  Searching: {q[:60]}...")
            results = web_search(q, max_results=5)
            all_web_results.extend(results)

        if not all_web_results:
            return AgentResult(
                agent=self.name, status="ok", findings=[],
                metrics={"queries": len(queries), "results": 0},
                notes="No web results",
                duration_seconds=self._time_end(start),
            )

        # Dedup by URL
        seen_urls = set()
        unique_results = []
        for r in all_web_results:
            if r.url in seen_urls:
                continue
            seen_urls.add(r.url)
            unique_results.append(r)

        self.log_info(f"Fetching top {min(5, len(unique_results))} pages...")
        unique_results = fetch_pages(unique_results, max_pages=5)

        fetched = [r for r in unique_results if r.fetched]
        if not fetched:
            return AgentResult(
                agent=self.name, status="ok", findings=[],
                metrics={"queries": len(queries), "results": len(unique_results)},
                notes="No pages fetched",
                duration_seconds=self._time_end(start),
            )

        self.log_info(f"Comparing with {len(fetched)} fetched page(s)...")

        # Sentence-level comparison
        sub_sents = [s for s in split_sentences(text) if len(s.split()) >= 8]
        if not sub_sents:
            return AgentResult(
                agent=self.name, status="ok", findings=[],
                notes="No comparable sentences",
                duration_seconds=self._time_end(start),
            )

        findings = []
        for page in fetched:
            page_sents = [
                s for s in split_sentences(page.full_text)
                if len(s.split()) >= 8
            ]
            if not page_sents:
                continue

            # Embed
            sub_vecs = self.embedder.embed_en(sub_sents)
            page_vecs = self.embedder.embed_en(page_sents)

            import numpy as np
            sim_matrix = np.dot(sub_vecs, page_vecs.T)

            evidence_list = []
            for i, sub_sent in enumerate(sub_sents):
                row = sim_matrix[i]
                best_j = int(np.argmax(row))
                score = float(row[best_j])
                if score < 0.78:
                    continue

                evidence_list.append(Evidence(
                    submitted_text=sub_sent,
                    source_text=page_sents[best_j],
                    source_document=page.url,
                    similarity=score,
                    method="web_semantic",
                    metadata={
                        "title": page.title,
                        "domain": page.domain,
                        "score": round(score, 4),
                    },
                ))

            if evidence_list:
                evidence_list.sort(key=lambda e: e.similarity, reverse=True)
                top = evidence_list[:5]
                avg = sum(e.similarity for e in top) / len(top)
                severity = (
                    Severity.HIGH if avg >= 0.88 else
                    Severity.MEDIUM if avg >= 0.82 else
                    Severity.LOW
                )

                findings.append(self.make_finding(
                    finding_type=FindingType.SOURCE_MATCH,
                    title=f"Web match: {page.domain}",
                    description=(
                        f"{len(evidence_list)} sentence(s) match content on "
                        f"'{page.title[:80]}'"
                    ),
                    confidence=min(0.95, avg),
                    evidence=top,
                    explanation=[
                        f"URL: {page.url}",
                        f"Domain: {page.domain}",
                        f"Matches: {len(evidence_list)}",
                        f"Avg similarity: {avg:.2%}",
                        "Sentence-level semantic matching",
                    ],
                    severity=severity,
                    metadata={
                        "url": page.url,
                        "domain": page.domain,
                        "title": page.title,
                        "match_count": len(evidence_list),
                        "avg_similarity": round(avg, 4),
                    },
                ))

        findings.sort(key=lambda f: f.confidence, reverse=True)
        duration = self._time_end(start)
        self.log_success(f"Web: {len(findings)} match(es) ({duration:.1f}s)")

        return AgentResult(
            agent=self.name, status="ok", findings=findings,
            metrics={
                "queries": len(queries),
                "results": len(unique_results),
                "pages_fetched": len(fetched),
                "matches": len(findings),
            },
            notes=f"Web: {len(findings)} match(es)",
            duration_seconds=duration,
        )


__all__ = ["WebPlagiarismAgent"]