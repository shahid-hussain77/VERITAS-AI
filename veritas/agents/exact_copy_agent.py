"""
Exact Copy Agent.

Detects:
- Exact phrase matches (n-gram)
- Near-duplicate sentences (Levenshtein, Jaro-Winkler)
- High MinHash Jaccard (approximate duplicate)
- High SimHash similarity (near-duplicate)
- TF-IDF cosine similarity

Input: Document A (submitted) + Document B or list (sources)
Output: Findings with evidence
"""
from __future__ import annotations

from typing import Optional

from veritas.agents.base import BaseAgent, AgentContext
from veritas.config import THRESHOLDS
from veritas.models.finding import (
    AgentResult, Evidence, Finding, FindingType, Severity,
)
from veritas.models.document import Document
from veritas.tools.ngram import (
    tokenize, ngram_jaccard, find_matching_spans,
)
from veritas.tools.similarity import (
    MinHash, SimHash, tfidf_cosine,
    levenshtein_ratio, jaro_winkler, best_sentence_match,
)
from veritas.tools.text_utils import split_sentences


class ExactCopyAgent(BaseAgent):
    """
    Detect exact and near-exact copying.

    Input:
        task: a label (unused)
        context: AgentContext with:
            - document_text: submitted document
            - extras["source_documents"]: list[Document] to compare against
            - extras["source_texts"]: list[str] alternate
    """

    name = "exact_copy_agent"
    description = "Detects exact and near-exact phrase/sentence matches."

    def __init__(self, run_id: str = "", verbose: bool = True):
        super().__init__(run_id=run_id, verbose=verbose)
        self._minhash = MinHash(num_perm=128, ngram=5)
        self._simhash = SimHash(bits=64, ngram=3)

    def run(self, task: str, context: Optional[AgentContext] = None) -> AgentResult:
        start = self._time_start()

        if context is None or not context.document_text:
            return AgentResult.failed(self.name, "No document text in context")

        submitted = context.document_text
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

        self.log_info(
            f"Comparing submitted doc against {len(sources)} source(s)"
        )

        all_findings: list[Finding] = []
        metrics = {
            "sources_compared": len(sources),
            "exact_spans_total": 0,
            "sentence_matches_total": 0,
            "high_similarity_sources": 0,
        }

        for src_name, src_text in sources:
            findings, src_metrics = self._compare_one(submitted, src_text, src_name)
            all_findings.extend(findings)
            metrics["exact_spans_total"] += src_metrics["exact_spans"]
            metrics["sentence_matches_total"] += src_metrics["sentence_matches"]
            if src_metrics["overall_similarity"] >= THRESHOLDS.exact_match_min:
                metrics["high_similarity_sources"] += 1

        # Sort by severity + confidence
        all_findings.sort(key=lambda f: (-f.confidence, f.title))

        duration = self._time_end(start)
        self.log_success(
            f"Found {len(all_findings)} finding(s) in {duration:.2f}s"
        )

        return AgentResult(
            agent=self.name,
            status="ok",
            findings=all_findings,
            metrics=metrics,
            notes=f"Compared {len(sources)} source(s)",
            duration_seconds=duration,
        )

    # ---------- Collect sources ----------
    def _collect_sources(self, context: AgentContext) -> list[tuple[str, str]]:
        """Extract (name, text) tuples from context."""
        sources: list[tuple[str, str]] = []

        docs = context.extras.get("source_documents", []) or []
        for d in docs:
            if isinstance(d, Document):
                sources.append((d.source_name, d.text))
            elif isinstance(d, str):
                sources.append(("source", d))

        texts = context.extras.get("source_texts", []) or []
        for t in texts:
            if isinstance(t, str) and t.strip():
                sources.append(("source", t))

        # Also context.other_documents
        for d in context.other_documents or []:
            if isinstance(d, Document):
                sources.append((d.source_name, d.text))

        return sources

    # ---------- Compare one source ----------
    def _compare_one(
        self,
        submitted: str,
        source: str,
        source_name: str,
    ) -> tuple[list[Finding], dict]:
        findings: list[Finding] = []
        metrics = {
            "exact_spans": 0,
            "sentence_matches": 0,
            "overall_similarity": 0.0,
        }

        # 1. Fast whole-document signals
        sub_tokens = tokenize(submitted)
        src_tokens = tokenize(source)

        if not sub_tokens or not src_tokens:
            return findings, metrics

        # MinHash
        try:
            sig_sub = self._minhash.signature(submitted)
            sig_src = self._minhash.signature(source)
            minhash_sim = MinHash.jaccard(sig_sub, sig_src)
        except Exception:
            minhash_sim = 0.0

        # SimHash
        try:
            h_sub = self._simhash.compute(submitted)
            h_src = self._simhash.compute(source)
            simhash_sim = SimHash.similarity(h_sub, h_src)
        except Exception:
            simhash_sim = 0.0

        # n-gram Jaccard
        jaccard = ngram_jaccard(sub_tokens, src_tokens, n=THRESHOLDS.ngram_size)

        # TF-IDF
        try:
            tfidf = float(tfidf_cosine([submitted], [source])[0, 0])
        except Exception:
            tfidf = 0.0

        overall = max(minhash_sim, simhash_sim, jaccard, tfidf)
        metrics["overall_similarity"] = overall

        # 2. Exact span matching (consecutive token runs)
        spans = find_matching_spans(
            submitted, source, min_words=THRESHOLDS.min_match_words,
        )
        metrics["exact_spans"] = len(spans)

        if spans:
            top_spans = spans[:5]
            evidence_list = []
            for s in top_spans:
                evidence_list.append(Evidence(
                    submitted_text=s["submitted_text"],
                    source_text=s["source_text"],
                    source_document=source_name,
                    similarity=1.0,
                    method="ngram_exact",
                    metadata={
                        "start_a": s["start_a"],
                        "start_b": s["start_b"],
                        "length_words": s["length_words"],
                    },
                ))

            severity = self._severity_from_words(
                max(s["length_words"] for s in top_spans)
            )

            findings.append(self.make_finding(
                finding_type=FindingType.EXACT_COPY,
                title=f"Exact phrase copy from {source_name}",
                description=(
                    f"{len(spans)} exact phrase match(es) found. "
                    f"Longest: {max(s['length_words'] for s in top_spans)} "
                    f"consecutive words."
                ),
                confidence=min(0.99, 0.7 + 0.03 * len(top_spans)),
                evidence=evidence_list,
                explanation=[
                    f"Found {len(spans)} matching phrase(s) of "
                    f"≥{THRESHOLDS.min_match_words} consecutive words",
                    f"Longest match: {max(s['length_words'] for s in top_spans)} words",
                    f"Matched source: {source_name}",
                ],
                severity=severity,
                metadata={
                    "source": source_name,
                    "spans": len(spans),
                    "minhash": round(minhash_sim, 4),
                    "simhash": round(simhash_sim, 4),
                    "jaccard": round(jaccard, 4),
                    "tfidf": round(tfidf, 4),
                },
            ))

        # 3. Sentence-level matching
        sub_sentences = split_sentences(submitted)
        src_sentences = split_sentences(source)

        if sub_sentences and src_sentences:
            sentence_evidence = []
            seen_idx = set()
            for sub_sent in sub_sentences:
                if len(sub_sent.split()) < 6:
                    continue
                match = best_sentence_match(
                    sub_sent, src_sentences, threshold=0.75,
                )
                if not match:
                    continue
                idx, score = match
                if idx in seen_idx:
                    continue
                seen_idx.add(idx)

                # Verify with Levenshtein
                lev = levenshtein_ratio(sub_sent, src_sentences[idx])
                jw = jaro_winkler(sub_sent, src_sentences[idx])

                sentence_evidence.append(Evidence(
                    submitted_text=sub_sent,
                    source_text=src_sentences[idx],
                    source_document=source_name,
                    similarity=max(score, lev, jw),
                    method="sentence_match",
                    metadata={
                        "levenshtein": round(lev, 4),
                        "jaro_winkler": round(jw, 4),
                        "match_score": round(score, 4),
                    },
                ))

            if sentence_evidence:
                metrics["sentence_matches"] = len(sentence_evidence)
                # Sort by similarity
                sentence_evidence.sort(key=lambda e: e.similarity, reverse=True)
                top = sentence_evidence[:5]

                avg_sim = sum(e.similarity for e in top) / len(top)
                severity = (
                    Severity.HIGH if avg_sim >= 0.9
                    else Severity.MEDIUM if avg_sim >= 0.75
                    else Severity.LOW
                )

                findings.append(self.make_finding(
                    finding_type=FindingType.EXACT_COPY,
                    title=f"Near-duplicate sentences from {source_name}",
                    description=(
                        f"{len(sentence_evidence)} sentence(s) closely match "
                        f"source '{source_name}'."
                    ),
                    confidence=min(0.95, avg_sim),
                    evidence=top,
                    explanation=[
                        f"Average sentence similarity: {avg_sim:.2%}",
                        f"Method: token overlap + Jaro-Winkler + Levenshtein",
                        f"Source: {source_name}",
                    ],
                    severity=severity,
                    metadata={
                        "source": source_name,
                        "sentence_count": len(sentence_evidence),
                    },
                ))

        return findings, metrics

    @staticmethod
    def _severity_from_words(word_count: int) -> Severity:
        """Severity based on matched phrase length."""
        if word_count >= 25:
            return Severity.CRITICAL
        if word_count >= 15:
            return Severity.HIGH
        if word_count >= 10:
            return Severity.MEDIUM
        return Severity.LOW


__all__ = ["ExactCopyAgent"]