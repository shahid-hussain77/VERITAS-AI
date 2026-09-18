"""
Orchestrator — runs all agents and fuses results.
"""
from __future__ import annotations
import time
from pathlib import Path
from typing import Optional

from rich.console import Console

from veritas.agents.base import AgentContext
from veritas.agents.document_agent import DocumentAgent
from veritas.agents.exact_copy_agent import ExactCopyAgent
from veritas.agents.semantic_agent import SemanticAgent
from veritas.agents.paraphrase_agent import ParaphraseAgent
from veritas.agents.cross_language_agent import CrossLanguageAgent
from veritas.agents.self_plagiarism_agent import SelfPlagiarismAgent
from veritas.agents.citation_agent import CitationAgent
from veritas.agents.fake_citation_agent import FakeCitationAgent
from veritas.agents.ai_writing_agent import AIWritingAgent
from veritas.agents.ai_tool_agent import AIToolAgent
from veritas.agents.style_agent import StyleAgent
from veritas.agents.batch_agent import BatchAgent
from veritas.agents.collusion_agent import CollusionAgent
from veritas.agents.source_discovery_agent import SourceDiscoveryAgent
from veritas.agents.research_integrity_agent import ResearchIntegrityAgent
from veritas.agents.web_plagiarism_agent import WebPlagiarismAgent
from veritas.agents.evidence_judge_agent import EvidenceJudgeAgent
from veritas.models.document import Document
from veritas.models.finding import AgentResult
from veritas.memory.store import memory


console = Console()


class Orchestrator:
    """Runs agents in sequence and fuses results."""

    def __init__(self, run_id: str = "", verbose: bool = True):
        self.run_id = run_id or f"run-{int(time.time())}"
        self.verbose = verbose

    def run(
        self,
        submitted_path: Optional[str] = None,
        submitted_text: Optional[str] = None,
        source_paths: Optional[list] = None,
        source_texts: Optional[list] = None,
        author_history_paths: Optional[list] = None,
        corpus_paths: Optional[list] = None,
        classroom_paths: Optional[list] = None,
        enable_web: bool = True,
    ) -> dict:
        """
        Full pipeline.

        Args:
            submitted_path: path to submitted doc
            submitted_text: or raw text
            source_paths: docs to compare against
            author_history_paths: author's own previous docs
            corpus_paths: local corpus for source discovery
            classroom_paths: batch comparison
            enable_web: enable online web plagiarism check
        """
        t0 = time.time()

        # ---------- 1. Parse submitted ----------
        if submitted_path:
            console.print(f"[cyan]Parsing submitted: {Path(submitted_path).name}[/cyan]")
            doc_agent = DocumentAgent(run_id=self.run_id, verbose=self.verbose)
            doc_result = doc_agent.run(submitted_path)
            if not doc_result.ok:
                return {"status": "failed", "error": doc_result.error}
            doc = memory.get_document(doc_result.metrics["document_id"])
            if doc is None:
                return {"status": "failed", "error": "Could not reload document"}
        elif submitted_text:
            doc = Document.from_text(submitted_text, path="<text>", title="Submitted")
        else:
            return {"status": "failed", "error": "No submitted document"}

        # ---------- 2. Parse sources ----------
        sources: list[Document] = []
        if source_paths:
            for p in source_paths:
                try:
                    d = DocumentAgent(run_id=self.run_id, verbose=False).run(p)
                    if d.ok:
                        doc_obj = memory.get_document(d.metrics["document_id"])
                        if doc_obj:
                            sources.append(doc_obj)
                except Exception as e:
                    console.print(f"[yellow]Could not parse {p}: {e}[/yellow]")

        source_texts = source_texts or []

        # ---------- 3. Parse history ----------
        history: list[Document] = []
        if author_history_paths:
            for p in author_history_paths:
                try:
                    d = DocumentAgent(run_id=self.run_id, verbose=False).run(p)
                    if d.ok:
                        doc_obj = memory.get_document(d.metrics["document_id"])
                        if doc_obj:
                            history.append(doc_obj)
                except Exception:
                    pass

        # ---------- 4. Parse corpus ----------
        corpus: list[Document] = []
        if corpus_paths:
            for p in corpus_paths:
                try:
                    d = DocumentAgent(run_id=self.run_id, verbose=False).run(p)
                    if d.ok:
                        doc_obj = memory.get_document(d.metrics["document_id"])
                        if doc_obj:
                            corpus.append(doc_obj)
                except Exception:
                    pass

        # ---------- 5. Parse classroom ----------
        classroom: list[Document] = []
        if classroom_paths:
            for p in classroom_paths:
                try:
                    d = DocumentAgent(run_id=self.run_id, verbose=False).run(p)
                    if d.ok:
                        doc_obj = memory.get_document(d.metrics["document_id"])
                        if doc_obj:
                            classroom.append(doc_obj)
                except Exception:
                    pass

        # ---------- 6. Build context ----------
        ctx = AgentContext(
            document_id=doc.id,
            document_path=doc.path,
            document_text=doc.text,
            paragraphs=doc.paragraphs,
            metadata={
                "document_name": doc.source_name,
                "page_count": doc.page_count,
            },
            other_documents=sources,
            extras={
                "source_texts": source_texts,
                "author_history": history,
                "corpus_documents": corpus,
                "classroom_documents": classroom,
                "references": doc.references,
                "agent_results": [],
            },
        )

        # ---------- 7. Run agents ----------
        agents = [
            ("Exact Copy", ExactCopyAgent(run_id=self.run_id, verbose=self.verbose)),
            ("Semantic", SemanticAgent(run_id=self.run_id, verbose=self.verbose)),
            ("Paraphrase", ParaphraseAgent(run_id=self.run_id, verbose=self.verbose)),
            ("Cross-Language", CrossLanguageAgent(run_id=self.run_id, verbose=self.verbose)),
            ("Self-Plagiarism", SelfPlagiarismAgent(run_id=self.run_id, verbose=self.verbose)),
            ("Citation", CitationAgent(run_id=self.run_id, verbose=self.verbose)),
            ("Fake Citation", FakeCitationAgent(run_id=self.run_id, verbose=self.verbose)),
            ("AI-Writing", AIWritingAgent(run_id=self.run_id, verbose=self.verbose)),
            ("AI-Tool Detection", AIToolAgent(run_id=self.run_id, verbose=self.verbose)),
            ("Style", StyleAgent(run_id=self.run_id, verbose=self.verbose)),
            ("Source Discovery", SourceDiscoveryAgent(run_id=self.run_id, verbose=self.verbose)),
            ("Research Integrity", ResearchIntegrityAgent(run_id=self.run_id, verbose=self.verbose)),
        ]

        # Web plagiarism — only if enabled
        if enable_web:
            agents.append(
                ("Web Plagiarism", WebPlagiarismAgent(run_id=self.run_id, verbose=self.verbose))
            )

        # Batch/Collusion only if classroom docs
        if len(classroom) >= 2:
            agents.append(("Batch", BatchAgent(run_id=self.run_id, verbose=self.verbose)))
        if len(classroom) >= 3:
            agents.append(("Collusion", CollusionAgent(run_id=self.run_id, verbose=self.verbose)))

        agent_results: list[AgentResult] = []
        for name, agent in agents:
            console.print(f"[dim]→ {name}...[/dim]")
            try:
                r = agent.run("analyze", ctx)
                agent_results.append(r)
                if r.ok and r.findings:
                    console.print(f"[green]  ✓ {name}: {len(r.findings)} finding(s)[/green]")
                elif r.ok:
                    console.print(f"[dim]  · {name}: no findings[/dim]")
                else:
                    console.print(f"[yellow]  ! {name}: {r.error}[/yellow]")
            except Exception as e:
                console.print(f"[red]  ✗ {name} crashed: {e}[/red]")
                agent_results.append(AgentResult.failed(name, str(e)))

        ctx.extras["agent_results"] = agent_results

        # ---------- 8. Judge ----------
        console.print("[bold cyan]→ Evidence Judge...[/bold cyan]")
        judge = EvidenceJudgeAgent(run_id=self.run_id, verbose=self.verbose)
        judge_result = judge.run("fuse", ctx)
        agent_results.append(judge_result)

        total_duration = time.time() - t0
        report_dict = judge_result.metrics.get("report", {})

        return {
            "status": "ok",
            "document": doc.to_dict(),
            "report": report_dict,
            "agent_results": [r.to_dict() for r in agent_results],
            "duration_seconds": round(total_duration, 3),
        }


__all__ = ["Orchestrator"]