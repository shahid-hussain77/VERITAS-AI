"""
Document Intelligence Agent.

Wraps DocumentParser + persists to memory.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from veritas.agents.base import BaseAgent, AgentContext
from veritas.models.finding import AgentResult
from veritas.tools.document_parser import DocumentParser
from veritas.memory.store import memory


class DocumentAgent(BaseAgent):
    """Parse a document and store it in memory."""

    name = "document_agent"
    description = "Parses PDF/DOCX/TXT/images into structured text."

    def __init__(self, run_id: str = "", verbose: bool = True):
        super().__init__(run_id=run_id, verbose=verbose)
        self.parser = DocumentParser(verbose=verbose)

    def run(self, task: str, context: Optional[AgentContext] = None) -> AgentResult:
        """
        Args:
            task: file path as string
            context: optional AgentContext (unused)
        """
        start = self._time_start()

        try:
            path = Path(task)
        except Exception as e:
            return AgentResult.failed(self.name, f"Invalid path: {e}")

        if not path.exists():
            return AgentResult.failed(self.name, f"File not found: {path}")

        self.log_info(f"Parsing: {path.name}")

        try:
            doc = self.parser.parse(path)
        except Exception as e:
            self.log_error(f"Parse failed: {e}")
            return AgentResult.failed(self.name, f"Parse failed: {e}")

        # Persist
        try:
            memory.save_document(doc)
            self.log_info(f"Saved document {doc.id[:8]} to memory")
        except Exception as e:
            self.log_warn(f"Could not persist to memory: {e}")

        duration = self._time_end(start)
        self.log_success(
            f"Parsed {doc.word_count} words in {duration:.2f}s"
        )

        return AgentResult(
            agent=self.name,
            status="ok",
            findings=[],  # no findings in parsing phase
            metrics={
                "document_id": doc.id,
                "word_count": doc.word_count,
                "paragraphs": len(doc.paragraphs),
                "references": len(doc.references),
                "page_count": doc.page_count,
                "source_type": doc.metadata.get("source_type", "unknown"),
                "ocr_pages": doc.metadata.get("ocr_pages", []),
            },
            notes=f"Parsed {path.name}",
            duration_seconds=duration,
        )


__all__ = ["DocumentAgent"]