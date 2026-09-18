"""
BaseAgent — foundation for all VERITAS agents.

Design principles:
- Every agent is independent
- Every agent returns AgentResult
- Agents never crash the pipeline
- LLM access is optional and lazy
- Logging is built-in
"""
from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from rich.console import Console

from veritas.config import LLM_MODEL, LLM_TEMPERATURE, LLM_TIMEOUT, LOG_LEVEL
from veritas.models.finding import AgentResult, Finding, Severity, FindingType


console = Console()


@dataclass
class AgentContext:
    """Shared context passed to agents."""
    document_id: str = ""
    document_path: str = ""
    document_text: str = ""
    paragraphs: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    other_documents: list = field(default_factory=list)
    extras: dict = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Abstract base class for all VERITAS agents.

    Subclasses must implement:
        name: unique agent name
        run(task, context) -> AgentResult
    """

    name: str = "base"
    description: str = ""
    temperature: float = LLM_TEMPERATURE

    def __init__(self, run_id: str = "", verbose: bool = True):
        self.run_id = run_id or f"{self.name}-{int(time.time())}"
        self.verbose = verbose
        self._llm = None
        self._embedder = None
        self.log_prefix = f"[{self.name}]"

    # ---------- Logging ----------
    def log_info(self, msg: str) -> None:
        if self.verbose:
            console.print(f"[cyan]{self.log_prefix}[/cyan] {msg}")

    def log_warn(self, msg: str) -> None:
        if self.verbose:
            console.print(f"[yellow]{self.log_prefix} WARN[/yellow] {msg}")

    def log_error(self, msg: str) -> None:
        if self.verbose:
            console.print(f"[red]{self.log_prefix} ERROR[/red] {msg}")

    def log_success(self, msg: str) -> None:
        if self.verbose:
            console.print(f"[green]{self.log_prefix}[/green] {msg}")

    # ---------- LLM (lazy) ----------
    def get_llm(self):
        """Lazy-load Ollama LLM client."""
        if self._llm is None:
            try:
                import ollama
                self._llm = ollama.Client()
                self.log_info(f"LLM loaded: {LLM_MODEL}")
            except Exception as e:
                self.log_warn(f"LLM unavailable: {e}")
                self._llm = False
        return self._llm if self._llm is not False else None

    def ask_llm(self, prompt: str, system: str = "") -> Optional[str]:
        """Ask the local LLM. Returns None on failure."""
        client = self.get_llm()
        if not client:
            return None
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            response = client.chat(
                model=LLM_MODEL,
                messages=messages,
                options={"temperature": self.temperature},
            )
            return response["message"]["content"]
        except Exception as e:
            self.log_warn(f"LLM call failed: {e}")
            return None

    # ---------- Embeddings (lazy) ----------
    def get_embedder(self, multilingual: bool = False):
        """Lazy-load sentence-transformers embedder."""
        key = "multi" if multilingual else "en"
        if not hasattr(self, "_embedders"):
            self._embedders = {}

        if key not in self._embedders:
            from veritas.config import (
                EMBEDDING_MODEL_EN, EMBEDDING_MODEL_MULTI,
            )
            from sentence_transformers import SentenceTransformer
            model_name = (
                EMBEDDING_MODEL_MULTI if multilingual else EMBEDDING_MODEL_EN
            )
            self.log_info(f"Loading embedder: {model_name}")
            self._embedders[key] = SentenceTransformer(model_name)
        return self._embedders[key]

    # ---------- Abstract ----------
    @abstractmethod
    def run(self, task: str, context: Optional[AgentContext] = None) -> AgentResult:
        """Run the agent. Must be implemented by subclasses."""
        raise NotImplementedError

    # ---------- Helpers ----------
    def _time_start(self) -> float:
        return time.time()

    def _time_end(self, start: float) -> float:
        return time.time() - start

    def make_finding(
        self,
        finding_type: FindingType,
        title: str,
        description: str,
        confidence: float,
        evidence: list = None,
        explanation: list = None,
        severity: Severity = Severity.MEDIUM,
        metadata: dict = None,
    ) -> Finding:
        """Convenience factory for findings."""
        return Finding(
            type=finding_type,
            title=title,
            description=description,
            confidence=confidence,
            evidence=evidence or [],
            explanation=explanation or [],
            severity=severity,
            agent=self.name,
            metadata=metadata or {},
        )


__all__ = ["BaseAgent", "AgentContext"]