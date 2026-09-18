"""
Document data models.

A Document is the core input unit. It contains:
- Raw text
- Structured paragraphs (with page numbers)
- Metadata (title, author, source path)
- Extracted references
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import hashlib
import re


@dataclass
class Paragraph:
    """A single paragraph from a document."""
    text: str
    page: int = 0
    index: int = 0
    is_reference: bool = False
    is_heading: bool = False
    word_count: int = 0

    def __post_init__(self):
        if not self.word_count:
            self.word_count = len(self.text.split())

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "page": self.page,
            "index": self.index,
            "is_reference": self.is_reference,
            "is_heading": self.is_heading,
            "word_count": self.word_count,
        }


@dataclass
class Document:
    """
    A parsed document.

    Attributes:
        id: Unique identifier (hash of content)
        path: Original file path
        title: Document title (if detected)
        text: Full extracted text
        paragraphs: List of Paragraph objects
        metadata: Arbitrary metadata
    """
    id: str = ""
    path: str = ""
    title: str = ""
    text: str = ""
    paragraphs: list[Paragraph] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    page_count: int = 0
    word_count: int = 0

    def __post_init__(self):
        if not self.id and self.text:
            self.id = hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]
        if not self.word_count and self.text:
            self.word_count = len(self.text.split())

    @property
    def source_name(self) -> str:
        """Return just the filename."""
        return Path(self.path).name if self.path else "unknown"

    def get_sentences(self) -> list[str]:
        """Split text into sentences (basic heuristic)."""
        text = re.sub(r"\s+", " ", self.text)
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
        return [s.strip() for s in sentences if len(s.strip()) > 20]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "path": self.path,
            "title": self.title,
            "text": self.text[:500] + "..." if len(self.text) > 500 else self.text,
            "paragraph_count": len(self.paragraphs),
            "reference_count": len(self.references),
            "page_count": self.page_count,
            "word_count": self.word_count,
            "source_name": self.source_name,
        }

    @classmethod
    def from_text(cls, text: str, path: str = "", title: str = "") -> "Document":
        """Convenience constructor from raw text."""
        paragraphs = [
            Paragraph(text=p.strip(), index=i)
            for i, p in enumerate(text.split("\n\n"))
            if p.strip()
        ]
        return cls(
            path=path,
            title=title,
            text=text,
            paragraphs=paragraphs,
        )


__all__ = ["Document", "Paragraph"]