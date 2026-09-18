"""PlagioVeritas data models."""
from veritas.models.document import Document, Paragraph
from veritas.models.finding import Finding, Evidence, AgentResult
from veritas.models.report import Report, IntegrityScore

__all__ = [
    "Document", "Paragraph",
    "Finding", "Evidence", "AgentResult",
    "Report", "IntegrityScore",
]