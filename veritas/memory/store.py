"""
MemoryStore — SQLite-backed storage for VERITAS-AI.

Stores:
- Documents (id, path, title, text, metadata)
- Findings (per document)
- Agent runs (metrics, timings)
- Embeddings (via FAISS, separate files)
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from veritas.config import DB_PATH
from veritas.models.document import Document
from veritas.models.finding import Finding, AgentResult


SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    path TEXT,
    title TEXT,
    text TEXT,
    page_count INTEGER DEFAULT 0,
    word_count INTEGER DEFAULT 0,
    metadata TEXT DEFAULT '{}',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS findings (
    id TEXT PRIMARY KEY,
    document_id TEXT,
    type TEXT,
    severity TEXT,
    title TEXT,
    description TEXT,
    confidence REAL,
    agent TEXT,
    evidence TEXT DEFAULT '[]',
    explanation TEXT DEFAULT '[]',
    metadata TEXT DEFAULT '{}',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES documents(id)
);

CREATE TABLE IF NOT EXISTS agent_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT,
    agent TEXT,
    status TEXT,
    findings_count INTEGER,
    metrics TEXT DEFAULT '{}',
    notes TEXT,
    error TEXT,
    duration_seconds REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES documents(id)
);

CREATE INDEX IF NOT EXISTS idx_findings_doc ON findings(document_id);
CREATE INDEX IF NOT EXISTS idx_findings_type ON findings(type);
CREATE INDEX IF NOT EXISTS idx_runs_doc ON agent_runs(document_id);
"""


class MemoryStore:
    """SQLite-backed persistent store."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path or DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ---------- Documents ----------
    def save_document(self, doc: Document) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO documents
                (id, path, title, text, page_count, word_count, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    doc.id, doc.path, doc.title, doc.text,
                    doc.page_count, doc.word_count,
                    json.dumps(doc.metadata),
                ),
            )

    def get_document(self, doc_id: str) -> Optional[Document]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE id = ?", (doc_id,)
            ).fetchone()
        if not row:
            return None
        return Document(
            id=row["id"],
            path=row["path"],
            title=row["title"],
            text=row["text"],
            page_count=row["page_count"],
            word_count=row["word_count"],
            metadata=json.loads(row["metadata"] or "{}"),
        )

    def list_documents(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, path, title, word_count, created_at "
                "FROM documents ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_document(self, doc_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM findings WHERE document_id = ?", (doc_id,))
            conn.execute("DELETE FROM agent_runs WHERE document_id = ?", (doc_id,))
            conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

    # ---------- Findings ----------
    def save_finding(self, doc_id: str, finding: Finding) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO findings
                (id, document_id, type, severity, title, description,
                 confidence, agent, evidence, explanation, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    finding.id, doc_id, finding.type.value,
                    finding.severity.value, finding.title, finding.description,
                    finding.confidence, finding.agent,
                    json.dumps([e.to_dict() for e in finding.evidence]),
                    json.dumps(finding.explanation),
                    json.dumps(finding.metadata),
                ),
            )

    def get_findings(self, doc_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM findings WHERE document_id = ? "
                "ORDER BY confidence DESC",
                (doc_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ---------- Agent runs ----------
    def save_agent_run(self, doc_id: str, result: AgentResult) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO agent_runs
                (document_id, agent, status, findings_count,
                 metrics, notes, error, duration_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    doc_id, result.agent, result.status, len(result.findings),
                    json.dumps(result.metrics), result.notes, result.error,
                    result.duration_seconds,
                ),
            )

    def get_agent_runs(self, doc_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM agent_runs WHERE document_id = ? "
                "ORDER BY created_at DESC",
                (doc_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ---------- Stats ----------
    def stats(self) -> dict:
        with self._conn() as conn:
            docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            finds = conn.execute("SELECT COUNT(*) FROM findings").fetchone()[0]
            runs = conn.execute("SELECT COUNT(*) FROM agent_runs").fetchone()[0]
        return {"documents": docs, "findings": finds, "agent_runs": runs}

    def reset(self) -> None:
        with self._conn() as conn:
            conn.execute("DROP TABLE IF EXISTS findings")
            conn.execute("DROP TABLE IF EXISTS agent_runs")
            conn.execute("DROP TABLE IF EXISTS documents")
        self._init_db()


# Global singleton
memory = MemoryStore()


__all__ = ["MemoryStore", "memory"]