"""
Vector memory — persistent per-document vector storage.

Each Document gets its own FAISS index at:
    workspace/vectors/{document_id}.faiss + .meta.json

Global index also maintained for cross-document search.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from veritas.config import VECTORS_DIR, PROCESSING
from veritas.models.document import Document
from veritas.tools.embeddings import Embedder
from veritas.tools.vector_store import VectorStore
from veritas.tools.text_utils import split_paragraphs


class VectorMemory:
    """Manages per-document + global vector indexes."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.embedder = Embedder.get(verbose=verbose)
        self._global_store: Optional[VectorStore] = None
        self._doc_stores: dict[str, VectorStore] = {}

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[vector_memory] {msg}")

    # ---------- Global index ----------
    def get_global(self) -> VectorStore:
        """Lazy-load or create global index."""
        if self._global_store is None:
            path = VECTORS_DIR / "global"
            if path.with_suffix(".faiss").exists():
                try:
                    self._global_store = VectorStore.load(path, verbose=self.verbose)
                    return self._global_store
                except Exception as e:
                    self._log(f"Could not load global: {e}")
            self._global_store = VectorStore(dim=self.embedder.dim, verbose=self.verbose)
        return self._global_store

    def add_to_global(self, vectors: np.ndarray, metadata: list[dict]) -> list[int]:
        store = self.get_global()
        return store.add(vectors, metadata)

    def search_global(self, query: str | np.ndarray, k: int = 5,
                      multilingual: bool = False) -> list[dict]:
        store = self.get_global()
        if isinstance(query, str):
            query = self.embedder.embed(query, multilingual=multilingual)
        return store.search(query, k=k)

    def save_global(self) -> None:
        if self._global_store and self._global_store.size > 0:
            self._global_store.save(VECTORS_DIR / "global")

    # ---------- Per-document ----------
    def _doc_path(self, doc_id: str) -> Path:
        return VECTORS_DIR / doc_id

    def get_doc_store(self, doc_id: str) -> VectorStore:
        """Get or create per-document store."""
        if doc_id in self._doc_stores:
            return self._doc_stores[doc_id]

        path = self._doc_path(doc_id)
        if path.with_suffix(".faiss").exists():
            try:
                store = VectorStore.load(path, verbose=self.verbose)
                self._doc_stores[doc_id] = store
                return store
            except Exception as e:
                self._log(f"Could not load {doc_id}: {e}")

        store = VectorStore(dim=self.embedder.dim, verbose=self.verbose)
        self._doc_stores[doc_id] = store
        return store

    def save_doc_store(self, doc_id: str) -> None:
        store = self._doc_stores.get(doc_id)
        if store and store.size > 0:
            store.save(self._doc_path(doc_id))

    # ---------- Chunking ----------
    @staticmethod
    def chunk_text(
        text: str,
        chunk_size: int = 200,
        overlap: int = 40,
    ) -> list[str]:
        """
        Split text into overlapping chunks by words.

        Returns list of chunk strings.
        """
        words = text.split()
        if len(words) <= chunk_size:
            return [text]

        chunks = []
        step = max(1, chunk_size - overlap)
        for i in range(0, len(words), step):
            chunk_words = words[i : i + chunk_size]
            if len(chunk_words) < 20 and chunks:
                break
            chunks.append(" ".join(chunk_words))
        return chunks

    def index_document(
        self,
        doc: Document,
        add_to_global: bool = True,
    ) -> dict:
        """
        Chunk document, embed chunks, store per-doc + optionally global.

        Returns metrics dict.
        """
        self._log(f"Indexing document {doc.id[:8]} ({doc.word_count} words)")

        # Use paragraphs as primary chunks (fall back to word chunks)
        chunks = []
        chunk_meta = []

        for i, para in enumerate(doc.paragraphs):
            if para.is_reference:
                continue
            if para.word_count < 10:
                continue

            if para.word_count > PROCESSING.chunk_size_words:
                sub_chunks = self.chunk_text(
                    para.text,
                    chunk_size=PROCESSING.chunk_size_words,
                    overlap=PROCESSING.chunk_overlap_words,
                )
                for j, sub in enumerate(sub_chunks):
                    chunks.append(sub)
                    chunk_meta.append({
                        "document_id": doc.id,
                        "paragraph_index": i,
                        "sub_index": j,
                        "is_heading": para.is_heading,
                        "word_count": len(sub.split()),
                    })
            else:
                chunks.append(para.text)
                chunk_meta.append({
                    "document_id": doc.id,
                    "paragraph_index": i,
                    "sub_index": 0,
                    "is_heading": para.is_heading,
                    "word_count": para.word_count,
                })

        if not chunks:
            # Fallback: whole text
            chunks = self.chunk_text(doc.text)
            chunk_meta = [
                {"document_id": doc.id, "paragraph_index": -1, "sub_index": i}
                for i in range(len(chunks))
            ]

        # Embed
        vectors = self.embedder.embed(chunks, multilingual=False)

        # Store per-doc
        store = self.get_doc_store(doc.id)
        store.clear()
        store.add(vectors, chunk_meta)
        self.save_doc_store(doc.id)

        # Add to global
        if add_to_global:
            global_meta = []
            for i, m in enumerate(chunk_meta):
                global_meta.append({
                    **m,
                    "text": chunks[i][:500],  # store preview
                    "source_name": doc.source_name,
                })
            self.add_to_global(vectors, global_meta)

        self._log(f"Indexed {len(chunks)} chunks")
        return {
            "document_id": doc.id,
            "chunks": len(chunks),
            "dim": self.embedder.dim,
        }

    def search_document(
        self,
        doc_id: str,
        query: str,
        k: int = 5,
    ) -> list[dict]:
        """Search within a document."""
        store = self.get_doc_store(doc_id)
        qvec = self.embedder.embed(query)
        return store.search(qvec, k=k)

    def compare_documents(
        self,
        doc_a_id: str,
        doc_b_id: str,
        min_score: float = 0.7,
    ) -> list[dict]:
        """
        Compare two documents chunk-by-chunk.

        For each chunk in A, find most similar chunk in B.

        Returns list of matches sorted by score.
        """
        store_a = self.get_doc_store(doc_a_id)
        store_b = self.get_doc_store(doc_b_id)

        if store_a.size == 0 or store_b.size == 0:
            return []

        # Get all vectors from A (re-embed from metadata? No — FAISS doesn't expose vectors easily)
        # Workaround: re-embed chunk texts from metadata
        # We stored "text" preview in global, but per-doc metadata may not have it.
        # So: use global store as the source of truth for text.

        global_store = self.get_global()
        meta_all = global_store.get_all_metadata()

        # Re-embed texts from A
        a_texts = []
        a_meta = []
        for m in meta_all:
            if m.get("document_id") == doc_a_id:
                t = m.get("text", "")
                if t:
                    a_texts.append(t)
                    a_meta.append(m)

        if not a_texts:
            return []

        a_vecs = self.embedder.embed(a_texts)

        # For each A chunk, search in B's store
        matches = []
        results = store_b.search_batch(a_vecs, k=1, min_score=min_score)

        for i, res_list in enumerate(results):
            if not res_list:
                continue
            r = res_list[0]
            matches.append({
                "submitted": {
                    "text": a_texts[i],
                    "paragraph_index": a_meta[i].get("paragraph_index", -1),
                    "document_id": doc_a_id,
                },
                "source": {
                    "text": r["metadata"].get("text", "") or "(in same document)",
                    "paragraph_index": r["metadata"].get("paragraph_index", -1),
                    "document_id": doc_b_id,
                },
                "score": r["score"],
            })

        matches.sort(key=lambda x: x["score"], reverse=True)
        return matches


# Global singleton
_vector_memory: Optional[VectorMemory] = None


def get_vector_memory(verbose: bool = True) -> VectorMemory:
    global _vector_memory
    if _vector_memory is None:
        _vector_memory = VectorMemory(verbose=verbose)
    return _vector_memory


__all__ = ["VectorMemory", "get_vector_memory"]