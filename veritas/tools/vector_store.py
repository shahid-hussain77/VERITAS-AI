"""
FAISS-backed vector store.

Usage:
    store = VectorStore(dim=384)
    store.add(vectors, metadata=[{"text": "..."}])
    results = store.search(query_vec, k=5)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np


class VectorStore:
    """FAISS index with metadata + persistence."""

    def __init__(self, dim: int = 384, verbose: bool = False):
        self.dim = dim
        self.verbose = verbose
        self._index = None
        self._metadata: list[dict] = []
        self._init_index()

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[vector_store] {msg}")

    def _init_index(self) -> None:
        try:
            import faiss
        except ImportError as e:
            raise RuntimeError(f"faiss not installed: {e}")

        # Inner product index (vectors normalized → cosine sim)
        self._index = faiss.IndexFlatIP(self.dim)

    @property
    def size(self) -> int:
        return self._index.ntotal if self._index else 0

    # ---------- Add ----------
    def add(
        self,
        vectors: np.ndarray,
        metadata: Optional[list[dict]] = None,
    ) -> list[int]:
        """
        Add vectors.

        Args:
            vectors: (N, D) float32
            metadata: list of N dicts
        Returns:
            list of assigned IDs
        """
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        vectors = np.ascontiguousarray(vectors.astype(np.float32))
        n = vectors.shape[0]

        if metadata is None:
            metadata = [{} for _ in range(n)]
        elif len(metadata) != n:
            raise ValueError(
                f"metadata length {len(metadata)} != vectors length {n}"
            )

        start_id = self._index.ntotal
        ids = list(range(start_id, start_id + n))

        self._index.add(vectors)
        self._metadata.extend(metadata)

        self._log(f"Added {n} vectors (total {self.size})")
        return ids

    # ---------- Search ----------
    def search(
        self,
        query: np.ndarray,
        k: int = 5,
        min_score: float = 0.0,
    ) -> list[dict]:
        """
        Search for top-k similar vectors.

        Args:
            query: (D,) or (1, D)
            k: number of results
            min_score: filter results below this similarity
        Returns:
            list of dicts with 'id', 'score', 'metadata'
        """
        if self.size == 0:
            return []

        if query.ndim == 1:
            query = query.reshape(1, -1)
        query = np.ascontiguousarray(query.astype(np.float32))

        k = min(k, self.size)
        scores, ids = self._index.search(query, k)

        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0 or score < min_score:
                continue
            results.append({
                "id": int(idx),
                "score": float(score),
                "metadata": self._metadata[int(idx)],
            })
        return results

    def search_batch(
        self,
        queries: np.ndarray,
        k: int = 5,
        min_score: float = 0.0,
    ) -> list[list[dict]]:
        """Batch search."""
        if self.size == 0:
            return [[] for _ in range(len(queries))]

        if queries.ndim == 1:
            queries = queries.reshape(1, -1)

        k = min(k, self.size)
        scores, ids = self._index.search(queries, k)

        all_results = []
        for row_scores, row_ids in zip(scores, ids):
            results = []
            for score, idx in zip(row_scores, row_ids):
                if idx < 0 or score < min_score:
                    continue
                results.append({
                    "id": int(idx),
                    "score": float(score),
                    "metadata": self._metadata[int(idx)],
                })
            all_results.append(results)
        return all_results

    # ---------- Persistence ----------
    def save(self, path: str | Path) -> None:
        """Save index + metadata."""
        import faiss

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        index_path = path.with_suffix(".faiss")
        meta_path = path.with_suffix(".meta.json")

        faiss.write_index(self._index, str(index_path))
        meta_path.write_text(
            json.dumps({
                "dim": self.dim,
                "size": self.size,
                "metadata": self._metadata,
            }, ensure_ascii=False),
            encoding="utf-8",
        )
        self._log(f"Saved index → {index_path}")

    @classmethod
    def load(cls, path: str | Path, verbose: bool = False) -> "VectorStore":
        """Load index + metadata."""
        import faiss

        path = Path(path)
        index_path = path.with_suffix(".faiss")
        meta_path = path.with_suffix(".meta.json")

        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"No index at {path}")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        store = cls(dim=meta["dim"], verbose=verbose)
        store._index = faiss.read_index(str(index_path))
        store._metadata = meta["metadata"]
        store._log(f"Loaded index ({store.size} vectors)")
        return store

    # ---------- Utilities ----------
    def clear(self) -> None:
        self._init_index()
        self._metadata = []

    def get_metadata(self, idx: int) -> dict:
        if 0 <= idx < len(self._metadata):
            return self._metadata[idx]
        return {}

    def get_all_metadata(self) -> list[dict]:
        return list(self._metadata)


__all__ = ["VectorStore"]