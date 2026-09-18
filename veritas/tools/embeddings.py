"""
Embeddings wrapper — sentence-transformers.

Two models:
- English: all-MiniLM-L6-v2 (fast, 384-dim, ~80 MB)
- Multilingual: paraphrase-multilingual-MiniLM-L12-v2 (384-dim, ~470 MB)

Both lazy-loaded. Singleton pattern for reuse.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from veritas.config import (
    EMBEDDING_MODEL_EN,
    EMBEDDING_MODEL_MULTI,
    EMBEDDING_DIM_EN,
    EMBEDDING_DIM_MULTI,
)


class Embedder:
    """Lazy-loading embedder with English + multilingual models."""

    _instance: Optional["Embedder"] = None

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._en_model = None
        self._multi_model = None

    @classmethod
    def get(cls, verbose: bool = True) -> "Embedder":
        """Singleton getter."""
        if cls._instance is None:
            cls._instance = cls(verbose=verbose)
        return cls._instance

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[embedder] {msg}")

    # ---------- English ----------
    def _load_en(self):
        if self._en_model is None:
            from sentence_transformers import SentenceTransformer
            self._log(f"Loading English model: {EMBEDDING_MODEL_EN}")
            self._en_model = SentenceTransformer(EMBEDDING_MODEL_EN)
            self._log("English model loaded")
        return self._en_model

    def embed_en(self, texts: list[str] | str) -> np.ndarray:
        """
        Embed English text(s).

        Returns:
            np.ndarray of shape (N, 384) for list, or (384,) for single string.
        """
        if isinstance(texts, str):
            single = True
            texts = [texts]
        else:
            single = False

        if not texts:
            return np.zeros((0, EMBEDDING_DIM_EN), dtype=np.float32)

        model = self._load_en()
        vecs = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,  # for cosine similarity via dot product
            show_progress_bar=False,
        )
        vecs = vecs.astype(np.float32)
        return vecs[0] if single else vecs

    # ---------- Multilingual ----------
    def _load_multi(self):
        if self._multi_model is None:
            from sentence_transformers import SentenceTransformer
            self._log(f"Loading multilingual model: {EMBEDDING_MODEL_MULTI}")
            self._multi_model = SentenceTransformer(EMBEDDING_MODEL_MULTI)
            self._log("Multilingual model loaded")
        return self._multi_model

    def embed_multi(self, texts: list[str] | str) -> np.ndarray:
        """Embed multilingual text(s)."""
        if isinstance(texts, str):
            single = True
            texts = [texts]
        else:
            single = False

        if not texts:
            return np.zeros((0, EMBEDDING_DIM_MULTI), dtype=np.float32)

        model = self._load_multi()
        vecs = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vecs = vecs.astype(np.float32)
        return vecs[0] if single else vecs

    # ---------- Universal ----------
    def embed(self, texts: list[str] | str, multilingual: bool = False) -> np.ndarray:
        """Pick model based on `multilingual` flag."""
        if multilingual:
            return self.embed_multi(texts)
        return self.embed_en(texts)

    @property
    def dim(self) -> int:
        return EMBEDDING_DIM_EN

    # ---------- Similarity ----------
    @staticmethod
    def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity. Vectors assumed normalized."""
        if a.ndim == 1 and b.ndim == 1:
            return float(np.dot(a, b))
        # Batch
        return float(np.dot(a, b))

    @staticmethod
    def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Cosine similarity between all pairs.

        Args:
            a: (N, D) normalized
            b: (M, D) normalized
        Returns:
            (N, M) similarity matrix
        """
        if a.ndim == 1:
            a = a.reshape(1, -1)
        if b.ndim == 1:
            b = b.reshape(1, -1)
        return np.dot(a, b.T)


# Convenience
def embed(texts, multilingual: bool = False) -> np.ndarray:
    return Embedder.get().embed(texts, multilingual=multilingual)


__all__ = ["Embedder", "embed"]