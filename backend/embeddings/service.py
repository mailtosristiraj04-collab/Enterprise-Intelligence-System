"""Reusable sentence-transformer embedding service."""

from __future__ import annotations

import os
from threading import RLock
from typing import Any, Sequence

import numpy as np


class EmbeddingService:
    """Load one sentence-transformer model and reuse it for all embeddings."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dimension: int = 384,
        batch_size: int = 64,
        device: str = "cpu",
        model: Any | None = None,
    ) -> None:
        if dimension <= 0 or batch_size <= 0:
            raise ValueError("dimension and batch_size must be positive.")
        self.model_name = model_name
        self.dimension = dimension
        self.batch_size = batch_size
        self.device = device
        self._model = model
        self._lock = RLock()

    def _get_model(self) -> Any:
        with self._lock:
            if self._model is None:
                try:
                    os.environ.setdefault("USE_TF", "0")
                    from sentence_transformers import SentenceTransformer
                except ImportError as exc:
                    raise RuntimeError("sentence-transformers is required for embeddings.") from exc
                self._model = SentenceTransformer(self.model_name, device=self.device)
            return self._model

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a batch of documents as normalized float32 vectors."""
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Embedding inputs must be non-empty strings.")
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        model = self._get_model()
        with self._lock:
            vectors = model.encode(
                list(texts),
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
                device=self.device,
            )
        return self._validate(vectors, len(texts))

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one query as a normalized float32 vector."""
        vectors = self.embed_documents([text])
        return vectors[0]

    def _validate(self, vectors: Any, expected_count: int) -> np.ndarray:
        result = np.asarray(vectors, dtype=np.float32)
        if result.shape != (expected_count, self.dimension):
            raise ValueError(
                f"Embedding model returned shape {result.shape}; "
                f"expected ({expected_count}, {self.dimension})."
            )
        norms = np.linalg.norm(result, axis=1)
        if not np.all(np.isfinite(result)) or np.any(norms <= 0):
            raise ValueError("Embedding model returned invalid or zero vectors.")
        result = result / norms[:, np.newaxis]
        return np.ascontiguousarray(result)
