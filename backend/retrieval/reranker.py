"""Reusable cross-encoder reranking."""

import os
from threading import RLock
from typing import Any

from backend.retrieval.models import RetrievalCandidate


class CrossEncoderReranker:
    """Load one cross-encoder and score query/candidate pairs in batches."""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 32,
        device: str = "cpu",
        model: Any | None = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model = model
        self._lock = RLock()

    def _get_model(self) -> Any:
        with self._lock:
            if self._model is None:
                os.environ.setdefault("USE_TF", "0")
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name, device=self.device)
            return self._model

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalCandidate],
        top_n: int,
    ) -> list[RetrievalCandidate]:
        if not query.strip() or not candidates or top_n <= 0:
            return []
        model = self._get_model()
        scores = model.predict(
            [(query, candidate.text) for candidate in candidates],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        ranked = sorted(zip(candidates, scores), key=lambda item: float(item[1]), reverse=True)
        return [
            candidate.__class__(
                **{
                    **candidate.__dict__,
                    "cross_encoder_score": float(score),
                    "final_rank": index,
                }
            )
            for index, (candidate, score) in enumerate(ranked[:top_n], start=1)
        ]
