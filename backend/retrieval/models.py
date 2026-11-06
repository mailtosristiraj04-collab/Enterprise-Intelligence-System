"""Shared retrieval result types."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RetrievalCandidate:
    """A chunk returned by one or more retrieval stages."""

    chunk_id: str
    document_id: str
    text: str
    page_number: int | None
    dense_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    cross_encoder_score: float | None = None
    dense_rank: int | None = None
    bm25_rank: int | None = None
    final_rank: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
