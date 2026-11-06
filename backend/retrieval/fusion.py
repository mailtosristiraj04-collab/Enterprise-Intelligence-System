"""Reciprocal Rank Fusion for dense and sparse rankings."""

from dataclasses import replace

from backend.retrieval.models import RetrievalCandidate


def reciprocal_rank_fusion(
    rankings: list[list[RetrievalCandidate]],
    constant: int = 60,
) -> list[RetrievalCandidate]:
    """Merge ranked candidates using reciprocal rank fusion."""
    if constant <= 0:
        raise ValueError("RRF constant must be positive.")
    merged: dict[str, RetrievalCandidate] = {}
    scores: dict[str, float] = {}
    for ranking in rankings:
        for position, candidate in enumerate(ranking, start=1):
            rank = candidate.dense_rank or candidate.bm25_rank or position
            existing = merged.get(candidate.chunk_id)
            if existing is None:
                merged[candidate.chunk_id] = candidate
            else:
                merged[candidate.chunk_id] = replace(
                    existing,
                    dense_score=(
                        candidate.dense_score
                        if candidate.dense_score is not None
                        else existing.dense_score
                    ),
                    bm25_score=(
                        candidate.bm25_score
                        if candidate.bm25_score is not None
                        else existing.bm25_score
                    ),
                    dense_rank=(
                        candidate.dense_rank
                        if candidate.dense_rank is not None
                        else existing.dense_rank
                    ),
                    bm25_rank=(
                        candidate.bm25_rank
                        if candidate.bm25_rank is not None
                        else existing.bm25_rank
                    ),
                )
            scores[candidate.chunk_id] = scores.get(candidate.chunk_id, 0.0) + 1 / (constant + rank)
    return [
        replace(merged[chunk_id], rrf_score=scores[chunk_id])
        for chunk_id in sorted(scores, key=scores.get, reverse=True)
    ]
