"""Deterministic information-retrieval metrics."""

from math import log2
from typing import Iterable


def _relevance_ids(retrieved: Iterable[str], relevant: set[str]) -> list[int]:
    return [int(item in relevant) for item in retrieved]


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    return sum(_relevance_ids(retrieved[:k], relevant)) / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float | None:
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant:
        return None
    return sum(_relevance_ids(retrieved[:k], relevant)) / len(relevant)


def mean_reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    for rank, item in enumerate(retrieved, 1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant:
        return 0.0
    actual = sum(
        relevance / log2(rank + 1)
        for rank, relevance in enumerate(_relevance_ids(retrieved[:k], relevant), 1)
    )
    ideal_count = min(len(relevant), k)
    ideal = sum(1 / log2(rank + 1) for rank in range(1, ideal_count + 1))
    return actual / ideal if ideal else 0.0


def aggregate_metrics(per_query: list[dict[str, float | None]]) -> dict[str, float | None]:
    """Average metrics, excluding undefined recall from no-label queries."""
    result: dict[str, float | None] = {}
    for name in ("precision", "recall", "mrr", "ndcg"):
        values = [row[name] for row in per_query if row[name] is not None]
        result[name] = sum(values) / len(values) if values else None
    return result
