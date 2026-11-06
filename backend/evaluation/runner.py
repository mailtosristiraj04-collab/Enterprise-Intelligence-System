"""Evaluation runner for retrieval configurations and latency measurements."""

from datetime import datetime, timezone
import json
from statistics import median
from time import perf_counter
from typing import Any
import uuid

from sqlalchemy.orm import Session

from backend.core.config import Settings
from backend.database.models import EvaluationRun
from backend.evaluation.dataset import EvaluationDataset
from backend.evaluation.metrics import (
    aggregate_metrics,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from backend.retrieval.fusion import reciprocal_rank_fusion
from backend.retrieval.models import RetrievalCandidate
from backend.retrieval.service import RetrievalService


CONFIGURATIONS = ("dense", "bm25", "hybrid", "hybrid_reranked")


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return ordered[index]


def _latency_summary(values: list[float]) -> dict[str, float | None]:
    return {
        "average_ms": sum(values) / len(values) if values else None,
        "median_ms": median(values) if values else None,
        "p95_ms": _percentile(values, 0.95),
        "sample_count": len(values),
    }


class EvaluationRunner:
    """Run reproducible retrieval evaluations against the current index."""

    def __init__(self, settings: Settings, retrieval: RetrievalService):
        self.settings = settings
        self.retrieval = retrieval

    def _retrieve(
        self, session: Session, query: str, configuration: str, k: int
    ) -> tuple[list[RetrievalCandidate], dict[str, float]]:
        timings: dict[str, float] = {}
        started = perf_counter()
        if configuration == "dense":
            results = self.retrieval.dense.retrieve(session, query, k)
        elif configuration == "bm25":
            results = self.retrieval.bm25.retrieve(session, query, k)
        else:
            dense_started = perf_counter()
            dense = self.retrieval.dense.retrieve(session, query, k)
            timings["dense_ms"] = (perf_counter() - dense_started) * 1000
            sparse_started = perf_counter()
            sparse = self.retrieval.bm25.retrieve(session, query, k)
            timings["bm25_ms"] = (perf_counter() - sparse_started) * 1000
            fusion_started = perf_counter()
            results = reciprocal_rank_fusion([dense, sparse], self.settings.rrf_constant)[:k]
            timings["rrf_ms"] = (perf_counter() - fusion_started) * 1000
            if configuration == "hybrid_reranked" and self.retrieval.reranker is not None:
                rerank_started = perf_counter()
                results = self.retrieval.reranker.rerank(query, results, k)
                timings["rerank_ms"] = (perf_counter() - rerank_started) * 1000
        timings["total_ms"] = (perf_counter() - started) * 1000
        return results, timings

    def run(
        self,
        session: Session,
        dataset: EvaluationDataset,
        *,
        ks: list[int] | None = None,
        configurations: list[str] | None = None,
    ) -> dict[str, Any]:
        ks = sorted(set(ks or [5, 10]))
        if not ks or any(k <= 0 for k in ks):
            raise ValueError("Evaluation k values must be positive.")
        configurations = configurations or list(CONFIGURATIONS)
        invalid = set(configurations) - set(CONFIGURATIONS)
        if invalid:
            raise ValueError(f"Unsupported evaluation configurations: {sorted(invalid)}")
        results: dict[str, Any] = {}
        for configuration in configurations:
            per_query: list[dict[str, Any]] = []
            stage_latencies: dict[str, list[float]] = {}
            for sample in dataset.samples:
                candidates, timings = self._retrieve(session, sample.question, configuration, max(ks))
                ids = [candidate.chunk_id for candidate in candidates]
                relevant = set(sample.relevant_chunk_ids)
                metrics_by_k = {
                    str(k): {
                        "precision": precision_at_k(ids, relevant, k),
                        "recall": recall_at_k(ids, relevant, k),
                        "ndcg": ndcg_at_k(ids, relevant, k),
                    }
                    for k in ks
                }
                for name, value in timings.items():
                    stage_latencies.setdefault(name, []).append(value)
                per_query.append({
                    "question": sample.question,
                    "category": sample.category,
                    "retrieved_chunk_ids": ids,
                    "relevant_chunk_ids": sorted(relevant),
                    "mrr": mean_reciprocal_rank(ids, relevant),
                    "metrics": metrics_by_k,
                    "latency_ms": timings,
                })
            aggregate = {}
            for k in ks:
                rows = [
                    {
                        "precision": item["metrics"][str(k)]["precision"],
                        "recall": item["metrics"][str(k)]["recall"],
                        "mrr": item["mrr"],
                        "ndcg": item["metrics"][str(k)]["ndcg"],
                    }
                    for item in per_query
                ]
                aggregate[str(k)] = aggregate_metrics(rows)
            results[configuration] = {
                "aggregate": aggregate,
                "per_query": per_query,
                "latency": {name: _latency_summary(values) for name, values in stage_latencies.items()},
            }
        return {
            "run_id": str(uuid.uuid4()),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_source": dataset.source,
            "sample_count": len(dataset.samples),
            "k_values": ks,
            "configurations": results,
            "metadata": {
                "embedding_model": self.settings.embedding_model,
                "reranker_model": self.settings.reranker_model,
                "hardware": self.settings.embedding_device,
                "notes": "Local measurements only; not production-scale benchmarks.",
            },
        }

    @staticmethod
    def persist(session: Session, report: dict[str, Any]) -> None:
        session.add(EvaluationRun(
            id=report["run_id"],
            run_type="retrieval",
            results_json=json.dumps(report),
        ))
        session.flush()

    @staticmethod
    def history(session: Session) -> list[dict[str, Any]]:
        runs = session.query(EvaluationRun).order_by(EvaluationRun.created_at.desc()).all()
        return [
            {
                "run_id": run.id,
                "run_type": run.run_type,
                "created_at": run.created_at.isoformat(),
                "sample_count": json.loads(run.results_json).get("sample_count", 0),
            }
            for run in runs
        ]

    @staticmethod
    def get(session: Session, run_id: str) -> dict[str, Any] | None:
        run = session.get(EvaluationRun, run_id)
        return json.loads(run.results_json) if run else None
