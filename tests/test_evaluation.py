"""Evaluation dataset and metric tests."""

import json

import pytest

from backend.evaluation.dataset import DatasetValidationError, EvaluationDataset
from backend.evaluation.metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from backend.evaluation.rag import evaluate_answer


def test_retrieval_metrics_are_deterministic() -> None:
    retrieved = ["a", "x", "b"]
    relevant = {"a", "b"}
    assert precision_at_k(retrieved, relevant, 2) == 0.5
    assert recall_at_k(retrieved, relevant, 2) == 0.5
    assert mean_reciprocal_rank(retrieved, relevant) == 1.0
    assert ndcg_at_k(retrieved, relevant, 3) > 0


def test_no_relevance_labels_have_explicit_recall_edge_case() -> None:
    assert recall_at_k(["a"], set(), 5) is None
    assert mean_reciprocal_rank(["a"], set()) == 0
    assert ndcg_at_k(["a"], set(), 5) == 0


def test_rag_evaluation_rejects_citations_not_in_retrieved_context() -> None:
    result = evaluate_answer(
        "The policy says seven years.",
        "Seven years.",
        [{"chunk_id": "not-retrieved"}],
        {"retrieved"},
        True,
    )
    assert result["citation_correctness"] == 0.0
    assert result["faithfulness"] is None


def test_dataset_loads_valid_jsonl(tmp_path) -> None:
    path = tmp_path / "dataset.jsonl"
    path.write_text(
        json.dumps({
            "question": "What is the policy?",
            "reference_answer": "Seven years.",
            "relevant_chunk_ids": ["chunk-1"],
            "relevant_document_ids": ["doc-1"],
            "expected_pages": [1],
            "category": "direct_factual",
            "sufficient_evidence": True,
        }),
        encoding="utf-8",
    )
    dataset = EvaluationDataset.load(path)
    assert len(dataset.samples) == 1
    assert dataset.samples[0].relevant_chunk_ids == {"chunk-1"}


@pytest.mark.parametrize(
    "payload",
    [
        {"category": "direct_factual", "sufficient_evidence": True},
        {"question": "Q", "category": "direct_factual"},
        {"question": "Q", "category": "direct_factual", "sufficient_evidence": True},
    ],
)
def test_dataset_rejects_missing_ground_truth(payload, tmp_path) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DatasetValidationError):
        EvaluationDataset.load(path)
