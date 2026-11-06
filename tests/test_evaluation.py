"""Evaluation dataset and metric tests."""

import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.database.models import Base, Chunk, Document
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


def _evaluation_session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    session.add(
        Document(
            id="doc-real",
            filename="policy.txt",
            original_filename="policy.txt",
            file_type=".txt",
            file_size=10,
            file_hash="a" * 64,
            stored_file_path="",
        )
    )
    session.add(
        Chunk(
            id="chunk-real",
            document_id="doc-real",
            chunk_index=0,
            text="The policy requires seven years.",
            page_number=1,
            char_count=34,
            token_estimate=6,
        )
    )
    session.add(
        Document(
            id="doc-other",
            filename="other.txt",
            original_filename="other.txt",
            file_type=".txt",
            file_size=10,
            file_hash="b" * 64,
            stored_file_path="",
        )
    )
    session.add(
        Chunk(
            id="chunk-other",
            document_id="doc-other",
            chunk_index=0,
            text="Unrelated content.",
            page_number=1,
            char_count=18,
            token_estimate=2,
        )
    )
    session.commit()
    return session


def _write_sample(tmp_path, **overrides):
    sample = {
        "question": "How long is the retention period?",
        "reference_answer": "Seven years.",
        "relevant_document_ids": ["doc-real"],
        "relevant_chunk_ids": ["chunk-real"],
        "expected_pages": [1],
        "category": "direct_factual",
        "sufficient_evidence": True,
    }
    sample.update(overrides)
    path = tmp_path / "references.jsonl"
    path.write_text(json.dumps(sample), encoding="utf-8")
    return path


def test_dataset_validates_references_against_database(tmp_path) -> None:
    session = _evaluation_session()
    try:
        dataset = EvaluationDataset.load(_write_sample(tmp_path), session)
        assert dataset.samples[0].relevant_chunk_ids == {"chunk-real"}
    finally:
        session.close()


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"relevant_document_ids": ["missing"]}, "missing document"),
        ({"relevant_chunk_ids": ["missing"]}, "missing chunk"),
        ({"relevant_chunk_ids": ["chunk-other"]}, "do not match"),
        ({"expected_pages": [9]}, "expected pages"),
    ],
)
def test_dataset_rejects_invalid_database_references(tmp_path, overrides, message) -> None:
    session = _evaluation_session()
    try:
        with pytest.raises(DatasetValidationError, match=message):
            EvaluationDataset.load(_write_sample(tmp_path, **overrides), session)
    finally:
        session.close()


def test_dataset_rejects_labels_on_insufficient_sample(tmp_path) -> None:
    with pytest.raises(DatasetValidationError, match="must not contain"):
        EvaluationDataset.load(
            _write_sample(
                tmp_path,
                reference_answer="This should not be supplied.",
                category="insufficient",
                sufficient_evidence=False,
            )
        )
