"""Validated JSONL evaluation dataset handling."""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.database.models import Chunk, Document


EVALUATION_CATEGORIES = frozenset(
    {
        "direct_factual",
        "multi_chunk",
        "multi_document",
        "semantic_paraphrase",
        "technical_terminology",
        "insufficient",
        "conflicting",
    }
)

@dataclass(frozen=True)
class EvaluationSample:
    """Human-authored question and relevance labels."""

    question: str
    reference_answer: str | None
    relevant_document_ids: frozenset[str]
    relevant_chunk_ids: frozenset[str]
    expected_pages: frozenset[int]
    category: str
    sufficient_evidence: bool
    metadata: dict[str, Any] = field(default_factory=dict)


class DatasetValidationError(ValueError):
    """Raised when an evaluation sample is malformed."""


class EvaluationDataset:
    """Load a reproducible JSONL dataset without touching production storage."""

    def __init__(self, samples: list[EvaluationSample], source: str):
        self.samples = samples
        self.source = source

    @classmethod
    def load(cls, path: Path, session: Session | None = None) -> "EvaluationDataset":
        if not path.exists():
            raise DatasetValidationError(f"Evaluation dataset does not exist: {path}")
        samples: list[EvaluationSample] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetValidationError(f"Invalid JSON on line {line_number}.") from exc
            samples.append(cls._parse_sample(raw, line_number))
        dataset = cls(samples, str(path))
        questions = [sample.question.casefold() for sample in samples]
        if len(questions) != len(set(questions)):
            raise DatasetValidationError("Duplicate evaluation questions are not allowed.")
        if session is not None:
            dataset.validate_references(session)
        return dataset

    @staticmethod
    def _parse_sample(raw: Any, line_number: int) -> EvaluationSample:
        if not isinstance(raw, dict):
            raise DatasetValidationError(f"Sample on line {line_number} must be an object.")
        question = raw.get("question")
        category = raw.get("category")
        if not isinstance(question, str) or not question.strip():
            raise DatasetValidationError(f"Sample on line {line_number} has no question.")
        if not isinstance(category, str) or not category.strip():
            raise DatasetValidationError(f"Sample on line {line_number} has no category.")
        if category not in EVALUATION_CATEGORIES:
            raise DatasetValidationError(
                f"Unsupported category '{category}' on line {line_number}."
            )
        chunks = raw.get("relevant_chunk_ids", [])
        documents = raw.get("relevant_document_ids", [])
        pages = raw.get("expected_pages", [])
        sufficient = raw.get("sufficient_evidence")
        if not isinstance(chunks, list) or not all(isinstance(item, str) and item for item in chunks):
            raise DatasetValidationError(f"Invalid relevant_chunk_ids on line {line_number}.")
        if not isinstance(documents, list) or not all(isinstance(item, str) and item for item in documents):
            raise DatasetValidationError(f"Invalid relevant_document_ids on line {line_number}.")
        if not isinstance(pages, list) or not all(isinstance(item, int) for item in pages):
            raise DatasetValidationError(f"Invalid expected_pages on line {line_number}.")
        if any(item <= 0 for item in pages):
            raise DatasetValidationError(f"Page numbers must be positive on line {line_number}.")
        if not isinstance(sufficient, bool):
            raise DatasetValidationError(f"Sample on line {line_number} needs sufficient_evidence.")
        if sufficient == (category == "insufficient"):
            raise DatasetValidationError(
                f"Category and sufficient_evidence disagree on line {line_number}."
            )
        reference_answer = raw.get("reference_answer")
        if sufficient:
            if not isinstance(reference_answer, str) or not reference_answer.strip():
                raise DatasetValidationError(
                    f"Answerable sample on line {line_number} needs a manual reference_answer."
                )
            if not chunks or not documents:
                raise DatasetValidationError(
                    f"Answerable sample on line {line_number} needs document and chunk labels."
                )
        elif chunks or documents or pages or reference_answer:
            raise DatasetValidationError(
                f"Insufficient sample on line {line_number} must not contain answer labels."
            )
        if len(chunks) != len(set(chunks)) or len(documents) != len(set(documents)):
            raise DatasetValidationError(
                f"Duplicate relevance labels on line {line_number}."
            )
        return EvaluationSample(
            question=question.strip(),
            reference_answer=reference_answer.strip() if isinstance(reference_answer, str) else None,
            relevant_document_ids=frozenset(documents),
            relevant_chunk_ids=frozenset(chunks),
            expected_pages=frozenset(pages),
            category=category.strip(),
            sufficient_evidence=sufficient,
            metadata=raw.get("metadata", {}),
        )

    def validate_references(self, session: Session) -> None:
        """Verify every answerable label against the current indexed database."""
        document_ids = set(session.scalars(select(Document.id)).all())
        chunk_rows = session.scalars(select(Chunk)).all()
        chunks = {row.id: row for row in chunk_rows}
        for index, sample in enumerate(self.samples, 1):
            if not sample.sufficient_evidence:
                continue
            missing_documents = sample.relevant_document_ids - document_ids
            if missing_documents:
                raise DatasetValidationError(
                    f"Sample {index} references missing document IDs: "
                    f"{sorted(missing_documents)}."
                )
            missing_chunks = sample.relevant_chunk_ids - chunks.keys()
            if missing_chunks:
                raise DatasetValidationError(
                    f"Sample {index} references missing chunk IDs: "
                    f"{sorted(missing_chunks)}."
                )
            chunk_documents = {chunks[item].document_id for item in sample.relevant_chunk_ids}
            if chunk_documents != sample.relevant_document_ids:
                raise DatasetValidationError(
                    f"Sample {index} document labels do not match labeled chunk parents."
                )
            if sample.expected_pages:
                chunk_pages = {
                    chunks[item].page_number
                    for item in sample.relevant_chunk_ids
                    if chunks[item].page_number is not None
                }
                if not sample.expected_pages <= chunk_pages:
                    raise DatasetValidationError(
                        f"Sample {index} has expected pages not present on labeled chunks."
                    )
