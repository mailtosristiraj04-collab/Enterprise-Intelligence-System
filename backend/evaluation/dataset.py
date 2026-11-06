"""Validated JSONL evaluation dataset handling."""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


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
    def load(cls, path: Path) -> "EvaluationDataset":
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
        return cls(samples, str(path))

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
        if not isinstance(sufficient, bool):
            raise DatasetValidationError(f"Sample on line {line_number} needs sufficient_evidence.")
        if sufficient and not chunks and not documents:
            raise DatasetValidationError(
                f"Sufficient sample on line {line_number} needs relevance labels."
            )
        return EvaluationSample(
            question=question.strip(),
            reference_answer=raw.get("reference_answer"),
            relevant_document_ids=frozenset(documents),
            relevant_chunk_ids=frozenset(chunks),
            expected_pages=frozenset(pages),
            category=category.strip(),
            sufficient_evidence=sufficient,
            metadata=raw.get("metadata", {}),
        )
