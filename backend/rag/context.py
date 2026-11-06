"""Bounded context construction and structured source citations."""

from dataclasses import dataclass
import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.database.models import Document
from backend.retrieval.models import RetrievalCandidate
from backend.utils.file_utils import estimate_tokens


@dataclass(frozen=True)
class Citation:
    """A traceable source supplied to the language model."""

    document_id: str
    document_name: str
    chunk_id: str
    page_number: int | None
    excerpt: str

    def as_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class ContextPackage:
    """Context text, citations, and the candidates actually supplied."""

    text: str
    citations: list[Citation]
    candidates: list[RetrievalCandidate]
    token_count: int


class ContextBuilder:
    """Select unique chunks under a token budget without mixing instructions."""

    def __init__(self, token_budget: int = 3000, max_chunks: int = 8):
        if token_budget <= 0 or max_chunks <= 0:
            raise ValueError("Context limits must be positive.")
        self.token_budget = token_budget
        self.max_chunks = max_chunks

    def build(
        self, session: Session, candidates: list[RetrievalCandidate]
    ) -> ContextPackage:
        unique: list[RetrievalCandidate] = []
        seen: set[str] = set()
        for candidate in candidates:
            if candidate.chunk_id in seen:
                continue
            if estimate_tokens(candidate.text) > self.token_budget:
                excerpt = candidate.text[: self.token_budget * 4]
            else:
                excerpt = candidate.text
            if not excerpt.strip():
                continue
            unique.append(candidate)
            seen.add(candidate.chunk_id)
            if len(unique) >= self.max_chunks:
                break

        documents = session.scalars(
            select(Document).where(Document.id.in_({item.document_id for item in unique}))
        ).all()
        names = {document.id: document.original_filename for document in documents}
        selected: list[RetrievalCandidate] = []
        citations: list[Citation] = []
        blocks: list[str] = []
        tokens = 0
        for candidate in unique:
            excerpt = candidate.text
            available = self.token_budget - tokens
            if available <= 0:
                break
            if estimate_tokens(excerpt) > available:
                excerpt = excerpt[: available * 4]
            if not excerpt.strip():
                continue
            tokens += estimate_tokens(excerpt)
            selected.append(candidate)
            citation = Citation(
                document_id=candidate.document_id,
                document_name=names.get(candidate.document_id, "Unknown document"),
                chunk_id=candidate.chunk_id,
                page_number=candidate.page_number,
                excerpt=excerpt,
            )
            citations.append(citation)
            blocks.append(
                f"[SOURCE {len(citations)}]\n"
                f"Document ID: {citation.document_id}\n"
                f"Document name: {citation.document_name}\n"
                f"Chunk ID: {citation.chunk_id}\n"
                f"Page: {citation.page_number if citation.page_number is not None else 'N/A'}\n"
                f"Content:\n{excerpt}"
            )
        return ContextPackage("\n\n".join(blocks), citations, selected, tokens)
