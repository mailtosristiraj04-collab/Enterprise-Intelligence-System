"""Pydantic schemas for the public API."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DocumentResponse(BaseModel):
    """Public document metadata."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    original_filename: str
    file_type: str
    file_size: int
    file_hash: str
    status: str
    page_count: int | None
    chunk_count: int
    uploaded_at: datetime
    processed_at: datetime | None


class DocumentStatusResponse(BaseModel):
    """Document processing status."""

    id: str
    status: str
    error_message: str | None = None


class ChatRequest(BaseModel):
    """Chat request payload."""

    question: str = Field(
        min_length=1,
        description="Question to answer from indexed documents.",
        examples=["What does the indexed policy say about retention?"],
    )
    conversation_id: str | None = Field(
        default=None,
        description="Existing conversation ID to continue.",
    )


class CitationResponse(BaseModel):
    """Traceable source citation."""

    document_id: str
    document_name: str
    chunk_id: str
    page_number: int | None
    excerpt: str


class ChatResponse(BaseModel):
    """Structured grounded answer."""

    answer: str
    citations: list[CitationResponse]
    conversation_id: str
    retrieved_chunks: list[dict[str, object]]
    retrieval: dict[str, object]
    insufficient_context: bool
    answer_mode: str
