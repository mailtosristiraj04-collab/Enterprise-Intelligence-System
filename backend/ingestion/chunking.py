"""Configurable paragraph-aware text chunking."""

from dataclasses import dataclass
import re
from typing import Any

from backend.utils.file_utils import estimate_tokens


@dataclass(frozen=True)
class TextChunk:
    """A stable, ordered chunk ready for database persistence."""

    chunk_index: int
    text: str
    page_number: int | None
    metadata: dict[str, Any]


def _split_long_text(text: str, target_chars: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > target_chars:
            pieces.append(current)
            current = ""
        current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def chunk_pages(
    pages: list[tuple[str, int | None]],
    *,
    chunk_size_tokens: int = 500,
    overlap_tokens: int = 100,
) -> list[TextChunk]:
    """Chunk cleaned pages without unnecessarily breaking paragraphs."""
    if chunk_size_tokens <= 0 or overlap_tokens < 0 or overlap_tokens >= chunk_size_tokens:
        raise ValueError("chunk_size_tokens must be positive and overlap_tokens must be smaller.")
    target_chars = chunk_size_tokens * 4
    overlap_chars = overlap_tokens * 4
    chunks: list[TextChunk] = []
    current = ""
    current_page: int | None = None
    for page_text, page_number in pages:
        for paragraph in (part.strip() for part in page_text.split("\n\n") if part.strip()):
            pieces = _split_long_text(paragraph, target_chars) if len(paragraph) > target_chars else [paragraph]
            for piece in pieces:
                if current and current_page is None:
                    current_page = page_number
                if current and len(current) + len(piece) + 2 > target_chars:
                    chunks.append(TextChunk(len(chunks), current, current_page, {"token_estimate": estimate_tokens(current)}))
                    overlap = current[-overlap_chars:] if overlap_chars else ""
                    current = overlap
                    current_page = page_number
                if not current:
                    current_page = page_number
                current = f"{current}\n\n{piece}".strip()
    if current:
        chunks.append(TextChunk(len(chunks), current, current_page, {"token_estimate": estimate_tokens(current)}))
    return chunks
