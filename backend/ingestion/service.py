"""Transactional document ingestion orchestration."""

import json
from datetime import datetime, timezone
from pathlib import Path
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.core.config import Settings
from backend.core.exceptions import DocumentNotFoundError, DuplicateDocumentError, EmptyDocumentError
from backend.database.models import Chunk, Document, ProcessingStatus
from backend.ingestion.chunking import chunk_pages
from backend.ingestion.extractors import get_extractor
from backend.ingestion.preprocessing import clean_text
from backend.utils.file_utils import compute_file_hash, get_file_extension, save_file, validate_file


class IngestionService:
    """Validate, extract, chunk, and persist documents atomically."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def ingest(
        self,
        session: Session,
        filename: str,
        content: bytes,
        content_type: str | None = None,
    ) -> Document:
        validate_file(
            filename, content,
            max_file_size_bytes=self.settings.max_file_size_bytes,
            allowed_extensions=self.settings.allowed_extensions,
            content_type=content_type,
        )
        if not content:
            raise EmptyDocumentError(filename)
        file_hash = compute_file_hash(content)
        if session.scalar(select(Document).where(Document.file_hash == file_hash)):
            raise DuplicateDocumentError(file_hash)
        extension = get_file_extension(filename)
        result = get_extractor(extension).extract(content, filename)
        pages = [(clean_text(page.text), page.page_number) for page in result.pages]
        pages = [(text, number) for text, number in pages if text]
        if not pages:
            raise EmptyDocumentError(filename)
        chunks = chunk_pages(
            pages,
            chunk_size_tokens=self.settings.chunk_size,
            overlap_tokens=self.settings.chunk_overlap,
        )
        if not chunks:
            raise EmptyDocumentError(filename)
        stored_path = save_file(content, self.settings.get_upload_path(), filename)
        document = Document(
            filename=stored_path.name,
            original_filename=Path(filename).name,
            file_type=extension,
            file_size=len(content),
            file_hash=file_hash,
            stored_file_path=str(stored_path),
            metadata_json=json.dumps(result.metadata, default=str),
            extraction_warnings_json=json.dumps(result.warnings),
            status=ProcessingStatus.COMPLETED.value,
            page_count=len(result.pages),
            chunk_count=len(chunks),
            processed_at=datetime.now(timezone.utc),
        )
        try:
            session.add(document)
            session.flush()
            session.add_all(
                Chunk(
                    document_id=document.id,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    page_number=chunk.page_number,
                    char_count=len(chunk.text),
                    token_estimate=chunk.metadata["token_estimate"],
                    metadata_json=json.dumps(chunk.metadata),
                )
                for chunk in chunks
            )
            session.commit()
            session.refresh(document)
            return document
        except Exception:
            session.rollback()
            stored_path.unlink(missing_ok=True)
            raise

    @staticmethod
    def require(session: Session, document_id: str) -> Document:
        document = session.get(Document, document_id)
        if document is None:
            raise DocumentNotFoundError(document_id)
        return document

    @staticmethod
    def delete(session: Session, document_id: str) -> None:
        document = IngestionService.require(session, document_id)
        path = Path(document.stored_file_path)
        session.delete(document)
        session.commit()
        path.unlink(missing_ok=True)
