"""Transactional document ingestion orchestration."""

import json
from datetime import datetime, timezone
from pathlib import Path
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.core.config import Settings
from backend.core.exceptions import DocumentNotFoundError, DuplicateDocumentError, EmptyDocumentError
from backend.database.models import Chunk, Document, ProcessingStatus
from backend.embeddings.service import EmbeddingService
from backend.vectorstore.faiss_store import FaissVectorStore
from backend.ingestion.chunking import chunk_pages
from backend.ingestion.extractors import get_extractor
from backend.ingestion.preprocessing import clean_text
from backend.utils.file_utils import compute_file_hash, get_file_extension, save_file, validate_file


class IngestionService:
    """Validate, extract, chunk, and persist documents atomically."""

    def __init__(
        self,
        settings: Settings,
        embedding_service: EmbeddingService | None = None,
        vector_store: FaissVectorStore | None = None,
    ):
        self.settings = settings
        self.embedding_service = embedding_service or EmbeddingService(
            model_name=settings.embedding_model,
            dimension=settings.embedding_dimension,
            batch_size=settings.embedding_batch_size,
            device=settings.embedding_device,
        )
        self.vector_store = vector_store or FaissVectorStore(
            settings.get_vectorstore_path(),
            settings.embedding_dimension,
            settings.vector_index_filename,
            settings.vector_mapping_filename,
        )

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
            status=ProcessingStatus.PENDING.value,
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
        except Exception:
            session.rollback()
            stored_path.unlink(missing_ok=True)
            raise
        try:
            document.status = ProcessingStatus.EMBEDDING.value
            session.commit()
            chunk_rows = list(document.chunks)
            vectors = self.embedding_service.embed_documents([chunk.text for chunk in chunk_rows])
            document.status = ProcessingStatus.INDEXING.value
            session.commit()
            faiss_ids = self.vector_store.add(
                vectors, [chunk.id for chunk in chunk_rows], document.id
            )
            if len(faiss_ids) != len(chunk_rows):
                raise RuntimeError("Not all document chunks were indexed.")
            for chunk, faiss_id in zip(chunk_rows, faiss_ids):
                chunk.faiss_index_id = faiss_id
            document.status = ProcessingStatus.COMPLETED.value
            session.commit()
            session.refresh(document)
            return document
        except Exception as exc:
            session.rollback()
            try:
                self.vector_store.delete_document(document.id)
            finally:
                failed = session.get(Document, document.id)
                if failed is not None:
                    failed.status = ProcessingStatus.FAILED.value
                    failed.error_message = str(exc)
                    session.commit()
            raise

    @staticmethod
    def require(session: Session, document_id: str) -> Document:
        document = session.get(Document, document_id)
        if document is None:
            raise DocumentNotFoundError(document_id)
        return document

    def delete(self, session: Session, document_id: str) -> None:
        document = self.require(session, document_id)
        path = Path(document.stored_file_path)
        self.vector_store.delete_document(document.id)
        session.delete(document)
        session.commit()
        path.unlink(missing_ok=True)
