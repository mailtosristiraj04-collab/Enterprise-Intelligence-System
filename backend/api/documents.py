"""Document ingestion and lifecycle API routes."""

from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.orm import Session

from backend.core.config import get_settings
from backend.database.database import get_db_session
from backend.database.models import Document
from backend.ingestion.service import IngestionService

router = APIRouter(prefix="/documents", tags=["documents"])


def _document_response(document: Document) -> dict[str, object]:
    return {
        "id": document.id,
        "filename": document.filename,
        "original_filename": document.original_filename,
        "file_type": document.file_type,
        "file_size": document.file_size,
        "file_hash": document.file_hash,
        "status": document.status,
        "page_count": document.page_count,
        "chunk_count": document.chunk_count,
        "uploaded_at": document.uploaded_at,
        "processed_at": document.processed_at,
    }


@router.post("/upload", status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    session: Session = Depends(get_db_session),
) -> dict[str, object]:
    content = await file.read()
    document = IngestionService(get_settings()).ingest(
        session,
        file.filename or "",
        content,
        file.content_type,
    )
    return _document_response(document)


@router.get("")
def list_documents(session: Session = Depends(get_db_session)) -> list[dict[str, object]]:
    return [_document_response(document) for document in session.query(Document).order_by(Document.uploaded_at.desc()).all()]


@router.get("/{document_id}")
def get_document(document_id: str, session: Session = Depends(get_db_session)) -> dict[str, object]:
    return _document_response(IngestionService.require(session, document_id))


@router.get("/{document_id}/status")
def get_document_status(document_id: str, session: Session = Depends(get_db_session)) -> dict[str, object]:
    document = IngestionService.require(session, document_id)
    return {"id": document.id, "status": document.status, "error_message": document.error_message}


@router.delete("/{document_id}", status_code=204)
def delete_document(document_id: str, session: Session = Depends(get_db_session)) -> None:
    IngestionService(get_settings()).delete(session, document_id)
