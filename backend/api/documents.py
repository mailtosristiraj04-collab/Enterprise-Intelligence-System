"""Document ingestion and lifecycle API routes."""

from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.orm import Session

from backend.api.dependencies import get_db_session, get_ingestion_service
from backend.api.schemas import DocumentResponse, DocumentStatusResponse
from backend.database.models import Document
from backend.ingestion.service import IngestionService

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "/upload",
    status_code=201,
    response_model=DocumentResponse,
    summary="Upload and index a document",
    description="Extracts, chunks, embeds, and indexes a PDF, DOC, DOCX, or TXT document.",
)
def upload_document(
    file: UploadFile = File(..., description="Document to ingest and index."),
    session: Session = Depends(get_db_session),
    ingestion: IngestionService = Depends(get_ingestion_service),
) -> Document:
    content = file.file.read()
    return ingestion.ingest(session, file.filename or "", content, file.content_type)


@router.get("", response_model=list[DocumentResponse], summary="List indexed documents")
def list_documents(session: Session = Depends(get_db_session)) -> list[Document]:
    return session.query(Document).order_by(Document.uploaded_at.desc()).all()


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get document metadata",
)
def get_document(document_id: str, session: Session = Depends(get_db_session)) -> Document:
    return IngestionService.require(session, document_id)


@router.get(
    "/{document_id}/status",
    response_model=DocumentStatusResponse,
    summary="Get document processing status",
)
def get_document_status(
    document_id: str, session: Session = Depends(get_db_session)
) -> dict[str, object]:
    document = IngestionService.require(session, document_id)
    return {"id": document.id, "status": document.status, "error_message": document.error_message}


@router.delete(
    "/{document_id}",
    status_code=204,
    response_model=None,
    summary="Delete a document and its index entries",
)
def delete_document(
    document_id: str,
    session: Session = Depends(get_db_session),
    ingestion: IngestionService = Depends(get_ingestion_service),
) -> None:
    ingestion.delete(session, document_id)
