"""RAG chat API."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.api.dependencies import get_db_session, get_rag_service
from backend.api.schemas import ChatRequest, ChatResponse
from backend.rag.service import RAGService

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post(
    "",
    response_model=ChatResponse,
    summary="Answer a question using indexed documents",
    description="Runs hybrid retrieval, builds bounded context, and returns a grounded answer with citations.",
)
def chat(
    request: ChatRequest,
    session: Session = Depends(get_db_session),
    rag: RAGService = Depends(get_rag_service),
) -> dict[str, object]:
    return rag.answer(session, request.question, request.conversation_id)
