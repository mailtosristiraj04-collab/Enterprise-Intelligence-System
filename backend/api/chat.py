"""RAG chat API."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.core.config import get_settings
from backend.database.database import get_db_session
from backend.rag.service import RAGService

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    conversation_id: str | None = None


@router.post("")
def chat(
    request: ChatRequest,
    session: Session = Depends(get_db_session),
) -> dict[str, object]:
    return RAGService(get_settings()).answer(
        session, request.question, request.conversation_id
    )
