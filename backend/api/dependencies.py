"""FastAPI dependencies and application-scoped service container."""

from dataclasses import dataclass
from typing import Generator

from fastapi import Request
from sqlalchemy.orm import Session

from backend.core.config import Settings
from backend.database.database import Database
from backend.ingestion.service import IngestionService
from backend.memory.service import ConversationService
from backend.rag.service import RAGService
from backend.retrieval.service import RetrievalService
from backend.evaluation.runner import EvaluationRunner


@dataclass
class ServiceContainer:
    """Services shared for the lifetime of one FastAPI application."""

    settings: Settings
    database: Database
    ingestion: IngestionService
    retrieval: RetrievalService
    conversations: ConversationService
    rag: RAGService
    evaluation: EvaluationRunner


def get_container(request: Request) -> ServiceContainer:
    """Return the initialized application service container."""
    return request.app.state.services


def get_db_session(request: Request) -> Generator[Session, None, None]:
    """Yield one request-scoped database session and always close it."""
    container = get_container(request)
    session = container.database.get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_ingestion_service(request: Request) -> IngestionService:
    return get_container(request).ingestion


def get_retrieval_service(request: Request) -> RetrievalService:
    return get_container(request).retrieval


def get_conversation_service(request: Request) -> ConversationService:
    return get_container(request).conversations


def get_rag_service(request: Request) -> RAGService:
    return get_container(request).rag


def get_evaluation_runner(request: Request) -> EvaluationRunner:
    return get_container(request).evaluation
