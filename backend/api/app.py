"""FastAPI application factory and HTTP exception handlers."""

from contextlib import asynccontextmanager
from typing import AsyncIterator
import re
import traceback

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from backend.api.dependencies import ServiceContainer
from backend.core.config import get_settings
from backend.core.exceptions import (
    AppException,
    LLMDeadlineExceededError,
    LLMServiceUnavailableError,
)
from backend.core.logging import get_logger, setup_logging
from backend.api.documents import router as documents_router
from backend.api.chat import router as chat_router
from backend.database.database import get_database
from backend.ingestion.service import IngestionService
from backend.memory.service import ConversationService
from backend.rag.service import RAGService
from backend.retrieval.service import RetrievalService
from backend.evaluation.runner import EvaluationRunner
from backend.api.evaluation import router as evaluation_router

logger = get_logger(__name__)


def _safe_traceback(exc: BaseException) -> str:
    """Redact sensitive values from a chained provider traceback."""
    text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    text = re.sub(
        r"(?i)(authorization|x-goog-api-key|api[-_ ]?key)\s*[:=]\s*\S+",
        r"\1=[REDACTED]",
        text,
    )
    text = re.sub(r"(?i)bearer\s+\S+", "******", text)
    text = re.sub(r"(?i)(key|token|secret)=\S+", r"\1=[REDACTED]", text)
    return text


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialise application-wide foundation services."""
    settings = get_settings()
    setup_logging(settings.log_level)
    settings.ensure_directories()
    database = get_database()
    retrieval = RetrievalService(settings)
    conversations = ConversationService()
    rag = RAGService(settings, retrieval=retrieval, conversations=conversations)
    app.state.services = ServiceContainer(
        settings=settings,
        database=database,
        ingestion=IngestionService(
            settings,
            embedding_service=retrieval.dense.embeddings,
            vector_store=retrieval.dense.vector_store,
        ),
        retrieval=retrieval,
        conversations=conversations,
        rag=rag,
        evaluation=EvaluationRunner(settings, retrieval),
    )
    try:
        yield
    finally:
        database.engine.dispose()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Enterprise Knowledge Intelligence System",
        version="0.1.0",
        description=(
            "Backend for document ingestion, persistent indexing, hybrid retrieval, "
            "and grounded RAG answers."
        ),
        lifespan=lifespan,
    )
    app.include_router(documents_router)
    app.include_router(chat_router)
    app.include_router(evaluation_router)

    @app.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        """Return the service health status."""
        return {"status": "ok"}

    @app.exception_handler(AppException)
    async def app_exception_handler(
        _: Request, exc: AppException
    ) -> JSONResponse:
        if isinstance(exc, (LLMServiceUnavailableError, LLMDeadlineExceededError)):
            cause = exc.__cause__
            if cause is not None:
                logger.error(
                    "llm_provider_error",
                    error_type=type(cause).__name__,
                    provider_status=exc.status_code,
                    traceback=_safe_traceback(cause),
                )
        else:
            logger.warning("application_error", message=exc.message, status_code=exc.status_code)
        body: dict[str, object] = {"detail": exc.message}
        if exc.detail is not None and not isinstance(
            exc, (LLMServiceUnavailableError, LLMDeadlineExceededError)
        ):
            body["metadata"] = exc.detail
        return JSONResponse(status_code=exc.status_code, content=body)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_application_error", error=str(exc))
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal server error occurred."},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        _: Request, exc: RequestValidationError
    ) -> JSONResponse:
        logger.warning("request_validation_error", error_count=len(exc.errors()))
        return JSONResponse(
            status_code=422,
            content={"detail": "Request validation failed."},
        )

    return app


app = create_app()
