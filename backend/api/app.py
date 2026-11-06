"""FastAPI application factory and HTTP exception handlers."""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from backend.core.config import get_settings
from backend.core.exceptions import AppException
from backend.core.logging import get_logger, setup_logging
from backend.api.documents import router as documents_router

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Initialise application-wide foundation services."""
    settings = get_settings()
    setup_logging(settings.log_level)
    settings.ensure_directories()
    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Enterprise Knowledge Intelligence System",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(documents_router)

    @app.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        """Return the service health status."""
        return {"status": "ok"}

    @app.exception_handler(AppException)
    async def app_exception_handler(
        _: Request, exc: AppException
    ) -> JSONResponse:
        logger.warning("application_error", message=exc.message, status_code=exc.status_code)
        body: dict[str, object] = {"detail": exc.message}
        if exc.detail is not None:
            body["metadata"] = exc.detail
        return JSONResponse(status_code=exc.status_code, content=body)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_application_error", error=str(exc))
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal server error occurred."},
        )

    return app


app = create_app()
