from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.api.app import create_app
from backend.api.dependencies import (
    get_db_session,
    get_ingestion_service,
    get_rag_service,
)
from backend.api.schemas import ChatResponse
from backend.core.exceptions import (
    ConversationNotFoundError,
    LLMDeadlineExceededError,
    LLMError,
    LLMServiceUnavailableError,
)
from backend.database.models import Document


class FakeIngestion:
    def __init__(self):
        self.document = Document(
            id="doc-1",
            filename="stored.txt",
            original_filename="source.txt",
            file_type=".txt",
            file_size=5,
            file_hash="e" * 64,
            stored_file_path="",
            status="completed",
            page_count=1,
            chunk_count=1,
            uploaded_at=datetime.now(timezone.utc),
        )

    def ingest(self, session, filename, content, content_type=None):
        self.document.original_filename = filename
        return self.document

    def delete(self, session, document_id):
        if document_id != self.document.id:
            from backend.core.exceptions import DocumentNotFoundError

            raise DocumentNotFoundError(document_id)


class FakeRAG:
    def answer(self, session, question, conversation_id=None):
        if conversation_id == "missing":
            raise ConversationNotFoundError(conversation_id)
        return {
            "answer": "Grounded answer",
            "citations": [],
            "conversation_id": conversation_id or "conversation-1",
            "retrieved_chunks": [],
            "retrieval": {"candidate_count": 0},
            "insufficient_context": True,
        }


class FailingRAG:
    def answer(self, session, question, conversation_id=None):
        raise LLMError("model unavailable")


class ProviderFailingRAG:
    def __init__(self, error):
        self.error = error

    def answer(self, session, question, conversation_id=None):
        raise self.error


def _session_override():
    class Session:
        def commit(self):
            pass

        def rollback(self):
            pass

        def close(self):
            pass

        def query(self, model):
            class Query:
                def order_by(self, *args):
                    return []

            return Query()

    session = Session()
    yield session


def _client():
    app = create_app()
    ingestion = FakeIngestion()
    rag = FakeRAG()
    app.dependency_overrides[get_db_session] = _session_override
    app.dependency_overrides[get_ingestion_service] = lambda: ingestion
    app.dependency_overrides[get_rag_service] = lambda: rag
    return TestClient(app), app


def test_api_endpoints_and_openapi() -> None:
    client, app = _client()
    with client:
        health = client.get("/health")
        upload = client.post(
            "/documents/upload",
            files={"file": ("source.txt", b"hello", "text/plain")},
        )
        chat = client.post("/chat", json={"question": "What is here?"})
        schema = client.get("/openapi.json")

    assert health.status_code == 200
    assert upload.status_code == 201
    assert upload.json()["original_filename"] == "source.txt"
    assert chat.status_code == 200
    assert chat.json()["insufficient_context"] is True
    assert schema.status_code == 200
    assert "/documents/upload" in schema.json()["paths"]
    assert "/chat" in schema.json()["paths"]
    app.dependency_overrides.clear()


def test_chat_invalid_conversation_and_invalid_request() -> None:
    client, app = _client()
    with client:
        invalid_conversation = client.post(
            "/chat",
            json={"question": "Question", "conversation_id": "missing"},
        )
        invalid_request = client.post("/chat", json={"question": ""})

    assert invalid_conversation.status_code == 404
    assert invalid_request.status_code == 422
    app.dependency_overrides.clear()


def test_api_llm_failure_is_explicit() -> None:
    client, app = _client()
    app.dependency_overrides[get_rag_service] = lambda: FailingRAG()
    with client:
        response = client.post("/chat", json={"question": "Question"})

    assert response.status_code == 502
    assert response.json() == {"detail": "model unavailable"}
    app.dependency_overrides.clear()


@pytest.mark.parametrize(
    ("error", "status_code", "message"),
    [
        (
            LLMServiceUnavailableError("HTTP status 503: secret=do-not-expose"),
            503,
            "The AI provider is temporarily unavailable. Please try again later.",
        ),
        (
            LLMDeadlineExceededError(),
            504,
            "The AI provider did not complete the request in time. Please try again.",
        ),
    ],
)
def test_api_provider_failures_have_safe_status_and_body(
    error, status_code, message
) -> None:
    client, app = _client()
    provider_error = RuntimeError("authorization=secret provider payload")
    error.__cause__ = provider_error
    app.dependency_overrides[get_rag_service] = lambda: ProviderFailingRAG(error)

    with client:
        response = client.post("/chat", json={"question": "Question"})

    assert response.status_code == status_code
    assert response.json() == {"detail": message}
    assert "secret" not in response.text
    assert "authorization" not in response.text
    app.dependency_overrides.clear()


def test_api_provider_failure_logs_sanitized_chained_traceback(monkeypatch) -> None:
    import importlib
    from unittest.mock import Mock

    app_module = importlib.import_module("backend.api.app")
    logger = Mock()
    monkeypatch.setattr(app_module, "logger", logger)
    error = LLMDeadlineExceededError()
    error.__cause__ = RuntimeError("authorization=secret provider payload")
    client, app = _client()
    app.dependency_overrides[get_rag_service] = lambda: ProviderFailingRAG(error)

    with client:
        response = client.post("/chat", json={"question": "Question"})

    assert response.status_code == 504
    logger.error.assert_called_once()
    logged = logger.error.call_args.kwargs
    assert "secret" not in logged["traceback"]
    assert "authorization=[REDACTED]" in logged["traceback"]
    assert logged["provider_status"] == 504
    app.dependency_overrides.clear()


def test_api_database_failure_is_not_a_success_response() -> None:
    client, app = _client()
    client = TestClient(app, raise_server_exceptions=False)

    def failing_session():
        raise RuntimeError("database unavailable")
        yield

    app.dependency_overrides[get_db_session] = failing_session
    with client:
        response = client.get("/documents")

    assert response.status_code == 500
    assert response.json() == {"detail": "An internal server error occurred."}
    app.dependency_overrides.clear()
