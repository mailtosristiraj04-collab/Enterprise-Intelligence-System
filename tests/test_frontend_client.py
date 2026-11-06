"""Contract tests for the HTTP-only Streamlit client."""

from unittest.mock import Mock, patch
from contextlib import contextmanager

import pytest
import requests

from frontend.api.client import APIClient, APIClientError


def response(status: int, payload=None) -> Mock:
    result = Mock()
    result.status_code = status
    result.ok = 200 <= status < 300
    result.json.return_value = payload
    return result


def test_client_lists_documents_and_uses_public_route() -> None:
    with patch("frontend.api.client.requests.request", return_value=response(200, [{"id": "doc-1"}])) as request:
        assert APIClient("http://backend", 4).list_documents() == [{"id": "doc-1"}]
    request.assert_called_once_with("GET", "http://backend/documents", timeout=4)


def test_client_uploads_multipart_document() -> None:
    with patch("frontend.api.client.requests.request", return_value=response(201, {"id": "doc-1"})) as request:
        result = APIClient("http://backend", 4).upload_document("notes.txt", b"hello", "text/plain")
    assert result["id"] == "doc-1"
    assert request.call_args.kwargs["files"]["file"] == ("notes.txt", b"hello", "text/plain")


def test_client_sends_conversation_id_for_continuation() -> None:
    with patch("frontend.api.client.requests.request", return_value=response(200, {"conversation_id": "c-1"})) as request:
        APIClient("http://backend", 4).chat("Continue", "c-1")
    assert request.call_args.kwargs["json"] == {"question": "Continue", "conversation_id": "c-1"}


def test_client_maps_backend_errors_without_exposing_response_details() -> None:
    failed = response(422, {"detail": "Question is required"})
    with patch("frontend.api.client.requests.request", return_value=failed):
        with pytest.raises(APIClientError, match="Question is required"):
            APIClient("http://backend", 4).chat("")


@pytest.mark.parametrize(
    ("status", "message"),
    [
        (503, "The AI provider is temporarily unavailable. Please try again later."),
        (504, "The AI provider did not complete the request in time. Please try again."),
    ],
)
def test_client_maps_provider_errors_to_safe_messages(status, message) -> None:
    failed = response(status, {"detail": "raw provider diagnostic"})
    with patch("frontend.api.client.requests.request", return_value=failed):
        with pytest.raises(APIClientError) as raised:
            APIClient("http://backend", 4).chat("Question")

    assert str(raised.value) == message
    assert "raw provider" not in str(raised.value)


def test_client_handles_timeout_and_connection_failure() -> None:
    with patch(
        "frontend.api.client.requests.request",
        side_effect=requests.Timeout(),
    ):
        with pytest.raises(APIClientError, match="timed out"):
            APIClient("http://backend", 4).health()
    with patch(
        "frontend.api.client.requests.request",
        side_effect=requests.ConnectionError(),
    ):
        with pytest.raises(APIClientError, match="unavailable"):
            APIClient("http://backend", 4).health()


def test_client_handles_delete_without_response_body() -> None:
    with patch("frontend.api.client.requests.request", return_value=response(204)) as request:
        assert APIClient("http://backend", 4).delete_document("doc-1") is None
    request.assert_called_once_with("DELETE", "http://backend/documents/doc-1", timeout=4)


def test_assistant_failed_request_can_retry_without_duplicate_user_message() -> None:
    import frontend.views.assistant as assistant

    class SessionState(dict):
        __getattr__ = dict.__getitem__

        def __setattr__(self, key, value):
            self[key] = value

    class FakeStreamlit:
        def __init__(self):
            self.session_state = SessionState()
            self.questions = iter(["Question", "Question"])
            self.errors = []

        def title(self, value):
            pass

        def caption(self, value):
            pass

        def chat_input(self, value):
            return next(self.questions)

        @contextmanager
        def spinner(self, value):
            yield

        def error(self, value):
            self.errors.append(value)

    streamlit = FakeStreamlit()
    client = Mock()
    client.chat.side_effect = [
        APIClientError("The AI provider did not complete the request in time. Please try again."),
        {"conversation_id": "conversation-1", "answer": "Answer", "citations": [], "insufficient_context": False},
    ]

    with patch.object(assistant, "st", streamlit), patch.object(assistant, "render_message"):
        assistant.render(client)
        assert streamlit.session_state.chat_messages == []
        assistant.render(client)

    assert [message["role"] for message in streamlit.session_state.chat_messages] == [
        "user",
        "assistant",
    ]
    assert streamlit.session_state.chat_messages[0]["content"] == "Question"
    assert streamlit.errors == [
        "The AI provider did not complete the request in time. Please try again."
    ]
