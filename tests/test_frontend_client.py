"""Contract tests for the HTTP-only Streamlit client."""

from unittest.mock import Mock, patch

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
