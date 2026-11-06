"""HTTP-only client for the public FastAPI contract."""

from dataclasses import dataclass
from typing import Any

import requests

from frontend.config import backend_base_url, request_timeout


class APIClientError(RuntimeError):
    """Safe, user-facing API client error."""


@dataclass(frozen=True)
class APIClient:
    """Reusable client with centralized routes and timeout behavior."""

    base_url: str = ""
    timeout: float = 0.0

    def __post_init__(self) -> None:
        if not self.base_url:
            object.__setattr__(self, "base_url", backend_base_url())
        if not self.timeout:
            object.__setattr__(self, "timeout", request_timeout())

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = requests.request(
                method,
                f"{self.base_url}{path}",
                timeout=self.timeout,
                **kwargs,
            )
        except requests.Timeout as exc:
            raise APIClientError("The backend request timed out.") from exc
        except requests.RequestException as exc:
            raise APIClientError("The backend is unavailable.") from exc
        if not response.ok:
            try:
                detail = response.json().get("detail", "The backend request failed.")
            except ValueError:
                detail = "The backend request failed."
            raise APIClientError(str(detail))
        if response.status_code == 204:
            return None
        try:
            return response.json()
        except ValueError as exc:
            raise APIClientError("The backend returned an invalid response.") from exc

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def list_documents(self) -> list[dict[str, Any]]:
        return self._request("GET", "/documents")

    def get_document(self, document_id: str) -> dict[str, Any]:
        return self._request("GET", f"/documents/{document_id}")

    def get_document_status(self, document_id: str) -> dict[str, Any]:
        return self._request("GET", f"/documents/{document_id}/status")

    def upload_document(self, filename: str, content: bytes, content_type: str | None) -> dict[str, Any]:
        return self._request(
            "POST",
            "/documents/upload",
            files={"file": (filename, content, content_type or "application/octet-stream")},
        )

    def delete_document(self, document_id: str) -> None:
        self._request("DELETE", f"/documents/{document_id}")

    def chat(self, question: str, conversation_id: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"question": question}
        if conversation_id:
            payload["conversation_id"] = conversation_id
        return self._request("POST", "/chat", json=payload)
