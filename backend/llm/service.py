"""Reusable Gemini API client with bounded retry and safe error handling."""

from __future__ import annotations

import time
import re
from threading import RLock
from typing import Any

from backend.core.exceptions import (
    LLMDeadlineExceededError,
    LLMError,
    LLMRateLimitError,
    LLMServiceUnavailableError,
    LLMTimeoutError,
)


def _provider_diagnostic(error: BaseException) -> str:
    """Return sanitized provider diagnostics without credentials or payloads."""
    status_code = (
        getattr(error, "status_code", None)
        or getattr(error, "code", None)
        or getattr(getattr(error, "response", None), "status_code", None)
    )
    message = str(error).strip() or type(error).__name__
    message = re.sub(r"(?i)(authorization|x-goog-api-key|api[-_ ]?key)\s*[:=]\s*\S+", r"\1=[REDACTED]", message)
    message = re.sub(r"(?i)bearer\s+\S+", "Bearer [REDACTED]", message)
    message = re.sub(r"(?i)(key|token|secret)=\S+", r"\1=[REDACTED]", message)
    message = " ".join(message.split())
    if len(message) > 240:
        message = f"{message[:237]}..."
    return f"HTTP status {status_code}: {message}" if status_code else message


def _provider_status_code(error: BaseException) -> int | None:
    """Return a provider HTTP status from SDK or HTTP-style exceptions."""
    status_code = (
        getattr(error, "status_code", None)
        or getattr(error, "code", None)
        or getattr(getattr(error, "response", None), "status_code", None)
    )
    try:
        return int(status_code) if status_code is not None else None
    except (TypeError, ValueError):
        return None


class GeminiService:
    """Generate text through one lazily initialized Google GenAI client."""

    def __init__(
        self,
        api_key: str,
        model_name: str,
        temperature: float = 0.2,
        max_output_tokens: int = 2048,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        client: Any | None = None,
    ) -> None:
        if not api_key and client is None:
            raise LLMError("Gemini API key is not configured.")
        if timeout_seconds <= 0 or max_retries < 0:
            raise ValueError("timeout_seconds must be positive and max_retries non-negative.")
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self._client = client
        self._lock = RLock()

    def _get_client(self) -> Any:
        with self._lock:
            if self._client is None:
                try:
                    from google import genai
                    from google.genai import types

                    self._client = genai.Client(
                        api_key=self.api_key,
                        http_options=types.HttpOptions(timeout=int(self.timeout_seconds * 1000)),
                    )
                except Exception as exc:
                    diagnostic = _provider_diagnostic(exc)
                    raise LLMError(
                        f"Unable to initialize the Gemini client. {diagnostic}"
                    ) from exc
            return self._client

    def generate(self, system_instruction: str, prompt: str) -> str:
        """Generate a non-empty answer with bounded transient-error retries."""
        if not system_instruction.strip() or not prompt.strip():
            raise ValueError("system_instruction and prompt must be non-empty.")
        client = self._get_client()
        try:
            from google.genai import types
        except ImportError as exc:
            raise LLMError("The Google GenAI SDK is not available.") from exc

        for attempt in range(self.max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=self.temperature,
                        max_output_tokens=self.max_output_tokens,
                    ),
                )
                text = getattr(response, "text", None)
                if not isinstance(text, str) or not text.strip():
                    raise LLMError("Gemini returned an empty or malformed response.")
                return text.strip()
            except LLMError:
                raise
            except TimeoutError as exc:
                if attempt >= self.max_retries:
                    raise LLMTimeoutError(_provider_diagnostic(exc)) from exc
            except Exception as exc:
                status_code = _provider_status_code(exc)
                if status_code == 504:
                    raise LLMDeadlineExceededError() from exc
                if status_code == 503:
                    if attempt >= self.max_retries:
                        raise LLMServiceUnavailableError(_provider_diagnostic(exc)) from exc
                elif status_code == 429 or "rate" in type(exc).__name__.lower():
                    if attempt >= self.max_retries:
                        raise LLMRateLimitError(_provider_diagnostic(exc)) from exc
                else:
                    raise LLMError(
                        f"The language model request failed. {_provider_diagnostic(exc)}"
                    ) from exc
            time.sleep(2**attempt)
        raise LLMError("The language model request failed.")
