"""Standalone Gemini API connectivity smoke test.

Run from the repository root with:
    python scripts/test_gemini_connection.py
"""

from dataclasses import dataclass
from pathlib import Path
import socket
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.config import Settings, get_settings
from backend.core.exceptions import LLMError, LLMRateLimitError, LLMTimeoutError
from backend.llm.service import GeminiService


SMOKE_TEST_PROMPT = "Reply with exactly: GEMINI_CONNECTION_SUCCESS"


@dataclass(frozen=True)
class SmokeTestResult:
    """Safe smoke-test output; never contains the configured API key."""

    api_key_configured: bool
    request_succeeded: bool
    model_name: str
    response_text: str | None = None
    error_category: str | None = None
    error_message: str | None = None


def _error_category(error: BaseException) -> str:
    """Map SDK/service failures to stable, non-sensitive categories."""
    if isinstance(error, LLMRateLimitError):
        return "quota_or_rate_limit"
    if isinstance(error, LLMTimeoutError) or isinstance(error, TimeoutError):
        return "timeout"

    cause = error.__cause__
    status_code = (
        getattr(cause, "status_code", None)
        or getattr(cause, "code", None)
        or getattr(getattr(cause, "response", None), "status_code", None)
    )
    if status_code in {401, 403}:
        return "authentication"
    if status_code == 429:
        return "quota_or_rate_limit"

    error_name = type(cause or error).__name__.casefold()
    if any(term in error_name for term in ("auth", "permission", "unauthorized", "forbidden")):
        return "authentication"
    cause_text = str(cause or "").casefold()
    if any(
        term in cause_text
        for term in ("api key", "invalid key", "unauthorized", "permission denied", "forbidden")
    ):
        return "authentication"
    if any(term in error_name for term in ("timeout", "timedout")):
        return "timeout"
    if any(term in error_name for term in ("connection", "network", "dns", "socket", "http")):
        return "network"
    if isinstance(cause, (ConnectionError, OSError, socket.gaierror)):
        return "network"
    if isinstance(error, LLMError):
        return "service_error"
    return "unknown"


def run_smoke_test(
    settings: Settings | None = None,
    service: GeminiService | None = None,
) -> SmokeTestResult:
    """Send exactly one minimal request using the existing Gemini service."""
    active_settings = settings or get_settings()
    configured = bool(active_settings.gemini_api_key.strip())
    model_name = active_settings.gemini_model
    if not configured:
        return SmokeTestResult(
            api_key_configured=False,
            request_succeeded=False,
            model_name=model_name,
            error_category="configuration",
            error_message="GEMINI_API_KEY is not configured.",
        )

    active_service = service or GeminiService(
        api_key=active_settings.gemini_api_key,
        model_name=active_settings.gemini_model,
        temperature=active_settings.gemini_temperature,
        max_output_tokens=active_settings.gemini_max_output_tokens,
        timeout_seconds=active_settings.gemini_timeout_seconds,
        max_retries=active_settings.gemini_max_retries,
    )
    try:
        response = active_service.generate(SMOKE_TEST_PROMPT, SMOKE_TEST_PROMPT)
    except (LLMError, TimeoutError, ConnectionError, OSError) as exc:
        return SmokeTestResult(
            api_key_configured=True,
            request_succeeded=False,
            model_name=model_name,
            error_category=_error_category(exc),
            error_message=str(exc),
        )
    except Exception as exc:
        return SmokeTestResult(
            api_key_configured=True,
            request_succeeded=False,
            model_name=model_name,
            error_category=_error_category(exc),
            error_message="Unexpected Gemini smoke-test failure.",
        )
    return SmokeTestResult(
        api_key_configured=True,
        request_succeeded=True,
        model_name=model_name,
        response_text=response,
    )


def main() -> int:
    """Print safe smoke-test results and return a process status."""
    result = run_smoke_test()
    print(f"API key configured: {result.api_key_configured}")
    print(f"API request succeeded: {result.request_succeeded}")
    print(f"Model name: {result.model_name}")
    print(f"Response text: {result.response_text or '(none)'}")
    if result.error_category:
        print(f"Error category: {result.error_category}")
        print(f"Error: {result.error_message}")
    return 0 if result.request_succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
