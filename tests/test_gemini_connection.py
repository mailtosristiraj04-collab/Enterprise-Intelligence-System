"""Mocked tests for the standalone Gemini connectivity smoke test."""

from backend.core.config import Settings
from backend.core.exceptions import LLMError, LLMRateLimitError, LLMTimeoutError
from scripts.test_gemini_connection import (
    SMOKE_TEST_PROMPT,
    run_smoke_test,
)


class FakeService:
    def __init__(self, response: str = "GEMINI_CONNECTION_SUCCESS", error=None):
        self.response = response
        self.error = error
        self.calls = []

    def generate(self, system_instruction: str, prompt: str) -> str:
        self.calls.append((system_instruction, prompt))
        if self.error:
            raise self.error
        return self.response


def _settings() -> Settings:
    return Settings(gemini_api_key="test-key", gemini_model="gemini-test")


def test_loaded_default_model_is_the_migrated_model() -> None:
    settings = Settings(_env_file=None, gemini_api_key="test-key")

    assert settings.gemini_model == "gemini-3.8-flash"


def test_smoke_test_reports_success_without_calling_gemini() -> None:
    service = FakeService()
    result = run_smoke_test(_settings(), service)

    assert result.api_key_configured is True
    assert result.request_succeeded is True
    assert result.model_name == "gemini-test"
    assert result.response_text == "GEMINI_CONNECTION_SUCCESS"
    assert service.calls == [(SMOKE_TEST_PROMPT, SMOKE_TEST_PROMPT)]


def test_smoke_test_reports_missing_key_without_creating_service() -> None:
    result = run_smoke_test(Settings(gemini_api_key="", gemini_model="gemini-test"))

    assert result.api_key_configured is False
    assert result.request_succeeded is False
    assert result.error_category == "configuration"


def test_smoke_test_classifies_authentication_failure() -> None:
    class AuthError(Exception):
        status_code = 401

    error = LLMError("The language model request failed.")
    error.__cause__ = AuthError()
    result = run_smoke_test(_settings(), FakeService(error=error))

    assert result.request_succeeded is False
    assert result.error_category == "authentication"
    assert "test-key" not in (result.error_message or "")


def test_smoke_test_classifies_rate_limit_and_timeout() -> None:
    rate_result = run_smoke_test(_settings(), FakeService(error=LLMRateLimitError()))
    timeout_result = run_smoke_test(_settings(), FakeService(error=LLMTimeoutError()))

    assert rate_result.error_category == "quota_or_rate_limit"
    assert timeout_result.error_category == "timeout"


def test_smoke_test_classifies_network_failure() -> None:
    result = run_smoke_test(_settings(), FakeService(error=ConnectionError("offline")))

    assert result.request_succeeded is False
    assert result.error_category == "network"
