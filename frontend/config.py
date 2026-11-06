"""Frontend-only configuration loaded from environment or Streamlit secrets."""

import os


def backend_base_url() -> str:
    """Return the configured backend URL without exposing credentials."""
    try:
        import streamlit as st

        configured = st.secrets.get("BACKEND_URL", "")
    except Exception:
        configured = ""
    return (configured or os.getenv("BACKEND_URL", "http://127.0.0.1:8000")).rstrip("/")


def request_timeout() -> float:
    """Return the frontend HTTP timeout in seconds."""
    return float(os.getenv("BACKEND_TIMEOUT_SECONDS", "30"))


APP_VERSION = "0.1.0"
