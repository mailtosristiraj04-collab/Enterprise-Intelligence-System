"""Safe frontend settings page."""

import streamlit as st

from frontend.config import APP_VERSION, backend_base_url, request_timeout
from frontend.api.client import APIClient, APIClientError


def render(client: APIClient) -> None:
    st.title("Settings")
    st.caption("Connection and application configuration visible to the frontend.")
    st.markdown(f"**Application version:** {APP_VERSION}")
    st.markdown(f"**Backend URL:** `{backend_base_url()}`")
    st.markdown(f"**Request timeout:** `{request_timeout():.0f} seconds`")
    try:
        health = client.health()
        st.success(f"Backend status: {health.get('status', 'unknown')}")
    except APIClientError as exc:
        st.error(str(exc))
    st.caption("Secrets and API keys are intentionally not displayed.")
