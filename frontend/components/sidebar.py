"""Sidebar navigation and connection status."""

import streamlit as st

from frontend.api.client import APIClient, APIClientError


def render_sidebar(client: APIClient) -> str:
    with st.sidebar:
        st.markdown("## EKI")
        st.caption("Enterprise Knowledge Intelligence")
        page = st.radio(
            "Workspace",
            ["Overview", "Knowledge Base", "AI Assistant", "Evaluation", "Settings"],
            label_visibility="collapsed",
        )
        st.divider()
        try:
            status = client.health().get("status")
            st.success("Backend connected" if status == "ok" else "Backend degraded")
        except APIClientError:
            st.error("Backend unavailable")
        st.caption("Phase 7 frontend")
    return page
