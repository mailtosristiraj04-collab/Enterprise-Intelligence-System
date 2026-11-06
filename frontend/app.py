"""Streamlit enterprise frontend entry point."""

import sys
from pathlib import Path

import streamlit as st

# Streamlit adds the script directory, rather than the repository root, to
# sys.path when the app is launched as ``streamlit run frontend/app.py``.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.api.client import APIClient
from frontend.components.sidebar import render_sidebar
from frontend.views import assistant, evaluation, knowledge_base, overview, settings


def _load_css() -> None:
    css_path = Path(__file__).parent / "styles" / "theme.css"
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(
        page_title="Enterprise Knowledge Intelligence",
        page_icon=":books:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _load_css()
    client = APIClient()
    page = render_sidebar(client)
    if page == "Overview":
        overview.render(client)
    elif page == "Knowledge Base":
        knowledge_base.render(client)
    elif page == "AI Assistant":
        assistant.render(client)
    elif page == "Evaluation":
        evaluation.render(client)
    else:
        settings.render(client)


if __name__ == "__main__":
    main()
