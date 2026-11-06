"""Evaluation page with an honest empty state until backend metrics exist."""

import streamlit as st


def render(_client) -> None:
    st.title("Evaluation")
    st.caption("Monitor retrieval and answer quality when evaluation runs are available.")
    st.info("No evaluation runs are available yet. Metrics will appear here after evaluation data is exposed by the backend.")
