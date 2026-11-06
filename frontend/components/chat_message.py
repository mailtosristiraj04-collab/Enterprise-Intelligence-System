"""Assistant conversation message rendering."""

import streamlit as st

from frontend.components.citation_view import render_citations


def render_message(message: dict[str, object]) -> None:
    role = message.get("role", "assistant")
    with st.chat_message("user" if role == "user" else "assistant"):
        if message.get("answer_mode") == "extractive_fallback":
            st.caption(
                "AI generation is temporarily unavailable. Showing relevant information "
                "directly from your documents."
            )
        if message.get("insufficient_context"):
            st.warning("There is not enough indexed evidence to answer this question.")
        st.markdown(str(message.get("content", "")))
        render_citations(message.get("citations", []))
