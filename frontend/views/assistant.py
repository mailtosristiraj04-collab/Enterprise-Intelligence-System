"""AI Assistant page."""

import streamlit as st

from frontend.api.client import APIClient, APIClientError
from frontend.components.chat_message import render_message


def render(client: APIClient) -> None:
    st.title("AI assistant")
    st.caption("Ask questions grounded in your indexed documents.")
    st.session_state.setdefault("conversation_id", None)
    st.session_state.setdefault("chat_messages", [])
    for message in st.session_state.chat_messages:
        render_message(message)
    question = st.chat_input("Ask about your enterprise knowledge")
    if not question:
        return
    st.session_state.chat_messages.append({"role": "user", "content": question})
    render_message(st.session_state.chat_messages[-1])
    with st.spinner("Searching the knowledge base…"):
        try:
            response = client.chat(question, st.session_state.conversation_id)
            st.session_state.conversation_id = response.get("conversation_id")
            assistant_message = {
                "role": "assistant",
                "content": response.get("answer", ""),
                "citations": response.get("citations", []),
                "insufficient_context": response.get("insufficient_context", False),
            }
            st.session_state.chat_messages.append(assistant_message)
            render_message(assistant_message)
        except APIClientError as exc:
            st.error(str(exc))
