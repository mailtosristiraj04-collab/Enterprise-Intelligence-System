"""Knowledge-base document display."""

from datetime import datetime
from typing import Any

import streamlit as st


def _format_date(value: Any) -> str:
    if not value:
        return "—"
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).strftime("%d %b %Y")
    except ValueError:
        return str(value)


def render_documents(documents: list[dict[str, Any]], on_delete) -> None:
    if not documents:
        st.info("No documents have been indexed yet.")
        return
    for document in documents:
        with st.container(border=True):
            columns = st.columns([3, 1, 1, 1, 1, .7])
            columns[0].markdown(f"**{document.get('original_filename', document.get('filename'))}**")
            columns[0].caption(f"Uploaded {_format_date(document.get('uploaded_at'))}")
            columns[1].metric("Pages", document.get("page_count") or 0)
            columns[2].metric("Chunks", document.get("chunk_count") or 0)
            columns[3].write(str(document.get("file_type", "")).upper())
            columns[4].write(document.get("status", "unknown").title())
            if columns[5].button("Delete", key=f"delete-{document['id']}"):
                on_delete(document["id"])
