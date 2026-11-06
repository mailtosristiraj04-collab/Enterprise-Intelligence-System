"""Citation rendering."""

import streamlit as st


def render_citations(citations: list[dict[str, object]]) -> None:
    if not citations:
        return
    st.markdown("**Sources**")
    for index, citation in enumerate(citations, start=1):
        page = citation.get("page_number")
        label = f"{citation.get('document_name', 'Document')} · Page {page}" if page else str(
            citation.get("document_name", "Document")
        )
        with st.expander(f"Source {index} · {label}"):
            st.caption(f"Document ID: {citation.get('document_id')} · Chunk ID: {citation.get('chunk_id')}")
            st.write(citation.get("excerpt", ""))
