"""Overview page."""

from typing import Any

import streamlit as st

from frontend.api.client import APIClient, APIClientError
from frontend.components.metrics import metric_cards
from frontend.components.document_table import _format_date


def render(client: APIClient) -> None:
    st.markdown('<div class="eki-brand">Knowledge intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="eki-subtitle">A governed view of your indexed enterprise knowledge.</div>', unsafe_allow_html=True)
    try:
        documents: list[dict[str, Any]] = client.list_documents()
    except APIClientError as exc:
        st.error(str(exc))
        return
    indexed = [item for item in documents if item.get("status") == "completed"]
    failed = [item for item in documents if item.get("status") == "failed"]
    chunks = sum(int(item.get("chunk_count") or 0) for item in indexed)
    metric_cards([("Total documents", len(documents)), ("Indexed", len(indexed)), ("Failed", len(failed)), ("Indexed chunks", chunks)])
    st.divider()
    st.subheader("Recent activity")
    if not documents:
        st.info("Your knowledge base is empty. Upload a document to begin.")
    else:
        for document in documents[:5]:
            st.markdown(
                f"**{document.get('original_filename', document.get('filename'))}** · "
                f"{document.get('status', 'unknown').title()} · {_format_date(document.get('uploaded_at'))}"
            )
