"""Knowledge base page."""

import streamlit as st

from frontend.api.client import APIClient, APIClientError
from frontend.components.document_table import render_documents


def render(client: APIClient) -> None:
    st.title("Knowledge base")
    st.caption("Upload source material, monitor indexing, and manage your evidence base.")
    with st.expander("Upload a document", expanded=True):
        st.caption("Supported: PDF, DOC, DOCX, TXT · Maximum size: 50 MB")
        upload = st.file_uploader("Choose a document", type=["pdf", "doc", "docx", "txt"])
        if upload and st.button("Upload and index", type="primary"):
            with st.spinner("Extracting, embedding, and indexing…"):
                try:
                    result = client.upload_document(upload.name, upload.getvalue(), upload.type)
                    st.success(f"{result.get('original_filename', upload.name)} is {result.get('status', 'processed')}.")
                    st.rerun()
                except APIClientError as exc:
                    st.error(str(exc))
    try:
        documents = client.list_documents()
    except APIClientError as exc:
        st.error(str(exc))
        return
    st.subheader("Documents")
    render_documents(documents, lambda document_id: _delete(client, document_id))


def _delete(client: APIClient, document_id: str) -> None:
    try:
        client.delete_document(document_id)
        st.success("Document deleted.")
        st.rerun()
    except APIClientError as exc:
        st.error(str(exc))
