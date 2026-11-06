import io
from pathlib import Path

import fitz
import pytest
from docx import Document as DocxDocument
from fastapi.testclient import TestClient

from backend.api.app import create_app
from backend.core.config import Settings
from backend.core.exceptions import ExtractionDependencyError
from backend.api.dependencies import get_db_session, get_ingestion_service
from backend.database.database import Database
from backend.database.models import Chunk, Document
from backend.ingestion.chunking import chunk_pages
from backend.ingestion.extractors import DocExtractor, DocxExtractor, PdfExtractor, TxtExtractor
from backend.ingestion.preprocessing import clean_text
from backend.ingestion.service import IngestionService


class FakeEmbeddingService:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def embed_documents(self, texts: list[str]):
        import numpy as np

        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        vectors[:, 0] = 1.0
        return vectors


def _pdf_bytes() -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "A PDF paragraph on page one.")
    content = document.tobytes()
    document.close()
    return content


def _docx_bytes() -> bytes:
    document = DocxDocument()
    document.add_paragraph("A DOCX paragraph.")
    output = io.BytesIO()
    document.save(output)
    return output.getvalue()


def test_pdf_extraction_preserves_page_metadata() -> None:
    result = PdfExtractor().extract(_pdf_bytes(), "sample.pdf")

    assert result.pages[0].page_number == 1
    assert "page one" in result.text
    assert result.metadata["page_count"] == 1


def test_docx_extraction() -> None:
    result = DocxExtractor().extract(_docx_bytes(), "sample.docx")

    assert "DOCX paragraph" in result.text


def test_txt_extraction_fallback_encoding() -> None:
    result = TxtExtractor().extract("café".encode("cp1252"), "sample.txt")

    assert result.text == "café"
    assert result.warnings


def test_doc_extraction_fails_gracefully_when_converter_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "backend.ingestion.extractors.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError),
    )
    with pytest.raises(ExtractionDependencyError):
        DocExtractor().extract(b"legacy", "sample.doc")


def test_clean_text_preserves_paragraphs() -> None:
    assert clean_text(" first \r\n\r\n second\x00\n\n\nthird ") == "first\n\nsecond\n\nthird"


def test_chunking_has_order_overlap_and_page_metadata() -> None:
    chunks = chunk_pages(
        [("Sentence one. " * 100, 2), ("Next page content.", 3)],
        chunk_size_tokens=20,
        overlap_tokens=5,
    )

    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert chunks[0].page_number == 2
    assert chunks[1].text[-20:] in chunks[0].text


def _test_database(settings: Settings) -> Database:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    settings.ensure_directories()
    database = Database.__new__(Database)
    database.engine = create_engine(settings.get_database_url(), connect_args={"check_same_thread": False})
    database.SessionLocal = sessionmaker(bind=database.engine)
    database.create_tables()
    return database


def test_ingestion_persists_chunks_and_deletion_removes_file(tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path)
    database = _test_database(settings)
    session = database.get_session()
    try:
        vector_store = __import__(
            "backend.vectorstore.faiss_store", fromlist=["FaissVectorStore"]
        ).FaissVectorStore(settings.get_vectorstore_path(), settings.embedding_dimension)
        service = IngestionService(settings, FakeEmbeddingService(settings.embedding_dimension), vector_store)
        document = service.ingest(session, "sample.txt", b"first paragraph.\n\nsecond paragraph.")
        stored_path = Path(document.stored_file_path)
        assert session.query(Chunk).filter_by(document_id=document.id).count() > 0
        assert stored_path.exists()
        service.delete(session, document.id)
        assert session.get(Document, document.id) is None
        assert not stored_path.exists()
    finally:
        session.close()


def test_api_upload_duplicate_and_missing_document(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(project_root=tmp_path)
    database = _test_database(settings)
    vector_store = __import__(
        "backend.vectorstore.faiss_store", fromlist=["FaissVectorStore"]
    ).FaissVectorStore(settings.get_vectorstore_path(), settings.embedding_dimension)
    ingestion = IngestionService(
        settings, FakeEmbeddingService(settings.embedding_dimension), vector_store
    )

    def session_override():
        session = database.get_session()
        try:
            yield session
        finally:
            session.close()

    app = create_app()
    app.dependency_overrides[get_db_session] = session_override
    app.dependency_overrides[get_ingestion_service] = lambda: ingestion
    with TestClient(app) as client:
        response = client.post("/documents/upload", files={"file": ("sample.txt", b"hello", "text/plain")})
        duplicate = client.post("/documents/upload", files={"file": ("again.txt", b"hello", "text/plain")})
        missing = client.get("/documents/not-found")

    assert response.status_code == 201
    assert response.json()["chunk_count"] == 1
    assert duplicate.status_code == 409
    assert missing.status_code == 404
