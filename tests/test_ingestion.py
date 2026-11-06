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
from backend.database.models import Chunk, Document, ProcessingStatus
from backend.retrieval.bm25 import BM25Retriever
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


class FailingEmbeddingService(FakeEmbeddingService):
    def embed_documents(self, texts: list[str]):
        raise RuntimeError("embedding failed")


class FailingVectorStore:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def add(self, vectors, chunk_ids, document_id):
        raise RuntimeError("FAISS insertion failed")

    def delete_document(self, document_id):
        return self.vector_store.delete_document(document_id)


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


@pytest.mark.parametrize("failure_stage", ["embedding", "faiss"])
def test_failed_ingestion_removes_sqlite_file_and_bm25_state(
    tmp_path: Path, failure_stage: str
) -> None:
    settings = Settings(project_root=tmp_path)
    database = _test_database(settings)
    session = database.get_session()
    try:
        from backend.vectorstore.faiss_store import FaissVectorStore

        vector_store = FaissVectorStore(settings.get_vectorstore_path(), settings.embedding_dimension)
        embeddings = (
            FailingEmbeddingService(settings.embedding_dimension)
            if failure_stage == "embedding"
            else FakeEmbeddingService(settings.embedding_dimension)
        )
        failing_store = (
            vector_store
            if failure_stage == "embedding"
            else FailingVectorStore(vector_store)
        )
        service = IngestionService(settings, embeddings, failing_store)

        with pytest.raises(RuntimeError, match="failed"):
            service.ingest(session, "failed.txt", b"failed ingestion content")

        failed = session.query(Document).one()
        assert failed.status == ProcessingStatus.FAILED.value
        expected_error = (
            "embedding failed" if failure_stage == "embedding" else "FAISS insertion failed"
        )
        assert failed.error_message == expected_error
        assert session.query(Chunk).count() == 0
        assert not list(settings.get_upload_path().glob("*"))
        bm25 = BM25Retriever(settings.get_vectorstore_path() / settings.bm25_index_filename)
        assert bm25.retrieve(session, "failed ingestion", 5) == []
    finally:
        session.close()


def test_failed_document_is_excluded_from_dense_and_bm25_retrieval(tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path)
    database = _test_database(settings)
    session = database.get_session()
    try:
        from backend.retrieval.dense import DenseRetriever
        from backend.retrieval.models import RetrievalCandidate
        from backend.vectorstore.faiss_store import FaissVectorStore

        failed = Document(
            id="failed-document",
            filename="failed.txt",
            original_filename="failed.txt",
            file_type=".txt",
            file_size=10,
            file_hash="f" * 64,
            stored_file_path="",
            status=ProcessingStatus.FAILED.value,
        )
        chunk = Chunk(
            id="failed-chunk",
            document_id=failed.id,
            chunk_index=0,
            text="failed ingestion content",
            char_count=24,
            token_estimate=3,
        )
        session.add_all([failed, chunk])
        session.commit()
        store = FaissVectorStore(settings.get_vectorstore_path(), 2)
        import numpy as np

        store.add(np.array([[1.0, 0.0]], dtype=np.float32), [chunk.id], failed.id)
        bm25 = BM25Retriever(settings.get_vectorstore_path() / "failed.bm25.json")
        bm25.rebuild(session)
        assert bm25.retrieve(session, "failed ingestion", 5) == []
        dense = DenseRetriever(
            type("Embeddings", (), {"embed_query": lambda self, query: np.array([1.0, 0.0], dtype=np.float32)})(),
            store,
        )
        assert dense.retrieve(session, "failed ingestion", 5) == []
    finally:
        session.close()


def test_bm25_cleanup_failure_preserves_original_error_and_removes_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(project_root=tmp_path)
    database = _test_database(settings)
    session = database.get_session()
    bm25_path = settings.get_vectorstore_path() / settings.bm25_index_filename
    bm25_path.parent.mkdir(parents=True, exist_ok=True)
    bm25_path.write_text(
        '[{"chunk_id":"stale-chunk","document_id":"failed-document",'
        '"text":"failed ingestion content","page_number":null,'
        '"tokens":["failed","ingestion","content"]}]',
        encoding="utf-8",
    )
    try:
        from backend.vectorstore.faiss_store import FaissVectorStore

        vector_store = FaissVectorStore(settings.get_vectorstore_path(), settings.embedding_dimension)
        service = IngestionService(
            settings,
            FailingEmbeddingService(settings.embedding_dimension),
            vector_store,
        )

        def fail_rebuild(self, cleanup_session):
            raise RuntimeError("BM25 rebuild failed")

        monkeypatch.setattr(BM25Retriever, "rebuild", fail_rebuild)
        with pytest.raises(RuntimeError, match="embedding failed"):
            service.ingest(session, "failed.txt", b"failed ingestion content")

        failed = session.query(Document).one()
        assert failed.status == ProcessingStatus.FAILED.value
        assert failed.error_message == "embedding failed"
        assert session.query(Chunk).count() == 0
        assert not list(settings.get_upload_path().glob("*"))
    finally:
        session.close()

    monkeypatch.undo()
    stale_bm25 = BM25Retriever(bm25_path)
    cleanup_session = database.get_session()
    try:
        assert stale_bm25.retrieve(cleanup_session, "failed ingestion", 5) == []
    finally:
        cleanup_session.close()


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
