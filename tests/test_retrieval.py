import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.core.config import Settings
from backend.database.models import Base, Chunk, Document, ProcessingStatus
from backend.retrieval.bm25 import BM25Retriever
from backend.retrieval.fusion import reciprocal_rank_fusion
from backend.retrieval.models import RetrievalCandidate
from backend.retrieval.reranker import CrossEncoderReranker
from backend.retrieval.service import RetrievalService
from backend.vectorstore.faiss_store import FaissVectorStore


class FakeEmbeddings:
    def embed_query(self, query: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)


class FakeReranker:
    def rerank(self, query, candidates, top_n):
        return [
            candidate.__class__(
                **{**candidate.__dict__, "cross_encoder_score": float(len(candidate.text)), "final_rank": index}
            )
            for index, candidate in enumerate(
                sorted(candidates, key=lambda candidate: len(candidate.text), reverse=True)[:top_n],
                start=1,
            )
        ]


class FakeCrossEncoder:
    def predict(self, pairs, **kwargs):
        return np.arange(len(pairs), dtype=np.float32)


def _session(tmp_path):
    settings = Settings(project_root=tmp_path)
    settings.ensure_directories()
    engine = create_engine(settings.get_database_url(), connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    document = Document(
        filename="knowledge.txt",
        original_filename="knowledge.txt",
        file_type=".txt",
        file_size=10,
        file_hash="a" * 64,
        stored_file_path=str(tmp_path / "knowledge.txt"),
        status="completed",
    )
    session.add(document)
    session.flush()
    chunks = [
        Chunk(
            document_id=document.id,
            chunk_index=0,
            text="Python database retrieval and indexing",
            char_count=40,
            token_estimate=6,
        ),
        Chunk(
            document_id=document.id,
            chunk_index=1,
            text="Unrelated operational notes",
            char_count=28,
            token_estimate=4,
        ),
    ]
    session.add_all(chunks)
    session.commit()
    return settings, session, document, chunks


def test_bm25_retrieval_rebuild_and_deleted_document_exclusion(tmp_path):
    settings, session, document, chunks = _session(tmp_path)
    bm25 = BM25Retriever(settings.get_vectorstore_path() / "bm25.json")

    bm25.rebuild(session)
    results = bm25.retrieve(session, "database retrieval", 5)
    assert results[0].chunk_id == chunks[0].id
    session.delete(document)
    session.commit()
    bm25.rebuild(session)
    assert bm25.retrieve(session, "database retrieval", 5) == []
    session.close()


def test_dense_retrieval_filter_and_empty_query(tmp_path):
    settings, session, document, chunks = _session(tmp_path)
    store = FaissVectorStore(settings.get_vectorstore_path(), 2)
    store.add(np.array([[1, 0], [0, 1]], dtype=np.float32), [chunk.id for chunk in chunks], document.id)
    from backend.retrieval.dense import DenseRetriever

    retriever = DenseRetriever(FakeEmbeddings(), store)
    assert retriever.retrieve(session, "database", 5)[0].chunk_id == chunks[0].id
    assert retriever.retrieve(session, "", 5) == []
    assert retriever.retrieve(session, "database", 5, "missing") == []
    session.close()


def test_dense_retrieval_empty_index_returns_no_results(tmp_path):
    settings, session, _, _ = _session(tmp_path)
    store = FaissVectorStore(settings.get_vectorstore_path(), 2)
    from backend.retrieval.dense import DenseRetriever

    retriever = DenseRetriever(FakeEmbeddings(), store)

    assert store.count == 0
    assert retriever.retrieve(session, "database", 5) == []
    session.close()


def test_dense_retrieval_skips_many_stale_vectors_without_shortening_top_k(tmp_path):
    settings, session, document, chunks = _session(tmp_path)
    stale_documents = []
    stale_chunks = []
    for index in range(10):
        stale_document = Document(
            id=f"failed-{index}",
            filename=f"failed-{index}.txt",
            original_filename=f"failed-{index}.txt",
            file_type=".txt",
            file_size=10,
            file_hash=f"{index + 1:064d}",
            stored_file_path="",
            status=ProcessingStatus.FAILED.value,
        )
        stale_documents.append(stale_document)
        stale_chunks.append(
            Chunk(
                id=f"failed-chunk-{index}",
                document_id=stale_document.id,
                chunk_index=0,
                text=f"failed stale content {index}",
                char_count=22,
                token_estimate=4,
            )
        )
    session.add_all(stale_documents + stale_chunks)
    session.commit()

    store = FaissVectorStore(settings.get_vectorstore_path(), 2)
    stale_vectors = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (10, 1))
    valid_vectors = np.array([[0.99, 0.141067], [0.97, 0.2431]], dtype=np.float32)
    valid_vectors /= np.linalg.norm(valid_vectors, axis=1, keepdims=True)
    store.add(
        np.vstack([stale_vectors, valid_vectors]),
        [item.id for item in stale_chunks] + [item.id for item in chunks],
        "mixed",
    )
    from backend.retrieval.dense import DenseRetriever

    retriever = DenseRetriever(FakeEmbeddings(), store)
    results = retriever.retrieve(session, "database", 2)

    assert [result.chunk_id for result in results] == [chunks[0].id, chunks[1].id]
    assert all(result.document_id == document.id for result in results)
    session.close()


def test_rrf_merges_duplicates_and_preserves_source_ranks():
    first = RetrievalCandidate("a", "d", "alpha", 1, dense_score=0.9, dense_rank=1)
    second = RetrievalCandidate("a", "d", "alpha", 1, bm25_score=4.0, bm25_rank=2)
    merged = reciprocal_rank_fusion([[first], [second]], constant=10)

    assert len(merged) == 1
    assert merged[0].dense_rank == 1
    assert merged[0].bm25_rank == 2
    assert merged[0].rrf_score == 1 / 11 + 1 / 12


def test_cross_encoder_reranking_is_batched_and_ranked():
    candidates = [
        RetrievalCandidate("a", "d", "short", None),
        RetrievalCandidate("b", "d", "longer text", None),
    ]
    result = CrossEncoderReranker("fake", model=FakeCrossEncoder()).rerank("query", candidates, 2)

    assert [item.chunk_id for item in result] == ["b", "a"]
    assert result[0].final_rank == 1


def test_complete_hybrid_pipeline_without_reranker(tmp_path):
    settings, session, document, chunks = _session(tmp_path)
    settings.reranker_enabled = False
    store = FaissVectorStore(settings.get_vectorstore_path(), 2)
    store.add(np.array([[1, 0], [0, 1]], dtype=np.float32), [chunk.id for chunk in chunks], document.id)
    service = RetrievalService(
        settings,
        embedding_service=FakeEmbeddings(),
        vector_store=store,
        bm25=BM25Retriever(settings.get_vectorstore_path() / "bm25.json"),
    )
    results = service.retrieve(session, "database retrieval", top_k=2, rerank=False)

    assert results
    assert results[0].final_rank == 1
    assert results[0].dense_score is not None
    assert results[0].bm25_score is not None
    assert results[0].rrf_score is not None
    session.close()
