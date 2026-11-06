"""Hybrid dense, BM25, RRF, and optional cross-encoder retrieval."""

from sqlalchemy.orm import Session

from backend.core.config import Settings
from backend.core.logging import get_logger
from backend.embeddings.service import EmbeddingService
from backend.retrieval.bm25 import BM25Retriever
from backend.retrieval.dense import DenseRetriever
from backend.retrieval.fusion import reciprocal_rank_fusion
from backend.retrieval.models import RetrievalCandidate
from backend.retrieval.reranker import CrossEncoderReranker
from backend.vectorstore.faiss_store import FaissVectorStore

logger = get_logger(__name__)


class RetrievalService:
    """Coordinate all retrieval stages with reusable indexes and models."""

    def __init__(
        self,
        settings: Settings,
        embedding_service: EmbeddingService | None = None,
        vector_store: FaissVectorStore | None = None,
        bm25: BM25Retriever | None = None,
        reranker: CrossEncoderReranker | None = None,
    ):
        self.settings = settings
        embeddings = embedding_service or EmbeddingService(
            settings.embedding_model,
            settings.embedding_dimension,
            settings.embedding_batch_size,
            settings.embedding_device,
        )
        store = vector_store or FaissVectorStore(
            settings.get_vectorstore_path(),
            settings.embedding_dimension,
            settings.vector_index_filename,
            settings.vector_mapping_filename,
        )
        self.dense = DenseRetriever(embeddings, store)
        self.bm25 = bm25 or BM25Retriever(settings.get_vectorstore_path() / settings.bm25_index_filename)
        self.reranker = reranker
        if reranker is None and settings.reranker_enabled:
            self.reranker = CrossEncoderReranker(
                settings.reranker_model,
                batch_size=settings.embedding_batch_size,
                device=settings.embedding_device,
            )

    def retrieve(
        self,
        session: Session,
        query: str,
        *,
        top_k: int | None = None,
        rerank: bool | None = None,
        document_id: str | None = None,
    ) -> list[RetrievalCandidate]:
        if not query.strip():
            return []
        retrieval_k = top_k or self.settings.retrieval_top_k
        dense = self.dense.retrieve(session, query, retrieval_k, document_id)
        sparse = self.bm25.retrieve(session, query, retrieval_k, document_id)
        fused = reciprocal_rank_fusion([dense, sparse], self.settings.rrf_constant)
        candidates = fused[: self.settings.rerank_candidate_count]
        should_rerank = self.settings.reranker_enabled if rerank is None else rerank
        if should_rerank and self.reranker is not None:
            return self.reranker.rerank(query, candidates, self.settings.rerank_top_k)
        return [
            candidate.__class__(**{**candidate.__dict__, "final_rank": index})
            for index, candidate in enumerate(candidates[:retrieval_k], start=1)
        ]
