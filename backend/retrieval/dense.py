"""Dense retrieval over the persistent FAISS vector store."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.database.models import Chunk
from backend.embeddings.service import EmbeddingService
from backend.retrieval.models import RetrievalCandidate
from backend.vectorstore.faiss_store import FaissVectorStore


class DenseRetriever:
    """Embed a query, search FAISS, and hydrate chunk metadata from SQLite."""

    def __init__(self, embeddings: EmbeddingService, vector_store: FaissVectorStore):
        self.embeddings = embeddings
        self.vector_store = vector_store

    def retrieve(
        self,
        session: Session,
        query: str,
        top_k: int,
        document_id: str | None = None,
    ) -> list[RetrievalCandidate]:
        if not query.strip() or top_k <= 0:
            return []
        search_k = self.vector_store.count if document_id else top_k * 2
        results = self.vector_store.search(self.embeddings.embed_query(query), search_k)
        ids = [result.chunk_id for result in results]
        if not ids:
            return []
        rows = session.scalars(select(Chunk).where(Chunk.id.in_(ids))).all()
        by_id = {row.id: row for row in rows}
        candidates: list[RetrievalCandidate] = []
        for result in results:
            chunk = by_id.get(result.chunk_id)
            if chunk is None or (document_id and chunk.document_id != document_id):
                continue
            candidates.append(
                RetrievalCandidate(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    text=chunk.text,
                    page_number=chunk.page_number,
                    dense_score=result.score,
                    dense_rank=len(candidates) + 1,
                    metadata={"faiss_id": result.faiss_id},
                )
            )
            if len(candidates) == top_k:
                break
        return candidates
