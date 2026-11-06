"""Cached and persistent BM25 retrieval over SQLite chunks."""

from __future__ import annotations

import json
import re
from pathlib import Path
from threading import RLock

from rank_bm25 import BM25Okapi
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.database.models import Chunk
from backend.retrieval.models import RetrievalCandidate


def tokenize(text: str) -> list[str]:
    """Normalize text into simple case-insensitive word tokens."""
    return re.findall(r"\b\w+\b", text.casefold())


class BM25Retriever:
    """Maintain a reusable BM25 index and rebuild it explicitly when needed."""

    def __init__(self, index_path: Path):
        self.index_path = index_path
        self._lock = RLock()
        self._index: BM25Okapi | None = None
        self._chunks: list[dict[str, object]] = []
        self._signature: tuple[int, ...] | None = None
        self._load()

    def rebuild(self, session: Session) -> None:
        rows = session.scalars(select(Chunk).order_by(Chunk.document_id, Chunk.chunk_index)).all()
        self._chunks = [
            {
                "chunk_id": row.id,
                "document_id": row.document_id,
                "text": row.text,
                "page_number": row.page_number,
                "tokens": tokenize(row.text),
            }
            for row in rows
        ]
        self._signature = tuple(hash((item["chunk_id"], item["text"])) for item in self._chunks)
        with self._lock:
            self._index = BM25Okapi([item["tokens"] for item in self._chunks]) if self._chunks else None
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            self.index_path.write_text(json.dumps(self._chunks), encoding="utf-8")

    def _load(self) -> None:
        if not self.index_path.exists():
            return
        try:
            self._chunks = json.loads(self.index_path.read_text(encoding="utf-8"))
            self._index = BM25Okapi([item["tokens"] for item in self._chunks]) if self._chunks else None
        except (OSError, ValueError, TypeError):
            self._chunks = []
            self._index = None

    def retrieve(
        self,
        session: Session,
        query: str,
        top_k: int,
        document_id: str | None = None,
    ) -> list[RetrievalCandidate]:
        if not query.strip() or top_k <= 0:
            return []
        current_ids = tuple(
            row for row in session.scalars(
                select(Chunk.id).order_by(Chunk.document_id, Chunk.chunk_index)
            ).all()
        )
        signature = tuple(hash((item["chunk_id"], item["text"])) for item in self._chunks)
        if self._index is None or signature != self._signature or set(current_ids) != {item["chunk_id"] for item in self._chunks}:
            self.rebuild(session)
        if self._index is None:
            return []
        scores = self._index.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        results: list[RetrievalCandidate] = []
        for index, score in ranked:
            item = self._chunks[index]
            if document_id and item["document_id"] != document_id:
                continue
            results.append(
                RetrievalCandidate(
                    chunk_id=str(item["chunk_id"]),
                    document_id=str(item["document_id"]),
                    text=str(item["text"]),
                    page_number=item["page_number"],
                    bm25_score=float(score),
                    bm25_rank=len(results) + 1,
                )
            )
            if len(results) == top_k:
                break
        return results
