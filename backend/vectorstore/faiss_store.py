"""Persistent FAISS storage for normalized chunk embeddings.

The store uses IndexIDMap2 over IndexFlatIP. Inner product equals cosine
similarity for normalized vectors. Deletion uses FAISS ID removal and rewrites
the index/mapping atomically; this favors correctness over deletion throughput.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Iterable

import faiss
import numpy as np

from backend.core.exceptions import VectorStoreError


@dataclass(frozen=True)
class VectorSearchResult:
    """A nearest-neighbor result mapped back to a chunk."""

    chunk_id: str
    score: float
    faiss_id: int
    document_id: str


class FaissVectorStore:
    """A disk-backed normalized-vector index and chunk mapping."""

    def __init__(
        self,
        directory: Path,
        dimension: int,
        index_filename: str = "chunks.faiss",
        mapping_filename: str = "chunks.mapping.json",
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive.")
        self.directory = directory
        self.dimension = dimension
        self.index_path = directory / index_filename
        self.mapping_path = directory / mapping_filename
        self._lock = RLock()
        self._index = self._new_index()
        self._mapping: dict[int, dict[str, str]] = {}
        self.load()

    def _new_index(self):
        return faiss.IndexIDMap2(faiss.IndexFlatIP(self.dimension))

    @property
    def count(self) -> int:
        return int(self._index.ntotal)

    def load(self) -> None:
        """Load both files, rejecting partial or corrupt persisted state."""
        with self._lock:
            index_exists = self.index_path.exists()
            mapping_exists = self.mapping_path.exists()
            if not index_exists and not mapping_exists:
                return
            if index_exists != mapping_exists:
                raise VectorStoreError("FAISS index and mapping files are inconsistent.")
            try:
                index = faiss.read_index(str(self.index_path))
                mapping_data = json.loads(self.mapping_path.read_text(encoding="utf-8"))
                mapping = {int(key): value for key, value in mapping_data.items()}
            except Exception as exc:
                raise VectorStoreError("FAISS index or mapping file is corrupt.") from exc
            if index.d != self.dimension or index.ntotal != len(mapping):
                raise VectorStoreError("FAISS index dimensions or mapping count are inconsistent.")
            self._index = index
            self._mapping = mapping

    def add(
        self,
        vectors: np.ndarray,
        chunk_ids: Iterable[str],
        document_id: str,
    ) -> list[int]:
        """Insert vectors, skipping already mapped chunks safely."""
        values = self._validate_vectors(vectors)
        chunk_list = list(chunk_ids)
        if len(chunk_list) != len(values):
            raise ValueError("Each vector must have a matching chunk ID.")
        with self._lock:
            existing = {
                value["chunk_id"] for value in self._mapping.values()
            }
            pending = [
                (vector, chunk_id)
                for vector, chunk_id in zip(values, chunk_list)
                if chunk_id not in existing
            ]
            if not pending:
                return []
            next_id = max(self._mapping, default=-1) + 1
            ids = np.arange(next_id, next_id + len(pending), dtype=np.int64)
            self._index.add_with_ids(
                np.ascontiguousarray([item[0] for item in pending], dtype=np.float32),
                ids,
            )
            for faiss_id, (_, chunk_id) in zip(ids.tolist(), pending):
                self._mapping[faiss_id] = {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                }
            self.save()
            return ids.tolist()

    def search(self, vector: np.ndarray, top_k: int = 5) -> list[VectorSearchResult]:
        """Search nearest neighbors, returning an empty list for an empty index."""
        if top_k <= 0:
            return []
        query = self._validate_vectors(np.asarray(vector, dtype=np.float32).reshape(1, -1))
        with self._lock:
            if self._index.ntotal == 0:
                return []
            scores, ids = self._index.search(query, min(top_k, self._index.ntotal))
            return [
                VectorSearchResult(
                    chunk_id=self._mapping[int(faiss_id)]["chunk_id"],
                    score=float(score),
                    faiss_id=int(faiss_id),
                    document_id=self._mapping[int(faiss_id)]["document_id"],
                )
                for score, faiss_id in zip(scores[0], ids[0])
                if int(faiss_id) >= 0
            ]

    def delete_document(self, document_id: str) -> list[str]:
        """Remove all vectors belonging to a document and persist the result."""
        with self._lock:
            ids = [
                faiss_id
                for faiss_id, value in self._mapping.items()
                if value["document_id"] == document_id
            ]
            if not ids:
                return []
            self._index.remove_ids(np.asarray(ids, dtype=np.int64))
            removed_chunks = [self._mapping.pop(faiss_id)["chunk_id"] for faiss_id in ids]
            self.save()
            return removed_chunks

    def get_faiss_id(self, chunk_id: str) -> int | None:
        """Return the persisted vector ID for a chunk, if indexed."""
        with self._lock:
            return next(
                (
                    faiss_id
                    for faiss_id, value in self._mapping.items()
                    if value["chunk_id"] == chunk_id
                ),
                None,
            )

    def save(self) -> None:
        """Persist index and mapping using temporary files and atomic replacement."""
        self.directory.mkdir(parents=True, exist_ok=True)
        index_tmp = self.index_path.with_suffix(".faiss.tmp")
        mapping_tmp = self.mapping_path.with_suffix(".json.tmp")
        try:
            faiss.write_index(self._index, str(index_tmp))
            mapping_tmp.write_text(json.dumps(self._mapping, sort_keys=True), encoding="utf-8")
            os.replace(index_tmp, self.index_path)
            os.replace(mapping_tmp, self.mapping_path)
        except Exception as exc:
            index_tmp.unlink(missing_ok=True)
            mapping_tmp.unlink(missing_ok=True)
            raise VectorStoreError("Unable to persist FAISS index.") from exc

    def _validate_vectors(self, vectors: np.ndarray) -> np.ndarray:
        values = np.asarray(vectors, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError(f"Expected vectors with shape (n, {self.dimension}).")
        if values.shape[0] == 0:
            return np.empty((0, self.dimension), dtype=np.float32)
        if not np.all(np.isfinite(values)):
            raise ValueError("Vectors must contain finite values.")
        norms = np.linalg.norm(values, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4):
            raise ValueError("Vectors must be normalized for cosine similarity.")
        return np.ascontiguousarray(values)
