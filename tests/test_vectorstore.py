import json

import numpy as np
import pytest

from backend.core.exceptions import VectorStoreError
from backend.vectorstore.faiss_store import FaissVectorStore


def test_faiss_insert_search_reload_and_document_delete(tmp_path) -> None:
    store = FaissVectorStore(tmp_path, 3)
    vectors = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

    ids = store.add(vectors, ["chunk-a", "chunk-b"], "document-a")
    assert len(ids) == 2
    assert store.search(np.array([1, 0, 0], dtype=np.float32), 1)[0].chunk_id == "chunk-a"

    reloaded = FaissVectorStore(tmp_path, 3)
    assert reloaded.count == 2
    assert reloaded.search(np.array([0, 1, 0], dtype=np.float32), 1)[0].chunk_id == "chunk-b"
    assert reloaded.delete_document("document-a") == ["chunk-a", "chunk-b"]
    assert reloaded.count == 0
    assert reloaded.search(np.array([1, 0, 0], dtype=np.float32)) == []


def test_faiss_duplicate_insert_is_safe_and_empty_index_searches(tmp_path) -> None:
    store = FaissVectorStore(tmp_path, 2)
    vector = np.array([[1, 0]], dtype=np.float32)

    assert store.search(vector[0]) == []
    assert store.add(vector, ["chunk-a"], "document-a")
    assert store.add(vector, ["chunk-a"], "document-a") == []
    assert store.count == 1


def test_faiss_rejects_corrupt_or_wrong_dimension_state(tmp_path) -> None:
    store = FaissVectorStore(tmp_path, 2)
    store.add(np.array([[1, 0]], dtype=np.float32), ["chunk-a"], "document-a")
    store.mapping_path.write_text("{bad json", encoding="utf-8")

    with pytest.raises(VectorStoreError):
        FaissVectorStore(tmp_path, 2)


def test_faiss_rejects_non_normalized_vectors(tmp_path) -> None:
    store = FaissVectorStore(tmp_path, 2)

    with pytest.raises(ValueError):
        store.add(np.array([[2, 0]], dtype=np.float32), ["chunk-a"], "document-a")
