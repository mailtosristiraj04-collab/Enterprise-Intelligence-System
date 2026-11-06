import numpy as np
import pytest

from backend.embeddings.service import EmbeddingService


class FakeModel:
    def encode(self, texts, **kwargs):
        vectors = np.zeros((len(texts), 3), dtype=np.float32)
        for index, text in enumerate(texts):
            vectors[index, len(text) % 3] = 1.0
        return vectors


def test_embedding_batch_and_query_are_normalized() -> None:
    service = EmbeddingService(model=FakeModel(), dimension=3)

    vectors = service.embed_documents(["one", "two"])
    query = service.embed_query("three")

    assert vectors.shape == (2, 3)
    assert query.shape == (3,)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert np.isclose(np.linalg.norm(query), 1.0)


def test_embedding_model_is_loaded_once() -> None:
    model = FakeModel()
    service = EmbeddingService(model=model, dimension=3)

    service.embed_documents(["one"])
    service.embed_documents(["two"])

    assert service._model is model


def test_empty_embedding_batch_and_invalid_input() -> None:
    service = EmbeddingService(model=FakeModel(), dimension=3)

    assert service.embed_documents([]).shape == (0, 3)
    with pytest.raises(ValueError):
        service.embed_documents([""])


def test_embedding_dimension_is_validated() -> None:
    class WrongModel(FakeModel):
        def encode(self, texts, **kwargs):
            return np.zeros((len(texts), 2), dtype=np.float32)

    with pytest.raises(ValueError):
        EmbeddingService(model=WrongModel(), dimension=3).embed_documents(["text"])
