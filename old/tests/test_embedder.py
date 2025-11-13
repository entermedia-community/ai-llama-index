import pytest

from src.embedder.model import EmbeddingModel


def test_embed_shapes():
    # Use a small sentence-transformers model; this test is a smoke test and
    # may be skipped in CI if large downloads are undesired.
    m = EmbeddingModel(model_name="sentence-transformers/all-MiniLM-L6-v2")
    texts = ["hello world", "test"]
    embs = m.embed(texts)
    assert len(embs) == 2
    assert len(embs[0]) == len(embs[1])
