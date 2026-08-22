"""Semantic cache tests: exact TTL store + semantic lookup w/ privacy."""

import numpy as np
import pytest

from app.cache.semantic_cache import SemanticCache


class FakeEmbedder:
    """Deterministic word-hash embeddings: shared words -> shared vector.

    Identical text -> identical embedding (cosine 1.0); texts with no
    shared words -> cosine 0."""

    BUCKETS = 16

    def _vec(self, text: str) -> np.ndarray:
        vec = np.zeros(self.BUCKETS, dtype="float32")
        for word in text.lower().split():
            vec[sum(ord(c) for c in word) % self.BUCKETS] += 1.0
        return vec

    def encode(self, texts, normalize=True):
        vecs = np.vstack([self._vec(t) for t in texts])
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


def make_cache(tmp_path):
    cache = SemanticCache(cache_path=tmp_path / "cache", threshold=0.5)
    cache._get_embedder = lambda: FakeEmbedder()
    cache.embedder = FakeEmbedder()
    return cache


def test_exact_roundtrip(tmp_path):
    c = make_cache(tmp_path)
    assert c.exact_get("hello") is None
    c.exact_put("hello", "hi back")
    assert c.exact_get("hello") == "hi back"


def test_semantic_hit_and_miss(tmp_path):
    c = make_cache(tmp_path)
    c.semantic_put("what is the capital of france", "paris")
    # same text -> identical embedding -> hit above threshold
    assert c.semantic_get("what is the capital of france") == "paris"
    # different text -> miss
    assert c.semantic_get("compute 2 plus 2") is None


def test_lookup_bypasses_for_private(tmp_path):
    c = make_cache(tmp_path)
    c.semantic_put("hot query", "cached answer")
    assert c.lookup("hot query", policy="private", private=True, offline_only=False) is None
    assert c.lookup("hot query", policy="ephemeral", private=False, offline_only=False) is None
    assert c.lookup("hot query", policy="restricted", private=False, offline_only=False) is None
    assert c.lookup("hot query", policy="public", private=False, offline_only=False) == "cached answer"


@pytest.mark.skipif(True, reason="sentence-transformers download not needed for unit tests")
def test_transformer_embedder_shape():
    from app.rag.embeddings import get_embedder

    emb = get_embedder("sentence-transformers/all-MiniLM-L6-v2")
    vec = emb.encode(["hello world"])
    assert vec.shape[1] >= 64