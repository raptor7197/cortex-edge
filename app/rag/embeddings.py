"""Embedding providers for RAG + semantic cache.

Preferred: sentence-transformers (all-MiniLM-L6-v2, 384-dim) — the
PLAN's embedding model. Fallback: a zero-dependency deterministic
hashing embedder (512-dim bag-of-words with TF-IDF weighting) so the
pipeline runs even before the transformer model is downloaded.
"""

import hashlib
import math
import re
from functools import lru_cache

import numpy as np

HASH_DIM = 512


@lru_cache(maxsize=1)
def transformer_available() -> bool:
    """True if sentence-transformers (and its torch dependency) imports
    cleanly. Broken/CUDA-only torch builds raise at import time, so we
    catch everything and fall back to the hashing embedder."""
    try:
        import sentence_transformers  # noqa: F401

        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def faiss_available() -> bool:
    """True if the FAISS shared library loads (catches GPU/CUDA
    misconfiguration that raises at import time)."""
    try:
        import faiss  # noqa: F401

        return True
    except Exception:
        return False


class HashEmbedder:
    """Deterministic hashing embedder: token hashes -> sparse 512-dim
    vector with TF-IDF-style weighting. Good enough for dev/testing,
    not a quality replacement for sentence-transformers."""

    name = "hash-512"

    def __init__(self):
        self._doc_freq: dict[int, int] = {}
        self._doc_count = 0

    def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        vectors = np.zeros((len(texts), HASH_DIM), dtype="float32")
        for i, text in enumerate(texts):
            tokens = re.findall(r"\b[a-z0-9]+\b", text.lower())
            counts: dict[int, int] = {}
            for tok in tokens:
                h = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16) % HASH_DIM
                counts[h] = counts.get(h, 0) + 1
            self._doc_count += 1
            for h, c in counts.items():
                self._doc_freq[h] = self._doc_freq.get(h, 0) + 1
            for h, c in counts.items():
                idf = math.log((self._doc_count + 1) / (self._doc_freq.get(h, 1) + 1)) + 1
                vectors[i, h] = c * idf
        if normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vectors /= norms
        return vectors


class TransformerEmbedder:
    """Sentence-transformers wrapper (lazy model load)."""

    name = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def _get(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        vecs = self._get().encode(texts, normalize_embeddings=normalize)
        return np.asarray(vecs, dtype="float32")


@lru_cache(maxsize=4)
def _embedder_instance(model_name: str):
    if transformer_available():
        return TransformerEmbedder(model_name)
    return HashEmbedder()


def get_embedder(model_name: str | None = None):
    """Return a cached embedder instance (transformer if available)."""
    name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
    return _embedder_instance(name)
