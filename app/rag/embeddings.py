"""Embedding providers for RAG + semantic cache.

Provider preference (first available wins):
  1. Ollama (`nomic-embed-text`) — already on the box, no extra download,
     fast on CPU, 768-dim.
  2. sentence-transformers (all-MiniLM-L6-v2, 384-dim) — the PLAN's model.
  3. A zero-dependency deterministic hashing embedder (512-dim TF-IDF
     bag-of-words) so the pipeline still runs with no model at all.

The provider name and dimension are reported by `embedder_name()` /
`embedding_dim()` so indexes and caches can detect a provider change
instead of silently mixing vector spaces.
"""

import hashlib
import math
import re
from functools import lru_cache

import numpy as np

HASH_DIM = 512
EMBED_TIMEOUT_S = 30


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
    not a quality replacement for a real embedding model."""

    name = "hash-512"
    dim = HASH_DIM

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
    dim = 384

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


class OllamaEmbedder:
    """Embeddings from the local Ollama server (`nomic-embed-text`).

    No download beyond the GGUF already on disk, and the request rides the
    same keep-alive HTTP session as chat traffic.
    """

    def __init__(self, model: str = "nomic-embed-text"):
        self.model = model
        self.name = f"ollama:{model}"
        self.dim = 768  # nomic-embed-text-v1.5; corrected on first response
        self._session = None
        self._base = None

    def _client(self):
        if self._session is None:
            from app.inference import llm_client

            self._session = llm_client.session()
            self._base = llm_client.ollama_base()
        return self._session, self._base

    def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        from app.config import settings

        sess, base = self._client()
        r = sess.post(
            f"{base}/api/embed",
            json={
                "model": self.model,
                "input": texts,
                # not -1: a permanently pinned embedder occupies one of
                # Ollama's few resident slots and evicts the chat tiers
                "keep_alive": settings.ollama_embed_keep_alive,
            },
            timeout=EMBED_TIMEOUT_S,
        )
        r.raise_for_status()
        data = r.json()
        vecs = np.asarray(data["embeddings"], dtype="float32")
        self.dim = int(vecs.shape[1])
        if normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vecs = vecs / norms
        return vecs


def ollama_embed_available(model: str = "nomic-embed-text") -> bool:
    """True when the Ollama server has the embedding model pulled."""
    try:
        from app.inference import llm_client

        tags = llm_client.ollama_tags()
        return any(t == model or t.split(":")[0] == model for t in tags)
    except Exception:
        return False


@lru_cache(maxsize=4)
def _embedder_instance(model_name: str, prefer_ollama: bool = True):
    if prefer_ollama:
        from app.config import settings

        candidate = settings.ollama_embed_model
        if ollama_embed_available(candidate):
            return OllamaEmbedder(candidate)
    if transformer_available():
        return TransformerEmbedder(model_name)
    return HashEmbedder()


def get_embedder(model_name: str | None = None):
    """Return a cached embedder instance (Ollama > transformer > hashing)."""
    from app.config import settings

    name = model_name or settings.embedding_model
    return _embedder_instance(name)


def embedder_name() -> str:
    return getattr(get_embedder(), "name", "unknown")


def embedding_dim() -> int:
    return int(getattr(get_embedder(), "dim", HASH_DIM))
