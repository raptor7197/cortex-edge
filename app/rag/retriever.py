"""RAG retrieval (PLAN 15.10): FAISS top-k search + prompt builder."""

import json
from pathlib import Path

import numpy as np

from app.config import settings
from app.rag.embeddings import faiss_available, get_embedder


class Retriever:
    def __init__(self, index_path: Path | None = None, chunk_path: Path | None = None):
        self.index_path = index_path or settings.faiss_index_path
        self.chunk_path = chunk_path or settings.chunk_store_path
        self.embedder = get_embedder(settings.embedding_model)
        self._index = None
        self._vectors: np.ndarray | None = None  # cached numpy fallback
        self.records = self._load_chunks()

    def _load_chunks(self) -> list[dict]:
        if not self.chunk_path.exists():
            raise FileNotFoundError(
                f"RAG index not built: {self.chunk_path} is missing. "
                "Run `make ingest` (or scripts/ingest_docs.py) first."
            )
        return json.loads(self.chunk_path.read_text(encoding="utf-8"))

    def _get_index(self):
        if self._index is not None:
            return self._index
        if not faiss_available():
            raise RuntimeError("FAISS is unavailable; using numpy fallback")
        import faiss

        self._index = faiss.read_index(str(self.index_path))
        return self._index

    def _get_vectors(self) -> np.ndarray:
        """Cached (N, d) document matrix for the no-FAISS path."""
        if self._vectors is not None:
            result: np.ndarray = self._vectors
            return result
        npy_path = Path(str(self.index_path) + ".npy")
        if npy_path.exists() and len(self.records):
            try:
                loaded = np.load(str(npy_path))
                if loaded.ndim == 2 and loaded.shape[0] == len(self.records):
                    result = loaded.astype("float32")
                    self._vectors = result
                    return result
            except Exception:
                pass
        result = self.embedder.encode([r["text"] for r in self.records]).astype("float32")
        self._vectors = result
        return result

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        if not self.records:
            return []
        top_k = max(1, min(top_k, len(self.records)))
        q = np.ascontiguousarray(self.embedder.encode([query]), dtype="float32")
        try:
            index = self._get_index()
            scores, ids = index.search(q, top_k)
            hits = []
            for rank, i in enumerate(ids[0]):
                if i >= 0 and i < len(self.records):
                    hits.append(self.records[i] | {"score": round(float(scores[0][rank]), 4)})
            return hits
        except Exception:
            pass  # fall through to numpy path
        # numpy fallback (no FAISS): cached precomputed vectors (BUG 4)
        matrix = self._get_vectors()
        scores = matrix @ q[0]
        order = np.argsort(scores)[::-1][:top_k]
        return [
            self.records[int(i)] | {"score": round(float(scores[i]), 4)}
            for i in order
        ]


def build_rag_prompt(question: str, passages: list[dict]) -> str:
    context = "\n\n".join(
        f"[Source: {p['source']}, chunk {p['chunk']}]\n{p['text']}" for p in passages
    )
    return f"""Answer using only the supplied context. If the answer is absent, say so.
Cite sources using [filename, chunk number].

CONTEXT
{context}

QUESTION
{question}

ANSWER"""