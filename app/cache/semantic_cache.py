"""Exact + semantic response cache (PLAN 15.9, improved per 16.4).

- Exact: diskcache, SHA-256 key, 7-day TTL.
- Semantic: embeddings + FAISS IndexFlatIP (batch-built) with dot
  product similarity; threshold from settings (default 0.95).

Caching is bypassed for private/restricted/ephemeral policies and
offline-only requests (PLAN 16.7).
"""

import hashlib
import json
import threading
from pathlib import Path

import numpy as np

from app.config import settings
from app.rag.embeddings import faiss_available, get_embedder


def _cache_allowed(policy: str, private: bool, offline_only: bool) -> bool:
    return policy == "public" and not private and not offline_only


class SemanticCache:
    def __init__(self, cache_path: Path | None = None, threshold: float | None = None):
        self.cache_path = cache_path or settings.cache_path
        self.threshold = threshold if threshold is not None else settings.semantic_cache_threshold
        self.cache_path.mkdir(parents=True, exist_ok=True)
        from diskcache import Cache

        self.exact_store = Cache(str(self.cache_path / "exact"))
        self.records_path = self.cache_path / "semantic.jsonl"
        self.records = self._load_records()
        self.embedder = None  # lazy
        # Persistent search structures (BUG 3): built once, updated
        # incrementally on put — never rebuilt per lookup.
        self._lock = threading.Lock()
        self._index = None  # faiss.IndexFlatIP when FAISS is available
        self._matrix: np.ndarray | None = None  # (N, d) fallback matrix
        self._dim: int | None = None
        self._row_map: list[int] = []  # matrix row -> records index

    def _get_embedder(self):
        if self.embedder is None:
            self.embedder = get_embedder(settings.embedding_model)
        return self.embedder

    def _load_records(self) -> list[dict]:
        if not self.records_path.exists():
            return []
        out = []
        for line in self.records_path.read_text(encoding="utf-8").splitlines():
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return out

    @staticmethod
    def exact_key(text: str, route: str = "auto") -> str:
        return hashlib.sha256(f"{route}|{text.strip()}".encode()).hexdigest()

    def exact_get(self, text: str, route: str = "auto"):
        return self.exact_store.get(self.exact_key(text, route))

    def exact_put(self, text: str, response: str, route: str = "auto"):
        self.exact_store.set(self.exact_key(text, route), response, expire=7 * 86400)

    def _build_search_structures(self) -> None:
        """Build the FAISS index (or numpy matrix) from current records.
        Records whose embedding dim differs from the majority dim are
        skipped (guards against embedder changes across runs)."""
        if not self.records:
            self._index = None
            self._matrix = None
            self._dim = None
            return
        dim_counts: dict[int, int] = {}
        for r in self.records:
            d = len(r["embedding"])
            dim_counts[d] = dim_counts.get(d, 0) + 1
        self._dim = sorted(dim_counts.items(), key=lambda kv: -kv[1])[0][0]
        usable = [
            (i, r) for i, r in enumerate(self.records)
            if len(r["embedding"]) == self._dim
        ]
        matrix = np.ascontiguousarray(
            np.asarray([r["embedding"] for _, r in usable], dtype="float32")
        )
        # remap compacted rows to original record indices
        self._row_map = [i for i, _ in usable]
        if faiss_available():
            import faiss

            index = faiss.IndexFlatIP(matrix.shape[1])
            index.add(matrix)
            self._index = index
            self._matrix = matrix
        else:
            self._index = None
            self._matrix = matrix

    def _ensure_built(self):
        if self._index is None and self._matrix is None:
            self._build_search_structures()
        return self._dim is not None

    def semantic_get(self, text: str):
        """Nearest cached query above threshold, else None."""
        with self._lock:
            if not self.records or not self._ensure_built():
                return None
            q = self._get_embedder().encode([text])[0].astype("float32")
            if len(q) != self._dim:
                return None  # embedder changed since records were written
            if self._index is not None:
                import faiss

                scores, ids = self._index.search(
                    np.ascontiguousarray(q.reshape(1, -1)), 1
                )
                matrix_row = int(ids[0][0])
                score = float(scores[0][0])
            else:
                assert self._matrix is not None
                scores = self._matrix @ q
                matrix_row = int(np.argmax(scores))
                score = float(scores[matrix_row])
        if score < self.threshold or not 0 <= matrix_row < len(self._row_map):
            return None
        row = self._row_map[matrix_row]
        return self.records[row]["response"]

    def semantic_put(self, text: str, response: str):
        emb = self._get_embedder().encode([text])[0].astype("float32")
        record = {
            "query": text,
            "response": response,
            "embedding": emb.tolist(),
        }
        with self._lock:
            self.records.append(record)
            try:
                if self._ensure_built() and len(emb) == self._dim:
                    vec = np.ascontiguousarray(emb.reshape(1, -1))
                    if self._index is not None:
                        self._index.add(vec)
                    if self._matrix is not None:
                        self._matrix = np.vstack([self._matrix, vec])
                else:
                    # first record ever, or dim changed -> rebuild structures
                    self._build_search_structures()
            except Exception:
                # never fail inference because cache indexing broke
                self._index = None
                self._matrix = None
            with self.records_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

    def lookup(self, text: str, policy: str, private: bool, offline_only: bool,
               route: str = "auto") -> str | None:
        """Combined lookup honouring privacy; returns cached response or None."""
        if not _cache_allowed(policy, private, offline_only):
            return None
        hit = self.exact_get(text, route)
        if hit is not None:
            return hit
        return self.semantic_get(text)