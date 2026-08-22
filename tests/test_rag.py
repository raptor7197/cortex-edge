"""RAG ingestion/retrieval tests, using the hash embedder + FAISS."""

import json

import pytest

from app.rag.embeddings import HashEmbedder
from app.rag.ingest import build_index, chunk_text
from app.rag.retriever import Retriever, build_rag_prompt


@pytest.fixture
def hash_embedder(monkeypatch):
    monkeypatch.setattr("app.rag.ingest.get_embedder", lambda name: HashEmbedder())
    monkeypatch.setattr("app.rag.retriever.get_embedder", lambda name: HashEmbedder())


def test_chunking_word_counts():
    text = " ".join(f"word{i}" for i in range(1000))
    chunks = chunk_text(text, size=450, overlap=70)
    assert len(chunks) >= 3
    assert all(len(c.split()) >= 40 for c in chunks)


def test_build_index_and_search(tmp_path, hash_embedder):
    doc = tmp_path / "sample.txt"
    doc.write_text(
        "The Raspberry Pi 5 has a quad-core CPU. "
        "CortexEdge routes queries between local models and the cloud. "
        "The INA219 sensor measures power consumption. "
        "Semantic caching reduces repeated computation. "
        "FAISS indexes document embeddings for retrieval augmented generation. "
        "Latency budgets constrain route selection to meet response time targets. "
    )
    idx = tmp_path / "faiss.index"
    chunks = tmp_path / "chunks.json"
    n = build_index(tmp_path, idx, chunks)
    assert n > 0

    retriever = Retriever(idx, chunks)
    hits = retriever.search("what does the INA219 sensor measure?")
    assert hits, "expected at least one hit"
    assert hits[0]["score"] > 0.0

    prompt = build_rag_prompt("q", hits)
    assert "[Source: sample.txt" in prompt
    assert "ANSWER" in prompt


def test_prompt_citations():
    passages = [{"source": "a.pdf", "chunk": 0, "text": "body", "score": 0.9}]
    prompt = build_rag_prompt("question?", passages)
    assert "[Source: a.pdf, chunk 0]" in prompt


def test_chunks_json_schema(tmp_path, hash_embedder):
    doc = tmp_path / "d.txt"
    doc.write_text("word " * 500)
    idx = tmp_path / "idx"
    chunks = tmp_path / "ch.json"
    build_index(tmp_path, idx, chunks)
    records = json.loads(chunks.read_text())
    assert {"source", "chunk", "text"} <= set(records[0])