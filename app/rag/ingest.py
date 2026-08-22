"""RAG ingestion (PLAN 15.10): PDF/txt extraction, chunking, embeddings,
FAISS index build. Run: `make ingest` or `python scripts/ingest_docs.py`.
"""

import json
from pathlib import Path

import numpy as np

from app.config import settings
from app.rag.embeddings import faiss_available, get_embedder


def extract_pdf(path: Path) -> str:
    import pymupdf

    with pymupdf.open(path) as pdf:
        return "\n".join(page.get_text("text") for page in pdf)  # type: ignore[arg-type]


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(path)
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".docx":
        try:
            import docx  # python-docx (optional)

            return "\n".join(p.text for p in docx.Document(str(path)).paragraphs)
        except ImportError:
            raise RuntimeError(
                f"Cannot extract {path.name}: install python-docx for .docx files"
            )
    raise ValueError(f"Unsupported document type: {suffix}")


def chunk_text(text: str, size: int | None = None, overlap: int | None = None) -> list[str]:
    """Word-based chunking with overlap (default 450/70 per PLAN)."""
    size = size or settings.rag_chunk_size
    overlap = overlap or settings.rag_chunk_overlap
    words = text.split()
    chunks, step = [], size - overlap
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + size]).strip()
        if len(chunk.split()) >= 40:
            chunks.append(chunk)
    return chunks


def build_index(document_dir: str | Path, index_path: Path | None = None,
                chunk_path: Path | None = None) -> int:
    """Embed every supported file in `document_dir` and write a FAISS
    IndexFlatIP + chunks.json. Returns number of chunks indexed."""
    document_dir = Path(document_dir)
    index_path = index_path or settings.faiss_index_path
    chunk_path = chunk_path or settings.chunk_store_path

    embedder = get_embedder(settings.embedding_model)
    records: list[dict] = []
    files = sorted(p for p in document_dir.iterdir()
                   if p.suffix.lower() in {".pdf", ".txt", ".md", ".docx"})
    if not files:
        print(f"No documents found in {document_dir}")
        return 0
    for path in files:
        try:
            text = extract_text(path)
        except Exception as exc:
            print(f"skip {path.name}: {exc}")
            continue
        for number, chunk in enumerate(chunk_text(text)):
            records.append({"source": path.name, "chunk": number, "text": chunk})

    if not records:
        print("No chunks produced")
        return 0

    vectors = embedder.encode([r["text"] for r in records])
    vectors = np.ascontiguousarray(vectors, dtype="float32")

    if faiss_available():
        import faiss

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        faiss.write_index(index, str(index_path))
    else:
        np.save(str(index_path) + ".npy", vectors)

    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    print(f"Indexed {len(records)} chunks ({len(files)} files) -> {index_path}")
    return len(records)