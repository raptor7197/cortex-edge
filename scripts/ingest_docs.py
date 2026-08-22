import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

"""Build the local RAG index from datasets/documents (PLAN 15.10).

Usage: python scripts/ingest_docs.py [--dir datasets/documents]
"""

import argparse

from app.config import settings
from app.rag.ingest import build_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="datasets/documents")
    args = parser.parse_args()
    build_index(args.dir, settings.faiss_index_path, settings.chunk_store_path)


if __name__ == "__main__":
    main()