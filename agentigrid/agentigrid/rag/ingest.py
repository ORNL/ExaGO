"""Build the RAG index from a corpus folder. Run whenever the corpus changes.
Usage (from the agentigrid project root, venv active):
    OLLAMA_HOST=http://localhost:11434 python -m agentigrid.rag.ingest rag/corpus
    OLLAMA_HOST=... python -m agentigrid.rag.ingest rag/corpus_docs --store rag/store_docs
Reads .txt/.md files under the corpus dir, chunks them, embeds, and stores them
in a persistent Chroma index. The collection is REBUILT from scratch (no stale
chunks from removed files), and ingest_manifest.json in the store dir records the
corpus SHA-256 it was built from, so corpus_hash.corpus_status() can verify
that a store matches the frozen corpus it claims.
"""
from __future__ import annotations

import os
import sys

from .store import VectorStore


def chunk(text: str, size: int | None = None, overlap: int | None = None) -> list[str]:
    """Paragraph-first chunking; only paragraphs longer than ``size`` are wrapped
    (with ``overlap``). Defaults come from corpus_hash.CHUNKING, so every worked
    example and schema exemplar (one paragraph each) stays a single chunk."""
    from .corpus_hash import CHUNKING
    size = CHUNKING["size"] if size is None else size
    overlap = CHUNKING["overlap"] if overlap is None else overlap
    out: list[str] = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if len(para) <= size:
            out.append(para)
        else:
            i = 0
            while i < len(para):
                out.append(para[i : i + size])
                i += size - overlap
    return out


def ingest(
    corpus_dir: str = "rag/corpus",
    store_path: str = "rag/store",
    collection: str = "agentigrid_kb",
    host: str = "http://localhost:11434",
    model: str = "nomic-embed-text",
) -> None:
    import datetime
    import json
    from pathlib import Path

    from .corpus_hash import CHUNKING, INGEST_MANIFEST_NAME, corpus_files, corpus_hash

    store = VectorStore(store_path, collection, host, model)
    store.reset()
    n, wrapped, longest = 0, 0, 0
    for p in corpus_files(Path(corpus_dir)):
        path = str(p)
        text = open(path, encoding="utf-8", errors="ignore").read()
        for para in (q.strip() for q in text.split("\n\n")):
            longest = max(longest, len(para))
            wrapped += len(para) > CHUNKING["size"]
        for i, c in enumerate(chunk(text)):
            store.add(f"{path}#{i}", c, {"source": os.path.basename(path)})
            n += 1
    digest, per_file = corpus_hash(Path(corpus_dir))
    Path(store_path, INGEST_MANIFEST_NAME).write_text(json.dumps({
        "corpus_dir": str(corpus_dir), "corpus_sha256": digest, "files": len(per_file),
        "chunks": n, "collection": collection, "embed_model": model,
        "chunking": {**CHUNKING, "wrapped_paragraphs": wrapped, "longest_paragraph_chars": longest},
        "ingested_at": datetime.datetime.now().isoformat(),
    }, indent=2) + "\n")
    print(f"Ingested {n} chunks from {len(per_file)} file(s); collection '{collection}' holds "
          f"{store.count()}; corpus sha256 {digest}.")
    print(f"Chunking {CHUNKING['scheme']}: longest paragraph {longest} chars; "
          f"{wrapped} paragraph(s) longer than {CHUNKING['size']} were split.")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build the RAG index from a corpus folder.")
    ap.add_argument("corpus", nargs="?", default="rag/corpus")
    ap.add_argument("--store", default="rag/store")
    ap.add_argument("--collection", default="agentigrid_kb")
    a = ap.parse_args()
    ingest(corpus_dir=a.corpus, store_path=a.store, collection=a.collection,
           host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
