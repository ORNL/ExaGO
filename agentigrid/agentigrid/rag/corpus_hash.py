"""Corpus identity: which files form a corpus and its SHA-256.

Shared by rag/tools/corpus_guard.py (freeze), agentigrid.rag.ingest (records the
hash of what it indexed) and corpus_status() (checks that a store was built
from the frozen corpus it claims). One definition, so they can never disagree.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

CORPUS_SUFFIXES = (".txt", ".md")          # what ingest indexes
MANIFEST_NAME = "corpus_manifest.json"     # written by corpus_guard --freeze
INGEST_MANIFEST_NAME = "ingest_manifest.json"  # written by ingest into the store dir

# How ingest splits the corpus into chunks. Part of the store's identity: the same
# corpus chunked differently retrieves different text, so corpus_status refuses a
# store built with another scheme. v2 keeps every paragraph (one worked example,
# one schema exemplar) whole up to CHUNK_SIZE characters; v1 (800/100) cut most
# worked examples mid-JSON.
CHUNKING = {"scheme": "paragraph-v2", "size": 3000, "overlap": 200}


def corpus_files(corpus_dir: Path) -> list[Path]:
    corpus_dir = Path(corpus_dir)
    return sorted(p for p in corpus_dir.rglob("*")
                  if p.is_file() and p.suffix in CORPUS_SUFFIXES and p.name != MANIFEST_NAME)


def corpus_hash(corpus_dir: Path) -> tuple[str, dict[str, str]]:
    """Overall SHA-256 over (relative path, content hash) pairs, order-independent of disk."""
    corpus_dir = Path(corpus_dir)
    per_file = {}
    for p in corpus_files(corpus_dir):
        per_file[p.relative_to(corpus_dir).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    h = hashlib.sha256()
    for rel in sorted(per_file):
        h.update(f"{rel}\0{per_file[rel]}\n".encode())
    return h.hexdigest(), per_file


def corpus_status(corpus_dir: Path, store_dir: Path) -> dict:
    """Is *corpus_dir* frozen, unchanged since freezing, and is *store_dir* built
    from exactly that corpus? Returns {"ok": bool, "reason": str|None, "corpus_sha256": ...}."""
    corpus_dir, store_dir = Path(corpus_dir), Path(store_dir)
    out = {"ok": False, "reason": None, "corpus_dir": str(corpus_dir), "store_dir": str(store_dir),
           "corpus_sha256": None}
    mpath = corpus_dir / MANIFEST_NAME
    if not mpath.exists():
        out["reason"] = f"corpus not frozen ({mpath} missing; run corpus_guard.py --freeze)"
        return out
    frozen = json.loads(mpath.read_text()).get("corpus_sha256")
    digest, _ = corpus_hash(corpus_dir)
    out["corpus_sha256"] = digest
    if digest != frozen:
        out["reason"] = "corpus changed since it was frozen (re-run corpus_guard.py --freeze, then ingest)"
        return out
    ipath = store_dir / INGEST_MANIFEST_NAME
    if not ipath.exists():
        out["reason"] = f"store not built by the current ingest ({ipath} missing; re-run ingest)"
        return out
    ing = json.loads(ipath.read_text())
    if ing.get("corpus_sha256") != digest:
        out["reason"] = "store was built from a different corpus version (re-run ingest)"
        return out
    built = ing.get("chunking") or {}
    if {k: built.get(k) for k in CHUNKING} != CHUNKING:
        out["reason"] = (f"store was built with different chunking ({built or 'v1, 800/100'}; "
                         f"current {CHUNKING}); re-run ingest")
        return out
    out["ok"] = True
    return out
