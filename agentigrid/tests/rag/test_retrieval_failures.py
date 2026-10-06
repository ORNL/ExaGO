"""Retrieval failures are counted (not hidden), and the index identity is checked."""

from __future__ import annotations

import json

from agentigrid.rag.corrective import CorrectiveRetriever
from agentigrid.rag.retriever import Retriever

class _DownStore:
    def query(self, q, k):
        raise ConnectionError("embedding server unreachable")


def _retriever():
    r = Retriever(enabled=False)
    r.enabled, r._store = True, _DownStore()
    return r


def test_basic_retriever_counts_errors():
    r = _retriever()
    assert r.retrieve("reduce cost") == ""
    assert r.stats == {"calls": 1, "errors": 1, "empty": 0}
    assert r.describe()["stats"]["errors"] == 1


def test_corrective_exposes_base_errors():
    c = CorrectiveRetriever(_retriever())
    assert c.retrieve("reduce cost") == ""
    assert c.describe()["stats"]["retrieval_errors"] >= 1


def test_chunking_keeps_worked_examples_whole():
    from agentigrid.rag.ingest import chunk
    example = "Goal: lower cost\nStep applied: x\nCorrect response (JSON): " + '{"a": 1}, ' * 180 + "\nResult: ok"
    assert 800 < len(example) < 3000
    assert chunk(example + "\n\nsecond paragraph") == [example, "second paragraph"]
    long = "y" * 7000
    parts = chunk(long)
    assert len(parts) == 3 and all(len(p) <= 3000 for p in parts)


def test_status_refuses_store_with_other_chunking(tmp_path):
    from agentigrid.rag.corpus_hash import CHUNKING, INGEST_MANIFEST_NAME, MANIFEST_NAME, corpus_hash, corpus_status
    corpus, store = tmp_path / "corpus", tmp_path / "store"
    corpus.mkdir(); store.mkdir()
    (corpus / "a.txt").write_text("hello")
    digest, _ = corpus_hash(corpus)
    (corpus / MANIFEST_NAME).write_text(json.dumps({"corpus_sha256": digest}))
    (store / INGEST_MANIFEST_NAME).write_text(json.dumps({"corpus_sha256": digest}))          # v1 store
    st = corpus_status(corpus, store)
    assert not st["ok"] and "chunking" in st["reason"]
    (store / INGEST_MANIFEST_NAME).write_text(json.dumps({"corpus_sha256": digest, "chunking": dict(CHUNKING)}))
    assert corpus_status(corpus, store)["ok"]
