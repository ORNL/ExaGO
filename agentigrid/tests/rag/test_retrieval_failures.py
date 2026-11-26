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


# --- startup check (preflight) and recorded store identity ------------------------

import pytest

from agentigrid.rag import RAGUnavailable


class _Store:
    def __init__(self, n=3, embed_ok=True):
        self.n, self.embed_ok = n, embed_ok

    def count(self):
        return self.n

    def query(self, q, k):
        if not self.embed_ok:
            raise ConnectionError("embedding server unreachable")
        return [("doc", {}, 0.9)]


def _with_store(store, path="rag/store"):
    r = Retriever(enabled=False, path=path)
    r.enabled = r.requested = True
    r._store = store
    return r


def test_preflight_is_a_noop_when_rag_is_off():
    Retriever(enabled=False).preflight()


def test_preflight_refuses_a_store_that_did_not_open():
    r = Retriever(enabled=False)
    r.requested, r.init_error = True, "ModuleNotFoundError: No module named 'chromadb'"
    with pytest.raises(RAGUnavailable, match="could not be opened.*chromadb"):
        r.preflight()


def test_preflight_refuses_an_empty_store():
    with pytest.raises(RAGUnavailable, match="empty"):
        _with_store(_Store(n=0)).preflight()


def test_preflight_refuses_when_the_embedder_is_down():
    with pytest.raises(RAGUnavailable, match="embedding server"):
        _with_store(_Store(embed_ok=False)).preflight()


def test_preflight_passes_and_corrective_delegates():
    r = _with_store(_Store())
    r.preflight()
    CorrectiveRetriever(r).preflight()
    with pytest.raises(RAGUnavailable):
        CorrectiveRetriever(_with_store(_Store(n=0))).preflight()


def test_describe_records_the_store_manifest(tmp_path):
    from agentigrid.rag.corpus_hash import INGEST_MANIFEST_NAME
    (tmp_path / INGEST_MANIFEST_NAME).write_text(json.dumps({
        "corpus_dir": "/somewhere/rag/corpus", "corpus_sha256": "ab" * 32, "files": 2, "chunks": 5,
        "collection": "agentigrid_kb", "embed_model": "nomic-embed-text",
        "chunking": {"scheme": "paragraph-v2"}, "ingested_at": "2026-11-26T12:00:00"}))
    r = _with_store(_Store(), path=str(tmp_path))
    d = r.describe()
    assert d["store_manifest"]["corpus_sha256"] == "ab" * 32
    assert d["store_manifest"]["embed_model"] == "nomic-embed-text"
    assert "corpus_dir" not in d["store_manifest"]          # no machine-specific paths in journals
    assert d["embed_model"] == "nomic-embed-text"
    assert CorrectiveRetriever(r).describe()["store_manifest"]["chunks"] == 5


def test_describe_without_manifest_is_none(tmp_path):
    assert _with_store(_Store(), path=str(tmp_path)).describe()["store_manifest"] is None


def test_embed_uses_a_short_connect_timeout(monkeypatch):
    """An unreachable server must fail fast (connect timeout), not after the read timeout."""
    import agentigrid.rag.embed as emb
    seen = {}

    def fake_post(url, json=None, timeout=None):
        seen["timeout"] = timeout
        raise ConnectionError("unreachable")

    monkeypatch.setattr(emb.requests, "post", fake_post)
    with pytest.raises(ConnectionError):
        emb.embed("x", "http://10.255.255.1:11434")
    assert seen["timeout"] == (5, 60)


def test_ingest_stops_before_touching_the_store_when_the_embedder_is_down(tmp_path, monkeypatch):
    import agentigrid.rag.ingest as ing
    import agentigrid.rag.embed as emb

    def down(*a, **k):
        raise ConnectionError("refused")

    monkeypatch.setattr(emb, "embed", down)
    touched = []
    monkeypatch.setattr(ing, "VectorStore", lambda *a, **k: touched.append(1))
    (tmp_path / "a.txt").write_text("fact")
    with pytest.raises(ing.EmbedderUnavailable, match=r"(?s)not usable at http://h:1.*not changed"):
        ing.ingest(str(tmp_path), str(tmp_path / "store"), host="http://h:1")
    assert touched == []
