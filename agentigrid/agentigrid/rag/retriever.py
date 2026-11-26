"""Retrieval step for the GENERATION stage only.

Design rule: retrieval grounds the LLM's spec/proposal generation. It must
never be consulted by the deterministic semantic validator — keep it out of
engine/validation.py so verification stays deterministic and reproducible.

`preflight()` checks, before a run starts, that the store opens, is not empty
and the embedding server answers; it raises RAGUnavailable otherwise, so a run
that asked for retrieval never silently runs without it. During the run,
`retrieve()` stays defensive: a failed query returns "" for that iteration and
is counted in `stats`.

Relevance is filtered by a cosine-similarity threshold (`min_score`) so weak
matches are dropped instead of always returning `k` chunks.
"""
from __future__ import annotations

import json
import hashlib
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("agentigrid.rag")


class RAGUnavailable(RuntimeError):
    """Retrieval was requested but cannot work (store missing or empty,
    embedding server unreachable). Raised before the first iteration."""


def hits_summary(hits, scores=None) -> list[dict]:
    """Compact, JSON-safe record of scored hits: source file, similarity, length,
    a 16-hex content hash, and the grader's score when one is given."""
    out = []
    for i, (doc, meta, score) in enumerate(hits or []):
        meta = meta or {}
        rec = {"source": meta.get("source"), "score": round(float(score), 4), "chars": len(doc or ""),
               # content hash: shows directly whether two iterations got the same chunk
               "sha": hashlib.sha256((doc or "").encode("utf-8")).hexdigest()[:16]}
        if scores is not None and i < len(scores):
            rec["grade"] = round(float(scores[i]), 4)
        out.append(rec)
    return out


class Retriever:
    def __init__(
        self,
        enabled: bool = True,
        path: str = "rag/store",
        collection: str = "agentigrid_kb",
        host: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
        k: int = 3,
        min_score: float = 0.35,
    ):
        self.enabled = enabled
        # True when retrieval was asked for, even if the store then failed to open
        # (enabled turns False); preflight() uses it to refuse such a run.
        self.requested = enabled
        self.init_error: Optional[str] = None
        self.host, self.model = host, model
        # Per-session retrieval telemetry (journaled via describe()): a failed
        # query (e.g. embedding server down) silently degrades to the no-RAG
        # baseline, so errors must be counted or a retrieval run is mislabelled.
        self.stats = {"calls": 0, "errors": 0, "empty": 0}
        # What the last retrieve() call did (journaled per iteration by the agent
        # loop as rag_log): scored hits by source, how many were kept, outcome.
        self.last_trace: dict = {}
        self.k = k
        self.min_score = min_score
        self.store_path = path
        self.collection = collection
        self._store = None
        if enabled:
            try:
                from .store import VectorStore

                self._store = VectorStore(path, collection, host, model)
            except Exception as exc:
                self.enabled = False
                self.init_error = f"{type(exc).__name__}: {exc}"

    def preflight(self) -> None:
        """Refuse to start a run that asked for retrieval when retrieval cannot
        work. No-op when retrieval was not requested."""
        if not self.requested:
            return
        where = f"store {self.store_path!r}, collection {self.collection!r}"
        if self._store is None:
            raise RAGUnavailable(f"RAG is on but the vector store could not be opened ({where}): "
                                 f"{self.init_error or 'unknown error'}. Install requirements-rag.txt "
                                 "and build the store with `python -m agentigrid.rag.ingest`.")
        try:
            n = self._store.count()
        except Exception as exc:
            raise RAGUnavailable(f"RAG is on but the vector store cannot be read ({where}): {exc}") from exc
        if n == 0:
            raise RAGUnavailable(f"RAG is on but the vector store is empty ({where}). "
                                 "Build it with `python -m agentigrid.rag.ingest <corpus dir>` or "
                                 "point AGENTIGRID_RAG_STORE at a built store.")
        try:
            self._store.query("preflight", 1)
        except Exception as exc:
            raise RAGUnavailable(f"RAG is on but the embedding server does not answer "
                                 f"(host {self.host}, model {self.model!r}): {type(exc).__name__}: {exc}. "
                                 "Start Ollama and `ollama pull` the model, or set OLLAMA_HOST.") from exc

    def store_identity(self) -> Optional[dict]:
        """What the store was built from, read from its ingest manifest: corpus
        SHA-256, chunking, embedding model, chunk count. None if the store has no
        manifest (built by an older ingest) or cannot be read."""
        try:
            from .corpus_hash import INGEST_MANIFEST_NAME
            m = json.loads((Path(self.store_path) / INGEST_MANIFEST_NAME).read_text(encoding="utf-8"))
        except Exception:
            return None
        keep = ("corpus_sha256", "files", "chunks", "collection", "embed_model",
                "chunking", "ingested_at")
        return {k: m[k] for k in keep if k in m}

    def retrieve(self, query: str, k: Optional[int] = None) -> str:
        """Return a formatted context block for `query`, or "" if disabled/empty.

        Only chunks with cosine similarity >= min_score are kept.
        """
        if not self.enabled or self._store is None or not query:
            self.last_trace = {"mode": "basic", "action": "disabled"}
            return ""
        self.stats["calls"] += 1
        try:
            hits = self._store.query(query, k or self.k)
        except Exception as exc:
            self.stats["errors"] += 1
            self.last_trace = {"mode": "basic", "action": "error", "error": type(exc).__name__}
            logger.warning("Retrieval failed (%s: %s); this iteration runs without references",
                           type(exc).__name__, exc)
            return ""  # retrieval must never break a run
        all_hits = hits
        hits = [(doc, meta, score) for (doc, meta, score) in hits if score >= self.min_score]
        self.last_trace = {"mode": "basic", "hits": hits_summary(all_hits), "kept": len(hits),
                           "action": "used" if hits else "empty"}
        if not hits:
            self.stats["empty"] += 1
            return ""
        lines = [
            f"[ref {i + 1} | score {score:.2f}] {doc}"
            for i, (doc, _meta, score) in enumerate(hits)
        ]
        return (
            "Use the following retrieved reference material only when relevant; "
            "do not invent facts beyond it.\n" + "\n".join(lines)
        )

    def describe(self) -> dict:
        """Effective configuration for this run (shown in the UI / logged)."""
        return {
            "mode": "basic",
            "enabled": self.enabled,
            "k": self.k,
            "min_score": self.min_score,
            "store_path": self.store_path,
            "collection": self.collection,
            "embed_host": self.host,
            "embed_model": self.model,
            "store_manifest": self.store_identity() if self.requested else None,
            "stats": dict(self.stats),
        }

    def query_hits(self, query: str, k: Optional[int] = None) -> list[tuple[str, dict, float]]:
        """Raw scored hits (document, metadata, similarity), best first, with NO
        ``min_score`` filtering. Advanced modes (e.g. Corrective RAG) grade these
        with their own thresholds. Returns [] on disabled/empty/any failure.
        """
        if not self.enabled or self._store is None or not query:
            return []
        try:
            return self._store.query(query, k or self.k)
        except Exception as exc:
            self.stats["errors"] += 1
            logger.warning("Retrieval failed (%s: %s)", type(exc).__name__, exc)
            return []

    def embed_text(self, text: str) -> list[float]:
        """Embed `text` with the same Ollama host/model as the store, for
        strip-level re-scoring in advanced modes. Raises on failure (callers
        that must degrade gracefully wrap this in try/except).
        """
        from .embed import embed as _embed

        return _embed(text, self._store._host, self._store._model)

    @classmethod
    def from_config(cls, cfg) -> "Retriever":
        """Build a Retriever from the app config's optional `rag:` block.

        Falls back to llm.ollama_host so the embedder reaches the same Ollama as
        the chat backend. ADAPT the attribute access to AgentiGrid's config object.
        """
        rag = getattr(cfg, "rag", None)
        get = (rag.get if isinstance(rag, dict) else (lambda k, d=None: getattr(rag, k, d))) if rag else (lambda k, d=None: d)
        llm = getattr(cfg, "llm", None)
        llm_host = getattr(llm, "ollama_host", None) if llm else None
        return cls(
            enabled=bool(get("enabled", False)),
            path=get("store_path", "rag/store"),
            collection=get("collection", "agentigrid_kb"),
            host=get("ollama_host", None) or llm_host or "http://localhost:11434",
            model=get("embed_model", "nomic-embed-text"),
            k=int(get("top_k", 3)),
            min_score=float(get("min_score", 0.35)),
        )
