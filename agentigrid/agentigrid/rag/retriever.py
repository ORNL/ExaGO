"""Retrieval step for the GENERATION stage only.

Design rule: retrieval grounds the LLM's spec/proposal generation. It must
never be consulted by the deterministic semantic validator — keep it out of
engine/validation.py so verification stays deterministic and reproducible.

`retrieve()` is defensive: any failure (Ollama down, empty index, missing
chromadb) returns "" so a run behaves exactly like the no-RAG baseline.

Relevance is filtered by a cosine-similarity threshold (`min_score`) so weak
matches are dropped instead of always returning `k` chunks.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("agentigrid.rag")


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
        # Per-session retrieval telemetry (journaled via describe()): a failed
        # query (e.g. embedding server down) silently degrades to the no-RAG
        # baseline, so errors must be counted or a retrieval run is mislabelled.
        self.stats = {"calls": 0, "errors": 0, "empty": 0}
        self.k = k
        self.min_score = min_score
        self.store_path = path
        self.collection = collection
        self._store = None
        if enabled:
            try:
                from .store import VectorStore

                self._store = VectorStore(path, collection, host, model)
            except Exception:
                self.enabled = False  # degrade gracefully to baseline

    def retrieve(self, query: str, k: Optional[int] = None) -> str:
        """Return a formatted context block for `query`, or "" if disabled/empty.

        Only chunks with cosine similarity >= min_score are kept.
        """
        if not self.enabled or self._store is None or not query:
            return ""
        self.stats["calls"] += 1
        try:
            hits = self._store.query(query, k or self.k)
        except Exception as exc:
            self.stats["errors"] += 1
            logger.warning("Retrieval failed (%s: %s); this iteration runs without references",
                           type(exc).__name__, exc)
            return ""  # retrieval must never break a run
        hits = [(doc, meta, score) for (doc, meta, score) in hits if score >= self.min_score]
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
