"""Local cross-encoder relevance grader for Corrective RAG.

Fills the grader seam of `CorrectiveRetriever`:

    grader(query, hits) -> [relevance score in 0..1 per hit]

A cross-encoder reads the query and a retrieved chunk *together* and scores
their relevance, which is the standard stronger alternative to the embedding
cosine similarity used by the default grader. It runs entirely locally (no
network after the one-time model download), in line with the air-gapped design.

Model: `AGENTIGRID_RERANKER_MODEL` (default ``BAAI/bge-reranker-base``, MIT
licensed). May be a Hugging Face id or a LOCAL DIRECTORY, so an air-gapped
machine can load a copied model. Optional `AGENTIGRID_RERANKER_DEVICE`
(e.g. ``cpu``, ``cuda``).

Scores are mapped to (0, 1) with a sigmoid when the model returns raw logits.
They are relevance SCORES, not calibrated probabilities (calibration, e.g. Platt
scaling on labeled pairs, is future work). The corrective thresholds for this
grader should therefore be set from labeled examples, via
`AGENTIGRID_CRAG_TAU_LOWER` / `AGENTIGRID_CRAG_TAU_UPPER` (see rag/__init__.py).

Design rules, as for every grader:
  * generation-stage only — never touches engine/validation.py;
  * failures are COUNTED and reported (describe()), so a run whose reranker fell
    back to cosine is visible in the journal instead of being silently mislabeled.
"""
from __future__ import annotations

import importlib.util
import math
import os
import time
from typing import Optional

Hit = tuple[str, dict, float]
DEFAULT_MODEL = "BAAI/bge-reranker-base"


def reranker_available() -> bool:
    """True if the cross-encoder library is installed (the model itself is
    loaded, and may be downloaded, on first use)."""
    return importlib.util.find_spec("sentence_transformers") is not None


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def to_unit_interval(raw: list[float]) -> list[float]:
    """Map scores into [0, 1]: keep them if already probabilities, else sigmoid."""
    vals = [float(x) for x in raw]
    if all(0.0 <= v <= 1.0 for v in vals):
        return vals
    return [_sigmoid(v) for v in vals]


class RerankerGrader:
    """Callable cross-encoder grader. Loads the model lazily on first use."""

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None,
                 batch_size: int = 16, _model=None):
        self.model_name = model_name or os.environ.get("AGENTIGRID_RERANKER_MODEL") or DEFAULT_MODEL
        self.device = device or os.environ.get("AGENTIGRID_RERANKER_DEVICE") or None
        self.batch_size = batch_size
        self._model = _model            # injectable for tests
        self._load_error: Optional[str] = None
        self.calls = 0
        self.failures = 0
        self.seconds = 0.0

    # -- model ---------------------------------------------------------
    def _ensure_model(self):
        if self._model is not None or self._load_error is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
            kwargs = {"device": self.device} if self.device else {}
            self._model = CrossEncoder(self.model_name, **kwargs)
        except Exception as exc:  # missing library, no network, bad path ...
            self._load_error = f"{type(exc).__name__}: {exc}"[:300]
            self._model = None
        return self._model

    @property
    def available(self) -> bool:
        return self._ensure_model() is not None

    # -- seam ----------------------------------------------------------
    def __call__(self, query: str, hits: list[Hit]) -> list[float]:
        if not hits:
            return []
        self.calls += 1
        model = self._ensure_model()
        if model is None:
            self.failures += 1
            raise RuntimeError(f"reranker unavailable: {self._load_error}")
        pairs = [(query, doc) for (doc, _m, _s) in hits]
        t0 = time.perf_counter()
        try:
            raw = model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        except TypeError:  # older/newer sentence-transformers signatures
            raw = model.predict(pairs)
        except Exception:
            self.failures += 1
            raise
        finally:
            self.seconds += time.perf_counter() - t0
        scores = to_unit_interval(list(raw))
        if len(scores) != len(hits):
            self.failures += 1
            raise RuntimeError("reranker returned a score list of the wrong length")
        return scores

    def describe(self) -> dict:
        ok = self._model is not None or (self._load_error is None and self.calls == 0)
        name = "reranker" if ok and not self.failures else "reranker (fallback: cosine)"
        return {
            "grader": name,
            "model": self.model_name,
            "calls": self.calls,
            "failures": self.failures,
            "mean_latency_s": round(self.seconds / self.calls, 4) if self.calls else None,
            "load_error": self._load_error,
        }
