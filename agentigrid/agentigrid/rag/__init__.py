"""RAG (retrieval-augmented generation) support for AgentiGrid.

Generation-stage only: retrieval grounds the LLM's spec/proposal generation.
It is NEVER used by the deterministic validator (engine/validation.py).

Retrieval is OFF by default. The mode is selected by an env var:

    AGENTIGRID_RAG_MODE = off | basic | corrective

For backward compatibility the legacy switch still works:
    AGENTIGRID_RAG=1  ->  basic
    AGENTIGRID_RAG=0/unset (and no _MODE) -> off

Corrective-mode grader (AGENTIGRID_CRAG_GRADER): cosine (default) | reranker
(local cross-encoder, needs sentence-transformers). Thresholds:
AGENTIGRID_CRAG_TAU_LOWER / AGENTIGRID_CRAG_TAU_UPPER.

Use `build_retriever(**kwargs)` to get the retriever for the active mode; every
mode exposes the same surface (`.enabled` and `.retrieve(query) -> str`).
"""
from __future__ import annotations

import os

from .retriever import Retriever
from .corrective import CorrectiveRetriever

VALID_MODES = ("off", "basic", "corrective")
# Graders implemented behind the corrective-RAG seam.
VALID_CRAG_GRADERS = ("cosine", "reranker")


def resolve_rag_mode() -> str:
    """Return the active RAG mode, honoring AGENTIGRID_RAG_MODE and falling back
    to the legacy AGENTIGRID_RAG switch. Unknown values degrade to 'basic'.
    """
    m = os.environ.get("AGENTIGRID_RAG_MODE")
    if m:
        m = m.strip().lower()
        return m if m in VALID_MODES else "basic"
    return "basic" if os.environ.get("AGENTIGRID_RAG", "0") == "1" else "off"


def resolve_crag_grader() -> str:
    """Return the corrective-RAG grader name from AGENTIGRID_CRAG_GRADER.
    'cosine' (default/unset) or 'reranker'. Unknown values fall back to 'cosine'.
    """
    name = os.environ.get("AGENTIGRID_CRAG_GRADER", "").strip().lower()
    return name if name in VALID_CRAG_GRADERS else "cosine"


def _make_crag_grader():
    """Build the optional grader for corrective mode (selected by
    AGENTIGRID_CRAG_GRADER). 'cosine' -> None (deterministic cosine buckets, the
    reproducible default). 'reranker' -> RerankerGrader, imported lazily so
    sentence-transformers is only needed when it is actually requested.
    """
    name = resolve_crag_grader()
    if name == "reranker":
        from .grader_reranker import RerankerGrader
        return RerankerGrader()  # loads lazily; failures are counted, not hidden
    return None  # 'cosine' (and any unknown value already normalized)


def _tau_overrides() -> dict:
    """Corrective thresholds from AGENTIGRID_CRAG_TAU_LOWER / _UPPER, if set.

    Each grader's score scale differs (cosine vs. reranker score vs. calibrated
    probability), so its thresholds are fixed on labeled data and passed per
    run; unset = the module defaults (0.30 / 0.50, cosine).
    """
    out = {}
    for env, key in (("AGENTIGRID_CRAG_TAU_LOWER", "tau_lower"), ("AGENTIGRID_CRAG_TAU_UPPER", "tau_upper")):
        val = os.environ.get(env)
        if val not in (None, ""):
            out[key] = float(val)
    if "tau_lower" in out and "tau_upper" in out and out["tau_lower"] > out["tau_upper"]:
        raise ValueError("AGENTIGRID_CRAG_TAU_LOWER must not exceed AGENTIGRID_CRAG_TAU_UPPER")
    return out


def build_retriever(**base_kwargs):
    """Construct the retriever for the active mode.

    `base_kwargs` are passed to the basic Retriever (e.g. host=, path=, model=).
    Returns a Retriever (off/basic) or a CorrectiveRetriever (corrective); both
    share the `.enabled` / `.retrieve()` surface.
    """
    mode = resolve_rag_mode()
    base_kwargs.pop("enabled", None)
    # Corpus selection (e.g. core vs. a larger documentation corpus): each
    # corpus is ingested into its own store directory.
    if os.environ.get("AGENTIGRID_RAG_STORE"):
        base_kwargs["path"] = os.environ["AGENTIGRID_RAG_STORE"]
    if os.environ.get("AGENTIGRID_RAG_COLLECTION"):
        base_kwargs["collection"] = os.environ["AGENTIGRID_RAG_COLLECTION"]
    if mode == "off":
        return Retriever(enabled=False)
    base = Retriever(enabled=True, **base_kwargs)
    if mode == "corrective":
        return CorrectiveRetriever(base, grader=_make_crag_grader(), **_tau_overrides())
    return base  # 'basic' (and any unknown value already normalized to basic)


__all__ = [
    "Retriever", "CorrectiveRetriever", "build_retriever",
    "resolve_rag_mode", "resolve_crag_grader",
]
