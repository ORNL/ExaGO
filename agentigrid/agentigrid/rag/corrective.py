"""Corrective RAG (CRAG) retriever — generation-stage only.

Extends basic retrieval with a CRAG-style retrieval evaluator + knowledge
refinement, adapted to an OFFLINE OPF setting (there is no web-search fallback):

  1. Retrieve candidate chunks (reuses the basic Retriever's vector store).
  2. Grade them -> CORRECT / AMBIGUOUS / INCORRECT.
     The default grader is DETERMINISTIC (cosine-score buckets), so runs are
     reproducible and it works even when the base LLM is too weak to grade.
     The grader is a single swappable seam (the `grader=` callable) — an
     LLM-based evaluator can be dropped in later WITHOUT touching anything else
     here; it only has to return one relevance score per hit.
  3. Corrective action:
       CORRECT   -> refined internal knowledge only.
       AMBIGUOUS -> refined internal knowledge + a hedge instruction.
       INCORRECT -> reformulate the query once and re-retrieve; if it is still
                    INCORRECT, withhold context entirely (degrade to the no-RAG
                    baseline) rather than ground the model on misleading refs.
  4. Knowledge refinement (the CRAG decompose-recompose step): split kept chunks
     into strips, re-score each strip against the query, drop weak strips, and
     recompose the survivors — so only the relevant sentences reach the prompt.

Like the basic retriever it is DEFENSIVE: any failure returns "" so a run
behaves exactly like the no-RAG baseline. It NEVER touches the deterministic
validator (engine/validation.py) — retrieval grounds generation only.

Thresholds below are initial defaults, meant to be tuned on the real corpus;
they are not empirically calibrated yet.
"""
from __future__ import annotations

import math
import re
from typing import Callable, Optional

from .retriever import Retriever

Hit = tuple[str, dict, float]

CORRECT, AMBIGUOUS, INCORRECT = "correct", "ambiguous", "incorrect"


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _strips(text: str) -> list[str]:
    """Sentence/line-level decomposition; keep only non-trivial strips."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if len(p.strip()) >= 20]


# Deterministic query reformulation: append domain keywords implied by the query.
# This is the offline stand-in for CRAG's web-search fallback.
_EXPANSION = {
    "cost": "generation dispatch redispatch economic",
    "load": "load scaling demand feasibility voltage limits",
    "infeasible": "feasibility voltage limits thermal constraints",
    "voltage": "voltage limits bus vlimits reactive power",
    "outage": "N-1 contingency security scopflow",
    "contingency": "N-1 contingency security scopflow",
    "thermal": "branch rate line loading thermal limits",
}


class CorrectiveRetriever:
    """CRAG retriever. Same public surface as ``Retriever``: ``.enabled`` and
    ``.retrieve(query) -> str``, so the engine and UI use it interchangeably.
    """

    def __init__(
        self,
        base: Retriever,
        *,
        tau_lower: float = 0.30,
        tau_upper: float = 0.50,
        strip_min_score: float = 0.30,
        max_refine_docs: int = 3,
        max_strips_per_doc: int = 4,
        grader: Optional[Callable[[str, list[Hit]], list[float]]] = None,
    ):
        self._base = base
        self.enabled = base.enabled
        self.k = base.k
        self.tau_lower = tau_lower
        self.tau_upper = tau_upper
        self.strip_min_score = strip_min_score
        self.max_refine_docs = max_refine_docs
        self.max_strips_per_doc = max_strips_per_doc
        # === Extension seam for an LLM grader ===
        # Provide a callable (query, hits) -> [score per hit in 0..1]; it replaces
        # the cosine scores in _grade(). Everything else stays identical.
        self._grader = grader
        # Per-session telemetry (journaled via describe() as rag_config): lets the
        # analysis see what corrective retrieval actually did in each run.
        self.stats = {"calls": 0, "correct": 0, "ambiguous": 0, "incorrect": 0,
                      "rewrites": 0, "withheld": 0, "grader_fallbacks": 0}

    def describe(self) -> dict:
        """Effective configuration for this run (shown in the UI / logged)."""
        return {
            "mode": "corrective",
            "enabled": self.enabled,
            "k": self.k,
            "min_score": getattr(self._base, "min_score", None),
            "store_path": getattr(self._base, "store_path", None),
            "collection": getattr(self._base, "collection", None),
            "tau_lower": self.tau_lower,
            "tau_upper": self.tau_upper,
            "strip_min_score": self.strip_min_score,
            "grader": self._grader_name(),
            "grader_detail": self._grader_detail(),
            "stats": {**dict(self.stats),
                      "retrieval_errors": getattr(self._base, "stats", {}).get("errors", 0)},
        }

    def _grader_detail(self) -> Optional[dict]:
        describe = getattr(self._grader, "describe", None)
        if callable(describe):
            try:
                return describe()
            except Exception:
                return None
        return None

    def _grader_name(self) -> str:
        """Name of the active grader: 'cosine' when none is injected, otherwise
        the grader's own reported name (e.g. 'reranker'), or its class name."""
        if self._grader is None:
            return "cosine"
        describe = getattr(self._grader, "describe", None)
        if callable(describe):
            try:
                return describe().get("grader", type(self._grader).__name__)
            except Exception:
                return type(self._grader).__name__
        return getattr(self._grader, "__name__", type(self._grader).__name__)

    # -- grading -------------------------------------------------------
    def _scores(self, query: str, hits: list[Hit]) -> list[float]:
        if self._grader is not None:
            try:
                return list(self._grader(query, hits))
            except Exception:
                # any grader failure -> fall back to the cosine scores, COUNTED so
                # the run is visibly not a clean run of the requested grader
                self.stats["grader_fallbacks"] += 1
        return [s for (_d, _m, s) in hits]

    def _grade(self, scores: list[float]) -> str:
        top = max(scores) if scores else 0.0
        if top >= self.tau_upper:
            verdict = CORRECT
        elif top < self.tau_lower:
            verdict = INCORRECT
        else:
            verdict = AMBIGUOUS
        self.stats[verdict] += 1
        return verdict

    # -- refinement (decompose-recompose) ------------------------------
    def _refine(self, query: str, hits: list[Hit]) -> list[tuple[str, float]]:
        try:
            q_emb = self._base.embed_text(query)
        except Exception:
            # can't embed -> fall back to whole surviving chunks
            return [(d, s) for (d, _m, s) in hits]
        out: list[tuple[str, float]] = []
        for (doc, _m, _s) in hits[: self.max_refine_docs]:
            strips = _strips(doc)[: self.max_strips_per_doc] or [doc]
            for strip in strips:
                try:
                    sc = _cosine(q_emb, self._base.embed_text(strip))
                except Exception:
                    continue
                if sc >= self.strip_min_score:
                    out.append((strip, sc))
        out.sort(key=lambda t: t[1], reverse=True)
        return out

    # -- query reformulation (offline fallback) ------------------------
    def _reformulate(self, query: str) -> str:
        ql = query.lower()
        adds: list[str] = []
        for key, exp in _EXPANSION.items():
            if key in ql:
                adds.extend(exp.split())
        if not adds:
            return query
        # de-duplicate while preserving order
        return query + " " + " ".join(dict.fromkeys(adds))

    # -- formatting ----------------------------------------------------
    def _format(self, refined: list[tuple[str, float]], verdict: str) -> str:
        lines = [
            f"[ref {i + 1} | score {sc:.2f}] {txt}"
            for i, (txt, sc) in enumerate(refined)
        ]
        header = (
            "Use the following retrieved reference material only when relevant; "
            "do not invent facts beyond it."
        )
        if verdict == AMBIGUOUS:
            header += (
                " (Retrieval confidence is moderate — rely on these references "
                "only where they clearly apply to the task.)"
            )
        return header + "\n" + "\n".join(lines)

    # -- public --------------------------------------------------------
    def retrieve(self, query: str, k: Optional[int] = None) -> str:
        """Return a CRAG-refined context block, or "" to fall back to baseline."""
        if not self.enabled or not query:
            return ""
        self.stats["calls"] += 1
        try:
            hits = self._base.query_hits(query, k or self.k)
        except Exception:
            return ""
        verdict = self._grade(self._scores(query, hits))

        if verdict == INCORRECT:
            rq = self._reformulate(query)
            if rq == query:
                self.stats["withheld"] += 1
                return ""  # nothing to correct with -> withhold context
            self.stats["rewrites"] += 1
            try:
                hits2 = self._base.query_hits(rq, k or self.k)
            except Exception:
                hits2 = []
            verdict2 = self._grade(self._scores(rq, hits2)) if hits2 else INCORRECT
            if verdict2 == INCORRECT:
                self.stats["withheld"] += 1
                return ""  # still weak -> withhold rather than mislead
            hits, verdict, query = hits2, verdict2, rq

        refined = self._refine(query, hits)
        if not refined:
            return ""
        return self._format(refined, verdict)
