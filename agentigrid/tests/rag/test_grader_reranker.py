"""Local cross-encoder grader for corrective RAG: seam contract, score mapping, failure
accounting, env wiring and journaled telemetry. Uses a fake model, so no
download or sentence-transformers install is needed."""

from __future__ import annotations

import pytest

from agentigrid.rag import build_retriever, resolve_crag_grader, _tau_overrides
from agentigrid.rag.corrective import CorrectiveRetriever
from agentigrid.rag.grader_reranker import RerankerGrader, to_unit_interval

HITS = [("chunk about voltage limits", {}, 0.41), ("chunk about generator cost", {}, 0.33)]


class FakeCE:
    def __init__(self, scores):
        self.scores = scores
        self.seen = None

    def predict(self, pairs, batch_size=16, show_progress_bar=False):
        self.seen = pairs
        return self.scores


def test_scores_pass_through_probabilities_and_sigmoid_logits():
    assert to_unit_interval([0.2, 0.9]) == [0.2, 0.9]
    out = to_unit_interval([3.0, -3.0])
    assert out[0] > 0.95 and out[1] < 0.05


def test_grader_scores_query_chunk_pairs():
    fake = FakeCE([0.8, 0.1])
    g = RerankerGrader(model_name="fake", _model=fake)
    assert g("set voltage band", HITS) == [0.8, 0.1]
    assert fake.seen == [("set voltage band", HITS[0][0]), ("set voltage band", HITS[1][0])]
    d = g.describe()
    assert d["grader"] == "reranker" and d["calls"] == 1 and d["failures"] == 0


def test_unavailable_model_is_counted_and_reported(monkeypatch):
    g = RerankerGrader(model_name="/nonexistent/model")
    monkeypatch.setattr(g, "_ensure_model", lambda: None)
    g._load_error = "OSError: not found"
    with pytest.raises(RuntimeError):
        g("q", HITS)
    assert g.failures == 1 and "fallback" in g.describe()["grader"]


class _Base:
    enabled, k, min_score = True, 3, 0.35

    def query_hits(self, q, k):
        return HITS

    def embed_text(self, t):
        return [1.0, 0.0]


def test_corrective_counts_fallbacks_and_verdicts():
    class Broken:
        def __call__(self, q, h):
            raise RuntimeError("boom")

        def describe(self):
            return {"grader": "reranker (fallback: cosine)"}

    cr = CorrectiveRetriever(_Base(), grader=Broken())
    cr.retrieve("voltage limits")
    d = cr.describe()
    assert d["stats"]["grader_fallbacks"] >= 1
    assert d["stats"]["calls"] == 1
    assert d["grader"] == "reranker (fallback: cosine)"
    assert sum(d["stats"][v] for v in ("correct", "ambiguous", "incorrect")) >= 1


def test_corrective_uses_reranker_scores_for_verdict():
    cr = CorrectiveRetriever(_Base(), grader=RerankerGrader(_model=FakeCE([0.9, 0.2])))
    cr.retrieve("voltage limits")
    assert cr.stats["correct"] == 1 and cr.stats["grader_fallbacks"] == 0


def test_env_selects_reranker_and_thresholds(monkeypatch):
    monkeypatch.setenv("AGENTIGRID_RAG_MODE", "corrective")
    monkeypatch.setenv("AGENTIGRID_CRAG_GRADER", "reranker")
    monkeypatch.setenv("AGENTIGRID_CRAG_TAU_LOWER", "0.2")
    monkeypatch.setenv("AGENTIGRID_CRAG_TAU_UPPER", "0.7")
    assert resolve_crag_grader() == "reranker"
    r = build_retriever(host="http://127.0.0.1:1")   # store may be unavailable; config still applies
    assert isinstance(r, CorrectiveRetriever)
    assert (r.tau_lower, r.tau_upper) == (0.2, 0.7)
    assert isinstance(r._grader, RerankerGrader)


def test_tau_overrides_validate(monkeypatch):
    monkeypatch.setenv("AGENTIGRID_CRAG_TAU_LOWER", "0.8")
    monkeypatch.setenv("AGENTIGRID_CRAG_TAU_UPPER", "0.2")
    with pytest.raises(ValueError):
        _tau_overrides()


def test_journal_carries_rag_config(tmp_path):
    import json
    from agentigrid.engine.journal import SearchJournal

    j = SearchJournal()
    j.rag_enabled = True
    j.rag_config = {"mode": "corrective", "mode_env": "corrective", "grader": "reranker",
                    "stats": {"grader_fallbacks": 0, "correct": 3, "ambiguous": 1, "incorrect": 0, "withheld": 0}}
    out = tmp_path / "j.json"
    j.export_json(out)
    data = json.loads(out.read_text())
    assert data["rag_enabled"] is True
    assert data["rag_config"]["grader"] == "reranker"
    assert data["rag_config"]["stats"]["correct"] == 3
