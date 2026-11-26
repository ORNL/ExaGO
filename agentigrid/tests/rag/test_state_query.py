"""State-dependent retrieval query, per-iteration retrieval log, whole-chunk refinement."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agentigrid.rag import build_retriever, resolve_crag_refine
from agentigrid.rag.corrective import CorrectiveRetriever
from agentigrid.rag.query import build_query, build_state_query, resolve_query_mode
from agentigrid.rag.retriever import Retriever

GOAL = "Eliminate bus-voltage violations outside 0.95-1.05 pu"
EXEMPLAR = 'Goal: lower voltage. Step: {"action": "modify", "commands": [{"action": "set_gen_voltage", "bus": 3, "Vg": 1.02}]}. Result: 0 violations.'


def _opflow(**kw):
    base = dict(convergence_status="CONVERGED", feasibility_detail="infeasible", num_violations=2,
                voltage_min=0.97, voltage_max=1.06, max_line_loading_pct=71.3,
                violation_details=["Bus 100 voltage 1.060 > 1.050", "Bus 15 voltage 1.055 > 1.050"])
    base.update(kw)
    return SimpleNamespace(**base)


# -- query mode ------------------------------------------------------------
def test_query_mode_default_and_values(monkeypatch):
    monkeypatch.delenv("AGENTIGRID_RAG_QUERY", raising=False)
    assert resolve_query_mode() == "goal"
    monkeypatch.setenv("AGENTIGRID_RAG_QUERY", "State")
    assert resolve_query_mode() == "state"
    monkeypatch.setenv("AGENTIGRID_RAG_QUERY", "adaptive")
    with pytest.raises(ValueError):
        resolve_query_mode()


def test_goal_mode_returns_goal_unchanged():
    assert build_query("goal", GOAL, _opflow(), "Command errors:\nSkipped X") == GOAL


def test_state_query_contents_and_determinism():
    fb = "Command errors:\nSkipped SetGenVoltage(bus=100, Vg=1.04): No generator at bus 100\n"
    q1 = build_state_query(GOAL, _opflow(), fb)
    q2 = build_state_query(GOAL, _opflow(), fb)
    assert q1 == q2
    lines = q1.splitlines()
    assert lines[0] == GOAL
    assert "2 violations" in q1 and "max line loading 71.3%" in q1
    assert "Bus 100 voltage 1.060 > 1.050" in q1
    assert "Last command errors: Skipped SetGenVoltage(bus=100, Vg=1.04): No generator at bus 100" in q1
    assert "Command errors:" not in q1  # header dropped


def test_state_query_without_state_is_goal_and_is_capped():
    assert build_state_query(GOAL) == GOAL
    many = _opflow(violation_details=[f"Bus {i} voltage 1.07 > 1.05 " + "x" * 300 for i in range(20)])
    q = build_state_query(GOAL, many, "e\n" * 50)
    assert len(q) <= 1500


# -- retrieval trace -------------------------------------------------------
class _Store:
    def __init__(self, hits):
        self.hits = hits

    def query(self, q, k):
        return self.hits[:k]


def _basic(hits, min_score=0.35):
    r = Retriever(enabled=False, min_score=min_score)
    r.enabled, r._store = True, _Store(hits)
    return r


def test_basic_trace_records_hits_and_kept():
    r = _basic([(EXEMPLAR, {"source": "run_exemplars.md"}, 0.62), ("noise", {"source": "docs.md"}, 0.20)])
    out = r.retrieve("q")
    assert "[ref 1" in out
    t = r.last_trace
    assert t["action"] == "used" and t["kept"] == 1
    assert [h["source"] for h in t["hits"]] == ["run_exemplars.md", "docs.md"]


def test_basic_trace_empty():
    r = _basic([("noise", {"source": "docs.md"}, 0.10)])
    assert r.retrieve("q") == ""
    assert r.last_trace["action"] == "empty"


# -- corrective: whole-chunk refinement -------------------------------------
def test_whole_refine_keeps_exemplar_intact_and_drops_weak_chunks():
    base = _basic([(EXEMPLAR, {"source": "run_exemplars.md"}, 0.62),
                   ("SOPFLOW scenario files list wind profiles.", {"source": "sopflow.md"}, 0.25)])
    c = CorrectiveRetriever(base, refine="whole")
    out = c.retrieve("lower the bus voltage")
    assert EXEMPLAR in out                      # not split into strips
    assert "SOPFLOW" not in out                 # grade 0.25 < tau_lower 0.30
    t = c.last_trace
    assert t["verdict"] == "correct" and t["action"] == "used" and t["injected_refs"] == 1
    assert t["refine"] == "whole" and t["grader"] == "cosine"
    assert c.describe()["refine"] == "whole"


def test_corrective_trace_withheld():
    base = _basic([("unrelated", {"source": "x.md"}, 0.05)])
    c = CorrectiveRetriever(base, refine="whole")
    assert c.retrieve("zzz") == ""             # no expansion keyword -> withheld
    assert c.last_trace["verdict"] == "incorrect" and c.last_trace["action"] == "withheld"


def test_refine_mode_validation(monkeypatch):
    with pytest.raises(ValueError):
        CorrectiveRetriever(_basic([]), refine="sentences")
    monkeypatch.delenv("AGENTIGRID_CRAG_REFINE", raising=False)
    assert resolve_crag_refine() == "strips"
    monkeypatch.setenv("AGENTIGRID_CRAG_REFINE", "bogus")
    with pytest.raises(ValueError):
        resolve_crag_refine()


def test_default_corrective_is_still_strips(monkeypatch):
    monkeypatch.delenv("AGENTIGRID_CRAG_REFINE", raising=False)
    monkeypatch.setenv("AGENTIGRID_RAG_MODE", "corrective")
    monkeypatch.setenv("AGENTIGRID_RAG_STORE", "/nonexistent/store")
    r = build_retriever(host="http://127.0.0.1:9")
    assert isinstance(r, CorrectiveRetriever) and r.refine == "strips"


# -- runner guard and journal ------------------------------------------------

def test_journal_exports_rag_log(tmp_path):
    from agentigrid.exago_engine.journal import SearchJournal
    j = SearchJournal()
    j.rag_log = [{"iteration": 1, "query_mode": "state", "action": "used"}]
    p = tmp_path / "j.json"
    j.export_json(p)
    assert json.loads(p.read_text())["rag_log"][0]["query_mode"] == "state"
    j2 = SearchJournal()
    j2.export_json(p)
    assert "rag_log" not in json.loads(p.read_text())


def test_agent_loop_logs_each_retrieval():
    from agentigrid.exago_engine.agent_loop import AgentLoopController as AgentLoop
    from agentigrid.exago_engine.journal import SearchJournal
    loop = AgentLoop.__new__(AgentLoop)
    loop._rag_query_mode = "state"
    loop._journal = SearchJournal()
    loop._journal.rag_log = []
    loop._retriever = SimpleNamespace(enabled=True, last_trace={"mode": "corrective", "verdict": "ambiguous",
                                                                "action": "used"})
    loop._log_retrieval(2, "goal\nCurrent state: ...", "[ref 1 | score 0.41] text")
    rec = loop._journal.rag_log[0]
    assert rec["iteration"] == 2 and rec["query_mode"] == "state" and rec["verdict"] == "ambiguous"
    assert len(rec["query_sha256"]) == 64 and rec["injected_chars"] > 0
    loop._retriever = SimpleNamespace(enabled=False)
    loop._log_retrieval(3, "q", "")
    assert len(loop._journal.rag_log) == 1    # RAG off: nothing journaled


# -- error-first query, severity order, chunk hashes ----------------
from agentigrid.rag.query import build_error_query  # noqa: E402


def _mixed_opflow():
    return _opflow(violation_details=[
        "Bus 3: Vm=1.033 pu > 1.03 (overvoltage)",
        "Bus 4: Vm=1.031 pu > 1.03 (overvoltage)",
        "Branch 123-124: Sf=446.20 > Slim=400.00 MVA",
        "Bus 100: Vm=1.060 pu > 1.03 (overvoltage)",
        "Contingency summary unavailable",
        "Bus 7: Vm=0.930 pu < 0.95 (undervoltage)",
    ])


def test_error_query_has_errors_first_and_no_goal(monkeypatch):
    monkeypatch.setenv("AGENTIGRID_RAG_QUERY", "error")
    assert resolve_query_mode() == "error"
    fb = "Command errors:\nSkipped SetGenVoltage(bus=100, Vg=1.04): No generator at bus 100"
    q = build_query("error", GOAL, _opflow(), fb)
    lines = q.splitlines()
    assert lines[0] == "Command errors: Skipped SetGenVoltage(bus=100, Vg=1.04): No generator at bus 100"
    assert lines[1].startswith("Violations: ")
    assert lines[2].startswith("Current state: ")
    assert GOAL not in q
    assert build_query("error", GOAL, _opflow(), fb) == q          # deterministic


def test_error_query_falls_back_to_goal():
    clean = _opflow(num_violations=0, violation_details=[])
    assert build_error_query(GOAL, clean, None) == GOAL
    assert build_error_query(GOAL, None, None) == GOAL


def test_violations_sorted_by_severity_in_both_state_modes():
    for q in (build_state_query(GOAL, _mixed_opflow()), build_error_query(GOAL, _mixed_opflow())):
        viol = next(l for l in q.splitlines() if l.startswith("Violations: "))[len("Violations: "):].split("; ")
        # relative excess: branch 11.6%, bus 100 2.9%, bus 7 2.1%, bus 3 0.29%, bus 4 0.10%
        assert viol == ["Branch 123-124: Sf=446.20 > Slim=400.00 MVA",
                        "Bus 100: Vm=1.060 pu > 1.03 (overvoltage)",
                        "Bus 7: Vm=0.930 pu < 0.95 (undervoltage)",
                        "Bus 3: Vm=1.033 pu > 1.03 (overvoltage)",
                        "Bus 4: Vm=1.031 pu > 1.03 (overvoltage)"]


def test_unparseable_violations_keep_order_after_parsed_ones():
    op = _opflow(violation_details=["odd line A", "Bus 1: Vm=1.06 pu > 1.05 (overvoltage)", "odd line B"])
    q = build_error_query(GOAL, op)
    assert "Violations: Bus 1: Vm=1.06 pu > 1.05 (overvoltage); odd line A; odd line B" in q


def test_hits_carry_content_hash():
    r = _basic([(EXEMPLAR, {"source": "run_exemplars.md"}, 0.62), (EXEMPLAR, {"source": "copy.md"}, 0.61),
                ("other text", {"source": "x.md"}, 0.6)])
    r.retrieve("q")
    shas = [h["sha"] for h in r.last_trace["hits"]]
    assert len(shas[0]) == 16 and shas[0] == shas[1] != shas[2]
