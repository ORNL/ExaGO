"""Retrieval in the agent loop: off by default, generation-stage only, journaled."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentigrid.engine.agent_loop import AgentLoopController
from tests.agent_loop.test_agent_loop import (
    BASE_CASE, MockBackend, _has_base_case, _has_sample, _make_config, _make_llm_response,
    _make_sim_result, _objectives_response, _sample_stdout,
)

pytestmark = [
    pytest.mark.skipif(not _has_base_case, reason="Base case .m file not found"),
    pytest.mark.skipif(not _has_sample, reason="sample_opflow_output.txt not found"),
]

_COMPLETE = {"action": "complete", "reasoning": "Done", "findings": {"summary": "ok"}}
_REF = "[ref 1 | score 0.80] [source: test] Use set_all_bus_vlimits inside a modify action."


class _FakeRetriever:
    enabled = True

    def __init__(self):
        self.queries: list[str] = []

    def retrieve(self, query, k=None):
        self.queries.append(query)
        return _REF

    def describe(self):
        return {"mode": "basic", "enabled": True, "k": 3, "min_score": 0.35,
                "stats": {"calls": len(self.queries), "errors": 0, "empty": 0}}


def _run(tmp_path, monkeypatch, retriever=None, mode=None):
    if mode is None:
        monkeypatch.delenv("AGENTIGRID_RAG_MODE", raising=False)
        monkeypatch.delenv("AGENTIGRID_RAG", raising=False)
    else:
        monkeypatch.setenv("AGENTIGRID_RAG_MODE", mode)
    cfg = _make_config(tmp_path, max_iterations=3)
    backend = MockBackend([_objectives_response(), _make_llm_response(_COMPLETE)])
    phases: list[str] = []
    patches = [
        patch("agentigrid.engine.agent_loop.create_backend", return_value=backend),
        patch("agentigrid.engine.agent_loop.SimulationExecutor"),
    ]
    if retriever is not None:
        patches.append(patch("agentigrid.engine.agent_loop.build_retriever", return_value=retriever))
    with patches[0], patches[1] as mock_exec_cls, (patches[2] if len(patches) > 2 else _nullcontext()):
        mock_executor = MagicMock()
        mock_executor.run.return_value = _make_sim_result(stdout=_sample_stdout(), success=True)
        mock_exec_cls.return_value = mock_executor
        session = AgentLoopController(cfg, on_phase=lambda it, ph: phases.append(ph)).run(BASE_CASE, "Lower the cost")
    return session, backend, phases


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _iteration_prompts(backend):
    """User prompts of the per-iteration calls: the objective-extraction call comes
    first and the post-search goal classification last; this run has one iteration."""
    return [user for _, user in backend.calls[1:-1]]


def test_rag_is_off_by_default(tmp_path, monkeypatch):
    session, backend, phases = _run(tmp_path, monkeypatch)
    assert all("Reference Material" not in p for p in _iteration_prompts(backend))
    assert not any(ph.startswith("rag_retrieved:") for ph in phases)
    assert "llm_request" in phases
    assert session.journal.rag_enabled is False
    assert session.journal.rag_config["mode_env"] == "off"


def test_retrieved_references_reach_the_prompt_and_journal(tmp_path, monkeypatch):
    fake = _FakeRetriever()
    session, backend, phases = _run(tmp_path, monkeypatch, retriever=fake, mode="basic")
    prompts = _iteration_prompts(backend)
    assert len(prompts) == 1
    assert prompts[0].startswith("=== Section B: Reference Material (retrieved) ===")
    assert _REF in prompts[0]
    assert "Reference Material" not in backend.calls[-1][1]          # post-search analysis is not grounded
    assert fake.queries and set(fake.queries) == {"Lower the cost"}   # the query is the goal text
    i = phases.index("rag_retrieved:1:0.80")
    assert phases[i + 1] == "llm_request"
    assert session.journal.rag_enabled is True
    cfg = session.journal.rag_config
    assert cfg["mode"] == "basic" and cfg["mode_env"] == "basic" and cfg["stats"]["calls"] == len(fake.queries)


def test_failed_retrieval_leaves_the_prompt_unchanged(tmp_path, monkeypatch):
    class _Empty(_FakeRetriever):
        def retrieve(self, query, k=None):
            self.queries.append(query)
            return ""

    session, backend, phases = _run(tmp_path, monkeypatch, retriever=_Empty(), mode="basic")
    assert all("Reference Material" not in p for p in _iteration_prompts(backend))
    assert "rag_retrieved:0:0.00" in phases


def test_journal_file_records_rag_fields(tmp_path, monkeypatch):
    session, _, _ = _run(tmp_path, monkeypatch, retriever=_FakeRetriever(), mode="basic")
    out = tmp_path / "j.json"
    session.journal.export_json(out)
    data = json.loads(out.read_text())
    assert data["rag_enabled"] is True and data["rag_config"]["mode"] == "basic"
