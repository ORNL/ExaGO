"""ExaGO agent loop and LLM API errors: retry, api_error / quota_exhausted, exit codes.

An API error is never a JSON parse failure, never "completed", and leaves no best result.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentigrid.backends.base import LLMResponse
from agentigrid.cli import exit_for_termination
from agentigrid.config import RetryConfig
from agentigrid.exago_engine.agent_loop import AgentLoopController
from tests.agent_loop.test_agent_loop import (
    BASE_CASE, MockBackend, _has_base_case, _has_sample, _make_config, _make_llm_response,
    _make_sim_result, _objectives_response, _sample_stdout,
)

pytestmark = [pytest.mark.skipif(not _has_base_case, reason="Base case .m file not found"),
              pytest.mark.skipif(not _has_sample, reason="sample_opflow_output.txt not found")]

E503 = ("OpenAI API error: Error code: 503 - [{'error': {'code': 503, 'message': 'This model is currently "
        "experiencing high demand.', 'status': 'UNAVAILABLE'}}]")
E429Q = ("OpenAI API error: Error code: 429 - [{'error': {'code': 429, 'message': 'You exceeded your current "
         "quota, please check your plan and billing details.', 'status': 'RESOURCE_EXHAUSTED'}}]")
MODIFY = {"action": "modify", "reasoning": "r", "description": "scale",
          "commands": [{"action": "scale_all_loads", "factor": 1.05}]}
COMPLETE = {"action": "complete", "reasoning": "done", "findings": {"summary": "done"}}


def _err(text: str, kind: str) -> LLMResponse:
    return LLMResponse(text, None, text, "test-model", "test", None, None, error_kind=kind)


def _run(tmp_path: Path, responses: list[LLMResponse]):
    cfg = _make_config(tmp_path, max_iterations=6)
    cfg = replace(cfg, llm=replace(cfg.llm, retry=RetryConfig(5, (10, 30, 60, 120, 240))))
    backend = MockBackend(responses)
    with patch("agentigrid.exago_engine.agent_loop.create_backend", return_value=backend), \
         patch("agentigrid.exago_engine.agent_loop.SimulationExecutor") as exec_cls:
        executor = MagicMock()
        executor.run.return_value = _make_sim_result(stdout=_sample_stdout(), success=True)
        exec_cls.return_value = executor
        controller = AgentLoopController(cfg, quiet=True)
        waits: list[float] = []
        controller._sleep = waits.append
        session = controller.run(BASE_CASE, "Scale loads a little")
    journal = json.loads(sorted((tmp_path / "workdir").glob("journal_*.json"))[-1].read_text())
    return session, controller, backend, waits, journal


def test_transient_then_success(tmp_path):
    session, controller, backend, waits, journal = _run(tmp_path, [
        _objectives_response(), _err(E503, "transient"),
        _make_llm_response(MODIFY), _make_llm_response(COMPLETE)])
    assert session.termination_reason == "completed" and session.error is None
    assert waits == [10]
    events = journal["llm_events"]
    assert len(events) == 1
    assert events[0]["phase"] == "search" and events[0]["iteration"] == 1
    assert events[0]["outcome"] == "retry" and events[0]["wait_s"] == 10 and "503" in events[0]["error"]
    assert journal["termination_reason"] == "completed"
    # tokens only for answers: the failed attempt is not in the ledger
    usage = journal["token_usage"]
    assert usage["calls_without_usage"] == 0
    assert usage["by_phase"]["search"]["calls"] == 2
    assert controller._consecutive_parse_failures == 0


def test_retries_exhausted_is_api_error(tmp_path, capsys):
    session, controller, backend, waits, journal = _run(tmp_path, [
        _objectives_response(), _err(E503, "transient")])          # the mock repeats the last reply
    assert session.termination_reason == "api_error"
    assert "503" in session.error
    assert waits == [10, 30, 60, 120, 240]
    assert backend._call_count == 1 + 6                             # no post-search analysis call
    assert [e["outcome"] for e in journal["llm_events"]] == ["retry"] * 5 + ["api_error"]
    assert journal["termination_reason"] == "api_error"
    assert controller._consecutive_parse_failures == 0             # never counted as parse failures
    assert session.goal_classification is None and session.analysis_text is None
    out = capsys.readouterr().out
    assert "Best solution:  none (session ended: api_error)" in out
    assert "Best objective" not in out
    with pytest.raises(SystemExit) as e:
        exit_for_termination(session.termination_reason)
    assert e.value.code == 3


def test_quota_is_not_retried(tmp_path):
    session, controller, backend, waits, journal = _run(tmp_path, [
        _objectives_response(), _err(E429Q, "quota"), _make_llm_response(COMPLETE)])
    assert session.termination_reason == "quota_exhausted"
    assert waits == [] and backend._call_count == 2
    assert [e["outcome"] for e in journal["llm_events"]] == ["quota_exhausted"]
    with pytest.raises(SystemExit) as e:
        exit_for_termination(session.termination_reason)
    assert e.value.code == 4


def test_quota_in_objective_extraction_stops_before_the_loop(tmp_path):
    session, controller, backend, waits, journal = _run(tmp_path, [
        _err(E429Q, "quota"), _make_llm_response(COMPLETE)])
    assert session.termination_reason == "quota_exhausted"
    assert backend._call_count == 1
    assert len(session.journal) == 1                                 # base case only
    assert journal["llm_events"][0]["phase"] == "objective_extraction"


def test_analysis_call_failure_keeps_the_search_result(tmp_path, capsys, caplog):
    # objectives, modify, complete; the post-search analysis call then hits the quota
    session, controller, backend, waits, journal = _run(tmp_path, [
        _objectives_response(), _make_llm_response(MODIFY), _make_llm_response(COMPLETE),
        _err(E429Q, "quota")])
    assert backend._call_count == 4
    assert session.termination_reason == "completed" and session.error is None
    assert session.analysis_error.startswith("quota_exhausted: ") and "429" in session.analysis_error
    assert session.goal_classification is None and session.analysis_text is None
    assert journal["termination_reason"] == "completed"
    assert journal["analysis_error"] == session.analysis_error
    assert [(e["phase"], e["outcome"]) for e in journal["llm_events"]] == [("analysis", "quota_exhausted")]
    assert "Post-search analysis call failed" in caplog.text
    out = capsys.readouterr().out
    assert "Termination:    completed" in out
    assert "Analysis:       not generated (quota_exhausted:" in out
    assert "Best solution:  none" not in out
    assert "Best objective" in out                                   # from the journal statistics
    assert exit_for_termination(session.termination_reason) is None


def test_no_analysis_error_field_when_the_call_succeeds(tmp_path):
    session, controller, backend, waits, journal = _run(tmp_path, [
        _objectives_response(), _make_llm_response(COMPLETE)])
    assert session.analysis_error is None and "analysis_error" not in journal


def test_real_parse_failures_are_not_completed(tmp_path):
    session, controller, backend, waits, journal = _run(tmp_path, [
        _objectives_response(), _make_llm_response(None, raw_text="not json")])
    assert session.termination_reason == "parse_failures"
    assert "llm_events" not in journal and waits == []
    with pytest.raises(SystemExit) as e:
        exit_for_termination(session.termination_reason)
    assert e.value.code == 1


@pytest.mark.parametrize("termination", ["completed", "max_iterations", "user_stopped"])
def test_normal_endings_exit_zero(termination):
    assert exit_for_termination(termination) is None
