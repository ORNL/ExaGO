"""Every LLM call of an ExaGO session is counted, with its phase.

Covers:
- a session with objective extraction, a steering directive, search iterations
  (incl. the 'analyze' LLM fallback) and the post-search analysis: ledger calls
  == backend calls, totals == sum of all calls, phases with the right counts
- the journal export carries the ``token_usage`` block; an old journal without
  it still loads
- session checkpoints save ``token_usage``; old checkpoints give totals only
- report regeneration records its call under ``report`` on top of the
  journal's own usage
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentigrid.backends.base import LLMResponse
from agentigrid.exago_engine.agent_loop import AgentLoopController
from agentigrid.exago_engine.journal import SearchJournal
from agentigrid.exago_engine.regenerate_report import (
    build_session_stub,
    load_journal_export,
    regenerate_from_journal,
)
from agentigrid.exago_engine.session_io import load_session, save_session
from agentigrid.token_usage import TokenLedger
from tests.agent_loop.test_agent_loop import (
    BASE_CASE,
    MockBackend,
    _has_base_case,
    _has_sample,
    _make_config,
    _make_llm_response,
    _make_sim_result,
    _objectives_response,
    _sample_stdout,
)


def _text_response(text: str, prompt: int, completion: int) -> LLMResponse:
    return LLMResponse(raw_text=text, json_data=None, json_error=None, model="test-model",
                       backend="test", prompt_tokens=prompt, completion_tokens=completion)


def _classification_response() -> LLMResponse:
    return _text_response(
        "Analysis.\n```json\n{\"goal_type\": \"cost_minimization\", \"best_iteration\": 0, "
        "\"best_iteration_rationale\": \"base\"}\n```", 700, 300,
    )


def _run_session(tmp_path: Path) -> tuple:
    cfg = _make_config(tmp_path, max_iterations=10)
    responses = [
        _objectives_response(),                                  # objective extraction (50+20)
        _objectives_response(),                                  # steering directive (50+20)
        _make_llm_response({                                     # iter 1 (100+50)
            "action": "analyze", "reasoning": "curious",
            "query": "tell me a short story about the sky",      # no pattern: LLM fallback
        }),
        _text_response("Once upon a time.", 30, 10),             # analyze fallback (30+10)
        _make_llm_response({                                     # iter 2 (100+50)
            "action": "modify", "reasoning": "+10%", "mode": "fresh",
            "description": "Scale all loads +10%",
            "commands": [{"action": "scale_all_loads", "factor": 1.1}],
        }),
        _make_llm_response({                                     # iter 3 (100+50)
            "action": "complete", "reasoning": "done",
            "findings": {"summary": "Done"},
        }),
        _classification_response(),                              # post-search analysis (700+300)
    ]
    backend = MockBackend(responses)
    with patch("agentigrid.exago_engine.agent_loop.create_backend", return_value=backend), \
         patch("agentigrid.exago_engine.agent_loop.SimulationExecutor") as mock_exec_cls:
        mock_executor = MagicMock()
        mock_executor.run.return_value = _make_sim_result(stdout=_sample_stdout())
        mock_exec_cls.return_value = mock_executor
        controller = AgentLoopController(cfg)
        controller.inject_steering("Also keep generation cost low")
        session = controller.run(BASE_CASE, "Find max feasible load increase")
    return cfg, backend, controller, session


@pytest.mark.skipif(not _has_base_case, reason="Base case .m file not found")
@pytest.mark.skipif(not _has_sample, reason="sample_opflow_output.txt not found")
class TestExagoSessionTokens:

    def test_every_call_counted(self, tmp_path: Path):
        _, backend, controller, session = _run_session(tmp_path)
        ledger = session.token_ledger

        assert ledger is controller.token_ledger
        assert len(backend.calls) == 7
        assert len(ledger) == len(backend.calls)
        by_phase = ledger.by_phase
        assert {p: a["calls"] for p, a in by_phase.items()} == {
            "objective_extraction": 1, "search": 4, "analysis": 1, "steering": 1,
        }
        assert by_phase["search"]["total_tokens"] == 3 * 150 + 40
        assert by_phase["analysis"]["total_tokens"] == 1000
        assert ledger.total_prompt == 50 + 50 + 3 * 100 + 30 + 700
        assert ledger.total_completion == 20 + 20 + 3 * 50 + 10 + 300
        assert ledger.total == sum(
            (c["prompt_tokens"] or 0) + (c["completion_tokens"] or 0) for c in ledger.calls
        )
        # Session totals now cover every phase
        assert session.total_prompt_tokens == ledger.total_prompt
        assert session.total_completion_tokens == ledger.total_completion
        # The analyze fallback is recorded with the iteration it ran in
        search_iters = [c["iteration"] for c in ledger.calls if c["phase"] == "search"]
        assert search_iters == [1, 1, 2, 3]

    def test_char_lengths_match_backend_calls(self, tmp_path: Path):
        _, backend, _, session = _run_session(tmp_path)
        ledger = session.token_ledger
        assert [c["prompt_chars"] for c in ledger.calls] == [
            len(system) + len(user) for system, user in backend.calls
        ]
        assert all(c["completion_chars"] is not None for c in ledger.calls)
        assert ledger.calls[3]["completion_chars"] == len("Once upon a time.")   # analyze fallback
        assert ledger.total_prompt_chars == sum(c["prompt_chars"] for c in ledger.calls)
        assert ledger.total_completion_chars == sum(c["completion_chars"] for c in ledger.calls)
        assert ledger.to_dict()["prompt_chars"] == ledger.total_prompt_chars

    def test_journal_has_token_usage(self, tmp_path: Path):
        cfg, _, controller, _ = _run_session(tmp_path)
        exports = sorted(cfg.output.workdir.glob("journal_*.json"))
        assert exports, "journal was not exported"
        data = json.loads(exports[-1].read_text())
        assert data["token_usage"] == controller.token_ledger.to_dict()
        assert data["token_usage"]["calls"] == 7
        # and it loads back onto the journal
        assert load_journal_export(exports[-1]).token_usage == data["token_usage"]

    def test_classification_prompt_gets_tokens_before_analysis(self, tmp_path: Path):
        _, backend, _, _ = _run_session(tmp_path)
        before = 50 + 20 + 50 + 20 + 3 * 150 + 40
        _, classification_user_prompt = backend.calls[-1]
        assert f"Tokens used before this analysis: ~{before:,}" in classification_user_prompt

    def test_summary_line_printed(self, tmp_path: Path, capsys):
        _run_session(tmp_path)
        out = capsys.readouterr().out
        assert ("Tokens: 1,630 total (1,130 prompt + 500 completion) in 7 LLM calls; "
                "objective extraction 70, search 490, analysis 1,000, steering 70") in out


def test_old_journal_without_token_usage_loads(tmp_path: Path):
    journal = SearchJournal()
    path = tmp_path / "old.json"
    journal.export_json(path)
    raw = json.loads(path.read_text())
    assert "token_usage" not in raw
    loaded = load_journal_export(path)
    assert loaded.token_usage is None
    assert len(TokenLedger.from_dict(loaded.token_usage)) == 0


def _save(tmp_path: Path, token_usage: dict | None) -> Path:
    return save_session(
        save_dir=tmp_path / "save", goal="g", application="opflow",
        base_case_path=tmp_path / "case.m", config_path=None, journal=SearchJournal(),
        steering_history=[], active_steering_directives=[], current_network=None,
        total_prompt_tokens=300, total_completion_tokens=100, last_iteration=0,
        token_usage=token_usage,
    )


def test_session_checkpoint_round_trip(tmp_path: Path):
    ledger = TokenLedger()
    ledger.record("search", _text_response("", 300, 100), 1)
    loaded = load_session(_save(tmp_path, ledger.to_dict()))
    assert TokenLedger.from_dict(loaded["token_usage"]).calls == ledger.calls


def test_old_session_checkpoint_gives_legacy_totals(tmp_path: Path):
    save_dir = _save(tmp_path, None)
    raw = json.loads((save_dir / "session.json").read_text())
    raw["format_version"] = "1.1"
    raw.pop("token_usage")
    (save_dir / "session.json").write_text(json.dumps(raw))
    loaded = load_session(save_dir)
    assert loaded["token_usage"] is None
    ledger = TokenLedger.from_legacy_totals(loaded["total_prompt_tokens"],
                                            loaded["total_completion_tokens"])
    assert ledger.total == 400 and ledger.by_phase["search"]["calls"] == 1


def test_regenerate_records_report_call(tmp_path: Path):
    from tests.journal.test_regenerate_report import _cfg

    journal = SearchJournal()
    session_ledger = TokenLedger()
    session_ledger.record("search", _text_response("", 1000, 200), 1)
    journal.token_usage = session_ledger.to_dict()
    export = tmp_path / "journal.json"
    journal.export_json(export)

    backend = MagicMock()
    backend.complete.return_value = _text_response("no json here", 400, 100)
    captured = {}

    class _SpyGenerator:
        def generate(self, session, **kwargs):
            captured["session"] = session
            return b"%PDF-1.4\n%%EOF"

    with patch("agentigrid.exago_engine.regenerate_report._import_report_generator",
               return_value=_SpyGenerator):
        regenerate_from_journal(journal_json_path=export, config=_cfg(tmp_path),
                                backend=backend, out_dir=tmp_path / "out", goal="g",
                                generate_analysis=True)

    session = captured["session"]
    assert backend.complete.call_count == 1
    assert {p: a["calls"] for p, a in session.token_ledger.by_phase.items()} == {
        "search": 1, "report": 1,
    }
    assert session.total_prompt_tokens + session.total_completion_tokens == 1700


def test_session_stub_takes_journal_usage(tmp_path: Path):
    from tests.journal.test_regenerate_report import _cfg

    journal = SearchJournal()
    ledger = TokenLedger()
    ledger.record("analysis", _text_response("", 10, 5))
    journal.token_usage = ledger.to_dict()
    session = build_session_stub(journal, _cfg(tmp_path), goal="g")
    assert session.total_prompt_tokens == 10 and session.total_completion_tokens == 5
    assert len(session.token_ledger) == 1


def _stub_session(tmp_path: Path):
    from tests.journal.test_regenerate_report import _cfg

    journal = SearchJournal()
    ledger = TokenLedger()
    ledger.record("search", _text_response("", 1000, 200), 1)
    journal.token_usage = ledger.to_dict()
    return build_session_stub(journal, _cfg(tmp_path), goal="g")


def test_gui_summary_analysis_uses_session_ledger(tmp_path: Path):
    from launcher.session_manager import SessionManager

    session = _stub_session(tmp_path)
    backend = MockBackend([_classification_response()])
    with patch("launcher.session_manager.create_backend", return_value=backend):
        SessionManager().get_summary_analysis(session)

    assert len(backend.calls) == 1
    assert {p: a["calls"] for p, a in session.token_ledger.by_phase.items()} == {
        "search": 1, "analysis": 1,
    }
    assert session.total_prompt_tokens == 1700
    assert session.total_completion_tokens == 500
    assert session.journal.token_usage == session.token_ledger.to_dict()


def test_report_token_line_shows_breakdown(tmp_path: Path):
    report_generator = pytest.importorskip("launcher.report_generator")

    session = _stub_session(tmp_path)
    session.token_ledger.record("analysis", _text_response("", 300, 100))
    session.total_prompt_tokens = session.token_ledger.total_prompt
    session.total_completion_tokens = session.token_ledger.total_completion
    assert report_generator._token_usage_line(session) == (
        "Token usage: 1,600 in 2 LLM calls (search 1,200, analysis 400)"
    )
    session.total_prompt_tokens = session.total_completion_tokens = 0
    assert report_generator._token_usage_line(session) == "Token usage: N/A"
