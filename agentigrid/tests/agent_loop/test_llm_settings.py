"""The ExaGO journal records the run's LLM settings, without any key material."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agentigrid.exago_engine.agent_loop import AgentLoopController
from agentigrid.exago_engine.regenerate_report import load_journal_export
from tests.agent_loop.test_agent_loop import (
    BASE_CASE, _has_base_case, _has_sample, _make_config, _make_sim_result, _sample_stdout,
)

SECRET = "sk-test-SECRET-do-not-leak-0123456789"


def _completion(content: str, prompt: int = 20, completion: int = 5, total: int | None = None):
    usage = SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion,
                            total_tokens=total if total is not None else prompt + completion)
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                           usage=usage)


@pytest.mark.skipif(not _has_base_case, reason="Base case .m file not found")
@pytest.mark.skipif(not _has_sample, reason="sample_opflow_output.txt not found")
def test_exago_journal_llm_settings(tmp_path: Path, monkeypatch, capsys):
    pytest.importorskip("openai")
    from agentigrid.backends.openai_backend import OpenAIBackend

    monkeypatch.setenv("TEST_KEY", SECRET)               # _make_config uses api_key_env TEST_KEY
    cfg = _make_config(tmp_path, max_iterations=3)
    client = MagicMock()
    client.base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    client.chat.completions.create.side_effect = [
        _completion('{"objectives": [{"name": "generation_cost", "direction": "minimize", '
                    '"priority": "primary"}]}'),
        _completion('{"action": "complete", "reasoning": "done", "findings": {"summary": "ok"}}',
                    prompt=100, completion=10, total=150),          # 40 thinking tokens
        _completion('Analysis.\n```json\n{"goal_type": "cost_minimization", "best_iteration": 0, '
                    '"best_iteration_rationale": "base"}\n```'),
    ]
    with patch("agentigrid.backends.openai_backend.openai.OpenAI", return_value=client):
        backend = OpenAIBackend(cfg.llm)
    with patch("agentigrid.exago_engine.agent_loop.create_backend", return_value=backend), \
         patch("agentigrid.exago_engine.agent_loop.SimulationExecutor") as mock_exec_cls:
        mock_executor = MagicMock()
        mock_executor.run.return_value = _make_sim_result(stdout=_sample_stdout())
        mock_exec_cls.return_value = mock_executor
        session = AgentLoopController(cfg).run(BASE_CASE, "Minimize cost")

    export = sorted(cfg.output.workdir.glob("journal_*.json"))[-1]
    text = export.read_text()
    assert SECRET not in text
    settings = json.loads(text)["llm_settings"]
    assert settings["backend"] == "openai" and settings["model"] == "test-model"
    assert settings["endpoint_host"] == "generativelanguage.googleapis.com"
    assert settings["temperature"] == 0.3 and settings["reasoning_effort"] is None
    assert settings["max_tokens"] == 4096
    assert "openai" in settings["packages"] and "ollama" in settings["packages"]
    assert load_journal_export(export).llm_settings == settings
    # derived reasoning tokens reach the ledger and the summary line
    ledger = session.token_ledger
    assert ledger.total_reasoning == 40 and ledger.calls_with_total_mismatch == 1
    out = capsys.readouterr().out
    assert out.count("LLM: openai test-model @ generativelanguage.googleapis.com") == 1
    assert "prompt+completion (+40 reasoning)" in out
    assert SECRET not in out
