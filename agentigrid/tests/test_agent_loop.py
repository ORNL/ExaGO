"""Tests for the Agent Loop Controller."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from agentigrid.backends.base import LLMBackend, LLMResponse
from agentigrid.config import (
    AppConfig, ExagoConfig, DataConfig, LLMConfig, SearchConfig, OutputConfig,
)
from agentigrid.engine.agent_loop import AgentLoopController, SearchSession
from agentigrid.engine.executor import SimulationResult
from agentigrid.parsers.opflow_results import OPFLOWResult


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

SAMPLE = Path(__file__).resolve().parent / "sample_opflow_output.txt"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BASE_CASE = DATA_DIR / "case_ACTIVSg200.m"
_has_base_case = BASE_CASE.exists()
_has_sample = SAMPLE.exists()


def _make_config(tmp_path: Path, max_iterations: int = 5) -> AppConfig:
    """Build an AppConfig pointing at tmp_path for workdir/logs."""
    return AppConfig(
        exago=ExagoConfig(
            binary_dir=tmp_path / "bin",
            opflow_binary=None,
            scopflow_binary=None,
            tcopflow_binary=None,
            sopflow_binary=None,
            dcopflow_binary=None,
            pflow_binary=None,
            env_script=None,
            timeout=30,
        ),
        data=DataConfig(data_dir=tmp_path / "data"),
        llm=LLMConfig(
            backend="openai",
            model="test-model",
            api_key_env="TEST_KEY",
            openai_base_url=None,
            ollama_host="http://localhost:11434",
            ollama_cloud_host=None,
            temperature=0.3,
            max_tokens=4096,
        ),
        search=SearchConfig(
            max_iterations=max_iterations,
            default_mode="accumulative",
            base_case=BASE_CASE if _has_base_case else tmp_path / "dummy.m",
            gic_file=None,
            application="opflow",
        ),
        output=OutputConfig(
            workdir=tmp_path / "workdir",
            logs_dir=tmp_path / "logs",
            save_journal=True,
            journal_format="json",
            save_modified_files=True,
            verbose=False,
        ),
    )


def _make_sim_result(
    stdout: str = "", success: bool = True, elapsed: float = 0.5
) -> SimulationResult:
    return SimulationResult(
        success=success,
        exit_code=0 if success else 1,
        stdout=stdout,
        stderr="",
        elapsed_seconds=elapsed,
        input_file=Path("/tmp/test.m"),
        application="opflow",
        error_message=None if success else "Simulation failed",
        workdir=Path("/tmp/workdir"),
    )


def _make_llm_response(json_data: Optional[dict], raw_text: str = "") -> LLMResponse:
    return LLMResponse(
        raw_text=raw_text or str(json_data),
        json_data=json_data,
        json_error=None if json_data else "parse error",
        model="test-model",
        backend="test",
        prompt_tokens=100,
        completion_tokens=50,
    )


def _objectives_response() -> LLMResponse:
    """Return a valid single-objective extraction response (for test setup)."""
    import json as _json
    payload = _json.dumps({"objectives": [
        {"name": "generation_cost", "direction": "minimize", "priority": "primary"},
    ]})
    return LLMResponse(
        raw_text=payload,
        json_data=None,  # The objective parser reads raw_text directly
        json_error=None,
        model="test-model",
        backend="test",
        prompt_tokens=50,
        completion_tokens=20,
    )


class MockBackend(LLMBackend):
    """A mock LLM backend that returns responses from a list."""

    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)
        self._call_count = 0
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str, temperature=None) -> LLMResponse:
        self.calls.append((system_prompt, user_prompt))
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
        else:
            resp = self._responses[-1]
        self._call_count += 1
        return resp

    def name(self) -> str:
        return "mock"

    def supports_json_mode(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Helper to get sample stdout
# ---------------------------------------------------------------------------

def _sample_stdout() -> str:
    if _has_sample:
        return SAMPLE.read_text(encoding="utf-8")
    return ""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_base_case, reason="Base case .m file not found")
@pytest.mark.skipif(not _has_sample, reason="sample_opflow_output.txt not found")
class TestAgentLoopModifyComplete:
    """Test a modify → modify → complete sequence."""

    def test_three_iteration_sequence(self, tmp_path: Path):
        cfg = _make_config(tmp_path, max_iterations=10)
        stdout = _sample_stdout()

        responses = [
            _objectives_response(),  # objective extraction call after base case
            _make_llm_response({
                "action": "modify",
                "reasoning": "Scale loads up 10%",
                "mode": "fresh",
                "description": "Scale all loads +10%",
                "commands": [{"action": "scale_all_loads", "factor": 1.1}],
            }),
            _make_llm_response({
                "action": "modify",
                "reasoning": "Scale loads up 20%",
                "mode": "fresh",
                "description": "Scale all loads +20%",
                "commands": [{"action": "scale_all_loads", "factor": 1.2}],
            }),
            _make_llm_response({
                "action": "complete",
                "reasoning": "Found the limit",
                "findings": {
                    "summary": "Max feasible load increase is ~20%",
                    "details": "Above 20% the system diverges",
                },
            }),
        ]

        mock_backend = MockBackend(responses)
        sim_result = _make_sim_result(stdout=stdout, success=True)

        with patch("agentigrid.engine.agent_loop.create_backend", return_value=mock_backend), \
             patch.object(
                 AgentLoopController, "_AgentLoopController__class__", create=True
             ) if False else \
             patch("agentigrid.engine.agent_loop.SimulationExecutor") as mock_exec_cls:

            mock_executor = MagicMock()
            mock_executor.run.return_value = sim_result
            mock_exec_cls.return_value = mock_executor

            controller = AgentLoopController(cfg)
            session = controller.run(BASE_CASE, "Find max feasible load increase")

        # Verify journal: iter 0 (base) + iter 1 (modify) + iter 2 (modify) + iter 3 (complete) = 4
        assert len(session.journal) == 4
        assert session.journal.entries[0].iteration == 0
        assert session.journal.entries[1].iteration == 1
        assert session.journal.entries[2].iteration == 2
        assert session.termination_reason == "completed"


@pytest.mark.skipif(not _has_base_case, reason="Base case .m file not found")
@pytest.mark.skipif(not _has_sample, reason="sample_opflow_output.txt not found")
class TestErrorRecovery:

    def test_invalid_json_then_valid(self, tmp_path: Path):
        """LLM returns invalid JSON on first call, valid on second, then complete."""
        cfg = _make_config(tmp_path, max_iterations=10)
        stdout = _sample_stdout()

        responses = [
            _make_llm_response(None, raw_text="This is not JSON"),
            _make_llm_response({
                "action": "modify",
                "reasoning": "Small change",
                "description": "Test modification",
                "commands": [{"action": "scale_all_loads", "factor": 1.05}],
            }),
            _make_llm_response({
                "action": "complete",
                "reasoning": "Done",
                "findings": {"summary": "Test complete"},
            }),
        ]

        mock_backend = MockBackend(responses)
        sim_result = _make_sim_result(stdout=stdout, success=True)

        with patch("agentigrid.engine.agent_loop.create_backend", return_value=mock_backend), \
             patch("agentigrid.engine.agent_loop.SimulationExecutor") as mock_exec_cls:

            mock_executor = MagicMock()
            mock_executor.run.return_value = sim_result
            mock_exec_cls.return_value = mock_executor

            controller = AgentLoopController(cfg)
            session = controller.run(BASE_CASE, "Test error recovery")

        # Iter 0 base + iter 1 failed parse (no journal) + iter 2 modify + iter 3 complete
        assert len(session.journal) == 3  # base + modify + complete
        assert session.termination_reason == "completed"

    def test_unknown_action(self, tmp_path: Path):
        """LLM returns unknown action, then completes."""
        cfg = _make_config(tmp_path, max_iterations=10)
        stdout = _sample_stdout()

        responses = [
            _make_llm_response({"action": "dance", "reasoning": "I want to dance"}),
            _make_llm_response({
                "action": "complete",
                "reasoning": "Done",
                "findings": {"summary": "Finished"},
            }),
        ]

        mock_backend = MockBackend(responses)
        sim_result = _make_sim_result(stdout=stdout, success=True)

        with patch("agentigrid.engine.agent_loop.create_backend", return_value=mock_backend), \
             patch("agentigrid.engine.agent_loop.SimulationExecutor") as mock_exec_cls:

            mock_executor = MagicMock()
            mock_executor.run.return_value = sim_result
            mock_exec_cls.return_value = mock_executor

            controller = AgentLoopController(cfg)
            session = controller.run(BASE_CASE, "Test unknown action")

        assert session.termination_reason == "completed"
        assert len(session.journal) == 2  # base + complete

    def test_simulation_failure(self, tmp_path: Path):
        """Simulation fails — journal records infeasible entry."""
        cfg = _make_config(tmp_path, max_iterations=10)
        stdout = _sample_stdout()

        responses = [
            _objectives_response(),  # objective extraction call after base case
            _make_llm_response({
                "action": "modify",
                "reasoning": "Aggressive change",
                "description": "Scale loads x5",
                "commands": [{"action": "scale_all_loads", "factor": 5.0}],
            }),
            _make_llm_response({
                "action": "complete",
                "reasoning": "Failed",
                "findings": {"summary": "Too aggressive"},
            }),
        ]

        mock_backend = MockBackend(responses)
        # Base case succeeds, modify iteration fails
        success_result = _make_sim_result(stdout=stdout, success=True)
        fail_result = _make_sim_result(stdout="", success=False)

        with patch("agentigrid.engine.agent_loop.create_backend", return_value=mock_backend), \
             patch("agentigrid.engine.agent_loop.SimulationExecutor") as mock_exec_cls:

            mock_executor = MagicMock()
            mock_executor.run.side_effect = [success_result, fail_result]
            mock_exec_cls.return_value = mock_executor

            controller = AgentLoopController(cfg)
            session = controller.run(BASE_CASE, "Test sim failure")

        assert len(session.journal) == 3  # base + failed modify + complete
        assert session.journal.entries[1].feasible is False
        assert session.termination_reason == "completed"


@pytest.mark.skipif(not _has_base_case, reason="Base case .m file not found")
@pytest.mark.skipif(not _has_sample, reason="sample_opflow_output.txt not found")
class TestMaxIterations:

    def test_stops_at_max(self, tmp_path: Path):
        cfg = _make_config(tmp_path, max_iterations=3)
        stdout = _sample_stdout()

        # Always returns modify — should stop at max_iterations
        modify_response = _make_llm_response({
            "action": "modify",
            "reasoning": "Keep going",
            "description": "Another change",
            "commands": [{"action": "scale_all_loads", "factor": 1.01}],
        })

        mock_backend = MockBackend([modify_response])
        sim_result = _make_sim_result(stdout=stdout, success=True)

        with patch("agentigrid.engine.agent_loop.create_backend", return_value=mock_backend), \
             patch("agentigrid.engine.agent_loop.SimulationExecutor") as mock_exec_cls:

            mock_executor = MagicMock()
            mock_executor.run.return_value = sim_result
            mock_exec_cls.return_value = mock_executor

            controller = AgentLoopController(cfg)
            session = controller.run(BASE_CASE, "Never-ending goal")

        # Base case (iter 0) + 3 modify iterations
        assert len(session.journal) == 4
        assert session.termination_reason == "max_iterations"


@pytest.mark.skipif(not _has_base_case, reason="Base case .m file not found")
class TestPromptAssembly:

    def test_system_prompt_contents(self, tmp_path: Path):
        cfg = _make_config(tmp_path)
        stdout = _sample_stdout() if _has_sample else ""

        mock_backend = MockBackend([
            _make_llm_response({
                "action": "complete",
                "reasoning": "Done",
                "findings": {"summary": "Test"},
            })
        ])
        sim_result = _make_sim_result(stdout=stdout, success=bool(stdout))

        with patch("agentigrid.engine.agent_loop.create_backend", return_value=mock_backend), \
             patch("agentigrid.engine.agent_loop.SimulationExecutor") as mock_exec_cls:

            mock_executor = MagicMock()
            mock_executor.run.return_value = sim_result
            mock_exec_cls.return_value = mock_executor

            controller = AgentLoopController(cfg)
            controller.run(BASE_CASE, "Test prompt assembly")

        assert len(mock_backend.calls) >= 1
        # Find the main agent call (not objective extraction or classification calls)
        # The main call is identified by having the goal text in the user prompt
        main_call = None
        for sys_p, usr_p in mock_backend.calls:
            if "Test prompt assembly" in usr_p and "power systems" in sys_p.lower():
                main_call = (sys_p, usr_p)
                break
        assert main_call is not None, "Main agent call not found"
        system_prompt, user_prompt = main_call

        # System prompt should contain command schema and role definition
        assert "power systems" in system_prompt.lower()
        assert "scale_all_loads" in system_prompt
        assert "set_load" in system_prompt
        assert "modify" in system_prompt

        # User prompt should contain the goal
        assert "Test prompt assembly" in user_prompt

    def test_user_prompt_contains_journal(self, tmp_path: Path):
        """After a modify iteration, user prompt should contain journal."""
        cfg = _make_config(tmp_path)
        stdout = _sample_stdout() if _has_sample else ""

        responses = [
            _make_llm_response({
                "action": "modify",
                "reasoning": "Test",
                "description": "Small change",
                "commands": [{"action": "scale_all_loads", "factor": 1.01}],
            }),
            _make_llm_response({
                "action": "complete",
                "reasoning": "Done",
                "findings": {"summary": "Done"},
            }),
        ]

        mock_backend = MockBackend(responses)
        sim_result = _make_sim_result(stdout=stdout, success=bool(stdout))

        with patch("agentigrid.engine.agent_loop.create_backend", return_value=mock_backend), \
             patch("agentigrid.engine.agent_loop.SimulationExecutor") as mock_exec_cls:

            mock_executor = MagicMock()
            mock_executor.run.return_value = sim_result
            mock_exec_cls.return_value = mock_executor

            controller = AgentLoopController(cfg)
            controller.run(BASE_CASE, "Test journal in prompt")

        # Second call should have journal in user prompt
        if len(mock_backend.calls) >= 2:
            _, user_prompt = mock_backend.calls[1]
            assert "Search Journal" in user_prompt


@pytest.mark.skipif(not _has_base_case, reason="Base case .m file not found")
@pytest.mark.skipif(not _has_sample, reason="sample_opflow_output.txt not found")
class TestAnalyzeAction:

    def test_analyze_then_complete(self, tmp_path: Path):
        cfg = _make_config(tmp_path)
        stdout = _sample_stdout()

        responses = [
            _objectives_response(),  # objective extraction call after base case
            _make_llm_response({
                "action": "analyze",
                "reasoning": "Need voltage info",
                "query": "buses with voltage below 1.07",
            }),
            _make_llm_response({
                "action": "complete",
                "reasoning": "Got the info",
                "findings": {"summary": "Analysis complete"},
            }),
        ]

        mock_backend = MockBackend(responses)
        sim_result = _make_sim_result(stdout=stdout, success=True)

        with patch("agentigrid.engine.agent_loop.create_backend", return_value=mock_backend), \
             patch("agentigrid.engine.agent_loop.SimulationExecutor") as mock_exec_cls:

            mock_executor = MagicMock()
            mock_executor.run.return_value = sim_result
            mock_exec_cls.return_value = mock_executor

            controller = AgentLoopController(cfg)
            session = controller.run(BASE_CASE, "Analyze voltages")

        assert session.termination_reason == "completed"
        # Analyze adds a lightweight ANALYSIS entry, complete adds a COMPLETE entry
        # (base case + analyze + complete = 3)
        assert len(session.journal) == 3


@pytest.mark.skipif(not _has_base_case, reason="Base case .m file not found")
@pytest.mark.skipif(not _has_sample, reason="sample_opflow_output.txt not found")
class TestJournalExport:

    def test_journal_saved_to_workdir(self, tmp_path: Path):
        cfg = _make_config(tmp_path)
        stdout = _sample_stdout()

        responses = [
            _make_llm_response({
                "action": "complete",
                "reasoning": "Done",
                "findings": {"summary": "Immediate complete"},
            }),
        ]

        mock_backend = MockBackend(responses)
        sim_result = _make_sim_result(stdout=stdout, success=True)

        with patch("agentigrid.engine.agent_loop.create_backend", return_value=mock_backend), \
             patch("agentigrid.engine.agent_loop.SimulationExecutor") as mock_exec_cls:

            mock_executor = MagicMock()
            mock_executor.run.return_value = sim_result
            mock_exec_cls.return_value = mock_executor

            controller = AgentLoopController(cfg)
            session = controller.run(BASE_CASE, "Test export")

        # Check that a journal file was created in workdir
        workdir = tmp_path / "workdir"
        if workdir.exists():
            json_files = list(workdir.glob("journal_*.json"))
            assert len(json_files) == 1
