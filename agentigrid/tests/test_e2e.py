"""End-to-end integration tests for AgentiGrid."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from agentigrid.backends.base import LLMBackend, LLMResponse
from agentigrid.config import (
    AppConfig, ExagoConfig, DataConfig, LLMConfig, SearchConfig, OutputConfig,
)
from agentigrid.engine.agent_loop import AgentLoopController, SearchSession

# ---------------------------------------------------------------------------
# Paths and availability checks
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE = PROJECT_ROOT / "tests" / "sample_opflow_output.txt"
DATA_DIR = PROJECT_ROOT / "data"
BASE_CASE = DATA_DIR / "case_ACTIVSg200.m"
APPS_DIR = PROJECT_ROOT / "applications"
OPFLOW_BIN = APPS_DIR / "opflow"
ENV_SCRIPT = PROJECT_ROOT / "configs" / "env_setup.sh"

_has_base_case = BASE_CASE.exists()
_has_opflow = OPFLOW_BIN.exists()
_has_sample = SAMPLE.exists()
_has_env_script = ENV_SCRIPT.exists()
_has_anthropic_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
_has_openai_key = bool(os.environ.get("OPENAI_API_KEY"))
_has_any_api_key = _has_anthropic_key or _has_openai_key


# ---------------------------------------------------------------------------
# Mock backend
# ---------------------------------------------------------------------------

class MockBackend(LLMBackend):
    """Returns responses from a predefined list."""

    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)
        self._idx = 0

    def complete(self, system_prompt: str, user_prompt: str, temperature=None) -> LLMResponse:
        if self._idx < len(self._responses):
            resp = self._responses[self._idx]
        else:
            resp = self._responses[-1]
        self._idx += 1
        return resp

    def name(self) -> str:
        return "mock"

    def supports_json_mode(self) -> bool:
        return False


def _mock_response(json_data: Optional[dict]) -> LLMResponse:
    return LLMResponse(
        raw_text=str(json_data) if json_data else "",
        json_data=json_data,
        json_error=None if json_data else "parse error",
        model="mock-model",
        backend="mock",
        prompt_tokens=100,
        completion_tokens=50,
    )


def _objectives_response() -> LLMResponse:
    """Return a valid objective extraction response (for test setup)."""
    import json as _json
    payload = _json.dumps({"objectives": [
        {"name": "generation_cost", "direction": "minimize", "priority": "primary"},
    ]})
    return LLMResponse(
        raw_text=payload,
        json_data=None,
        json_error=None,
        model="mock-model",
        backend="mock",
        prompt_tokens=50,
        completion_tokens=20,
    )


def _make_config(
    tmp_path: Path,
    max_iterations: int = 3,
    env_script: Optional[Path] = None,
) -> AppConfig:
    return AppConfig(
        exago=ExagoConfig(
            binary_dir=APPS_DIR,
            opflow_binary=OPFLOW_BIN if _has_opflow else None,
            scopflow_binary=None,
            tcopflow_binary=None,
            sopflow_binary=None,
            dcopflow_binary=None,
            pflow_binary=None,
            env_script=env_script,
            timeout=60,
        ),
        data=DataConfig(data_dir=DATA_DIR),
        llm=LLMConfig(
            backend="anthropic" if _has_anthropic_key else "openai",
            model="claude-sonnet-4-20250514" if _has_anthropic_key else "gpt-4o-mini",
            api_key_env="ANTHROPIC_API_KEY" if _has_anthropic_key else "OPENAI_API_KEY",
            openai_base_url=None,
            ollama_host="http://localhost:11434",
            ollama_cloud_host=None,
            temperature=0.3,
            max_tokens=4096,
        ),
        search=SearchConfig(
            max_iterations=max_iterations,
            default_mode="fresh",
            base_case=BASE_CASE,
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


# ===========================================================================
# Variant B — Mock LLM with real opflow
# ===========================================================================

@pytest.mark.skipif(not _has_opflow, reason="opflow binary not available")
@pytest.mark.skipif(not _has_base_case, reason="base case not available")
@pytest.mark.skipif(not _has_env_script, reason="configs/env_setup.sh not available")
class TestE2EMockLLMRealOpflow:
    """Full pipeline with mock LLM and real opflow binary."""

    def test_base_case_then_complete(self, tmp_path: Path):
        """Run base case, LLM immediately completes."""
        cfg = _make_config(
            tmp_path,
            env_script=ENV_SCRIPT if _has_env_script else None,
        )

        responses = [
            _mock_response({
                "action": "complete",
                "reasoning": "Base case is converged, reporting results.",
                "findings": {
                    "summary": "Base case converges with cost ~$27,557.",
                    "details": "Voltage range 1.062-1.100 pu, 49 generators online.",
                },
            }),
        ]

        mock_backend = MockBackend(responses)

        with patch("agentigrid.engine.agent_loop.create_backend", return_value=mock_backend):
            controller = AgentLoopController(cfg, quiet=True)
            session = controller.run(BASE_CASE, "Report the base case results")

        # Base case ran with real opflow
        assert len(session.journal) >= 1
        base_entry = session.journal.entries[0]
        assert base_entry.iteration == 0
        assert base_entry.feasible is True
        assert base_entry.convergence_status == "CONVERGED"
        assert abs(base_entry.objective_value - 27557.57) < 1.0
        assert session.termination_reason == "completed"

    def test_modify_then_complete(self, tmp_path: Path):
        """Run base case, modify (scale loads +5%), then complete."""
        cfg = _make_config(
            tmp_path,
            env_script=ENV_SCRIPT if _has_env_script else None,
        )

        responses = [
            _objectives_response(),  # objective extraction call after base case
            _mock_response({
                "action": "modify",
                "reasoning": "Testing small load increase.",
                "mode": "fresh",
                "description": "Scale all loads +5%",
                "commands": [{"action": "scale_all_loads", "factor": 1.05}],
            }),
            _mock_response({
                "action": "complete",
                "reasoning": "Load increase converged successfully.",
                "findings": {"summary": "5% load increase is feasible."},
            }),
        ]

        mock_backend = MockBackend(responses)

        with patch("agentigrid.engine.agent_loop.create_backend", return_value=mock_backend):
            controller = AgentLoopController(cfg, quiet=True)
            session = controller.run(BASE_CASE, "Test 5% load increase")

        assert len(session.journal) == 3  # base + modify + complete
        assert session.journal.entries[0].feasible is True
        assert session.journal.entries[1].feasible is True
        # Modified case should have higher cost than base
        assert session.journal.entries[1].objective_value > session.journal.entries[0].objective_value
        assert session.termination_reason == "completed"

        # Journal should be saved
        json_files = list((tmp_path / "workdir").glob("journal_*.json"))
        assert len(json_files) == 1


# ===========================================================================
# Variant A — Real LLM integration test (requires API key)
# ===========================================================================

@pytest.mark.slow
@pytest.mark.skipif(not _has_opflow, reason="opflow binary not available")
@pytest.mark.skipif(not _has_base_case, reason="base case not available")
@pytest.mark.skipif(not _has_any_api_key, reason="no LLM API key available")
class TestE2ERealLLM:
    """Full end-to-end test with real opflow and real LLM."""

    def test_simple_goal(self, tmp_path: Path):
        """Ask LLM a simple goal with real API."""
        cfg = _make_config(
            tmp_path,
            max_iterations=3,
            env_script=ENV_SCRIPT if _has_env_script else None,
        )

        controller = AgentLoopController(cfg, quiet=True)
        session = controller.run(
            BASE_CASE,
            "Run the base case and report the objective value and voltage range. "
            "Then use the complete action.",
        )

        assert len(session.journal) >= 1
        assert session.journal.entries[0].feasible is True
        # Should complete without errors
        assert session.termination_reason in ("completed", "max_iterations")


# ===========================================================================
# CLI tests
# ===========================================================================

class TestCLIDryRun:
    """Test the CLI with --dry-run (no simulation needed)."""

    def test_dry_run(self):
        result = subprocess.run(
            [
                sys.executable, "-m", "agentigrid",
                "data/case_ACTIVSg200.m", "test goal",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=10,
        )
        assert result.returncode == 0
        assert "Resolved configuration" in result.stdout

    def test_dry_run_concurrent_pflow(self):
        result = subprocess.run(
            [
                sys.executable, "-m", "agentigrid",
                "data/case_ACTIVSg200.m", "test pflow goal",
                "--app", "pflow",
                "--concurrent-pflow",
                "--max-variants", "4",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=10,
        )
        assert result.returncode == 0
        assert "concurrent_pflow: True" in result.stdout
        assert "max_variants: 4" in result.stdout

    def test_version(self):
        result = subprocess.run(
            [sys.executable, "-m", "agentigrid", "--version"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=10,
        )
        assert result.returncode == 0
        assert "0.1.0" in result.stdout
