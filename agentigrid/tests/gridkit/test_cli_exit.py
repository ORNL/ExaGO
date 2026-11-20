"""CLI exit codes for how a session ended: 3 api_error, 4 quota_exhausted, 1 other failures."""

from __future__ import annotations

import argparse
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agentigrid import cli
from agentigrid.config import load_config

CASES = [("api_error", 3), ("quota_exhausted", 4), ("parse_failures", 1),
         ("base_case_failed", 1), ("completed", None), ("max_iterations", None)]


@pytest.mark.parametrize("termination,code", CASES)
def test_gridkit_exit_code(tmp_path, termination, code):
    case = tmp_path / "case.json"
    case.write_text("{}")
    cfg = load_config(None, cli_overrides={"output.workdir": str(tmp_path / "w"),
                                            "output.logs_dir": str(tmp_path / "l")})
    args = argparse.Namespace(resume=None, base_case=str(case), goal="g", from_exago=False,
                              dry_run=False, quiet=True)

    class FakeLoop:
        def __init__(self, *a, **k):
            pass

        def run(self, *a, **k):
            return SimpleNamespace(termination_reason=termination)

    with patch("agentigrid.gridkit_engine.agent_loop.GridkitAgentLoop", FakeLoop):
        if code is None:
            cli.run_gridkit(cfg, args)
        else:
            with pytest.raises(SystemExit) as e:
                cli.run_gridkit(cfg, args)
            assert e.value.code == code


@pytest.mark.parametrize("termination,code", CASES)
def test_exago_exit_code(termination, code):
    class FakeController:
        def __init__(self, *a, **k):
            pass

        def run(self, *a, **k):
            return SimpleNamespace(termination_reason=termination)

    cfg = load_config(None)
    with patch("agentigrid.exago_engine.agent_loop.AgentLoopController", FakeController), \
         patch.object(cli.sys.stdin, "isatty", return_value=False):
        if code is None:
            cli.run_search(cfg, "g", quiet=True)
        else:
            with pytest.raises(SystemExit) as e:
                cli.run_search(cfg, "g", quiet=True)
            assert e.value.code == code
