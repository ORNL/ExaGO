"""exago.save_solutions: which solved cases (opflowout.m / pflowout.m) are kept."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentigrid.config import load_config
from agentigrid.exago_engine.agent_loop import AgentLoopController
from agentigrid.exago_engine.executor import SOLUTION_FILES, SimulationExecutor
from agentigrid.exago_parsers.matpower_model import MATNetwork

# Stands in for opflow/pflow: writes the solved case only when asked to.
FAKE_APP = """#!/bin/sh
for a in "$@"; do
  if [ "$a" = "-save_output" ]; then echo "mpc.converged = 1;" > {out}; fi
done
echo "fake solve"
"""


def _executor(tmp_path: Path, mode: str) -> SimulationExecutor:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for app, out in (("opflow", "opflowout.m"), ("pflow", "pflowout.m"), ("tcopflow", "tcopflowout")):
        exe = bin_dir / app
        exe.write_text(FAKE_APP.format(out=out))
        exe.chmod(0o755)
    cfg = load_config(None, cli_overrides={
        "exago.binary_dir": str(bin_dir), "exago.save_solutions": mode,
        "output.workdir": str(tmp_path / "work"),
    })
    return SimulationExecutor(cfg.exago, cfg.output)


def _net() -> MATNetwork:
    return MATNetwork(casename="case", version="2", baseMVA=100.0, buses=[], generators=[],
                      branches=[], gencost=[], header_comments="")


def _solve_batch(ex: SimulationExecutor, n: int, first_tag: int = 1) -> list[Path]:
    # Distinct tags: run folders are named iter_<tag>_<time to the second>.
    ex.begin_batch()
    for i in range(n):
        ex.run(_net(), "opflow", iteration=-(first_tag + i))
    return ex.end_batch()


def _saved(dirs: list[Path]) -> list[Path]:
    return [d for d in dirs if any((d / f).exists() for f in SOLUTION_FILES)]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_default_is_best():
    assert load_config(None).exago.save_solutions == "best"


def test_set_on_command_line():
    cfg = load_config(None, cli_overrides={"exago.save_solutions": "newest"})
    assert cfg.exago.save_solutions == "newest"


def test_unknown_value_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="agentigrid.config"):
        load_config(None, cli_overrides={"exago.save_solutions": "some"})
    assert "save_solutions" in caplog.text


@pytest.mark.parametrize("name", ["default_config.yaml.template", "local_config.yaml.template"])
def test_templates_set_best(name):
    path = Path(__file__).resolve().parents[2] / "configs" / name
    assert load_config(path).exago.save_solutions == "best"


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode,written", [("all", True), ("best", True), ("newest", True),
                                          ("none", False)])
@pytest.mark.parametrize("app,out", [("opflow", "opflowout.m"), ("pflow", "pflowout.m")])
def test_written_unless_none(tmp_path, mode, written, app, out):
    res = _executor(tmp_path, mode).run(_net(), app, iteration=0)
    assert (res.workdir / out).exists() == written
    assert ("-save_output" in res.argv) == written


def test_none_leaves_other_applications_alone(tmp_path):
    # TCOPFLOW needs -save_output for its own per-period results.
    res = _executor(tmp_path, "none").run(_net(), "tcopflow", iteration=0)
    assert "-save_output" in res.argv


# ---------------------------------------------------------------------------
# Pruning one batch
# ---------------------------------------------------------------------------

def test_batch_records_runs_in_order(tmp_path):
    ex = _executor(tmp_path, "best")
    dirs = _solve_batch(ex, 3)
    assert len(dirs) == 3 and len(set(dirs)) == 3
    ex.run(_net(), "opflow", iteration=9)            # outside a batch: not recorded
    assert ex.end_batch() == []


def test_best_keeps_best(tmp_path):
    ex = _executor(tmp_path, "best")
    dirs = _solve_batch(ex, 3)
    assert ex.prune_solutions(dirs, best=dirs[0]) == dirs[0]
    assert _saved(dirs) == [dirs[0]]
    assert all((d / "case.m").exists() for d in dirs)    # input cases stay


def test_best_without_best_keeps_most_recent(tmp_path):
    ex = _executor(tmp_path, "best")
    dirs = _solve_batch(ex, 3)
    assert ex.prune_solutions(dirs, best=None) == dirs[-1]
    assert _saved(dirs) == [dirs[-1]]


def test_newest_ignores_best(tmp_path):
    ex = _executor(tmp_path, "newest")
    dirs = _solve_batch(ex, 3)
    ex.prune_solutions(dirs, best=dirs[0])
    assert _saved(dirs) == [dirs[-1]]


def test_all_keeps_everything(tmp_path):
    ex = _executor(tmp_path, "all")
    dirs = _solve_batch(ex, 3)
    assert ex.prune_solutions(dirs, best=dirs[0]) is None
    assert _saved(dirs) == dirs


# ---------------------------------------------------------------------------
# Agent loop: which run is "best", and explore -> select
# ---------------------------------------------------------------------------

def _loop(ex):
    return SimpleNamespace(_executor=ex, _pending_explore=None, _batch_best=None)


def test_single_solve_action_is_kept(tmp_path):
    ex = _executor(tmp_path, "newest")
    loop = _loop(ex)
    dirs = _solve_batch(ex, 1)
    AgentLoopController._prune_saved_solutions(loop, "modify", dirs)
    assert _saved(dirs) == dirs


def test_sweep_keeps_marked_best(tmp_path):
    ex = _executor(tmp_path, "best")
    loop = _loop(ex)
    dirs = _solve_batch(ex, 4)
    loop._batch_best = dirs[1]
    AgentLoopController._prune_saved_solutions(loop, "sweep", dirs)
    assert _saved(dirs) == [dirs[1]]


def test_explore_waits_and_select_keeps_choice(tmp_path):
    ex = _executor(tmp_path, "best")
    loop = _loop(ex)
    variants = _solve_batch(ex, 3)
    loop._batch_best = variants[0]                           # Pareto-best variant
    AgentLoopController._prune_saved_solutions(loop, "explore", variants)
    assert _saved(variants) == variants                      # nothing pruned yet
    loop._batch_best = variants[2]                           # the LLM selects another one
    AgentLoopController._prune_saved_solutions(loop, "select", [])
    assert _saved(variants) == [variants[2]]
    assert loop._pending_explore is None


def test_explore_followed_by_other_action_keeps_pareto_best(tmp_path):
    ex = _executor(tmp_path, "best")
    loop = _loop(ex)
    variants = _solve_batch(ex, 3)
    loop._batch_best = variants[1]
    AgentLoopController._prune_saved_solutions(loop, "explore", variants)
    loop._batch_best = None
    later = _solve_batch(ex, 1, first_tag=100)
    AgentLoopController._prune_saved_solutions(loop, "modify", later)
    assert _saved(variants) == [variants[1]]
    assert _saved(later) == later
