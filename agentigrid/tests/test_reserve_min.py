"""Tests for the greedy minimum-hot-reserve de-commitment search (C.8-revised, Path A).

Covers the pure ``reserve.minimize_hot_reserve``:
- accept only while BOTH feasibility checks (base solve + N-1 screen) pass
- de-commit order is descending Pmax (tie-break ascending bus, gen_id)
- revert on failure and skip-and-continue (a blocked unit does not stop the pass)
- ``min_reserve <= reserve_full``; ``lower_bound_pg`` == largest remaining on-unit Pg
- budget guard: ``hit_budget`` set and the search stops at ``max_solves``
- determinism: identical inputs → identical ``ReserveMinResult``
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agentigrid.engine import reserve as R


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

# (bus, gen_id, Pmax) for a 3-unit system; one gen per bus so gen_id == 0.
_UNITS = [(1, 0, 500.0), (2, 0, 300.0), (3, 0, 200.0)]


def _net():
    """A MATNetwork-like object exposing ``.generators`` with bus/status/Pmax."""
    gens = [SimpleNamespace(bus=b, status=1, Pmax=pmax) for (b, _g, pmax) in _UNITS]
    return SimpleNamespace(generators=gens)


def _opflow_for(off: frozenset):
    """A solved-dispatch OPFLOWResult-like object for a given de-commitment set.

    On-units carry Pg=100; off-units carry status=0 (excluded from reserve).
    """
    off = set(off)
    gens = []
    for (bus, gid, pmax) in _UNITS:
        status = 0 if (bus, gid) in off else 1
        pg = 0.0 if status == 0 else 100.0
        gens.append(SimpleNamespace(bus=bus, status=status, Pg=pg, Pmax=pmax))
    return SimpleNamespace(generators=gens)


def _base_solve_always():
    def fn(off):
        return _opflow_for(off)
    return fn


# ---------------------------------------------------------------------------
# Greedy correctness: revert on failure + skip-and-continue
# ---------------------------------------------------------------------------

def _n1_screen_selective(off):
    """N-1 infeasible if unit A(1,0) is off, or if BOTH B(2,0) and C(3,0) are off."""
    off = set(off)
    if (1, 0) in off:
        feasible = False
    elif (2, 0) in off and (3, 0) in off:
        feasible = False
    else:
        feasible = True
    total = len(_UNITS) - len(off)
    passed = total if feasible else max(0, total - 1)
    return feasible, passed, total


def test_greedy_reverts_and_skips():
    res = R.minimize_hot_reserve(
        _net(), 0.9, 1.1,
        base_solve_fn=_base_solve_always(),
        n1_screen_fn=_n1_screen_selective,
        max_solves=100_000,
    )
    # A blocked (revert), B accepted, C blocked (B+C both off is infeasible).
    assert res.decommitted == [(2, 0)]
    assert res.final_on_count == 2
    assert res.reserve_full == pytest.approx(700.0)   # (500+300+200) - 3*100
    assert res.min_reserve == pytest.approx(500.0)    # A,C on: (500-100)+(200-100)
    assert res.min_reserve <= res.reserve_full
    assert res.lower_bound_pg == pytest.approx(100.0)  # largest remaining on-unit Pg
    assert res.n1_secure is True


# ---------------------------------------------------------------------------
# De-commit order is descending Pmax
# ---------------------------------------------------------------------------

def test_decommit_order_descending_pmax():
    # Everything feasible → every unit de-committed, in descending-Pmax order.
    res = R.minimize_hot_reserve(
        _net(), 0.9, 1.1,
        base_solve_fn=_base_solve_always(),
        n1_screen_fn=lambda off: (True, len(_UNITS) - len(set(off)), len(_UNITS) - len(set(off))),
        max_solves=100_000,
    )
    assert res.decommitted == [(1, 0), (2, 0), (3, 0)]  # 500, 300, 200
    assert res.final_on_count == 0
    assert res.min_reserve == pytest.approx(0.0)
    assert res.lower_bound_bus == -1  # no committed units remain


def test_tie_break_bus_then_genid():
    # Two units with equal Pmax at different buses → ordered by ascending bus.
    units = [SimpleNamespace(bus=5, status=1, Pmax=300.0),
             SimpleNamespace(bus=2, status=1, Pmax=300.0)]
    net = SimpleNamespace(generators=units)

    def base_solve(off):
        off = set(off)
        gens = []
        for u in units:
            gid = 0
            status = 0 if (u.bus, gid) in off else 1
            gens.append(SimpleNamespace(bus=u.bus, status=status, Pg=50.0 if status else 0.0, Pmax=300.0))
        return SimpleNamespace(generators=gens)

    res = R.minimize_hot_reserve(
        net, 0.9, 1.1,
        base_solve_fn=base_solve,
        n1_screen_fn=lambda off: (True, 0, len(units) - len(set(off))),
        max_solves=100_000,
    )
    assert res.decommitted == [(2, 0), (5, 0)]  # equal Pmax → ascending bus


# ---------------------------------------------------------------------------
# Budget guard
# ---------------------------------------------------------------------------

def test_budget_guard_stops_early():
    # Tiny budget: the full-commitment startup screen alone exhausts it.
    res = R.minimize_hot_reserve(
        _net(), 0.9, 1.1,
        base_solve_fn=_base_solve_always(),
        n1_screen_fn=lambda off: (True, len(_UNITS) - len(set(off)), len(_UNITS) - len(set(off))),
        max_solves=6,
    )
    assert res.hit_budget is True
    assert len(res.decommitted) < len(_UNITS)  # did not finish the pass


# ---------------------------------------------------------------------------
# Base infeasible at full commitment
# ---------------------------------------------------------------------------

def test_base_infeasible_at_full_commitment():
    res = R.minimize_hot_reserve(
        _net(), 0.9, 1.1,
        base_solve_fn=lambda off: None,   # never converges
        n1_screen_fn=lambda off: (True, 0, 0),
        max_solves=100_000,
    )
    assert res.decommitted == []
    assert res.min_reserve == pytest.approx(0.0)
    assert res.n1_secure is False
    assert res.hit_budget is False


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_determinism():
    kwargs = dict(
        base_solve_fn=_base_solve_always(),
        n1_screen_fn=_n1_screen_selective,
        max_solves=100_000,
    )
    a = R.minimize_hot_reserve(_net(), 0.9, 1.1, **kwargs)
    b = R.minimize_hot_reserve(_net(), 0.9, 1.1, **kwargs)
    assert a == b


# ---------------------------------------------------------------------------
# Handler wiring: _handle_reserve_screen(minimize=True) maps the search result
# into reserve_meta, sets the completed-minimization description, and journals it.
# ---------------------------------------------------------------------------

from agentigrid.config import (  # noqa: E402
    AppConfig, ExagoConfig, DataConfig, LLMConfig, SearchConfig, OutputConfig,
)
from agentigrid.engine.agent_loop import AgentLoopController  # noqa: E402
from agentigrid.engine.executor import SimulationResult  # noqa: E402
from agentigrid.engine import contingency as C  # noqa: E402
from agentigrid.parsers.matpower_parser import parse_matpower  # noqa: E402
from agentigrid.parsers.opflow_results import GenResult, OPFLOWResult  # noqa: E402

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_IEEE118 = _DATA_DIR / "ieee_118_bus_v10.m"


def _cfg(tmp_path):
    return AppConfig(
        exago=ExagoConfig(
            binary_dir=tmp_path / "bin", opflow_binary=None, scopflow_binary=None,
            tcopflow_binary=None, sopflow_binary=None, dcopflow_binary=None,
            pflow_binary=None, env_script=None, timeout=30,
        ),
        data=DataConfig(data_dir=tmp_path / "data"),
        llm=LLMConfig(
            backend="openai", model="test-model", api_key_env="TEST_KEY",
            openai_base_url=None, ollama_host="http://localhost:11434",
            ollama_cloud_host=None, temperature=0.3, max_tokens=4096,
        ),
        search=SearchConfig(
            max_iterations=5, default_mode="accumulative",
            base_case=_IEEE118, gic_file=None, application="opflow",
        ),
        output=OutputConfig(
            workdir=tmp_path / "wd", logs_dir=tmp_path / "logs", save_journal=False,
            journal_format="json", save_modified_files=False, verbose=False,
        ),
    )


def _sim():
    return SimulationResult(
        success=True, exit_code=0, stdout="ok", stderr="", elapsed_seconds=0.1,
        input_file=Path("/tmp/x.m"), application="opflow", error_message=None,
        workdir=Path("/tmp"),
    )


def _gr(bus, pg, pmax, status=1):
    return GenResult(bus=bus, status=status, fuel="COAL", Pg=pg, Qg=0.0,
                     Pmin=0.0, Pmax=pmax, Qmin=-100.0, Qmax=100.0)


def _opf(feasible=True, generators=None):
    return OPFLOWResult(
        converged=feasible, objective_value=1000.0,
        convergence_status="CONVERGED" if feasible else "DID NOT CONVERGE",
        solver="IPOPT", model="POWER_BALANCE", objective_type="MIN_GEN_COST",
        num_iterations=10, solve_time=0.1, branches=[], buses=[],
        generators=generators or [], voltage_min=0.96, voltage_max=1.04,
        max_line_loading_pct=80.0, num_violations=0 if feasible else 1,
        feasibility_detail="feasible" if feasible else "infeasible",
    )


@pytest.mark.skipif(not _IEEE118.exists(), reason="ieee_118_bus_v10.m not available")
def test_handler_minimize_maps_result_and_journals(tmp_path):
    cfg = _cfg(tmp_path)
    backend_mock = MagicMock()
    with patch("agentigrid.engine.agent_loop.create_backend", return_value=backend_mock), \
         patch("agentigrid.engine.agent_loop.SimulationExecutor") as mock_exec_cls:
        mock_executor = MagicMock()
        mock_executor.run.return_value = _sim()
        mock_executor.run_parallel.side_effect = (
            lambda tasks, max_workers=4, thread_limit=None, on_progress=None:
            {i: _sim() for i in range(len(tasks))}
        )
        mock_exec_cls.return_value = mock_executor
        controller = AgentLoopController(cfg)
    net = parse_matpower(_IEEE118)
    controller._base_network = net
    controller._current_network = net

    base_gens = [_gr(10, 100.0, 300.0), _gr(20, 250.0, 400.0), _gr(30, 50.0, 200.0)]
    fake_ctgs = [
        C.Contingency(elements=(C.OutageElement(
            kind="gen", neighbor_bus=b, hop=0, bus=b, gen_id=0),))
        for b in (10, 20, 30)
    ]

    # Reading-2 assessment: base feasible, all three unit outages feasible.
    def fake_parse(sim, application="opflow", bus_limits=None):
        return _opf(True, generators=base_gens)

    canned = R.ReserveMinResult(
        reserve_full=700.0, min_reserve=250.0, decommitted=[(20, 0)],
        final_on_count=2, lower_bound_pg=100.0, lower_bound_bus=10,
        n1_secure=True, solves_used=42, hit_budget=False,
    )

    with patch("agentigrid.engine.agent_loop.contingency.all_generator_contingencies",
               return_value=fake_ctgs), \
         patch("agentigrid.engine.agent_loop.parse_simulation_result_for_app",
               side_effect=fake_parse), \
         patch.object(AgentLoopController, "_run_reserve_minimization",
                      return_value=canned) as mock_min:
        kind, ok = controller._handle_reserve_screen(1, {
            "mode": "reserve", "minimize": True,
            "feasibility": {"Vmin": 0.9, "Vmax": 1.1},
            "description": "Minimize N-1 hot reserve",
        })

    assert (kind, ok) == ("sweep", True)
    assert mock_min.called
    entry = controller._journal.entries[-1]
    assert entry.mode == "reserve"
    assert entry.convergence_status == "CONTINGENCY"
    assert entry.description.startswith("[reserve N-1 minimize]")
    rm = entry.reserve_meta
    assert rm["minimize"] is True
    assert rm["reserve_full"] == pytest.approx(700.0)
    assert rm["min_reserve"] == pytest.approx(250.0)
    assert rm["n_decommitted"] == 1
    assert rm["decommitted"] == [(20, 0)]
    assert rm["final_on_count"] == 2
    assert rm["lower_bound_pg"] == pytest.approx(100.0)
    assert rm["lower_bound_bus"] == 10
    assert rm["n1_secure_min"] is True
    assert rm["hit_budget"] is False
    # Full-commitment assessment fields are preserved alongside the minimization.
    assert rm["hot_reserve_available"] == pytest.approx(500.0)  # 200+150+150
    assert "minimum feasible hot reserve" in (controller._latest_results_text or "").lower()
