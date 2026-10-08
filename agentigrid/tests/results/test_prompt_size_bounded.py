"""Regression guards: LLM-facing summaries must stay O(1) in network size.

At 100k buses the un-capped generator table in ``network_summary`` blew the
system prompt past the model's context limit on iteration 1. These tests pin the
bounded behavior (and the config-defaults trap that would silently null the cap).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentigrid.config import load_config
from agentigrid.parsers.matpower_model import (
    Bus, Branch, Generator, GenCost, MATNetwork,
)
from agentigrid.parsers.network_summary import network_summary
from agentigrid.parsers.opflow_results import (
    OPFLOWResult, BusResult, BranchResult, GenResult,
)
from agentigrid.parsers.results_summary import results_summary
from agentigrid.parsers.pflow_summary import pflow_results_summary
from agentigrid.parsers.scopflow_summary import scopflow_results_summary
from agentigrid.parsers.dcopflow_summary import dcopflow_results_summary
from agentigrid.parsers.tcopflow_summary import tcopflow_results_summary


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _bus(i: int) -> Bus:
    return Bus(
        bus_i=i, type=1, Pd=10.0, Qd=3.0, Gs=0.0, Bs=0.0, area=1,
        Vm=1.0, Va=0.0, baseKV=138.0, zone=1, Vmax=1.1, Vmin=0.9,
    )


def _gen(bus: int, pmax: float, pg: float = 0.0, status: int = 1) -> Generator:
    return Generator(
        bus=bus, Pg=pg, Qg=0.0, Qmax=50.0, Qmin=-50.0, Vg=1.0,
        mBase=100.0, status=status, Pmax=pmax, Pmin=0.0, extra=[],
    )


def _branch(f: int, t: int, ratio: float = 0.0) -> Branch:
    return Branch(
        fbus=f, tbus=t, r=0.01, x=0.1, b=0.0, rateA=100.0, rateB=100.0,
        rateC=100.0, ratio=ratio, angle=0.0, status=1, angmin=-360.0,
        angmax=360.0, extra=[],
    )


def _network(n_gen: int, genfuel: list[str] | None = None) -> MATNetwork:
    buses = [_bus(i + 1) for i in range(max(n_gen, 3))]
    generators = [_gen(bus=i + 1, pmax=10.0 + i, pg=1.0) for i in range(n_gen)]
    branches = [_branch(1, 2), _branch(2, 3, ratio=1.02)]
    extra: dict[str, str] = {}
    if genfuel is not None:
        body = "\n".join(f"\t'{f}';" for f in genfuel)
        extra["genfuel"] = "mpc.genfuel = {\n" + body + "\n};"
    return MATNetwork(
        casename="synthetic", version="2", baseMVA=100.0,
        buses=buses, generators=generators, branches=branches,
        gencost=[GenCost(model=2, startup=0.0, shutdown=0.0, ncost=3,
                         coeffs=[0.0, 20.0, 0.0]) for _ in range(n_gen)],
        header_comments="function mpc = synthetic\n", extra_sections=extra,
    )


def _big_opflow(n_bus: int, n_branch: int, n_gen: int) -> OPFLOWResult:
    buses = [
        BusResult(bus_id=i + 1, Pd=10.0, Pd_loss=0.0, Qd=3.0, Qd_loss=0.0,
                  Vm=1.0, Va=0.0, mult_Pmis=0.0, mult_Qmis=0.0,
                  Pslack=0.0, Qslack=0.0)
        for i in range(n_bus)
    ]
    branches = [
        BranchResult(from_bus=(i % n_bus) + 1, to_bus=((i + 1) % n_bus) + 1,
                     status=1, Sf=50.0, St=48.0, Slim=100.0,
                     mult_Sf=0.0, mult_St=0.0)
        for i in range(n_branch)
    ]
    gens = [
        GenResult(bus=(i % n_bus) + 1, status=1, fuel="COAL", Pg=5.0, Qg=1.0,
                  Pmin=0.0, Pmax=50.0, Qmin=-20.0, Qmax=20.0)
        for i in range(n_gen)
    ]
    return OPFLOWResult(
        converged=True, objective_value=12345.0, convergence_status="CONVERGED",
        solver="IPOPT", model="POWER_BALANCE", objective_type="MIN_GEN_COST",
        num_iterations=10, solve_time=1.0, buses=buses, branches=branches,
        generators=gens, total_gen_mw=25000.0, total_load_mw=24000.0,
        total_gen_mvar=1000.0, total_load_mvar=900.0, voltage_min=0.98,
        voltage_max=1.02, voltage_mean=1.0, max_line_loading_pct=50.0,
        num_violations=0, losses_mw=1000.0,
    )


# ---------------------------------------------------------------------------
# Part 1: config knob default (guards the config-defaults trap)
# ---------------------------------------------------------------------------

def test_report_config_default(tmp_path):
    # A YAML with an explicitly EMPTY report section must still yield the
    # dataclass default (40), not None — the field is in BOTH DEFAULTS and the
    # dataclass, so _build_section's raw.get finds 40.
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text("report: {}\n", encoding="utf-8")
    cfg = load_config(cfg_file)
    assert cfg.report.network_summary_max_generators == 40

    # And with no file at all (pure defaults).
    assert load_config(None).report.network_summary_max_generators == 40


def test_report_config_cli_override():
    cfg = load_config(None, cli_overrides={"report.network_summary_max_generators": 7})
    assert cfg.report.network_summary_max_generators == 7


def test_report_config_cli_set_flag():
    # The `--set report.network_summary_max_generators=<n>` CLI path (with int
    # coercion) reaches the config as an int, not a string.
    from agentigrid.cli import build_parser, _cli_overrides

    args = build_parser().parse_args(
        ["case.m", "some goal",
         "--set", "report.network_summary_max_generators=12"]
    )
    overrides = _cli_overrides(args)
    assert overrides["report.network_summary_max_generators"] == 12
    cfg = load_config(None, cli_overrides=overrides)
    assert cfg.report.network_summary_max_generators == 12


# ---------------------------------------------------------------------------
# Part 2: network_summary bounded at scale
# ---------------------------------------------------------------------------

def test_network_summary_bounded_large():
    fuels = [("WIND" if i % 3 == 0 else "COAL") for i in range(5000)]
    net = _network(5000, genfuel=fuels)
    # Give one generator a distinctive, unambiguous largest Pmax.
    net.generators[1234] = _gen(bus=88888, pmax=1_000_000.0, pg=10.0)

    summary = network_summary(net, max_generators=40)

    assert len(summary.splitlines()) < 120
    assert len(summary) < 20_000
    # Largest-Pmax generator (top of the ranked list) is present.
    assert "88888" in summary
    # Truncation marker + fuel-mix histogram appear only when truncating.
    assert "more generators omitted" in summary
    assert "Fuel mix" in summary
    # Aggregate lines are always kept.
    assert "Total Pg:" in summary
    assert "Online capacity:" in summary


def test_network_summary_small_identical():
    net = _network(10)  # <= default cap
    summary = network_summary(net, max_generators=40)

    # No truncation artifacts on a small network.
    assert "Fuel mix" not in summary
    assert "omitted" not in summary

    # Generators listed in ORIGINAL file order (bus 1, 2, 3, ...).
    gen_rows = [
        ln for ln in summary.splitlines()
        if ln.startswith("  ") and "ON" in ln and "." in ln
    ]
    listed_buses = [int(ln.split()[0]) for ln in gen_rows]
    assert listed_buses == [g.bus for g in net.generators]

    # Deterministic.
    assert network_summary(net, max_generators=40) == summary


def test_network_summary_small_matches_uncapped_default():
    # At or below the cap, changing the cap must not change the output
    # (byte-identical small-network behavior).
    net = _network(10)
    assert network_summary(net, max_generators=40) == network_summary(
        net, max_generators=10_000
    )


# ---------------------------------------------------------------------------
# Part 5: all result summaries bounded
# ---------------------------------------------------------------------------

def test_all_result_summaries_bounded():
    result = _big_opflow(n_bus=5000, n_branch=8000, n_gen=5000)
    assert len(results_summary(result)) < 8_000
    assert len(pflow_results_summary(result)) < 8_000
    assert len(scopflow_results_summary(result)) < 8_000
    assert len(dcopflow_results_summary(result)) < 8_000


# ---------------------------------------------------------------------------
# Part 4: tcopflow per-period table bounded on long horizons
# ---------------------------------------------------------------------------

def test_tcopflow_period_table_bounded():
    result = _big_opflow(n_bus=50, n_branch=60, n_gen=20)
    period_data = [
        {"period": i, "total_load_mw": 100.0 + i, "total_gen_mw": 105.0 + i,
         "voltage_min": 0.98, "voltage_max": 1.02, "max_line_loading_pct": 40.0,
         "losses_mw": 5.0}
        for i in range(500)
    ]
    summary = tcopflow_results_summary(
        result, num_steps=500, duration_min=1500.0, dT_min=3.0,
        period_data=period_data,
    )
    assert "more periods omitted" in summary
    # Aggregates still span all periods (kept, not truncated).
    assert "Aggregated metrics" in summary
    # Bounded overall.
    assert len(summary.splitlines()) < 160


def test_tcopflow_short_horizon_no_truncation():
    result = _big_opflow(n_bus=50, n_branch=60, n_gen=20)
    period_data = [
        {"period": i, "total_load_mw": 100.0 + i, "total_gen_mw": 105.0 + i,
         "voltage_min": 0.98, "voltage_max": 1.02, "max_line_loading_pct": 40.0,
         "losses_mw": 5.0}
        for i in range(4)
    ]
    summary = tcopflow_results_summary(
        result, num_steps=4, period_data=period_data,
    )
    assert "more periods omitted" not in summary
    assert "Per-period summary" in summary
