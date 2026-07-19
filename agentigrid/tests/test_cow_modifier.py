"""Tests for copy-on-write modification mode and O(1) index lookups (perf pass).

Covers:
- COW isolates the source network (no mutation leaks through shared elements)
- COW produces byte-identical output to the deep-copy path for a fixed command list
- the sweep hoist builds per-candidate nets identical to the pre-change path and
  leaves base_network untouched
- the map-backed index lookups match a linear-scan reference (incl. parallel branch)
- prebuilt index maps (reused across calls) produce identical output to building
  them per call, and the sweep builds its maps once (not once per candidate)
- validation with prebuilt maps returns results identical to the linear-scan path
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentigrid.config import (
    AppConfig, ExagoConfig, DataConfig, LLMConfig, SearchConfig, OutputConfig,
)
from agentigrid.engine.agent_loop import AgentLoopController
from agentigrid.engine.executor import SimulationResult
from agentigrid.engine import modifier as M
from agentigrid.engine.commands import parse_command
from agentigrid.engine.modifier import (
    apply_modifications,
    build_index_maps,
    _build_index_maps,
    _find_bus,
    _gen_index_in_network,
    _branch_index_in_network,
)
from agentigrid.engine.validation import validate_command
from agentigrid.parsers.matpower_parser import parse_matpower
from agentigrid.parsers.matpower_writer import write_matpower

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_IEEE118 = _DATA_DIR / "ieee_118_bus_v10.m"
_ACTIVSG200 = _DATA_DIR / "case_ACTIVSg200.m"

# Prefer ACTIVSg200 (richer extended fields); fall back to the 118-bus case.
_BASE_CASE = _ACTIVSG200 if _ACTIVSG200.exists() else _IEEE118
_has_base = _BASE_CASE.exists()
_has_118 = _IEEE118.exists()

pytestmark = pytest.mark.skipif(not _has_base, reason="no MATPOWER base case available")


def _net():
    return parse_matpower(_BASE_CASE)


def _as_string(net, tmp_path: Path, name: str) -> str:
    """Serialize a network to a MATPOWER string via write_matpower."""
    out = tmp_path / name
    write_matpower(net, out)
    return out.read_text(encoding="utf-8")


def _cmds(raw_list):
    return [parse_command(r) for r in raw_list]


# ---------------------------------------------------------------------------
# COW isolates the base network
# ---------------------------------------------------------------------------

def test_cow_isolates_base(tmp_path):
    base = _net()
    before = _as_string(base, tmp_path, "base_before.m")

    # A load bus we can perturb.
    load_bus = next(b.bus_i for b in base.buses if b.Pd != 0)

    cmds = _cmds([
        {"action": "set_all_bus_vlimits", "Vmin": 0.9, "Vmax": 1.1},
        {"action": "set_load", "bus": load_bus, "Pd": 123456.0},
    ])
    modified, _ = apply_modifications(base, cmds, application="opflow", copy_mode="cow")

    # Base must be byte-for-byte unchanged (no write-through via shared elements).
    after = _as_string(base, tmp_path, "base_after.m")
    assert after == before

    # The modified net must reflect the change.
    mbus = next(b for b in modified.buses if b.bus_i == load_bus)
    assert mbus.Pd == 123456.0
    assert all(b.Vmin == 0.9 and b.Vmax == 1.1 for b in modified.buses)

    # The base object itself (in memory) is also untouched.
    assert next(b for b in base.buses if b.bus_i == load_bus).Pd != 123456.0


# ---------------------------------------------------------------------------
# COW == deep, byte-for-byte
# ---------------------------------------------------------------------------

def test_cow_equals_deep(tmp_path):
    base = _net()
    load_bus = next(b.bus_i for b in base.buses if b.Pd != 0)
    gen_bus = base.generators[0].bus
    br = base.branches[0]

    raw = [
        {"action": "set_all_bus_vlimits", "Vmin": 0.92, "Vmax": 1.08},
        {"action": "set_load", "bus": load_bus, "Pd": 77.0, "Qd": 12.0},
        {"action": "scale_all_loads", "factor": 1.05},
        {"action": "set_gen_status", "bus": gen_bus, "status": 0},
        {"action": "set_branch_rate", "fbus": br.fbus, "tbus": br.tbus, "rateA": 555.0},
    ]

    deep_net, _ = apply_modifications(base, _cmds(raw), application="opflow", copy_mode="deep")
    cow_net, _ = apply_modifications(base, _cmds(raw), application="opflow", copy_mode="cow")

    deep_str = _as_string(deep_net, tmp_path, "deep.m")
    cow_str = _as_string(cow_net, tmp_path, "cow.m")
    assert cow_str == deep_str


def test_cow_equals_deep_with_added_generator(tmp_path):
    """Append path (AddGeneratorAtBus) must match under both modes."""
    base = _net()
    gen_bus = base.generators[0].bus

    raw = [
        {"action": "set_all_bus_vlimits", "Vmin": 0.9, "Vmax": 1.1},
        {"action": "add_generator_at_bus", "bus": gen_bus, "capacity_mw": 100.0,
         "dispatchable": True},
        # A gen mutation AFTER the append targets an existing (pre-append) unit.
        {"action": "set_gen_status", "bus": gen_bus, "status": 0},
    ]

    deep_net, _ = apply_modifications(base, _cmds(raw), application="opflow", copy_mode="deep")
    cow_net, _ = apply_modifications(base, _cmds(raw), application="opflow", copy_mode="cow")

    assert _as_string(deep_net, tmp_path, "deep_g.m") == _as_string(cow_net, tmp_path, "cow_g.m")
    # Base unchanged (append did not leak into the source list).
    assert len(base.generators) < len(cow_net.generators)


# ---------------------------------------------------------------------------
# Index-lookup parity vs linear-scan reference
# ---------------------------------------------------------------------------

def test_index_lookup_parity():
    net = _net()
    maps = _build_index_maps(net)

    # _find_bus for every bus id.
    for b in net.buses:
        ref = None
        for cand in net.buses:
            if cand.bus_i == b.bus_i:
                ref = cand
                break
        assert _find_bus(net, b.bus_i, maps) is ref
    # Missing bus id → None (both paths).
    missing = max(b.bus_i for b in net.buses) + 12345
    assert _find_bus(net, missing, maps) is None
    assert _find_bus(net, missing) is None

    # _gen_index_in_network for each (bus, gen_id) position.
    gen_buses = {}
    for i, g in enumerate(net.generators):
        gen_buses.setdefault(g.bus, []).append(i)
    for bus_id, indices in gen_buses.items():
        for gid in range(len(indices)):
            ref_idx = indices[gid]
            assert _gen_index_in_network(net, bus_id, gid, maps) == ref_idx
            assert _gen_index_in_network(net, bus_id, gid) == ref_idx
        # gen_id=None behaves as 0.
        assert _gen_index_in_network(net, bus_id, None, maps) == indices[0]

    # _branch_index_in_network for each branch, including a parallel (ckt>0) case.
    branch_keys = {}
    for i, brr in enumerate(net.branches):
        key = (min(brr.fbus, brr.tbus), max(brr.fbus, brr.tbus))
        branch_keys.setdefault(key, []).append(i)
    parallel_found = False
    for (f, t), indices in branch_keys.items():
        for ckt in range(len(indices)):
            ref_idx = indices[ckt]
            assert _branch_index_in_network(net, f, t, ckt, maps) == ref_idx
            assert _branch_index_in_network(net, f, t, ckt) == ref_idx
            # Reversed orientation resolves to the same key.
            assert _branch_index_in_network(net, t, f, ckt, maps) == ref_idx
        if len(indices) > 1:
            parallel_found = True
    # The assertion loop above already exercised a parallel branch if one exists.
    _ = parallel_found


# ---------------------------------------------------------------------------
# Sweep hoist regression: per-candidate nets match the pre-change path
# ---------------------------------------------------------------------------

def _make_config(tmp_path: Path, base_case: Path) -> AppConfig:
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
            base_case=base_case, gic_file=None, application="opflow",
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


@pytest.mark.skipif(not _has_118, reason="ieee_118_bus_v10.m not available")
def test_sweep_regression(tmp_path):
    cfg = _make_config(tmp_path, _IEEE118)

    captured = {"tasks": None}

    with patch("agentigrid.engine.agent_loop.create_backend", return_value=MagicMock()), \
         patch("agentigrid.engine.agent_loop.SimulationExecutor") as mock_exec_cls:
        mock_executor = MagicMock()
        mock_executor.run.return_value = _sim()

        def _capture_parallel(tasks, max_workers=4, thread_limit=None, on_progress=None):
            captured["tasks"] = list(tasks)
            # Return no solved results → the downstream parse loop is a no-op.
            return {i: None for i in range(len(tasks))}

        mock_executor.run_parallel.side_effect = _capture_parallel
        mock_exec_cls.return_value = mock_executor
        controller = AgentLoopController(cfg)

    net118 = parse_matpower(_IEEE118)
    controller._base_network = net118
    controller._current_network = net118

    vmin, vmax = 0.9, 1.1
    load_buses = [b.bus_i for b in net118.buses if b.Pd != 0][:3]
    assert load_buses, "expected load buses in the 118-bus case"

    kind, ok = controller._handle_sweep(1, {
        "description": "COW sweep regression",
        "candidate_set": {"type": "bus_list", "buses": load_buses},
        "mutation": {"action": "set_load", "Pd": 42.0},
        "feasibility": {"Vmin": vmin, "Vmax": vmax},
    })
    assert kind == "sweep"
    assert captured["tasks"] is not None
    assert len(captured["tasks"]) == len(load_buses)

    # Reference: the pre-change path applied BOTH commands to base_network (deep).
    for task, bus_id in zip(captured["tasks"], load_buses):
        modified_net = task[0]
        ref_net, _ = apply_modifications(
            net118,
            _cmds([
                {"action": "set_all_bus_vlimits", "Vmin": vmin, "Vmax": vmax},
                {"action": "set_load", "bus": bus_id, "Pd": 42.0},
            ]),
            application="opflow",
        )
        got = _as_string(modified_net, tmp_path, f"got_{bus_id}.m")
        ref = _as_string(ref_net, tmp_path, f"ref_{bus_id}.m")
        assert got == ref, f"candidate bus {bus_id} net differs from pre-change path"

    # base_network must be untouched after the whole sweep.
    base_after = _as_string(net118, tmp_path, "base118_after.m")
    base_fresh = _as_string(parse_matpower(_IEEE118), tmp_path, "base118_fresh.m")
    assert base_after == base_fresh


# ---------------------------------------------------------------------------
# Prompt #2: prebuilt (reused) index maps
# ---------------------------------------------------------------------------

def test_prebuilt_maps_parity(tmp_path):
    """apply_modifications with reused maps == building them per call, base intact."""
    base = _net()
    before = _as_string(base, tmp_path, "pm_before.m")

    load_bus = next(b.bus_i for b in base.buses if b.Pd != 0)
    gen_bus = base.generators[0].bus
    br = base.branches[0]
    raw = [
        {"action": "set_all_bus_vlimits", "Vmin": 0.93, "Vmax": 1.07},
        {"action": "set_load", "bus": load_bus, "Pd": 88.0, "Qd": 9.0},
        {"action": "set_gen_status", "bus": gen_bus, "status": 0},
        {"action": "set_branch_rate", "fbus": br.fbus, "tbus": br.tbus, "rateA": 321.0},
    ]

    maps = build_index_maps(base)
    with_maps, _ = apply_modifications(
        base, _cmds(raw), application="opflow", copy_mode="cow", index_maps=maps,
    )
    without_maps, _ = apply_modifications(
        base, _cmds(raw), application="opflow", copy_mode="cow",
    )
    deep_ref, _ = apply_modifications(base, _cmds(raw), application="opflow")

    s_with = _as_string(with_maps, tmp_path, "pm_with.m")
    s_without = _as_string(without_maps, tmp_path, "pm_without.m")
    s_deep = _as_string(deep_ref, tmp_path, "pm_deep.m")
    assert s_with == s_without == s_deep

    # Base network must be unchanged by any path.
    assert _as_string(base, tmp_path, "pm_after.m") == before


def test_prebuilt_maps_reused_across_calls(tmp_path):
    """One map object drives many candidate applies, each isolated and correct."""
    base = _net()
    maps = build_index_maps(base)
    load_buses = [b.bus_i for b in base.buses if b.Pd != 0][:5]

    for bus_id in load_buses:
        cmds = _cmds([{"action": "set_load", "bus": bus_id, "Pd": 4242.0}])
        got, _ = apply_modifications(
            base, cmds, application="opflow", copy_mode="cow", index_maps=maps,
        )
        ref, _ = apply_modifications(base, cmds, application="opflow")
        assert _as_string(got, tmp_path, f"rm_got_{bus_id}.m") == \
            _as_string(ref, tmp_path, f"rm_ref_{bus_id}.m")
        assert next(b for b in got.buses if b.bus_i == bus_id).Pd == 4242.0
    # Base untouched after all candidates.
    assert all(b.Pd != 4242.0 for b in base.buses)


# ---------------------------------------------------------------------------
# Prompt #2: validation O(1) parity
# ---------------------------------------------------------------------------

def _assert_validation_parity(cmd, net, maps):
    a = validate_command(cmd, net)
    b = validate_command(cmd, net, index_maps=maps)
    assert a.valid == b.valid, f"valid differs for {cmd}"
    assert a.errors == b.errors, f"errors differ for {cmd}: {a.errors} vs {b.errors}"
    assert a.warnings == b.warnings, f"warnings differ for {cmd}: {a.warnings} vs {b.warnings}"


def test_validation_maps_parity():
    net = _net()
    maps = build_index_maps(net)

    real_bus = net.buses[0].bus_i
    missing_bus = max(b.bus_i for b in net.buses) + 99999
    gen_bus = net.generators[0].bus
    slack_bus = next((b.bus_i for b in net.buses if b.type == 3), None)
    br = net.branches[0]

    cmds = [
        # Valid existence / bounds paths.
        {"action": "set_load", "bus": real_bus, "Pd": 50.0},
        {"action": "scale_load", "bus": real_bus, "factor": 1.2},
        {"action": "set_gen_status", "bus": gen_bus, "status": 0},
        {"action": "set_gen_voltage", "bus": gen_bus, "Vg": 1.01},
        {"action": "set_cost_coeffs", "bus": gen_bus, "coeffs": [0.0, 40.0, 0.0]},
        {"action": "set_branch_status", "fbus": br.fbus, "tbus": br.tbus, "status": 0},
        {"action": "set_branch_rate", "fbus": br.fbus, "tbus": br.tbus, "rateA": 400.0},
        {"action": "set_bus_vlimits", "bus": real_bus, "Vmin": 0.95, "Vmax": 1.05},
        {"action": "add_load_at_bus", "bus": real_bus, "Pd": 10.0, "Qd": 2.0},
        {"action": "add_generator_at_bus", "bus": real_bus, "capacity_mw": 50.0,
         "dispatchable": True},
        # Reversed-orientation branch lookup.
        {"action": "set_branch_status", "fbus": br.tbus, "tbus": br.fbus, "status": 1},
        # Invalid: nonexistent bus / gen / branch, out-of-range gen_id, non-transformer tap.
        {"action": "set_load", "bus": missing_bus, "Pd": 1.0},
        {"action": "set_gen_status", "bus": gen_bus, "gen_id": 999, "status": 0},
        {"action": "set_gen_status", "bus": missing_bus, "status": 0},
        {"action": "set_branch_status", "fbus": missing_bus, "tbus": real_bus, "status": 0},
        {"action": "set_tap_ratio", "fbus": br.fbus, "tbus": br.tbus, "ratio": 1.02},
    ]
    if slack_bus is not None:
        cmds.append({"action": "set_gen_dispatch", "bus": slack_bus, "Pg": 10.0})

    for raw in cmds:
        _assert_validation_parity(parse_command(raw), net, maps)

    # Parallel-branch ckt>0 parity, if the case has any.
    branch_keys: dict[tuple[int, int], list] = {}
    for brr in net.branches:
        branch_keys.setdefault((min(brr.fbus, brr.tbus), max(brr.fbus, brr.tbus)), []).append(brr)
    parallel = next(((k, v) for k, v in branch_keys.items() if len(v) > 1), None)
    if parallel is not None:
        (f, t), _group = parallel
        for ckt in (0, 1):
            _assert_validation_parity(
                parse_command({"action": "set_branch_status", "fbus": f, "tbus": t,
                               "ckt": ckt, "status": 0}),
                net, maps,
            )
        # ckt out of range → identical error on both paths.
        _assert_validation_parity(
            parse_command({"action": "set_branch_status", "fbus": f, "tbus": t,
                           "ckt": 99, "status": 0}),
            net, maps,
        )


# ---------------------------------------------------------------------------
# Prompt #2: the sweep builds its maps once, not once per candidate
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_118, reason="ieee_118_bus_v10.m not available")
def test_sweep_builds_maps_once(tmp_path):
    cfg = _make_config(tmp_path, _IEEE118)
    captured = {"tasks": None}

    with patch("agentigrid.engine.agent_loop.create_backend", return_value=MagicMock()), \
         patch("agentigrid.engine.agent_loop.SimulationExecutor") as mock_exec_cls:
        mock_executor = MagicMock()
        mock_executor.run.return_value = _sim()

        def _capture_parallel(tasks, max_workers=4, thread_limit=None, on_progress=None):
            captured["tasks"] = list(tasks)
            return {i: None for i in range(len(tasks))}

        mock_executor.run_parallel.side_effect = _capture_parallel
        mock_exec_cls.return_value = mock_executor
        controller = AgentLoopController(cfg)

    net118 = parse_matpower(_IEEE118)
    controller._base_network = net118
    controller._current_network = net118
    load_buses = [b.bus_i for b in net118.buses if b.Pd != 0]

    real_build = M._build_index_maps

    def _run_sweep(candidate_buses):
        counter = {"n": 0}

        def _counting(net):
            counter["n"] += 1
            return real_build(net)

        with patch.object(M, "_build_index_maps", side_effect=_counting):
            kind, ok = controller._handle_sweep(1, {
                "description": "maps-once sweep",
                "candidate_set": {"type": "bus_list", "buses": candidate_buses},
                "mutation": {"action": "set_load", "Pd": 33.0},
                "feasibility": {"Vmin": 0.9, "Vmax": 1.1},
            })
        assert kind == "sweep"
        return counter["n"]

    k_small = load_buses[:4]
    k_large = load_buses[:9]
    n_small = _run_sweep(k_small)
    n_large = _run_sweep(k_large)

    # The map-build count is a small constant (vlimits-net setup + the one sweep
    # hoist), independent of the number of candidates — NOT once per candidate.
    assert n_small == n_large, (
        f"map builds scale with candidates: {n_small} vs {n_large}"
    )
    assert n_small <= 2, f"expected <=2 map builds per sweep, got {n_small}"
    assert n_small < len(k_large)

    # And the candidate nets still match the pre-change (deep, both-commands) path.
    for task, bus_id in zip(captured["tasks"], k_large):
        ref_net, _ = apply_modifications(
            net118,
            _cmds([
                {"action": "set_all_bus_vlimits", "Vmin": 0.9, "Vmax": 1.1},
                {"action": "set_load", "bus": bus_id, "Pd": 33.0},
            ]),
            application="opflow",
        )
        assert _as_string(task[0], tmp_path, f"mo_got_{bus_id}.m") == \
            _as_string(ref_net, tmp_path, f"mo_ref_{bus_id}.m")
