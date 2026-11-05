"""Tests for starting GridKit from the most recent ExaGO steady state."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from agentigrid.exago_engine.executor import _CMD_BUILDERS
from agentigrid.exago_parsers.matpower_parser import parse_matpower
from agentigrid.gridkit_engine.from_exago import (
    SteadyStateError,
    apply_steady_state,
    exago_shifts_to_degrees,
    find_latest_steady_state,
    goal_wants_steady_state,
)
from agentigrid.gridkit_parsers.case_parser import MACHINE_CLASSES

ROOT = Path(__file__).resolve().parents[2]
EXAGO_200 = ROOT / "data" / "exago" / "examples" / "case_ACTIVSg200.m"
GRIDKIT_200 = ROOT / "data" / "gridkit" / "examples" / "ACTIVSg200.case.json"
needs_cases = pytest.mark.skipif(
    not (EXAGO_200.exists() and GRIDKIT_200.exists()),
    reason="ACTIVSg200 ExaGO/GridKit cases not linked",
)


# ---------------------------------------------------------------------------
# Goal phrase and ExaGO command
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("goal", [
    "Please use the most recent steady state simulation results and fault bus 5",
    "start from the latest steady-state solution",
    "Use the last steady state",
])
def test_goal_asks_for_steady_state(goal):
    assert goal_wants_steady_state(goal)


@pytest.mark.parametrize("goal", ["Fault every bus", "steady state check of bus 3", ""])
def test_goal_does_not_ask(goal):
    assert not goal_wants_steady_state(goal)


@pytest.mark.parametrize("app,saves", [("opflow", True), ("pflow", True), ("dcopflow", False),
                                       ("scopflow", False)])
def test_exago_saves_steady_state(app, saves):
    cmd = _CMD_BUILDERS[app](Path("/bin/x"), Path("/wd/case.m"), None)
    assert ("-save_output" in cmd) == saves


# ---------------------------------------------------------------------------
# Finding the most recent steady state
# ---------------------------------------------------------------------------

def _entry(it, run_dir, app="opflow", status="CONVERGED", mode="single", desc="d"):
    return {
        "iteration": it, "description": desc, "convergence_status": status,
        "exago_command": {"mode": mode, "application": app,
                          "argv": [f"/bin/{app}", "-netfile", str(run_dir / "case.m"), "-print_output"]},
    }


def _workdir(tmp_path, entries, name="journal_20261007_120000.json", solved=("opflowout.m",)):
    for e in entries:
        if e.get("exago_command"):
            d = Path(e["exago_command"]["argv"][2]).parent
            d.mkdir(parents=True, exist_ok=True)
            for f in solved:
                (d / f).write_text("")
    (tmp_path / name).write_text(json.dumps({"entries": entries}))
    return tmp_path


def test_last_simulation_of_newest_journal(tmp_path):
    _workdir(tmp_path, [_entry(0, tmp_path / "old")], name="journal_20261007_100000.json")
    _workdir(tmp_path, [_entry(0, tmp_path / "a"), _entry(1, tmp_path / "b", desc="gen off"),
                        {"iteration": 2, "description": "Search completed", "exago_command": None}])
    src = find_latest_steady_state(tmp_path)
    assert src.iteration == 1 and src.run_dir == tmp_path / "b"
    assert src.solved_file == tmp_path / "b" / "opflowout.m"
    assert "gen off" in src.describe()


def test_pflow_uses_journal_status(tmp_path):
    # PFLOW's saved file says converged = 0 even when it converged.
    _workdir(tmp_path, [_entry(0, tmp_path / "a", app="pflow", status="CONVERGED")],
             solved=("pflowout.m",))
    assert find_latest_steady_state(tmp_path).application == "pflow"


@pytest.mark.parametrize("entry,match", [
    (dict(status="DID NOT CONVERGE"), "did not converge"),
    (dict(mode="sweep"), "not a single steady state"),
    (dict(app="dcopflow"), "only"),
])
def test_most_recent_must_be_usable(tmp_path, entry, match):
    # An older usable iteration exists, but there is no fallback to it.
    _workdir(tmp_path, [_entry(0, tmp_path / "a"), _entry(1, tmp_path / "b", **entry)])
    with pytest.raises(SteadyStateError, match=match):
        find_latest_steady_state(tmp_path)


def test_missing_solution_file(tmp_path):
    _workdir(tmp_path, [_entry(0, tmp_path / "a")], solved=())
    with pytest.raises(SteadyStateError, match="missing"):
        find_latest_steady_state(tmp_path)


def test_no_journal(tmp_path):
    with pytest.raises(SteadyStateError, match="No ExaGO journal"):
        find_latest_steady_state(tmp_path)


# ---------------------------------------------------------------------------
# Conversion (ACTIVSg200: same bus numbers in both cases)
# ---------------------------------------------------------------------------

@pytest.fixture
def pair():
    return json.loads(GRIDKIT_200.read_text()), parse_matpower(EXAGO_200)


def _devices(case, cls):
    return [d for d in case["devices"] if d["class"] == cls]


@needs_cases
def test_operating_point_copied(pair):
    case, net = pair
    out, _ = apply_steady_state(case, net)
    bus = {b["number"]: b for b in out["buses"]}[135]
    mb = {b.bus_i: b for b in net.buses}[135]
    assert math.hypot(bus["init"]["Vr"], bus["init"]["Vi"]) == pytest.approx(mb.Vm)
    assert math.degrees(math.atan2(bus["init"]["Vi"], bus["init"]["Vr"])) == pytest.approx(mb.Va)
    gen = next(g for g in net.generators if g.bus == 135)
    m = next(d for d in _devices(out, "Genrou") if d["ports"]["bus"] == 135)
    assert m["params"]["p0"] == pytest.approx(gen.Pg / net.baseMVA)
    assert m["params"]["q0"] == pytest.approx(gen.Qg / net.baseMVA)
    load2 = next(d for d in _devices(out, "LoadZIP") if d["ports"]["bus"] == 2)
    assert load2["params"]["Pnom"] == pytest.approx(7.39 / 100)
    assert case["buses"][0]["init"] != out["buses"][0]["init"]   # input case not changed


@needs_cases
def test_offline_machines_removed_with_controls(pair):
    case, net = pair
    out, notes = apply_steady_state(case, net)
    ids = {d["id"] for d in out["devices"]}
    for unit in ("161_1", "197_1"):
        assert not any(i.startswith(unit) for i in ids)
    used = {v for d in out["devices"] for k, v in d["ports"].items() if not k.startswith("bus")}
    assert {s["signal_id"] for s in out["signals"]} == used
    assert any("161_1_genrou" in n for n in notes)


@needs_cases
def test_generator_outage_removes_its_machine(pair):
    case, net = pair
    next(g for g in net.generators if g.bus == 135).status = 0
    out, _ = apply_steady_state(case, net)
    assert not any(d["ports"].get("bus") == 135 for d in out["devices"] if d["class"] in MACHINE_CLASSES)


@needs_cases
def test_shunts_and_branches_from_exago(pair):
    case, net = pair
    net.branches[0].status = 0
    out, notes = apply_steady_state(case, net)
    shunts = {d["ports"]["bus"]: d for d in _devices(out, "LoadZIP") if d["id"].startswith("shunt_")}
    assert sorted(shunts) == [15, 95, 100, 194]
    vm = {b.bus_i: b.Vm for b in net.buses}[15]
    assert shunts[15]["params"]["Qnom"] == pytest.approx(-30 * vm * vm / 100)
    assert len(_devices(out, "Branch")) == sum(1 for b in net.branches if b.status)
    assert any("64-82" in n for n in notes)


@needs_cases
def test_exago_phase_shifts_read_as_radians(pair):
    # ExaGO's saved solution writes shift angles in radians; GridKit wants radians.
    case, net = pair
    br = net.branches[0]
    br.angle = math.radians(-12.0)
    notes = exago_shifts_to_degrees(net)
    assert br.angle == pytest.approx(-12.0) and "radians to degrees" in notes[0]
    out, _ = apply_steady_state(case, net)
    phases = [d["params"]["phase"] for d in _devices(out, "Branch") if d["params"]["phase"]]
    assert phases == [pytest.approx(math.radians(-12.0))]


def _with_static_gen(case, net, name):
    # A 5 MW renewable at bus 2, as GridKit's 10k case writes it; ExaGO nets it from Pd.
    case["header"]["case_name"] = name
    case["devices"].append({"class": "LoadZIP", "id": "load_2_static_gen_1", "ports": {"bus": 2},
                            "params": {"Pnom": -0.05, "Qnom": -0.01, "alphaI": 1.0, "alphaP": 0.0}})
    b = next(b for b in net.buses if b.bus_i == 2)
    b.Pd -= 5.0
    b.Qd -= 1.0
    return b


@needs_cases
def test_static_gen_kept_for_10k_case(pair):
    case, net = pair
    _with_static_gen(case, net, "ACTIVSg10k")
    out, notes = apply_steady_state(case, net)
    at2 = {d["id"]: d["params"] for d in _devices(out, "LoadZIP") if d["ports"]["bus"] == 2}
    assert at2["load_2_static_gen_1"] == {"Pnom": -0.05, "Qnom": -0.01, "alphaI": 1.0, "alphaP": 0.0}
    assert at2["load_2"]["Pnom"] == pytest.approx(7.39 / 100) and at2["load_2"]["alphaI"] == 0.0
    assert any("static_gen" in n for n in notes)


@needs_cases
def test_static_gen_rule_off_for_other_cases(pair):
    case, net = pair
    b = _with_static_gen(case, net, "ACTIVSg200")
    out, notes = apply_steady_state(case, net)
    at2 = [d for d in _devices(out, "LoadZIP") if d["ports"]["bus"] == 2]
    assert [d["id"] for d in at2] == ["load_2"] or [d["id"] for d in at2] == ["load_2_static_gen_1"]
    assert at2[0]["params"]["Pnom"] == pytest.approx(b.Pd / 100)
    assert not any("static_gen" in n for n in notes)


@needs_cases
def test_machines_pair_in_file_order(pair):
    # ACTIVSg10k lists a bus's units as 1, 10, 2, ...; pairing must not re-sort them.
    case, net = pair
    gen = next(g for g in net.generators if g.bus == 135 and g.status)
    m1 = next(d for d in _devices(case, "Genrou") if d["ports"]["bus"] == 135)
    m10 = copy.deepcopy(m1)
    m10["id"] = m1["id"].replace("_1_", "_10_", 1)
    case["devices"].insert(case["devices"].index(m1) + 1, m10)
    second = copy.deepcopy(gen)
    second.Pg = 12.34
    net.generators.insert(net.generators.index(gen) + 1, second)
    out, _ = apply_steady_state(case, net)
    p0 = {d["id"]: d["params"]["p0"] for d in _devices(out, "Genrou") if d["ports"]["bus"] == 135}
    assert p0[m1["id"]] == pytest.approx(gen.Pg / 100)
    assert p0[m10["id"]] == pytest.approx(12.34 / 100)


@needs_cases
@pytest.mark.parametrize("offset,snapped", [(0.005, True), (-0.005, True), (0.05, False)])
def test_dispatch_at_limit_snapped(pair, offset, snapped):
    case, net = pair
    gen = next(g for g in net.generators if g.bus == 135 and g.status)
    gen.Pg = gen.Pmax + offset
    out, notes = apply_steady_state(case, net)
    m = next(d for d in _devices(out, "Genrou") if d["ports"]["bus"] == 135)
    assert m["params"]["p0"] == pytest.approx((gen.Pmax if snapped else gen.Pg) / 100)
    assert any("set to the limit" in n for n in notes) == snapped


@needs_cases
def test_dispatch_at_valve_limit_set_inside(pair):
    # Tgov1 rejects a start outside [Pvmin, Pvmax] with a 4-epsilon tolerance.
    case, net = pair
    gov = next(d for d in _devices(case, "Tgov1"))
    m = next(d for d in case["devices"] if d["class"] in MACHINE_CLASSES
             and d["ports"].get("pmech") == gov["ports"]["pmech"])
    gen = next(g for g in net.generators if g.bus == m["ports"]["bus"] and g.status)
    gen.Pg = gov["params"]["Pvmax"] * gov["params"]["Trate"] + 0.004
    out, notes = apply_steady_state(case, net)
    p0 = next(d for d in out["devices"] if d["id"] == m["id"])["params"]["p0"]
    assert p0 * 100 / gov["params"]["Trate"] <= gov["params"]["Pvmax"]
    assert p0 * 100 == pytest.approx(gov["params"]["Pvmax"] * gov["params"]["Trate"])
    assert any("valve limit" in n for n in notes)


@needs_cases
def test_generator_without_model_is_an_error(pair):
    case, net = pair
    machine_buses = {d["ports"]["bus"] for d in case["devices"] if d["class"] in MACHINE_CLASSES}
    g = next(g for g in net.generators if g.bus not in machine_buses)
    g.status = 1
    with pytest.raises(SteadyStateError, match=f"bus\\(es\\) \\[{g.bus}\\].*no dynamic model"):
        apply_steady_state(case, net)


@needs_cases
def test_bus_mismatch_is_an_error(pair):
    case, net = pair
    case["buses"] = case["buses"][1:]
    with pytest.raises(SteadyStateError, match="Bus numbers differ"):
        apply_steady_state(case, net)
