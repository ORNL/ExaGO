"""Tests for the GridKit helpers: commands, topology, validation, modifier,
check registry, schema text and results reader."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import pytest

from agentigrid.gridkit_engine import criteria
from agentigrid.gridkit_engine.commands import (
    AddBusFault,
    FaultStudy,
    RecordVariables,
    parse_command,
    parse_fault_study,
)
from agentigrid.gridkit_engine.modifier import apply_modifications, prepare_fault_case, write_case
from agentigrid.gridkit_engine.schema_description import command_schema_text
from agentigrid.gridkit_engine.sweep_metrics import (
    CHECKS,
    FaultOutcome,
    required_variables,
    run_checks,
    screen_verdict,
)
from agentigrid.gridkit_engine.topology import (
    branches_at_bus,
    build_adjacency,
    buses_within,
    fault_locations,
    format_locations_view,
)
from agentigrid.gridkit_engine.validation import (
    resolve_fault_buses,
    validate_command,
    validate_fault_study,
)
from agentigrid.gridkit_parsers.case_parser import build_element_map, load_case
from agentigrid.gridkit_parsers.results_parser import read_results
from agentigrid.gridkit_parsers.status_parser import RunStatus, parse_run_status

ROOT = Path(__file__).resolve().parents[2]
BINARY = ROOT / "applications" / "gridkit" / "DynamicSimulation"
IEEE39 = ROOT / "data" / "gridkit" / "examples" / "IEEE39.case.json"


def _chain_case() -> dict:
    """Buses 1-2-3-4 in a chain plus 2-5; machines at 1 and 4; one fault at 3;
    a case-level monitor entry."""
    branches = [(1, 2), (2, 3), (3, 4), (2, 5)]
    return {
        "header": {"case_name": "Chain"},
        "buses": [{"number": b, "name": str(b)} for b in range(1, 6)],
        "monitors": [{"file_name": "mon.csv", "format": "csv"}],
        "devices": (
            [{"class": "Branch", "id": f"br_{a}_{b}", "ports": {"bus1": a, "bus2": b},
              "params": {"R": 0.0, "X": 0.1}} for a, b in branches]
            + [{"class": "Genrou", "id": "g1", "ports": {"bus": 1}, "mon": ["omega"]},
               {"class": "GenClassical", "id": "g4", "ports": {"bus": 4}},
               {"class": "BusFault", "id": "f3", "ports": {"bus": 3},
                "params": {"state0": False, "R": 0.0, "X": 0.05}}]
        ),
    }


@pytest.fixture
def case() -> dict:
    return _chain_case()


@pytest.fixture
def emap(case) -> dict:
    return build_element_map(case)


class TestCommands:
    def test_parse(self):
        assert parse_command({"action": "add_bus_fault", "bus": 3}) == AddBusFault(bus=3)
        rec = parse_command({"action": "record_variables", "buses": ["Vm", "Va"]})
        assert rec.buses == ["Vm", "Va"] and "delta" in rec.machines

    def test_parse_errors(self):
        with pytest.raises(ValueError, match="Unknown action"):
            parse_command({"action": "trip_line"})
        with pytest.raises(ValueError, match="missing"):
            parse_command({"action": "add_bus_fault"})

    def test_fault_study_defaults_are_pjm_values(self):
        s = parse_fault_study({"action": "fault_screen", "poi": 2, "reasoning": "x"})
        assert s.tmax_s == criteria.SIM_LENGTH_S
        assert s.duration_s() == pytest.approx(6.25 / 60)
        assert s.clear_time_s() == pytest.approx(1.0 + 6.25 / 60)

    def test_fault_study_unknown_field(self):
        with pytest.raises(ValueError, match="reclose"):
            parse_fault_study({"poi": 2, "reclose": True})


class TestTopology:
    def test_adjacency_and_hops(self, emap):
        assert build_adjacency(emap)[2] == {1, 3, 5}
        assert buses_within(emap, 2, 1) == {2: 0, 1: 1, 3: 1, 5: 1}
        assert buses_within(emap, 1, 2) == {1: 0, 2: 1, 3: 2, 5: 2}

    def test_fault_locations_poi_first(self, emap):
        assert fault_locations(emap, 3, 1) == [3, 2, 4]

    def test_unknown_bus(self, emap):
        with pytest.raises(ValueError):
            buses_within(emap, 99, 1)

    def test_branches_and_view(self, emap):
        assert {d["id"] for d in branches_at_bus(emap, 2)} == {"br_1_2", "br_2_3", "br_2_5"}
        view = format_locations_view(emap, 3, 1)
        assert "POI: 3" in view and "1 away: 2, 4" in view


class TestValidation:
    def test_command_checks(self, emap):
        assert validate_command(AddBusFault(bus=2), emap).valid
        assert not validate_command(AddBusFault(bus=9), emap).valid
        assert not validate_command(AddBusFault(bus=2, R=0, X=0), emap).valid
        assert validate_command(AddBusFault(bus=3), emap).warnings
        assert not validate_command(RecordVariables(machines=["efd"]), emap).valid

    def test_study_checks(self, emap):
        assert validate_fault_study(FaultStudy(poi=2), emap).valid
        assert not validate_fault_study(FaultStudy(), emap).valid                      # no location
        assert not validate_fault_study(FaultStudy(poi=2, all_buses=True), emap).valid
        assert not validate_fault_study(FaultStudy(buses=[2, 9]), emap).valid
        assert not validate_fault_study(FaultStudy(poi=2, tmax_s=1.05), emap).valid     # clears after end
        assert not validate_fault_study(FaultStudy(poi=2, checks=["relay"]), emap).valid
        assert not validate_fault_study(FaultStudy(all_buses=True), emap, max_faults=3).valid

    def test_study_warnings(self, emap):
        v = validate_fault_study(FaultStudy(poi=2, tmax_s=5.0), emap)
        assert v.valid and any("damping" in w for w in v.warnings)

    def test_resolve(self, emap):
        assert resolve_fault_buses(FaultStudy(all_buses=True), emap) == [1, 2, 3, 4, 5]
        assert resolve_fault_buses(FaultStudy(buses=[4, 2, 4]), emap) == [4, 2]


class TestModifier:
    def test_overlay_does_not_touch_original(self, case, emap):
        before = json.dumps(case, sort_keys=True)
        work, report = apply_modifications(case, [AddBusFault(bus=2)], emap)
        assert json.dumps(case, sort_keys=True) == before
        assert "monitors" not in work
        assert any(d["id"] == "agentigrid_fault_2" for d in work["devices"])
        assert not report.errors

    def test_existing_fault_reused_with_new_impedance(self, case, emap):
        work, _ = apply_modifications(case, [AddBusFault(bus=3, X=0.02)], emap)
        faults = [d for d in work["devices"] if d["class"] == "BusFault"]
        assert len(faults) == 1 and faults[0]["params"]["X"] == 0.02

    def test_invalid_command_skipped(self, case, emap):
        work, report = apply_modifications(case, [AddBusFault(bus=9)], emap)
        assert report.errors and len(work["devices"]) == len(case["devices"])

    def test_prepare_fault_case(self, case, emap):
        work, wmap, fids, _ = prepare_fault_case(case, FaultStudy(poi=3, hops=1), emap)
        assert fids == {3: 0, 2: 1, 4: 2}            # existing fault keeps element 0
        assert {f["bus"] for f in wmap["bus_faults"]} == {2, 3, 4}
        machines = [d for d in work["devices"] if d["class"] in ("Genrou", "GenClassical")]
        assert all(set(d["mon"]) >= {"delta", "omega"} for d in machines)
        assert all(b["mon"] == ["Vm"] for b in work["buses"])


class TestChecksAndVerdict:
    def test_registry(self):
        assert set(FaultStudy().checks) <= set(CHECKS)
        need = required_variables(["angle_stability", "voltage_recovery"])
        assert need == {"machines": {"delta"}, "buses": {"Vm"}}

    def test_verdict_pass_fail_and_skipped(self):
        ok = RunStatus(solved=True, reason="solved")
        bad = RunStatus(solved=False, reason="solver failed: IDASolve IDA_CONV_FAIL")
        passed = criteria.CheckResult("damping", "pass", 0.05, 0.03, "5%", "src")
        failed = criteria.CheckResult("voltage_recovery", "fail", 0.6, 0.7, "0.600 pu at bus 7", "src")
        outs = [FaultOutcome(1, ok, [passed]), FaultOutcome(2, ok, [failed]), FaultOutcome(3, bad)]
        text = screen_verdict("Case", outs, "note")
        assert text.startswith("VERDICT: Case does NOT pass: 2 of 3 bus faults failed")
        assert "0.600 pu at bus 7" in text and "IDA_CONV_FAIL" in text
        assert "D3: high-speed reclosing" in text

    def test_verdict_not_confirmed_when_unjudged(self):
        ok = RunStatus(solved=True, reason="solved")
        nt = criteria.CheckResult("damping", "not_tested", None, 0.03, "run too short", "src")
        text = screen_verdict("Case", [FaultOutcome(1, ok, [nt])], "note")
        assert "NOT CONFIRMED" in text and "run too short" in text

    def test_steady_state(self):
        t = [i * 0.1 for i in range(50)]
        flat = {"a": [0.1] * 50, "b": [0.3] * 50}
        drift = {"a": [0.1] * 50, "b": [0.3 + 0.001 * i for i in range(50)]}
        vm = {1: [1.0] * 50}
        assert criteria.check_steady_state(t, flat, vm).status == "pass"
        assert criteria.check_steady_state(t, drift, vm).status == "fail"


class TestSchemaAndReader:
    def test_schema_mentions_checks_and_skipped(self):
        text = command_schema_text()
        assert "fault_screen" in text
        assert all(name in text for name in CHECKS)
        assert all(tid in text for tid, _, _ in criteria.SKIPPED_TESTS)

    def test_read_results(self, tmp_path, emap):
        csv_file = tmp_path / "out.csv"
        csv_file.write_text(
            "t,Bus_2_Vm,Genrou_g1_delta,Genrou_g1_omega,Other_x_y\n"
            "0,1.0,0.5,0,0\n"
            "1,0.9,0.6,0.001,0\n"
        )
        s = read_results(csv_file, emap)
        assert s.t == [0.0, 1.0]
        assert s.bus_variable("Vm") == {2: [1.0, 0.9]}
        assert s.device_variable("delta") == {"g1": [0.5, 0.6]}
        assert s.device_bus["g1"] == 1
        assert s.unmatched == ["Other_x_y"]


@pytest.mark.skipif(
    not (BINARY.exists() and IEEE39.exists()),
    reason="GridKit DynamicSimulation or IEEE39 case not linked",
)
def test_live_poi_screen(tmp_path):
    """IEEE39, POI bus 16 and one bus away, 3 s runs: every fault solves and
    passes angle stability and voltage recovery."""
    case = load_case(IEEE39)
    emap = build_element_map(case, IEEE39)
    study = FaultStudy(poi=16, hops=1, tmax_s=4.0, checks=["angle_stability", "voltage_recovery"])
    assert validate_fault_study(study, emap).valid
    work, wmap, fids, _ = prepare_fault_case(case, study, emap)
    write_case(work, tmp_path / "case.json")
    outcomes = []
    for bus, eid in fids.items():
        run = tmp_path / f"bus_{bus}"
        run.mkdir()
        (run / "s.json").write_text(json.dumps({
            "system_model_file": "../case.json", "dt_monitor": 0.01, "tmax": study.tmax_s,
            "output_file": "out.csv",
            "events": [{"time": study.start_s, "type": "fault_on", "element_id": eid},
                       {"time": study.clear_time_s(), "type": "fault_off", "element_id": eid}],
        }))
        p = subprocess.run([str(BINARY), "s.json"], cwd=run, capture_output=True, text=True, timeout=120)
        status = parse_run_status(p.returncode, p.stdout, p.stderr, run / "out.csv", study.tmax_s)
        checks = run_checks(read_results(run / "out.csv", wmap), study.checks, study.clear_time_s())
        outcomes.append(FaultOutcome(bus, status, checks))
    assert [o.bus for o in outcomes] == [16, 15, 17, 19, 21, 24]
    assert all(o.status == "pass" for o in outcomes), screen_verdict("IEEE39", outcomes, "")
