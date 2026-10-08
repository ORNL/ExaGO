"""Fault-screen options: start_from (case / latest ExaGO steady state) and
application (DynamicSimulation / ContingencyAnalysis)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentigrid.exago_parsers.matpower_parser import parse_matpower
from agentigrid.exago_parsers.matpower_writer import write_matpower
from agentigrid.gridkit_engine.commands import parse_fault_study
from agentigrid.gridkit_engine.modifier import prepare_fault_case
from agentigrid.gridkit_parsers.case_parser import build_element_map

from tests.gridkit.test_agent_loop import BINARY_DIR, IEEE39, _loop

ROOT = Path(__file__).resolve().parents[2]
EXAGO_200 = ROOT / "data" / "exago" / "examples" / "case_ACTIVSg200.m"
GRIDKIT_200 = ROOT / "data" / "gridkit" / "examples" / "ACTIVSg200.case.json"

needs_gridkit = pytest.mark.skipif(
    not ((BINARY_DIR / "DynamicSimulation").exists() and IEEE39.exists()),
    reason="GridKit DynamicSimulation or IEEE39 case not linked",
)
needs_ca = pytest.mark.skipif(
    not ((BINARY_DIR / "ContingencyAnalysis").exists() and IEEE39.exists()),
    reason="GridKit ContingencyAnalysis or IEEE39 case not linked",
)
needs_200 = pytest.mark.skipif(
    not ((BINARY_DIR / "DynamicSimulation").exists() and EXAGO_200.exists()
         and GRIDKIT_200.exists()),
    reason="GridKit or ACTIVSg200 ExaGO/GridKit cases not linked",
)

COMPLETE = {"action": "complete", "summary": "VERDICT: ...", "details": "none"}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_defaults():
    study = parse_fault_study({"action": "fault_screen", "buses": [3]})
    assert study.start_from == "case" and study.application == "DynamicSimulation"


@pytest.mark.parametrize("field,value", [("start_from", "yesterday"),
                                         ("application", "PowerFlow")])
def test_unknown_values_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        parse_fault_study({"action": "fault_screen", "buses": [3], field: value})


# ---------------------------------------------------------------------------
# Case copy: other fault devices
# ---------------------------------------------------------------------------

@needs_gridkit
@pytest.mark.parametrize("only,n_faults", [(False, 39), (True, 2)])
def test_other_faults_in_case_copy(only, n_faults):
    # IEEE39 ships a fault device at every bus.
    case = json.loads(IEEE39.read_text())
    emap = build_element_map(case)
    study = parse_fault_study({"buses": [16, 3]})
    work, wmap, fault_ids, _ = prepare_fault_case(case, study, emap, only_study_faults=only)
    assert len(wmap["bus_faults"]) == n_faults
    assert set(fault_ids) == {16, 3}
    # element ids point at the right devices
    by_id = {f["element_id"]: f["bus"] for f in wmap["bus_faults"]}
    assert all(by_id[eid] == bus for bus, eid in fault_ids.items())


# ---------------------------------------------------------------------------
# ContingencyAnalysis gives the same results as DynamicSimulation
# ---------------------------------------------------------------------------

def _screen_rows(session):
    journal = json.loads(session.journal_path.read_text())
    return next(e for e in journal["entries"] if e["action"] == "fault_screen")


@needs_ca
def test_contingency_analysis_matches_dynamic_simulation(tmp_path):
    screen = {"action": "fault_screen", "buses": [16, 3], "tmax_s": 4.0,
              "checks": ["angle_stability", "voltage_recovery"]}
    results = {}
    for app in ("DynamicSimulation", "ContingencyAnalysis"):
        loop, backend = _loop(tmp_path / app, [{**screen, "application": app}, COMPLETE])
        session = loop.run(IEEE39, "Fault buses 16 and 3.")
        assert session.termination_reason == "completed"
        results[app] = _screen_rows(session)
        assert f"GridKit {app}" in results[app]["verdict"]

    ca = results["ContingencyAnalysis"]
    run_dir = Path(ca["run_dir"]) / "contingency"
    case_copy = json.loads((Path(ca["run_dir"]) / "case.json").read_text())
    assert sorted(d["ports"]["bus"] for d in case_copy["devices"] if d["class"] == "BusFault") == [3, 16]
    assert sorted(p.name for p in run_dir.glob("output_*.csv")) == ["output_0.csv", "output_1.csv"]

    def values(entry):
        return {r["bus"]: (r["solved"], r["status"]) for r in entry["rows"]}

    assert values(ca) == values(results["DynamicSimulation"])
    assert all(solved for solved, _ in values(ca).values())
    assert ca["verdict"].split("(")[0] == results["DynamicSimulation"]["verdict"].split("(")[0]


# ---------------------------------------------------------------------------
# start_from: latest_steady_state
# ---------------------------------------------------------------------------

def _exago_workdir(root: Path, solved_from: Path) -> Path:
    """An ExaGO workdir whose newest journal ends with a converged OPFLOW run
    (its "solution" is *solved_from*, a case whose stored state is steady)."""
    run_dir = root / "iter_001"
    run_dir.mkdir(parents=True)
    write_matpower(parse_matpower(solved_from), run_dir / "opflowout.m")
    entry = {"iteration": 1, "description": "gen out", "convergence_status": "CONVERGED",
             "exago_command": {"mode": "single", "application": "opflow",
                               "argv": ["/bin/opflow", "-netfile", str(run_dir / "case.m")]}}
    (root / "journal_20261008_120000.json").write_text(json.dumps({"entries": [entry]}))
    return root


def _loop_with_exago(tmp_path, replies, exago_dir):
    from dataclasses import replace
    loop, backend = _loop(tmp_path, replies)
    loop._settings = replace(loop._settings, exago_workdir=exago_dir)
    return loop, backend


@needs_200
def test_latest_steady_state_chosen_by_llm(tmp_path):
    exago_dir = _exago_workdir(tmp_path / "exago", EXAGO_200)
    screen = {"action": "fault_screen", "buses": [135], "tmax_s": 3.0,
              "checks": ["angle_stability"], "start_from": "latest_steady_state"}
    loop, backend = _loop_with_exago(tmp_path, [screen, COMPLETE], exago_dir)
    session = loop.run(GRIDKIT_200, "Fault bus 135.")

    assert session.termination_reason == "completed"
    assert session.steady_state and "OPFLOW iteration 1" in session.steady_state
    journal = json.loads(session.journal_path.read_text())
    actions = [e["action"] for e in journal["entries"]]
    # case base run, ExaGO start, its own base run, the screen
    assert actions[:4] == ["base_case", "steady_state", "base_case", "fault_screen"]
    assert (session.session_dir / "from_exago.case.json").exists()
    copy = json.loads((Path(journal["entries"][3]["run_dir"]) / "case.json").read_text())
    machines = {d["id"] for d in copy["devices"] if d["class"] in ("Genrou", "Gensal", "GenClassical")}
    # ExaGO has generators 161 and 197 off: their machines are not in the copy
    assert not any(m.startswith(("161_", "197_")) for m in machines)
    assert any(m.startswith("135_") for m in machines)
    assert "starting from the latest ExaGO steady state" in journal["entries"][3]["verdict"]
    # what the LLM was told it may choose
    assert "latest_steady_state" in backend.prompts[0][0]


@needs_gridkit
def test_goal_asking_for_steady_state_needs_start_from(tmp_path):
    loop, backend = _loop_with_exago(tmp_path, [
        {"action": "fault_screen", "buses": [16], "tmax_s": 2.0, "checks": ["angle_stability"]},
        COMPLETE,
    ], tmp_path / "no_exago")
    session = loop.run(IEEE39, "Use the most recent steady state and fault bus 16.")
    assert session.termination_reason == "completed"
    assert '"start_from": "latest_steady_state"' in backend.prompts[1][1]


@needs_gridkit
def test_unusable_steady_state_is_reported_not_run(tmp_path):
    screen = {"action": "fault_screen", "buses": [16], "tmax_s": 2.0,
              "checks": ["angle_stability"], "start_from": "latest_steady_state"}
    loop, backend = _loop_with_exago(tmp_path, [screen, COMPLETE], tmp_path / "no_exago")
    session = loop.run(IEEE39, "Fault bus 16.")
    assert session.termination_reason == "completed"
    assert "NOT run" in backend.prompts[1][1] and "No ExaGO journal" in backend.prompts[1][1]
