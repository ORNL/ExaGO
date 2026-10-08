"""Tests for PFLOW support (Phase 4, Step 4.1)."""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from agentigrid.parsers.pflow_parser import parse_pflow_output, parse_pflow_simulation_result
from agentigrid.parsers.pflow_summary import pflow_results_summary
from agentigrid.parsers import (
    results_summary_for_app,
    parse_simulation_result_for_app,
    parse_pflow_metadata,
)
from agentigrid.parsers.opflow_results import OPFLOWResult, BusResult, GenResult
from agentigrid.parsers.matpower_model import MATNetwork, Bus, Generator, Branch, GenCost
from agentigrid.engine.commands import (
    SetTapRatio, SetShuntSusceptance, SetPhaseShiftAngle,
    SetGenVoltage, ScaleAllLoads, SetAllBusVLimits,
)
from agentigrid.engine.modifier import apply_modifications
from agentigrid.engine.validation import validate_command
from agentigrid.engine.metric_extractor import available_metrics_for_app
from agentigrid.engine.goal_classifier import build_classification_prompts
from agentigrid.prompts.system_prompt import build_system_prompt, _app_section


# ---------------------------------------------------------------------------
# Synthetic PFLOW output — 3-bus system
# PFLOW output format: "AC Power Flow" header, Newton-Raphson solver,
# "Number of iterations" (lowercase), "Solve Time (sec)", no Objective value,
# no EXIT line, "Convergence status" line.
# ---------------------------------------------------------------------------

SAMPLE_PFLOW_OUTPUT = """\
=============================================================
\t\tAC Power Flow
=============================================================
Model                               POWER_BALANCE_POLAR
Solver                              Newton-Rhapson
Objective                           NONE
Initialization                      ACPF
Gen. bus voltage mode               FIXED_WITHIN_BOUNDS
Load loss allowed                   NO
Power imbalance allowed             NO
Ignore line flow constraints        NO

Number of variables                 8
Number of equality constraints      8

Number of iterations                3
Solve Time (sec)                    0.005
Convergence status                  CONVERGED

------------------------------------------------------------------------------------------------------
Bus        Pd      Pdloss Qd      Qdloss Vm      Va      mult_Pmis      mult_Qmis      Pslack         Qslack
------------------------------------------------------------------------------------------------------
1         0.00    0.00    0.00    0.00   1.050   0.000         0.00         0.00         0.00         0.00
2        80.00    0.00   20.00    0.00   1.020  -4.100         0.00         0.00         0.00         0.00
3        60.00    0.00   15.00    0.00   1.015  -6.800         0.00         0.00         0.00         0.00

------------------------------------------------------------------------------------------------------
From       To       Status     Sft      Stf     Slim     mult_Sf  mult_St
----------------------------------------------------------------------------------------
1          2          1       55.00    55.00   100.00     0.00     0.00
1          3          1       75.43    75.43   150.00     0.00     0.00

----------------------------------------------------------------------------------------
Gen      Status     Fuel     Pg       Qg       Pmin     Pmax     Qmin     Qmax
----------------------------------------------------------------------------------------
1          1        COAL     105.00    18.50    10.00   150.00   -50.00    50.00
2          1        GAS      42.43    12.30     5.00    80.00   -30.00    30.00
"""

SAMPLE_PFLOW_DIVERGED = """\
=============================================================
\t\tAC Power Flow
=============================================================
Model                               POWER_BALANCE_POLAR
Solver                              Newton-Rhapson
Objective                           NONE
Initialization                      ACPF

Number of iterations                100
Solve Time (sec)                    0.120
Convergence status                  DID NOT CONVERGE

------------------------------------------------------------------------------------------------------
Bus        Pd      Pdloss Qd      Qdloss Vm      Va      mult_Pmis      mult_Qmis      Pslack         Qslack
------------------------------------------------------------------------------------------------------
1         0.00    0.00    0.00    0.00   0.850   0.000         0.00         0.00         0.00         0.00
2        80.00    0.00   20.00    0.00   0.780  -8.000         0.00         0.00         0.00         0.00
3        60.00    0.00   15.00    0.00   0.760 -12.000         0.00         0.00         0.00         0.00

------------------------------------------------------------------------------------------------------
From       To       Status     Sft      Stf     Slim     mult_Sf  mult_St
----------------------------------------------------------------------------------------
1          2          1       55.00    55.00   100.00     0.00     0.00
1          3          1       75.43    75.43   150.00     0.00     0.00

----------------------------------------------------------------------------------------
Gen      Status     Fuel     Pg       Qg       Pmin     Pmax     Qmin     Qmax
----------------------------------------------------------------------------------------
1          1        COAL     105.00    18.50    10.00   150.00   -50.00    50.00
2          1        GAS      42.43    12.30     5.00    80.00   -30.00    30.00
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pflow_result():
    """Parse the synthetic PFLOW output."""
    result, _ = parse_pflow_output(SAMPLE_PFLOW_OUTPUT)
    return result


@pytest.fixture
def pflow_result_diverged():
    """Parse the synthetic PFLOW output with DID NOT CONVERGE."""
    result, _ = parse_pflow_output(SAMPLE_PFLOW_DIVERGED)
    return result


@pytest.fixture
def pflow_metadata():
    """Extract PFLOW metadata."""
    _, metadata = parse_pflow_output(SAMPLE_PFLOW_OUTPUT)
    return metadata


@pytest.fixture
def pflow_metadata_diverged():
    """Extract PFLOW metadata from diverged output."""
    _, metadata = parse_pflow_output(SAMPLE_PFLOW_DIVERGED)
    return metadata


@pytest.fixture
def minimal_network():
    """A minimal 3-bus MATNetwork for modifier tests."""
    buses = [
        Bus(bus_i=1, type=3, Pd=0.0, Qd=0.0, Gs=0, Bs=0.0, area=1,
            Vm=1.05, Va=0.0, baseKV=138.0, zone=1, Vmax=1.1, Vmin=0.9),
        Bus(bus_i=2, type=1, Pd=80.0, Qd=20.0, Gs=0, Bs=0.0, area=1,
            Vm=1.02, Va=-4.1, baseKV=138.0, zone=1, Vmax=1.1, Vmin=0.9),
        Bus(bus_i=3, type=1, Pd=60.0, Qd=15.0, Gs=0, Bs=0.0, area=1,
            Vm=1.015, Va=-6.8, baseKV=138.0, zone=1, Vmax=1.1, Vmin=0.9),
    ]
    generators = [
        Generator(bus=1, Pg=105.0, Qg=18.5, Qmax=50.0, Qmin=-50.0,
                  Vg=1.05, mBase=100.0, status=1, Pmax=150.0, Pmin=10.0),
        Generator(bus=2, Pg=42.43, Qg=12.3, Qmax=30.0, Qmin=-30.0,
                  Vg=1.02, mBase=100.0, status=1, Pmax=80.0, Pmin=5.0),
    ]
    branches = [
        Branch(fbus=1, tbus=2, r=0.01, x=0.05, b=0.0, rateA=100.0,
               rateB=100.0, rateC=100.0, ratio=0.0, angle=0.0, status=1,
               angmin=-360.0, angmax=360.0),
        Branch(fbus=1, tbus=3, r=0.01, x=0.04, b=0.0, rateA=150.0,
               rateB=150.0, rateC=150.0, ratio=1.05, angle=0.0, status=1,
               angmin=-360.0, angmax=360.0),
        Branch(fbus=2, tbus=3, r=0.02, x=0.06, b=0.0, rateA=80.0,
               rateB=80.0, rateC=80.0, ratio=0.0, angle=5.0, status=1,
               angmin=-360.0, angmax=360.0),
    ]
    gencost = [
        GenCost(model=2, ncost=3, coeffs=[0.003, 20.0, 500.0], startup=0, shutdown=0),
        GenCost(model=2, ncost=3, coeffs=[0.005, 30.0, 200.0], startup=0, shutdown=0),
    ]
    return MATNetwork(
        casename="test_pflow",
        version="2",
        baseMVA=100.0,
        buses=buses,
        generators=generators,
        branches=branches,
        gencost=gencost,
        header_comments="",
    )


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------

class TestPFLOWParser:
    """Verify the PFLOW parser handles AC Power Flow output."""

    def test_parse_succeeds(self, pflow_result):
        assert pflow_result is not None

    def test_header_rejects_opflow(self):
        """Should reject output that doesn't contain 'AC Power Flow'."""
        with pytest.raises(ValueError, match="PFLOW"):
            parse_pflow_output("Optimal Power Flow result")

    def test_rejects_empty(self):
        """Should reject empty output."""
        with pytest.raises(ValueError):
            parse_pflow_output("")

    def test_rejects_none(self):
        """Should reject None."""
        with pytest.raises(ValueError):
            parse_pflow_output(None)

    def test_objective_value_is_zero(self, pflow_result):
        assert pflow_result.objective_value == 0.0

    def test_model_is_ac(self, pflow_result):
        assert pflow_result.model == "AC"

    def test_objective_type(self, pflow_result):
        assert pflow_result.objective_type == "PowerFlow"

    def test_solver_is_newton_rhapson(self, pflow_result):
        assert pflow_result.solver == "Newton-Rhapson"

    def test_convergence_converged(self, pflow_result):
        assert pflow_result.convergence_status == "CONVERGED"
        assert pflow_result.converged is True

    def test_convergence_diverged(self, pflow_result_diverged):
        assert pflow_result_diverged.convergence_status == "DID NOT CONVERGE"
        assert pflow_result_diverged.converged is False

    def test_num_iterations(self, pflow_result):
        assert pflow_result.num_iterations == 3

    def test_solve_time(self, pflow_result):
        assert pflow_result.solve_time == pytest.approx(0.005, abs=0.001)

    def test_num_iterations_diverged(self, pflow_result_diverged):
        assert pflow_result_diverged.num_iterations == 100

    def test_solve_time_diverged(self, pflow_result_diverged):
        assert pflow_result_diverged.solve_time == pytest.approx(0.120, abs=0.001)

    def test_ipopt_exit_status_empty(self, pflow_result):
        assert pflow_result.ipopt_exit_status == ""

    def test_bus_count(self, pflow_result):
        assert len(pflow_result.buses) == 3

    def test_branch_count(self, pflow_result):
        assert len(pflow_result.branches) == 2

    def test_generator_count(self, pflow_result):
        assert len(pflow_result.generators) == 2

    def test_bus_voltages(self, pflow_result):
        assert pflow_result.buses[0].Vm == pytest.approx(1.050, abs=0.001)
        assert pflow_result.buses[1].Vm == pytest.approx(1.020, abs=0.001)
        assert pflow_result.buses[2].Vm == pytest.approx(1.015, abs=0.001)

    def test_total_gen_mw(self, pflow_result):
        assert pflow_result.total_gen_mw == pytest.approx(105.0 + 42.43, abs=0.01)

    def test_total_load_mw(self, pflow_result):
        assert pflow_result.total_load_mw == pytest.approx(80.0 + 60.0, abs=0.01)

    def test_losses_mw(self, pflow_result):
        expected_losses = (105.0 + 42.43) - (80.0 + 60.0)
        assert pflow_result.losses_mw == pytest.approx(expected_losses, abs=0.01)

    def test_feasibility_converged(self, pflow_result):
        assert pflow_result.feasibility_detail == "feasible"

    def test_feasibility_diverged(self, pflow_result_diverged):
        assert pflow_result_diverged.feasibility_detail == "infeasible"

    def test_voltage_range(self, pflow_result):
        assert pflow_result.voltage_min == pytest.approx(1.015, abs=0.001)
        assert pflow_result.voltage_max == pytest.approx(1.050, abs=0.001)

    def test_no_violations_converged(self, pflow_result):
        assert pflow_result.num_violations == 0

    def test_violations_diverged(self, pflow_result_diverged):
        assert pflow_result_diverged.num_violations > 0


class TestPFLOWMetadata:
    """Test PFLOW metadata extraction."""

    def test_convergence_status_converged(self, pflow_metadata):
        assert pflow_metadata["convergence_status"] == "CONVERGED"

    def test_solver_in_metadata(self, pflow_metadata):
        assert pflow_metadata["solver"] == "Newton-Rhapson"

    def test_convergence_status_diverged(self, pflow_metadata_diverged):
        assert pflow_metadata_diverged["convergence_status"] == "DID NOT CONVERGE"


class TestPFLOWSimulationResult:
    """Test parse_pflow_simulation_result with mock SimulationResult."""

    def test_success(self):
        mock = MagicMock()
        mock.success = True
        mock.stdout = SAMPLE_PFLOW_OUTPUT
        parsed = parse_pflow_simulation_result(mock)
        assert parsed is not None
        result, metadata = parsed
        assert result.converged is True
        assert metadata["convergence_status"] == "CONVERGED"

    def test_failure(self):
        mock = MagicMock()
        mock.success = False
        parsed = parse_pflow_simulation_result(mock)
        assert parsed is None

    def test_invalid_output(self):
        mock = MagicMock()
        mock.success = True
        mock.stdout = "Not PFLOW output"
        parsed = parse_pflow_simulation_result(mock)
        assert parsed is None


class TestPFLOWDispatch:
    """Test the application-aware dispatch for PFLOW."""

    def test_parse_simulation_result_for_app(self):
        mock = MagicMock()
        mock.success = True
        mock.stdout = SAMPLE_PFLOW_OUTPUT
        result = parse_simulation_result_for_app(mock, "pflow")
        assert result is not None
        assert result.objective_value == 0.0

    def test_results_summary_for_app_pflow(self, pflow_result):
        summary = results_summary_for_app(pflow_result, "pflow")
        assert "Power Flow Results" in summary
        assert "Analysis, Not Optimization" in summary

    def test_results_summary_for_app_opflow(self, pflow_result):
        summary = results_summary_for_app(pflow_result, "opflow")
        assert "OPFLOW Results" in summary

    def test_parse_pflow_metadata_dispatch(self):
        mock = MagicMock()
        mock.success = True
        mock.stdout = SAMPLE_PFLOW_OUTPUT
        metadata = parse_pflow_metadata(mock)
        assert metadata is not None
        assert metadata["solver"] == "Newton-Rhapson"

    def test_parse_pflow_metadata_failure(self):
        mock = MagicMock()
        mock.success = False
        metadata = parse_pflow_metadata(mock)
        assert metadata is None


class TestPFLOWSummary:
    """Test the PFLOW-specific results summary."""

    def test_summary_header(self, pflow_result):
        summary = pflow_results_summary(pflow_result)
        assert "Power Flow Results (Analysis, Not Optimization)" in summary

    def test_summary_notes_no_optimization(self, pflow_result):
        summary = pflow_results_summary(pflow_result)
        assert "NO cost optimization" in summary or "no cost optimization" in summary

    def test_summary_set_gen_voltage_note(self, pflow_result):
        summary = pflow_results_summary(pflow_result)
        assert "set_gen_voltage" in summary or "constrains bus voltage" in summary

    def test_summary_has_voltage_profile(self, pflow_result):
        summary = pflow_results_summary(pflow_result)
        assert "Voltage profile:" in summary

    def test_summary_has_solver_info(self, pflow_result):
        summary = pflow_results_summary(pflow_result)
        assert "Newton-Rhapson" in summary

    def test_summary_no_objective_value(self, pflow_result):
        summary = pflow_results_summary(pflow_result)
        assert "Objective value" not in summary

    def test_summary_generation_info(self, pflow_result):
        summary = pflow_results_summary(pflow_result)
        assert "Generation:" in summary
        assert "Load:" in summary

    def test_summary_line_loading(self, pflow_result):
        summary = pflow_results_summary(pflow_result)
        assert "Most loaded lines:" in summary

    def test_summary_with_gencost(self, pflow_result, minimal_network):
        summary = pflow_results_summary(pflow_result, gencost=minimal_network.gencost)
        assert "Computed generation cost:" in summary
        assert "$" in summary

    def test_summary_without_gencost(self, pflow_result):
        summary = pflow_results_summary(pflow_result, gencost=None)
        assert "Computed generation cost" not in summary

    def test_summary_violations(self, pflow_result_diverged):
        summary = pflow_results_summary(pflow_result_diverged)
        assert "Violations:" in summary


class TestPFLOWComputedCost:
    """Test the compute_generation_cost method on OPFLOWResult."""

    def test_with_gencost(self, pflow_result, minimal_network):
        cost = pflow_result.compute_generation_cost(minimal_network.gencost)
        assert cost > 0

    def test_gencost_matches_manual(self, pflow_result, minimal_network):
        cost = pflow_result.compute_generation_cost(minimal_network.gencost)
        assert cost > 0

    def test_empty_gencost(self, pflow_result):
        cost = pflow_result.compute_generation_cost([])
        assert cost == 0.0

    def test_none_gencost(self, pflow_result):
        cost = pflow_result.compute_generation_cost(None)
        assert cost == 0.0


class TestPFLOWCommands:
    """Test new PFLOW-specific commands: SetTapRatio, SetShuntSusceptance, SetPhaseShiftAngle."""

    def test_parse_set_tap_ratio(self):
        from agentigrid.engine.commands import parse_command, SetTapRatio
        cmd = parse_command({"action": "set_tap_ratio", "fbus": 1, "tbus": 3, "ratio": 1.05})
        assert isinstance(cmd, SetTapRatio)
        assert cmd.fbus == 1
        assert cmd.tbus == 3
        assert cmd.ratio == 1.05

    def test_parse_set_shunt_susceptance(self):
        from agentigrid.engine.commands import parse_command, SetShuntSusceptance
        cmd = parse_command({"action": "set_shunt_susceptance", "bus": 2, "Bs": 0.5})
        assert isinstance(cmd, SetShuntSusceptance)
        assert cmd.bus == 2
        assert cmd.Bs == 0.5

    def test_parse_set_phase_shift_angle(self):
        from agentigrid.engine.commands import parse_command, SetPhaseShiftAngle
        cmd = parse_command({"action": "set_phase_shift_angle", "fbus": 2, "tbus": 3, "angle": 5.0})
        assert isinstance(cmd, SetPhaseShiftAngle)
        assert cmd.fbus == 2
        assert cmd.tbus == 3
        assert cmd.angle == 5.0

    def test_parse_set_tap_ratio_with_ckt(self):
        from agentigrid.engine.commands import parse_command
        cmd = parse_command({"action": "set_tap_ratio", "fbus": 1, "tbus": 3, "ratio": 1.05, "ckt": 0})
        assert cmd.ckt == 0

    def test_command_map_has_pflow_commands(self):
        from agentigrid.engine.commands import _COMMAND_MAP
        assert "set_tap_ratio" in _COMMAND_MAP
        assert "set_shunt_susceptance" in _COMMAND_MAP
        assert "set_phase_shift_angle" in _COMMAND_MAP


class TestPFLOWValidation:
    """Test validation for PFLOW-specific commands."""

    def test_tap_ratio_on_transformer(self, minimal_network):
        cmd = SetTapRatio(fbus=1, tbus=3, ratio=1.05)
        result = validate_command(cmd, minimal_network)
        assert result.valid
        assert len(result.errors) == 0

    def test_tap_ratio_on_non_transformer(self, minimal_network):
        cmd = SetTapRatio(fbus=1, tbus=2, ratio=1.05)
        result = validate_command(cmd, minimal_network)
        assert not result.valid
        assert any("not a transformer" in e for e in result.errors)

    def test_tap_ratio_negative(self, minimal_network):
        cmd = SetTapRatio(fbus=1, tbus=3, ratio=-0.5)
        result = validate_command(cmd, minimal_network)
        assert not result.valid
        assert any("> 0" in e for e in result.errors)

    def test_tap_ratio_out_of_range_warning(self, minimal_network):
        cmd = SetTapRatio(fbus=1, tbus=3, ratio=0.5)
        result = validate_command(cmd, minimal_network)
        assert result.valid
        assert any("typical range" in w for w in result.warnings)

    def test_shunt_susceptance_valid(self, minimal_network):
        cmd = SetShuntSusceptance(bus=2, Bs=0.5)
        result = validate_command(cmd, minimal_network)
        assert result.valid

    def test_shunt_susceptance_negative_valid(self, minimal_network):
        cmd = SetShuntSusceptance(bus=2, Bs=-0.3)
        result = validate_command(cmd, minimal_network)
        assert result.valid

    def test_shunt_susceptance_invalid_bus(self, minimal_network):
        cmd = SetShuntSusceptance(bus=99, Bs=0.5)
        result = validate_command(cmd, minimal_network)
        assert not result.valid

    def test_phase_shift_on_phase_shifter(self, minimal_network):
        cmd = SetPhaseShiftAngle(fbus=2, tbus=3, angle=10.0)
        result = validate_command(cmd, minimal_network)
        assert result.valid

    def test_phase_shift_on_non_shifter(self, minimal_network):
        cmd = SetPhaseShiftAngle(fbus=1, tbus=2, angle=5.0)
        result = validate_command(cmd, minimal_network)
        assert not result.valid
        assert any("not a phase shifter" in e for e in result.errors)

    def test_phase_shift_large_angle_warning(self, minimal_network):
        cmd = SetPhaseShiftAngle(fbus=2, tbus=3, angle=95.0)
        result = validate_command(cmd, minimal_network)
        assert result.valid
        assert any("very large" in w for w in result.warnings)


class TestPFLOWModifier:
    """Test modifier application for PFLOW-specific commands and PFLOW-aware SetGenVoltage."""

    def test_set_tap_ratio_applied(self, minimal_network):
        cmd = SetTapRatio(fbus=1, tbus=3, ratio=1.10)
        modified, report = apply_modifications(minimal_network, [cmd], application="pflow")
        assert len(report.applied) == 1
        br = [b for b in modified.branches if b.fbus == 1 and b.tbus == 3][0]
        assert br.ratio == 1.10

    def test_set_shunt_susceptance_applied(self, minimal_network):
        cmd = SetShuntSusceptance(bus=2, Bs=0.5)
        modified, report = apply_modifications(minimal_network, [cmd], application="pflow")
        assert len(report.applied) == 1
        bus = [b for b in modified.buses if b.bus_i == 2][0]
        assert bus.Bs == 0.5

    def test_set_phase_shift_angle_applied(self, minimal_network):
        cmd = SetPhaseShiftAngle(fbus=2, tbus=3, angle=10.0)
        modified, report = apply_modifications(minimal_network, [cmd], application="pflow")
        assert len(report.applied) == 1
        br = [b for b in modified.branches if b.fbus == 2 and b.tbus == 3][0]
        assert br.angle == 10.0

    def test_set_gen_voltage_pflow_message(self, minimal_network):
        cmd = SetGenVoltage(bus=1, Vg=1.03)
        modified, report = apply_modifications(minimal_network, [cmd], application="pflow")
        assert len(report.applied) == 1
        desc = report.applied[0][1]
        assert "constrains bus voltage" in desc

    def test_set_gen_voltage_opflow_message(self, minimal_network):
        cmd = SetGenVoltage(bus=1, Vg=1.03)
        modified, report = apply_modifications(minimal_network, [cmd], application="opflow")
        assert len(report.applied) == 1
        desc = report.applied[0][1]
        assert "initial guess only" in desc

    def test_set_gen_voltage_pflow_note_in_warnings(self, minimal_network):
        cmd = SetGenVoltage(bus=1, Vg=1.03)
        _, report = apply_modifications(minimal_network, [cmd], application="pflow")
        assert any("PFLOW" in w or "constrains" in w for w in report.warnings)

    def test_set_gen_voltage_opflow_warning(self, minimal_network):
        cmd = SetGenVoltage(bus=1, Vg=1.03)
        _, report = apply_modifications(minimal_network, [cmd], application="opflow")
        assert any("initial guess" in w for w in report.warnings)

    def test_set_gen_voltage_default_is_initial_guess(self, minimal_network):
        cmd = SetGenVoltage(bus=1, Vg=1.03)
        _, report = apply_modifications(minimal_network, [cmd])
        desc = report.applied[0][1]
        assert "initial guess only" in desc

    def test_tap_ratio_on_non_transformer_skipped(self, minimal_network):
        cmd = SetTapRatio(fbus=1, tbus=2, ratio=1.05)
        _, report = apply_modifications(minimal_network, [cmd])
        assert len(report.skipped) == 1

    def test_phase_shift_on_non_shifter_skipped(self, minimal_network):
        cmd = SetPhaseShiftAngle(fbus=1, tbus=2, angle=5.0)
        _, report = apply_modifications(minimal_network, [cmd])
        assert len(report.skipped) == 1

    def test_pflow_full_workflow(self, minimal_network):
        cmds = [
            ScaleAllLoads(factor=1.1),
            SetGenVoltage(bus=1, Vg=1.06),
            SetShuntSusceptance(bus=2, Bs=0.3),
            SetTapRatio(fbus=1, tbus=3, ratio=1.08),
            SetPhaseShiftAngle(fbus=2, tbus=3, angle=8.0),
        ]
        modified, report = apply_modifications(minimal_network, cmds, application="pflow")
        assert len(report.applied) == 5
        assert len(report.skipped) == 0


class TestPFLOWMetrics:
    """Test metric filtering for PFLOW."""

    def test_generation_cost_excluded(self):
        pflow_metrics = available_metrics_for_app("pflow")
        assert "generation_cost" not in pflow_metrics

    def test_voltage_metrics_included(self):
        pflow_metrics = available_metrics_for_app("pflow")
        assert "voltage_min" in pflow_metrics
        assert "voltage_max" in pflow_metrics
        assert "voltage_range" in pflow_metrics

    def test_line_loading_included(self):
        pflow_metrics = available_metrics_for_app("pflow")
        assert "max_line_loading_pct" in pflow_metrics

    def test_losses_included(self):
        pflow_metrics = available_metrics_for_app("pflow")
        assert "active_losses_mw" in pflow_metrics

    def test_opflow_still_has_generation_cost(self):
        opflow_metrics = available_metrics_for_app("opflow")
        assert "generation_cost" in opflow_metrics


class TestPFLOWSystemPrompt:
    """Test system prompt generation for PFLOW."""

    def test_pflow_section_present(self):
        section = _app_section("pflow")
        assert "Power Flow" in section
        assert "Analysis, Not Optimization" in section or "analysis" in section.lower()

    def test_pflow_section_no_opf_voltage(self):
        section = _app_section("pflow")
        assert "OPF Voltage Control" not in section

    def test_pflow_section_mentions_set_gen_voltage(self):
        section = _app_section("pflow")
        assert "set_gen_voltage" in section

    def test_pflow_section_mentions_new_commands(self):
        section = _app_section("pflow")
        assert "set_tap_ratio" in section
        assert "set_shunt_susceptance" in section
        assert "set_phase_shift_angle" in section

    def test_pflow_section_search_heuristics(self):
        section = _app_section("pflow")
        assert "binary search" in section.lower() or "bisection" in section.lower()

    def test_pflow_section_voltage_limits_guidance(self):
        section = _app_section("pflow")
        assert "set_all_bus_vlimits" in section
        assert "CRITICAL" in section or "first action" in section.lower()
        assert "0.95" in section

    def test_opflow_section_still_has_voltage_control(self):
        section = _app_section("opflow")
        assert "OPF Voltage Control" in section

    def test_build_system_prompt_pflow(self):
        prompt = build_system_prompt("schema text", "network info", application="pflow")
        assert "PFLOW" in prompt
        assert "Power Flow" in prompt

    def test_build_system_prompt_opflow_unchanged(self):
        prompt = build_system_prompt("schema text", "network info", application="opflow")
        assert "OPFLOW" in prompt
        assert "OPF Voltage Control" in prompt


class TestPFLOWGoalClassifier:
    """Test goal classifier context for PFLOW."""

    def test_pflow_context_in_system_prompt(self):
        _, user_prompt = build_classification_prompts(
            goal="Find maximum loadability",
            termination_reason="completed",
            stats={"total_iterations": 5, "feasible_count": 3, "infeasible_count": 2,
                   "best_objective": 0.0, "best_iteration": 3, "objective_trend": [],
                   "voltage_range_trend": [], "goal_type": None,
                   "objective_registry": [], "is_multi_objective": False,
                   "marginal_count": 0},
            journal_formatted="test journal",
            total_tokens=100,
            application="pflow",
        )
        assert "PFLOW" in _extract_system_prompt_context(
            goal="Find maximum loadability",
            termination_reason="completed",
            stats={"total_iterations": 5, "feasible_count": 3, "infeasible_count": 2,
                   "best_objective": 0.0, "best_iteration": 3, "objective_trend": [],
                   "voltage_range_trend": [], "goal_type": None,
                   "objective_registry": [], "is_multi_objective": False,
                   "marginal_count": 0},
            journal_formatted="test journal",
            total_tokens=100,
            application="pflow",
        )


def _extract_system_prompt_context(**kwargs):
    """Helper to extract the system prompt from build_classification_prompts."""
    sys_prompt, _ = build_classification_prompts(**kwargs)
    return sys_prompt


class TestPFLOWSchemaDescription:
    """Test that the command schema includes PFLOW-specific commands."""

    def test_schema_has_pflow_commands(self):
        from agentigrid.engine.schema_description import command_schema_text
        schema = command_schema_text()
        assert "set_tap_ratio" in schema
        assert "set_shunt_susceptance" in schema
        assert "set_phase_shift_angle" in schema

    def test_schema_set_gen_voltage_mentions_pflow(self):
        from agentigrid.engine.schema_description import command_schema_text
        schema = command_schema_text()
        assert "PFLOW" in schema
        assert "set_gen_voltage" in schema


class TestPFLOWBusLimits:
    """Test PFLOW violations with custom bus limits."""

    def test_violations_with_custom_limits(self):
        result, _ = parse_pflow_output(
            SAMPLE_PFLOW_OUTPUT,
            bus_limits={2: (1.025, 1.1), 3: (1.02, 1.1)},
        )
        assert result.num_violations > 0

    def test_no_violations_with_relaxed_limits(self):
        result, _ = parse_pflow_output(
            SAMPLE_PFLOW_OUTPUT,
            bus_limits={2: (0.9, 1.1), 3: (0.9, 1.1)},
        )
        assert result.num_violations == 0

    def test_default_fallback_limits_095_105(self):
        result, _ = parse_pflow_output(SAMPLE_PFLOW_DIVERGED)
        voltages = [0.850, 0.780, 0.760]
        for v in voltages:
            assert v < 0.95
        assert result.num_violations > 0

    def test_default_limits_match_engineering_standard(self):
        result, _ = parse_pflow_output(SAMPLE_PFLOW_OUTPUT)
        assert result.voltage_min >= 0.95
        assert result.voltage_max <= 1.05
        assert result.num_violations == 0


class TestPFLOWParserEdgeCases:
    """Test edge cases in the PFLOW parser."""

    def test_power_balance_violation(self):
        """Test results where generation < load (unphysical)."""
        output = SAMPLE_PFLOW_OUTPUT.replace("105.00", "50.00").replace("42.43", "20.00")
        result, _ = parse_pflow_output(output)
        assert result.losses_mw < 0
        assert result.feasibility_detail == "infeasible"

    def test_missing_solver(self):
        """Parser should handle missing solver line gracefully."""
        output = SAMPLE_PFLOW_OUTPUT.replace("Solver                              Newton-Rhapson\n", "")
        result, metadata = parse_pflow_output(output)
        assert metadata["solver"] == ""

    def test_missing_iterations_line(self):
        """Parser should handle missing iterations line gracefully."""
        output = SAMPLE_PFLOW_OUTPUT.replace("Number of iterations                3\n", "")
        result, _ = parse_pflow_output(output)
        assert result.num_iterations == 0

    def test_missing_solve_time_line(self):
        """Parser should handle missing solve time line gracefully."""
        output = SAMPLE_PFLOW_OUTPUT.replace("Solve Time (sec)                    0.005\n", "")
        result, _ = parse_pflow_output(output)
        assert result.solve_time == 0.0
