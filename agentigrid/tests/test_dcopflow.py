"""Tests for DCOPFLOW support (Phase 3, Step 3.1)."""

from __future__ import annotations

import pytest
from pathlib import Path

from agentigrid.parsers.opflow_parser import parse_opflow_output
from agentigrid.parsers.dcopflow_summary import dcopflow_results_summary
from agentigrid.parsers import results_summary_for_app, parse_simulation_result_for_app
from agentigrid.engine.metric_extractor import available_metrics, available_metrics_for_app
from agentigrid.engine.commands import SetGenVoltage, SetBusVLimits, SetAllBusVLimits, ScaleAllLoads
from agentigrid.engine.modifier import apply_modifications
from agentigrid.parsers.matpower_model import MATNetwork, Bus, Generator, Branch, GenCost


# ---------------------------------------------------------------------------
# Synthetic DCOPFLOW output — 3-bus system
# Same column layout as sample_opflow_output.txt but with:
#   Model=DCOPF, all Vm=1.000, all Qd=Qg=0.00
# ---------------------------------------------------------------------------

SAMPLE_DCOPFLOW_OUTPUT = """\


******************************************************************************
This program contains Ipopt, a library for large-scale nonlinear optimization.
 Ipopt is released as open source code under the Eclipse Public License (EPL).
         For more information visit https://github.com/coin-or/Ipopt
******************************************************************************

This is Ipopt version 3.14.20, running with linear solver ma27.

Number of nonzeros in equality constraint Jacobian...:       12
Number of nonzeros in inequality constraint Jacobian.:        4
Number of nonzeros in Lagrangian Hessian.............:        6

Total number of variables............................:        5
                     variables with only lower bounds:        0
                variables with lower and upper bounds:        4
                     variables with only upper bounds:        0
Total number of equality constraints.................:        4
Total number of inequality constraints...............:        4
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:        4

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  1.2500000e+03 1.20e+00 2.50e+01  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  1.2345000e+03 1.00e-04 1.00e-03  -2.5 1.50e-01    -  1.00e+00 1.00e+00f  1

Number of Iterations....: 2

                                   (scaled)                 (unscaled)
Objective...............:   1.2345000000000000e+03    1.2345000000000000e+03
Dual infeasibility......:   1.0000000000000000e-08    1.0000000000000000e-08
Constraint violation....:   1.0000000000000000e-08    1.0000000000000000e-08
Variable bound violation:   0.0000000000000000e+00    0.0000000000000000e+00
Complementarity.........:   1.0000000000000000e-08    1.0000000000000000e-08
Overall NLP error.......:   1.0000000000000000e-08    1.0000000000000000e-08


Number of objective function evaluations             = 2
Number of objective gradient evaluations             = 2
Number of equality constraint evaluations            = 2
Number of inequality constraint evaluations          = 2
Number of equality constraint Jacobian evaluations   = 2
Number of inequality constraint Jacobian evaluations = 2
Number of Lagrangian Hessian evaluations             = 1
Total seconds in IPOPT                               = 0.008

EXIT: Optimal Solution Found.
=============================================================
		Optimal Power Flow
=============================================================
Model                               DCOPF
Solver                              IPOPT
Objective                           MIN_GEN_COST
Initialization                      DCOPF
Gen. bus voltage mode               VARIABLE_WITHIN_BOUNDS
Load loss allowed                   NO
Power imbalance allowed             NO
Ignore line flow constraints        NO
Allow line flow violation           NO

Number of variables                 5
Number of equality constraints      4
Number of inequality constraints    4

Convergence status                  CONVERGED
Objective value                     1234.56

------------------------------------------------------------------------------------------------------
Bus        Pd      Pdloss Qd      Qdloss Vm      Va      mult_Pmis      mult_Qmis      Pslack         Qslack
------------------------------------------------------------------------------------------------------
1         0.00    0.00    0.00    0.00   1.000   0.000         2.50         0.00         0.00         0.00
2        80.00    0.00    0.00    0.00   1.000  -5.250         2.50         0.00         0.00         0.00
3        60.00    0.00    0.00    0.00   1.000  -8.100         2.50         0.00         0.00         0.00

------------------------------------------------------------------------------------------------------
From       To       Status     Sft      Stf     Slim     mult_Sf  mult_St
----------------------------------------------------------------------------------------
1          2          1       60.00    60.00   100.00     0.00     0.00
1          3          1       82.43    82.43   150.00     0.00     0.00

----------------------------------------------------------------------------------------
Gen      Status     Fuel     Pg       Qg       Pmin     Pmax     Qmin     Qmax
----------------------------------------------------------------------------------------
1          1        COAL    100.00     0.00    10.00   150.00   -50.00    50.00
2          1        COAL     42.43     0.00     5.00    80.00   -30.00    30.00
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dcopflow_result():
    """Parse the synthetic DCOPFLOW output."""
    return parse_opflow_output(SAMPLE_DCOPFLOW_OUTPUT)


@pytest.fixture
def minimal_network():
    """A minimal 3-bus MATNetwork for modifier tests."""
    buses = [
        Bus(bus_i=1, type=3, Pd=0.0, Qd=0.0, Gs=0, Bs=0, area=1,
            Vm=1.0, Va=0.0, baseKV=138.0, zone=1, Vmax=1.1, Vmin=0.9),
        Bus(bus_i=2, type=1, Pd=80.0, Qd=0.0, Gs=0, Bs=0, area=1,
            Vm=1.0, Va=0.0, baseKV=138.0, zone=1, Vmax=1.1, Vmin=0.9),
        Bus(bus_i=3, type=1, Pd=60.0, Qd=0.0, Gs=0, Bs=0, area=1,
            Vm=1.0, Va=0.0, baseKV=138.0, zone=1, Vmax=1.1, Vmin=0.9),
    ]
    generators = [
        Generator(bus=1, Pg=100.0, Qg=0.0, Qmax=50.0, Qmin=-50.0,
                  Vg=1.0, mBase=100.0, status=1, Pmax=150.0, Pmin=10.0),
        Generator(bus=2, Pg=42.43, Qg=0.0, Qmax=30.0, Qmin=-30.0,
                  Vg=1.0, mBase=100.0, status=1, Pmax=80.0, Pmin=5.0),
    ]
    branches = [
        Branch(fbus=1, tbus=2, r=0.01, x=0.05, b=0.0, rateA=100.0,
               rateB=100.0, rateC=100.0, ratio=0.0, angle=0.0, status=1,
               angmin=-360.0, angmax=360.0),
        Branch(fbus=1, tbus=3, r=0.01, x=0.04, b=0.0, rateA=150.0,
               rateB=150.0, rateC=150.0, ratio=0.0, angle=0.0, status=1,
               angmin=-360.0, angmax=360.0),
    ]
    return MATNetwork(
        casename="test_dc",
        version="2",
        baseMVA=100.0,
        buses=buses,
        generators=generators,
        branches=branches,
        gencost=[],
        header_comments="",
    )


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------

class TestDCOPFLOWParser:
    """Verify the existing OPFLOW parser handles DCOPFLOW output."""

    def test_parse_succeeds(self, dcopflow_result):
        """parse_opflow_output should parse DCOPFLOW output without error."""
        assert dcopflow_result is not None

    def test_model_is_dcopf(self, dcopflow_result):
        """Model field should be 'DCOPF'."""
        assert dcopflow_result.model == "DCOPF"

    def test_all_voltages_unity(self, dcopflow_result):
        """All Vm values should be 1.0 in DC output."""
        for bus in dcopflow_result.buses:
            assert bus.Vm == pytest.approx(1.0, abs=0.001)

    def test_all_reactive_zero(self, dcopflow_result):
        """All Qg values should be 0.0 in DC output."""
        for gen in dcopflow_result.generators:
            assert gen.Qg == pytest.approx(0.0, abs=0.001)

    def test_objective_value(self, dcopflow_result):
        """Objective value should be parsed correctly."""
        assert dcopflow_result.objective_value == pytest.approx(1234.56, abs=0.01)

    def test_convergence_status(self, dcopflow_result):
        """Convergence status should be CONVERGED."""
        assert dcopflow_result.convergence_status == "CONVERGED"
        assert dcopflow_result.converged is True

    def test_bus_count(self, dcopflow_result):
        """Should have 3 buses."""
        assert len(dcopflow_result.buses) == 3

    def test_branch_count(self, dcopflow_result):
        """Should have 2 branches."""
        assert len(dcopflow_result.branches) == 2

    def test_generator_count(self, dcopflow_result):
        """Should have 2 generators."""
        assert len(dcopflow_result.generators) == 2


class TestDCOPFLOWSummary:
    """Test the DCOPFLOW-specific results summary."""

    def test_summary_header(self, dcopflow_result):
        """Summary should include 'DCOPFLOW Results'."""
        summary = dcopflow_results_summary(dcopflow_result)
        assert "DCOPFLOW Results" in summary

    def test_summary_has_dc_note(self, dcopflow_result):
        """Summary should note DC approximation."""
        summary = dcopflow_results_summary(dcopflow_result)
        assert "DC approximation" in summary or "DC Approximation" in summary

    def test_summary_has_angle_profile(self, dcopflow_result):
        """Summary should include phase angle profile."""
        summary = dcopflow_results_summary(dcopflow_result)
        assert "Phase angle" in summary or "phase angle" in summary

    def test_summary_no_voltage_magnitude(self, dcopflow_result):
        """Summary should NOT contain 'Voltage profile' (AC-style)."""
        summary = dcopflow_results_summary(dcopflow_result)
        assert "Voltage profile:" not in summary

    def test_summary_has_generation_info(self, dcopflow_result):
        """Summary should include generation/load info."""
        summary = dcopflow_results_summary(dcopflow_result)
        assert "Generation:" in summary

    def test_summary_has_line_loading(self, dcopflow_result):
        """Summary should include line loading section."""
        summary = dcopflow_results_summary(dcopflow_result)
        assert "Most loaded lines:" in summary

    def test_summary_no_reactive(self, dcopflow_result):
        """Summary should NOT contain reactive power line (Reactive: Gen ...)."""
        summary = dcopflow_results_summary(dcopflow_result)
        assert "Reactive: Gen" not in summary


class TestDCOPFLOWDispatch:
    """Test the application-aware dispatch functions."""

    def test_dispatch_dcopflow(self, dcopflow_result):
        """results_summary_for_app('dcopflow') should use dcopflow summary."""
        summary = results_summary_for_app(dcopflow_result, "dcopflow")
        assert "DCOPFLOW Results" in summary
        assert "Phase angle" in summary

    def test_dispatch_opflow(self, dcopflow_result):
        """results_summary_for_app('opflow') should use standard summary."""
        summary = results_summary_for_app(dcopflow_result, "opflow")
        assert "OPFLOW Results" in summary
        assert "Voltage profile:" in summary

    def test_dispatch_unknown_falls_back(self, dcopflow_result):
        """Unknown app should fall back to OPFLOW summary without raising."""
        summary = results_summary_for_app(dcopflow_result, "unknown_app")
        # Falls back to OPFLOW summary
        assert "OPFLOW Results" in summary


class TestDCOPFLOWMetrics:
    """Test metric filtering for DCOPFLOW."""

    def test_excluded_metrics(self):
        """Voltage-related metrics should be excluded for dcopflow."""
        dc_metrics = available_metrics_for_app("dcopflow")
        assert "voltage_min" not in dc_metrics
        assert "voltage_max" not in dc_metrics
        assert "voltage_deviation" not in dc_metrics
        assert "voltage_range" not in dc_metrics
        assert "total_reactive_gen_mvar" not in dc_metrics

    def test_phase_angle_range_included(self):
        """phase_angle_range should be in dcopflow metrics."""
        dc_metrics = available_metrics_for_app("dcopflow")
        assert "phase_angle_range" in dc_metrics

    def test_opflow_has_all_metrics(self):
        """OPFLOW should still have all metrics including voltage ones."""
        opflow_metrics = available_metrics_for_app("opflow")
        assert "voltage_min" in opflow_metrics
        assert "voltage_max" in opflow_metrics
        assert "total_reactive_gen_mvar" in opflow_metrics
        assert "phase_angle_range" in opflow_metrics

    def test_dcopflow_has_non_voltage_metrics(self):
        """DCOPFLOW should still have non-voltage metrics."""
        dc_metrics = available_metrics_for_app("dcopflow")
        assert "generation_cost" in dc_metrics
        assert "max_line_loading_pct" in dc_metrics
        assert "total_generation_mw" in dc_metrics
        assert "active_losses_mw" in dc_metrics

    def test_available_metrics_unchanged(self):
        """available_metrics() (no app) should still return all metrics."""
        all_metrics = available_metrics()
        assert "voltage_min" in all_metrics
        assert "phase_angle_range" in all_metrics


class TestDCOPFLOWModifier:
    """Test that voltage commands are skipped for DCOPFLOW."""

    def test_set_gen_voltage_skipped(self, minimal_network):
        """set_gen_voltage should be skipped with warning for dcopflow."""
        from agentigrid.engine.commands import SetGenVoltage
        cmd = SetGenVoltage(bus=1, Vg=1.05)
        _, report = apply_modifications(minimal_network, [cmd], application="dcopflow")
        assert len(report.applied) == 0
        assert any("set_gen_voltage" in w for w in report.warnings)
        assert any("DCOPFLOW" in w for w in report.warnings)

    def test_set_bus_vlimits_skipped(self, minimal_network):
        """set_bus_vlimits should be skipped with warning for dcopflow."""
        from agentigrid.engine.commands import SetBusVLimits
        cmd = SetBusVLimits(bus=1, Vmin=0.95, Vmax=1.05)
        _, report = apply_modifications(minimal_network, [cmd], application="dcopflow")
        assert len(report.applied) == 0
        assert any("set_bus_vlimits" in w for w in report.warnings)

    def test_set_all_bus_vlimits_skipped(self, minimal_network):
        """set_all_bus_vlimits should be skipped with warning for dcopflow."""
        from agentigrid.engine.commands import SetAllBusVLimits
        cmd = SetAllBusVLimits(Vmin=0.95, Vmax=1.05)
        _, report = apply_modifications(minimal_network, [cmd], application="dcopflow")
        assert len(report.applied) == 0
        assert any("set_all_bus_vlimits" in w for w in report.warnings)

    def test_load_commands_still_work(self, minimal_network):
        """Non-voltage commands should work normally for dcopflow."""
        from agentigrid.engine.commands import ScaleAllLoads
        cmd = ScaleAllLoads(factor=1.1)
        modified, report = apply_modifications(minimal_network, [cmd], application="dcopflow")
        assert len(report.applied) == 1
        # Loads should be scaled
        for orig_bus, new_bus in zip(minimal_network.buses, modified.buses):
            assert new_bus.Pd == pytest.approx(orig_bus.Pd * 1.1, abs=0.001)

    def test_voltage_cmds_applied_for_opflow(self, minimal_network):
        """Voltage commands should still be applied for opflow (not skipped)."""
        from agentigrid.engine.commands import SetAllBusVLimits
        cmd = SetAllBusVLimits(Vmin=0.95, Vmax=1.05)
        _, report = apply_modifications(minimal_network, [cmd], application="opflow")
        assert len(report.applied) == 1
