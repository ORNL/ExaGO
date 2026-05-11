"""Tests for TCOPFLOW support (Phase 3, Step 3.3)."""

from __future__ import annotations

import csv
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from agentigrid.parsers.tcopflow_parser import (
    parse_tcopflow_output,
    parse_tcopflow_simulation_result,
    parse_tcopflow_period_files,
    parse_tcopflow_metadata,
)
from agentigrid.parsers.tcopflow_summary import tcopflow_results_summary
from agentigrid.parsers import (
    results_summary_for_app,
    parse_simulation_result_for_app,
    parse_tcopflow_metadata as dispatch_tcopflow_metadata,
)
from agentigrid.parsers.opflow_parser import parse_opflow_output
from agentigrid.config import SearchConfig
from agentigrid.engine.metric_extractor import available_metrics, available_metrics_for_app
from agentigrid.engine.commands import ScaleLoadProfile, parse_command
from agentigrid.engine.modifier import apply_modifications, scale_load_profile_csv, ModificationReport
from agentigrid.parsers.matpower_model import MATNetwork, Bus, Generator, Branch, GenCost
from agentigrid.engine.journal import SearchJournal, JournalEntry
from agentigrid.prompts.system_prompt import build_system_prompt
from agentigrid.engine.schema_description import command_schema_text
from agentigrid.engine.goal_classifier import build_classification_prompts


# ---------------------------------------------------------------------------
# Synthetic TCOPFLOW output — 3-bus system
# Same bus/branch/gen table format as OPFLOW but with the TCOPFLOW header
# and multi-period metadata. AC OPF: realistic Vm, non-zero Qg/Qd.
# ---------------------------------------------------------------------------

SAMPLE_TCOPFLOW_OUTPUT = """\


******************************************************************************
This program contains Ipopt, a library for large-scale nonlinear optimization.
 Ipopt is released as open source code under the Eclipse Public License (EPL).
         For more information visit https://github.com/coin-or/Ipopt
******************************************************************************

This is Ipopt version 3.14.20, running with linear solver ma27.

Number of Iterations....: 10

Total seconds in IPOPT                               = 0.035

EXIT: Optimal Solution Found.
=============================================================
	Multi-Period Optimal Power Flow
=============================================================
OPFLOW Model                        POWER_BALANCE_POLAR
Solver                              IPOPT
Duration (minutes)                  60.00
Time-step (minutes)                 15.00 
Number of steps                     4
Active power demand profile         /data/case9_load_P.csv
Rective power demand profile        /data/case9_load_Q.csv
Wind generation profile             NOT SET
Load loss allowed                   NO
Power imbalance allowed             NO
Ignore line flow constraints        NO

Number of variables                 120
Number of equality constraints      90
Number of inequality constraints    90
Number of coupling constraints      6

Convergence status                  CONVERGED
Objective value                     15234.56

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
1          1        COAL    105.00    18.50    10.00   150.00   -50.00    50.00
2          1        GAS      42.43    12.30     5.00    80.00   -30.00    30.00
"""

SAMPLE_TCOPFLOW_DID_NOT_CONVERGE = """\
EXIT: Maximum Number of Iterations Exceeded.
=============================================================
	Multi-Period Optimal Power Flow
=============================================================
OPFLOW Model                        POWER_BALANCE_POLAR
Solver                              IPOPT
Duration (minutes)                  60.00
Time-step (minutes)                 15.00 
Number of steps                     4
Active power demand profile         /data/case9_load_P.csv
Rective power demand profile        /data/case9_load_Q.csv
Wind generation profile             NOT SET
Load loss allowed                   NO
Power imbalance allowed             NO
Ignore line flow constraints        NO

Number of variables                 120
Number of equality constraints      90
Number of inequality constraints    90
Number of coupling constraints      6

Convergence status                  DID NOT CONVERGE
Objective value                     0.00

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
1          1        COAL    105.00    18.50    10.00   150.00   -50.00    50.00
2          1        GAS      42.43    12.30     5.00    80.00   -30.00    30.00
"""

SAMPLE_TCOPFLOW_WITH_WIND = SAMPLE_TCOPFLOW_OUTPUT.replace(
    "Wind generation profile             NOT SET",
    "Wind generation profile             /data/case9_wind.csv",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tcopflow_result_and_meta():
    return parse_tcopflow_output(SAMPLE_TCOPFLOW_OUTPUT)


@pytest.fixture
def tcopflow_result(tcopflow_result_and_meta):
    return tcopflow_result_and_meta[0]


@pytest.fixture
def tcopflow_meta(tcopflow_result_and_meta):
    return tcopflow_result_and_meta[1]


@pytest.fixture
def mock_sim_result():
    sr = MagicMock()
    sr.success = True
    sr.stdout = SAMPLE_TCOPFLOW_OUTPUT
    sr.workdir = Path("/tmp/nonexistent")
    return sr


@pytest.fixture
def minimal_network():
    buses = [
        Bus(bus_i=1, type=3, Pd=0.0, Qd=0.0, Gs=0, Bs=0, area=1,
            Vm=1.0, Va=0.0, baseKV=138.0, zone=1, Vmax=1.1, Vmin=0.9),
        Bus(bus_i=2, type=1, Pd=80.0, Qd=20.0, Gs=0, Bs=0, area=1,
            Vm=1.0, Va=0.0, baseKV=138.0, zone=1, Vmax=1.1, Vmin=0.9),
        Bus(bus_i=3, type=1, Pd=60.0, Qd=15.0, Gs=0, Bs=0, area=1,
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
    ]
    return MATNetwork(
        casename="test_tcopflow",
        version="2",
        baseMVA=100.0,
        buses=buses,
        generators=generators,
        branches=branches,
        gencost=[],
        header_comments="",
    )


@pytest.fixture
def sample_period_data():
    return [
        {"period": 0, "total_gen_mw": 475.0, "total_load_mw": 265.0,
         "total_gen_mvar": 30.0, "total_load_mvar": 35.0,
         "voltage_min": 1.010, "voltage_max": 1.050,
         "max_line_loading_pct": 55.3, "losses_mw": 10.0,
         "converged": True, "objective": 3800.0,
         "num_buses": 9, "num_gens_on": 3},
        {"period": 1, "total_gen_mw": 478.0, "total_load_mw": 270.0,
         "total_gen_mvar": 31.0, "total_load_mvar": 36.0,
         "voltage_min": 1.008, "voltage_max": 1.049,
         "max_line_loading_pct": 58.1, "losses_mw": 11.0,
         "converged": True, "objective": 3850.0,
         "num_buses": 9, "num_gens_on": 3},
        {"period": 2, "total_gen_mw": 482.0, "total_load_mw": 278.0,
         "total_gen_mvar": 32.0, "total_load_mvar": 37.0,
         "voltage_min": 1.005, "voltage_max": 1.048,
         "max_line_loading_pct": 62.7, "losses_mw": 12.0,
         "converged": True, "objective": 3900.0,
         "num_buses": 9, "num_gens_on": 3},
    ]


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------

class TestTCOPFLOWParser:
    def test_parse_succeeds(self, tcopflow_result_and_meta):
        result, meta = tcopflow_result_and_meta
        assert result is not None
        assert meta is not None

    def test_duration(self, tcopflow_meta):
        assert tcopflow_meta["duration_min"] == pytest.approx(60.0, abs=0.1)

    def test_dT(self, tcopflow_meta):
        assert tcopflow_meta["dT_min"] == pytest.approx(15.0, abs=0.1)

    def test_num_steps(self, tcopflow_meta):
        assert tcopflow_meta["num_steps"] == 4

    def test_pload_profile(self, tcopflow_meta):
        assert "case9_load_P.csv" in tcopflow_meta["pload_profile"]

    def test_qload_profile(self, tcopflow_meta):
        assert "case9_load_Q.csv" in tcopflow_meta["qload_profile"]

    def test_wind_profile_not_set(self, tcopflow_meta):
        assert tcopflow_meta["wind_profile"] == "NOT SET"

    def test_wind_profile_set(self):
        result, meta = parse_tcopflow_output(SAMPLE_TCOPFLOW_WITH_WIND)
        assert "case9_wind.csv" in meta["wind_profile"]
        assert meta["wind_profile"] != "NOT SET"

    def test_load_loss_not_allowed(self, tcopflow_meta):
        assert tcopflow_meta["load_loss_allowed"] is False

    def test_power_imbalance_not_allowed(self, tcopflow_meta):
        assert tcopflow_meta["power_imbalance_allowed"] is False

    def test_coupling_constraints(self, tcopflow_meta):
        assert tcopflow_meta["num_coupling_constraints"] == 6

    def test_convergence_status(self, tcopflow_result, tcopflow_meta):
        assert tcopflow_meta["convergence_status"] == "CONVERGED"
        assert tcopflow_result.converged is True

    def test_objective_value(self, tcopflow_result):
        assert tcopflow_result.objective_value == pytest.approx(15234.56, abs=0.01)

    def test_feasibility_detail(self, tcopflow_result):
        assert tcopflow_result.feasibility_detail == "feasible"

    def test_bus_count(self, tcopflow_result):
        assert len(tcopflow_result.buses) == 3

    def test_branch_count(self, tcopflow_result):
        assert len(tcopflow_result.branches) == 2

    def test_gen_count(self, tcopflow_result):
        assert len(tcopflow_result.generators) == 2

    def test_rejects_non_tcopflow(self):
        with pytest.raises(ValueError, match="TCOPFLOW"):
            parse_tcopflow_output("This is OPFLOW output")

    def test_rejects_opflow(self):
        with pytest.raises(ValueError, match="TCOPFLOW"):
            parse_tcopflow_output("Optimal Power Flow\nConvergence status CONVERGED")

    def test_voltages_non_unity(self, tcopflow_result):
        vms = [b.Vm for b in tcopflow_result.buses]
        assert any(abs(vm - 1.0) > 0.01 for vm in vms)

    def test_reactive_non_zero(self, tcopflow_result):
        qgs = [g.Qg for g in tcopflow_result.generators]
        assert any(abs(qg) > 0.1 for qg in qgs)


class TestTCOPFLOWDidNotConverge:
    def test_did_not_converge(self):
        result, meta = parse_tcopflow_output(SAMPLE_TCOPFLOW_DID_NOT_CONVERGE)
        assert meta["convergence_status"] == "DID NOT CONVERGE"
        assert result.converged is False
        assert result.feasibility_detail == "marginal"

    def test_marginal_with_exit_status(self):
        result, meta = parse_tcopflow_output(SAMPLE_TCOPFLOW_DID_NOT_CONVERGE)
        assert result.ipopt_exit_status == "Maximum Number of Iterations Exceeded."
        assert result.feasibility_detail == "marginal"

    def test_infeasible_with_negative_losses(self):
        output = SAMPLE_TCOPFLOW_OUTPUT.replace(
            "Bus        Pd      Pdloss Qd      Qdloss Vm      Va",
            "Bus        Pd      Pdloss Qd      Qdloss Vm      Va",
        )
        output = output.replace(
            "1         0.00    0.00    0.00    0.00   1.050",
            "1         0.00    0.00    0.00    0.00   0.050",
        ).replace("2        80.00", "2       300.00").replace("3        60.00", "3       400.00")
        result, meta = parse_tcopflow_output(output)
        assert result.feasibility_detail == "infeasible"

    def test_near_boundary_voltage_marginal(self):
        """Non-converged with 'Infeasible Problem Detected' but voltage near limit → marginal."""
        output = SAMPLE_TCOPFLOW_DID_NOT_CONVERGE.replace(
            "Maximum Number of Iterations Exceeded.",
            "Infeasible Problem Detected.",
        )
        bus_limits = {1: (0.95, 1.05), 2: (0.95, 1.05), 3: (0.95, 1.05)}
        result, meta = parse_tcopflow_output(output, bus_limits=bus_limits)
        assert result.converged is False
        assert result.feasibility_detail == "marginal"

    def test_near_boundary_line_loading_marginal(self):
        """Non-converged with line loading near 100% -> marginal."""
        output = (
            "EXIT: Infeasible Problem Detected.\n"
            "Multi-Period Optimal Power Flow\n"
            "Convergence status                  DID NOT CONVERGE\n"
            "Objective value                     0.00\n"
            "Duration (minutes)                  60.00\n"
            "Time-step (minutes)                 15.00\n"
            "Number of steps                     2\n"
            "Active power demand profile         /data/case9_load_P.csv\n"
            "Rective power demand profile        /data/case9_load_Q.csv\n"
            "------------------------------------------------------------------------------------------------------\n"
            "Bus        Pd      Pdloss Qd      Qdloss Vm      Va      mult_Pmis      mult_Qmis      Pslack         Qslack\n"
            "------------------------------------------------------------------------------------------------------\n"
            "1         0.00    0.00    0.00    0.00   1.050   0.000         0.00         0.00         0.00         0.00\n"
            "2        80.00    0.00   20.00    0.00   1.020  -4.100         0.00         0.00         0.00         0.00\n"
            "------------------------------------------------------------------------------------------------------\n"
            "From       To       Status     Sft      Stf     Slim     mult_Sf  mult_St\n"
            "----------------------------------------------------------------------------------------\n"
            "1          2          1       97.00    97.00   100.00     0.00     0.00\n"
            "1          3          1       60.00    60.00   150.00     0.00     0.00\n"
            "----------------------------------------------------------------------------------------\n"
            "Gen      Status     Fuel     Pg       Qg       Pmin     Pmax     Qmin     Qmax\n"
            "----------------------------------------------------------------------------------------\n"
            "1          1        COAL    105.00    18.50    10.00   150.00   -50.00    50.00\n"
        )
        result, meta = parse_tcopflow_output(output)
        assert result.converged is False
        assert result.feasibility_detail == "marginal"

    def test_far_from_boundary_infeasible(self):
        """Non-converged with metrics far from limits -> infeasible."""
        output = (
            "EXIT: Infeasible Problem Detected.\n"
            "Multi-Period Optimal Power Flow\n"
            "Convergence status                  DID NOT CONVERGE\n"
            "Objective value                     0.00\n"
            "Duration (minutes)                  60.00\n"
            "Time-step (minutes)                 15.00\n"
            "Number of steps                     2\n"
            "Active power demand profile         /data/case9_load_P.csv\n"
            "Rective power demand profile        /data/case9_load_Q.csv\n"
            "------------------------------------------------------------------------------------------------------\n"
            "Bus        Pd      Pdloss Qd      Qdloss Vm      Va      mult_Pmis      mult_Qmis      Pslack         Qslack\n"
            "------------------------------------------------------------------------------------------------------\n"
            "1         0.00    0.00    0.00    0.00   0.800   0.000         0.00         0.00         0.00         0.00\n"
            "2        80.00    0.00   20.00    0.00   0.750  -4.100         0.00         0.00         0.00         0.00\n"
            "------------------------------------------------------------------------------------------------------\n"
            "From       To       Status     Sft      Stf     Slim     mult_Sf  mult_St\n"
            "----------------------------------------------------------------------------------------\n"
            "1          2          1       55.00    55.00   200.00     0.00     0.00\n"
            "1          3          1       40.00    40.00   300.00     0.00     0.00\n"
            "----------------------------------------------------------------------------------------\n"
            "Gen      Status     Fuel     Pg       Qg       Pmin     Pmax     Qmin     Qmax\n"
            "----------------------------------------------------------------------------------------\n"
            "1          1        COAL    105.00    18.50    10.00   150.00   -50.00    50.00\n"
        )
        result, meta = parse_tcopflow_output(output)
        assert result.converged is False
        assert result.feasibility_detail == "infeasible"


class TestTCOPFLOWSummary:
    def test_header_mentions_multi_period(self, tcopflow_result):
        summary = tcopflow_results_summary(tcopflow_result, num_steps=4, duration_min=60.0, dT_min=15.0)
        assert "Multi-Period" in summary

    def test_time_horizon_shown(self, tcopflow_result):
        summary = tcopflow_results_summary(tcopflow_result, num_steps=4, duration_min=60.0, dT_min=15.0)
        assert "4 steps" in summary
        assert "60 min" in summary

    def test_coupling_shown(self, tcopflow_result):
        summary = tcopflow_results_summary(tcopflow_result, num_steps=4, is_coupling=True)
        assert "coupling enabled" in summary.lower() or "ramp constraints" in summary.lower()

    def test_has_voltage_profile(self, tcopflow_result):
        summary = tcopflow_results_summary(tcopflow_result)
        assert "Period-0 voltage" in summary

    def test_has_generation_info(self, tcopflow_result):
        summary = tcopflow_results_summary(tcopflow_result)
        assert "Generation:" in summary
        assert "Load:" in summary

    def test_has_line_loading(self, tcopflow_result):
        summary = tcopflow_results_summary(tcopflow_result)
        assert "loaded lines" in summary.lower()

    def test_no_period_data(self, tcopflow_result):
        summary = tcopflow_results_summary(tcopflow_result, num_steps=4)
        assert "TCOPFLOW Results" in summary
        assert "Per-period summary" not in summary

    def test_with_period_data(self, tcopflow_result, sample_period_data):
        summary = tcopflow_results_summary(
            tcopflow_result, num_steps=4, duration_min=60.0, dT_min=15.0,
            period_data=sample_period_data,
        )
        assert "Per-period summary" in summary
        assert "265.0" in summary
        assert "Aggregated metrics" in summary


class TestTCOPFLOWDispatch:
    def test_dispatch_tcopflow_summary(self, tcopflow_result):
        summary = results_summary_for_app(tcopflow_result, "tcopflow", num_steps=4)
        assert "TCOPFLOW Results" in summary
        assert "Multi-Period" in summary

    def test_dispatch_tcopflow_with_period_data(self, tcopflow_result, sample_period_data):
        summary = results_summary_for_app(
            tcopflow_result, "tcopflow", num_steps=4,
            duration_min=60.0, dT_min=15.0, period_data=sample_period_data,
        )
        assert "Per-period summary" in summary

    def test_parse_dispatch_tcopflow(self, mock_sim_result):
        result = parse_simulation_result_for_app(mock_sim_result, "tcopflow")
        assert result is not None
        assert result.objective_value == pytest.approx(15234.56, abs=0.01)

    def test_parse_dispatch_tcopflow_metadata(self, mock_sim_result):
        meta = dispatch_tcopflow_metadata(mock_sim_result)
        assert meta is not None
        assert meta["num_steps"] == 4
        assert meta["duration_min"] == pytest.approx(60.0, abs=0.1)
        assert meta["dT_min"] == pytest.approx(15.0, abs=0.1)

    def test_parse_tcopflow_simulation_result_failed(self):
        sr = MagicMock()
        sr.success = False
        result = parse_tcopflow_simulation_result(sr)
        assert result is None


class TestTCOPFLOWConfig:
    def test_tcopflow_profile_fields_in_config(self):
        cfg = SearchConfig(
            max_iterations=10,
            default_mode="accumulative",
            base_case=None,
            gic_file=None,
            application="tcopflow",
            pload_profile=None,
            qload_profile=None,
        )
        assert cfg.pload_profile is None
        assert cfg.qload_profile is None

    def test_profile_paths_stored(self, tmp_path):
        p_file = tmp_path / "load_P.csv"
        q_file = tmp_path / "load_Q.csv"
        p_file.touch()
        q_file.touch()
        cfg = SearchConfig(
            max_iterations=10,
            default_mode="accumulative",
            base_case=None,
            gic_file=None,
            application="tcopflow",
            pload_profile=p_file,
            qload_profile=q_file,
        )
        assert cfg.pload_profile == p_file
        assert cfg.qload_profile == q_file

    def test_tcopflow_params_defaults(self):
        cfg = SearchConfig(
            max_iterations=10,
            default_mode="accumulative",
            base_case=None,
            gic_file=None,
            application="tcopflow",
        )
        assert cfg.tcopflow_duration == 1.0
        assert cfg.tcopflow_dT == 60.0
        assert cfg.tcopflow_iscoupling == 1

    def test_tcopflow_params_stored(self):
        cfg = SearchConfig(
            max_iterations=10,
            default_mode="accumulative",
            base_case=None,
            gic_file=None,
            application="tcopflow",
            tcopflow_duration=2.0,
            tcopflow_dT=30.0,
            tcopflow_iscoupling=0,
        )
        assert cfg.tcopflow_duration == 2.0
        assert cfg.tcopflow_dT == 30.0
        assert cfg.tcopflow_iscoupling == 0


class TestTCOPFLOWMetrics:
    def test_tcopflow_has_all_metrics(self):
        tc_metrics = available_metrics_for_app("tcopflow")
        all_metrics = available_metrics()
        assert set(tc_metrics) == set(all_metrics)

    def test_tcopflow_has_voltage_metrics(self):
        tc_metrics = available_metrics_for_app("tcopflow")
        assert "voltage_min" in tc_metrics
        assert "voltage_max" in tc_metrics

    def test_tcopflow_has_reactive_metrics(self):
        tc_metrics = available_metrics_for_app("tcopflow")
        assert "total_reactive_gen_mvar" in tc_metrics


class TestTCOPFLOWModifier:
    def test_scale_load_profile_skipped_for_opflow(self, minimal_network):
        cmd = ScaleLoadProfile(factor=1.2)
        _, report = apply_modifications(minimal_network, [cmd], application="opflow")
        assert len(report.applied) == 0
        assert any("TCOPFLOW" in w for w in report.warnings)

    def test_scale_load_profile_requires_profiles(self, minimal_network):
        cmd = ScaleLoadProfile(factor=1.2)
        _, report = apply_modifications(
            minimal_network, [cmd], application="tcopflow",
            pload_profile=None, qload_profile=None,
        )
        assert len(report.applied) == 0
        assert len(report.skipped) == 1

    def test_scale_load_profile_csv(self, tmp_path):
        csv_path = tmp_path / "test_load_P.csv"
        csv_path.write_text("Timestamp,5,6,8\n2024-01-01,100.0,90.0,80.0\n")
        out_dir = tmp_path / "scaled"
        out_path = scale_load_profile_csv(csv_path, 1.1, out_dir)
        assert out_path.exists()
        lines = out_path.read_text().strip().split("\n")
        assert lines[0] == "Timestamp,5,6,8"
        vals = lines[1].split(",")
        assert float(vals[1]) == pytest.approx(110.0, abs=0.01)
        assert float(vals[2]) == pytest.approx(99.0, abs=0.01)
        assert float(vals[3]) == pytest.approx(88.0, abs=0.01)

    def test_scale_load_profile_csv_preserves_timestamp(self, tmp_path):
        csv_path = tmp_path / "test_load_Q.csv"
        csv_path.write_text("Timestamp,5,6\n2024-01-01 00:00,50.0,60.0\n")
        out_dir = tmp_path / "scaled"
        out_path = scale_load_profile_csv(csv_path, 0.9, out_dir)
        lines = out_path.read_text().strip().split("\n")
        assert lines[1].split(",")[0] == "2024-01-01 00:00"

    def test_scale_load_profile_modifies_both(self, tmp_path):
        p_path = tmp_path / "load_P.csv"
        q_path = tmp_path / "load_Q.csv"
        p_path.write_text("Timestamp,5,6\nT0,100.0,90.0\n")
        q_path.write_text("Timestamp,5,6\nT0,50.0,45.0\n")
        net = MATNetwork(
            casename="test", version="2", baseMVA=100.0,
            buses=[], generators=[], branches=[], gencost=[],
            header_comments="",
        )
        cmd = ScaleLoadProfile(factor=1.5)
        out_dir = tmp_path / "profiles"
        _, report = apply_modifications(
            net, [cmd], application="tcopflow",
            pload_profile=p_path, qload_profile=q_path,
            profile_output_dir=out_dir,
        )
        assert len(report.applied) == 1
        assert "pload_profile" in report.profile_paths
        assert "qload_profile" in report.profile_paths

    def test_scale_load_profile_cumulative(self, tmp_path):
        p_path = tmp_path / "load_P.csv"
        p_path.write_text("T,B1\n0,100.0\n")
        net = MATNetwork(
            casename="test", version="2", baseMVA=100.0,
            buses=[], generators=[], branches=[], gencost=[],
            header_comments="",
        )
        out_dir = tmp_path / "profiles_iter1"
        cmd1 = ScaleLoadProfile(factor=1.2)
        _, report1 = apply_modifications(
            net, [cmd1], application="tcopflow",
            pload_profile=p_path, qload_profile=p_path,
            profile_output_dir=out_dir,
        )
        assert "pload_profile" in report1.profile_paths
        scaled_p = report1.profile_paths["pload_profile"]
        lines = scaled_p.read_text().strip().split("\n")
        assert float(lines[1].split(",")[1]) == pytest.approx(120.0, abs=0.1)


class TestTCOPFLOWCommandParsing:
    def test_parse_scale_load_profile(self):
        cmd = parse_command({"action": "scale_load_profile", "factor": 1.2})
        assert isinstance(cmd, ScaleLoadProfile)
        assert cmd.factor == pytest.approx(1.2, abs=0.01)

    def test_parse_scale_load_profile_missing_factor(self):
        with pytest.raises(ValueError, match="factor"):
            parse_command({"action": "scale_load_profile"})

    def test_command_schema_includes_scale_load_profile(self):
        schema = command_schema_text()
        assert "scale_load_profile" in schema


class TestTCOPFLOWJournalEntry:
    def test_num_steps_default_zero(self):
        entry = JournalEntry(
            iteration=0, description="test", commands=[],
            objective_value=100.0, feasible=True, convergence_status="CONVERGED",
            violations_count=0, voltage_min=0.95, voltage_max=1.05,
            max_line_loading_pct=50.0, total_gen_mw=100.0, total_load_mw=90.0,
            llm_reasoning="test", mode="fresh", elapsed_seconds=1.0,
        )
        assert entry.num_steps == 0

    def test_num_steps_set(self):
        entry = JournalEntry(
            iteration=0, description="test", commands=[],
            objective_value=100.0, feasible=True, convergence_status="CONVERGED",
            violations_count=0, voltage_min=0.95, voltage_max=1.05,
            max_line_loading_pct=50.0, total_gen_mw=100.0, total_load_mw=90.0,
            llm_reasoning="test", mode="fresh", elapsed_seconds=1.0,
            num_steps=7,
        )
        assert entry.num_steps == 7

    def test_add_from_results_with_num_steps(self):
        journal = SearchJournal()
        from agentigrid.parsers.opflow_results import OPFLOWResult
        result = OPFLOWResult(
            converged=True, objective_value=100.0, convergence_status="CONVERGED",
            solver="IPOPT", model="POWER_BALANCE_POLAR",
            objective_type="MIN_GEN_COST", num_iterations=5, solve_time=0.1,
            total_gen_mw=100.0, total_load_mw=90.0,
        )
        entry = journal.add_from_results(
            iteration=0, description="test", commands=[],
            opflow_result=result, sim_elapsed=1.0,
            llm_reasoning="test", mode="fresh", num_steps=4,
        )
        assert entry.num_steps == 4


class TestTCOPFLOWSystemPrompt:
    def test_tcopflow_prompt_includes_multi_period(self):
        prompt = build_system_prompt(
            command_schema=command_schema_text(),
            network_summary="Network: 9 buses",
            application="tcopflow",
        )
        assert "Multi-Period" in prompt
        assert "scale_load_profile" in prompt

    def test_tcopflow_prompt_includes_voltage_control(self):
        prompt = build_system_prompt(
            command_schema=command_schema_text(),
            network_summary="Network: 9 buses",
            application="tcopflow",
        )
        assert "OPF Voltage Control" in prompt

    def test_opflow_prompt_unchanged(self):
        prompt = build_system_prompt(
            command_schema=command_schema_text(),
            network_summary="Network: 9 buses",
            application="opflow",
        )
        assert "Multi-Period" not in prompt
        assert "OPF Voltage Control" in prompt


class TestTCOPFLOWGoalClassifier:
    def test_application_kwarg_accepted(self):
        sys_prompt, user_prompt = build_classification_prompts(
            goal="test", termination_reason="completed",
            stats={"total_iterations": 1, "feasible_count": 1,
                   "infeasible_count": 0, "best_objective": 100, "best_iteration": 0},
            journal_formatted="test", total_tokens=0,
            application="tcopflow",
        )
        assert "TCOPFLOW" in sys_prompt
        assert "Multi-Period" in sys_prompt

    def test_scopflow_application_context(self):
        sys_prompt, _ = build_classification_prompts(
            goal="test", termination_reason="completed",
            stats={"total_iterations": 1, "feasible_count": 1,
                   "infeasible_count": 0, "best_objective": 100, "best_iteration": 0},
            journal_formatted="test", total_tokens=0,
            application="scopflow",
        )
        assert "SCOPFLOW" in sys_prompt

    def test_default_application_no_extra_context(self):
        sys_prompt, _ = build_classification_prompts(
            goal="test", termination_reason="completed",
            stats={"total_iterations": 1, "feasible_count": 1,
                   "infeasible_count": 0, "best_objective": 100, "best_iteration": 0},
            journal_formatted="test", total_tokens=0,
            application="opflow",
        )
        assert "Multi-Period Optimal Power Flow" not in sys_prompt
        assert "Security-Constrained OPF" not in sys_prompt


class TestTCOPFLOWPeriodFiles:
    def test_no_tcopflowout_dir(self, tmp_path):
        result = parse_tcopflow_period_files(tmp_path)
        assert result == []

    def test_empty_tcopflowout_dir(self, tmp_path):
        (tmp_path / "tcopflowout").mkdir()
        result = parse_tcopflow_period_files(tmp_path)
        assert result == []


class TestTCOPFLOWProfileMatching:
    def test_case_stem_variants_mod(self):
        from launcher.config_builder import _case_stem_variants
        variants = _case_stem_variants("case9mod")
        assert variants[0] == "case9mod"
        assert "case9" in variants

    def test_case_stem_variants_plain(self):
        from launcher.config_builder import _case_stem_variants
        variants = _case_stem_variants("case9")
        assert variants == ["case9"]

    def test_case_stem_variants_complex(self):
        from launcher.config_builder import _case_stem_variants
        variants = _case_stem_variants("case_ACTIVSg200")
        assert variants[0] == "case_ACTIVSg200"

    def test_match_profiles_for_case(self, tmp_path):
        from launcher.config_builder import match_profiles_for_case
        (tmp_path / "case9_load_P.csv").write_text("T,5\n0,100\n")
        (tmp_path / "case9_load_Q.csv").write_text("T,5\n0,50\n")
        case_path = tmp_path / "case9mod.m"
        case_path.touch()
        matches = match_profiles_for_case(case_path, tmp_path)
        assert len(matches["pload"]) == 1
        assert len(matches["qload"]) == 1
        assert "case9_load_P.csv" in matches["pload"][0].name

    def test_match_profiles_no_match(self, tmp_path):
        from launcher.config_builder import match_profiles_for_case
        (tmp_path / "other_load_P.csv").write_text("T,B\n0,100\n")
        case_path = tmp_path / "case99.m"
        case_path.touch()
        matches = match_profiles_for_case(case_path, tmp_path)
        assert len(matches["pload"]) == 1  # fallback: all profiles returned