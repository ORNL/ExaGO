"""Tests for SOPFLOW support (Phase 3, Step 3.4)."""

from __future__ import annotations

import csv
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from agentigrid.parsers.sopflow_parser import (
    parse_sopflow_output,
    parse_sopflow_simulation_result,
)
from agentigrid.parsers.sopflow_summary import sopflow_results_summary
from agentigrid.parsers import (
    results_summary_for_app,
    parse_simulation_result_for_app,
    parse_sopflow_metadata as dispatch_sopflow_metadata,
)
from agentigrid.parsers.opflow_parser import parse_opflow_output
from agentigrid.config import SearchConfig
from agentigrid.engine.metric_extractor import available_metrics, available_metrics_for_app
from agentigrid.engine.commands import ScaleWindScenario, parse_command
from agentigrid.engine.modifier import (
    apply_modifications,
    scale_wind_scenario_csv,
    scale_load_profile_csv,
    ModificationReport,
)
from agentigrid.parsers.matpower_model import MATNetwork, Bus, Generator, Branch, GenCost
from agentigrid.engine.journal import SearchJournal, JournalEntry
from agentigrid.prompts.system_prompt import build_system_prompt
from agentigrid.engine.schema_description import command_schema_text
from agentigrid.engine.goal_classifier import build_classification_prompts


# ---------------------------------------------------------------------------
# Synthetic SOPFLOW output — 3-bus system
# Same bus/branch/gen table format as OPFLOW but with "Stochastic Optimal
# Power Flow" header and stochastic metadata. AC OPF: realistic Vm, non-zero
# Qg/Qd. Includes a WIND generator row.
# ---------------------------------------------------------------------------

SAMPLE_SOPFLOW_OUTPUT = """\


******************************************************************************
This program contains Ipopt, a library for large-scale nonlinear optimization.
 Ipopt is released as open source code under the Eclipse Public License (EPL).
         For more information visit https://github.com/coin-or/Ipopt
******************************************************************************

This is Ipopt version 3.14.20, running with linear solver ma27.

Number of Iterations....: 8

Total seconds in IPOPT                               = 0.032

EXIT: Optimal Solution Found.
=============================================================
\tStochastic Optimal Power Flow
=============================================================
OPFLOW Model                        POWER_BALANCE_POLAR
Solver                              IPOPT
Number of scenarios                  10
Multi-contingency scenarios?         NO
Contingencies per scenario          0
Initialization                      MIDPOINT
Load loss allowed                    NO
Power imbalance allowed             NO
Ignore line flow constraints         NO

Convergence status                  CONVERGED
Objective value (base)              13250.78

------------------------------------------------------------------------------------------------------
Bus        Pd      Pdloss Qd      Qdloss Vm      Va      mult_Pmis      mult_Qmis      Pslack         Qslack        
------------------------------------------------------------------------------------------------------
1         0.00    0.00    0.00    0.00   1.045   0.000         0.00         0.00         0.00         0.00
2        80.00    0.00   20.00    0.00   1.025  -3.200         0.00         0.00         0.00         0.00
3        60.00    0.00   15.00    0.00   1.018  -5.500         0.00         0.00         0.00         0.00

------------------------------------------------------------------------------------------------------
From       To       Status     Sft      Stf     Slim     mult_Sf  mult_St 
----------------------------------------------------------------------------------------
1          2          1       50.00    50.00   100.00     0.00     0.00
1          3          1       70.43    70.43   150.00     0.00     0.00

----------------------------------------------------------------------------------------
Gen      Status     Fuel     Pg       Qg       Pmin     Pmax     Qmin     Qmax  
----------------------------------------------------------------------------------------
1          1        COAL     95.00    16.50    10.00   150.00   -50.00    50.00
2          1        GAS      35.43    10.20     5.00    80.00   -30.00    30.00
3          1        WIND     15.00     2.00     0.00    50.00   -10.00    10.00
"""

SAMPLE_SOPFLOW_DID_NOT_CONVERGE = """\
EXIT: Maximum Number of Iterations Exceeded.
=============================================================
\tStochastic Optimal Power Flow
=============================================================
OPFLOW Model                        POWER_BALANCE_POLAR
Solver                              IPOPT
Number of scenarios                  10
Multi-contingency scenarios?         NO
Contingencies per scenario          0
Initialization                      MIDPOINT
Load loss allowed                    NO
Power imbalance allowed             NO
Ignore line flow constraints         NO

Convergence status                  DID NOT CONVERGE
Objective value (base)              0.00

------------------------------------------------------------------------------------------------------
Bus        Pd      Pdloss Qd      Qdloss Vm      Va      mult_Pmis      mult_Qmis      Pslack         Qslack        
------------------------------------------------------------------------------------------------------
1         0.00    0.00    0.00    0.00   1.045   0.000         0.00         0.00         0.00         0.00
2        80.00    0.00   20.00    0.00   1.025  -3.200         0.00         0.00         0.00         0.00
3        60.00    0.00   15.00    0.00   1.018  -5.500         0.00         0.00         0.00         0.00

------------------------------------------------------------------------------------------------------
From       To       Status     Sft      Stf     Slim     mult_Sf  mult_St 
----------------------------------------------------------------------------------------
1          2          1       50.00    50.00   100.00     0.00     0.00
1          3          1       70.43    70.43   150.00     0.00     0.00

----------------------------------------------------------------------------------------
Gen      Status     Fuel     Pg       Qg       Pmin     Pmax     Qmin     Qmax  
----------------------------------------------------------------------------------------
1          1        COAL     95.00    16.50    10.00   150.00   -50.00    50.00
2          1        GAS      35.43    10.20     5.00    80.00   -30.00    30.00
3          1        WIND     15.00     2.00     0.00    50.00   -10.00    10.00
"""

SAMPLE_SOPFLOW_EMPAR = SAMPLE_SOPFLOW_OUTPUT.replace(
    "Solver                              IPOPT",
    "Solver                              EMPAR",
)

SAMPLE_SOPFLOW_MULTI_CONTINGENCY = SAMPLE_SOPFLOW_OUTPUT.replace(
    "Multi-contingency scenarios?         NO",
    "Multi-contingency scenarios?         YES",
)

SAMPLE_SOPFLOW_COUPLING = SAMPLE_SOPFLOW_OUTPUT


SAMPLE_SOPFLOW_NO_EXIT_MSG = """\
=============================================================
\tStochastic Optimal Power Flow
=============================================================
OPFLOW Model                        POWER_BALANCE_POLAR
Solver                              IPOPT
Number of scenarios                  10
Multi-contingency scenarios?         NO
Contingencies per scenario          0
Initialization                      MIDPOINT
Load loss allowed                    NO
Power imbalance allowed             NO
Ignore line flow constraints         NO

Convergence status                  CONVERGED
Objective value (base)              13250.78

------------------------------------------------------------------------------------------------------
Bus        Pd      Pdloss Qd      Qdloss Vm      Va      mult_Pmis      mult_Qmis      Pslack         Qslack        
------------------------------------------------------------------------------------------------------
1         0.00    0.00    0.00    0.00   1.045   0.000         0.00         0.00         0.00         0.00
2        80.00    0.00   20.00    0.00   1.025  -3.200         0.00         0.00         0.00         0.00
3        60.00    0.00   15.00    0.00   1.018  -5.500         0.00         0.00         0.00         0.00

------------------------------------------------------------------------------------------------------
From       To       Status     Sft      Stf     Slim     mult_Sf  mult_St 
----------------------------------------------------------------------------------------
1          2          1       50.00    50.00   100.00     0.00     0.00
1          3          1       70.43    70.43   150.00     0.00     0.00

----------------------------------------------------------------------------------------
Gen      Status     Fuel     Pg       Qg       Pmin     Pmax     Qmin     Qmax  
----------------------------------------------------------------------------------------
1          1        COAL     95.00    16.50    10.00   150.00   -50.00    50.00
2          1        GAS      35.43    10.20     5.00    80.00   -30.00    30.00
3          1        WIND     15.00     2.00     0.00    50.00   -10.00    10.00
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sopflow_result_and_meta():
    return parse_sopflow_output(SAMPLE_SOPFLOW_OUTPUT)


@pytest.fixture
def sopflow_result(sopflow_result_and_meta):
    return sopflow_result_and_meta[0]


@pytest.fixture
def sopflow_meta(sopflow_result_and_meta):
    return sopflow_result_and_meta[1]


@pytest.fixture
def mock_sim_result():
    sr = MagicMock()
    sr.success = True
    sr.stdout = SAMPLE_SOPFLOW_OUTPUT
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
        casename="test_sopflow",
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

class TestSOPFLOWParser:
    def test_parse_succeeds(self, sopflow_result_and_meta):
        result, meta = sopflow_result_and_meta
        assert result is not None
        assert meta is not None

    def test_num_scenarios(self, sopflow_meta):
        assert sopflow_meta["num_scenarios"] == 10

    def test_multi_contingency_no(self, sopflow_meta):
        assert sopflow_meta["multi_contingency"] is False

    def test_multi_contingency_yes(self):
        result, meta = parse_sopflow_output(SAMPLE_SOPFLOW_MULTI_CONTINGENCY)
        assert meta["multi_contingency"] is True

    def test_is_coupling_default_false(self, sopflow_meta):
        assert sopflow_meta["is_coupling"] is False

    def test_is_coupling_always_false_without_config(self):
        result, meta = parse_sopflow_output(SAMPLE_SOPFLOW_MULTI_CONTINGENCY)
        assert meta["is_coupling"] is False

    def test_contingencies_per_scenario(self, sopflow_meta):
        assert sopflow_meta["contingencies_per_scenario"] == 0

    def test_contingencies_per_scenario_nonzero(self):
        output = SAMPLE_SOPFLOW_OUTPUT.replace(
            "Contingencies per scenario          0",
            "Contingencies per scenario          5",
        )
        result, meta = parse_sopflow_output(output)
        assert meta["contingencies_per_scenario"] == 5

    def test_load_loss_not_allowed(self, sopflow_meta):
        assert sopflow_meta["load_loss_allowed"] is False

    def test_power_imbalance_not_allowed(self, sopflow_meta):
        assert sopflow_meta["power_imbalance_allowed"] is False

    def test_ignore_lineflow_no(self, sopflow_meta):
        assert sopflow_meta["ignore_lineflow"] is False

    def test_convergence_status(self, sopflow_result, sopflow_meta):
        assert sopflow_meta["convergence_status"] == "CONVERGED"
        assert sopflow_result.converged is True

    def test_objective_value(self, sopflow_result):
        assert sopflow_result.objective_value == pytest.approx(13250.78, abs=0.01)

    def test_objective_value_base(self, sopflow_meta):
        assert sopflow_meta["objective_value"] == pytest.approx(13250.78, abs=0.01)

    def test_solver_ipopt(self, sopflow_meta):
        assert sopflow_meta["solver"] == "IPOPT"

    def test_solver_empar(self):
        result, meta = parse_sopflow_output(SAMPLE_SOPFLOW_EMPAR)
        assert meta["solver"] == "EMPAR"

    def test_feasibility_detail(self, sopflow_result):
        assert sopflow_result.feasibility_detail == "feasible"

    def test_bus_count(self, sopflow_result):
        assert len(sopflow_result.buses) == 3

    def test_branch_count(self, sopflow_result):
        assert len(sopflow_result.branches) == 2

    def test_gen_count(self, sopflow_result):
        assert len(sopflow_result.generators) == 3

    def test_wind_generator_present(self, sopflow_result):
        wind_gens = [g for g in sopflow_result.generators if "wind" in g.fuel.lower()]
        assert len(wind_gens) == 1
        assert wind_gens[0].fuel == "WIND"
        assert wind_gens[0].Pg == pytest.approx(15.0, abs=0.01)

    def test_rejects_non_sopflow(self):
        with pytest.raises(ValueError, match="SOPFLOW"):
            parse_sopflow_output("This is OPFLOW output")

    def test_rejects_plain_opflow(self):
        with pytest.raises(ValueError, match="SOPFLOW"):
            parse_sopflow_output("Optimal Power Flow\nConvergence status CONVERGED")

    def test_voltages_non_unity(self, sopflow_result):
        vms = [b.Vm for b in sopflow_result.buses]
        assert any(abs(vm - 1.0) > 0.01 for vm in vms)

    def test_reactive_non_zero(self, sopflow_result):
        qgs = [g.Qg for g in sopflow_result.generators]
        assert any(abs(qg) > 0.1 for qg in qgs)

    def test_converged_override_no_exit_msg(self):
        """SOPFLOW without IPOPT EXIT line: converged=True based on convergence_status."""
        result, meta = parse_sopflow_output(SAMPLE_SOPFLOW_NO_EXIT_MSG)
        assert meta["convergence_status"] == "CONVERGED"
        assert result.converged is True
        assert result.feasibility_detail == "feasible"


class TestSOPFLOWDidNotConverge:
    def test_did_not_converge(self):
        result, meta = parse_sopflow_output(SAMPLE_SOPFLOW_DID_NOT_CONVERGE)
        assert meta["convergence_status"] == "DID NOT CONVERGE"
        assert result.converged is False

    def test_marginal_with_exit_status(self):
        result, meta = parse_sopflow_output(SAMPLE_SOPFLOW_DID_NOT_CONVERGE)
        assert result.ipopt_exit_status == "Maximum Number of Iterations Exceeded."
        assert result.feasibility_detail == "marginal"

    def test_infeasible_with_negative_losses(self):
        output = SAMPLE_SOPFLOW_OUTPUT.replace(
            "1         0.00    0.00    0.00    0.00   1.045",
            "1         0.00    0.00    0.00    0.00   0.050",
        ).replace("2        80.00", "2       300.00").replace("3        60.00", "3       400.00")
        result, meta = parse_sopflow_output(output)
        assert result.feasibility_detail == "infeasible"

    def test_near_boundary_voltage_marginal(self):
        """Non-converged with 'Infeasible Problem Detected' but voltage near limit → marginal."""
        output = SAMPLE_SOPFLOW_DID_NOT_CONVERGE.replace(
            "Maximum Number of Iterations Exceeded.",
            "Infeasible Problem Detected.",
        )
        bus_limits = {1: (0.95, 1.05), 2: (0.95, 1.05), 3: (0.95, 1.05)}
        result, meta = parse_sopflow_output(output, bus_limits=bus_limits)
        assert result.converged is False
        assert result.feasibility_detail == "marginal"

    def test_near_boundary_line_loading_marginal(self):
        """Non-converged with line loading near 100% → marginal."""
        output = (
            "EXIT: Infeasible Problem Detected.\n"
            "Stochastic Optimal Power Flow\n"
            "Convergence status                  DID NOT CONVERGE\n"
            "Objective value (base)              0.00\n"
            "Number of scenarios                  10\n"
            "Solver                              IPOPT\n"
            "Contingencies per scenario          0\n"
            "Load loss allowed                    NO\n"
            "Power imbalance allowed              NO\n"
            "Ignore line flow constraints         NO\n"
            "Multi-contingency scenarios?         NO\n"
            "------------------------------------------------------------------------------------------------------\n"
            "Bus        Pd      Pdloss Qd      Qdloss Vm      Va      mult_Pmis      mult_Qmis      Pslack         Qslack\n"
            "------------------------------------------------------------------------------------------------------\n"
            "1         0.00    0.00    0.00    0.00   1.045   0.000         0.00         0.00         0.00         0.00\n"
            "2        80.00    0.00   20.00    0.00   1.025  -4.100         0.00         0.00         0.00         0.00\n"
            "------------------------------------------------------------------------------------------------------\n"
            "From       To       Status     Sft      Stf     Slim     mult_Sf  mult_St\n"
            "----------------------------------------------------------------------------------------\n"
            "1          2          1       97.00    97.00   100.00     0.00     0.00\n"
            "1          3          1       60.00    60.00   150.00     0.00     0.00\n"
            "----------------------------------------------------------------------------------------\n"
            "Gen      Status     Fuel     Pg       Qg       Pmin     Pmax     Qmin     Qmax\n"
            "----------------------------------------------------------------------------------------\n"
            "1          1        COAL     95.00    16.50    10.00   150.00   -50.00    50.00\n"
        )
        result, meta = parse_sopflow_output(output)
        assert result.converged is False
        assert result.feasibility_detail == "marginal"

    def test_far_from_boundary_infeasible(self):
        """Non-converged with metrics far from limits → infeasible."""
        output = (
            "EXIT: Infeasible Problem Detected.\n"
            "Stochastic Optimal Power Flow\n"
            "Convergence status                  DID NOT CONVERGE\n"
            "Objective value (base)              0.00\n"
            "Number of scenarios                  10\n"
            "Solver                              IPOPT\n"
            "Contingencies per scenario          0\n"
            "Load loss allowed                    NO\n"
            "Power imbalance allowed              NO\n"
            "Ignore line flow constraints         NO\n"
            "Multi-contingency scenarios?         NO\n"
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
            "1          1        COAL     95.00    16.50    10.00   150.00   -50.00    50.00\n"
        )
        result, meta = parse_sopflow_output(output)
        assert result.converged is False
        assert result.feasibility_detail == "infeasible"


class TestSOPFLOWSummary:
    def test_header_mentions_stochastic(self, sopflow_result):
        summary = sopflow_results_summary(sopflow_result, num_scenarios=10)
        assert "Stochastic" in summary

    def test_sopflow_label(self, sopflow_result):
        summary = sopflow_results_summary(sopflow_result)
        assert "SOPFLOW" in summary

    def test_scenarios_shown(self, sopflow_result):
        summary = sopflow_results_summary(sopflow_result, num_scenarios=10)
        assert "10" in summary

    def test_has_voltage_profile(self, sopflow_result):
        summary = sopflow_results_summary(sopflow_result)
        assert "Voltage" in summary

    def test_has_generation_info(self, sopflow_result):
        summary = sopflow_results_summary(sopflow_result)
        assert "Generation:" in summary
        assert "Load:" in summary

    def test_has_line_loading(self, sopflow_result):
        summary = sopflow_results_summary(sopflow_result)
        assert "loaded lines" in summary.lower()

    def test_wind_gen_info(self, sopflow_result):
        summary = sopflow_results_summary(sopflow_result)
        assert "Wind" in summary

    def test_first_stage_note(self, sopflow_result):
        summary = sopflow_results_summary(sopflow_result)
        assert "first-stage" in summary.lower() or "here-and-now" in summary.lower()

    def test_negative_losses_flagged(self):
        output = SAMPLE_SOPFLOW_OUTPUT.replace(
            "1         0.00    0.00    0.00    0.00   1.045",
            "1         0.00    0.00    0.00    0.00   0.050",
        ).replace("2        80.00", "2       300.00").replace("3        60.00", "3       400.00")
        result, _ = parse_sopflow_output(output)
        summary = sopflow_results_summary(result)
        assert "UNPHYSICAL" in summary or "negative losses" in summary.lower()

    def test_stale_per_scenario_disclaimer_removed(self, sopflow_result):
        # The old "per-scenario data not available" disclaimer is obsolete:
        # per-scenario second-stage dispatch IS now available.
        summary = sopflow_results_summary(sopflow_result, num_scenarios=10)
        assert "not available" not in summary.lower()

    def test_wind_absorption_block_rendered(self, sopflow_result):
        wind_absorption = {
            "num_scenarios": 10,
            "total_available_mw": 27574.0,
            "total_dispatched_mw": 7534.5,
            "total_curtailment_mw": 20039.5,
            "curtailment_pct": 72.7,
            "avg_dispatched_mw": 753.45,
        }
        summary = sopflow_results_summary(
            sopflow_result, num_scenarios=10, wind_absorption=wind_absorption
        )
        assert "Wind absorption" in summary
        assert "Offered" in summary and "Dispatched" in summary and "Curtailed" in summary
        assert "Curtailment is active" in summary

    def test_wind_absorption_all_absorbed_message(self, sopflow_result):
        wind_absorption = {
            "num_scenarios": 10,
            "total_available_mw": 5514.8,
            "total_dispatched_mw": 5501.8,
            "total_curtailment_mw": 13.0,
            "curtailment_pct": 0.2,
            "avg_dispatched_mw": 550.18,
        }
        summary = sopflow_results_summary(
            sopflow_result, num_scenarios=10, wind_absorption=wind_absorption
        )
        assert "All offered wind is absorbed" in summary

    def test_no_absorption_block_without_data(self, sopflow_result):
        summary = sopflow_results_summary(sopflow_result, num_scenarios=0)
        assert "Wind absorption" not in summary

    def test_wind_capacity_utilization(self, sopflow_result):
        summary = sopflow_results_summary(sopflow_result, num_scenarios=10)
        assert "utilization" in summary.lower() or "capacity" in summary.lower()

    def test_stale_wind_at_pmax_warning_removed(self, sopflow_result):
        # The old "wind at maximum capacity / scale_wind_scenario won't change
        # dispatch" WARNING is obsolete and must no longer be emitted.
        summary = sopflow_results_summary(sopflow_result, num_scenarios=10)
        assert "scale_wind_scenario alone will NOT" not in summary
        assert "maximum capacity in the base-case" not in summary


class TestSOPFLOWDispatch:
    def test_dispatch_sopflow_summary(self, sopflow_result):
        summary = results_summary_for_app(sopflow_result, "sopflow", num_scenarios=10)
        assert "SOPFLOW" in summary
        assert "Stochastic" in summary

    def test_dispatch_sopflow_with_scenarios(self, sopflow_result):
        summary = results_summary_for_app(sopflow_result, "sopflow", num_scenarios=10)
        assert "10" in summary

    def test_parse_dispatch_sopflow(self, mock_sim_result):
        result = parse_simulation_result_for_app(mock_sim_result, "sopflow")
        assert result is not None
        assert result.objective_value == pytest.approx(13250.78, abs=0.01)

    def test_parse_dispatch_sopflow_metadata(self, mock_sim_result):
        meta = dispatch_sopflow_metadata(mock_sim_result)
        assert meta is not None
        assert meta["num_scenarios"] == 10

    def test_parse_sopflow_simulation_result_failed(self):
        sr = MagicMock()
        sr.success = False
        result = parse_sopflow_simulation_result(sr)
        assert result is None

    def test_dispatch_opflow_unchanged(self, mock_sim_result):
        result = parse_simulation_result_for_app(mock_sim_result, "opflow")
        assert result is not None


class TestSOPFLOWConfig:
    def test_sopflow_scenario_field_in_config(self):
        cfg = SearchConfig(
            max_iterations=10,
            default_mode="accumulative",
            base_case=None,
            gic_file=None,
            application="sopflow",
            scenario_file=None,
        )
        assert cfg.scenario_file is None

    def test_scenario_path_stored(self, tmp_path):
        scenario = tmp_path / "case9_scenarios.csv"
        scenario.touch()
        cfg = SearchConfig(
            max_iterations=10,
            default_mode="accumulative",
            base_case=None,
            gic_file=None,
            application="sopflow",
            scenario_file=scenario,
        )
        assert cfg.scenario_file == scenario

    def test_sopflow_params_defaults(self):
        cfg = SearchConfig(
            max_iterations=10,
            default_mode="accumulative",
            base_case=None,
            gic_file=None,
            application="sopflow",
        )
        assert cfg.sopflow_solver == "IPOPT"
        assert cfg.sopflow_iscoupling == 0

    def test_sopflow_params_stored(self):
        cfg = SearchConfig(
            max_iterations=10,
            default_mode="accumulative",
            base_case=None,
            gic_file=None,
            application="sopflow",
            sopflow_solver="EMPAR",
            sopflow_iscoupling=1,
        )
        assert cfg.sopflow_solver == "EMPAR"
        assert cfg.sopflow_iscoupling == 1


class TestSOPFLOWMetrics:
    def test_sopflow_has_all_metrics(self):
        sf_metrics = available_metrics_for_app("sopflow")
        all_metrics = available_metrics()
        assert set(sf_metrics) == set(all_metrics)

    def test_sopflow_has_voltage_metrics(self):
        sf_metrics = available_metrics_for_app("sopflow")
        assert "voltage_min" in sf_metrics
        assert "voltage_max" in sf_metrics

    def test_sopflow_has_reactive_metrics(self):
        sf_metrics = available_metrics_for_app("sopflow")
        assert "total_reactive_gen_mvar" in sf_metrics


class TestSOPFLOWModifier:
    def test_scale_wind_scenario_skipped_for_opflow(self, minimal_network):
        cmd = ScaleWindScenario(factor=1.5)
        _, report = apply_modifications(minimal_network, [cmd], application="opflow")
        assert len(report.applied) == 0
        assert any("SOPFLOW" in w for w in report.warnings)

    def test_scale_wind_scenario_skipped_for_tcopflow(self, minimal_network):
        cmd = ScaleWindScenario(factor=1.5)
        _, report = apply_modifications(minimal_network, [cmd], application="tcopflow")
        assert len(report.applied) == 0
        assert any("SOPFLOW" in w for w in report.warnings)

    def test_scale_wind_scenario_requires_scenario_file(self, minimal_network):
        cmd = ScaleWindScenario(factor=1.5)
        _, report = apply_modifications(
            minimal_network, [cmd], application="sopflow",
            scenario_file=None,
        )
        assert len(report.applied) == 0
        assert len(report.skipped) == 1

    def test_scale_wind_scenario_csv_single_period(self, tmp_path):
        csv_path = tmp_path / "case9_10_scenarios.csv"
        csv_path.write_text(
            "scenario_nr,3_Wind_3,9_Wind_9,weight\n"
            "1,10.0,20.0,0.1\n"
            "2,15.0,25.0,0.2\n"
        )
        out_dir = tmp_path / "scaled"
        out_path = scale_wind_scenario_csv(csv_path, 1.5, out_dir)
        assert out_path.exists()
        lines = out_path.read_text().strip().split("\n")
        assert lines[0] == "scenario_nr,3_Wind_3,9_Wind_9,weight"
        vals = lines[1].split(",")
        assert vals[0] == "1"
        assert float(vals[1]) == pytest.approx(15.0, abs=0.01)
        assert float(vals[2]) == pytest.approx(30.0, abs=0.01)
        assert vals[3] == "0.1"

    def test_scale_wind_scenario_csv_multi_period(self, tmp_path):
        csv_path = tmp_path / "case9_scenarios.csv"
        csv_path.write_text(
            "sim_timestamp,scenario_nr,3_Wind_3,9_Wind_9\n"
            "2024-01-01,1,10.0,20.0\n"
            "2024-01-01,2,15.0,25.0\n"
        )
        out_dir = tmp_path / "scaled"
        out_path = scale_wind_scenario_csv(csv_path, 2.0, out_dir)
        lines = out_path.read_text().strip().split("\n")
        vals = lines[1].split(",")
        assert vals[0] == "2024-01-01"
        assert vals[1] == "1"
        assert float(vals[2]) == pytest.approx(20.0, abs=0.01)
        assert float(vals[3]) == pytest.approx(40.0, abs=0.01)

    def test_scale_wind_scenario_csv_preserves_non_numeric(self, tmp_path):
        csv_path = tmp_path / "case9_10_scenarios.csv"
        csv_path.write_text(
            "scenario_nr,3_Wind_3,weight\n"
            "1,10.0,0.1\n"
        )
        out_dir = tmp_path / "scaled"
        out_path = scale_wind_scenario_csv(csv_path, 1.2, out_dir)
        lines = out_path.read_text().strip().split("\n")
        vals = lines[1].split(",")
        assert vals[0] == "1"
        assert float(vals[1]) == pytest.approx(12.0, abs=0.01)
        assert vals[2] == "0.1"

    def test_scale_wind_scenario_modifies_file(self, tmp_path):
        scenario = tmp_path / "case9_scenarios.csv"
        scenario.write_text(
            "sim_timestamp,scenario_nr,3_Wind_3\n"
            "2024-01-01,1,10.0\n"
        )
        net = MATNetwork(
            casename="test", version="2", baseMVA=100.0,
            buses=[], generators=[], branches=[], gencost=[],
            header_comments="",
        )
        cmd = ScaleWindScenario(factor=1.5)
        out_dir = tmp_path / "scenarios"
        _, report = apply_modifications(
            net, [cmd], application="sopflow",
            scenario_file=scenario,
            scenario_output_dir=out_dir,
        )
        assert len(report.applied) == 1
        assert "scenario_file" in report.scenario_paths

    def test_scale_wind_scenario_cumulative(self, tmp_path):
        scenario = tmp_path / "case9_scenarios.csv"
        scenario.write_text(
            "scenario_nr,3_Wind_3\n"
            "1,10.0\n"
        )
        net = MATNetwork(
            casename="test", version="2", baseMVA=100.0,
            buses=[], generators=[], branches=[], gencost=[],
            header_comments="",
        )
        out_dir = tmp_path / "scenarios_iter1"
        cmd1 = ScaleWindScenario(factor=1.2)
        _, report1 = apply_modifications(
            net, [cmd1], application="sopflow",
            scenario_file=scenario,
            scenario_output_dir=out_dir,
        )
        assert "scenario_file" in report1.scenario_paths
        scaled = report1.scenario_paths["scenario_file"]
        lines = scaled.read_text().strip().split("\n")
        assert float(lines[1].split(",")[1]) == pytest.approx(12.0, abs=0.1)

    def test_scale_load_profile_skipped_for_sopflow(self, tmp_path):
        from agentigrid.engine.commands import ScaleLoadProfile
        p_path = tmp_path / "load_P.csv"
        q_path = tmp_path / "load_Q.csv"
        p_path.write_text("T,B1\n0,100.0\n")
        q_path.write_text("T,B1\n0,50.0\n")
        net = MATNetwork(
            casename="test", version="2", baseMVA=100.0,
            buses=[], generators=[], branches=[], gencost=[],
            header_comments="",
        )
        cmd = ScaleLoadProfile(factor=1.2)
        _, report = apply_modifications(
            net, [cmd], application="sopflow",
            pload_profile=p_path, qload_profile=q_path,
        )
        assert len(report.applied) == 0
        assert any("TCOPFLOW" in w for w in report.warnings)


class TestSOPFLOWCommandParsing:
    def test_parse_scale_wind_scenario(self):
        cmd = parse_command({"action": "scale_wind_scenario", "factor": 1.5})
        assert isinstance(cmd, ScaleWindScenario)
        assert cmd.factor == pytest.approx(1.5, abs=0.01)

    def test_parse_scale_wind_scenario_missing_factor(self):
        with pytest.raises(ValueError, match="factor"):
            parse_command({"action": "scale_wind_scenario"})

    def test_command_schema_includes_scale_wind_scenario(self):
        schema = command_schema_text()
        assert "scale_wind_scenario" in schema


class TestSOPFLOWJournalEntry:
    def test_num_scenarios_default_zero(self):
        entry = JournalEntry(
            iteration=0, description="test", commands=[],
            objective_value=100.0, feasible=True, convergence_status="CONVERGED",
            violations_count=0, voltage_min=0.95, voltage_max=1.05,
            max_line_loading_pct=50.0, total_gen_mw=100.0, total_load_mw=90.0,
            llm_reasoning="test", mode="fresh", elapsed_seconds=1.0,
        )
        assert entry.num_scenarios == 0

    def test_num_scenarios_set(self):
        entry = JournalEntry(
            iteration=0, description="test", commands=[],
            objective_value=100.0, feasible=True, convergence_status="CONVERGED",
            violations_count=0, voltage_min=0.95, voltage_max=1.05,
            max_line_loading_pct=50.0, total_gen_mw=100.0, total_load_mw=90.0,
            llm_reasoning="test", mode="fresh", elapsed_seconds=1.0,
            num_scenarios=10,
        )
        assert entry.num_scenarios == 10

    def test_add_from_results_with_num_scenarios(self):
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
            llm_reasoning="test", mode="fresh", num_scenarios=10,
        )
        assert entry.num_scenarios == 10


class TestSOPFLOWSystemPrompt:
    def test_sopflow_prompt_includes_stochastic(self):
        prompt = build_system_prompt(
            command_schema=command_schema_text(),
            network_summary="Network: 9 buses",
            application="sopflow",
        )
        assert "Stochastic" in prompt
        assert "scale_wind_scenario" in prompt

    def test_sopflow_prompt_includes_two_stage(self):
        prompt = build_system_prompt(
            command_schema=command_schema_text(),
            network_summary="Network: 9 buses",
            application="sopflow",
        )
        assert "two-stage" in prompt.lower() or "first stage" in prompt.lower()

    def test_sopflow_prompt_includes_wind_scenario(self):
        prompt = build_system_prompt(
            command_schema=command_schema_text(),
            network_summary="Network: 9 buses",
            application="sopflow",
        )
        assert "wind" in prompt.lower()
        assert "scenario" in prompt.lower()

    def test_sopflow_prompt_includes_absorption_guidance(self):
        prompt = build_system_prompt(
            command_schema=command_schema_text(),
            network_summary="Network: 9 buses",
            application="sopflow",
        )
        # Post-fix guidance: absorption/curtailment is the primary signal,
        # and "max feasible wind scale" is ill-posed.
        assert "curtail" in prompt.lower()
        assert "absorb" in prompt.lower()

    def test_sopflow_prompt_uses_scenfile_not_windgen(self):
        prompt = build_system_prompt(
            command_schema=command_schema_text(),
            network_summary="Network: 9 buses",
            application="sopflow",
        )
        assert "-windgen" not in prompt
        assert "-scenfile" in prompt

    def test_opflow_prompt_unchanged(self):
        prompt = build_system_prompt(
            command_schema=command_schema_text(),
            network_summary="Network: 9 buses",
            application="opflow",
        )
        assert "Stochastic Optimal Power Flow" not in prompt
        assert "Stochastic OPF" not in prompt


class TestSOPFLOWGoalClassifier:
    def test_sopflow_application_context(self):
        sys_prompt, _ = build_classification_prompts(
            goal="test", termination_reason="completed",
            stats={"total_iterations": 1, "feasible_count": 1,
                   "infeasible_count": 0, "best_objective": 100, "best_iteration": 0},
            journal_formatted="test", total_tokens=0,
            application="sopflow",
        )
        assert "SOPFLOW" in sys_prompt
        assert "Stochastic" in sys_prompt

    def test_default_application_no_sopflow_context(self):
        sys_prompt, _ = build_classification_prompts(
            goal="test", termination_reason="completed",
            stats={"total_iterations": 1, "feasible_count": 1,
                   "infeasible_count": 0, "best_objective": 100, "best_iteration": 0},
            journal_formatted="test", total_tokens=0,
            application="opflow",
        )
        assert "Stochastic Optimal Power Flow" not in sys_prompt


class TestSOPFLOWScenarioMatching:
    def test_scan_scenario_files(self, tmp_path):
        (tmp_path / "case9_scenarios.csv").write_text("scenario_nr,wind\n1,10.0\n")
        (tmp_path / "case9_10_scenarios.csv").write_text("scenario_nr,wind,weight\n1,10.0,0.1\n")
        (tmp_path / "other_file.csv").write_text("x,y\n1,2\n")
        from launcher.config_builder import scan_scenario_files
        files = scan_scenario_files(tmp_path)
        names = [f.name for f in files]
        assert "case9_scenarios.csv" in names
        assert "case9_10_scenarios.csv" in names
        assert "other_file.csv" not in names

    def test_match_scenarios_for_case(self, tmp_path):
        (tmp_path / "case9_scenarios.csv").write_text("scenario_nr,wind\n1,10.0\n")
        (tmp_path / "case9_10_scenarios.csv").write_text("scenario_nr,wind,weight\n1,10.0,0.1\n")
        from launcher.config_builder import match_scenarios_for_case
        case_path = tmp_path / "case9mod_gen3_wind.m"
        case_path.touch()
        matches = match_scenarios_for_case(case_path, tmp_path)
        assert len(matches) >= 1
        names = [p.name for p in matches]
        assert any("case9" in n for n in names)

    def test_match_scenarios_no_match(self, tmp_path):
        (tmp_path / "other_scenarios.csv").write_text("scenario_nr,wind\n1,10.0\n")
        from launcher.config_builder import match_scenarios_for_case
        case_path = tmp_path / "case99.m"
        case_path.touch()
        matches = match_scenarios_for_case(case_path, tmp_path)


class TestScenarioRowCount:
    def test_count_scenario_rows_single_period(self, tmp_path):
        from agentigrid.engine.agent_loop import _count_scenario_rows
        csv_path = tmp_path / "case9_10_scenarios.csv"
        csv_path.write_text("scenario_nr,3_Wind_1,weight\n1,85.0,0.1\n2,105.0,0.1\n")
        assert _count_scenario_rows(csv_path) == 2

    def test_count_scenario_rows_multi_period(self, tmp_path):
        from agentigrid.engine.agent_loop import _count_scenario_rows
        csv_path = tmp_path / "case9_scenarios.csv"
        csv_path.write_text(
            "sim_timestamp,scenario_nr,3_Wind_1\n"
            "2024-01-01,1,85.0\n"
            "2024-01-01,2,105.0\n"
            "2024-01-01,3,65.0\n"
        )
        assert _count_scenario_rows(csv_path) == 3

    def test_count_scenario_rows_missing_file(self, tmp_path):
        from agentigrid.engine.agent_loop import _count_scenario_rows
        csv_path = tmp_path / "nonexistent.csv"
        assert _count_scenario_rows(csv_path) == 1


class TestSOPFLOWAnalyzeHandler:
    """Test that the SOPFLOW analyze handler doesn't blanket-intercept queries
    about voltage/loading that mention 'scenario' or 'wind'."""

    @pytest.fixture
    def sopflow_loop(self):
        from agentigrid.engine.agent_loop import AgentLoopController
        result, _ = parse_sopflow_output(SAMPLE_SOPFLOW_OUTPUT)
        loop = MagicMock(spec=AgentLoopController)
        loop._latest_opflow = result
        loop._sopflow_num_scenarios = 10
        loop._tcopflow_period_data = []
        loop._tcopflow_num_steps = 0
        loop._config = MagicMock()
        loop._config.search.application = "sopflow"
        loop._run_analysis_query = AgentLoopController._run_analysis_query.__get__(loop)
        return loop

    def test_pure_scenario_query_returns_summary(self, sopflow_loop):
        resp = sopflow_loop._run_analysis_query("how many scenarios are there?")
        assert "10 scenarios" in resp
        assert "SOPFLOW" in resp

    def test_scenario_query_with_voltage_falls_through(self, sopflow_loop):
        resp = sopflow_loop._run_analysis_query(
            "voltage variations across wind scenarios"
        )
        assert "10 scenarios" not in resp

    def test_scenario_query_with_loading_falls_through(self, sopflow_loop):
        resp = sopflow_loop._run_analysis_query(
            "line loading across all wind scenarios"
        )
        assert "10 scenarios" not in resp

    def test_scenario_query_with_generator_falls_through(self, sopflow_loop):
        resp = sopflow_loop._run_analysis_query(
            "generator dispatch in stochastic scenarios"
        )
        assert "10 scenarios" not in resp

    def test_pure_wind_query_returns_summary(self, sopflow_loop):
        resp = sopflow_loop._run_analysis_query("wind scenario summary")
        assert "10 scenarios" in resp
        assert "SOPFLOW" in resp

    def test_pure_stochastic_query_returns_summary(self, sopflow_loop):
        resp = sopflow_loop._run_analysis_query("stochastic optimization summary")
        assert "10 scenarios" in resp

    def test_scenario_query_includes_wind_utilization(self, sopflow_loop):
        resp = sopflow_loop._run_analysis_query("wind scenario info")
        assert "utilization" in resp.lower() or "capacity" in resp.lower()

    def test_scenario_query_warns_when_wind_at_pmax(self, sopflow_loop):
        wind_gens = [g for g in sopflow_loop._latest_opflow.generators
                     if "wind" in g.fuel.lower() and g.status == 1]
        if wind_gens and all(g.Pg >= g.Pmax * 0.995 for g in wind_gens):
            resp = sopflow_loop._run_analysis_query("scenario summary")
            assert "WARNING" in resp or "maximum capacity" in resp
