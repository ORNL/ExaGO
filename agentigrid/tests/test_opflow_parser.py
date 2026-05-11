"""Tests for the OPFLOW results parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentigrid.parsers.opflow_parser import parse_opflow_output, parse_simulation_result
from agentigrid.parsers.opflow_results import OPFLOWResult
from agentigrid.parsers.results_summary import results_summary

SAMPLE = Path(__file__).resolve().parent / "sample_opflow_output.txt"
_has_sample = SAMPLE.exists()


@pytest.fixture(scope="module")
def sample_text() -> str:
    if not _has_sample:
        pytest.skip("sample_opflow_output.txt not found")
    return SAMPLE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def result(sample_text: str) -> OPFLOWResult:
    return parse_opflow_output(sample_text)


# ===========================================================================
# Parse tests
# ===========================================================================

class TestParseOPFLOW:

    def test_converged(self, result: OPFLOWResult):
        assert result.converged is True

    def test_convergence_status(self, result: OPFLOWResult):
        assert result.convergence_status == "CONVERGED"

    def test_objective_value(self, result: OPFLOWResult):
        assert abs(result.objective_value - 27557.57) < 0.01

    def test_solver(self, result: OPFLOWResult):
        assert result.solver == "IPOPT"

    def test_model(self, result: OPFLOWResult):
        assert result.model == "POWER_BALANCE_POLAR"

    def test_objective_type(self, result: OPFLOWResult):
        assert result.objective_type == "MIN_GEN_COST"

    def test_num_iterations(self, result: OPFLOWResult):
        assert result.num_iterations == 23

    def test_solve_time(self, result: OPFLOWResult):
        assert result.solve_time == pytest.approx(0.042, abs=0.001)

    def test_bus_count(self, result: OPFLOWResult):
        assert len(result.buses) == 200

    def test_branch_count(self, result: OPFLOWResult):
        assert len(result.branches) == 245

    def test_gen_count(self, result: OPFLOWResult):
        assert len(result.generators) == 49


# ===========================================================================
# Specific value tests
# ===========================================================================

class TestSpecificValues:

    def test_bus_189_slack(self, result: OPFLOWResult):
        """Bus 189 is the slack bus — Va should be 0.0."""
        bus189 = next(b for b in result.buses if b.bus_id == 189)
        assert bus189.Va == 0.0

    def test_bus_100_highest_vm(self, result: OPFLOWResult):
        """Bus with highest Vm."""
        max_bus = max(result.buses, key=lambda b: b.Vm)
        # Verify voltage_max matches
        assert abs(max_bus.Vm - result.voltage_max) < 1e-6

    def test_branch_187_189_highest_sf(self, result: OPFLOWResult):
        """Branch 187->189 should have the highest Sf."""
        max_br = max(result.branches, key=lambda br: br.Sf)
        assert max_br.from_bus == 187
        assert max_br.to_bus == 189
        assert abs(max_br.Sf - 387.19) < 0.01

    def test_gen_189(self, result: OPFLOWResult):
        """Gen at bus 189: Pg≈383.40, fuel=COAL."""
        gen189 = next(g for g in result.generators if g.bus == 189)
        assert abs(gen189.Pg - 383.40) < 0.01
        assert gen189.fuel == "COAL"


# ===========================================================================
# Derived metrics tests
# ===========================================================================

class TestDerivedMetrics:

    def test_voltage_min(self, result: OPFLOWResult):
        assert abs(result.voltage_min - 1.062) < 0.001

    def test_voltage_max(self, result: OPFLOWResult):
        assert abs(result.voltage_max - 1.100) < 0.001

    def test_gen_exceeds_load(self, result: OPFLOWResult):
        assert result.total_gen_mw > result.total_load_mw

    def test_max_line_loading(self, result: OPFLOWResult):
        # Highest loading is ~71.3% (bus 146->147: 93.07/130.60)
        assert result.max_line_loading_pct > 70.0
        assert result.max_line_loading_pct < 75.0


# ===========================================================================
# Summary test
# ===========================================================================

class TestResultsSummary:

    def test_summary_content(self, result: OPFLOWResult):
        text = results_summary(result)
        assert "CONVERGED" in text
        assert "27,557.57" in text
        assert "IPOPT" in text
        assert "Most loaded lines:" in text
        assert "146->147" in text
        assert "Generators:" in text
        assert "Violations:" in text


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:

    def test_empty_string(self):
        with pytest.raises(ValueError, match="not.*OPFLOW"):
            parse_opflow_output("")

    def test_random_text(self):
        with pytest.raises(ValueError, match="not.*OPFLOW"):
            parse_opflow_output("This is just random text\nwith no OPFLOW output")

    def test_non_converged(self, sample_text: str):
        """Replace EXIT message to simulate non-convergence."""
        modified = sample_text.replace(
            "EXIT: Optimal Solution Found.",
            "EXIT: Maximum Number of Iterations Exceeded.",
        ).replace(
            "Convergence status                  CONVERGED",
            "Convergence status                  DIVERGED",
        )
        result = parse_opflow_output(modified)
        assert result.converged is False
        assert result.convergence_status == "DIVERGED"

    def test_parse_simulation_result_failed(self):
        """parse_simulation_result returns None for failed simulations."""

        class FakeResult:
            success = False
            stdout = ""

        assert parse_simulation_result(FakeResult()) is None

    def test_parse_simulation_result_success(self, sample_text: str):
        """parse_simulation_result works for successful simulations."""

        class FakeResult:
            success = True
            stdout = sample_text

        result = parse_simulation_result(FakeResult())
        assert result is not None
        assert result.converged is True


# ===========================================================================
# Bus-limits violation checking
# ===========================================================================

def _minimal_opflow_stdout(vm_value: float) -> str:
    """Build the minimum valid OPFLOW stdout with a single bus at *vm_value*."""
    bus_row = f"  1    0.00    0.00    0.00    0.00    {vm_value:.4f}    0.00    0.00    0.00    0.00    0.00"
    return (
        "Optimal Power Flow\n"
        "EXIT: Optimal Solution Found.\n"
        "Model                             POWER_BALANCE_POLAR\n"
        "Solver                            IPOPT\n"
        "Objective                         MIN_GEN_COST\n"
        "Convergence status                CONVERGED\n"
        "Objective value                   1000.00\n"
        "Bus      Pd      Pd      Qd      Qd      Vm      Va      mult_Pmis  mult_Qmis  Pslack  Qslack\n"
        "------------------------------------------------------------\n"
        f"{bus_row}\n"
        "------------------------------------------------------------\n"
    )


class TestBusLimitsViolations:
    """Verify that violation detection uses per-bus limits when provided."""

    def test_no_violation_within_default_limits(self):
        """Vm=1.06 is within 0.9–1.1 (default), so no violations."""
        result = parse_opflow_output(_minimal_opflow_stdout(1.06))
        assert result.num_violations == 0

    def test_violation_against_tighter_bus_limits(self):
        """Vm=1.06 violates Vmax=1.05 when bus_limits are provided."""
        result = parse_opflow_output(
            _minimal_opflow_stdout(1.06),
            bus_limits={1: (0.95, 1.05)},
        )
        assert result.num_violations == 1
        assert "Bus 1" in result.violation_details[0]
        assert "> 1.05" in result.violation_details[0]

    def test_violation_below_tighter_vmin(self):
        """Vm=0.96 violates Vmin=0.97 when bus_limits are provided."""
        result = parse_opflow_output(
            _minimal_opflow_stdout(0.96),
            bus_limits={1: (0.97, 1.05)},
        )
        assert result.num_violations == 1
        assert "< 0.97" in result.violation_details[0]

    def test_no_violation_with_limits_exactly_met(self):
        """Vm exactly at Vmax should not be flagged."""
        result = parse_opflow_output(
            _minimal_opflow_stdout(1.05),
            bus_limits={1: (0.95, 1.05)},
        )
        assert result.num_violations == 0

    def test_fallback_to_hardcoded_when_bus_not_in_limits(self):
        """If bus_limits provided but bus not in dict, fallback to 0.9/1.1."""
        result = parse_opflow_output(
            _minimal_opflow_stdout(1.06),
            bus_limits={99: (0.95, 1.05)},  # bus 1 not in dict
        )
        assert result.num_violations == 0  # 1.06 is within 0.9-1.1

    def test_parse_simulation_result_forwards_bus_limits(self):
        """parse_simulation_result passes bus_limits to parse_opflow_output."""
        class FakeResult:
            success = True
            stdout = _minimal_opflow_stdout(1.06)

        result = parse_simulation_result(FakeResult(), bus_limits={1: (0.95, 1.05)})
        assert result is not None
        assert result.num_violations == 1


# ===========================================================================
# Power balance violation checking
# ===========================================================================

def _minimal_opflow_stdout_with_gen(gen_mw: float, load_mw: float) -> str:
    """Build OPFLOW stdout with one bus (load) and one generator."""
    return (
        "Optimal Power Flow\n"
        "EXIT: Optimal Solution Found.\n"
        "Model                             POWER_BALANCE_POLAR\n"
        "Solver                            IPOPT\n"
        "Objective                         MIN_GEN_COST\n"
        "Convergence status                CONVERGED\n"
        "Objective value                   1000.00\n"
        "Bus        Pd      Pdloss Qd      Qdloss Vm      Va      mult_Pmis      mult_Qmis      Pslack         Qslack\n"
        "------------------------------------------------------------------------------------------------------\n"
        f"  1        {load_mw:.2f}    0.00    0.00    0.00   1.050   0.000         0.00         0.00         0.00         0.00\n"
        "------------------------------------------------------------------------------------------------------\n"
        "From       To       Status     Sft      Stf     Slim     mult_Sf  mult_St\n"
        "----------------------------------------------------------------------------------------\n"
        "1          2          1       55.00    55.00   100.00     0.00     0.00\n"
        "----------------------------------------------------------------------------------------\n"
        "Gen      Status     Fuel     Pg       Qg       Pmin     Pmax     Qmin     Qmax\n"
        "----------------------------------------------------------------------------------------\n"
        f"1          1        COAL     {gen_mw:.2f}    18.50    10.00   150.00   -50.00    50.00\n"
        "----------------------------------------------------------------------------------------\n"
    )


class TestPowerBalanceViolations:
    """Verify that negative losses (gen < load) are flagged as violations."""

    def test_positive_losses_no_violation(self):
        """Gen > Load should produce 0 power balance violations."""
        result = parse_opflow_output(
            _minimal_opflow_stdout_with_gen(gen_mw=150.0, load_mw=100.0),
        )
        assert result.losses_mw > 0
        assert result.power_balance_mismatch_pct > 0
        pb_violations = [
            v for v in result.violation_details if "Power balance" in v
        ]
        assert len(pb_violations) == 0

    def test_negative_losses_flagged(self):
        """Gen < Load should produce a power balance violation."""
        result = parse_opflow_output(
            _minimal_opflow_stdout_with_gen(gen_mw=75.0, load_mw=100.0),
        )
        assert result.losses_mw < 0
        assert result.power_balance_mismatch_pct < 0
        pb_violations = [
            v for v in result.violation_details if "Power balance" in v
        ]
        assert len(pb_violations) == 1
        assert "generation" in pb_violations[0]

    def test_zero_load_no_violation(self):
        """Zero load should not trigger power balance violation."""
        result = parse_opflow_output(
            _minimal_opflow_stdout_with_gen(gen_mw=0.0, load_mw=0.0),
        )
        assert result.losses_mw == 0.0
        pb_violations = [
            v for v in result.violation_details if "Power balance" in v
        ]
        assert len(pb_violations) == 0

    def test_violation_count_includes_power_balance(self):
        """num_violations should include power balance violations."""
        result = parse_opflow_output(
            _minimal_opflow_stdout_with_gen(gen_mw=50.0, load_mw=100.0),
        )
        pb_violations = [
            v for v in result.violation_details if "Power balance" in v
        ]
        assert len(pb_violations) == 1
        assert result.num_violations >= 1


# ===========================================================================
# IPOPT exit status and feasibility_detail
# ===========================================================================

def _minimal_opflow_stdout_with_convergence(convergence_status: str, exit_msg: str = "") -> str:
    """Build minimal OPFLOW stdout with specific convergence and EXIT messages."""
    bus_row = "  1    0.00    0.00    0.00    0.00    1.050    0.00    0.00    0.00    0.00    0.00"
    exit_section = f"EXIT: {exit_msg}\n" if exit_msg else ""
    return (
        f"{exit_section}"
        "Optimal Power Flow\n"
        f"Convergence status                {convergence_status}\n"
        "Objective value                   1000.00\n"
        "Model                             POWER_BALANCE_POLAR\n"
        "Solver                            IPOPT\n"
        "Bus      Pd      Pd      Qd      Qd      Vm      Va      mult_Pmis  mult_Qmis  Pslack  Qslack\n"
        "------------------------------------------------------------\n"
        f"{bus_row}\n"
        "------------------------------------------------------------\n"
    )


class TestIPOPTExitStatus:
    """Verify IPOPT exit status parsing and feasibility_detail classification."""

    def test_optimal_solution_is_feasible(self):
        result = parse_opflow_output(
            _minimal_opflow_stdout_with_convergence("CONVERGED", "Optimal Solution Found.")
        )
        assert result.converged is True
        assert result.ipopt_exit_status == "Optimal Solution Found."
        assert result.feasibility_detail == "feasible"

    def test_max_iterations_is_marginal(self):
        result = parse_opflow_output(
            _minimal_opflow_stdout_with_convergence(
                "DID NOT CONVERGE", "Maximum Number of Iterations Exceeded."
            )
        )
        assert result.converged is False
        assert result.ipopt_exit_status == "Maximum Number of Iterations Exceeded."
        assert result.feasibility_detail == "marginal"

    def test_infeasible_problem_detected(self):
        result = parse_opflow_output(
            _minimal_opflow_stdout_with_convergence(
                "DID NOT CONVERGE", "Infeasible Problem Detected."
            )
        )
        assert result.converged is False
        assert result.ipopt_exit_status == "Infeasible Problem Detected."
        assert result.feasibility_detail == "infeasible"

    def test_search_direction_too_small_is_marginal(self):
        result = parse_opflow_output(
            _minimal_opflow_stdout_with_convergence(
                "DID NOT CONVERGE", "Search Direction Becomes Too Small."
            )
        )
        assert result.converged is False
        assert result.feasibility_detail == "marginal"

    def test_no_exit_message_but_not_converged(self, sample_text):
        """Without EXIT line, non-converged is classified as infeasible."""
        modified = sample_text.replace(
            "EXIT: Optimal Solution Found.", ""
        ).replace(
            "Convergence status                  CONVERGED",
            "Convergence status                  DID NOT CONVERGE",
        )
        result = parse_opflow_output(modified)
        assert result.converged is False
        assert result.feasibility_detail == "infeasible"

    def test_convergence_status_full_did_not_converge(self, sample_text):
        """Regex now captures full 'DID NOT CONVERGE' instead of just 'DID'."""
        modified = sample_text.replace(
            "Convergence status                  CONVERGED",
            "Convergence status                  DID NOT CONVERGE",
        ).replace(
            "EXIT: Optimal Solution Found.",
            "EXIT: Maximum Number of Iterations Exceeded.",
        )
        result = parse_opflow_output(modified)
        # Strip trailing whitespace
        assert result.convergence_status.strip() == "DID NOT CONVERGE"
        assert result.feasibility_detail == "marginal"

    def test_near_boundary_voltage_classified_marginal(self):
        """Non-converged with 'Infeasible Problem Detected' but voltage near limit → marginal."""
        bus_row = "  1    0.00    0.00    0.00    0.00    0.951    0.00    0.00    0.00    0.00    0.00"
        output = (
            "EXIT: Infeasible Problem Detected.\n"
            "Optimal Power Flow\n"
            "Convergence status                DID NOT CONVERGE\n"
            "Objective value                   1000.00\n"
            "Model                             POWER_BALANCE_POLAR\n"
            "Solver                            IPOPT\n"
            "Bus      Pd      Pd      Qd      Qd      Vm      Va      mult_Pmis  mult_Qmis  Pslack  Qslack\n"
            "------------------------------------------------------------\n"
            f"{bus_row}\n"
            "------------------------------------------------------------\n"
        )
        bus_limits = {1: (0.95, 1.05)}
        result = parse_opflow_output(output, bus_limits=bus_limits)
        assert result.converged is False
        assert result.feasibility_detail == "marginal"

    def test_near_boundary_line_loading_classified_marginal(self):
        """Non-converged with line loading near 100% → marginal."""
        bus_row = "  1    0.00    0.00    0.00    0.00    1.050    0.00    0.00    0.00    0.00    0.00"
        output = (
            "EXIT: Infeasible Problem Detected.\n"
            "Optimal Power Flow\n"
            "Convergence status                DID NOT CONVERGE\n"
            "Objective value                   1000.00\n"
            "Model                             POWER_BALANCE_POLAR\n"
            "Solver                            IPOPT\n"
            "Bus      Pd      Pd      Qd      Qd      Vm      Va      mult_Pmis  mult_Qmis  Pslack  Qslack\n"
            "------------------------------------------------------------\n"
            f"{bus_row}\n"
            "------------------------------------------------------------\n"
            "From     To      Status     Sft      Stf     Slim     mult_Sf  mult_St\n"
            "------------------------------------------------------------\n"
            "  1        2         1      95.00    95.00    100.00    0.00   0.00\n"
            "------------------------------------------------------------\n"
        )
        result = parse_opflow_output(output)
        assert result.converged is False
        assert result.feasibility_detail == "marginal"

    def test_far_from_boundary_classified_infeasible(self):
        """Non-converged with metrics far from limits → infeasible."""
        bus_row = "  1    0.00    0.00    0.00    0.00    0.850    0.00    0.00    0.00    0.00    0.00"
        output = (
            "EXIT: Infeasible Problem Detected.\n"
            "Optimal Power Flow\n"
            "Convergence status                DID NOT CONVERGE\n"
            "Objective value                   1000.00\n"
            "Model                             POWER_BALANCE_POLAR\n"
            "Solver                            IPOPT\n"
            "Bus      Pd      Pd      Qd      Qd      Vm      Va      mult_Pmis  mult_Qmis  Pslack  Qslack\n"
            "------------------------------------------------------------\n"
            f"{bus_row}\n"
            "------------------------------------------------------------\n"
        )
        result = parse_opflow_output(output)
        assert result.converged is False
        assert result.feasibility_detail == "infeasible"
