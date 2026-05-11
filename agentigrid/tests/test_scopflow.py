"""Tests for SCOPFLOW support (Phase 3, Step 3.2)."""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from agentigrid.parsers.scopflow_parser import parse_scopflow_output, parse_scopflow_simulation_result
from agentigrid.parsers.scopflow_summary import scopflow_results_summary
from agentigrid.parsers import (
    results_summary_for_app,
    parse_simulation_result_for_app,
    parse_scopflow_metadata,
)
from agentigrid.parsers.opflow_parser import parse_opflow_output
from agentigrid.config import SearchConfig


# ---------------------------------------------------------------------------
# Synthetic SCOPFLOW output — 3-bus system
# Same bus/branch/gen table format as OPFLOW but with the SCOPFLOW header
# and "Objective value (base)" format. AC OPF: realistic Vm, non-zero Qg/Qd.
# ---------------------------------------------------------------------------

SAMPLE_SCOPFLOW_OUTPUT = """\


******************************************************************************
This program contains Ipopt, a library for large-scale nonlinear optimization.
 Ipopt is released as open source code under the Eclipse Public License (EPL).
         For more information visit https://github.com/coin-or/Ipopt
******************************************************************************

This is Ipopt version 3.14.20, running with linear solver ma27.

Number of Iterations....: 8

Total seconds in IPOPT                               = 0.021

EXIT: Optimal Solution Found.
=============================================================
\tSecurity-Constrained Optimal Power Flow
=============================================================
Number of contingencies         5
Multi-period contingencies?     NO
Solver                          IPOPT
Initialization                  ACPF
Load loss allowed               NO
Power imbalance allowed         NO
Ignore line flow constraints    NO

Convergence status              CONVERGED
Objective value (base)          28500.00

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

SAMPLE_OPFLOW_OUTPUT = """\
EXIT: Optimal Solution Found.
=============================================================
\tOptimal Power Flow
=============================================================
Model                               POWER_BALANCE_POLAR
Solver                              IPOPT
Objective                           MIN_GEN_COST
Initialization                      ACPF

Convergence status                  CONVERGED
Objective value                     27557.57

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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def scopflow_result_and_meta():
    """Parse the synthetic SCOPFLOW output."""
    return parse_scopflow_output(SAMPLE_SCOPFLOW_OUTPUT)


@pytest.fixture
def scopflow_result(scopflow_result_and_meta):
    return scopflow_result_and_meta[0]


@pytest.fixture
def scopflow_meta(scopflow_result_and_meta):
    return scopflow_result_and_meta[1]


@pytest.fixture
def mock_sim_result():
    """A mock SimulationResult for testing dispatch functions."""
    sr = MagicMock()
    sr.success = True
    sr.stdout = SAMPLE_SCOPFLOW_OUTPUT
    return sr


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------

class TestSCOPFLOWParser:
    """Verify SCOPFLOW parser extracts metadata and delegates table parsing."""

    def test_parse_succeeds(self, scopflow_result_and_meta):
        """Should parse without error."""
        result, meta = scopflow_result_and_meta
        assert result is not None
        assert meta is not None

    def test_num_contingencies(self, scopflow_meta):
        """Should extract number of contingencies from header."""
        assert scopflow_meta["num_contingencies"] == 5

    def test_multi_period_flag(self, scopflow_meta):
        """Should extract multi-period flag."""
        assert scopflow_meta["multi_period"] is False

    def test_objective_value(self, scopflow_result):
        """Should parse 'Objective value (base)' format."""
        assert scopflow_result.objective_value == pytest.approx(28500.00, abs=0.01)

    def test_convergence(self, scopflow_result):
        """Should parse convergence status."""
        assert scopflow_result.convergence_status == "CONVERGED"
        assert scopflow_result.converged is True

    def test_bus_count(self, scopflow_result):
        """Should parse 3 buses."""
        assert len(scopflow_result.buses) == 3

    def test_branch_count(self, scopflow_result):
        """Should parse 2 branches."""
        assert len(scopflow_result.branches) == 2

    def test_gen_count(self, scopflow_result):
        """Should parse 2 generators."""
        assert len(scopflow_result.generators) == 2

    def test_rejects_non_scopflow(self):
        """Should raise ValueError for non-SCOPFLOW output."""
        with pytest.raises(ValueError, match="SCOPFLOW"):
            parse_scopflow_output("This is not SCOPFLOW output")

    def test_rejects_plain_opflow(self):
        """Should raise ValueError for plain OPFLOW output (no 'Security-Constrained' in header)."""
        with pytest.raises(ValueError, match="SCOPFLOW"):
            parse_scopflow_output(SAMPLE_OPFLOW_OUTPUT)

    def test_voltages_non_unity(self, scopflow_result):
        """AC OPF: bus voltages should not all be 1.0 pu."""
        vms = [b.Vm for b in scopflow_result.buses]
        assert any(abs(vm - 1.0) > 0.01 for vm in vms)

    def test_reactive_non_zero(self, scopflow_result):
        """AC OPF: generators should have non-zero Qg."""
        qgs = [g.Qg for g in scopflow_result.generators]
        assert any(abs(qg) > 0.1 for qg in qgs)


class TestSCOPFLOWSummary:
    """Test SCOPFLOW-specific results summary."""

    def test_header_mentions_security(self, scopflow_result):
        """Summary should identify as security-constrained."""
        summary = scopflow_results_summary(scopflow_result, num_contingencies=5)
        assert "Security-Constrained" in summary

    def test_contingency_count_shown(self, scopflow_result):
        """Summary should show number of contingencies."""
        summary = scopflow_results_summary(scopflow_result, num_contingencies=5)
        assert "5" in summary
        assert "ontingenc" in summary  # "Contingencies" or "contingencies"

    def test_has_voltage_profile(self, scopflow_result):
        """Summary should include voltage analysis (AC OPF)."""
        summary = scopflow_results_summary(scopflow_result, num_contingencies=5)
        assert "Voltage profile" in summary

    def test_has_preventive_dispatch_note(self, scopflow_result):
        """Summary should mention preventive dispatch."""
        summary = scopflow_results_summary(scopflow_result)
        assert "preventive dispatch" in summary.lower() or "preventive" in summary.lower()

    def test_has_generation_info(self, scopflow_result):
        """Summary should include generation/load information."""
        summary = scopflow_results_summary(scopflow_result)
        assert "Generation" in summary
        assert "Load" in summary

    def test_has_line_loading(self, scopflow_result):
        """Summary should include line loading section."""
        summary = scopflow_results_summary(scopflow_result)
        assert "loaded" in summary.lower()

    def test_default_zero_contingencies(self, scopflow_result):
        """Default num_contingencies=0 should not crash."""
        summary = scopflow_results_summary(scopflow_result)
        assert "SCOPFLOW Results" in summary


class TestSCOPFLOWDispatch:
    """Test application-aware dispatch for SCOPFLOW."""

    def test_dispatch_scopflow_summary(self, scopflow_result):
        """results_summary_for_app('scopflow') should use SCOPFLOW summary."""
        summary = results_summary_for_app(scopflow_result, "scopflow", num_contingencies=5)
        assert "SCOPFLOW Results" in summary
        assert "Security-Constrained" in summary

    def test_dispatch_scopflow_with_num_contingencies(self, scopflow_result):
        """num_contingencies kwarg should be forwarded to scopflow_results_summary."""
        summary = results_summary_for_app(scopflow_result, "scopflow", num_contingencies=195)
        assert "195" in summary

    def test_parse_dispatch_scopflow(self, mock_sim_result):
        """parse_simulation_result_for_app('scopflow') should return OPFLOWResult."""
        result = parse_simulation_result_for_app(mock_sim_result, "scopflow")
        assert result is not None
        assert result.objective_value == pytest.approx(28500.00, abs=0.01)

    def test_parse_dispatch_scopflow_metadata(self, mock_sim_result):
        """parse_scopflow_metadata() should return dict with num_contingencies."""
        meta = parse_scopflow_metadata(mock_sim_result)
        assert meta is not None
        assert meta["num_contingencies"] == 5
        assert meta["multi_period"] is False

    def test_parse_scopflow_simulation_result_failed(self):
        """parse_scopflow_simulation_result should return None if sim failed."""
        sr = MagicMock()
        sr.success = False
        result = parse_scopflow_simulation_result(sr)
        assert result is None


class TestOPFLOWParserRegression:
    """Verify the _OBJ_VALUE_RE regex change didn't break OPFLOW parsing."""

    def test_opflow_objective_still_parsed(self):
        """Standard OPFLOW 'Objective value  27557.57' still works."""
        result = parse_opflow_output(SAMPLE_OPFLOW_OUTPUT)
        assert result.objective_value == pytest.approx(27557.57, abs=0.01)

    def test_scopflow_objective_format_also_works(self):
        """SCOPFLOW 'Objective value (base)  28500.00' parsed by the same regex."""
        # The OPFLOW parser is called internally by parse_scopflow_output,
        # so also verify it can handle SCOPFLOW-format text that contains the string.
        result, _ = parse_scopflow_output(SAMPLE_SCOPFLOW_OUTPUT)
        assert result.objective_value == pytest.approx(28500.00, abs=0.01)


class TestSCOPFLOWConfig:
    """Test config support for contingency files."""

    def test_ctgc_file_in_config(self):
        """SearchConfig should accept ctgc_file."""
        cfg = SearchConfig(
            max_iterations=10,
            default_mode="accumulative",
            base_case=None,
            gic_file=None,
            application="scopflow",
            ctgc_file=None,
        )
        assert cfg.ctgc_file is None

    def test_ctgc_file_stored(self, tmp_path):
        """ctgc_file should be stored as provided."""
        p = tmp_path / "test.cont"
        p.touch()
        cfg = SearchConfig(
            max_iterations=10,
            default_mode="accumulative",
            base_case=None,
            gic_file=None,
            application="scopflow",
            ctgc_file=p,
        )
        assert cfg.ctgc_file == p

    def test_ctgc_file_default_is_none(self):
        """ctgc_file should default to None (backward compatible)."""
        cfg = SearchConfig(
            max_iterations=10,
            default_mode="accumulative",
            base_case=None,
            gic_file=None,
            application="opflow",
        )
        assert cfg.ctgc_file is None


# ---------------------------------------------------------------------------
# Synthetic SCOPFLOW EMPAR output (no Ipopt EXIT message)
# ---------------------------------------------------------------------------

SAMPLE_SCOPFLOW_EMPAR_CONVERGED = """\
=============================================================
\tSecurity-Constrained Optimal Power Flow
=============================================================
Number of contingencies         5
Multi-period contingencies?     NO
Solver                          EMPAR
Initialization                  ACPF
Load loss allowed               NO
Power imbalance allowed         NO
Ignore line flow constraints    NO

Convergence status              CONVERGED
Objective value (base)          28500.00

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

SAMPLE_SCOPFLOW_EMPAR_DID_NOT_CONVERGE = """\
=============================================================
\tSecurity-Constrained Optimal Power Flow
=============================================================
Number of contingencies         5
Multi-period contingencies?     NO
Solver                          EMPAR
Initialization                  ACPF
Load loss allowed               NO
Power imbalance allowed         NO
Ignore line flow constraints    NO

Convergence status              DID NOT CONVERGE
Objective value (base)          28500.00

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


class TestSCOPFLOWEMPARParsing:
    """EMPAR solver does not emit the Ipopt EXIT line; converged must still
    be set correctly from the SCOPFLOW Convergence status field."""

    def test_empar_no_exit_message(self):
        """CONVERGED in header should yield converged=True even without EXIT line."""
        assert "EXIT: Optimal Solution Found." not in SAMPLE_SCOPFLOW_EMPAR_CONVERGED
        result, _ = parse_scopflow_output(SAMPLE_SCOPFLOW_EMPAR_CONVERGED)
        assert result.converged is True
        assert result.convergence_status == "CONVERGED"

    def test_empar_did_not_converge(self):
        """'DID NOT CONVERGE' in header should yield converged=False."""
        assert "EXIT:" not in SAMPLE_SCOPFLOW_EMPAR_DID_NOT_CONVERGE
        result, _ = parse_scopflow_output(SAMPLE_SCOPFLOW_EMPAR_DID_NOT_CONVERGE)
        assert result.converged is False
        assert result.convergence_status == "DID NOT CONVERGE"

    def test_original_scopflow_with_exit_still_works(self):
        """Regression: original SCOPFLOW output (with EXIT line) still converged=True."""
        assert "EXIT: Optimal Solution Found." in SAMPLE_SCOPFLOW_OUTPUT
        result, _ = parse_scopflow_output(SAMPLE_SCOPFLOW_OUTPUT)
        assert result.converged is True
        assert result.convergence_status == "CONVERGED"
        assert result.feasibility_detail == "feasible"

    def test_empar_converged_feasibility_detail(self):
        """EMPAR CONVERGED with positive losses should be feasible."""
        result, _ = parse_scopflow_output(SAMPLE_SCOPFLOW_EMPAR_CONVERGED)
        assert result.feasibility_detail == "feasible"

    def test_empar_did_not_converge_feasibility_detail(self):
        """EMPAR DID NOT CONVERGE should be classified as infeasible (no EXIT status)."""
        result, _ = parse_scopflow_output(SAMPLE_SCOPFLOW_EMPAR_DID_NOT_CONVERGE)
        assert result.convergence_status == "DID NOT CONVERGE"
        assert result.feasibility_detail == "infeasible"

    def test_ipopt_scopflow_marginal(self):
        """SCOPFLOW with IPOPT max iterations exceeded should be marginal."""
        marginal_output = SAMPLE_SCOPFLOW_OUTPUT.replace(
            "EXIT: Optimal Solution Found.",
            "EXIT: Maximum Number of Iterations Exceeded.",
        ).replace(
            "Convergence status              CONVERGED",
            "Convergence status              DID NOT CONVERGE",
        )
        result, _ = parse_scopflow_output(marginal_output)
        assert result.convergence_status == "DID NOT CONVERGE"
        assert result.feasibility_detail == "marginal"
        assert result.converged is False
