"""Data model for parsed OPFLOW results."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BusResult:
    """Bus-level results from OPFLOW."""

    bus_id: int
    Pd: float
    Pd_loss: float
    Qd: float
    Qd_loss: float
    Vm: float
    Va: float
    mult_Pmis: float
    mult_Qmis: float
    Pslack: float
    Qslack: float


@dataclass
class BranchResult:
    """Branch-level results from OPFLOW."""

    from_bus: int
    to_bus: int
    status: int
    Sf: float
    St: float
    Slim: float
    mult_Sf: float
    mult_St: float


@dataclass
class GenResult:
    """Generator-level results from OPFLOW."""

    bus: int
    status: int
    fuel: str
    Pg: float
    Qg: float
    Pmin: float
    Pmax: float
    Qmin: float
    Qmax: float


@dataclass
class OPFLOWResult:
    """Complete parsed OPFLOW results."""

    # Convergence and objective
    converged: bool
    objective_value: float
    convergence_status: str

    # Solver metadata
    solver: str
    model: str
    objective_type: str
    num_iterations: int
    solve_time: float

    # Detailed results
    buses: list[BusResult] = field(default_factory=list)
    branches: list[BranchResult] = field(default_factory=list)
    generators: list[GenResult] = field(default_factory=list)

    # Derived summary metrics
    total_gen_mw: float = 0.0
    total_load_mw: float = 0.0
    total_gen_mvar: float = 0.0
    total_load_mvar: float = 0.0
    voltage_min: float = 0.0
    voltage_max: float = 0.0
    voltage_mean: float = 0.0
    max_line_loading_pct: float = 0.0
    num_violations: int = 0
    violation_details: list[str] = field(default_factory=list)
    losses_mw: float = 0.0
    power_balance_mismatch_pct: float = 0.0

    # IPOPT exit status and feasibility classification
    # ipopt_exit_status: full EXIT message from IPOPT (e.g. "Optimal Solution Found.")
    #   Empty string if no EXIT line found (EMPAR solver or missing).
    # feasibility_detail: one of "feasible", "infeasible", "marginal"
    #   feasible   = converged AND no power balance violation
    #   infeasible = NOT converged OR power balance violation (negative losses)
    #   marginal   = NOT converged but no structural violations, solution may be usable
    ipopt_exit_status: str = ""
    feasibility_detail: str = ""

    def compute_generation_cost(self, gencost: list) -> float:
        """Compute total generation cost from dispatch and cost curves.

        Uses the MATPOWER polynomial cost model (model=2):
          cost = c2*Pg^2 + c1*Pg + c0
        where coefficients are ordered as [c2, c1, c0] (highest degree first).

        Args:
            gencost: List of GenCost objects aligned with generators by index.

        Returns:
            Total generation cost, or 0.0 if gencost data is unavailable.
        """
        if not gencost or not self.generators:
            return 0.0

        total_cost = 0.0
        for i, gen in enumerate(self.generators):
            if gen.status != 1 or i >= len(gencost):
                continue
            cost_entry = gencost[i]
            if cost_entry.model != 2 or len(cost_entry.coeffs) < 3:
                continue
            c2, c1, c0 = cost_entry.coeffs[0], cost_entry.coeffs[1], cost_entry.coeffs[2]
            pg = gen.Pg
            total_cost += c2 * pg ** 2 + c1 * pg + c0

        return total_cost
