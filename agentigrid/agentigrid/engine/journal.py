"""Search journal — records the trajectory of the LLM-driven search."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from agentigrid.parsers.opflow_results import OPFLOWResult

logger = logging.getLogger("agentigrid.engine.journal")


@dataclass
class JournalEntry:
    """One iteration's record in the search journal."""

    iteration: int
    description: str
    commands: list[dict]
    objective_value: Optional[float]
    feasible: bool
    convergence_status: str
    violations_count: int
    voltage_min: float
    voltage_max: float
    max_line_loading_pct: float
    total_gen_mw: float
    total_load_mw: float
    llm_reasoning: str
    mode: str
    elapsed_seconds: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    steering_directive: Optional[str] = None  # Active steering directive at this iteration
    tracked_metrics: Optional[dict[str, float]] = None  # Multi-objective metric values
    feasibility_detail: str = ""  # "feasible", "infeasible", or "marginal"
    solver: str = ""  # Solver used (IPOPT, EMPAR, etc.)
    num_steps: int = 0  # TCOPFLOW: number of time periods
    num_scenarios: int = 0  # SOPFLOW: number of wind scenarios
    explored_variants: Optional[list[dict]] = None  # Explore/select: companion variants
    candidate_count: int = 0  # Sweep: total number of candidates tested
    feasible_buses: Optional[list[int]] = None  # Sweep: list of feasible bus ids
    exago_command: Optional[dict] = None  # Reproducible ExaGO invocation record (JSON journal only; see add_* methods)
    contingency_meta: Optional[dict] = None  # C.5: target bus, neighbors, order, pass/fail counts
    reserve_meta: Optional[dict] = None  # C.8: hot-reserve / N-1 generator security accounting


@dataclass
class ObjectiveEntry:
    """A tracked objective in the multi-objective registry."""

    name: str                          # e.g. "generation_cost", "max_voltage_deviation"
    direction: str                     # "minimize", "maximize", or "constraint"
    threshold: Optional[float] = None  # For constraint-type objectives (e.g., ≤ 0.85)
    priority: str = "primary"          # "primary", "secondary", or "watch"
    introduced_at: int = 0             # Iteration when this objective was registered
    source: str = "initial"            # "initial", "steering", or "llm_proposed"


class ObjectiveRegistry:
    """Tracks which objectives the search is optimizing for."""

    def __init__(self) -> None:
        self._objectives: list[ObjectiveEntry] = []
        self._history: list[dict] = []  # Records all add/update/reprioritize events

    def register(self, objective: ObjectiveEntry) -> None:
        """Register a new objective. If an objective with the same name exists, update it."""
        for i, existing in enumerate(self._objectives):
            if existing.name == objective.name:
                self._history.append({
                    "action": "updated",
                    "iteration": objective.introduced_at,
                    "name": objective.name,
                    "old_priority": existing.priority,
                    "new_priority": objective.priority,
                    "source": objective.source,
                })
                self._objectives[i] = objective
                return
        self._objectives.append(objective)
        self._history.append({
            "action": "registered",
            "iteration": objective.introduced_at,
            "name": objective.name,
            "direction": objective.direction,
            "priority": objective.priority,
            "source": objective.source,
        })

    def reprioritize(self, name: str, new_priority: str, iteration: int, source: str = "steering") -> bool:
        """Change the priority of an existing objective. Returns True if found."""
        for obj in self._objectives:
            if obj.name == name:
                old = obj.priority
                obj.priority = new_priority
                self._history.append({
                    "action": "reprioritized",
                    "iteration": iteration,
                    "name": name,
                    "old_priority": old,
                    "new_priority": new_priority,
                    "source": source,
                })
                return True
        return False

    @property
    def objectives(self) -> list[ObjectiveEntry]:
        return list(self._objectives)

    @property
    def history(self) -> list[dict]:
        return list(self._history)

    @property
    def is_multi_objective(self) -> bool:
        """True if more than one objective is registered (excluding 'watch' priority)."""
        active = [o for o in self._objectives if o.priority != "watch"]
        return len(active) > 1

    def get_primary(self) -> list[ObjectiveEntry]:
        return [o for o in self._objectives if o.priority == "primary"]

    def get_secondary(self) -> list[ObjectiveEntry]:
        return [o for o in self._objectives if o.priority == "secondary"]

    def format_for_prompt(self) -> str:
        """Format the registry for injection into the LLM prompt."""
        if not self._objectives:
            return ""
        lines = ["Tracked Objectives:"]
        for obj in self._objectives:
            dir_label = obj.direction
            if obj.direction == "constraint" and obj.threshold is not None:
                dir_label = f"constraint (\u2264 {obj.threshold})"
            lines.append(
                f"  - {obj.name} [{dir_label}] priority={obj.priority}"
                f" (since iter {obj.introduced_at}, source: {obj.source})"
            )
        return "\n".join(lines)

    def to_dict_list(self) -> list[dict]:
        """Serialize for JSON export."""
        return [asdict(o) for o in self._objectives]


class SearchJournal:
    """Maintains the search trajectory across iterations."""

    def __init__(self) -> None:
        self._entries: list[JournalEntry] = []
        self.objective_registry = ObjectiveRegistry()
        self.benchmark_result: Optional[dict] = None
        # session_best: lowest-cost feasible variant ever found across all
        # explore batches, including non-selected variants.
        self.session_best: Optional[dict] = None
        # load_factor: session-level load scaling factor (updated by set_load_factor action).
        self.load_factor: Optional[float] = None

    def update_session_best(
        self,
        label: str,
        iteration: int,
        cost: float,
        commands: list[dict],
    ) -> None:
        """Update the session-best record if cost is lower than the current best.

        Args:
            label: Variant label (e.g. "A").
            iteration: Explore-iteration number that produced this variant.
            cost: Computed generation cost (must be > 0 to be recorded).
            commands: Raw command list for this variant.
        """
        if cost <= 0:
            return
        if self.session_best is None or cost < self.session_best["cost"]:
            self.session_best = {
                "cost": cost,
                "iteration": iteration,
                "variant_label": label,
                "commands": commands,
            }

    def add_entry(self, entry: JournalEntry) -> None:
        """Append an entry to the journal."""
        self._entries.append(entry)

    def add_from_results(
        self,
        iteration: int,
        description: str,
        commands: list[dict],
        opflow_result: Optional[OPFLOWResult],
        sim_elapsed: float,
        llm_reasoning: str,
        mode: str,
        steering_directive: Optional[str] = None,
        num_steps: int = 0,
        num_scenarios: int = 0,
        explored_variants: Optional[list[dict]] = None,
        gencost: Optional[list] = None,
        exago_command: Optional[dict] = None,
    ) -> JournalEntry:
        """Create and append a journal entry from OPFLOW results.

        If opflow_result is None (simulation failed), fills in defaults
        indicating failure. Returns the created entry.

        For PFLOW (where the solver does not produce an objective value),
        pass the network's gencost data: the entry's objective_value and
        tracked_metrics["generation_cost"] will be populated from the
        computed Σ(c2·Pg² + c1·Pg + c0) across online generators.
        """
        if opflow_result is not None:
            # Determine feasible flag: use feasibility_detail if available,
            # otherwise fall back to converged + no power balance violation.
            # IMPORTANT: also require num_violations == 0 — thermal violations
            # (line overloads) set violations_count > 0 but do NOT change
            # feasibility_detail from "feasible". Without this guard, states
            # with active thermal violations are incorrectly classified as
            # feasible and accepted into session_best.
            if opflow_result.feasibility_detail:
                feasible = (
                    opflow_result.feasibility_detail == "feasible"
                    and opflow_result.num_violations == 0
                )
            else:
                has_power_balance_violation = (
                    opflow_result.losses_mw < 0 and opflow_result.total_load_mw > 0
                )
                feasible = opflow_result.converged and not has_power_balance_violation
            # Determine feasibility_detail for the journal entry
            if opflow_result.feasibility_detail:
                feasibility_detail = opflow_result.feasibility_detail
            else:
                has_power_balance_violation = (
                    opflow_result.losses_mw < 0 and opflow_result.total_load_mw > 0
                )
                if feasible:
                    feasibility_detail = "feasible"
                elif has_power_balance_violation:
                    feasibility_detail = "infeasible"
                else:
                    feasibility_detail = "infeasible"

            # Compute the entry's objective_value. PFLOW has no native
            # objective: when gencost is supplied, derive cost from the
            # dispatch instead of accepting the placeholder 0.0. For other
            # applications, pass through opflow_result.objective_value.
            if gencost is not None:
                try:
                    computed_cost = opflow_result.compute_generation_cost(gencost)
                except Exception:
                    computed_cost = None
                # 0.0 from compute_generation_cost means "unable to compute"
                # (no gencost data, or all generators offline). Treat as None
                # rather than the misleading sentinel.
                objective_value = computed_cost if computed_cost else None
            else:
                objective_value = opflow_result.objective_value

            # Tracked metrics: ensure generation_cost is populated for PFLOW
            tracked_metrics: Optional[dict[str, float]] = None
            if gencost is not None and objective_value is not None:
                tracked_metrics = {"generation_cost": objective_value}

            entry = JournalEntry(
                iteration=iteration,
                description=description,
                commands=commands,
                objective_value=objective_value,
                feasible=feasible,
                convergence_status=opflow_result.convergence_status,
                violations_count=opflow_result.num_violations,
                voltage_min=opflow_result.voltage_min,
                voltage_max=opflow_result.voltage_max,
                max_line_loading_pct=opflow_result.max_line_loading_pct,
                total_gen_mw=opflow_result.total_gen_mw,
                total_load_mw=opflow_result.total_load_mw,
                llm_reasoning=llm_reasoning,
                mode=mode,
                elapsed_seconds=sim_elapsed,
                steering_directive=steering_directive,
                feasibility_detail=feasibility_detail,
                solver=opflow_result.solver,
                num_steps=num_steps,
                num_scenarios=num_scenarios,
                explored_variants=explored_variants,
                tracked_metrics=tracked_metrics,
                exago_command=exago_command,
            )
        else:
            entry = JournalEntry(
                iteration=iteration,
                description=description,
                commands=commands,
                objective_value=None,
                feasible=False,
                convergence_status="FAILED",
                violations_count=0,
                voltage_min=0.0,
                voltage_max=0.0,
                max_line_loading_pct=0.0,
                total_gen_mw=0.0,
                total_load_mw=0.0,
                llm_reasoning=llm_reasoning,
                mode=mode,
                elapsed_seconds=sim_elapsed,
                steering_directive=steering_directive,
                feasibility_detail="infeasible",
                num_steps=num_steps,
                num_scenarios=num_scenarios,
                explored_variants=explored_variants,
                exago_command=exago_command,
            )

        self._entries.append(entry)
        return entry

    def add_explore(
        self,
        iteration: int,
        description: str,
        variant_info: list[dict],
        pareto_labels: list[str] | None = None,
        llm_reasoning: str = "",
        steering_directive: str | None = None,
        exago_command: Optional[dict] = None,
    ) -> JournalEntry:
        """Record an 'explore' action in the journal.

        Creates a lightweight entry (no simulation data for the explore itself)
        with explored_variants metadata so the search history shows what
        parameter ranges were tested.
        """
        entry = JournalEntry(
            iteration=iteration,
            description=description,
            commands=[],
            objective_value=None,
            feasible=False,
            convergence_status="EXPLORE",
            violations_count=0,
            voltage_min=0.0,
            voltage_max=0.0,
            max_line_loading_pct=0.0,
            total_gen_mw=0.0,
            total_load_mw=0.0,
            llm_reasoning=llm_reasoning,
            mode="explore",
            elapsed_seconds=0.0,
            steering_directive=steering_directive,
            feasibility_detail="",
            explored_variants=variant_info,
            exago_command=exago_command,
        )
        self._entries.append(entry)
        return entry

    def add_sweep(
        self,
        iteration: int,
        description: str,
        candidate_count: int,
        candidate_summaries: list[dict],
        feasible_buses: list[int],
        llm_reasoning: str = "",
        steering_directive: Optional[str] = None,
        exago_command: Optional[dict] = None,
    ) -> JournalEntry:
        """Record a 'sweep' action in the journal.

        Creates a lightweight entry (no single simulation result) with
        per-candidate summaries stored in explored_variants.
        """
        entry = JournalEntry(
            iteration=iteration,
            description=description,
            commands=[],
            objective_value=None,
            feasible=len(feasible_buses) > 0,
            convergence_status="SWEEP",
            violations_count=0,
            voltage_min=0.0,
            voltage_max=0.0,
            max_line_loading_pct=0.0,
            total_gen_mw=0.0,
            total_load_mw=0.0,
            llm_reasoning=llm_reasoning,
            mode="sweep",
            elapsed_seconds=0.0,
            steering_directive=steering_directive,
            feasibility_detail="",
            explored_variants=candidate_summaries,
            candidate_count=candidate_count,
            feasible_buses=feasible_buses,
            exago_command=exago_command,
        )
        self._entries.append(entry)
        return entry

    def add_contingency(
        self,
        iteration: int,
        description: str,
        target_bus: int,
        neighbors: list[tuple[int, int]],
        order: int,
        contingency_summaries: list[dict],
        passed_count: int,
        failed_count: int,
        llm_reasoning: str = "",
        steering_directive: Optional[str] = None,
        exago_command: Optional[dict] = None,
    ) -> JournalEntry:
        """Record a 'contingency' screen in the journal (C.5).

        Modeled on ``add_sweep``: a lightweight entry whose ``explored_variants``
        holds the full per-contingency pass/fail summaries and whose
        ``contingency_meta`` captures the reproducible study parameters (target
        bus, resolved neighbors, order, and pass/fail counts) so the JSON journal
        is a complete, reproducible record of the N-1/N-2 study.

        When a C.7 relief phase ran, each FAILED contingency summary carries a
        ``relief`` payload ({resolved, measure, detail, attempts}); these ride
        along inside ``contingency_summaries`` and are therefore persisted here.
        """
        entry = JournalEntry(
            iteration=iteration,
            description=description,
            commands=[],
            objective_value=None,
            feasible=passed_count > 0,
            convergence_status="CONTINGENCY",
            violations_count=0,
            voltage_min=0.0,
            voltage_max=0.0,
            max_line_loading_pct=0.0,
            total_gen_mw=0.0,
            total_load_mw=0.0,
            llm_reasoning=llm_reasoning,
            mode="contingency",
            elapsed_seconds=0.0,
            steering_directive=steering_directive,
            feasibility_detail="",
            explored_variants=contingency_summaries,
            candidate_count=len(contingency_summaries),
            feasible_buses=[],
            exago_command=exago_command,
            contingency_meta={
                "target_bus": target_bus,
                "neighbors": [[nb, hop] for nb, hop in neighbors],
                "order": order,
                "passed_count": passed_count,
                "failed_count": failed_count,
            },
        )
        self._entries.append(entry)
        return entry

    def add_reserve(
        self,
        iteration: int,
        description: str,
        reserve_meta: dict,
        contingency_summaries: list[dict],
        llm_reasoning: str = "",
        steering_directive: Optional[str] = None,
        exago_command: Optional[dict] = None,
    ) -> JournalEntry:
        """Record a hot-reserve / N-1 generator security screen in the journal (C.8).

        Modeled on ``add_contingency``: a lightweight entry whose
        ``explored_variants`` holds the per-generator N-1 pass/fail summaries. The
        ``reserve_meta`` dict carries the reserve accounting (available reserve,
        largest committed unit, required N-1 reserve, margin, N-1-secure flag, and
        pass/fail counts). When the run minimized reserve (C.8 Path A) it also
        carries the greedy de-commitment result: ``minimize`` (True), ``reserve_full``,
        ``min_reserve``, ``n_decommitted``, ``decommitted``, ``final_on_count``,
        ``lower_bound_pg``, ``lower_bound_bus``, ``solves_used``, and ``hit_budget``.

        ``convergence_status`` is ``"CONTINGENCY"`` so the entry routes through the
        contingency-aware PDF report path (C.5b); the report keys off
        ``reserve_meta`` to render the Hot Reserve Assessment block.
        """
        entry = JournalEntry(
            iteration=iteration,
            description=description,
            commands=[],
            objective_value=None,
            feasible=reserve_meta.get("n1_secure", False),
            convergence_status="CONTINGENCY",
            violations_count=0,
            voltage_min=0.0,
            voltage_max=0.0,
            max_line_loading_pct=0.0,
            total_gen_mw=0.0,
            total_load_mw=0.0,
            llm_reasoning=llm_reasoning,
            mode="reserve",
            elapsed_seconds=0.0,
            steering_directive=steering_directive,
            feasibility_detail="",
            explored_variants=contingency_summaries,
            candidate_count=len(contingency_summaries),
            feasible_buses=[],
            exago_command=exago_command,
            reserve_meta=reserve_meta,
        )
        self._entries.append(entry)
        return entry

    def add_analysis(
        self,
        iteration: int,
        query: str,
        result_summary: str,
    ) -> JournalEntry:
        """Record an 'analyze' action in the journal.

        Creates a lightweight entry (no simulation data) so analyze actions
        appear in the search history.
        """
        entry = JournalEntry(
            iteration=iteration,
            description=f"Analysis: {query[:80]}",
            commands=[],
            objective_value=None,
            feasible=False,
            convergence_status="ANALYSIS",
            violations_count=0,
            voltage_min=0.0,
            voltage_max=0.0,
            max_line_loading_pct=0.0,
            total_gen_mw=0.0,
            total_load_mw=0.0,
            llm_reasoning=result_summary,
            mode="analyze",
            elapsed_seconds=0.0,
            feasibility_detail="",
        )
        self._entries.append(entry)
        return entry

    def add_complete(
        self,
        iteration: int,
        summary: str,
    ) -> JournalEntry:
        """Record a 'complete' action in the journal.

        Creates a lightweight entry so the LLM's decision to end
        the search is visible in the search history.
        """
        entry = JournalEntry(
            iteration=iteration,
            description="Search completed by LLM",
            commands=[],
            objective_value=None,
            feasible=False,
            convergence_status="COMPLETE",
            violations_count=0,
            voltage_min=0.0,
            voltage_max=0.0,
            max_line_loading_pct=0.0,
            total_gen_mw=0.0,
            total_load_mw=0.0,
            llm_reasoning=summary,
            mode="complete",
            elapsed_seconds=0.0,
            feasibility_detail="",
        )
        self._entries.append(entry)
        return entry

    def get_sweep_entry(self) -> Optional[JournalEntry]:
        """Return the most recent sweep entry (mode='sweep'), or None."""
        for e in reversed(self._entries):
            if e.mode == "sweep":
                return e
        return None

    def get_contingency_entry(self) -> Optional[JournalEntry]:
        """Return the most recent contingency entry (convergence_status='CONTINGENCY'), or None."""
        for e in reversed(self._entries):
            if e.convergence_status == "CONTINGENCY":
                return e
        return None

    @property
    def entries(self) -> list[JournalEntry]:
        """All journal entries (read-only copy)."""
        return list(self._entries)

    @property
    def latest(self) -> Optional[JournalEntry]:
        """The most recent entry, or None if empty."""
        return self._entries[-1] if self._entries else None

    def __len__(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_row(e: JournalEntry) -> str:
        """Format a single entry as a table row."""
        desc_text = e.description
        if e.steering_directive:
            desc_text += f" [Steered: {e.steering_directive[:50]}]"
        desc = desc_text[:35].ljust(35)
        if e.convergence_status == "EXPLORE":
            desc = desc_text[:35].ljust(35)
            n_var = len(e.explored_variants) if e.explored_variants else 0
            pareto = ""
            if e.explored_variants:
                pareto_labels = [v["label"] for v in e.explored_variants if v.get("is_pareto")]
                if pareto_labels:
                    pareto = f" ★{','.join(pareto_labels)}"
            return (
                f"{e.iteration:>4} | {desc} | {'EXPLORE':>14} |  N/A  | "
                f"{n_var} variants{pareto:<13} |      N/A"
            )
        elif e.convergence_status == "SWEEP":
            desc = desc_text[:35].ljust(35)
            n_feas = len(e.feasible_buses) if e.feasible_buses is not None else 0
            n_cand = e.candidate_count or 0
            feas_summary = f"{n_feas}/{n_cand} feasible"
            return (
                f"{e.iteration:>4} | {desc} | {'SWEEP':>14} |  N/A  | "
                f"{feas_summary:<13} |      N/A"
            )
        elif e.feasible and e.objective_value is not None:
            cost = f"{e.objective_value:>12,.2f}"
            feas = "Yes"
            vrange = f"{e.voltage_min:.3f} - {e.voltage_max:.3f}"
            load = f"{e.max_line_loading_pct:>6.1f}%"
        elif e.feasibility_detail == "marginal":
            cost = f"{e.objective_value:>12,.2f}" if e.objective_value is not None else "         N/A"
            feas = "Marg"
            vrange = f"{e.voltage_min:.3f} - {e.voltage_max:.3f}" if e.voltage_min > 0 else "N/A          "
            load = f"{e.max_line_loading_pct:>6.1f}%" if e.max_line_loading_pct > 0 else "   N/A"
        else:
            cost = "         N/A" if e.objective_value is None else f"{e.objective_value:>12,.2f}"
            feas = "No "
            vrange = f"{e.voltage_min:.3f} - {e.voltage_max:.3f}" if e.voltage_min > 0 else "N/A          "
            load = f"{e.max_line_loading_pct:>6.1f}%" if e.max_line_loading_pct > 0 else "   N/A"
        return (
            f"{e.iteration:>4} | {desc} | {cost} | {feas}   "
            f"| {vrange:<13} | {load}"
        )

    def format_for_prompt(self, max_entries: Optional[int] = None) -> str:
        """Format the journal as a compact text table for LLM prompt injection."""
        if not self._entries:
            return "Search Journal: no iterations yet."

        header = (
            "Iter | Description                         "
            "| Cost($)      | Feas. | V_range(pu)   | Max_load(%)"
        )
        sep = (
            "-----|-------------------------------------"
            "|--------------|-------|---------------|------------"
        )

        lines = [f"Search Journal ({len(self._entries)} iterations):", "", header, sep]

        if max_entries is None or len(self._entries) <= max_entries:
            for e in self._entries:
                lines.append(self._format_row(e))
        else:
            # Always include entry 1, then last (max_entries - 1)
            lines.append(self._format_row(self._entries[0]))
            tail_start = len(self._entries) - (max_entries - 1)
            lines.append(
                f"  ... (entries 2-{tail_start} omitted)"
            )
            for e in self._entries[tail_start:]:
                lines.append(self._format_row(e))

        # Add footnote if any marginal or EMPAR entries exist
        has_marginal = any(e.feasibility_detail == "marginal" for e in self._entries)
        has_empar = any(e.solver.strip().upper() == "EMPAR" for e in self._entries)
        if has_marginal or has_empar:
            lines.append("")
            if has_marginal:
                lines.append("Note: Marg = marginal convergence (solver did not fully converge")
                lines.append("      but no constraint violations detected; use with caution).")
            if has_empar:
                lines.append(
                    "WARNING: EMPAR solver always reports CONVERGED and does not verify "
                    "N-1 security. Results reflect base-case feasibility only."
                )

        return "\n".join(lines)

    def format_detailed(self) -> str:
        """Format the full journal with all fields for post-session reporting."""
        if not self._entries:
            return "Search Journal: empty."

        parts: list[str] = [
            f"Search Journal — {len(self._entries)} iterations",
            "=" * 60,
        ]

        for e in self._entries:
            parts.append("")
            parts.append(f"--- Iteration {e.iteration} [{e.timestamp}] ---")
            parts.append(f"Mode: {e.mode}")
            parts.append(f"Description: {e.description}")
            parts.append(f"Convergence: {e.convergence_status}")
            if e.solver:
                parts.append(f"Solver: {e.solver}")
            if e.num_steps > 0:
                parts.append(f"Time periods: {e.num_steps}")
            if e.num_scenarios > 0:
                parts.append(f"Wind scenarios: {e.num_scenarios}")
            if e.candidate_count > 0:
                parts.append(f"Sweep candidates: {e.candidate_count}")
            if e.feasible_buses is not None:
                n_feas = len(e.feasible_buses)
                parts.append(f"Feasible buses ({n_feas}): {e.feasible_buses[:20]}"
                             + ("..." if n_feas > 20 else ""))
            if e.objective_value is not None:
                parts.append(f"Objective value: ${e.objective_value:,.2f}")
            else:
                parts.append("Objective value: N/A")
            parts.append(f"Feasible: {e.feasible}")
            if e.feasibility_detail:
                parts.append(f"Feasibility detail: {e.feasibility_detail}")
            parts.append(f"Violations: {e.violations_count}")
            parts.append(f"Voltage: {e.voltage_min:.3f} - {e.voltage_max:.3f} pu")
            parts.append(f"Max line loading: {e.max_line_loading_pct:.1f}%")
            parts.append(f"Gen: {e.total_gen_mw:.2f} MW / Load: {e.total_load_mw:.2f} MW")
            parts.append(f"Sim time: {e.elapsed_seconds:.2f}s")
            parts.append(f"Commands ({len(e.commands)}):")
            for cmd in e.commands:
                parts.append(f"  {json.dumps(cmd)}")
            parts.append(f"LLM reasoning: {e.llm_reasoning}")
            if e.steering_directive:
                parts.append(f"Steering directive: {e.steering_directive}")
            if e.tracked_metrics:
                parts.append(f"Tracked metrics: {json.dumps(e.tracked_metrics, indent=2)}")
            if e.explored_variants:
                parts.append(f"Explored variants ({len(e.explored_variants)}):")
                for v in e.explored_variants:
                    parts.append(f"  {json.dumps(v)}")

        return "\n".join(parts)

    def format_multi_objective_summary(self, max_entries: int | None = None) -> str:
        """Format a multi-objective tracking table for LLM prompt injection.

        Shows how each tracked metric has evolved across iterations.
        Only includes iterations that have tracked_metrics data.
        """
        registry = self.objective_registry
        if not registry.objectives:
            return ""

        tracked_entries = [e for e in self._entries if e.tracked_metrics]
        if not tracked_entries:
            return ""

        obj_names = [o.name for o in registry.objectives]

        # Apply max_entries limit
        if max_entries is not None and len(tracked_entries) > max_entries:
            display_entries = [tracked_entries[0]] + tracked_entries[-(max_entries - 1):]
            omitted = True
            omitted_range = (2, len(tracked_entries) - (max_entries - 1))
        else:
            display_entries = tracked_entries
            omitted = False
            omitted_range = (0, 0)

        # Build header
        col_headers = ["Iter"]
        for name in obj_names:
            obj = next(o for o in registry.objectives if o.name == name)
            short = name[:18]
            col_headers.append(f"{short}({obj.direction[0]})")

        header = " | ".join(f"{h:>20}" for h in col_headers)
        sep = "-" * len(header)

        lines = [
            f"Multi-Objective Tracking ({len(tracked_entries)} iterations, "
            f"{len(obj_names)} objectives):",
            "",
            header,
            sep,
        ]

        for e in display_entries:
            vals = [f"{e.iteration:>20}"]
            for name in obj_names:
                v = e.tracked_metrics.get(name)
                if v is not None:
                    vals.append(f"{v:>20.4f}")
                else:
                    vals.append(f"{'N/A':>20}")
            lines.append(" | ".join(vals))
            if omitted and e is display_entries[0]:
                lines.append(f"  ... (iterations {omitted_range[0]}-{omitted_range[1]} omitted)")

        # Trend analysis
        if len(tracked_entries) >= 2:
            lines.append("")
            lines.append("Trends (first tracked \u2192 latest):")
            first = tracked_entries[0]
            last = tracked_entries[-1]
            for name in obj_names:
                v0 = first.tracked_metrics.get(name)
                vn = last.tracked_metrics.get(name)
                if v0 is not None and vn is not None and v0 != 0:
                    delta_pct = (vn - v0) / abs(v0) * 100
                    direction = "\u2191" if vn > v0 else "\u2193" if vn < v0 else "\u2192"
                    lines.append(f"  {name}: {v0:.4f} \u2192 {vn:.4f} ({delta_pct:+.1f}%) {direction}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_json(self, path: Path) -> None:
        """Export the journal to a JSON file."""
        data = {
            "entries": [asdict(e) for e in self._entries],
            "objective_registry": self.objective_registry.to_dict_list(),
            "preference_history": self.objective_registry.history,
        }
        if self.benchmark_result is not None:
            data["benchmark_result"] = self.benchmark_result
        if self.session_best is not None:
            data["session_best"] = self.session_best
        if self.load_factor is not None:
            data["load_factor"] = self.load_factor
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Journal exported to %s (%d entries)", path, len(self._entries))

    def export_csv(self, path: Path) -> None:
        """Export the journal to a CSV file."""
        if not self._entries:
            path.write_text("", encoding="utf-8")
            return

        fieldnames = [
            "iteration", "description", "commands", "objective_value",
            "feasible", "convergence_status", "violations_count",
            "voltage_min", "voltage_max", "max_line_loading_pct",
            "total_gen_mw", "total_load_mw", "llm_reasoning",
            "mode", "elapsed_seconds", "timestamp", "steering_directive",
            "tracked_metrics", "feasibility_detail", "solver", "num_steps", "num_scenarios",
            "explored_variants", "candidate_count", "feasible_buses", "exago_command",
            "contingency_meta", "reserve_meta",
        ]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for e in self._entries:
                row = asdict(e)
                row["commands"] = json.dumps(row["commands"])
                row["tracked_metrics"] = json.dumps(row.get("tracked_metrics") or {})
                row["explored_variants"] = json.dumps(row.get("explored_variants") or [])
                row["feasible_buses"] = json.dumps(row.get("feasible_buses") or [])
                row["exago_command"] = json.dumps(row.get("exago_command") or None)
                row["contingency_meta"] = json.dumps(row.get("contingency_meta") or None)
                row["reserve_meta"] = json.dumps(row.get("reserve_meta") or None)
                writer.writerow(row)

        logger.info("Journal CSV exported to %s (%d entries)", path, len(self._entries))

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def summary_stats(
        self,
        best_iteration_override: int | None = None,
        goal_type: str | None = None,
    ) -> dict:
        """Compute summary statistics across all entries.

        Args:
            best_iteration_override: If provided, use this iteration as "best"
                instead of the default lowest-cost heuristic. This allows the
                LLM's goal classification to select the most relevant iteration.
            goal_type: Optional goal type string (e.g. "cost_minimization",
                "feasibility_boundary") to include in the returned dict.

        Returns:
            Dict with keys: total_iterations, best_objective, best_iteration,
            feasible_count, infeasible_count, objective_trend,
            voltage_range_trend, goal_type.
        """
        if not self._entries:
            return {
                "total_iterations": 0,
                "best_objective": None,
                "best_iteration": None,
                "best_bus": None,
                "feasible_count": 0,
                "infeasible_count": 0,
                "objective_trend": [],
                "voltage_range_trend": [],
                "goal_type": goal_type,
                "objective_registry": self.objective_registry.to_dict_list(),
                "is_multi_objective": self.objective_registry.is_multi_objective,
            }

        feasible = [e for e in self._entries if e.feasible and e.objective_value is not None]
        infeasible_count = sum(1 for e in self._entries if not e.feasible)
        marginal_count = sum(1 for e in self._entries if e.feasibility_detail == "marginal")

        # Use override if provided and valid
        best_bus = None
        if best_iteration_override is not None:
            override_entry = None
            for e in self._entries:
                if e.iteration == best_iteration_override:
                    override_entry = e
                    break
            if override_entry is not None:
                best_objective = override_entry.objective_value
                best_iteration = override_entry.iteration
            else:
                # Invalid override — fall back to default
                best_objective, best_iteration = self._best_by_cost(feasible)
        else:
            best_objective, best_iteration = self._best_by_cost(feasible)
            # Fix 2: sweep iterations carry no scalar cost (objective_value is None
            # → "SWEEP"), so the iteration-level minimum collapses to the base case.
            # For cost-style reductions, descend into the per-candidate variant costs
            # of cost sweeps (excluding boundary/metric sweeps, whose reduction is not
            # cost) and take the global minimum.
            if goal_type in (None, "cost_minimization"):
                sweep_best = self._best_cost_in_sweeps()
                if sweep_best is not None:
                    s_cost, s_iter, s_bus = sweep_best
                    if best_objective is None or s_cost < best_objective:
                        best_objective, best_iteration, best_bus = s_cost, s_iter, s_bus

        return {
            "total_iterations": len(self._entries),
            "best_objective": best_objective,
            "best_iteration": best_iteration,
            "best_bus": best_bus,
            "feasible_count": len(feasible),
            "infeasible_count": infeasible_count,
            "marginal_count": marginal_count,
            "objective_trend": [e.objective_value for e in self._entries],
            "voltage_range_trend": [
                (e.voltage_min, e.voltage_max) for e in self._entries
            ],
            "goal_type": goal_type,
            "objective_registry": self.objective_registry.to_dict_list(),
            "is_multi_objective": self.objective_registry.is_multi_objective,
        }

    @staticmethod
    def _best_by_cost(
        feasible: list[JournalEntry],
    ) -> tuple[float | None, int | None]:
        """Select the best iteration by lowest cost (default heuristic)."""
        if feasible:
            best = min(feasible, key=lambda e: e.objective_value)  # type: ignore[arg-type]
            return best.objective_value, best.iteration
        return None, None

    def _best_cost_in_sweeps(self) -> tuple[float, int, int | None] | None:
        """Lowest feasible per-candidate cost across all *cost* sweep entries.

        Descends into ``explored_variants``. Skips boundary sweeps (variants carry
        ``max_feasible_mw``, no cost reduction) and custom-metric sweeps (variants
        carry ``metric_name``, where the reduction is the metric, not cost), so a
        non-cost sweep's variant costs never hijack the cost-min summary.

        Returns ``(cost, iteration, bus)`` for the global minimum, or ``None``.
        """
        best: tuple[float, int, int | None] | None = None
        for e in self._entries:
            if e.mode != "sweep" or not e.explored_variants:
                continue
            variants = e.explored_variants
            if any(v.get("metric_name") for v in variants):
                continue  # custom-metric sweep — cost is not the reduction
            if any("max_feasible_mw" in v for v in variants):
                continue  # boundary sweep — capacity, not cost
            for v in variants:
                if v.get("feasible") and isinstance(v.get("cost"), (int, float)):
                    c = float(v["cost"])
                    if best is None or c < best[0]:
                        best = (c, e.iteration, v.get("bus"))
        return best
