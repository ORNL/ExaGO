"""Agent loop controller — central orchestrator for LLM-driven search."""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from agentigrid.backends import create_backend
from agentigrid.backends.base import LLMBackend, LLMResponse
from agentigrid.config import AppConfig
from agentigrid.engine.commands import parse_command
from agentigrid.engine.executor import SimulationExecutor, SimulationResult
from agentigrid.engine.explore import (
    ExploreCache,
    VariantResult,
    annotate_cost_equivalent_siblings,
    build_variant_description,
    compute_pareto_labels,
    format_variant_results,
)
from agentigrid.engine.journal import JournalEntry, ObjectiveEntry, SearchJournal
from agentigrid.engine.metric_extractor import available_metrics, available_metrics_for_app, extract_all_metrics
from agentigrid.engine.modifier import apply_modifications
from agentigrid.engine.objective_parser import (
    build_objective_extraction_prompt,
    parse_objective_extraction,
)
from agentigrid.engine.schema_description import command_schema_text
from agentigrid.parsers import (
    parse_matpower,
    network_metadata,
    network_summary,
    parse_simulation_result_for_app,
    results_summary_for_app,
)
from agentigrid.parsers.matpower_model import MATNetwork
from agentigrid.parsers.opflow_results import OPFLOWResult


def _count_scenario_rows(scenario_path: Path) -> int:
    """Count data rows (excluding header) in a SOPFLOW scenario CSV file."""
    try:
        lines = scenario_path.read_text(encoding="utf-8").strip().splitlines()
        return max(len(lines) - 1, 1)
    except (OSError, UnicodeDecodeError):
        return 1
from agentigrid.prompts import build_system_prompt, build_user_prompt
from agentigrid.engine.goal_classifier import build_classification_prompts, parse_goal_classification

logger = logging.getLogger("agentigrid.engine.agent_loop")

_MAX_CONSECUTIVE_PARSE_FAILURES = 3


def _bus_limits_from_network(net) -> dict[int, tuple[float, float]]:
    """Extract per-bus (Vmin, Vmax) from a MATNetwork for violation checking."""
    return {b.bus_i: (b.Vmin, b.Vmax) for b in net.buses}


def _benchmark_to_dict(bresult) -> dict:
    """Convert a BenchmarkResult to a serializable dict."""
    from agentigrid.engine.benchmark import DispatchComparison, LoadabilityResult
    d: dict = {
        "opflow_converged": bresult.opflow_converged,
        "opflow_objective": bresult.opflow_objective,
        "pflow_best_computed_cost": bresult.pflow_best_computed_cost,
        "cost_gap_pct": bresult.cost_gap_pct,
        "cost_gap_abs": bresult.cost_gap_abs,
        "summary_text": bresult.summary_text,
        "error": bresult.error,
    }
    if bresult.dispatch_comparison:
        d["dispatch_comparison"] = [
            {
                "bus": dc.bus,
                "fuel": dc.fuel,
                "opflow_pg": dc.opflow_pg,
                "pflow_pg": dc.pflow_pg,
                "delta": dc.delta,
                "opflow_pmax": dc.opflow_pmax,
            }
            for dc in bresult.dispatch_comparison
        ]
    if bresult.loadability is not None:
        d["loadability"] = {
            "opflow_max_factor": bresult.loadability.opflow_max_factor,
            "pflow_max_factor": bresult.loadability.pflow_max_factor,
            "gap_pct": bresult.loadability.gap_pct,
            "detail": bresult.loadability.detail,
        }
    return d


@dataclass
class SearchSession:
    """Complete record of a search session."""

    goal: str
    application: str
    base_case_path: Path
    config: AppConfig
    journal: SearchJournal
    start_time: str
    end_time: Optional[str] = None
    termination_reason: str = ""
    final_findings: Optional[dict] = None
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    goal_classification: Optional[dict] = None
    analysis_text: Optional[str] = None
    enforced_vmin: Optional[float] = None
    enforced_vmax: Optional[float] = None
    objective_registry_data: Optional[list[dict]] = None
    preference_history: Optional[list[dict]] = None
    tcopflow_period_data: Optional[list[dict]] = None
    tcopflow_dT_min: float = 0.0
    tcopflow_duration_min: float = 0.0
    tcopflow_is_coupling: bool = True
    sopflow_num_scenarios: int = 0
    benchmark_result: Optional[dict] = None


class AgentLoopController:
    """Drives the iterative LLM-driven search."""

    def __init__(
        self,
        config: AppConfig,
        quiet: bool = False,
        on_iteration: Callable[[int, JournalEntry, str, OPFLOWResult | None], None] | None = None,
        on_phase: Callable[[int, str], None] | None = None,
        on_pause_state: Callable[[bool], None] | None = None,
        on_explore: Callable[[int, list[dict]], None] | None = None,
    ) -> None:
        self._config = config
        self._backend: LLMBackend = create_backend(config.llm)
        self._executor = SimulationExecutor(config.exago, config.output)
        self._journal = SearchJournal()
        self._quiet = quiet
        self._on_iteration = on_iteration
        self._on_phase = on_phase
        self._on_pause_state = on_pause_state
        self._on_explore = on_explore
        self._stop_requested = False

        # Steering
        self._steering_queue: queue.Queue = queue.Queue()
        self._active_steering_directives: list[dict] = []
        self._steering_history: list[dict] = []

        # Pause/resume
        self._pause_event = threading.Event()
        self._pause_event.set()  # Not paused initially

        # State tracked across iterations
        self._base_network: Optional[MATNetwork] = None
        self._current_network: Optional[MATNetwork] = None
        self._latest_opflow: Optional[OPFLOWResult] = None
        self._base_opflow_result: Optional[OPFLOWResult] = None
        self._latest_results_text: Optional[str] = None
        self._error_feedback: Optional[str] = None
        self._consecutive_parse_failures = 0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._opflow_results_cache: dict[int, OPFLOWResult] = {}
        self._scopflow_num_contingencies: int = 0
        self._tcopflow_num_steps: int = 0
        self._tcopflow_duration_min: float = 0.0
        self._tcopflow_dT_min: float = 0.0
        self._tcopflow_is_coupling: bool = True
        self._tcopflow_period_data: list[dict] = []
        self._tcopflow_profile_overrides: dict[str, Path] = {}
        self._sopflow_num_scenarios: int = 0
        self._sopflow_scenario_override: Optional[Path] = None
        self._explore_cache: Optional[ExploreCache] = None
        # Sticky session-level load factor (Task 1)
        self._session_load_factor: Optional[float] = config.search.load_factor
        # Benchmark result computed at session start (Task 2)
        self._benchmark_result: Optional[dict] = None

    # ------------------------------------------------------------------
    # Output helper
    # ------------------------------------------------------------------

    def _print(self, msg: str) -> None:
        """Print progress message unless quiet mode is enabled."""
        if not self._quiet:
            print(msg)

    def request_stop(self) -> None:
        """Request graceful termination of the search loop."""
        self._stop_requested = True

    # ------------------------------------------------------------------
    # Steering & pause/resume API
    # ------------------------------------------------------------------

    def inject_steering(self, directive: str, mode: str = "augment") -> None:
        """Inject a user steering directive into the search.

        Args:
            directive: Natural language instruction from the user.
            mode: "augment" (add to original goal) or "replace" (override goal).
        """
        self._steering_queue.put({"directive": directive, "mode": mode})

    def pause(self) -> None:
        """Pause the search at the next iteration boundary."""
        self._pause_event.clear()
        if self._on_pause_state:
            self._on_pause_state(True)

    def resume(self) -> None:
        """Resume a paused search."""
        self._pause_event.set()
        if self._on_pause_state:
            self._on_pause_state(False)

    def is_paused(self) -> bool:
        """Return True if the search is currently paused."""
        return not self._pause_event.is_set()

    @property
    def steering_history(self) -> list[dict]:
        """Read-only copy of all steering directives injected so far."""
        return list(self._steering_history)

    # ------------------------------------------------------------------
    # Application-specific helpers
    # ------------------------------------------------------------------

    def _build_extra_args(self) -> list[str] | None:
        """Build application-specific extra CLI arguments for the executor."""
        args = []
        app = self._config.search.application

        if app == "scopflow":
            if self._config.search.ctgc_file:
                args.extend(["-ctgcfile", str(self._config.search.ctgc_file)])
                args.extend(["-scopflow_Nc", "-1"])
            if self._config.exago.mpi_np > 1:
                args.extend(["-scopflow_solver", "EMPAR"])

        if app == "tcopflow":
            pload = self._tcopflow_profile_overrides.get("pload_profile") or self._config.search.pload_profile
            qload = self._tcopflow_profile_overrides.get("qload_profile") or self._config.search.qload_profile
            if pload:
                args.extend(["-tcopflow_ploadprofile", str(pload)])
            if qload:
                args.extend(["-tcopflow_qloadprofile", str(qload)])
            if self._config.search.wind_profile:
                args.extend(["-tcopflow_windgenprofile", str(self._config.search.wind_profile)])
            dT = self._config.search.tcopflow_dT
            if dT != 60.0:
                args.extend(["-tcopflow_dT", str(dT)])
            duration = self._config.search.tcopflow_duration
            if duration != 1.0:
                args.extend(["-tcopflow_duration", str(duration)])
            iscoupling = self._config.search.tcopflow_iscoupling
            if iscoupling != 1:
                args.extend(["-tcopflow_iscoupling", str(iscoupling)])

        if app == "sopflow":
            scenario = self._sopflow_scenario_override or self._config.search.scenario_file
            if scenario:
                args.extend(["-windgen", str(scenario)])
                num_scenarios = _count_scenario_rows(Path(str(scenario)))
                args.extend(["-sopflow_Ns", str(num_scenarios)])
            solver = self._config.search.sopflow_solver
            args.extend(["-sopflow_solver", solver])
            iscoupling = self._config.search.sopflow_iscoupling
            if iscoupling != 0:
                args.extend(["-sopflow_iscoupling", str(iscoupling)])
            if self._config.exago.mpi_np > 1 and solver == "EMPAR":
                pass  # MPI is handled by the executor

        if self._config.search.gic_file:
            args.extend(["-gicfile", str(self._config.search.gic_file)])

        return args if args else None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, base_case: Path, goal: str) -> SearchSession:
        """Execute the full search loop."""
        session_start = time.monotonic()
        session = SearchSession(
            goal=goal,
            application=self._config.search.application,
            base_case_path=base_case,
            config=self._config,
            journal=self._journal,
            start_time=datetime.now().isoformat(),
        )

        # 1. Parse base case
        logger.info("Parsing base case: %s", base_case)
        self._base_network = parse_matpower(base_case)
        self._current_network = self._base_network
        net_summary_text = network_summary(self._base_network)
        net_metadata_text = network_metadata(self._base_network)
        self._network_metadata_text = net_metadata_text

        # 1b. Run OPFLOW benchmark at session start (if enabled)
        benchmark_text: Optional[str] = None
        if (
            self._config.search.application == "pflow"
            and self._config.search.benchmark_opflow
        ):
            try:
                from agentigrid.engine.benchmark import _run_opflow_on_base_case
                from agentigrid.prompts.system_prompt import format_benchmark_for_prompt
                self._print("[Session] Running OPFLOW baseline for benchmark reference...")
                opflow_baseline = _run_opflow_on_base_case(base_case, self._config)
                if opflow_baseline is not None:
                    opflow_cost = opflow_baseline.objective_value
                    pflow_baseline_cost = None
                    if self._base_network.gencost:
                        pflow_result_tmp = parse_simulation_result_for_app(
                            self._executor.run(self._base_network, "pflow", iteration=-2),
                            "pflow",
                            bus_limits=_bus_limits_from_network(self._base_network),
                        )
                        if pflow_result_tmp is not None:
                            pflow_baseline_cost = pflow_result_tmp.compute_generation_cost(
                                self._base_network.gencost
                            )
                    gap_pct = None
                    gap_abs = None
                    if opflow_cost and pflow_baseline_cost and opflow_cost != 0:
                        gap_abs = pflow_baseline_cost - opflow_cost
                        gap_pct = gap_abs / abs(opflow_cost) * 100
                    from agentigrid.engine.benchmark import _build_dispatch_comparison, DispatchComparison
                    dispatch_cmp = _build_dispatch_comparison(opflow_baseline, pflow_result_tmp if pflow_result_tmp is not None else None)
                    bench_dict: dict = {
                        "opflow_converged": opflow_baseline.converged,
                        "opflow_objective": opflow_cost,
                        "pflow_best_computed_cost": pflow_baseline_cost,
                        "cost_gap_pct": gap_pct,
                        "cost_gap_abs": gap_abs,
                        "dispatch_comparison": [
                            {
                                "bus": dc.bus,
                                "fuel": dc.fuel,
                                "opflow_pg": dc.opflow_pg,
                                "pflow_pg": dc.pflow_pg,
                                "delta": dc.delta,
                                "opflow_pmax": dc.opflow_pmax,
                            }
                            for dc in dispatch_cmp
                        ],
                    }
                    self._benchmark_result = bench_dict
                    self._journal.benchmark_result = bench_dict
                    benchmark_text = format_benchmark_for_prompt(bench_dict)
                    self._print(
                        f"[Session] Benchmark: OPFLOW=${opflow_cost:,.2f}"
                        if opflow_cost else "[Session] Benchmark: OPFLOW converged"
                    )
            except Exception as exc:
                logger.warning("Session-start benchmark failed: %s", exc)

        # Initialize journal load_factor from config
        if self._session_load_factor is not None:
            self._journal.load_factor = self._session_load_factor

        # Build system prompt once (static per session)
        from agentigrid.prompts.system_prompt import format_benchmark_for_prompt as _fmt_bench
        self._system_prompt = build_system_prompt(
            command_schema=command_schema_text(),
            network_summary=net_summary_text,
            application=self._config.search.application,
            search_mode=self._config.search.search_mode,
            concurrent_pflow=self._config.search.concurrent_pflow,
            network_metadata=net_metadata_text,
            benchmark_text=benchmark_text,
            session_load_factor=self._session_load_factor,
        )

        self._current_goal = goal

        # 2. Run base case simulation (iteration 0)
        self._print("[Iter 0] Running base case simulation...")
        sim_result = self._executor.run(
            self._base_network,
            self._config.search.application,
            iteration=0,
            extra_args=self._build_extra_args(),
        )

        # Extract SCOPFLOW metadata (num_contingencies) once from base case
        if self._config.search.application == "scopflow" and sim_result.success:
            from agentigrid.parsers import parse_scopflow_metadata
            meta = parse_scopflow_metadata(sim_result)
            if meta:
                self._scopflow_num_contingencies = meta.get("num_contingencies", 0)

        # Extract TCOPFLOW metadata and period files once from base case
        if self._config.search.application == "tcopflow" and sim_result.success:
            from agentigrid.parsers import parse_tcopflow_metadata, parse_tcopflow_period_files
            meta = parse_tcopflow_metadata(sim_result)
            if meta:
                self._tcopflow_num_steps = meta.get("num_steps", 0)
                self._tcopflow_duration_min = meta.get("duration_min", 0.0)
                self._tcopflow_dT_min = meta.get("dT_min", 0.0)
                self._tcopflow_is_coupling = meta.get("num_coupling_constraints", 0) > 0
            self._tcopflow_period_data = parse_tcopflow_period_files(sim_result.workdir)

        # Extract SOPFLOW metadata once from base case
        if self._config.search.application == "sopflow" and sim_result.success:
            from agentigrid.parsers import parse_sopflow_metadata
            meta = parse_sopflow_metadata(sim_result)
            if meta:
                self._sopflow_num_scenarios = meta.get("num_scenarios", 0)

        opflow = parse_simulation_result_for_app(
            sim_result,
            application=self._config.search.application,
            bus_limits=_bus_limits_from_network(self._base_network),
        )
        self._latest_opflow = opflow

        if opflow is not None:
            self._latest_results_text = results_summary_for_app(
                opflow,
                self._config.search.application,
                num_contingencies=self._scopflow_num_contingencies,
                num_steps=self._tcopflow_num_steps,
                duration_min=self._tcopflow_duration_min,
                dT_min=self._tcopflow_dT_min,
                is_coupling=self._tcopflow_is_coupling,
                period_data=self._tcopflow_period_data if self._tcopflow_period_data else None,
                num_scenarios=self._sopflow_num_scenarios,
                gencost=self._base_network.gencost if self._config.search.application == "pflow" else None,
            )
            self._base_opflow_result = opflow
            self._opflow_results_cache[0] = opflow
            self._journal.add_from_results(
                iteration=0,
                description="Base case (no modifications)",
                commands=[],
                opflow_result=opflow,
                sim_elapsed=sim_result.elapsed_seconds,
                llm_reasoning="Baseline run",
                mode="fresh",
                num_steps=self._tcopflow_num_steps,
                num_scenarios=self._sopflow_num_scenarios,
                gencost=self._base_network.gencost if self._config.search.application == "pflow" else None,
            )
            if self._config.search.application == "pflow":
                computed_cost = opflow.compute_generation_cost(self._base_network.gencost)
                self._print(
                    f"[Iter 0] Base case: {opflow.convergence_status}, "
                    f"computed cost=${computed_cost:,.2f}"
                )
            else:
                self._print(
                    f"[Iter 0] Base case: {opflow.convergence_status}, "
                    f"cost=${opflow.objective_value:,.2f}"
                )
        else:
            self._latest_results_text = None
            self._journal.add_from_results(
                iteration=0,
                description="Base case (no modifications)",
                commands=[],
                opflow_result=None,
                sim_elapsed=sim_result.elapsed_seconds,
                llm_reasoning="Baseline run",
                mode="fresh",
                num_steps=self._tcopflow_num_steps,
                num_scenarios=self._sopflow_num_scenarios,
            )
            self._error_feedback = (
                f"Base case simulation failed: {sim_result.error_message or 'unknown error'}"
            )
            self._print(f"[Iter 0] Base case simulation FAILED: {sim_result.error_message}")

        # 2b. Extract initial objectives from goal
        self._extract_initial_objectives(goal)

        # Backfill base case metrics now that objectives are registered
        if self._latest_opflow is not None:
            metric_names = [o.name for o in self._journal.objective_registry.objectives]
            base_metrics = extract_all_metrics(self._latest_opflow, metric_names)
            if base_metrics and self._journal.latest:
                self._journal.latest.tracked_metrics = base_metrics

        # Notify callback after base case
        if self._on_iteration:
            latest_entry = self._journal.latest
            if latest_entry:
                self._on_iteration(0, latest_entry, "base_case", self._latest_opflow)

        # 3. Agent loop
        max_iter = self._config.search.max_iterations
        for iteration in range(1, max_iter + 1):
            if self._stop_requested:
                session.termination_reason = "user_stopped"
                self._print("\nSearch stopped by user.")
                break
            action_type, should_continue = self._iteration(iteration, goal)
            # Notify callback after each iteration
            if self._on_iteration:
                latest_entry = self._journal.latest
                if latest_entry:
                    self._on_iteration(iteration, latest_entry, action_type, self._latest_opflow)
            if not should_continue:
                if not session.termination_reason:
                    session.termination_reason = "completed"
                break
        else:
            session.termination_reason = "max_iterations"
            self._print(f"\nMax iterations ({max_iter}) reached.")

        if not session.termination_reason:
            session.termination_reason = "completed"

        # Final notification with termination reason
        if self._on_iteration:
            latest_entry = self._journal.latest
            if latest_entry:
                self._on_iteration(
                    latest_entry.iteration,
                    latest_entry,
                    session.termination_reason,
                    self._latest_opflow,
                )

        session.end_time = datetime.now().isoformat()
        session.total_prompt_tokens = self._total_prompt_tokens
        session.total_completion_tokens = self._total_completion_tokens

        # Record the voltage limits that were enforced in the final network state
        if self._current_network is not None:
            limits = _bus_limits_from_network(self._current_network)
            if limits:
                session.enforced_vmin = min(v[0] for v in limits.values())
                session.enforced_vmax = max(v[1] for v in limits.values())

        # 4. Finalize session
        elapsed = time.monotonic() - session_start
        self._finalize(session, elapsed)
        return session

    # ------------------------------------------------------------------
    # Single iteration
    # ------------------------------------------------------------------

    def _iteration(
        self,
        iteration: int,
        goal: str,
    ) -> tuple[str, bool]:
        """Execute one iteration. Returns (action_type, should_continue)."""
        # Drain the steering queue at the iteration boundary
        new_directives: list[dict] = []
        while True:
            try:
                item = self._steering_queue.get_nowait()
            except queue.Empty:
                break
            directive = item["directive"]
            mode = item["mode"]
            if mode == "replace":
                self._active_steering_directives.clear()
            self._active_steering_directives.append(item)
            self._steering_history.append({"iteration": iteration, **item})
            new_directives.append(item)
            self._print(
                f"[Iter {iteration}] Steering [{mode.upper()}]: \"{directive[:80]}\""
            )

        # Extract objectives from any new steering directives
        for sd in new_directives:
            self._extract_objectives_from_steering(sd["directive"], iteration)

        # Pause: block here until resume() is called
        self._pause_event.wait()

        self._print(
            f"\n{'─' * 50}\n"
            f"[Iter {iteration}] Sending prompt to "
            f"{self._backend.name()} ({self._config.llm.model})..."
        )

        # Assemble and send prompt
        system_prompt, user_prompt = self._assemble_prompt(
            goal,
            self._latest_results_text,
            self._error_feedback,
            steering_directives=self._active_steering_directives or None,
            current_iteration=iteration,
        )
        self._error_feedback = None  # consumed

        if self._on_phase:
            self._on_phase(iteration, "llm_request")

        response = self._backend.complete(system_prompt, user_prompt)
        logger.debug("LLM raw response: %s", response.raw_text[:500])

        # Track tokens
        pt = response.prompt_tokens or 0
        ct = response.completion_tokens or 0
        self._total_prompt_tokens += pt
        self._total_completion_tokens += ct
        if pt or ct:
            self._print(
                f"[Iter {iteration}] Tokens: {pt} prompt + {ct} completion "
                f"(cumulative: ~{self._total_prompt_tokens + self._total_completion_tokens:,})"
            )

        # Parse JSON from response
        if response.json_data is None:
            self._consecutive_parse_failures += 1
            logger.warning(
                "Failed to parse JSON from LLM response (%d/%d): %s",
                self._consecutive_parse_failures,
                _MAX_CONSECUTIVE_PARSE_FAILURES,
                response.json_error,
            )
            self._print(f"[Iter {iteration}] Failed to parse LLM response as JSON")
            if self._consecutive_parse_failures >= _MAX_CONSECUTIVE_PARSE_FAILURES:
                self._print(f"[Iter {iteration}] Too many consecutive parse failures — aborting")
                return "error", False
            self._error_feedback = (
                "Failed to parse JSON from your response. "
                "Please respond with a valid JSON object."
            )
            return "error", True

        self._consecutive_parse_failures = 0
        data = response.json_data
        action = data.get("action", "").lower()

        # Dispatch action
        if action == "modify":
            self._explore_cache = None
            return self._handle_modify(iteration, data)
        elif action == "explore":
            return self._handle_explore(iteration, data)
        elif action == "select":
            return self._handle_select(iteration, data)
        elif action == "complete":
            return self._handle_complete(iteration, data)
        elif action == "analyze":
            return self._handle_analyze(iteration, data)
        elif action == "set_load_factor":
            return self._handle_set_load_factor(iteration, data)
        else:
            self._print(f"[Iter {iteration}] Unknown action: '{action}'")
            valid = "modify, explore, select, complete, analyze"
            if self._config.search.concurrent_pflow and self._config.search.application == "pflow":
                self._error_feedback = f"Unknown action '{action}'. Valid actions: {valid}."
            else:
                self._error_feedback = f"Unknown action '{action}'. Valid actions: modify, complete, analyze."
            return "error", True

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _handle_modify(
        self, iteration: int, data: dict
    ) -> tuple[str, bool]:
        """Handle a 'modify' action from the LLM."""
        description = data.get("description", "No description")
        reasoning = data.get("reasoning", "")
        raw_commands = data.get("commands", [])
        mode = data.get("mode", self._config.search.default_mode)

        self._print(f'[Iter {iteration}] LLM action: modify — "{description}"')

        if self._on_phase:
            self._on_phase(iteration, "applying_commands")

        # Choose base network for modifications
        if mode == "fresh":
            base_net = self._base_network
            self._tcopflow_profile_overrides = {}
            self._sopflow_scenario_override = None
        else:
            base_net = self._current_network

        # Auto-inject session load factor if set and not already in commands
        if (
            self._session_load_factor is not None
            and self._config.search.application == "pflow"
            and not any(
                r.get("action", "").lower() == "scale_all_loads" for r in raw_commands
            )
        ):
            raw_commands = [
                {"action": "scale_all_loads", "factor": self._session_load_factor}
            ] + list(raw_commands)

        # Parse and apply commands
        commands = []
        parse_errors = []
        for raw in raw_commands:
            try:
                commands.append(parse_command(raw))
            except ValueError as exc:
                logger.warning("Failed to parse command %s: %s", raw, exc)
                parse_errors.append(f"Invalid command {raw}: {exc}")

        if commands:
            # Build TCOPFLOW profile args for modifier
            _tcopflow_mod_kwargs = {}
            if self._config.search.application == "tcopflow":
                _tcopflow_mod_kwargs = {
                    "pload_profile": self._tcopflow_profile_overrides.get("pload_profile") or self._config.search.pload_profile,
                    "qload_profile": self._tcopflow_profile_overrides.get("qload_profile") or self._config.search.qload_profile,
                    "profile_output_dir": self._config.output.workdir / f"profiles_iter_{iteration:03d}",
                }
            # Build SOPFLOW scenario args for modifier
            _sopflow_mod_kwargs = {}
            if self._config.search.application == "sopflow":
                _sopflow_mod_kwargs = {
                    "scenario_file": self._sopflow_scenario_override or self._config.search.scenario_file,
                    "scenario_output_dir": self._config.output.workdir / f"scenarios_iter_{iteration:03d}",
                }
            modified_net, report = apply_modifications(
                base_net, commands, application=self._config.search.application,
                **_tcopflow_mod_kwargs,
                **_sopflow_mod_kwargs,
            )
            # Store profile overrides from modifier for subsequent iterations
            if report.profile_paths:
                self._tcopflow_profile_overrides.update(report.profile_paths)
            # Store scenario path override from modifier for subsequent iterations
            if report.scenario_paths:
                self._sopflow_scenario_override = report.scenario_paths.get("scenario_file")
            skipped_msgs = []
            for cmd, reasons in report.skipped:
                skipped_msgs.append(f"Skipped {cmd}: {'; '.join(reasons)}")
            applied_count = len(report.applied)
            skipped_count = len(report.skipped) + len(parse_errors)
        else:
            modified_net = base_net
            applied_count = 0
            skipped_count = len(parse_errors)
            skipped_msgs = []

        all_errors = parse_errors + skipped_msgs
        all_warnings = report.warnings if commands else []
        self._print(
            f"[Iter {iteration}] Applied {applied_count} command(s), "
            f"{skipped_count} skipped"
        )

        if all_errors:
            self._error_feedback = "Command errors:\n" + "\n".join(all_errors)

        if all_warnings:
            warning_text = "Warnings:\n" + "\n".join(all_warnings)
            if self._error_feedback:
                self._error_feedback += "\n\n" + warning_text
            else:
                self._error_feedback = warning_text

        # Run simulation
        if self._on_phase:
            self._on_phase(iteration, "running_simulation")
        self._print(f"[Iter {iteration}] Running {self._config.search.application} simulation...")
        sim_result = self._executor.run(
            modified_net,
            self._config.search.application,
            iteration=iteration,
            extra_args=self._build_extra_args(),
        )

        # Parse results
        if self._on_phase:
            self._on_phase(iteration, "parsing_results")
        opflow = parse_simulation_result_for_app(
            sim_result,
            application=self._config.search.application,
            bus_limits=_bus_limits_from_network(modified_net),
        )
        self._latest_opflow = opflow
        if opflow is not None:
            self._opflow_results_cache[iteration] = opflow

        # Parse TCOPFLOW period files after simulation
        if self._config.search.application == "tcopflow" and opflow is not None and sim_result.success:
            from agentigrid.parsers import parse_tcopflow_period_files
            self._tcopflow_period_data = parse_tcopflow_period_files(sim_result.workdir)

        if opflow is not None:
            self._latest_results_text = results_summary_for_app(
                opflow,
                self._config.search.application,
                num_contingencies=self._scopflow_num_contingencies,
                num_steps=self._tcopflow_num_steps,
                duration_min=self._tcopflow_duration_min,
                dT_min=self._tcopflow_dT_min,
                is_coupling=self._tcopflow_is_coupling,
                period_data=self._tcopflow_period_data if self._tcopflow_period_data else None,
                num_scenarios=self._sopflow_num_scenarios,
                gencost=self._current_network.gencost if self._config.search.application == "pflow" else None,
            )
            self._current_network = modified_net
            if self._config.search.application == "pflow":
                computed_cost = opflow.compute_generation_cost(modified_net.gencost)
                self._print(
                    f"[Iter {iteration}] Simulation completed in "
                    f"{sim_result.elapsed_seconds:.2f}s — "
                    f"{opflow.convergence_status}, computed cost=${computed_cost:,.2f}"
                )
            else:
                self._print(
                    f"[Iter {iteration}] Simulation completed in "
                    f"{sim_result.elapsed_seconds:.2f}s — "
                    f"{opflow.convergence_status}, cost=${opflow.objective_value:,.2f}"
                )
        else:
            self._latest_results_text = None
            error_msg = sim_result.error_message or "unknown error"
            feedback = f"Simulation failed: {error_msg}"
            if self._error_feedback:
                self._error_feedback += "\n" + feedback
            else:
                self._error_feedback = feedback
            self._print(
                f"[Iter {iteration}] Simulation FAILED in "
                f"{sim_result.elapsed_seconds:.2f}s — {error_msg}"
            )

        # Check for LLM-proposed objectives
        proposed = data.get("propose_objectives", [])
        if proposed and isinstance(proposed, list):
            for prop in proposed:
                name = prop.get("name", "")
                if name:
                    entry = ObjectiveEntry(
                        name=name,
                        direction=prop.get("direction", "minimize"),
                        threshold=prop.get("threshold"),
                        priority=prop.get("priority", "secondary"),
                        introduced_at=iteration,
                        source="llm_proposed",
                    )
                    self._journal.objective_registry.register(entry)
                    self._print(
                        f"[Objectives] LLM proposed: {name} ({entry.direction}, {entry.priority})"
                    )
            if proposed:
                self._backfill_metrics()

        # Update journal
        active_directive = (
            self._active_steering_directives[-1]["directive"]
            if self._active_steering_directives else None
        )
        self._journal.add_from_results(
            iteration=iteration,
            description=description,
            commands=raw_commands,
            opflow_result=opflow,
            sim_elapsed=sim_result.elapsed_seconds,
            llm_reasoning=reasoning,
            mode=mode,
            steering_directive=active_directive,
            num_steps=self._tcopflow_num_steps,
            num_scenarios=self._sopflow_num_scenarios,
            gencost=modified_net.gencost if self._config.search.application == "pflow" else None,
        )

        # Extract tracked metrics for multi-objective tracking
        if opflow is not None:
            metric_names = [o.name for o in self._journal.objective_registry.objectives]
            metrics = extract_all_metrics(opflow, metric_names)
            # For PFLOW: override generation_cost with the computed value
            # (extract_all_metrics reads opflow.objective_value, which is 0.0
            # for PFLOW). We have already populated the entry's objective_value
            # and tracked_metrics["generation_cost"] correctly in add_from_results;
            # preserve that value here too.
            if (
                self._config.search.application == "pflow"
                and "generation_cost" in metrics
                and self._journal.latest is not None
                and self._journal.latest.tracked_metrics is not None
                and "generation_cost" in self._journal.latest.tracked_metrics
            ):
                metrics["generation_cost"] = self._journal.latest.tracked_metrics["generation_cost"]
            if metrics and self._journal.latest:
                # Merge: keep any pre-populated values and add new ones
                if self._journal.latest.tracked_metrics:
                    merged = dict(self._journal.latest.tracked_metrics)
                    merged.update(metrics)
                    if "generation_cost" in self._journal.latest.tracked_metrics:
                        merged["generation_cost"] = self._journal.latest.tracked_metrics["generation_cost"]
                    self._journal.latest.tracked_metrics = merged
                else:
                    self._journal.latest.tracked_metrics = metrics

        return "modify", True

    @staticmethod
    def _current_bus_vlimits(net) -> tuple[float | None, float | None] | None:
        """Return the uniform (Vmin, Vmax) on the network, or None if not uniform.

        If all buses share the same Vmin and same Vmax, return that pair.
        Otherwise return None (mixed limits, cannot auto-inject).
        """
        if not net.buses:
            return None
        vmins = {round(b.Vmin, 6) for b in net.buses}
        vmaxs = {round(b.Vmax, 6) for b in net.buses}
        if len(vmins) == 1 and len(vmaxs) == 1:
            return (vmins.pop(), vmaxs.pop())
        return None

    @staticmethod
    def _variant_has_vlimits(raw_cmds: list[dict]) -> bool:
        """Check whether the variant commands include set_all_bus_vlimits."""
        for raw in raw_cmds:
            action = raw.get("action", "").lower()
            if action in ("set_all_bus_vlimits", "set_bus_vlimits"):
                return True
        return False

    def _handle_explore(
        self, iteration: int, data: dict
    ) -> tuple[str, bool]:
        """Handle an 'explore' action — evaluate multiple variants concurrently."""
        if not self._config.search.concurrent_pflow or self._config.search.application != "pflow":
            self._print(f"[Iter {iteration}] 'explore' requires --concurrent-pflow with pflow application")
            self._error_feedback = (
                "The 'explore' action requires concurrent PFLOW mode "
                "(--concurrent-pflow flag). Use 'modify' for single-point changes."
            )
            return "error", True

        description = data.get("description", "Neighborhood exploration")
        reasoning = data.get("reasoning", "")
        mode = data.get("mode", self._config.search.default_mode)
        raw_variants = data.get("variants", [])

        self._print(f'[Iter {iteration}] LLM action: explore — "{description}"')

        max_variants = self._config.search.max_variants

        # Validate variant count
        if not isinstance(raw_variants, list) or len(raw_variants) < 2:
            self._print(f"[Iter {iteration}] explore requires at least 2 variants, got {len(raw_variants) if isinstance(raw_variants, list) else 0}")
            self._error_feedback = (
                "The 'explore' action requires at least 2 variants. "
                "Each variant must have a 'label' and 'commands' list."
            )
            return "error", True

        if len(raw_variants) > max_variants:
            self._print(f"[Iter {iteration}] Truncating {len(raw_variants)} variants to max_variants={max_variants}")
            raw_variants = raw_variants[:max_variants]

        if self._on_phase:
            self._on_phase(iteration, "applying_commands")

        # Choose base network
        if mode == "fresh":
            base_net = self._base_network
            self._tcopflow_profile_overrides = {}
            self._sopflow_scenario_override = None
        else:
            base_net = self._current_network

        # Auto-inject set_all_bus_vlimits into variants missing it when
        # the current network has non-default voltage limits.
        _vlimits_inject = None
        if self._config.search.application == "pflow" and base_net is not None:
            limits = self._current_bus_vlimits(base_net)
            base_limits = self._current_bus_vlimits(self._base_network)
            if limits is not None and base_limits is not None and limits != base_limits:
                _vlimits_inject = limits
                self._print(
                    f"[Iter {iteration}] Auto-injecting set_all_bus_vlimits "
                    f"(Vmin={limits[0]}, Vmax={limits[1]}) into variants missing it"
                )

        # Parse and apply each variant
        variant_results: dict[str, VariantResult] = {}
        sim_tasks: list[tuple[MATNetwork, str, int, list[str] | None]] = []
        sim_labels: list[str] = []
        # Maps a variant label to its index in sim_tasks (and therefore in
        # results_map). Rejected variants are absent from this mapping.
        sim_idx_by_label: dict[str, int] = {}
        all_errors: list[str] = []
        all_warnings: list[str] = []
        rejected_labels: list[str] = []

        for raw_v in raw_variants:
            label = raw_v.get("label", "")
            if not label:
                label = chr(ord("A") + len(variant_results))
            raw_cmds = raw_v.get("commands", [])
            # LLM-provided description (if any). If absent or just a letter,
            # we auto-generate a human-readable description after apply_modifications.
            _llm_desc = raw_v.get("description")
            v_desc = _llm_desc if _llm_desc and _llm_desc != label else None

            if _vlimits_inject is not None and not self._variant_has_vlimits(raw_cmds):
                raw_cmds = [{"action": "set_all_bus_vlimits", "Vmin": _vlimits_inject[0], "Vmax": _vlimits_inject[1]}] + raw_cmds

            # Auto-inject session load factor if set and not already in variant
            if (
                self._session_load_factor is not None
                and not any(
                    r.get("action", "").lower() == "scale_all_loads" for r in raw_cmds
                )
            ):
                raw_cmds = [
                    {"action": "scale_all_loads", "factor": self._session_load_factor}
                ] + list(raw_cmds)

            commands: list = []
            parse_errors: list[str] = []
            for raw in raw_cmds:
                try:
                    commands.append(parse_command(raw))
                except ValueError as exc:
                    parse_errors.append(f"Variant {label}: Invalid command {raw}: {exc}")

            if parse_errors:
                all_errors.extend(parse_errors)
                continue

            rejected = False
            if commands:
                modified_net, report = apply_modifications(
                    base_net, commands, application=self._config.search.application,
                )
                applied = len(report.applied)
                skipped = len(report.skipped)
                self._print(f"[Iter {iteration}] Variant {label}: {applied} command(s) applied, {skipped} skipped")
                all_warnings.extend(report.warnings)
                if report.skipped:
                    for cmd, reasons in report.skipped:
                        all_errors.append(f"Variant {label}: Skipped {cmd}: {'; '.join(reasons)}")
                _skipped_cmds = list(report.skipped)
                # Auto-generate description from commands if the LLM didn't provide one.
                if v_desc is None:
                    v_desc = build_variant_description(commands, _skipped_cmds)
                # If every command in this variant was a no-op against the
                # base case, it would simulate to the base result and waste
                # a slot. Reject pre-execution and surface why.
                if applied == 0:
                    rejected = True
                    rejected_labels.append(label)
                    v_desc = f"[REJECTED — all commands no-op] {v_desc}"
                    self._print(
                        f"[Iter {iteration}] Variant {label}: REJECTED — all "
                        f"{skipped} command(s) would be no-ops against the base case"
                    )
            else:
                modified_net = base_net
                _skipped_cmds = []
                if v_desc is None:
                    v_desc = "(no commands)"

            if not rejected:
                # Use negative iteration numbers to create distinct workdirs for each variant
                variant_iter = -(iteration * max_variants + len(sim_tasks))

                gencost_for_summary = modified_net.gencost if self._config.search.application == "pflow" else None

                sim_idx_by_label[label] = len(sim_tasks)
                sim_tasks.append((
                    modified_net,
                    self._config.search.application,
                    variant_iter,
                    self._build_extra_args(),
                ))
                sim_labels.append(label)

            variant_results[label] = VariantResult(
                label=label,
                description=v_desc,
                commands=commands,
                raw_commands=raw_cmds,
                modified_net=modified_net,
                sim_result=None,
                opflow_result=None,
                skipped_commands=_skipped_cmds,
                rejected=rejected,
            )

        if len(sim_tasks) < 2:
            self._error_feedback = (
                "Not enough valid variants to explore. At least 2 variants with "
                "at least one effective (non-no-op) command are required."
            )
            if rejected_labels:
                self._error_feedback += (
                    f"\nRejected variants ({len(rejected_labels)}): "
                    f"{', '.join(rejected_labels)} — every proposed command "
                    "was a no-op against the base case."
                )
            if all_errors:
                self._error_feedback += "\n" + "\n".join(all_errors[:5])
            return "error", True

        # Run simulations concurrently
        if self._on_phase:
            if len(sim_tasks) > 1:
                self._on_phase(iteration, f"running_simulation ({len(sim_tasks)} variants)")
            else:
                self._on_phase(iteration, "running_simulation")
        self._print(f"[Iter {iteration}] Running {len(sim_tasks)} simulations concurrently...")

        results_map = self._executor.run_parallel(
            sim_tasks, max_workers=min(self._config.search.max_variants, len(sim_tasks)),
        )

        # Parse results — only for variants that were actually simulated
        # (rejected variants have no sim_idx entry and stay with empty results).
        for label, v in variant_results.items():
            if v.rejected:
                continue
            sim_idx = sim_idx_by_label.get(label)
            if sim_idx is None:
                continue
            sim_result = results_map.get(sim_idx)
            if sim_result is None:
                continue
            v.sim_result = sim_result

            opflow = parse_simulation_result_for_app(
                sim_result,
                application=self._config.search.application,
                bus_limits=_bus_limits_from_network(v.modified_net),
            )
            v.opflow_result = opflow

            if opflow is not None:
                self._print(
                    f"[Iter {iteration}] Variant {label}: {opflow.convergence_status}, "
                    f"feasibility={opflow.feasibility_detail}"
                )
            else:
                err_msg = sim_result.error_message or "simulation failed"
                self._print(f"[Iter {iteration}] Variant {label}: FAILED — {err_msg}")

        # Compute Pareto front
        if self._on_phase:
            self._on_phase(iteration, "computing_pareto_front")
        gencost = base_net.gencost if self._config.search.application == "pflow" else None
        pareto_labels = compute_pareto_labels(
            variant_results, self._journal.objective_registry.objectives, gencost,
        )

        # Flag identical-cost siblings and collect batch-level warning
        sibling_warning = annotate_cost_equivalent_siblings(variant_results, gencost)
        if sibling_warning:
            if self._error_feedback:
                self._error_feedback += "\n\n" + sibling_warning
            else:
                self._error_feedback = sibling_warning

        # Update session-best with any feasible variant cheaper than the current best
        if gencost is not None:
            for lbl, v in variant_results.items():
                if v.rejected or v.opflow_result is None:
                    continue
                if not (v.opflow_result.feasibility_detail == "feasible"
                        and v.opflow_result.num_violations == 0):
                    continue
                try:
                    cost = v.opflow_result.compute_generation_cost(gencost)
                except Exception:
                    continue
                if cost > 0:
                    self._journal.update_session_best(
                        label=lbl,
                        iteration=iteration,
                        cost=cost,
                        commands=v.raw_commands,
                    )

        # Build results text
        self._latest_results_text = format_variant_results(
            variant_results, pareto_labels, gencost,
        )

        # Store explore cache
        self._explore_cache = ExploreCache(
            variants=variant_results,
            description=description,
            reasoning=reasoning,
            iteration=iteration,
            base_network_snapshot=base_net,
            base_mode=mode,
        )

        self._latest_opflow = None
        if pareto_labels and pareto_labels[0] in variant_results:
            best = variant_results[pareto_labels[0]]
            if best.opflow_result is not None:
                self._latest_opflow = best.opflow_result

        self._print(
            f"[Iter {iteration}] Explored {len(variant_results)} variants. "
            f"Pareto: {', '.join(pareto_labels) if pareto_labels else 'none'}"
        )

        # Notify UI about explore results
        if self._on_explore:
            variant_summaries = []
            for lbl, v in variant_results.items():
                summary = {"label": lbl, "feasible": False}
                if v.opflow_result is not None:
                    summary["feasible"] = (
                        v.opflow_result.feasibility_detail == "feasible"
                        and v.opflow_result.num_violations == 0
                    )
                    summary["convergence_status"] = v.opflow_result.convergence_status
                    summary["voltage_min"] = v.opflow_result.voltage_min
                    summary["voltage_max"] = v.opflow_result.voltage_max
                    summary["max_line_loading_pct"] = v.opflow_result.max_line_loading_pct
                    summary["violations_count"] = v.opflow_result.num_violations
                summary["is_pareto"] = v.is_pareto
                variant_summaries.append(summary)
            self._on_explore(iteration, variant_summaries)

        if all_warnings:
            warning_text = "Warnings:\n" + "\n".join(all_warnings[:5])
            if self._error_feedback:
                self._error_feedback += "\n\n" + warning_text
            else:
                self._error_feedback = warning_text

        active_directive = (
            self._active_steering_directives[-1]["directive"]
            if self._active_steering_directives else None
        )

        variant_labels = sorted(variant_results.keys())
        variant_info = []
        for lbl in variant_labels:
            v = variant_results[lbl]
            info = {"label": lbl, "description": v.description, "commands": v.raw_commands, "is_pareto": v.is_pareto}
            if v.rejected:
                info["rejected"] = True
            if v.skipped_commands:
                skipped_summaries = []
                for cmd, reasons in v.skipped_commands:
                    cmd_name = type(cmd).__name__
                    reason_str = "; ".join(reasons)
                    skipped_summaries.append(f"{cmd_name}: {reason_str}")
                info["skipped"] = skipped_summaries
            if v.opflow_result is not None:
                info["feasible"] = (
                    v.opflow_result.feasibility_detail == "feasible"
                    and v.opflow_result.num_violations == 0
                )
                info["voltage_min"] = v.opflow_result.voltage_min
                info["voltage_max"] = v.opflow_result.voltage_max
                info["max_line_loading_pct"] = v.opflow_result.max_line_loading_pct
                info["violations_count"] = v.opflow_result.num_violations
                if gencost is not None:
                    try:
                        info["cost"] = v.opflow_result.compute_generation_cost(gencost)
                    except Exception:
                        pass
            else:
                info["feasible"] = False
            variant_info.append(info)

        self._journal.add_explore(
            iteration=iteration,
            description=f"[explore] {description}",
            variant_info=variant_info,
            pareto_labels=pareto_labels,
            llm_reasoning=reasoning,
            steering_directive=active_directive,
        )

        return "explore", True

    def _handle_select(
        self, iteration: int, data: dict
    ) -> tuple[str, bool]:
        """Handle a 'select' action — choose a variant from explore results."""
        if self._explore_cache is None:
            self._print(f"[Iter {iteration}] 'select' without prior 'explore'")
            self._error_feedback = (
                "No explore results to select from. Use 'explore' before 'select'."
            )
            return "error", True

        choice = data.get("choice", "")
        reasoning = data.get("reasoning", "")

        cache = self._explore_cache

        if choice not in cache.variants:
            available = sorted(cache.variants.keys())
            self._print(f"[Iter {iteration}] Invalid variant choice: '{choice}'")
            self._error_feedback = (
                f"Invalid variant '{choice}'. Available variants: {', '.join(available)}. "
                "Respond with a 'select' action choosing one of these."
            )
            return "error", True

        selected = cache.variants[choice]
        if selected.rejected:
            self._print(f"[Iter {iteration}] Cannot select rejected variant '{choice}'")
            available = sorted(
                lbl for lbl, v in cache.variants.items() if not v.rejected
            )
            self._error_feedback = (
                f"Variant '{choice}' was rejected (all commands were no-ops) and "
                f"cannot be selected. Selectable variants: {', '.join(available) or 'none'}."
            )
            return "error", True
        self._print(f'[Iter {iteration}] Selected variant {choice} — "{selected.description}"')

        # Update current state
        self._current_network = selected.modified_net
        self._latest_opflow = selected.opflow_result
        if selected.opflow_result is not None:
            self._opflow_results_cache[iteration] = selected.opflow_result
            self._latest_results_text = results_summary_for_app(
                selected.opflow_result,
                self._config.search.application,
                num_contingencies=self._scopflow_num_contingencies,
                num_steps=self._tcopflow_num_steps,
                duration_min=self._tcopflow_duration_min,
                dT_min=self._tcopflow_dT_min,
                is_coupling=self._tcopflow_is_coupling,
                period_data=self._tcopflow_period_data if self._tcopflow_period_data else None,
                num_scenarios=self._sopflow_num_scenarios,
                gencost=selected.modified_net.gencost if self._config.search.application == "pflow" else None,
            )

        # Build explored-variants summary for journal
        explored_variants = []
        gencost = (
            (cache.base_network_snapshot.gencost if cache.base_network_snapshot else None)
            if self._config.search.application == "pflow" else None
        )
        for lbl, v in cache.variants.items():
            entry = {"label": lbl, "description": v.description, "commands": v.raw_commands, "is_pareto": v.is_pareto}
            if v.rejected:
                entry["rejected"] = True
            if v.skipped_commands:
                skipped_summaries = []
                for cmd, reasons in v.skipped_commands:
                    cmd_name = type(cmd).__name__
                    reason_str = "; ".join(reasons)
                    skipped_summaries.append(f"{cmd_name}: {reason_str}")
                entry["skipped"] = skipped_summaries
            if v.opflow_result is not None:
                entry["feasible"] = (
                    v.opflow_result.feasibility_detail == "feasible"
                    and v.opflow_result.num_violations == 0
                )
                if gencost is not None:
                    try:
                        entry["cost"] = v.opflow_result.compute_generation_cost(gencost)
                    except Exception:
                        pass
            else:
                entry["feasible"] = False
            explored_variants.append(entry)

        # Add journal entry for the selected variant
        active_directive = (
            self._active_steering_directives[-1]["directive"]
            if self._active_steering_directives else None
        )
        self._journal.add_from_results(
            iteration=iteration,
            description=f"[select {choice}] {cache.description}",
            commands=selected.raw_commands,
            opflow_result=selected.opflow_result,
            sim_elapsed=selected.sim_result.elapsed_seconds if selected.sim_result else 0.0,
            llm_reasoning=f"Selected variant {choice}. {reasoning}",
            mode=cache.base_mode,
            steering_directive=active_directive,
            num_steps=self._tcopflow_num_steps,
            num_scenarios=self._sopflow_num_scenarios,
            explored_variants=explored_variants,
            gencost=selected.modified_net.gencost if self._config.search.application == "pflow" else None,
        )

        # Extract tracked metrics for multi-objective tracking
        if selected.opflow_result is not None and self._journal.latest:
            metric_names = [o.name for o in self._journal.objective_registry.objectives]
            metrics = extract_all_metrics(selected.opflow_result, metric_names)
            # For PFLOW: preserve the computed cost we populated above —
            # extract_all_metrics would otherwise overwrite it with the
            # raw opflow.objective_value (0.0 for PFLOW).
            if (
                self._config.search.application == "pflow"
                and "generation_cost" in metrics
                and self._journal.latest.tracked_metrics is not None
                and "generation_cost" in self._journal.latest.tracked_metrics
            ):
                metrics["generation_cost"] = self._journal.latest.tracked_metrics["generation_cost"]
            if metrics:
                if self._journal.latest.tracked_metrics:
                    merged = dict(self._journal.latest.tracked_metrics)
                    merged.update(metrics)
                    if "generation_cost" in self._journal.latest.tracked_metrics:
                        merged["generation_cost"] = self._journal.latest.tracked_metrics["generation_cost"]
                    self._journal.latest.tracked_metrics = merged
                else:
                    self._journal.latest.tracked_metrics = metrics

        # Clear explore cache
        self._explore_cache = None

        return "select", True

    def _handle_complete(
        self, iteration: int, data: dict
    ) -> tuple[str, bool]:
        """Handle a 'complete' action from the LLM."""
        findings = data.get("findings", {})
        reasoning = data.get("reasoning", "")
        summary_text = findings.get("summary", reasoning)

        self._print(f"[Iter {iteration}] LLM action: complete")
        self._print(f'[Iter {iteration}] Search completed: "{summary_text}"')

        self._journal.add_complete(
            iteration=iteration,
            summary=summary_text,
        )

        self._final_findings = findings
        return "complete", False

    def _handle_analyze(
        self, iteration: int, data: dict
    ) -> tuple[str, bool]:
        """Handle an 'analyze' action from the LLM."""
        query = data.get("query", "")

        self._print(f'[Iter {iteration}] LLM action: analyze — "{query}"')

        result_text = self._run_analysis_query(query)
        self._latest_results_text = result_text
        logger.info("Analysis query result: %s", result_text[:300])

        # Record analyze action in the journal
        self._journal.add_analysis(
            iteration=iteration,
            query=query,
            result_summary=result_text[:200],
        )

        return "analyze", True

    def _handle_set_load_factor(
        self, iteration: int, data: dict
    ) -> tuple[str, bool]:
        """Handle a 'set_load_factor' action — update the session-level load scaling factor."""
        new_factor = data.get("factor")
        if not isinstance(new_factor, (int, float)) or new_factor <= 0:
            self._error_feedback = (
                f"Invalid factor for set_load_factor: {new_factor!r}. "
                "Must be a positive number (e.g., 1.23)."
            )
            return "error", True

        old_factor = self._session_load_factor
        self._session_load_factor = float(new_factor)
        self._journal.load_factor = self._session_load_factor
        self._print(
            f"[Iter {iteration}] LLM action: set_load_factor — "
            f"{old_factor} → {self._session_load_factor}"
        )
        # Rebuild system prompt to reflect the new load factor note
        from agentigrid.prompts.system_prompt import format_benchmark_for_prompt as _fmt_bench
        net_summary_text = network_summary(self._base_network)
        net_metadata_text = self._network_metadata_text
        benchmark_text = _fmt_bench(self._benchmark_result) if self._benchmark_result else None
        self._system_prompt = build_system_prompt(
            command_schema=command_schema_text(),
            network_summary=net_summary_text,
            application=self._config.search.application,
            search_mode=self._config.search.search_mode,
            concurrent_pflow=self._config.search.concurrent_pflow,
            network_metadata=net_metadata_text,
            benchmark_text=benchmark_text,
            session_load_factor=self._session_load_factor,
        )
        self._error_feedback = (
            f"Load factor updated to {self._session_load_factor}×. "
            "All subsequent runs will automatically apply this scaling. "
            "Use 'explore' or 'modify' to continue the search."
        )
        return "set_load_factor", True

    def _run_analysis_query(self, query: str) -> str:
        """Execute an analysis query against the latest OPFLOW results.

        Handles pattern-matched queries for common analyses and falls back to
        an LLM sub-call for arbitrary questions.
        """
        if self._latest_opflow is None:
            return "No simulation results available to analyze."

        opf = self._latest_opflow
        q = query.lower()

        _dcopflow = self._config.search.application == "dcopflow"
        _tcopflow = self._config.search.application == "tcopflow"
        _sopflow = self._config.search.application == "sopflow"

        # ── TCOPFLOW time-period queries ──────────────────────────────────
        if _tcopflow and ("period" in q or "time step" in q or "timestep" in q or "temporal" in q):
            if not self._tcopflow_period_data:
                return "No per-period data available. Run a simulation first."
            lines = [f"Multi-period summary ({len(self._tcopflow_period_data)} periods):"]
            lines.append(
                f"{'Period':>6} | {'Load(MW)':>9} | {'Gen(MW)':>9} | "
                f"{'Vmin(pu)':>8} | {'Vmax(pu)':>8} | {'MaxLoad%':>8}"
            )
            lines.append("-" * 62)
            for p in self._tcopflow_period_data:
                lines.append(
                    f"{p['period']:>6} | {p['total_load_mw']:>9.1f} | "
                    f"{p['total_gen_mw']:>9.1f} | {p['voltage_min']:>8.3f} | "
                    f"{p['voltage_max']:>8.3f} | {p['max_line_loading_pct']:>7.1f}%"
                )
            peak = max(self._tcopflow_period_data, key=lambda p: p["total_load_mw"])
            worst_v = min(self._tcopflow_period_data, key=lambda p: p["voltage_min"])
            worst_load = max(self._tcopflow_period_data, key=lambda p: p["max_line_loading_pct"])
            lines.append(f"\nPeak load: period {peak['period']} ({peak['total_load_mw']:.1f} MW)")
            lines.append(f"Worst voltage: period {worst_v['period']} (Vmin={worst_v['voltage_min']:.3f} pu)")
            lines.append(f"Worst line loading: period {worst_load['period']} ({worst_load['max_line_loading_pct']:.1f}%)")
            return "\n".join(lines)

        # ── SOPFLOW scenario queries ─────────────────────────────────────
        # Only intercept queries that are purely about scenario/wind metadata.
        # If the query also asks about voltage, loading, or generators,
        # let it fall through to the standard pattern matchers so the LLM
        # gets useful data instead of a generic 4-line dead end.
        if _sopflow and ("scenario" in q or "wind" in q or "stochastic" in q):
            _wants_metrics = (
                "voltage" in q or "vm" in q or "bus" in q
                or "loading" in q or "line" in q or "branch" in q
                or "generator" in q or "load" in q or "loss" in q
                or "flow" in q or "slack" in q
            )
            if not _wants_metrics:
                wind_gens = [g for g in opf.generators if "wind" in g.fuel.lower() and g.status == 1]
                wind_info = ""
                if wind_gens:
                    wind_pg = sum(g.Pg for g in wind_gens)
                    wind_pmax = sum(g.Pmax for g in wind_gens)
                    wind_util = wind_pg / wind_pmax * 100 if wind_pmax > 0 else 0
                    wind_info = (
                        f"Wind generators: {len(wind_gens)} online, "
                        f"total {wind_pg:.2f} MW / {wind_pmax:.2f} MW capacity "
                        f"({wind_util:.0f}% utilization)"
                    )
                lines = [
                    f"SOPFLOW scenario summary ({self._sopflow_num_scenarios} scenarios):",
                    f"Solver: {opf.solver}",
                    wind_info,
                    f"Base-case dispatch cost: ${opf.objective_value:,.2f}",
                ]
                if opf.num_violations > 0:
                    lines.append(f"Constraints violated in base case: {opf.num_violations}")
                else:
                    lines.append("All scenarios satisfied network constraints (base case feasible).")
                lines.append(
                    "Note: SOPFLOW output only contains the base-case (first-stage) dispatch. "
                    "Per-scenario voltage/loading data is not available. "
                    "To explore scenario effects, modify the network (e.g., scale_wind_scenario, "
                    "scale_all_loads) and observe feasibility changes."
                )
                if wind_gens:
                    wind_pg = sum(g.Pg for g in wind_gens)
                    wind_pmax = sum(g.Pmax for g in wind_gens)
                    if wind_pmax > 0 and wind_pg / wind_pmax >= 0.995:
                        lines.append(
                            "WARNING: Wind generators are at maximum capacity in the base-case "
                            "dispatch. scale_wind_scenario alone will NOT change the first-stage "
                            "dispatch. Increase wind Pmax (set_gen_dispatch) or increase system "
                            "stress (scale_all_loads) to see feasibility changes."
                        )
                return "\n".join(lines)
        m = re.search(r"voltage\s+below\s+([\d.]+)", q)
        if m:
            if _dcopflow:
                return (
                    "Voltage magnitude analysis is not available in DCOPFLOW. "
                    "In the DC approximation, all bus voltages are fixed at 1.0 pu. "
                    "Use phase angle or line loading queries instead."
                )
            threshold = float(m.group(1))
            buses = [b for b in opf.buses if b.Vm < threshold]
            if not buses:
                return f"No buses with voltage below {threshold} pu."
            lines = [f"Buses with Vm < {threshold} pu:"]
            for b in sorted(buses, key=lambda b: b.Vm):
                lines.append(f"  Bus {b.bus_id}: Vm={b.Vm:.4f} pu")
            return "\n".join(lines)

        m = re.search(r"voltage\s+above\s+([\d.]+)", q)
        if m:
            if _dcopflow:
                return (
                    "Voltage magnitude analysis is not available in DCOPFLOW. "
                    "In the DC approximation, all bus voltages are fixed at 1.0 pu. "
                    "Use phase angle or line loading queries instead."
                )
            threshold = float(m.group(1))
            buses = [b for b in opf.buses if b.Vm > threshold]
            if not buses:
                return f"No buses with voltage above {threshold} pu."
            lines = [f"Buses with Vm > {threshold} pu:"]
            for b in sorted(buses, key=lambda b: -b.Vm):
                lines.append(f"  Bus {b.bus_id}: Vm={b.Vm:.4f} pu")
            return "\n".join(lines)

        # ── Phase angle queries (DCOPFLOW) ────────────────────────────────
        if re.search(r"phase\s+angle|angle\s+profile|bus\s+angle", q):
            if not opf.buses:
                return "No bus data available."
            sorted_by_angle = sorted(opf.buses, key=lambda b: b.Va)
            lines = [f"Phase angle profile ({len(opf.buses)} buses):"]
            lines.append(f"  Min: {sorted_by_angle[0].Va:.3f}° (bus {sorted_by_angle[0].bus_id})")
            lines.append(f"  Max: {sorted_by_angle[-1].Va:.3f}° (bus {sorted_by_angle[-1].bus_id})")
            ref = min(opf.buses, key=lambda b: abs(b.Va))
            lines.append(f"  Ref: bus {ref.bus_id} ({ref.Va:.3f}°)")
            lines.append("  Most extreme angles:")
            for b in sorted_by_angle[:5]:
                lines.append(f"    Bus {b.bus_id}: Va={b.Va:.3f}°")
            if len(sorted_by_angle) > 5:
                lines.append("    ...")
                for b in sorted_by_angle[-3:]:
                    lines.append(f"    Bus {b.bus_id}: Va={b.Va:.3f}°")
            return "\n".join(lines)

        # ── Line loading ─────────────────────────────────────────────────
        if "most loaded" in q or "loaded lines" in q or "line loading" in q:
            loaded = []
            for br in opf.branches:
                if br.Slim > 0:
                    pct = max(br.Sf, br.St) / br.Slim * 100
                    loaded.append((pct, br))
            loaded.sort(key=lambda x: -x[0])
            lines = ["Most loaded lines:"]
            for pct, br in loaded[:10]:
                flow = max(br.Sf, br.St)
                lines.append(
                    f"  {br.from_bus}->{br.to_bus}: {pct:.1f}% "
                    f"({flow:.2f}/{br.Slim:.2f} MVA)"
                )
            return "\n".join(lines)

        # ── Generator summary ────────────────────────────────────────────
        if "generator" in q:
            lines = ["Generators:"]
            gencost_list = None
            ref_bus_ids = set()
            net_for_info = getattr(self, "_current_network", None) or getattr(self, "_base_network", None)
            if net_for_info is not None and hasattr(net_for_info, "gencost"):
                gencost_list = net_for_info.gencost
                ref_bus_ids = {b.bus_i for b in net_for_info.buses if b.type == 3}
            show_cost = "cost" in q or "coeff" in q or "detail" in q or "dispatch" in q
            for i, g in enumerate(sorted(opf.generators, key=lambda g: -g.Pg)):
                status = "ON" if g.status == 1 else "OFF"
                bus_type = " [SLACK]" if g.bus in ref_bus_ids else ""
                cost_str = ""
                if show_cost and gencost_list is not None and i < len(gencost_list):
                    gc = gencost_list[i]
                    if hasattr(gc, "coeffs") and gc.coeffs:
                        cost_str = f" cost_coeffs={gc.coeffs}"
                lines.append(
                    f"  Bus {g.bus}: {status} Pg={g.Pg:.2f} MW "
                    f"[{g.Pmin:.0f}-{g.Pmax:.0f}] fuel={g.fuel}{bus_type}{cost_str}"
                )
            return "\n".join(lines)

        # ── Voltage profile for a specific kV level ──────────────────────
        m = re.search(r"(\d+(?:\.\d+)?)\s*kv", q)
        if m and ("voltage profile" in q or "kv buses" in q or "buses" in q):
            target_kv = float(m.group(1))
            kv_buses = [b for b in opf.buses if abs(b.base_kv - target_kv) < 1.0]
            if not kv_buses:
                return f"No buses found at {target_kv} kV."
            lines = [f"Voltage profile for {target_kv} kV buses ({len(kv_buses)} buses):"]
            for b in sorted(kv_buses, key=lambda b: b.bus_id):
                lines.append(f"  Bus {b.bus_id}: Vm={b.Vm:.4f} pu, Va={b.Va:.2f}°")
            return "\n".join(lines)

        # ── Area summary ─────────────────────────────────────────────────
        m = re.search(r"area\s+(\d+)", q)
        if m and ("summary" in q or "area" in q):
            area_id = int(m.group(1))
            area_buses = [b for b in opf.buses if b.area == area_id]
            if not area_buses:
                return f"No buses found in area {area_id}."
            total_load = sum(b.Pd for b in area_buses)
            total_gen = sum(
                g.Pg for g in opf.generators
                if any(b.bus_id == g.bus for b in area_buses)
            )
            vm_vals = [b.Vm for b in area_buses]
            lines = [
                f"Area {area_id} Summary ({len(area_buses)} buses):",
                f"  Total load:       {total_load:.2f} MW",
                f"  Total generation: {total_gen:.2f} MW",
                f"  Voltage range:    {min(vm_vals):.4f} – {max(vm_vals):.4f} pu",
            ]
            return "\n".join(lines)

        # ── Cost breakdown by fuel type ──────────────────────────────────
        if "cost breakdown" in q or "generation cost" in q:
            from collections import defaultdict
            fuel_mw: dict[str, float] = defaultdict(float)
            for g in opf.generators:
                if g.status == 1:
                    fuel = g.fuel or "unknown"
                    fuel_mw[fuel] += g.Pg
            if self._config.search.application == "pflow":
                computed_cost = opf.compute_generation_cost(
                    self._current_network.gencost if self._current_network else self._base_network.gencost
                )
                lines = [f"Generation cost breakdown (computed: ${computed_cost:,.2f}):"]
            else:
                lines = [f"Generation cost breakdown (total: ${opf.objective_value:,.2f}):"]
            for fuel, mw in sorted(fuel_mw.items(), key=lambda x: -x[1]):
                lines.append(f"  {fuel:15s}: {mw:8.2f} MW")
            return "\n".join(lines)

        # ── Constraint margins ───────────────────────────────────────────
        if "constraint margin" in q or "binding constraint" in q:
            lines = ["Constraint Margins:"]
            # Voltage limits
            v_min_limit = 0.95
            v_max_limit = 1.05
            voltage_margins = []
            for b in opf.buses:
                margin_low = b.Vm - v_min_limit
                margin_high = v_max_limit - b.Vm
                voltage_margins.append((min(margin_low, margin_high), b))
            voltage_margins.sort(key=lambda x: x[0])
            lines.append("  Tightest voltage margins:")
            for margin, b in voltage_margins[:5]:
                lines.append(f"    Bus {b.bus_id}: Vm={b.Vm:.4f} pu (margin={margin:.4f})")
            # Line loading
            line_margins = []
            for br in opf.branches:
                if br.Slim > 0:
                    loading = max(br.Sf, br.St) / br.Slim * 100
                    line_margins.append((100 - loading, br, loading))
            line_margins.sort(key=lambda x: x[0])
            lines.append("  Lines closest to thermal limit:")
            for headroom, br, loading in line_margins[:5]:
                lines.append(
                    f"    {br.from_bus}->{br.to_bus}: {loading:.1f}% loaded "
                    f"(headroom={headroom:.1f}%)"
                )
            return "\n".join(lines)

        # ── Compare with base case ───────────────────────────────────────
        if "compare with base" in q or "changes from base" in q:
            if self._base_opflow_result is None:
                return "Base case results not available for comparison."
            base = self._base_opflow_result
            curr = opf
            lines = ["Comparison: current vs base case:"]
            if self._config.search.application == "pflow":
                base_cost = base.compute_generation_cost(self._base_network.gencost)
                curr_gencost = self._current_network.gencost if self._current_network else self._base_network.gencost
                curr_cost = curr.compute_generation_cost(curr_gencost)
                delta = curr_cost - base_cost
                pct = delta / base_cost * 100 if base_cost != 0 else 0
                lines.append(
                    f"  Computed cost: ${base_cost:,.2f} → ${curr_cost:,.2f}"
                    f"  ({delta:+,.2f}, {pct:+.1f}%)"
                )
            elif base.objective_value is not None and curr.objective_value is not None:
                delta = curr.objective_value - base.objective_value
                pct = delta / base.objective_value * 100 if base.objective_value != 0 else 0
                lines.append(
                    f"  Cost:       ${base.objective_value:,.2f} → ${curr.objective_value:,.2f}"
                    f"  ({delta:+,.2f}, {pct:+.1f}%)"
                )
            lines.append(
                f"  Voltage:    [{base.voltage_min:.4f}, {base.voltage_max:.4f}] → "
                f"[{curr.voltage_min:.4f}, {curr.voltage_max:.4f}] pu"
            )
            lines.append(
                f"  Generation: {base.total_gen_mw:.2f} → {curr.total_gen_mw:.2f} MW "
                f"({curr.total_gen_mw - base.total_gen_mw:+.2f})"
            )
            lines.append(
                f"  Max loading:{base.max_line_loading_pct:.1f}% → "
                f"{curr.max_line_loading_pct:.1f}% "
                f"({curr.max_line_loading_pct - base.max_line_loading_pct:+.1f}pp)"
            )
            return "\n".join(lines)

        # ── LLM fallback for unrecognized queries ────────────────────────
        logger.info("Analysis query not matched by patterns; using LLM fallback: %s", query)
        try:
            system = (
                "You are analyzing power grid simulation results. "
                "Answer the user's question based on the provided data. Be concise."
            )
            context = self._latest_results_text or "No results available."
            user = f"Question: {query}\n\nCurrent results:\n{context}"
            response = self._backend.complete(system, user)
            return response.raw_text
        except Exception as exc:
            logger.warning("LLM fallback for analysis failed: %s", exc)
            return (
                f"Query not matched and LLM fallback failed: {exc}\n"
                "Available pattern queries:\n"
                "  - 'voltage below/above X'\n"
                "  - 'most loaded lines'\n"
                "  - 'generators'\n"
                "  - '<N>kV voltage profile'\n"
                "  - 'area N summary'\n"
                "  - 'cost breakdown'\n"
                "  - 'constraint margins'\n"
                "  - 'compare with base'"
            )

    # ------------------------------------------------------------------
    # Multi-objective helpers
    # ------------------------------------------------------------------

    def _extract_initial_objectives(self, goal: str) -> None:
        """Use the LLM to extract tracked objectives from the initial goal."""
        try:
            sys_prompt, user_prompt = build_objective_extraction_prompt(
                text=goal,
                available_metrics=available_metrics_for_app(self._config.search.application),
                context="initial_goal",
            )
            response = self._backend.complete(sys_prompt, user_prompt)
            parsed = parse_objective_extraction(response.raw_text)
            if parsed:
                for obj_data in parsed:
                    entry = ObjectiveEntry(
                        name=obj_data["name"],
                        direction=obj_data["direction"],
                        threshold=obj_data.get("threshold"),
                        priority=obj_data["priority"],
                        introduced_at=0,
                        source="initial",
                    )
                    self._journal.objective_registry.register(entry)
                self._print(
                    f"[Objectives] Registered {len(parsed)} objective(s): "
                    f"{', '.join(o['name'] for o in parsed)}"
                )
            else:
                self._journal.objective_registry.register(ObjectiveEntry(
                    name="generation_cost",
                    direction="minimize",
                    priority="primary",
                    introduced_at=0,
                    source="initial",
                ))
                self._print("[Objectives] Defaulted to generation_cost (minimize)")
        except Exception as exc:
            logger.warning("Failed to extract initial objectives: %s", exc)
            self._journal.objective_registry.register(ObjectiveEntry(
                name="generation_cost",
                direction="minimize",
                priority="primary",
                introduced_at=0,
                source="initial",
            ))

    def _extract_objectives_from_steering(self, directive: str, iteration: int) -> None:
        """Extract any new objectives from a steering directive."""
        try:
            sys_prompt, user_prompt = build_objective_extraction_prompt(
                text=directive,
                available_metrics=available_metrics_for_app(self._config.search.application),
                context="steering_directive",
            )
            response = self._backend.complete(sys_prompt, user_prompt)
            parsed = parse_objective_extraction(response.raw_text)
            if parsed:
                for obj_data in parsed:
                    entry = ObjectiveEntry(
                        name=obj_data["name"],
                        direction=obj_data["direction"],
                        threshold=obj_data.get("threshold"),
                        priority=obj_data["priority"],
                        introduced_at=iteration,
                        source="steering",
                    )
                    self._journal.objective_registry.register(entry)
                self._print(
                    f"[Objectives] Steering added/updated {len(parsed)} objective(s): "
                    f"{', '.join(o['name'] for o in parsed)}"
                )
                self._backfill_metrics()
        except Exception as exc:
            logger.warning("Failed to extract objectives from steering: %s", exc)

    def _backfill_metrics(self) -> None:
        """Backfill tracked metrics for all past iterations from stored OPFLOW results.

        Called when new objectives are registered mid-search, so earlier
        iterations get the newly tracked metric values.
        """
        metric_names = [o.name for o in self._journal.objective_registry.objectives]
        for entry in self._journal.entries:
            if entry.mode == "analyze":
                continue
            opflow = self._opflow_results_cache.get(entry.iteration)
            if opflow is not None:
                entry.tracked_metrics = extract_all_metrics(opflow, metric_names)

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------

    def _assemble_prompt(
        self,
        goal: str,
        latest_results_text: Optional[str],
        error_feedback: Optional[str] = None,
        steering_directives: list[dict] | None = None,
        current_iteration: Optional[int] = None,
    ) -> tuple[str, str]:
        """Assemble the system prompt and user prompt for the LLM."""
        journal_text = (
            self._journal.format_for_prompt() if len(self._journal) > 0 else None
        )

        # Multi-objective context
        multi_obj_text = None
        registry = self._journal.objective_registry
        if registry.objectives:
            parts = [registry.format_for_prompt()]
            mo_summary = self._journal.format_multi_objective_summary()
            if mo_summary:
                parts.append("")
                parts.append(mo_summary)
            multi_obj_text = "\n".join(parts)

        # If explore cache is active, inject variant comparison text
        explore_text = None
        if self._explore_cache is not None and self._latest_results_text is not None:
            explore_text = self._latest_results_text

        user_prompt = build_user_prompt(
            goal=goal,
            journal_text=journal_text,
            results_text=latest_results_text,
            error_feedback=error_feedback,
            steering_directives=steering_directives,
            multi_objective_text=multi_obj_text,
            explore_text=explore_text,
            session_best=self._journal.session_best,
            current_iteration=current_iteration,
            max_iterations=self._config.search.max_iterations,
            benchmark_result=self._benchmark_result,
        )
        return self._system_prompt, user_prompt

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def _finalize(self, session: SearchSession, elapsed_seconds: float) -> None:
        """Print summary and save journal."""
        total_tokens = self._total_prompt_tokens + self._total_completion_tokens

        # --- Post-search goal classification via LLM ---
        goal_classification: Optional[dict] = None
        analysis_text: Optional[str] = None
        try:
            raw_stats = self._journal.summary_stats()
            sys_prompt, user_prompt = build_classification_prompts(
                goal=session.goal,
                termination_reason=session.termination_reason,
                stats=raw_stats,
                journal_formatted=self._journal.format_detailed(),
                total_tokens=total_tokens,
                objective_registry=self._journal.objective_registry.to_dict_list(),
                preference_history=self._journal.objective_registry.history,
                application=self._config.search.application,
            )
            response = self._backend.complete(sys_prompt, user_prompt)
            analysis_text = response.raw_text
            valid_iters = {e.iteration for e in self._journal.entries}
            goal_classification = parse_goal_classification(analysis_text, valid_iters)
        except Exception as exc:
            logger.warning("Post-search goal classification failed: %s", exc)

        # Resolve stats with override if classification succeeded
        best_iter_override = (
            goal_classification["best_iteration"] if goal_classification else None
        )
        goal_type_override = (
            goal_classification["goal_type"] if goal_classification else None
        )
        stats = self._journal.summary_stats(
            best_iteration_override=best_iter_override,
            goal_type=goal_type_override,
        )

        # Store on session for downstream consumers (GUI, tests)
        session.goal_classification = goal_classification
        session.analysis_text = analysis_text
        session.objective_registry_data = self._journal.objective_registry.to_dict_list()
        session.preference_history = self._journal.objective_registry.history
        session.tcopflow_period_data = self._tcopflow_period_data if self._tcopflow_period_data else None
        session.tcopflow_dT_min = self._tcopflow_dT_min
        session.tcopflow_duration_min = self._tcopflow_duration_min
        session.tcopflow_is_coupling = self._tcopflow_is_coupling
        session.sopflow_num_scenarios = self._sopflow_num_scenarios

        # Always print the final summary (even in quiet mode)
        print()
        print("=" * 60)
        print("  AgentiGrid Search Complete")
        print("=" * 60)
        print(f"  Goal:           {session.goal}")
        print(f"  Application:    {session.application}")
        print(f"  Backend:        {self._backend.name()} ({self._config.llm.model})")
        print(
            f"  Iterations:     {stats['total_iterations']} "
            f"(of max {self._config.search.max_iterations})"
        )
        print(f"  Duration:       {elapsed_seconds:.1f} seconds")
        if total_tokens:
            print(
                f"  Tokens used:    ~{total_tokens:,} "
                f"(prompt: {self._total_prompt_tokens:,}, "
                f"completion: {self._total_completion_tokens:,})"
            )
        print(f"  Termination:    {session.termination_reason}")

        # Goal-aware best-solution line
        goal_type = stats.get("goal_type") or "cost_minimization"
        best_iter = stats["best_iteration"]
        best_obj = stats["best_objective"]
        rationale = (
            goal_classification["best_iteration_rationale"] if goal_classification else None
        )

        if best_obj is None:
            print("  Best solution:  N/A (no feasible solution found)")
        elif self._config.search.application == "pflow":
            best_entry = None
            for e in self._journal.entries:
                if e.iteration == best_iter:
                    best_entry = e
                    break
            feasible_str = "feasible" if best_entry and best_entry.feasible else "infeasible"
            print(
                f"  Best solution:  iteration {best_iter} ({feasible_str})"
            )
            if rationale:
                print(f"  Rationale:      {rationale}")
            print(f"  Search type:    {goal_type.replace('_', ' ')}")
        elif goal_type == "cost_minimization":
            print(
                f"  Best objective: ${best_obj:,.2f} "
                f"(iteration {best_iter})"
            )
        else:
            cost_str = f"${best_obj:,.2f}" if best_obj is not None else "N/A"
            print(f"  Best solution:  iteration {best_iter} — cost={cost_str}")
            if rationale:
                print(f"  Rationale:      {rationale}")
            print(f"  Search type:    {goal_type.replace('_', ' ')}")

        # Print findings if complete
        findings = getattr(self, "_final_findings", None)
        if findings:
            session.final_findings = findings
            summary = findings.get("summary", "")
            if summary:
                print(f"\n  Findings: {summary}")

        print("=" * 60)

        # Print journal table
        print()
        print(self._journal.format_for_prompt())

        # Print truncated analysis text
        if analysis_text:
            print()
            print("─" * 60)
            print("  Post-Search Analysis:")
            print("─" * 60)
            snippet = analysis_text[:500]
            if len(analysis_text) > 500:
                snippet += f"\n  ... ({len(analysis_text) - 500} chars truncated)"
            print(snippet)

        # Multi-objective summary
        registry = self._journal.objective_registry
        if registry.is_multi_objective:
            print()
            print("─" * 60)
            print("  Multi-Objective Summary:")
            print("─" * 60)
            for obj in registry.objectives:
                dir_str = obj.direction
                if obj.direction == "constraint" and obj.threshold is not None:
                    dir_str = f"constraint (\u2264 {obj.threshold})"
                print(f"  {obj.name}: {dir_str} [{obj.priority}] (from iter {obj.introduced_at})")
            if goal_classification and goal_classification.get("tradeoff_summary"):
                print()
                print(f"  Tradeoffs: {goal_classification['tradeoff_summary']}")
            if goal_classification and goal_classification.get("recommended_solutions"):
                recs = goal_classification["recommended_solutions"]
                if len(recs) > 1:
                    print(f"  Recommended solutions: iterations {recs}")

        # PFLOW vs OPFLOW benchmark
        if (
            self._config.search.application == "pflow"
            and self._config.search.benchmark_opflow
        ):
            try:
                from agentigrid.engine.benchmark import run_pflow_vs_opflow_benchmark

                best_iter_num = stats.get("best_iteration")
                goal_type_str = stats.get("goal_type") or goal_type
                pflow_best = self._opflow_results_cache.get(best_iter_num) if best_iter_num is not None else None
                bresult = run_pflow_vs_opflow_benchmark(
                    base_case_path=self._config.search.base_case,
                    pflow_journal=self._journal,
                    config=self._config,
                    goal_type=goal_type_str,
                    pflow_best_result=pflow_best,
                )
                session.benchmark_result = _benchmark_to_dict(bresult)
                self._journal.benchmark_result = _benchmark_to_dict(bresult)
                print()
                print(bresult.summary_text)
            except Exception as exc:
                logger.warning("PFLOW vs OPFLOW benchmark failed: %s", exc)

        # Save journal if configured
        if self._config.output.save_journal:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fmt = self._config.output.journal_format
            journal_path = (
                self._config.output.workdir / f"journal_{timestamp}.{fmt}"
            )
            journal_path.parent.mkdir(parents=True, exist_ok=True)
            if fmt == "csv":
                self._journal.export_csv(journal_path)
            else:
                self._journal.export_json(journal_path)
            print(f"\nJournal saved to: {journal_path}")

    # ------------------------------------------------------------------
    # Session save/resume
    # ------------------------------------------------------------------

    def resume_from(self, save_dir: Path) -> SearchSession:
        """Resume a search from a saved session checkpoint.

        Loads the saved state (journal, objectives, network, steering),
        restores the controller's internal state, and continues the
        agent loop from the next iteration.

        Args:
            save_dir: Path to the saved session directory.

        Returns:
            Completed SearchSession.
        """
        from agentigrid.engine.session_io import load_session

        saved = load_session(save_dir)

        # Restore journal
        for entry in saved["journal_entries"]:
            self._journal.add_entry(entry)

        # Restore objective registry
        self._journal.objective_registry = saved["objective_registry"]

        # Restore steering state
        self._steering_history = saved["steering_history"]
        self._active_steering_directives = saved["active_steering_directives"]

        # Restore TCOPFLOW period data
        self._tcopflow_period_data = saved.get("tcopflow_period_data") or []
        self._tcopflow_dT_min = saved.get("tcopflow_dT_min", 0.0)
        self._tcopflow_duration_min = saved.get("tcopflow_duration_min", 0.0)
        self._tcopflow_is_coupling = saved.get("tcopflow_is_coupling", True)
        if self._tcopflow_period_data:
            self._tcopflow_num_steps = max(len(self._tcopflow_period_data), 0)
        else:
            self._tcopflow_num_steps = max(
                (e.num_steps for e in saved["journal_entries"] if e.num_steps > 0), default=0
            )

        # Restore SOPFLOW scenario count and override
        self._sopflow_num_scenarios = saved.get("sopflow_num_scenarios", 0)
        if not self._sopflow_num_scenarios:
            self._sopflow_num_scenarios = max(
                (e.num_scenarios for e in saved["journal_entries"] if e.num_scenarios > 0), default=0
            )
        _sopflow_scenario_str = saved.get("sopflow_scenario_override")
        self._sopflow_scenario_override = Path(_sopflow_scenario_str) if _sopflow_scenario_str else None

        # Restore TCOPFLOW profile overrides
        _saved_profile_overrides = saved.get("tcopflow_profile_overrides")
        if _saved_profile_overrides:
            self._tcopflow_profile_overrides = {k: Path(v) for k, v in _saved_profile_overrides.items()}
        else:
            self._tcopflow_profile_overrides = {}

        # Restore token counts
        self._total_prompt_tokens = saved["total_prompt_tokens"]
        self._total_completion_tokens = saved["total_completion_tokens"]

        # Parse base case and restore networks
        base_case = saved["base_case_path"]
        goal = saved["goal"]
        self._current_goal = goal

        self._base_network = parse_matpower(base_case)
        self._current_network = saved["current_network"] or self._base_network

        # Rebuild system prompt
        net_summary_text = network_summary(self._base_network)
        net_metadata_text = network_metadata(self._base_network)
        self._network_metadata_text = net_metadata_text
        # Restore load_factor from journal if saved
        if self._journal.load_factor is not None:
            self._session_load_factor = self._journal.load_factor
        self._system_prompt = build_system_prompt(
            command_schema=command_schema_text(),
            network_summary=net_summary_text,
            application=self._config.search.application,
            search_mode=self._config.search.search_mode,
            concurrent_pflow=self._config.search.concurrent_pflow,
            network_metadata=net_metadata_text,
            benchmark_text=None,  # benchmark only shown in fresh sessions
            session_load_factor=self._session_load_factor,
        )

        self._latest_results_text = None

        # Handle explore cache: if an explore was in progress at save time,
        # clear it — the variant results cannot be fully restored, so the
        # LLM will need to re-explore from the current network state.
        self._explore_cache = None
        explore_info = saved.get("explore_cache_info")
        if explore_info and explore_info.get("was_active"):
            labels = explore_info.get("variant_labels", [])
            self._print(
                f"[Resume] Note: an explore action was in progress at save time "
                f"(iteration {explore_info.get('iteration', '?')}, "
                f"variants: {', '.join(labels)}). The variant results have been "
                f"discarded — the LLM will need to re-explore."
            )
            logger.info(
                "Cleared explore cache from saved session (variants: %s)",
                labels,
            )

        last_iteration = saved["last_iteration"]

        self._print(
            f"[Resume] Loaded session with {len(self._journal)} entries, "
            f"resuming from iteration {last_iteration + 1}"
        )

        # Notify callback for each restored entry so GUI can display them
        if self._on_iteration:
            for entry in saved["journal_entries"]:
                self._on_iteration(entry.iteration, entry, "restored", None)

        # Build session
        session_start = time.monotonic()
        session = SearchSession(
            goal=goal,
            application=saved["application"],
            base_case_path=base_case,
            config=self._config,
            journal=self._journal,
            start_time=datetime.now().isoformat(),
        )

        # Continue the agent loop from last_iteration + 1
        max_iter = self._config.search.max_iterations
        for iteration in range(last_iteration + 1, max_iter + 1):
            if self._stop_requested:
                session.termination_reason = "user_stopped"
                self._print("\nSearch stopped by user.")
                break
            action_type, should_continue = self._iteration(iteration, goal)
            if self._on_iteration:
                latest_entry = self._journal.latest
                if latest_entry:
                    self._on_iteration(iteration, latest_entry, action_type, self._latest_opflow)
            if not should_continue:
                if not session.termination_reason:
                    session.termination_reason = "completed"
                break
        else:
            session.termination_reason = "max_iterations"
            self._print(f"\nMax iterations ({max_iter}) reached.")

        if not session.termination_reason:
            session.termination_reason = "completed"

        # Final notification
        if self._on_iteration:
            latest_entry = self._journal.latest
            if latest_entry:
                self._on_iteration(
                    latest_entry.iteration, latest_entry,
                    session.termination_reason, self._latest_opflow,
                )

        session.end_time = datetime.now().isoformat()
        session.total_prompt_tokens = self._total_prompt_tokens
        session.total_completion_tokens = self._total_completion_tokens

        if self._current_network is not None:
            limits = _bus_limits_from_network(self._current_network)
            if limits:
                session.enforced_vmin = min(v[0] for v in limits.values())
                session.enforced_vmax = max(v[1] for v in limits.values())

        elapsed = time.monotonic() - session_start
        self._finalize(session, elapsed)
        return session

    def _serialize_explore_cache(self) -> dict | None:
        """Serialize explore cache metadata for session persistence.

        The full variant results (networks, simulation results) are not
        persisted because they're too large. On resume, the explore cache
        is cleared and the LLM starts fresh from the current network.
        We save only enough metadata to know an explore was in progress
        and log a warning on resume.
        """
        if self._explore_cache is None:
            return None
        cache = self._explore_cache
        return {
            "was_active": True,
            "description": cache.description,
            "reasoning": cache.reasoning,
            "iteration": cache.iteration,
            "variant_labels": list(cache.variants.keys()),
            "base_mode": cache.base_mode,
        }

    def save_session(self, save_dir: Path, config_path: Path | str | None = None) -> Path:
        """Save the current search state to disk for later resumption.

        Args:
            save_dir: Directory to save session files into.
            config_path: Path to the config YAML used (for reference).

        Returns:
            Path to the saved session directory.
        """
        from agentigrid.engine.session_io import save_session as _save_session

        last_entry = self._journal.latest
        last_iteration = last_entry.iteration if last_entry else 0

        enforced_vmin = None
        enforced_vmax = None
        if self._current_network is not None:
            limits = _bus_limits_from_network(self._current_network)
            if limits:
                enforced_vmin = min(v[0] for v in limits.values())
                enforced_vmax = max(v[1] for v in limits.values())

        return _save_session(
            save_dir=save_dir,
            goal=self._current_goal if hasattr(self, "_current_goal") else "",
            application=self._config.search.application,
            base_case_path=self._config.search.base_case,
            config_path=config_path,
            journal=self._journal,
            steering_history=self._steering_history,
            active_steering_directives=self._active_steering_directives,
            current_network=self._current_network,
            total_prompt_tokens=self._total_prompt_tokens,
            total_completion_tokens=self._total_completion_tokens,
            last_iteration=last_iteration,
            enforced_vmin=enforced_vmin,
            enforced_vmax=enforced_vmax,
            tcopflow_period_data=self._tcopflow_period_data if self._tcopflow_period_data else None,
            tcopflow_dT_min=self._tcopflow_dT_min,
            tcopflow_duration_min=self._tcopflow_duration_min,
            tcopflow_is_coupling=self._tcopflow_is_coupling,
            sopflow_num_scenarios=self._sopflow_num_scenarios,
            sopflow_scenario_override=str(self._sopflow_scenario_override) if self._sopflow_scenario_override else None,
            tcopflow_profile_overrides={k: str(v) for k, v in self._tcopflow_profile_overrides.items()} if self._tcopflow_profile_overrides else None,
            explore_cache_info=self._serialize_explore_cache(),
        )
