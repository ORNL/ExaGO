"""Session manager �� bridges Streamlit UI with the AgentiGrid AgentLoopController.

Manages a background thread running the search loop, collects iteration
updates via callbacks into a thread-safe queue, and stores key OPFLOWResult
objects for visualization.
"""

from __future__ import annotations

import logging
import queue
import threading
from pathlib import Path
from typing import Optional

from agentigrid.backends import create_backend
from agentigrid.config import load_config
from agentigrid.engine.agent_loop import AgentLoopController, SearchSession
from agentigrid.engine.goal_classifier import build_classification_prompts, parse_goal_classification
from agentigrid.engine.journal import JournalEntry
from agentigrid.parsers.opflow_results import OPFLOWResult

try:
    from config_builder import get_default_config_path
except ModuleNotFoundError:
    from launcher.config_builder import get_default_config_path

logger = logging.getLogger("launcher.session_manager")


class SessionManager:
    """Bridges Streamlit UI with the AgentiGrid AgentLoopController.

    Manages a background thread running the search loop, collects
    iteration updates via callbacks into a thread-safe queue, and
    stores key OPFLOWResult objects for visualization.
    """

    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._controller: Optional[AgentLoopController] = None
        self._session: Optional[SearchSession] = None
        self._update_queue: queue.Queue = queue.Queue()
        self._error: Optional[str] = None
        self._base_opflow: Optional[OPFLOWResult] = None
        self._best_opflow: Optional[OPFLOWResult] = None
        self._best_objective: Optional[float] = None
        self._opflow_by_iteration: dict[int, OPFLOWResult] = {}
        self._goal_classification: Optional[dict] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_search(
        self,
        config_overrides: dict,
        goal: str,
        config_path: str | Path | None = None,
    ) -> None:
        """Build config, create controller with callbacks, launch in background thread.

        Args:
            config_overrides: Dict from config_builder.build_config_overrides().
            goal: Natural-language search goal.
            config_path: Path to base config YAML. Defaults to configs/default_config.yaml.
        """
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("A search is already running.")
        # Reset controller reference from any previous run
        self._controller = None

        # Reset state
        self._session = None
        self._error = None
        self._base_opflow = None
        self._best_opflow = None
        self._best_objective = None
        self._opflow_by_iteration = {}
        self._goal_classification = None
        # Drain the queue
        while not self._update_queue.empty():
            try:
                self._update_queue.get_nowait()
            except queue.Empty:
                break

        # Build config
        if config_path is None:
            config_path = get_default_config_path()
        cfg = load_config(path=config_path, cli_overrides=config_overrides)

        # Create controller with callbacks
        controller = AgentLoopController(
            config=cfg,
            quiet=True,
            on_iteration=self._on_iteration_callback,
            on_phase=self._on_phase_callback,
            on_pause_state=self._on_pause_state_callback,
            on_explore=self._on_explore_callback,
        )
        self._controller = controller

        base_case_path = cfg.search.base_case

        # Launch background thread
        self._thread = threading.Thread(
            target=self._run_search,
            args=(controller, base_case_path, goal),
            daemon=True,
        )
        self._thread.start()

    def stop_search(self) -> None:
        """Request graceful stop of the running search."""
        if self._controller is not None:
            self._controller.request_stop()

    def poll_updates(self) -> list[dict]:
        """Non-blocking drain of the update queue.

        Returns all pending updates. Called by the Streamlit main loop
        during periodic reruns.

        Returns:
            List of update dicts (may be empty).
        """
        updates = []
        while True:
            try:
                updates.append(self._update_queue.get_nowait())
            except queue.Empty:
                break
        return updates

    def is_running(self) -> bool:
        """Check if the search thread is still alive."""
        return self._thread is not None and self._thread.is_alive()

    def get_session(self) -> SearchSession | None:
        """Get the completed SearchSession after search ends."""
        return self._session

    def get_error(self) -> str | None:
        """Get error message if the search thread crashed."""
        return self._error

    def get_base_opflow(self) -> OPFLOWResult | None:
        """Get the base case (iteration 0) OPFLOW result."""
        return self._base_opflow

    def get_best_opflow(self) -> OPFLOWResult | None:
        """Get the best feasible OPFLOW result found during the search.

        If a goal classification has been computed, returns the OPFLOW result
        for the LLM-classified best iteration. Otherwise falls back to
        the lowest-cost feasible result.
        """
        if self._goal_classification is not None:
            best_iter = self._goal_classification.get("best_iteration")
            if best_iter is not None and best_iter in self._opflow_by_iteration:
                return self._opflow_by_iteration[best_iter]
        return self._best_opflow

    def get_goal_classification(self) -> dict | None:
        """Get the LLM-determined goal classification, or None if not yet computed."""
        return self._goal_classification

    def get_rag_config(self) -> dict | None:
        """Effective RAG configuration for the run (mode + thresholds), or None."""
        if self._controller is not None and hasattr(self._controller, "_retriever"):
            r = self._controller._retriever
            if hasattr(r, "describe"):
                try:
                    return r.describe()
                except Exception:
                    return None
        return None

    def get_opflow_by_iteration(self, iteration: int) -> OPFLOWResult | None:
        """Get the OPFLOW result for a specific iteration."""
        return self._opflow_by_iteration.get(iteration)

    def inject_steering(self, directive: str, mode: str = "augment") -> None:
        """Forward a steering directive to the running controller."""
        if self._controller is not None:
            self._controller.inject_steering(directive, mode)

    def pause_search(self) -> None:
        """Pause the search at the next iteration boundary."""
        if self._controller is not None:
            self._controller.pause()

    def resume_search(self) -> None:
        """Resume a paused search."""
        if self._controller is not None:
            self._controller.resume()

    def is_paused(self) -> bool:
        """Check if the search is currently paused."""
        if self._controller is not None:
            return self._controller.is_paused()
        return False

    def get_steering_history(self) -> list[dict]:
        """Return the list of steering directives injected so far."""
        if self._controller is not None:
            return self._controller.steering_history
        return []

    def get_summary_analysis(self, session: SearchSession) -> str:
        """Generate an analytical summary of the completed search via a final LLM call.

        Also requests a structured goal classification (goal_type, best_iteration)
        from the LLM, parsed and stored in self._goal_classification.

        Args:
            session: The completed search session.

        Returns:
            Summary analysis text from the LLM.
        """
        # If the CLI path already ran classification during _finalize(), reuse it.
        if session.analysis_text is not None:
            if session.goal_classification is not None:
                self._goal_classification = session.goal_classification
            return session.analysis_text

        try:
            backend = create_backend(session.config.llm)
            stats = session.journal.summary_stats()
            total_tokens = session.total_prompt_tokens + session.total_completion_tokens

            objective_registry = None
            preference_history = None
            if hasattr(session.journal, "objective_registry"):
                objective_registry = session.journal.objective_registry.to_dict_list()
                preference_history = session.journal.objective_registry.history

            system_prompt, user_prompt = build_classification_prompts(
                goal=session.goal,
                termination_reason=session.termination_reason,
                stats=stats,
                journal_formatted=session.journal.format_detailed(),
                total_tokens=total_tokens,
                objective_registry=objective_registry,
                preference_history=preference_history,
                application=session.application,
                near_optimal_abs_tol=session.config.report.near_optimal_abs_tol,
            )

            response = backend.complete(system_prompt, user_prompt)
            raw_text = response.raw_text

            valid_iters = {e.iteration for e in session.journal.entries}
            self._goal_classification = parse_goal_classification(raw_text, valid_iters)

            return raw_text

        except Exception as exc:
            logger.warning("Failed to generate summary analysis: %s", exc)
            return f"Summary analysis could not be generated: {exc}"

    def get_objective_registry(self) -> list[dict] | None:
        """Get the objective registry data from the running or completed session."""
        if self._session and hasattr(self._session, "objective_registry_data"):
            return self._session.objective_registry_data
        if self._controller and hasattr(self._controller, "_journal"):
            return self._controller._journal.objective_registry.to_dict_list()
        return None

    def get_preference_history(self) -> list[dict] | None:
        """Get the preference evolution history."""
        if self._session and hasattr(self._session, "preference_history"):
            return self._session.preference_history
        if self._controller and hasattr(self._controller, "_journal"):
            return self._controller._journal.objective_registry.history
        return None

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _run_search(
        self, controller: AgentLoopController, base_case: Path, goal: str
    ) -> None:
        """Execute the search loop in a background thread."""
        try:
            session = controller.run(base_case, goal)
            self._session = session
        except Exception as exc:
            logger.exception("Search thread crashed: %s", exc)
            self._error = str(exc)
            self._update_queue.put({"type": "error", "message": str(exc)})
        finally:
            self._update_queue.put({"type": "search_finished"})

    def _on_iteration_callback(
        self,
        iteration: int,
        entry: JournalEntry,
        status: str,
        opflow_result: OPFLOWResult | None,
    ) -> None:
        """Called by AgentLoopController after each iteration."""
        # Store OPFLOW result indexed by iteration (for later lookup)
        if opflow_result is not None:
            self._opflow_by_iteration[iteration] = opflow_result

        # Store base case OPFLOW result
        if iteration == 0 and opflow_result is not None:
            self._base_opflow = opflow_result

        # Track best feasible OPFLOW result (by lowest cost — default heuristic)
        if entry.feasible and entry.objective_value is not None:
            if self._best_objective is None or entry.objective_value < self._best_objective:
                self._best_objective = entry.objective_value
                self._best_opflow = opflow_result

        # Enqueue update for the UI
        update = {
            "type": "iteration",
            "iteration": iteration,
            "timestamp": entry.timestamp,
            "description": entry.description,
            "status": status,
            "objective_value": entry.objective_value,
            "feasible": entry.feasible,
            "convergence_status": entry.convergence_status,
            "violations_count": entry.violations_count,
            "voltage_range": (entry.voltage_min, entry.voltage_max),
            "max_line_loading_pct": entry.max_line_loading_pct,
            "total_gen_mw": entry.total_gen_mw,
            "total_load_mw": entry.total_load_mw,
            "sim_elapsed": entry.elapsed_seconds,
            "llm_reasoning": entry.llm_reasoning,
            "commands_count": len(entry.commands),
            "commands": entry.commands,
            "mode": entry.mode,
            "tracked_metrics": entry.tracked_metrics,
            "explored_variants": entry.explored_variants if hasattr(entry, "explored_variants") else None,
        }
        self._update_queue.put(update)

    def _on_phase_callback(self, iteration: int, phase_name: str) -> None:
        """Called by AgentLoopController at phase transitions within an iteration."""
        self._update_queue.put({
            "type": "phase",
            "iteration": iteration,
            "phase": phase_name,
        })

    def _on_pause_state_callback(self, paused: bool) -> None:
        """Called by AgentLoopController when pause state changes."""
        self._update_queue.put({
            "type": "pause_state",
            "paused": paused,
        })

    def _on_explore_callback(self, iteration: int, variant_summaries: list[dict]) -> None:
        """Called by AgentLoopController after an explore action completes."""
        self._update_queue.put({
            "type": "explore_status",
            "iteration": iteration,
            "variant_summaries": variant_summaries,
        })

    # ------------------------------------------------------------------
    # Save / resume
    # ------------------------------------------------------------------

    def save_current_session(self, save_dir: Path | None = None) -> Path | None:
        """Save the current or completed session to disk.

        Args:
            save_dir: Directory to save into. If None, auto-generates
                a timestamped directory in workdir.

        Returns:
            Path to saved session directory, or None if nothing to save.
        """
        if self._controller is None and self._session is None:
            return None

        if save_dir is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = Path(f"workdir/saved_session_{timestamp}")

        if self._controller is not None:
            return self._controller.save_session(save_dir)

        # If search is completed, save from the session object
        if self._session is not None:
            from agentigrid.engine.session_io import save_session
            return save_session(
                save_dir=save_dir,
                goal=self._session.goal,
                application=self._session.application,
                base_case_path=self._session.base_case_path,
                config_path=None,
                journal=self._session.journal,
                steering_history=(
                    self._session.preference_history
                    if hasattr(self._session, "preference_history")
                    else []
                ),
                active_steering_directives=[],
                current_network=None,
                total_prompt_tokens=self._session.total_prompt_tokens,
                total_completion_tokens=self._session.total_completion_tokens,
                last_iteration=len(self._session.journal) - 1,
                termination_reason=self._session.termination_reason,
            )
        return None

    def resume_session(
        self,
        save_dir: Path,
        config_overrides: dict,
        config_path: str | Path | None = None,
    ) -> None:
        """Resume a saved session in a background thread.

        Args:
            save_dir: Directory containing the saved session.
            config_overrides: Config overrides from the GUI (for LLM backend, etc.).
            config_path: Path to base config YAML.
        """
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("A search is already running.")

        self._controller = None
        self._session = None
        self._error = None
        self._base_opflow = None
        self._best_opflow = None
        self._best_objective = None
        self._opflow_by_iteration = {}
        self._goal_classification = None
        while not self._update_queue.empty():
            try:
                self._update_queue.get_nowait()
            except queue.Empty:
                break

        if config_path is None:
            config_path = get_default_config_path()
        cfg = load_config(path=config_path, cli_overrides=config_overrides)

        controller = AgentLoopController(
            config=cfg,
            quiet=True,
            on_iteration=self._on_iteration_callback,
            on_phase=self._on_phase_callback,
            on_pause_state=self._on_pause_state_callback,
            on_explore=self._on_explore_callback,
        )
        self._controller = controller

        self._thread = threading.Thread(
            target=self._run_resume,
            args=(controller, save_dir),
            daemon=True,
        )
        self._thread.start()

    def _run_resume(self, controller: AgentLoopController, save_dir: Path) -> None:
        """Resume a saved session in a background thread."""
        try:
            session = controller.resume_from(save_dir)
            self._session = session
        except Exception as exc:
            logger.exception("Resume thread crashed: %s", exc)
            self._error = str(exc)
            self._update_queue.put({"type": "error", "message": str(exc)})
        finally:
            self._update_queue.put({"type": "search_finished"})
