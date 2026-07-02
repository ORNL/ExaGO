"""AgentiGrid Launcher — Streamlit GUI for LLM-driven power grid optimization."""

# ── Path Setup (must be first) ───────────────────────────────────────────────
# Ensure the project root is on sys.path so that agentigrid can be imported.
# Streamlit adds launcher/ (the script's directory) to sys.path, but not
# the project root where the agentigrid package lives.
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ── Imports ──────────────────────────────────────────────────────────────────

import os
import streamlit as st
import time
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go

from config_builder import (
    scan_data_files, scan_contingency_files, scan_profile_files,
    scan_scenario_files, match_profiles_for_case, match_scenarios_for_case,
    load_example_goals,
    build_config_overrides, get_default_config_path, get_project_root,
    DEFAULT_MODELS, BACKENDS, APPLICATIONS, FUTURE_APPLICATIONS, MODES,
    SEARCH_MODES,
)
from session_manager import SessionManager
from charts import (
    convergence_chart, voltage_range_chart, voltage_profile_chart,
    generator_dispatch_chart, line_loading_chart, multi_objective_trend_chart,
    reserve_trajectory_chart,
)

from agentigrid.parsers import parse_matpower, network_summary

# ── Page Configuration ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="AgentiGrid Launcher",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Session State Initialization ─────────────────────────────────────────────

def init_session_state():
    """Initialize session state variables."""
    defaults = {
        "search_running": False,
        "search_finished": False,
        "session_manager": None,
        "iteration_log": [],
        "current_phase": "",
        "current_iteration": 0,
        "stop_requested": False,
        "search_session": None,
        "search_error": None,
        "summary_analysis": None,
        "base_opflow": None,
        "best_opflow": None,
        "goal_classification": None,
        "completed_sessions": [],
        "selected_goal_label": "Custom",
        "steering_history": [],
        "search_paused": False,
        "explore_status": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


init_session_state()


# ── Network Info Caching ─────────────────────────────────────────────────────

@st.cache_data
def get_network_info(file_path: str) -> dict:
    """Parse a MATPOWER file and return summary info."""
    net = parse_matpower(Path(file_path))
    return {
        "buses": len(net.buses),
        "generators": len(net.generators),
        "branches": len(net.branches),
        "total_load_mw": sum(b.Pd for b in net.buses),
        "total_gen_capacity_mw": sum(g.Pmax for g in net.generators),
        "summary_text": network_summary(net),
    }


# ── Start Search Logic ───────────────────────────────────────────────────────

def start_search(base_case_path, goal, backend, model, temperature,
                   application, mode, max_iterations, search_mode="standard",
                   ctgc_file=None, mpi_np=1,
                   pload_profile=None, qload_profile=None, wind_profile=None,
                   tcopflow_duration=1.0, tcopflow_dT=60.0, tcopflow_iscoupling=1,
                   scenario_file=None, sopflow_solver="IPOPT", sopflow_iscoupling=0,
                   benchmark_opflow=False, concurrent_pflow=False, max_variants=8,
                   sweep_max_workers=0, load_factor=None):
    """Initialize and start a new search."""
    # Validate base case still exists
    if not Path(base_case_path).exists():
        st.error(f"Base case file no longer exists: {base_case_path}")
        return

    # Warn about missing API keys
    if backend == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        st.warning("ANTHROPIC_API_KEY environment variable not set. "
                   "The search may fail if the key is not available.")
    elif backend == "openai" and not os.environ.get("OPENAI_API_KEY"):
        st.warning("OPENAI_API_KEY environment variable not set. "
                   "The search may fail if the key is not available.")

    overrides = build_config_overrides(
        base_case=str(base_case_path),
        backend=backend,
        model=model,
        temperature=temperature,
        application=application,
        default_mode=mode,
        max_iterations=max_iterations,
        search_mode=search_mode,
        ctgc_file=ctgc_file,
        mpi_np=mpi_np,
        pload_profile=pload_profile,
        qload_profile=qload_profile,
        wind_profile=wind_profile,
        tcopflow_duration=tcopflow_duration,
        tcopflow_dT=tcopflow_dT,
        tcopflow_iscoupling=tcopflow_iscoupling,
        scenario_file=scenario_file,
        sopflow_solver=sopflow_solver,
        sopflow_iscoupling=sopflow_iscoupling,
        benchmark_opflow=benchmark_opflow,
        concurrent_pflow=concurrent_pflow,
        max_variants=max_variants,
        sweep_max_workers=sweep_max_workers,
        load_factor=load_factor,
    )

    if st.session_state.session_manager is None:
        st.session_state.session_manager = SessionManager()

    manager = st.session_state.session_manager

    # Clear previous results
    st.session_state.iteration_log = []
    st.session_state.current_phase = ""
    st.session_state.current_iteration = 0
    st.session_state.search_finished = False
    st.session_state.search_session = None
    st.session_state.search_error = None
    st.session_state.summary_analysis = None
    st.session_state.base_opflow = None
    st.session_state.best_opflow = None
    st.session_state.goal_classification = None
    st.session_state.stop_requested = False
    st.session_state.steering_history = []
    st.session_state.search_paused = False
    st.session_state.explore_status = None

    try:
        manager.start_search(config_overrides=overrides, goal=goal)
        st.session_state.search_running = True
        st.session_state.current_application = application
    except Exception as e:
        st.error(f"Failed to start search: {e}")


# ── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar() -> dict:
    """Render the sidebar configuration panel and return config values."""
    disabled = st.session_state.search_running

    with st.sidebar:
        st.title("⚡ AgentiGrid")
        st.caption("v0.1.0 — LLM-Driven Power Grid Optimization")

        # ── Base Case ────────────────────────────────────────────────────
        st.header("📁 Base Case")
        data_files = scan_data_files()

        if not data_files:
            st.warning("No .m files found in data/ directory")
            selected_file = None
        else:
            display_names = [f.name for f in data_files]
            selected_idx = st.selectbox(
                "Base case file",
                range(len(data_files)),
                format_func=lambda i: display_names[i],
                disabled=disabled,
            )
            selected_file = data_files[selected_idx]
            st.caption(f"`{selected_file}`")

        # ── LLM Backend ─────────────────────────────────────────────────
        st.header("🤖 LLM Backend")
        backend = st.selectbox("Backend", BACKENDS, disabled=disabled)
        model_key = f"model_input_{backend}"
        model = st.text_input(
            "Model",
            value=DEFAULT_MODELS.get(backend, ""),
            key=model_key,
            disabled=disabled,
        )
        temperature = st.slider(
            "Temperature", 0.0, 1.0, 0.3, step=0.05,
            disabled=disabled,
        )

        # ── Search Parameters ────────────────────────────────────────────
        st.header("⚙️ Search Parameters")
        application = st.selectbox(
            "Application", APPLICATIONS, disabled=disabled,
        )
        if FUTURE_APPLICATIONS:
            st.caption(f"Coming soon: {', '.join(FUTURE_APPLICATIONS)}")
        if application == "pflow":
            st.caption(
                "PFLOW solves power flow equations — there is no cost optimization. "
                "The LLM controls the search strategy. Enable 'Concurrent explore/select' "
                "below to allow the LLM to evaluate multiple configurations in parallel."
            )

        # Contingency file selector (SCOPFLOW only)
        ctgc_file = None
        if application == "scopflow":
            cont_files = scan_contingency_files()
            if cont_files:
                cont_names = [f.name for f in cont_files]
                cont_idx = st.selectbox(
                    "Contingency file (.cont)",
                    range(len(cont_files)),
                    format_func=lambda i: cont_names[i],
                    disabled=disabled,
                )
                ctgc_file = cont_files[cont_idx]
                st.caption(f"`{ctgc_file}`")
            else:
                st.warning(
                    "No .cont files found in data/ directory. "
                    "SCOPFLOW requires a contingency file."
                )

        # Load profile selectors (TCOPFLOW only)
        pload_profile = None
        qload_profile = None
        wind_profile = None
        tcopflow_duration = 1.0
        tcopflow_dT = 60.0
        tcopflow_iscoupling = 1
        if application == "tcopflow":
            # Auto-match profiles to selected base case
            profile_matches = (
                match_profiles_for_case(selected_file)
                if selected_file else {"pload": [], "qload": []}
            )
            p_matches = profile_matches["pload"]
            q_matches = profile_matches["qload"]

            if p_matches:
                p_names = [f.name for f in p_matches]
                p_idx = st.selectbox(
                    "Active load profile (P)",
                    range(len(p_matches)),
                    format_func=lambda i: p_names[i],
                    disabled=disabled,
                )
                pload_profile = p_matches[p_idx]
                st.caption(f"`{pload_profile}`")
            else:
                st.warning(
                    "No *_load_P.csv files found in data/ directory. "
                    "TCOPFLOW requires an active load profile."
                )

            if q_matches:
                q_names = [f.name for f in q_matches]
                q_idx = st.selectbox(
                    "Reactive load profile (Q)",
                    range(len(q_matches)),
                    format_func=lambda i: q_names[i],
                    disabled=disabled,
                )
                qload_profile = q_matches[q_idx]
                st.caption(f"`{qload_profile}`")
            else:
                st.warning(
                    "No *_load_Q.csv files found in data/ directory. "
                    "TCOPFLOW requires a reactive load profile."
                )

            # Optional wind profile
            all_profiles = scan_profile_files()
            wind_profiles = [p for p in all_profiles if "wind" in p.name.lower()]
            if wind_profiles:
                wind_names = [f.name for f in wind_profiles]
                wind_idx = st.selectbox(
                    "Wind generation profile (optional)",
                    range(len(wind_profiles) + 1),
                    format_func=lambda i: "None" if i == 0 else wind_names[i - 1],
                    disabled=disabled,
                )
                if wind_idx > 0:
                    wind_profile = wind_profiles[wind_idx - 1]

            # Temporal parameters
            st.subheader("⏱️ Temporal Parameters")
            tcopflow_duration = st.number_input(
                "Duration (hours)",
                min_value=0.1,
                max_value=168.0,
                value=1.0,
                step=0.5,
                disabled=disabled,
                help="Total time horizon for TCOPFLOW (default: 1.0 hour).",
            )
            tcopflow_dT = st.number_input(
                "Time-step (minutes)",
                min_value=1.0,
                max_value=1440.0,
                value=60.0,
                step=5.0,
                disabled=disabled,
                help="Time-step size for each period (default: 60 min).",
            )
            tcopflow_iscoupling = st.selectbox(
                "Generator ramp coupling",
                [1, 0],
                format_func=lambda x: "Enabled" if x == 1 else "Disabled",
                disabled=disabled,
                help="When enabled, generator ramp limits are enforced between periods.",
            )

        # Scenario file and solver for SOPFLOW only
        scenario_file = None
        sopflow_solver = "IPOPT"
        sopflow_iscoupling = 0
        if application == "sopflow":
            # Auto-match scenario files to selected base case
            scenario_matches = (
                match_scenarios_for_case(selected_file)
                if selected_file else []
            )
            if scenario_matches:
                scenario_names = [f.name for f in scenario_matches]
                scenario_idx = st.selectbox(
                    "Wind scenario file",
                    range(len(scenario_matches)),
                    format_func=lambda i: scenario_names[i],
                    disabled=disabled,
                    help="SOPFLOW requires a wind scenario file (-windgen). "
                    "Auto-matched to base case filename.",
                )
                scenario_file = scenario_matches[scenario_idx]
                st.caption(f"`{scenario_file}`")
            else:
                st.warning(
                    "No scenario files found in data/ directory. "
                    "SOPFLOW requires a wind scenario file."
                )

            sopflow_solver = st.selectbox(
                "SOPFLOW Solver",
                ["IPOPT", "EMPAR"],
                disabled=disabled,
                help="IPOPT: single-core deterministic solver. "
                "EMPAR: multi-core decomposition solver (requires MPI processes > 1).",
            )
            sopflow_iscoupling = st.selectbox(
                "First/second stage coupling",
                [0, 1],
                format_func=lambda x: "Enabled" if x == 1 else "Disabled",
                disabled=disabled,
                help="When enabled, first and second stage decisions are coupled.",
            )

        mode = st.selectbox("Mode", MODES, disabled=disabled)
        search_mode = st.selectbox(
            "Search Mode", SEARCH_MODES, disabled=disabled,
            help="Standard: goal-directed search. Stress Test: systematic contingency exploration.",
        )
        max_iterations = st.slider(
            "Max iterations", 1, 50, 20, disabled=disabled,
        )
        mpi_disabled = disabled or application not in ("scopflow", "sopflow")
        mpi_np = st.number_input(
            "MPI processes",
            min_value=1,
            max_value=64,
            value=1,
            disabled=mpi_disabled,
            help="Number of MPI processes for ExaGO (default: 1). "
            "SCOPFLOW and SOPFLOW support multi-core execution via EMPAR solver. "
            "OPFLOW, DCOPFLOW, TCOPFLOW, and PFLOW use single-core solvers only.",
        )
        if application not in ("scopflow", "sopflow"):
            mpi_np = 1

        benchmark_opflow = st.checkbox(
            "Benchmark vs OPFLOW",
            value=False,
            disabled=disabled or application != "pflow",
            help="After PFLOW search, run OPFLOW on the base case and compare "
            "cost, dispatch, and loadability results.",
        )
        if application != "pflow":
            benchmark_opflow = False

        concurrent_pflow = st.checkbox(
            "Concurrent explore/select",
            value=False,
            disabled=disabled or application != "pflow",
            help="Enable concurrent neighborhood search for PFLOW. The LLM can "
            "propose multiple variants per iteration, which are evaluated in "
            "parallel. A Pareto front identifies the best tradeoffs, and the "
            "LLM selects one variant as the new current point.",
        )
        max_variants = st.number_input(
            "Max variants per explore",
            min_value=2,
            max_value=16,
            value=8,
            disabled=disabled or application != "pflow" or not concurrent_pflow,
            help="Maximum number of PFLOW instances to run concurrently per "
            "explore action (default: 8). Each variant is a separate simulation.",
        )
        if application != "pflow":
            concurrent_pflow = False
            max_variants = 8

        parallel_sweep = st.checkbox(
            "Run sweep in parallel",
            value=True,
            disabled=disabled or application != "opflow",
            help="Solve sweep candidate buses concurrently. Disable for sequential "
                 "(serial) execution — useful for debugging or reproducibility.",
        )
        _default_workers = min(os.cpu_count() or 4, 16)
        sweep_workers = st.number_input(
            "Concurrent solves",
            min_value=1,
            max_value=128,
            value=_default_workers,
            step=1,
            disabled=disabled or application != "opflow" or not parallel_sweep,
            help="Number of OPFLOW solves to run at once. Set near your physical core "
                 "count. Each solve is pinned to a single BLAS/OpenMP thread to avoid "
                 "oversubscription.",
        )
        if application == "opflow":
            sweep_max_workers = int(sweep_workers) if parallel_sweep else 1
        else:
            sweep_max_workers = 0

        load_factor_input = st.number_input(
            "Session load factor",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            step=0.01,
            format="%.2f",
            disabled=disabled or application != "pflow",
            help="Session-level load scaling factor for PFLOW (e.g., 1.23 = 123% of base load). "
            "Automatically injected into every run — the LLM does not need to include "
            "scale_all_loads. Set to 1.0 to use the network's native load level.",
        )
        if application != "pflow" or load_factor_input == 1.0:
            load_factor = None
        else:
            load_factor = float(load_factor_input)

        # ── Search Goal ──────────────────────────────────────────────────
        st.header("🎯 Search Goal")
        example_goals = load_example_goals()
        goal_labels = ["Custom"] + [g["label"] for g in example_goals]

        selected_goal_label = st.selectbox(
            "Example Goals",
            goal_labels,
            index=goal_labels.index(st.session_state.selected_goal_label)
            if st.session_state.selected_goal_label in goal_labels
            else 0,
            disabled=disabled,
        )
        st.session_state.selected_goal_label = selected_goal_label

        # Determine default goal text
        goal_default = ""
        if selected_goal_label != "Custom":
            for g in example_goals:
                if g["label"] == selected_goal_label:
                    goal_default = g["goal"]
                    break

        goal = st.text_area(
            "Goal",
            value=goal_default,
            height=150,
            disabled=disabled,
        )

        # ── Actions ──────────────────────────────────────────────────────
        st.header("🚀 Actions")

        can_start = (
            not disabled
            and selected_file is not None
            and goal.strip() != ""
        )

        if st.button("▶️ Start Search", disabled=not can_start, type="primary"):
            start_search(
                base_case_path=selected_file,
                goal=goal.strip(),
                backend=backend,
                model=model,
                temperature=temperature,
                application=application,
                mode=mode,
                max_iterations=max_iterations,
                search_mode=search_mode,
                ctgc_file=ctgc_file,
                mpi_np=mpi_np,
                pload_profile=pload_profile,
                qload_profile=qload_profile,
                wind_profile=wind_profile,
                tcopflow_duration=tcopflow_duration,
                tcopflow_dT=tcopflow_dT,
                tcopflow_iscoupling=tcopflow_iscoupling,
                scenario_file=scenario_file,
                sopflow_solver=sopflow_solver,
                sopflow_iscoupling=sopflow_iscoupling,
                benchmark_opflow=benchmark_opflow,
                concurrent_pflow=concurrent_pflow,
                max_variants=max_variants,
                sweep_max_workers=sweep_max_workers,
                load_factor=load_factor,
            )
            st.rerun()

        if disabled:
            if st.button("⏹️ Stop Search", type="secondary"):
                manager = st.session_state.session_manager
                if manager is not None:
                    manager.stop_search()
                st.session_state.stop_requested = True
            if st.session_state.stop_requested:
                st.warning("Stop requested — waiting for current iteration to finish...")

        # ── Session History ────────────────────────────────────────────────
        if st.session_state.completed_sessions:
            st.header("📋 History")
            for i, sess_info in enumerate(st.session_state.completed_sessions):
                st.caption(
                    f"{i + 1}. {sess_info['goal'][:40]} — {sess_info['best_obj']}"
                )

        # ── Session Save/Resume ───────────────────────────────────────────
        st.markdown("---")
        st.subheader("💾 Session Save/Resume")

        save_enabled = (
            st.session_state.search_finished
            or st.session_state.search_running
        )
        if st.button("Save Session", disabled=not save_enabled):
            sm = st.session_state.session_manager
            if sm:
                save_path = sm.save_current_session()
                if save_path:
                    st.success(f"Saved to: {save_path}")
                else:
                    st.warning("Nothing to save.")

        saved_sessions_dir = Path("workdir")
        saved_dirs = sorted(
            [d for d in saved_sessions_dir.glob("saved_session_*") if d.is_dir()],
            reverse=True,
        ) if saved_sessions_dir.exists() else []

        if saved_dirs:
            selected_save = st.selectbox(
                "Resume from",
                ["(none)"] + [d.name for d in saved_dirs],
                disabled=st.session_state.search_running,
            )
            if selected_save != "(none)" and st.button(
                "Resume Search", disabled=st.session_state.search_running
            ):
                save_dir = saved_sessions_dir / selected_save
                overrides = build_config_overrides(
                    base_case=".",  # will be loaded from session
                    backend=backend,
                    model=model,
                    temperature=temperature,
                    application=application,
                    default_mode=mode,
                    max_iterations=max_iterations,
                    search_mode=search_mode,
                    mpi_np=mpi_np,
                )
                sm = SessionManager()
                st.session_state.session_manager = sm
                st.session_state.search_running = True
                st.session_state.search_finished = False
                st.session_state.iteration_log = []
                st.session_state.search_session = None
                st.session_state.search_error = None
                sm.resume_session(save_dir, overrides)
                st.rerun()

    return {
        "base_case": selected_file,
        "backend": backend,
        "model": model,
        "temperature": temperature,
        "application": application,
        "ctgc_file": ctgc_file,
        "mode": mode,
        "search_mode": search_mode,
        "max_iterations": max_iterations,
        "mpi_np": mpi_np,
        "goal": goal,
    }


# ── Main Area — State Transitions ────────────────────────────────────────────

def render_main_area(config: dict):
    """Route to the appropriate view based on current state."""
    if st.session_state.search_running:
        render_live_monitor()
    elif st.session_state.search_finished:
        render_results()
    else:
        render_welcome(config.get("base_case"))


# ── State A: Welcome / Pre-Search ────────────────────────────────────────────

def render_welcome(base_case: Path | None):
    """Render the welcome screen before any search starts."""
    st.title("⚡ AgentiGrid — LLM-Driven Power Grid Optimization")
    st.markdown(
        "AgentiGrid uses large language models to iteratively explore and optimize "
        "power grid configurations via ExaGO simulations. It translates natural-language "
        "goals into concrete parameter modifications, runs simulations, and learns from "
        "the results to converge toward an optimal solution."
    )

    st.info(
        "Select a base case in the sidebar, choose your LLM backend, "
        "write a search goal, and click **Start Search**."
    )

    # Environment info
    with st.expander("ℹ️ Environment Info"):
        st.caption(f"Project root: {get_project_root()}")
        st.caption(f"Config: {get_default_config_path()}")
        st.caption(f"Data files: {len(scan_data_files())} found")
        for key in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"]:
            status = "Set" if os.environ.get(key) else "Not set"
            st.caption(f"{key}: {status}")

    # Network summary card if a base case is selected
    if base_case is not None:
        try:
            info = get_network_info(str(base_case))
            st.subheader("📊 Network Summary")
            st.caption(f"File: `{base_case.name}`")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Buses", info["buses"])
            c2.metric("Generators", info["generators"])
            c3.metric("Branches", info["branches"])
            c4.metric("Total Load", f"{info['total_load_mw']:.1f} MW")
            c5.metric("Gen Capacity", f"{info['total_gen_capacity_mw']:.1f} MW")
        except Exception as e:
            st.error(f"Failed to parse base case: {e}")


# ── State B: Live Monitor ─────────────────────────────────────────────────────

def _iteration_icon(entry: dict) -> str:
    """Return a status icon for an iteration log entry."""
    if entry.get("convergence_status") == "FAILED" or entry.get("status") == "FAILED":
        return "❌"
    if entry.get("convergence_status") == "COMPLETE":
        return "🏁"
    if entry.get("convergence_status") == "EXPLORE":
        return "🔍"
    if entry.get("convergence_status") == "ANALYSIS":
        return "🔬"
    if entry.get("feasible"):
        return "✅"
    return "⚠️"


def _format_command(cmd: dict) -> str:
    """Format a single command dict as a compact summary line."""
    action = cmd.get("action", "unknown")
    parts = [f"{action}:"]
    for k, v in cmd.items():
        if k == "action":
            continue
        parts.append(f"{k}={v}")
    return " ".join(parts)


def render_live_monitor():
    """Render the live search monitor with iteration timeline and charts."""

    # 1. Poll for updates
    manager = st.session_state.session_manager
    if manager is not None:
        updates = manager.poll_updates()
        for update in updates:
            if update["type"] == "search_finished":
                st.session_state.search_running = False
                st.session_state.search_finished = True
                st.session_state.search_session = manager.get_session()
                st.session_state.search_error = manager.get_error()
                st.session_state.base_opflow = manager.get_base_opflow()
                st.session_state.best_opflow = manager.get_best_opflow()
                st.session_state.goal_classification = manager.get_goal_classification()
                # Append to session history
                session = manager.get_session()
                if session:
                    stats = session.journal.summary_stats()
                    st.session_state.completed_sessions.append({
                        "goal": session.goal[:50],
                        "best_obj": f"${stats['best_objective']:,.2f}" if stats["best_objective"] and session.application != "pflow" else ("N/A (analysis)" if session.application == "pflow" else "N/A"),
                        "iterations": stats["total_iterations"],
                        "termination": session.termination_reason,
                    })
                st.rerun()
                return
            elif update["type"] == "iteration":
                st.session_state.iteration_log.append(update)
                st.session_state.current_iteration = update["iteration"]
            elif update["type"] == "phase":
                phase_labels = {
                    "llm_request": "Sending prompt to LLM...",
                    "applying_commands": "Applying modifications...",
                    "running_simulation": "Running simulation...",
                    "parsing_results": "Parsing results...",
                    "computing_pareto_front": "Computing Pareto front...",
                }
                raw_phase = update["phase"]
                if raw_phase.startswith("running_simulation ("):
                    st.session_state.current_phase = f"Running {raw_phase.split('(')[1].rstrip(')')}..."
                else:
                    st.session_state.current_phase = phase_labels.get(
                        raw_phase, raw_phase
                    )
            elif update["type"] == "pause_state":
                st.session_state.search_paused = update["paused"]
            elif update["type"] == "explore_status":
                st.session_state.explore_status = update
            elif update["type"] == "error":
                st.session_state.search_error = update["message"]

    # Check if thread died unexpectedly
    if manager is not None and not manager.is_running() and not st.session_state.search_finished:
        st.session_state.search_running = False
        st.session_state.search_finished = True
        st.session_state.search_session = manager.get_session()
        st.session_state.search_error = manager.get_error()
        st.session_state.base_opflow = manager.get_base_opflow()
        st.session_state.best_opflow = manager.get_best_opflow()
        st.session_state.goal_classification = manager.get_goal_classification()
        if not st.session_state.search_error:
            st.session_state.search_error = "Search thread terminated unexpectedly."
        st.rerun()
        return

    # 2. Header
    st.header("🔄 Search in Progress...")

    # 3. Two-column layout
    left_col, right_col = st.columns([2, 1])

    # ── Left Column: Iteration Timeline ──────────────────────────────────
    with left_col:
        for entry in st.session_state.iteration_log:
            icon = _iteration_icon(entry)
            obj = entry.get("objective_value")
            is_pflow_live = st.session_state.get("current_application") == "pflow"
            if is_pflow_live:
                obj_str = "analysis" if obj is not None else "FAILED"
            else:
                obj_str = f"${obj:,.2f}" if obj is not None else "FAILED"
            elapsed = entry.get("sim_elapsed")
            time_str = f"{elapsed:.1f}s" if elapsed is not None else "—"
            label = (
                f"{icon} Iteration {entry['iteration']}: "
                f"{entry['description']} — {obj_str} ({time_str})"
            )

            with st.expander(label):
                # LLM Reasoning
                reasoning = entry.get("llm_reasoning")
                if reasoning:
                    st.markdown(f"*{reasoning}*")

                # Commands
                commands = entry.get("commands", [])
                if commands:
                    st.markdown(f"**Commands** ({len(commands)}):")
                    cmd_lines = [_format_command(c) for c in commands]
                    st.code("\n".join(cmd_lines), language=None)

                # Key Metrics
                v_min, v_max = entry.get("voltage_range", (0, 0))
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Voltage", f"{v_min:.3f} – {v_max:.3f} p.u."
                           if v_min > 0 else "—")
                mc2.metric("Max Line Load", f"{entry.get('max_line_loading_pct', 0):.1f}%")
                mc3.metric("Generation", f"{entry.get('total_gen_mw', 0):.1f} MW")
                mc4.metric("Violations", entry.get("violations_count", 0))

                # Mode
                st.caption(f"Mode: {entry.get('mode', '—')}")

                # Explored variants (concurrent PFLOW)
                explored_variants = entry.get("explored_variants")
                if explored_variants:
                    st.markdown(f"**Explored {len(explored_variants)} variants:**")
                    variant_rows = []
                    for v in explored_variants:
                        label = v.get("label", "?")
                        desc = v.get("description", "")
                        feasible = "✅" if v.get("feasible") else "❌"
                        cost = f"${v['cost']:,.0f}" if v.get("cost") is not None else "—"
                        pareto = " ★" if v.get("is_pareto") else ""
                        desc_str = f" ({desc})" if desc else ""
                        variant_rows.append(f"{label}{pareto}: {feasible} {cost}{desc_str}")
                    st.code("\n".join(variant_rows), language=None)

        # Current phase indicator
        if st.session_state.search_running and st.session_state.current_phase:
            with st.status(
                f"Iteration {st.session_state.current_iteration}: "
                f"{st.session_state.current_phase}",
                state="running",
            ):
                st.write("Waiting for results...")

    # ── Right Column: Live Charts and Stats ──────────────────────────────
    with right_col:
        # Convergence Chart
        is_pflow_live = st.session_state.get("current_application") == "pflow"
        if is_pflow_live:
            st.info("Convergence chart not shown for PFLOW (no optimization objective). See voltage range chart below.")
        else:
            iterations = []
            values = []
            colors = []
            for entry in st.session_state.iteration_log:
                if entry["objective_value"] is not None:
                    iterations.append(entry["iteration"])
                    values.append(entry["objective_value"])
                    colors.append("green" if entry["feasible"] else "red")

            fig = go.Figure()
            if iterations:
                fig.add_trace(go.Scatter(
                    x=iterations, y=values,
                    mode="lines+markers",
                    marker=dict(color=colors, size=8),
                    line=dict(color="rgba(100,100,100,0.5)"),
                    hovertemplate="Iter %{x}<br>$%{y:,.2f}<extra></extra>",
                ))
            fig.update_layout(
                title="Convergence",
                xaxis_title="Iteration",
                yaxis_title="Objective Value ($)",
                height=280,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig, width="stretch")

        # Voltage Range Chart
        iters_v = [e["iteration"] for e in st.session_state.iteration_log
                    if e["voltage_range"][0] > 0]
        v_min_list = [e["voltage_range"][0] for e in st.session_state.iteration_log
                      if e["voltage_range"][0] > 0]
        v_max_list = [e["voltage_range"][1] for e in st.session_state.iteration_log
                      if e["voltage_range"][0] > 0]

        fig2 = go.Figure()
        if iters_v:
            fig2.add_trace(go.Scatter(
                x=iters_v + iters_v[::-1],
                y=v_max_list + v_min_list[::-1],
                fill="toself",
                fillcolor="rgba(66, 133, 244, 0.2)",
                line=dict(color="rgba(0,0,0,0)"),
                name="V range",
            ))
            fig2.add_trace(go.Scatter(
                x=iters_v, y=v_max_list, mode="lines",
                line=dict(color="blue"), name="V_max",
            ))
            fig2.add_trace(go.Scatter(
                x=iters_v, y=v_min_list, mode="lines",
                line=dict(color="blue"), name="V_min",
            ))
        fig2.add_hline(y=0.95, line_dash="dash", line_color="red",
                       annotation_text="0.95 p.u.")
        fig2.add_hline(y=1.05, line_dash="dash", line_color="red",
                       annotation_text="1.05 p.u.")
        fig2.update_layout(
            title="Voltage Range",
            xaxis_title="Iteration",
            yaxis_title="Voltage (p.u.)",
            height=280,
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False,
        )
        st.plotly_chart(fig2, width="stretch")

        # Progress Stats
        st.markdown("---")
        n_iters = len(st.session_state.iteration_log)
        feasible_count = sum(
            1 for e in st.session_state.iteration_log if e.get("feasible")
        )
        st.metric("Iterations", n_iters)
        st.metric("Feasible", feasible_count)
        if st.session_state.iteration_log:
            feasible_entries = [
                e for e in st.session_state.iteration_log
                if e.get("feasible") and e.get("objective_value") is not None
            ]
            if feasible_entries:
                best = min(feasible_entries, key=lambda e: e["objective_value"])
                if st.session_state.get("current_application") == "pflow":
                    st.metric(
                        "Best Solution",
                        f"Iter {best['iteration']}",
                        delta=f"V: {best.get('voltage_min', 0):.3f}\u2013{best.get('voltage_max', 0):.3f} p.u." if best.get("voltage_min", 0) > 0 else None,
                    )
                else:
                    st.metric(
                        "Best Cost",
                        f"${best['objective_value']:,.2f}",
                        delta=f"Iter {best['iteration']}",
                    )

        # ── Explore Status ───────────────────────────────────────────────
        explore_status = st.session_state.get("explore_status")
        if explore_status:
            with st.expander(
                f"🔍 Explore: {len(explore_status['variant_summaries'])} variants evaluated",
                expanded=True,
            ):
                variants = explore_status["variant_summaries"]
                for v in variants:
                    label = v.get("label", "?")
                    feasible = v.get("feasible", False)
                    is_pareto = v.get("is_pareto", False)
                    icon = "✅" if feasible else "❌"
                    pareto_mark = " ★" if is_pareto else ""
                    v_min = v.get("voltage_min", 0)
                    v_max = v.get("voltage_max", 0)
                    violations = v.get("violations_count", 0)
                    loading = v.get("max_line_loading_pct", 0)
                    status = f"{icon} **{label}**{pareto_mark}"
                    if feasible:
                        status += f"  —  V: {v_min:.3f}–{v_max:.3f}  |  Load: {loading:.1f}%  |  Viol: {violations}"
                    else:
                        conv = v.get("convergence_status", "FAILED")
                        status += f"  —  {conv}"
                    st.markdown(status)
                pareto_variants = [v for v in variants if v.get("is_pareto")]
                if pareto_variants:
                    st.info(f"Pareto-optimal: {', '.join(v['label'] for v in pareto_variants)} — awaiting LLM selection")

        # ── Steering Panel ────────────────────────────────────────────
        st.markdown("---")
        st.markdown("**🎮 Steering**")

        if st.session_state.search_paused:
            st.warning("Search paused — waiting at iteration boundary.")

        directive_text = st.text_input(
            "Directive",
            placeholder="Enter steering directive...",
            label_visibility="collapsed",
            key="steering_input",
        )
        btn_col1, btn_col2, btn_col3 = st.columns(3)

        with btn_col1:
            if st.button("Augment", width='stretch', disabled=not st.session_state.search_running):
                if directive_text.strip() and manager is not None:
                    manager.inject_steering(directive_text.strip(), mode="augment")
                    st.session_state.steering_history.append({
                        "directive": directive_text.strip(),
                        "mode": "augment",
                        "iteration": st.session_state.current_iteration,
                    })

        with btn_col2:
            if st.button("Replace", width='stretch', disabled=not st.session_state.search_running):
                if directive_text.strip() and manager is not None:
                    manager.inject_steering(directive_text.strip(), mode="replace")
                    st.session_state.steering_history.append({
                        "directive": directive_text.strip(),
                        "mode": "replace",
                        "iteration": st.session_state.current_iteration,
                    })

        with btn_col3:
            if st.session_state.search_paused:
                if st.button("▶️ Resume", width='stretch'):
                    if manager is not None:
                        manager.resume_search()
            else:
                if st.button("⏸️ Pause", width='stretch', disabled=not st.session_state.search_running):
                    if manager is not None:
                        manager.pause_search()
                        st.session_state.search_paused = True

        # Steering history
        if st.session_state.steering_history:
            with st.expander(f"📋 Steering History ({len(st.session_state.steering_history)})"):
                for item in st.session_state.steering_history:
                    tag = "🔀 REPLACE" if item["mode"] == "replace" else "➕ AUGMENT"
                    st.caption(
                        f"Iter {item['iteration']} {tag}: {item['directive'][:60]}"
                    )

    # 4. Auto-rerun (MUST be last)
    if st.session_state.search_running:
        time.sleep(1)
        st.rerun()


# ── State C: Results ─────────────────────────────────────────────────────────

def _reset_for_new_search():
    """Reset session state for a new search."""
    st.session_state.search_finished = False
    st.session_state.search_session = None
    st.session_state.search_error = None
    st.session_state.summary_analysis = None
    st.session_state.iteration_log = []
    st.session_state.base_opflow = None
    st.session_state.best_opflow = None
    st.session_state.goal_classification = None
    st.session_state.steering_history = []
    st.session_state.search_paused = False
    st.session_state.explore_status = None
    st.rerun()


def render_results():
    """Render the full results view with three tabs."""
    session = st.session_state.search_session
    if session is None:
        if st.session_state.search_error:
            st.error(f"Search failed: {st.session_state.search_error}")
        else:
            st.warning("No session data available.")
        if st.button("🔄 Start New Search"):
            _reset_for_new_search()
        return

    st.header("✅ Search Complete")

    tab1, tab2, tab3 = st.tabs([
        "📊 Overview", "🔍 Detailed Results", "📝 Analysis & Report",
    ])

    with tab1:
        _render_overview_tab(session)

    with tab2:
        _render_detailed_tab(session)

    with tab3:
        _render_analysis_tab(session)

    # New Search button at the bottom
    st.markdown("---")
    if st.button("🔄 Start New Search", type="primary"):
        _reset_for_new_search()


# ── Tab 1: Overview ──────────────────────────────────────────────────────────

def _render_contingency_overview(session, entry):
    """Render Overview tab content for contingency-screen or reserve-screen runs."""
    rm = getattr(entry, "reserve_meta", None)
    meta = entry.contingency_meta or {}
    variants = entry.explored_variants or []

    def _certified_reason(v: dict) -> str:
        status = (v.get("status") or "").upper()
        return "Did not converge" if not status.startswith("CONVERGED") else "Constraint violation"

    if rm is not None:
        # ── Reserve run ──────────────────────────────────────────────────
        st.subheader("Hot Reserve / N-1 Generator Security")

        n_on = rm.get("n_on", 0)
        avail = rm.get("hot_reserve_available", 0.0)
        req = rm.get("required_reserve_n1", 0.0)
        margin = rm.get("margin", 0.0)
        n1_secure = rm.get("n1_secure", False)
        passed_count = rm.get("passed_count", 0)
        failed_count = rm.get("failed_count", 0)

        # Minimization result (C.8 Path A) leads, if present.
        if rm.get("minimize"):
            secure_min = rm.get("n1_secure_min", n1_secure)
            min_reserve = rm.get("min_reserve", 0.0)
            reserve_full = rm.get("reserve_full", 0.0)
            n_decommitted = rm.get("n_decommitted", 0)
            final_on = rm.get("final_on_count", n_on)
            lb_pg = rm.get("lower_bound_pg", 0.0)
            lb_bus = rm.get("lower_bound_bus", "?")
            st.markdown("**Minimum feasible hot reserve for N-1**")
            xm1, xm2, xm3, xm4 = st.columns(4)
            xm1.metric("Minimum Reserve (N-1)", f"{min_reserve:,.1f} MW")
            xm2.metric("At Full Commitment", f"{reserve_full:,.1f} MW")
            xm3.metric("Units De-committed", f"{n_decommitted} / {n_on}")
            xm4.metric("N-1 Secure at Min", "Yes" if secure_min else "No")
            st.caption(
                f"Greedy largest-Pmax-first de-commitment — upper bound on the true minimum, "
                f"global optimality not claimed. {final_on} units remain committed. "
                f"Largest committed unit at minimized commitment: gen@{lb_bus} ({lb_pg:,.1f} MW) "
                f"— its output is the N-1 reserve requirement at this operating point."
                + (" Solve budget reached — reported commitment is best-so-far."
                   if rm.get("hit_budget") else "")
            )
            trajectory = rm.get("trajectory") or []
            if trajectory:
                st.plotly_chart(
                    reserve_trajectory_chart(trajectory, height=300),
                    use_container_width=True,
                )
            st.markdown("**Full-commitment starting point**")

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Hot Reserve Available", f"{avail:,.1f} MW")
        mc2.metric("Minimum Required (N-1)", f"{req:,.1f} MW")
        mc3.metric("Margin", f"{margin:,.1f} MW")
        mc4.metric("N-1 Secure", "Yes" if n1_secure else "No")

        lg_bus = rm.get("largest_pg_bus", "?")
        lg_pg = rm.get("largest_pg", 0.0)
        lg_pmax = rm.get("largest_pmax", 0.0)
        st.caption(
            f"Largest committed unit: gen@{lg_bus} at {lg_pg:,.1f} MW "
            f"(capacity {lg_pmax:,.1f} MW). Required N-1 reserve = largest single "
            f"committed generation. Each unit loss is verified by full OPF redispatch "
            f"(deliverability), not copperplate."
        )
        st.markdown(
            f"**N-1 generator screen:** {passed_count}/{passed_count + failed_count} "
            f"unit outages feasible."
        )

        failed_variants = [v for v in variants if not v.get("passed")]
        if failed_variants:
            st.markdown(f"**Failed generator outages ({len(failed_variants)}):**")
            df_fail = pd.DataFrame([{
                "Contingency": v.get("label", "?"),
                "Status": _certified_reason(v),
            } for v in failed_variants])
            st.dataframe(df_fail, width="stretch", hide_index=True)

    else:
        # ── Contingency screening run ─────────────────────────────────────
        st.subheader("Contingency Screening")

        order = meta.get("order", "?")
        target_bus = meta.get("target_bus", "?")
        neighbors = meta.get("neighbors", [])
        kinds = meta.get("kinds", [])
        vmin = meta.get("vmin", 0.9)
        vmax = meta.get("vmax", 1.1)
        passed_count = meta.get("passed_count", 0)
        failed_count = meta.get("failed_count", 0)
        total_count = passed_count + failed_count

        neighbor_str = (
            ", ".join(f"bus {n[0]} (hop {n[1]})" for n in neighbors[:5])
            if neighbors else "—"
        )
        kind_str = ", ".join(sorted(set(kinds))) if kinds else "all"

        st.caption(
            f"N-{order} contingency screen | target bus {target_bus} | "
            f"neighbors: {neighbor_str} | components: {kind_str} | "
            f"V-band [{vmin:.2f}, {vmax:.2f}] p.u. | "
            f"{passed_count}/{total_count} contingencies feasible."
        )
        st.caption(
            "Note: Under OPFLOW, the voltage band and thermal limits (Rate A) are enforced "
            "as in-solve hard constraints. PASS = post-outage OPFLOW converges feasibly "
            "under V-band + Rate A."
        )

        failed_variants = [v for v in variants if not v.get("passed")]
        passed_variants = [v for v in variants if v.get("passed")]

        if failed_variants:
            has_relief = any(v.get("relief") for v in failed_variants)
            failed_rows = []
            for v in failed_variants:
                row = {
                    "Contingency": v.get("label", "?"),
                    "Component(s)": ", ".join(v.get("kinds", [])),
                    "Status": _certified_reason(v),
                }
                if has_relief:
                    relief = v.get("relief") or {}
                    resolved = relief.get("resolved", False) if relief else False
                    if relief and not resolved:
                        row["Relief Measure"] = "unresolved (no local measure)"
                        row["Relief Detail"] = "—"
                    elif relief:
                        row["Relief Measure"] = relief.get("measure", "—")
                        row["Relief Detail"] = relief.get("detail", "—")
                    else:
                        row["Relief Measure"] = "—"
                        row["Relief Detail"] = "—"
                failed_rows.append(row)

            st.markdown(f"**Failed contingencies ({len(failed_variants)}):**")
            st.dataframe(pd.DataFrame(failed_rows), width="stretch", hide_index=True)

            if has_relief:
                st.markdown("**Relief attempts audit:**")
                for v in failed_variants:
                    relief = v.get("relief") or {}
                    attempts = relief.get("attempts", [])
                    if attempts:
                        trail = ", ".join(
                            f"{a.get('action', '?')} {'✓' if a.get('resolved') else '✗'}"
                            + (f" [{a['note']}]" if a.get("note") else "")
                            for a in attempts
                        )
                        st.caption(f"{v.get('label', '?')}: {trail}")
                st.caption(
                    "Relief scope: local modifications only (transformer ratio, generator "
                    "redispatch at neighboring buses, line switching, load curtailment). "
                    "Load curtailment is applied uniformly across all loads at the affected bus."
                )

        if passed_variants:
            st.markdown(f"**{len(passed_variants)} contingencies passed.** Full data in the JSON journal.")

            top_loading = sorted(
                passed_variants,
                key=lambda v: v.get("max_line_loading_pct", 0.0),
                reverse=True,
            )[:10]
            if top_loading:
                st.markdown("*Top 10 by max line loading (%):*")
                st.dataframe(pd.DataFrame([{
                    "Contingency": v.get("label", "?"),
                    "Component(s)": ", ".join(v.get("kinds", [])),
                    "V_min (pu)": round(v.get("voltage_min", 0.0), 3),
                    "V_max (pu)": round(v.get("voltage_max", 0.0), 3),
                    "Max Load (%)": round(v.get("max_line_loading_pct", 0.0), 1),
                } for v in top_loading]), width="stretch", hide_index=True)

            lowest_vmin = sorted(
                passed_variants, key=lambda v: v.get("voltage_min", 1.0)
            )[:10]
            if lowest_vmin:
                st.markdown("*Lowest 10 by V_min:*")
                st.dataframe(pd.DataFrame([{
                    "Contingency": v.get("label", "?"),
                    "Component(s)": ", ".join(v.get("kinds", [])),
                    "V_min (pu)": round(v.get("voltage_min", 0.0), 3),
                    "V_max (pu)": round(v.get("voltage_max", 0.0), 3),
                    "Max Load (%)": round(v.get("max_line_loading_pct", 0.0), 1),
                } for v in lowest_vmin]), width="stretch", hide_index=True)


def _render_overview_tab(session):
    """Render the Overview tab with summary and convergence chart."""
    # Determine goal classification override
    gc = st.session_state.goal_classification
    best_iter_override = gc.get("best_iteration") if gc else None
    goal_type = gc.get("goal_type") if gc else None

    stats = session.journal.summary_stats(
        best_iteration_override=best_iter_override,
        goal_type=goal_type,
    )

    sweep_entry = session.journal.get_sweep_entry()
    is_sweep = sweep_entry is not None

    # Goal
    st.markdown(f"**Goal:** {session.goal}")

    # Summary metrics row
    start = datetime.fromisoformat(session.start_time)
    end = datetime.fromisoformat(session.end_time) if session.end_time else datetime.now()
    duration = end - start
    duration_str = f"{duration.total_seconds():.0f}s"
    total_tokens = session.total_prompt_tokens + session.total_completion_tokens

    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("Application", f"{session.application}")
    mc2.metric("Iterations", f"{stats['total_iterations']}")
    mc3.metric("Duration", duration_str)
    mc4.metric("Termination", session.termination_reason)
    mc5.metric("Tokens", f"{total_tokens:,}" if total_tokens > 0 else "—")

    base_entry = session.journal.entries[0] if session.journal.entries else None
    best_entry = None
    if stats.get("best_iteration") is not None:
        for e in session.journal.entries:
            if e.iteration == stats["best_iteration"]:
                best_entry = e
                break
    is_pflow = session.application == "pflow"

    # Contingency/reserve runs get their own layout; skip sweep/scalar paths.
    contingency_entry = session.journal.get_contingency_entry()
    if contingency_entry is not None:
        _render_contingency_overview(session, contingency_entry)
        return

    # Sweep summary replaces scalar-objective metrics
    if is_sweep:
        n_cand = sweep_entry.candidate_count or 0
        _variants_hdr = sweep_entry.explored_variants or []
        _is_boundary_hdr = any("max_feasible_mw" in v for v in _variants_hdr)
        if _is_boundary_hdr:
            _det = [v for v in _variants_hdr if v.get("max_feasible_mw") is not None]
            st.metric("Boundary determined", f"{len(_det)} / {n_cand} buses")
            if _det:
                _best = max(_det, key=lambda v: v.get("max_feasible_mw") or 0.0)
                st.metric(
                    "Highest hosting capacity",
                    f"{_best['max_feasible_mw']:,.1f} MW",
                    delta=f"at bus {_best['bus']}",
                    delta_color="off",
                )
            mut_desc = (sweep_entry.description or "").replace("[boundary sweep] ", "")
            st.markdown(f"**Boundary sweep:** {mut_desc}")
        else:
            n_feas = len(sweep_entry.feasible_buses or [])
            st.metric("Feasible buses", f"{n_feas} / {n_cand}")
            mut_desc = (sweep_entry.description or "").replace("[sweep] ", "")
            st.markdown(f"**Mutation:** {mut_desc}")
    elif stats["best_objective"] is not None and not is_pflow:
        if base_entry and base_entry.objective_value is not None and base_entry.objective_value != 0:
            pct = (stats["best_objective"] - base_entry.objective_value) / base_entry.objective_value * 100
            if goal_type in (None, "cost_minimization"):
                # Lower cost is better — show reduction as positive delta (green)
                delta_str = f"{-pct:.1f}% cost reduction vs base"
                st.metric(
                    "Best Objective",
                    f"${stats['best_objective']:,.2f}",
                    delta=delta_str,
                    delta_color="inverse",
                )
            elif goal_type == "feasibility_boundary":
                # Cost increase is expected (more load served) — neutral framing
                delta_str = f"{pct:+.1f}% vs base (increase expected)"
                st.metric(
                    "Cost at Best Solution",
                    f"${stats['best_objective']:,.2f}",
                    delta=delta_str,
                    delta_color="off",
                )
            else:
                delta_str = f"{pct:+.1f}% vs base case"
                st.metric(
                    "Cost at Best Solution",
                    f"${stats['best_objective']:,.2f}",
                    delta=delta_str,
                    delta_color="off",
                )
        else:
            label = "Best Objective" if goal_type in (None, "cost_minimization") else "Cost at Best Solution"
            st.metric(label, f"${stats['best_objective']:,.2f}")
    elif is_pflow:
        st.metric("Feasibility", f"{stats['feasible_count']} / {stats['total_iterations']} feasible")
        # Show session-best cost from the journal (covers non-selected variants too)
        sb = getattr(session.journal, "session_best", None) if hasattr(session, "journal") else None
        if sb and sb.get("cost"):
            st.metric(
                "Session Best Cost",
                f"${sb['cost']:,.2f}",
                delta=f"iter {sb['iteration']}, variant {sb['variant_label']}",
                delta_color="off",
            )
        elif best_entry:
            st.metric(
                "Best Solution",
                f"Iteration {best_entry.iteration}",
                delta=f"V: {best_entry.voltage_min:.3f}\u2013{best_entry.voltage_max:.3f} p.u.",
                delta_color="off",
            )

    # Show goal achievement if available
    if gc and gc.get("best_iteration_rationale"):
        st.info(f"**Goal achievement:** {gc['best_iteration_rationale']}")

    # PFLOW vs OPFLOW Benchmark
    br = session.benchmark_result
    if br:
        with st.expander("📊 PFLOW vs OPFLOW Benchmark", expanded=False):
            if br.get("error"):
                st.warning(f"Benchmark error: {br['error']}")
            else:
                bc1, bc2, bc3 = st.columns(3)
                if br.get("opflow_objective") is not None:
                    bc1.metric("OPFLOW Optimal Cost", f"${br['opflow_objective']:,.2f}")
                if br.get("pflow_best_computed_cost") is not None:
                    bc2.metric("Best PFLOW Cost", f"${br['pflow_best_computed_cost']:,.2f}")
                if br.get("cost_gap_pct") is not None:
                    sign = "+" if br["cost_gap_pct"] >= 0 else ""
                    bc3.metric("Cost Gap", f"{sign}{br['cost_gap_pct']:.2f}%")

                dc = br.get("dispatch_comparison", [])
                if dc:
                    st.markdown("**Dispatch comparison** (sorted by |delta|):")
                    dc_rows = []
                    for d in dc[:10]:
                        pct = (d["delta"] / d["opflow_pmax"] * 100) if d.get("opflow_pmax", 0) > 0 else 0
                        dc_rows.append({
                            "Gen bus": d["bus"],
                            "Fuel": d["fuel"],
                            "OPFLOW MW": f"{d['opflow_pg']:.2f}",
                            "PFLOW MW": f"{d['pflow_pg']:.2f}",
                            "Delta MW": f"{d['delta']:+.2f}",
                            "% of Pmax": f"{pct:+.1f}%",
                        })
                    st.dataframe(
                        pd.DataFrame(dc_rows), width="stretch", hide_index=True,
                    )

                loadability = br.get("loadability")
                if loadability:
                    st.markdown("**Loadability comparison:**")
                    lc1, lc2, lc3 = st.columns(3)
                    if loadability.get("opflow_max_factor") is not None:
                        lc1.metric("OPFLOW Max Factor", f"{loadability['opflow_max_factor']:.4f}")
                    if loadability.get("pflow_max_factor") is not None:
                        lc2.metric("PFLOW Max Factor", f"{loadability['pflow_max_factor']:.4f}")
                    if loadability.get("gap_pct") is not None:
                        lc3.metric("Boundary Gap", f"{loadability['gap_pct']:+.2f}%")

    def _fmt_cost(v):
        return f"${v:,.2f}" if v is not None else "—"

    def _fmt_obj(v, is_pflow=False):
        if is_pflow:
            return "N/A (no optimization)" if (v is None or v == 0.0) else f"${v:,.2f}"
        return f"${v:,.2f}" if v is not None else "—"

    is_pflow = session.application == "pflow"

    def _fmt_f(v, fmt=".1f"):
        return f"{v:{fmt}}" if v and v > 0 else "—"

    def _change(base_v, best_v, fmt=".1f", suffix=""):
        if base_v is None or best_v is None or base_v == 0:
            return "—"
        diff = best_v - base_v
        return f"{diff:+{fmt}}{suffix}"

    if not is_sweep:
        # Base Case vs Best Solution comparison table
        comparison_label = "Base Case vs Best Solution"
        if goal_type == "feasibility_boundary":
            comparison_label = "Base Case vs Maximum Feasible Configuration"
        elif goal_type == "constraint_satisfaction":
            comparison_label = "Base Case vs Best Constraint-Satisfying Configuration"
        elif goal_type == "parameter_exploration":
            comparison_label = "Base Case vs Selected Exploration Result"
        st.subheader(comparison_label)

        rows = []
        b = base_entry
        s = best_entry
        if b:
            cost_label = "Cost (computed)" if is_pflow else "Objective (cost)"
            rows.append({
                "Metric": cost_label,
                "Base Case": _fmt_obj(b.objective_value, is_pflow),
                "Best Solution": _fmt_obj(s.objective_value, is_pflow) if s else "N/A",
                "Change": _change(b.objective_value, s.objective_value, ",.2f") if s else "—",
            })
            rows.append({
                "Metric": "Total Generation (MW)",
                "Base Case": _fmt_f(b.total_gen_mw),
                "Best Solution": _fmt_f(s.total_gen_mw) if s else "N/A",
                "Change": _change(b.total_gen_mw, s.total_gen_mw, ".1f", " MW") if s else "—",
            })
            rows.append({
                "Metric": "Voltage Min (p.u.)",
                "Base Case": _fmt_f(b.voltage_min, ".4f"),
                "Best Solution": _fmt_f(s.voltage_min, ".4f") if s else "N/A",
                "Change": _change(b.voltage_min, s.voltage_min, ".4f") if s else "—",
            })
            rows.append({
                "Metric": "Voltage Max (p.u.)",
                "Base Case": _fmt_f(b.voltage_max, ".4f"),
                "Best Solution": _fmt_f(s.voltage_max, ".4f") if s else "N/A",
                "Change": _change(b.voltage_max, s.voltage_max, ".4f") if s else "—",
            })
            rows.append({
                "Metric": "Max Line Loading (%)",
                "Base Case": _fmt_f(b.max_line_loading_pct),
                "Best Solution": _fmt_f(s.max_line_loading_pct) if s else "N/A",
                "Change": _change(b.max_line_loading_pct, s.max_line_loading_pct, ".1f", " pp") if s else "—",
            })
            rows.append({
                "Metric": "Violations",
                "Base Case": str(b.violations_count),
                "Best Solution": str(s.violations_count) if s else "N/A",
                "Change": str(s.violations_count - b.violations_count) if s else "—",
            })
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        else:
            st.info("No iteration data available for comparison.")

        if not best_entry and stats["best_objective"] is None:
            st.warning("No feasible solution was found during the search.")
    else:
        # Sweep: show feasible-bus table and infeasible table
        variants = sweep_entry.explored_variants or []

        # Boundary (hosting-capacity) sweep: dedicated table keyed on max_feasible_mw.
        if any("max_feasible_mw" in v for v in variants):
            determined = sorted(
                [v for v in variants if v.get("max_feasible_mw") is not None],
                key=lambda v: v.get("max_feasible_mw") or 0.0, reverse=True,
            )
            undetermined = sorted(
                [v for v in variants if v.get("max_feasible_mw") is None],
                key=lambda v: v["bus"],
            )
            st.caption(
                "The boundary reported is the OPFLOW convergence boundary: the largest "
                "injection for which IPOPT converges with the voltage band and thermal "
                "limits (Rate A) as in-solve hard constraints. Non-convergence is treated "
                "as the infeasible signal that caps the bisection. The binding constraint "
                "is identified at the maximum-feasible operating point."
            )
            if determined:
                st.subheader(f"Hosting capacity by bus ({len(determined)})")
                df_b = pd.DataFrame([{
                    "Bus": v["bus"],
                    "Max Feasible (MW)": round(v.get("max_feasible_mw") or 0.0, 1),
                    "Binding Constraint": v.get("binding_constraint", ""),
                    "Boundary V_min": round(v.get("voltage_min", 0), 3),
                    "Boundary V_max": round(v.get("voltage_max", 0), 3),
                    "Max Load (%)": round(v.get("max_line_loading_pct", 0), 1),
                    "Probes": v.get("probes_used", 0),
                } for v in determined])
                st.dataframe(df_b, width="stretch", hide_index=True)
            else:
                st.warning("No boundary could be determined for any candidate bus.")
            if undetermined:
                bus_list = ", ".join(str(v["bus"]) for v in undetermined)
                st.caption(f"Undetermined buses ({len(undetermined)}): {bus_list}")
        else:
            # C2/C3 detection: dispatchable siting, custom metric/predicate.
            is_dispatchable = any(v.get("dispatched_pg") is not None for v in variants)
            has_dispatched_q = any(v.get("dispatched_q") is not None for v in variants)
            metric_name = next((v.get("metric_name") for v in variants if v.get("metric_name")), None)
            predicate_name = next((v.get("predicate_name") for v in variants if v.get("predicate_name")), None)

            if metric_name == "max_delta_v":
                feasible_rows = sorted(
                    [v for v in variants if v.get("feasible")],
                    key=lambda v: v.get("metric_value") or 0.0, reverse=True,
                )
            else:
                feasible_rows = sorted(
                    [v for v in variants if v.get("feasible")],
                    key=lambda v: v["bus"],
                )
            infeasible_rows = sorted(
                [v for v in variants if not v.get("feasible")],
                key=lambda v: v["bus"],
            )

            def _certified_reason(v: dict) -> str:
                """Derive displayed reason from solver status, not the stored heuristic label."""
                status = (v.get("status") or "").upper()
                return "Did not converge" if not status.startswith("CONVERGED") else "Constraint violation"

            # Fix 3 — voltage-criterion honesty note
            st.caption(
                "Note: Under OPFLOW, the voltage band and thermal limits (Rate A) are enforced "
                "as in-solve hard constraints. The stated voltage criterion is satisfied by "
                "construction for any converged candidate and is not an independent discriminating "
                "filter for feasibility."
            )

            if feasible_rows:
                if predicate_name == "reactive_adequacy":
                    st.subheader(f"Reactive-adequate buses ({len(feasible_rows)})")
                    st.caption(
                        "A bus is reactive-adequate if a feasible OPF exists with the added unit "
                        "forced to (P = Pmax, Q = Qmax) — a reactive-headroom test, not the "
                        "standard voltage/loading criterion."
                    )
                elif metric_name == "max_delta_v":
                    st.subheader(f"Buses by voltage step ({len(feasible_rows)})")
                    st.caption(
                        "Max ΔV is the worst system-wide voltage change |V − V_base| from "
                        "switching in the load block at the candidate bus. Ranked by largest step."
                    )
                else:
                    st.subheader(f"Feasible buses ({len(feasible_rows)})")
                    if is_dispatchable:
                        st.caption(
                            "Generator mode: dispatchable (Pmin = 0, Pmax = cap) under economic "
                            "dispatch. “Dispatched Pg” is the optimized output at that location."
                        )

                def _feas_row(v: dict) -> dict:
                    row = {
                        "Bus": v["bus"],
                        "V_min (pu)": round(v.get("voltage_min", 0), 3),
                        "V_max (pu)": round(v.get("voltage_max", 0), 3),
                        "Max Line Loading (%)": round(v.get("max_line_loading_pct", 0), 1),
                        "Violations": v.get("violations", 0),
                    }
                    if metric_name == "max_delta_v":
                        mv = v.get("metric_value")
                        row["Max ΔV (pu)"] = round(mv, 4) if isinstance(mv, (int, float)) else None
                    else:
                        row["System Cost ($)"] = v.get("cost")
                    if is_dispatchable:
                        pg = v.get("dispatched_pg")
                        row["Dispatched Pg (MW)"] = round(pg, 1) if isinstance(pg, (int, float)) else None
                    if has_dispatched_q:
                        q = v.get("dispatched_q")
                        row["Dispatched Q (MVAr)"] = round(q, 1) if isinstance(q, (int, float)) else None
                    return row

                df_feas = pd.DataFrame([_feas_row(v) for v in feasible_rows])
                st.dataframe(df_feas, width="stretch", hide_index=True)

                # Fix 2 — cost-minimization near-optimal ranking (cost sweeps only)
                if goal_type in (None, "cost_minimization") and metric_name != "max_delta_v":
                    feasible_with_cost = sorted(
                        [(v, v.get("cost")) for v in feasible_rows
                         if isinstance(v.get("cost"), (int, float))],
                        key=lambda x: x[1],
                    )
                    if feasible_with_cost:
                        top_k = getattr(session.config.report, "cost_min_top_k", 10)
                        abs_tol = getattr(session.config.report, "near_optimal_abs_tol", 5.0)
                        top_buses = feasible_with_cost[:top_k]
                        best_cost = top_buses[0][1]

                        st.subheader(f"Cost ranking — top {len(top_buses)} cheapest buses")
                        df_rank = pd.DataFrame([{
                            "Rank": rank,
                            "Bus": v["bus"],
                            "Cost ($/h)": f"{cost:,.2f}",
                            "Δ from best ($/h)": f"+{cost - best_cost:.2f}" if cost > best_cost else "0.00",
                        } for rank, (v, cost) in enumerate(top_buses, 1)])
                        st.dataframe(df_rank, hide_index=True)

                        if len(top_buses) >= 2:
                            gap = top_buses[1][1] - best_cost
                            if gap < abs_tol:
                                st.caption(
                                    f"The cost difference between the top candidates (${gap:.2f}/h) "
                                    f"is below the solver tolerance threshold (${abs_tol:.1f}/h). "
                                    f"These buses should be treated as equivalently optimal rather "
                                    f"than strictly ranked."
                                )
            else:
                st.warning("No feasible buses found in this sweep.")

            if infeasible_rows:
                st.subheader(f"Infeasible buses ({len(infeasible_rows)})")
                df_infeas = pd.DataFrame([{
                    "Bus": v["bus"],
                    "Status": _certified_reason(v),
                    "V_min pu (last iter.)": round(v.get("voltage_min", 0), 3),
                    "V_max pu (last iter.)": round(v.get("voltage_max", 0), 3),
                    "Max Load % (last iter.)": round(v.get("max_line_loading_pct", 0), 1),
                    "Viol. (last iter.)": v.get("violations", 0),
                } for v in infeasible_rows])
                st.dataframe(df_infeas, width="stretch", hide_index=True)
                st.caption(
                    "Status reflects solver certification only: a solve either converged to a "
                    "feasible operating point or it did not. The four rightmost columns are from "
                    "the solver’s last uncertified iterate and must not be read as the certified "
                    "cause of infeasibility — they are diagnostic hints only. No certified "
                    "operating point exists for \"Did not converge\" buses."
                )

    if not is_sweep:
        # Convergence chart
        st.subheader("Convergence")
        fig = convergence_chart(
            session.journal, highlight_best=True, height=450,
            best_iteration=best_iter_override,
        )
        st.plotly_chart(fig, width="stretch")

        # Voltage range chart
        fig_v = voltage_range_chart(
            session.journal,
            v_min_limit=session.enforced_vmin if session.enforced_vmin is not None else 0.95,
            v_max_limit=session.enforced_vmax if session.enforced_vmax is not None else 1.05,
            height=350,
        )
        st.plotly_chart(fig_v, width="stretch")

    # Multi-objective section (only when applicable)
    if session.journal.objective_registry.is_multi_objective:
        st.subheader("📐 Multi-Objective Tracking")
        if gc and gc.get("is_multi_objective") and gc.get("tradeoff_summary"):
            st.info(f"**Tradeoff Analysis:** {gc['tradeoff_summary']}")
        if gc and gc.get("recommended_solutions"):
            recs = gc["recommended_solutions"]
            if len(recs) > 1:
                st.write(f"**Recommended tradeoff solutions:** iterations {recs}")
        mo_chart = multi_objective_trend_chart(session.journal)
        if mo_chart is not None:
            st.plotly_chart(mo_chart, width='stretch')
        pref_history = session.journal.objective_registry.history
        if pref_history:
            with st.expander("Preference Evolution History"):
                for event in pref_history:
                    action = event.get("action", "?")
                    name = event.get("name", "?")
                    src = event.get("source", "?")
                    itr = event.get("iteration", "?")
                    if action == "reprioritized":
                        st.text(
                            f"Iter {itr}: reprioritized '{name}' [{src}] "
                            f"{event.get('old_priority', '?')} → {event.get('new_priority', '?')}"
                        )
                    else:
                        st.text(
                            f"Iter {itr}: {action} '{name}' [{src}] "
                            f"({event.get('direction', '?')}, {event.get('priority', '?')})"
                        )


# ── Tab 2: Detailed Results ──────────────────────────────────────────────────

def _render_detailed_tab(session):
    """Render the Detailed Results tab with comparison charts and history table."""
    base_opflow = st.session_state.base_opflow
    best_opflow = st.session_state.best_opflow
    is_pflow = session.application == "pflow"
    sweep_entry = session.journal.get_sweep_entry()
    is_sweep = sweep_entry is not None

    _NON_SIM = {"SWEEP", "EXPLORE", "ANALYSIS", "COMPLETE", "CONTINGENCY"}

    def _fmt_obj(v, is_pf=False):
        if is_pf:
            return "N/A (no optimization)" if (v is None or v == 0.0) else f"${v:,.2f}"
        return f"${v:,.2f}" if v is not None else "—"

    if not is_sweep:
        # Voltage Profile
        fig_vp = voltage_profile_chart(
            base_opflow, best_opflow,
            v_min_limit=session.enforced_vmin if session.enforced_vmin is not None else 0.95,
            v_max_limit=session.enforced_vmax if session.enforced_vmax is not None else 1.05,
        )
        if fig_vp is not None:
            st.plotly_chart(fig_vp, width="stretch")
        else:
            st.info("Voltage profile comparison not available (missing simulation results).")

        # Generator Dispatch and Line Loading side by side
        col1, col2 = st.columns(2)
        with col1:
            fig_gen = generator_dispatch_chart(base_opflow, best_opflow)
            if fig_gen:
                st.plotly_chart(fig_gen, width="stretch")
            else:
                st.info("Generator dispatch comparison not available.")

        with col2:
            fig_ll = line_loading_chart(base_opflow, best_opflow)
            if fig_ll:
                st.plotly_chart(fig_ll, width="stretch")
            else:
                st.info("Line loading comparison not available.")

    # Iteration History Table
    st.subheader("📋 Iteration History")
    rows = []
    cost_col = "Cost" if is_pflow else "Cost ($)"
    for e in session.journal.entries:
        if e.convergence_status == "CONTINGENCY":
            rm = getattr(e, "reserve_meta", None)
            cm = e.contingency_meta or {}
            _meta = rm or cm or {}
            p = _meta.get("passed_count", 0)
            f = _meta.get("failed_count", 0)
            cost_val = f"SCREEN ({p}/{p + f} passed)"
            feas_icon = "✅" if f == 0 else "❌"
        elif e.convergence_status in _NON_SIM:
            cost_val = e.convergence_status
            feas_icon = "—"
        else:
            cost_val = _fmt_obj(e.objective_value, is_pflow) if e.objective_value is not None else "FAILED"
            feas_icon = "✅" if e.feasible else "❌"
        rows.append({
            "Iteration": e.iteration,
            "Description": e.description[:50],
            cost_col: cost_val,
            "Feasible": feas_icon,
            "V_min (p.u.)": f"{e.voltage_min:.3f}" if e.voltage_min > 0 else "—",
            "V_max (p.u.)": f"{e.voltage_max:.3f}" if e.voltage_max > 0 else "—",
            "Max Load (%)": f"{e.max_line_loading_pct:.1f}" if e.max_line_loading_pct > 0 else "—",
            "Gen (MW)": f"{e.total_gen_mw:.1f}" if e.total_gen_mw > 0 else "—",
            "Time (s)": f"{e.elapsed_seconds:.1f}",
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.info("No iterations recorded.")


# ── Tab 3: Analysis & Report ─────────────────────────────────────────────────

def _render_analysis_tab(session):
    """Render the Analysis tab with LLM summary and report generation."""

    # LLM-Generated Summary Analysis
    st.subheader("🧠 LLM Analysis")

    if st.session_state.summary_analysis is not None:
        st.markdown(st.session_state.summary_analysis)
        # Show goal classification if available
        gc = st.session_state.goal_classification
        if gc:
            with st.expander("Goal Classification (from LLM)"):
                st.markdown(f"**Goal type:** `{gc.get('goal_type', 'unknown')}`")
                st.markdown(f"**Best iteration:** {gc.get('best_iteration', 'N/A')}")
                st.markdown(f"**Rationale:** {gc.get('best_iteration_rationale', '')}")
    else:
        st.info("Click below to generate an analytical summary of the search using the LLM.")
        if st.button("Generate Analysis", type="primary"):
            manager = st.session_state.session_manager
            if manager is not None:
                with st.spinner("Generating summary analysis..."):
                    summary = manager.get_summary_analysis(session)
                    st.session_state.summary_analysis = summary
                    # Capture goal classification and update best_opflow
                    gc = manager.get_goal_classification()
                    st.session_state.goal_classification = gc
                    if gc is not None:
                        best_opflow = manager.get_best_opflow()
                        if best_opflow is not None:
                            st.session_state.best_opflow = best_opflow
                st.rerun()
            else:
                st.error("Session manager not available.")

    # Search Narrative (auto-generated from journal, no LLM call)
    st.subheader("📖 Search Narrative")

    if not session.journal.entries:
        st.info("No iterations were recorded during this search.")

    for entry in session.journal.entries:
        is_pflow = session.application == "pflow"
        if entry.iteration == 0:
            st.markdown(
                "**Iteration 0 (Base Case):** Initial simulation of the unmodified network."
            )
            if entry.objective_value is not None and not is_pflow:
                st.markdown(
                    f"Base cost: ${entry.objective_value:,.2f}, "
                    f"voltage range: {entry.voltage_min:.3f}\u2013{entry.voltage_max:.3f} p.u."
                )
            elif is_pflow:
                st.markdown(
                    f"Base case: voltage range {entry.voltage_min:.3f}\u2013{entry.voltage_max:.3f} p.u."
                )
        else:
            status = "✅" if entry.feasible else "❌"
            obj_str = (
                f"${entry.objective_value:,.2f}"
                if entry.objective_value is not None and not is_pflow else
                ("analysis" if is_pflow else "FAILED")
            )
            st.markdown(
                f"**Iteration {entry.iteration}** {status}: {entry.description}"
            )
            if entry.llm_reasoning:
                truncated = entry.llm_reasoning[:300]
                if len(entry.llm_reasoning) > 300:
                    truncated += "..."
                st.markdown(f"> *{truncated}*")
            st.markdown(
                f"Result: {obj_str} | "
                f"V: {entry.voltage_min:.3f}\u2013{entry.voltage_max:.3f} p.u. | "
                f"Load: {entry.max_line_loading_pct:.1f}%"
            )

    # PDF Report
    st.subheader("📄 PDF Report")

    if st.button("Generate PDF Report"):
        try:
            from report_generator import ReportGenerator
        except ModuleNotFoundError:
            from launcher.report_generator import ReportGenerator

        pdf_bytes = None
        with st.spinner("Generating PDF report..."):
            try:
                generator = ReportGenerator()
                pdf_bytes = generator.generate(
                    session=session,
                    summary_text=st.session_state.summary_analysis,
                    base_result=st.session_state.base_opflow,
                    best_result=st.session_state.best_opflow,
                    goal_classification=st.session_state.goal_classification,
                    steering_history=st.session_state.steering_history or None,
                )
            except Exception as e:
                st.error(f"Failed to generate PDF: {e}")

        if pdf_bytes:
            st.download_button(
                label="📥 Download Report",
                data=pdf_bytes,
                file_name=f"agentigrid_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
            )


# ── Main Entry Point ─────────────────────────────────────────────────────────

config = render_sidebar()
render_main_area(config)
