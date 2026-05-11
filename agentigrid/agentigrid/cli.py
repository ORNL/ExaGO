"""CLI entry point for AgentiGrid."""

from __future__ import annotations

import argparse
import logging
import sys
import threading
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Any

from agentigrid import __version__
from agentigrid.config import AppConfig, load_config
from agentigrid.logging_setup import setup_logging

logger = logging.getLogger("agentigrid.cli")


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser."""
    parser = argparse.ArgumentParser(
        prog="agentigrid",
        description="AgentiGrid: LLM-driven iterative simulation for ExaGO",
    )
    parser.add_argument(
        "base_case",
        nargs="?",
        default=None,
        help="Path to MATPOWER .m base case file (not needed with --resume)",
    )
    parser.add_argument(
        "goal",
        nargs="?",
        default=None,
        help="Search goal in natural language (not needed with --resume)",
    )
    parser.add_argument(
        "--config",
        default="configs/default_config.yaml",
        help="Path to config YAML file (default: configs/default_config.yaml)",
    )
    parser.add_argument(
        "--backend",
        choices=["openai", "anthropic", "ollama", "ollama-cloud"],
        help="LLM backend: openai, anthropic, ollama, ollama-cloud",
    )
    parser.add_argument(
        "--model",
        help="Model name (overrides config)",
    )
    parser.add_argument(
        "--app",
        dest="application",
        choices=["opflow", "scopflow", "tcopflow", "sopflow", "dcopflow", "pflow"],
        help="ExaGO application: opflow, scopflow, tcopflow, sopflow, dcopflow, pflow",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        metavar="N",
        help="Maximum search iterations (default: 20)",
    )
    parser.add_argument(
        "--mode",
        choices=["accumulative", "fresh"],
        help="Iteration mode: accumulative, fresh",
    )
    parser.add_argument(
        "--search-mode",
        choices=["standard", "stress_test"],
        help="Search mode: standard (default) or stress_test (adversarial contingency exploration)",
    )
    parser.add_argument(
        "--gic",
        dest="gic_file",
        help="Path to .gic file (optional)",
    )
    parser.add_argument(
        "--ctgc",
        dest="ctgc_file",
        help="Path to .cont contingency file (required for SCOPFLOW)",
    )
    parser.add_argument(
        "--pload-profile",
        dest="pload_profile",
        help="Path to active load profile CSV (required for TCOPFLOW)",
    )
    parser.add_argument(
        "--qload-profile",
        dest="qload_profile",
        help="Path to reactive load profile CSV (required for TCOPFLOW)",
    )
    parser.add_argument(
        "--wind-profile",
        dest="wind_profile",
        help="Path to wind generation profile CSV (optional, TCOPFLOW only)",
    )
    parser.add_argument(
        "--tcopflow-duration",
        dest="tcopflow_duration",
        type=float,
        help="TCOPFLOW duration in hours (default: 1.0)",
    )
    parser.add_argument(
        "--tcopflow-dt",
        dest="tcopflow_dT",
        type=float,
        help="TCOPFLOW time-step in minutes (default: 60)",
    )
    parser.add_argument(
        "--tcopflow-iscoupling",
        dest="tcopflow_iscoupling",
        type=int,
        choices=[0, 1],
        help="TCOPFLOW generator ramp coupling: 1=on (default), 0=off",
    )
    parser.add_argument(
        "--scenario-file",
        dest="scenario_file",
        help="Path to wind scenario CSV file (required for SOPFLOW)",
    )
    parser.add_argument(
        "--sopflow-solver",
        dest="sopflow_solver",
        choices=["IPOPT", "EMPAR"],
        help="SOPFLOW solver: IPOPT (default) or EMPAR",
    )
    parser.add_argument(
        "--sopflow-iscoupling",
        dest="sopflow_iscoupling",
        type=int,
        choices=[0, 1],
        help="SOPFLOW first/second stage coupling: 0=off (default), 1=on",
    )
    parser.add_argument(
        "--np",
        dest="mpi_np",
        type=int,
        help="Number of MPI processes for ExaGO (default: 1). SCOPFLOW and SOPFLOW support multi-core via EMPAR.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=None,
        help="Enable detailed output",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=None,
        help="Suppress per-iteration progress output (only show final summary)",
    )
    parser.add_argument(
        "--resume",
        metavar="DIR",
        help="Resume a saved session from the given directory",
    )
    parser.add_argument(
        "--save-on-stop",
        action="store_true",
        help="Automatically save session state when stopped (via 'stop' command or Ctrl+C)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse config and validate without running (for testing setup)",
    )
    parser.add_argument(
        "--benchmark-opflow",
        action="store_true",
        default=None,
        help="After PFLOW search, run OPFLOW on the base case and compare results",
    )
    parser.add_argument(
        "--concurrent-pflow",
        action="store_true",
        default=None,
        help="Enable concurrent explore/select for PFLOW search",
    )
    parser.add_argument(
        "--max-variants",
        type=int,
        default=None,
        help="Max variants per explore action (default: 8, range: 2-16)",
    )
    parser.add_argument(
        "--load-factor",
        dest="load_factor",
        type=float,
        default=None,
        help="Session-level load scaling factor for PFLOW (e.g., 1.23). Auto-injected into every run.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def _cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Convert CLI arguments into dot-notation config overrides."""
    overrides: dict[str, Any] = {}
    if args.backend is not None:
        overrides["llm.backend"] = args.backend
    if args.model is not None:
        overrides["llm.model"] = args.model
    if args.application is not None:
        overrides["search.application"] = args.application
    if args.max_iter is not None:
        overrides["search.max_iterations"] = args.max_iter
    if args.mode is not None:
        overrides["search.default_mode"] = args.mode
    if args.gic_file is not None:
        overrides["search.gic_file"] = args.gic_file
    if args.ctgc_file is not None:
        overrides["search.ctgc_file"] = args.ctgc_file
    if getattr(args, "pload_profile", None) is not None:
        overrides["search.pload_profile"] = args.pload_profile
    if getattr(args, "qload_profile", None) is not None:
        overrides["search.qload_profile"] = args.qload_profile
    if getattr(args, "wind_profile", None) is not None:
        overrides["search.wind_profile"] = args.wind_profile
    if getattr(args, "tcopflow_duration", None) is not None:
        overrides["search.tcopflow_duration"] = args.tcopflow_duration
    if getattr(args, "tcopflow_dT", None) is not None:
        overrides["search.tcopflow_dT"] = args.tcopflow_dT
    if getattr(args, "tcopflow_iscoupling", None) is not None:
        overrides["search.tcopflow_iscoupling"] = args.tcopflow_iscoupling
    if args.scenario_file is not None:
        overrides["search.scenario_file"] = args.scenario_file
    if getattr(args, "sopflow_solver", None) is not None:
        overrides["search.sopflow_solver"] = args.sopflow_solver
    if getattr(args, "sopflow_iscoupling", None) is not None:
        overrides["search.sopflow_iscoupling"] = args.sopflow_iscoupling
    if args.mpi_np is not None:
        overrides["exago.mpi_np"] = args.mpi_np
    if args.search_mode is not None:
        overrides["search.search_mode"] = args.search_mode
    if getattr(args, "benchmark_opflow", None):
        overrides["search.benchmark_opflow"] = True
    if getattr(args, "concurrent_pflow", None):
        overrides["search.concurrent_pflow"] = True
    if getattr(args, "max_variants", None) is not None:
        overrides["search.max_variants"] = args.max_variants
    if getattr(args, "load_factor", None) is not None:
        overrides["search.load_factor"] = args.load_factor
    if args.verbose is not None:
        overrides["output.verbose"] = args.verbose
    # base_case is set via positional arg (may be None if --resume is used)
    if args.base_case is not None:
        overrides["search.base_case"] = args.base_case
    return overrides


def _print_config(cfg: AppConfig) -> None:
    """Pretty-print the resolved configuration."""
    def _section(name: str, obj: Any) -> None:
        print(f"\n  [{name}]")
        for f in fields(obj):
            print(f"    {f.name}: {getattr(obj, f.name)}")

    print("Resolved configuration:")
    _section("exago", cfg.exago)
    _section("data", cfg.data)
    _section("llm", cfg.llm)
    _section("search", cfg.search)
    _section("output", cfg.output)
    print()


def _print_banner(cfg: AppConfig, goal: str) -> None:
    """Print a startup banner with key settings."""
    print("=" * 60)
    print("  AgentiGrid — LLM-driven iterative simulation for ExaGO")
    print(f"  Version {__version__}")
    print("=" * 60)
    print(f"  Backend:        {cfg.llm.backend}")
    print(f"  Model:          {cfg.llm.model}")
    print(f"  Application:    {cfg.search.application}")
    print(f"  Base case:      {cfg.search.base_case}")
    print(f"  Goal:           {goal}")
    print(f"  Max iterations: {cfg.search.max_iterations}")
    print(f"  Mode:           {cfg.search.default_mode}")
    print("=" * 60)
    print()


def _start_stdin_listener(controller) -> None:
    """Start a background daemon thread that reads steering directives from stdin.

    Only called when stdin is a TTY (interactive terminal).
    """
    def _listen():
        print(
            "\n[Steering] Interactive steering active. Commands:\n"
            "  <text>         → inject directive (augment mode)\n"
            "  replace: <text>→ inject directive (replace mode)\n"
            "  pause          → pause at next iteration boundary\n"
            "  resume         → resume paused search\n"
            "  stop           → request graceful stop\n"
            "  save           → save session to disk\n"
            "  status         → show current steering state\n"
        )
        while True:
            try:
                line = input()
            except EOFError:
                break
            line = line.strip()
            if not line:
                continue
            lower = line.lower()
            if lower == "pause":
                controller.pause()
                print("[Steering] Paused. Type 'resume' to continue, or enter a directive to auto-resume.")
            elif lower == "resume":
                controller.resume()
                print("[Steering] Resumed.")
            elif lower == "stop":
                controller.request_stop()
                print("[Steering] Stop requested.")
            elif lower == "save":
                save_dir = Path(f"workdir/saved_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                controller.save_session(save_dir)
                print(f"[Steering] Session saved to: {save_dir}")
            elif lower == "status":
                directives = controller.steering_history
                print(
                    f"[Steering] Status: "
                    f"paused={controller.is_paused()}, "
                    f"directives_injected={len(directives)}"
                )
                for d in directives[-3:]:
                    print(f"  iter {d['iteration']} [{d['mode'].upper()}]: {d['directive'][:60]}")
            elif lower.startswith("replace:"):
                directive = line[8:].strip()
                if directive:
                    controller.inject_steering(directive, mode="replace")
                    if controller.is_paused():
                        controller.resume()
                    print(f"[Steering] Directive queued: '{directive[:60]}' (mode=replace)")
            else:
                controller.inject_steering(line, mode="augment")
                if controller.is_paused():
                    controller.resume()
                print(f"[Steering] Directive queued: '{line[:60]}' (mode=augment)")

    t = threading.Thread(target=_listen, name="steering-listener", daemon=True)
    t.start()


def run_search(cfg: AppConfig, goal: str, quiet: bool = False) -> None:
    """Run the LLM-driven search loop."""
    from agentigrid.engine.agent_loop import AgentLoopController

    controller = AgentLoopController(cfg, quiet=quiet)

    # Start interactive steering listener if running in a TTY
    if sys.stdin.isatty():
        _start_stdin_listener(controller)

    controller.run(cfg.search.base_case, goal)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Validate required positional args when not in resume mode
    if not args.resume and not getattr(args, "dry_run", False):
        if args.base_case is None or args.goal is None:
            parser.error("base_case and goal are required (unless --resume is used)")

    overrides = _cli_overrides(args)
    cfg = load_config(Path(args.config), cli_overrides=overrides)

    # Set up logging
    verbose = cfg.output.verbose
    setup_logging(cfg.output.logs_dir, verbose=verbose)

    logger.debug("CLI arguments: %s", args)

    # Validate base case (only when not resuming)
    if not args.resume and cfg.search.base_case and not cfg.search.base_case.exists():
        if args.dry_run:
            logger.warning("Base case file does not exist: %s (dry-run, continuing)", cfg.search.base_case)
        else:
            logger.error("Base case file does not exist: %s", cfg.search.base_case)
            sys.exit(1)

    if args.dry_run:
        _print_config(cfg)
        logger.info("Dry-run complete — exiting.")
        return

    quiet = getattr(args, "quiet", False) or False

    # Resume mode
    if args.resume:
        resume_dir = Path(args.resume)
        if not resume_dir.exists():
            logger.error("Resume directory does not exist: %s", resume_dir)
            sys.exit(1)
        if not quiet:
            print(f"Resuming session from: {resume_dir}")
        from agentigrid.engine.agent_loop import AgentLoopController
        controller = AgentLoopController(cfg, quiet=quiet)
        if sys.stdin.isatty():
            _start_stdin_listener(controller)
        controller.resume_from(resume_dir)
        return

    if not quiet:
        _print_banner(cfg, args.goal)
    run_search(cfg, args.goal, quiet=quiet)


if __name__ == "__main__":
    main()
