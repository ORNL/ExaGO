"""Offline analysis + PDF regeneration from a saved journal export.

Recovers the post-search analysis and PDF for a completed run WITHOUT re-solving:
useful when a large-sweep run's JSON journal was saved but the report step failed
(e.g. the classifier prompt overflowed the model context). Reads the top-level
``export_json`` schema, optionally re-runs the (now-bounded) goal classification,
and renders the same PDF the live run would have produced.

This module never imports or triggers ExaGO and never re-solves anything.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from agentigrid.config import AppConfig
from agentigrid.engine.goal_classifier import (
    build_classification_prompts,
    parse_goal_classification,
)
from agentigrid.engine.journal import (
    JournalEntry,
    ObjectiveEntry,
    ObjectiveRegistry,
    SearchJournal,
)

logger = logging.getLogger("agentigrid.engine.regenerate_report")


def _import_report_generator():
    """Import ReportGenerator, adding the (non-package) ``launcher`` dir to sys.path.

    The report generator lives in the top-level ``launcher/`` directory, which is
    not an importable package. Mirror the launcher's own import convention.
    """
    import sys

    launcher_dir = Path(__file__).resolve().parents[2] / "launcher"
    if launcher_dir.is_dir() and str(launcher_dir) not in sys.path:
        sys.path.insert(0, str(launcher_dir))
    try:
        from report_generator import ReportGenerator
    except ModuleNotFoundError:
        from launcher.report_generator import ReportGenerator
    return ReportGenerator


def load_journal_export(path: Path) -> SearchJournal:
    """Reconstruct a :class:`SearchJournal` from a top-level ``export_json`` file.

    Rebuilds every current :class:`JournalEntry` field (dict.get with dataclass
    defaults, so older exports still load), the :class:`ObjectiveRegistry` (with
    its saved preference history), and the optional
    ``benchmark_result`` / ``session_best`` / ``load_factor`` fields.

    Raises:
        ValueError: If *path* looks like a ``session.json`` (has ``format_version``
            and a nested ``journal``) instead of a journal export — the caller
            should pass the export_json journal, not a session file.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    if "format_version" in raw and isinstance(raw.get("journal"), dict):
        raise ValueError(
            f"{path} looks like a session.json (has 'format_version' and a nested "
            "'journal'). Pass the export_json journal file (top-level 'entries'), "
            "not a session.json."
        )
    if "entries" not in raw:
        raise ValueError(
            f"{path} is not a journal export: missing top-level 'entries' key."
        )

    journal = SearchJournal()

    # ``entries`` is a read-only copy property; append to the backing list, as the
    # journal's own add_* methods do.
    for ed in raw.get("entries", []):
        journal._entries.append(JournalEntry(
            iteration=ed["iteration"],
            description=ed.get("description", ""),
            commands=ed.get("commands", []),
            objective_value=ed.get("objective_value"),
            feasible=ed.get("feasible", False),
            convergence_status=ed.get("convergence_status", "UNKNOWN"),
            violations_count=ed.get("violations_count", 0),
            voltage_min=ed.get("voltage_min", 0.0),
            voltage_max=ed.get("voltage_max", 0.0),
            max_line_loading_pct=ed.get("max_line_loading_pct", 0.0),
            total_gen_mw=ed.get("total_gen_mw", 0.0),
            total_load_mw=ed.get("total_load_mw", 0.0),
            llm_reasoning=ed.get("llm_reasoning", ""),
            mode=ed.get("mode", "fresh"),
            elapsed_seconds=ed.get("elapsed_seconds", 0.0),
            timestamp=ed.get("timestamp", ""),
            steering_directive=ed.get("steering_directive"),
            tracked_metrics=ed.get("tracked_metrics"),
            feasibility_detail=ed.get("feasibility_detail", ""),
            solver=ed.get("solver", ""),
            num_steps=ed.get("num_steps", 0),
            num_scenarios=ed.get("num_scenarios", 0),
            explored_variants=ed.get("explored_variants"),
            candidate_count=ed.get("candidate_count", 0),
            feasible_buses=ed.get("feasible_buses"),
            exago_command=ed.get("exago_command"),
            contingency_meta=ed.get("contingency_meta"),
            reserve_meta=ed.get("reserve_meta"),
        ))

    # Rebuild the objective registry, then overwrite its history with the saved
    # preference history (register() would otherwise synthesize its own entries).
    registry = ObjectiveRegistry()
    for od in raw.get("objective_registry", []):
        registry.register(ObjectiveEntry(
            name=od["name"],
            direction=od["direction"],
            threshold=od.get("threshold"),
            priority=od.get("priority", "primary"),
            introduced_at=od.get("introduced_at", 0),
            source=od.get("source", "initial"),
        ))
    registry._history = raw.get("preference_history", [])
    journal.objective_registry = registry

    if raw.get("benchmark_result") is not None:
        journal.benchmark_result = raw["benchmark_result"]
    if raw.get("session_best") is not None:
        journal.session_best = raw["session_best"]
    if raw.get("load_factor") is not None:
        journal.load_factor = raw["load_factor"]

    logger.info("Loaded journal export %s (%d entries)", path, len(journal.entries))
    return journal


def build_session_stub(
    journal: SearchJournal,
    config: AppConfig,
    goal: str,
    termination_reason: str = "regenerated",
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> Any:
    """Build a :class:`SearchSession` carrying exactly what ``generate()`` reads.

    ``SearchSession`` is a plain dataclass constructible offline (no controller,
    no ExaGO), so the real object is used. Token counters are zeroed; timestamps
    default to now(); ``benchmark_result`` is carried over from the journal.
    """
    # Imported lazily so this module has no import-time coupling to the loop.
    from agentigrid.engine.agent_loop import SearchSession

    now = datetime.now().isoformat()
    return SearchSession(
        goal=goal,
        application=config.search.application,
        base_case_path=config.search.base_case,
        config=config,
        journal=journal,
        start_time=start_time or now,
        end_time=end_time or now,
        termination_reason=termination_reason,
        total_prompt_tokens=0,
        total_completion_tokens=0,
        objective_registry_data=journal.objective_registry.to_dict_list(),
        preference_history=journal.objective_registry.history,
        benchmark_result=journal.benchmark_result,
    )


def regenerate_from_journal(
    journal_json_path: Path,
    config: AppConfig,
    backend: Any,
    out_dir: Path,
    goal: str,
    generate_analysis: bool = True,
) -> Path:
    """Regenerate analysis (optional) and a PDF from a saved journal export.

    Args:
        journal_json_path: Path to a top-level ``export_json`` journal file.
        config: Loaded application config (provides application, report tuning,
            llm.backend/model for the PDF header).
        backend: LLM backend used ONLY when ``generate_analysis`` is True. May be
            None when analysis is disabled.
        out_dir: Directory to write ``<journal_stem>.pdf`` into (created if needed).
        goal: The original natural-language search goal.
        generate_analysis: When False, render from deterministic journal data only
            (no LLM call).

    Returns:
        Path to the written PDF.
    """
    journal = load_journal_export(journal_json_path)

    analysis_text: Optional[str] = None
    gc: Optional[dict] = None

    if generate_analysis:
        stats = journal.summary_stats()
        sys_p, user_p = build_classification_prompts(
            goal=goal,
            termination_reason="regenerated",
            stats=stats,
            journal_formatted=journal.format_for_classification(),
            total_tokens=0,
            objective_registry=journal.objective_registry.to_dict_list(),
            preference_history=journal.objective_registry.history,
            application=config.search.application,
            near_optimal_abs_tol=config.report.near_optimal_abs_tol,
        )
        try:
            resp = backend.complete(sys_p, user_p)
            analysis_text = resp.raw_text
            gc = parse_goal_classification(
                analysis_text, {e.iteration for e in journal.entries}
            )
        except Exception as exc:  # noqa: BLE001 — best-effort; PDF still renders
            logger.warning(
                "Analysis regeneration failed (%s); rendering PDF without it.", exc
            )
            analysis_text = None
            gc = None

    session = build_session_stub(journal, config, goal)

    # Imported here (not at module top) so the deterministic --no-analysis path
    # and unit tests don't pay the ReportLab/Plotly import cost unless rendering.
    ReportGenerator = _import_report_generator()

    pdf_bytes = ReportGenerator().generate(
        session,
        summary_text=analysis_text,
        base_result=None,
        best_result=None,
        goal_classification=gc,
        steering_history=None,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(journal_json_path).stem}.pdf"
    out_path.write_bytes(pdf_bytes)
    logger.info("Regenerated PDF written to %s (%d bytes)", out_path, len(pdf_bytes))
    return out_path
