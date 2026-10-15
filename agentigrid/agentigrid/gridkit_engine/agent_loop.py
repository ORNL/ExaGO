"""GridKit agent loop: goal -> LLM -> fault screens -> answer.

Mirrors the ExaGO loop: the LLM only picks one JSON action per turn; AgentiGrid
builds the case copy, runs GridKit, applies the named checks and writes the
verdict. Actions: ``fault_screen`` and ``complete``.

A fault screen starts from the GridKit case's own operating point or, with
``start_from: latest_steady_state``, from the most recent ExaGO solution
(see from_exago.py); either start must pass a no-fault run first.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from agentigrid.config import AppConfig
from agentigrid.gridkit_engine import criteria
from agentigrid.gridkit_engine.commands import FaultStudy, RecordVariables, parse_fault_study
from agentigrid.gridkit_engine.executor import RunJob, RunResult, run_contingency, run_many, run_one
from agentigrid.gridkit_engine.from_exago import (
    SteadyStateError, SteadyStateSource, build_from_exago, find_latest_steady_state,
    goal_wants_steady_state,
)
from agentigrid.gridkit_engine.journal import GridkitJournal, JournalEntry, outcome_row
from agentigrid.gridkit_engine.modifier import apply_modifications, prepare_fault_case, write_case
from agentigrid.gridkit_engine.schema_description import command_schema_text
from agentigrid.gridkit_engine.settings import GridkitSettings
from agentigrid.gridkit_engine.sweep_metrics import FaultOutcome, run_checks, screen_verdict
from agentigrid.gridkit_engine.validation import validate_fault_study
from agentigrid.gridkit_parsers.case_parser import (
    MACHINE_CLASSES, build_element_map, create_element_map, load_case,
)
from agentigrid.gridkit_parsers.network_summary import network_summary
from agentigrid.gridkit_parsers.results_parser import read_results
from agentigrid.prompts.gridkit import build_system_prompt

logger = logging.getLogger("agentigrid.gridkit_engine.agent_loop")

_MAX_CONSECUTIVE_PARSE_FAILURES = 3
# How the shared backends report a failed API call (no key, no connection, ...).
_BACKEND_ERROR_RE = re.compile(r"^(\w+ API error:|Cannot connect to Ollama)")
_FULL_TABLE_MAX = 50        # screens up to this size also show every passing fault


@dataclass
class GridkitSession:
    goal: str
    case_file: Path
    session_dir: Path
    termination_reason: str = ""
    error: Optional[str] = None
    final_answer: Optional[dict] = None
    journal_path: Optional[Path] = None
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    screens: list[str] = field(default_factory=list)   # verdict lines, in order
    steady_state: Optional[str] = None                  # ExaGO source, when started from it


@dataclass
class _Start:
    """An operating point fault screens can start from."""

    case: Optional[dict]          # None when it could not be built
    emap: Optional[dict]
    label: str                    # for verdicts and prompts
    ok: bool                      # built, and its no-fault run solved and stayed steady
    text: str                     # how it was built and what the no-fault run showed


class GridkitAgentLoop:
    def __init__(
        self,
        cfg: AppConfig,
        settings: GridkitSettings | None = None,
        backend=None,
        quiet: bool = False,
    ):
        self._cfg = cfg
        self._settings = settings or GridkitSettings.from_config(cfg)
        if backend is None:
            from agentigrid.backends import create_backend
            backend = create_backend(cfg.llm)
        self._backend = backend
        self._quiet = quiet
        self._screen_cache: dict[str, tuple[int, str]] = {}
        self._starts: dict[str, _Start] = {}     # by FaultStudy.start_from

    def _print(self, msg: str) -> None:
        """Print progress message unless quiet mode is enabled."""
        if not self._quiet:
            print(msg, flush=True)
        else:
            logger.info(msg)

    # ------------------------------------------------------------------
    def run(self, case_file: Path | str, goal: str,
            steady_state: Optional[SteadyStateSource] = None) -> GridkitSession:
        """Run the study on *case_file*. With *steady_state*, the case's dynamic
        models start from that ExaGO solution (written as from_exago.case.json in
        the session folder) instead of the case's own operating point."""
        case_file = Path(case_file)
        binary = self._settings.dynamic_simulation
        if not binary.exists():
            raise FileNotFoundError(
                f"GridKit DynamicSimulation not found at {binary} "
                "(see applications/gridkit/README.md)"
            )
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self._settings.workdir / f"session_{stamp}"
        self._goal = goal
        self._source_case_file = case_file
        source_text = None
        if steady_state is not None:
            case, notes = build_from_exago(load_case(case_file), steady_state)
            case_file = write_case(case, session_dir / "from_exago.case.json")
            source_text = "\n".join([f"Operating point: {steady_state.describe()}.",
                                      *(f"  {n}" for n in notes)])
        self._case = load_case(case_file)
        self._emap, _ = create_element_map(case_file, session_dir)
        self._case_name = self._emap.get("case_name") or case_file.stem
        self._session_dir = session_dir
        self._journal = GridkitJournal(goal, case_file, session_dir)
        session = GridkitSession(goal=goal, case_file=case_file, session_dir=session_dir)
        self._session = session
        if source_text:
            session.steady_state = steady_state.describe()
            self._journal.add(JournalEntry(0, "steady_state", "operating point from ExaGO",
                                           result_text=source_text, run_dir=str(steady_state.run_dir)))
            self._print(f"[Session] {source_text}")
        self._print(f"[Session] {self._case_name}: {self._emap['counts']['buses']} buses, "
                    f"{len(self._emap['machines'])} machines -> {session_dir}")

        base_text, base_ok = self._run_base_case(self._case, self._emap,
                                                 session_dir / "iter_000_base")
        if source_text:
            base_text = f"{source_text}\n{base_text}"
        # With an ExaGO start given up front (--from-exago), both start_from
        # values mean that start.
        start = _Start(self._case, self._emap,
                       _exago_label(steady_state) if steady_state else "the GridKit case's operating point",
                       base_ok, base_text)
        self._starts = {"case": start}
        if steady_state is not None:
            self._starts["latest_steady_state"] = start
        if not base_ok:
            session.termination_reason = "base_case_failed"
            return self._finalize(session)

        system_prompt = build_system_prompt(
            command_schema_text(self._settings.fault_start_s, self._settings.tmax_s),
            network_summary(self._emap), base_text,
        )
        latest: Optional[str] = None
        feedback: Optional[str] = None
        parse_failures = 0
        for iteration in range(1, self._settings.max_iterations + 1):
            user_prompt = self._user_prompt(goal, latest, feedback, iteration)
            feedback = None
            self._print(f"[Iter {iteration}] Asking {self._backend.name()} ({self._cfg.llm.model})...")
            response = self._backend.complete(system_prompt, user_prompt)
            session.total_prompt_tokens += response.prompt_tokens or 0
            session.total_completion_tokens += response.completion_tokens or 0
            data = response.json_data
            if data is None and _BACKEND_ERROR_RE.match(response.raw_text or ""):
                session.termination_reason = "llm_error"
                session.error = response.raw_text.strip()
                self._journal.add(JournalEntry(iteration, "error", "LLM call failed",
                                               result_text=session.error))
                break
            if data is None:
                parse_failures += 1
                self._journal.add(JournalEntry(iteration, "error", "reply was not JSON",
                                               result_text=response.raw_text[:2000]))
                if parse_failures >= _MAX_CONSECUTIVE_PARSE_FAILURES:
                    session.termination_reason = "parse_failures"
                    break
                feedback = "Your reply was not a valid JSON object. Reply with exactly one JSON object."
                continue
            parse_failures = 0
            action = str(data.get("action", "")).lower()
            if action == "fault_screen":
                latest, feedback = self._handle_fault_screen(iteration, data)
                if latest:
                    session.screens.append(latest.splitlines()[0])
            elif action == "complete":
                session.final_answer = self._handle_complete(iteration, data)
                session.termination_reason = "completed"
                break
            else:
                feedback = f"Unknown action '{action}'. Valid actions: fault_screen, complete."
                self._journal.add(JournalEntry(iteration, "error", feedback, request=data))
        else:
            session.termination_reason = "max_iterations"
        return self._finalize(session)

    # ------------------------------------------------------------------
    def _run_base_case(self, case: dict, emap: dict, run_dir: Path,
                       iteration: int = 0, what: str = "no fault") -> tuple[str, bool]:
        """No-fault run of a copy of *case*: it must solve and stay still."""
        work, report = apply_modifications(
            case, [RecordVariables(machines=["delta", "omega"], buses=["Vm"])], emap,
        )
        case_path = write_case(work, run_dir / "case.json")
        job = RunJob(run_dir, case_path, tmax_s=self._settings.tmax_s,
                     dt_monitor_s=self._settings.dt_monitor_s)
        self._print(f"[Iter {iteration}] Base case: {what}, {self._settings.tmax_s:g} s ...")
        res = run_one(job, self._settings)
        if res.status.solved:
            series = read_results(res.csv_path, emap)
            steady = criteria.check_steady_state(
                series.t, series.device_variable("delta", MACHINE_CLASSES), series.bus_variable("Vm"),
            )
            ok = steady.status == "pass"
            text = f"GridKit {res.status.reason}. Steady state: {steady.status} - {steady.detail}."
        else:
            ok = False
            text = f"GridKit base case did not solve: {res.status.reason}."
        if not ok:
            text += " Fault screens are not meaningful until the base case solves and stays steady."
        self._print(f"[Iter {iteration}] {text}")
        self._journal.add(JournalEntry(
            iteration, "base_case", what, verdict=text, result_text=text,
            elapsed_s=res.elapsed_s, run_dir=str(run_dir),
        ))
        return text, ok

    def _start(self, start_from: str, iteration: int) -> _Start:
        """The operating point a screen asked for, built (and checked) once."""
        if start_from in self._starts:
            return self._starts[start_from]
        # Only "latest_steady_state" gets here: the most recent ExaGO solution
        # applied to the GridKit case's dynamic models.
        try:
            source = find_latest_steady_state(self._settings.exago_workdir)
            case, notes = build_from_exago(load_case(self._source_case_file), source)
        except SteadyStateError as exc:
            text = f"The most recent ExaGO steady state cannot be used: {exc}"
            start = _Start(None, None, "the latest ExaGO steady state", False, text)
            self._journal.add(JournalEntry(iteration, "steady_state", "latest ExaGO steady state",
                                           verdict=text, result_text=text))
            self._print(f"[Iter {iteration}] {text}")
            self._starts[start_from] = start
            return start
        path = write_case(case, self._session_dir / "from_exago.case.json")
        emap = build_element_map(case, path)
        source_text = "\n".join([f"Operating point: {source.describe()}.", *(f"  {n}" for n in notes)])
        self._journal.add(JournalEntry(iteration, "steady_state", "operating point from ExaGO",
                                       result_text=source_text, run_dir=str(source.run_dir)))
        self._print(f"[Iter {iteration}] {source_text}")
        base_text, ok = self._run_base_case(
            case, emap, self._session_dir / f"iter_{iteration:03d}_steady_state_base",
            iteration, "no fault, from the latest ExaGO steady state",
        )
        start = _Start(case, emap, _exago_label(source), ok, f"{source_text}\n{base_text}")
        self._starts[start_from] = start
        return start

    def _user_prompt(self, goal: str, latest: Optional[str], feedback: Optional[str], iteration: int) -> str:
        parts = [f"GOAL: {goal}", "", "HISTORY", self._journal.digest() or "(none)"]
        if latest:
            parts += ["", "LATEST RESULT", latest]
        if feedback:
            parts += ["", "ERROR IN YOUR LAST REPLY", feedback]
        parts += ["", f"Iteration {iteration} of {self._settings.max_iterations}. "
                      "Reply with one JSON action."]
        return "\n".join(parts)

    # ------------------------------------------------------------------
    def _handle_fault_screen(self, iteration: int, data: dict) -> tuple[Optional[str], Optional[str]]:
        """Returns (result text for the LLM, error feedback)."""
        description = data.get("description", "fault screen")
        try:
            study = parse_fault_study(data, self._settings.study_defaults())
        except ValueError as exc:
            self._journal.add(JournalEntry(iteration, "error", str(exc), request=data))
            return None, str(exc)
        if study.start_from == "case" and goal_wants_steady_state(self._goal) \
                and "latest_steady_state" not in self._starts:
            msg = ('fault_screen rejected: the goal asks for the most recent steady state, '
                   'so set "start_from": "latest_steady_state".')
            self._journal.add(JournalEntry(iteration, "error", msg, request=data))
            return None, msg
        ca = study.application == "ContingencyAnalysis"
        if ca and not self._settings.contingency_analysis.exists():
            msg = (f"fault_screen rejected: GridKit ContingencyAnalysis not found at "
                   f"{self._settings.contingency_analysis}; use DynamicSimulation.")
            self._journal.add(JournalEntry(iteration, "error", msg, request=data))
            return None, msg
        start = self._start(study.start_from, iteration)
        if not start.ok:
            text = (f"The fault screen was NOT run: {start.label} is not usable.\n{start.text}")
            self._journal.add(JournalEntry(iteration, "fault_screen", f"{description} (not run)",
                                           request=data, verdict=text.splitlines()[0],
                                           result_text=text))
            return text, None
        if study.start_from == "latest_steady_state" and not self._session.steady_state:
            self._session.steady_state = start.label
        check = validate_fault_study(study, start.emap)
        if not check.valid:
            msg = "fault_screen rejected: " + "; ".join(check.errors)
            self._journal.add(JournalEntry(iteration, "error", msg, request=data))
            return None, msg

        key = json.dumps(asdict(study), sort_keys=True)
        if key in self._screen_cache:
            first_iter, text = self._screen_cache[key]
            text = (f"This identical fault screen already ran in iteration {first_iter}; "
                    f"its result is repeated below. Do not run it again.\n{text}")
            self._journal.add(JournalEntry(iteration, "fault_screen", f"{description} (cached)",
                                           request=data, verdict=text.splitlines()[1],
                                           result_text=text))
            return text, None

        iter_dir = self._session_dir / f"iter_{iteration:03d}"
        work, wmap, fault_ids, report = prepare_fault_case(start.case, study, start.emap,
                                                           only_study_faults=ca)
        if report.errors:
            msg = "fault_screen could not build the case copy: " + "; ".join(report.errors)
            self._journal.add(JournalEntry(iteration, "error", msg, request=data))
            return None, msg
        case_path = write_case(work, iter_dir / "case.json")
        (iter_dir / "element_map.json").write_text(json.dumps(wmap, indent=1))

        clear = study.clear_time_s()
        device_ids = {f["element_id"]: f["id"] for f in wmap["bus_faults"]}
        jobs = [
            RunJob(iter_dir / ("contingency" if ca else f"bus_{bus}"), case_path,
                   study.tmax_s, study.dt_monitor_s, element_id=eid, start_s=study.start_s,
                   clear_s=clear, bus=bus, fault_id=device_ids.get(eid))
            for bus, eid in fault_ids.items()
        ]
        how = ("ContingencyAnalysis, one process" if ca
               else f"{self._settings.resolved_workers(len(jobs))} at a time")
        self._print(f'[Iter {iteration}] fault_screen "{description}": {len(jobs)} bus faults, {how} ...')
        t0 = time.monotonic()
        results = run_contingency(jobs, self._settings) if ca else run_many(jobs, self._settings)
        outcomes = [self._judge(r, wmap, study, clear) for r in results]
        elapsed = time.monotonic() - t0

        cycles = study.clearing_cycles + study.margin_cycles
        note = (f"bus short circuit to ground at {study.start_s:g} s, cleared after {cycles:g} cycles "
                f"({study.duration_s():.3f} s), R = {study.R:g} pu, X = {study.X:g} pu, "
                f"{study.tmax_s:g} s simulated, GridKit {study.application}, "
                f"starting from {start.label}")
        text = screen_verdict(self._subject(study), outcomes, note)
        if check.warnings:
            text += "\n\nWARNINGS:\n" + "\n".join(f"  {w}" for w in check.warnings)
        if len(outcomes) <= _FULL_TABLE_MAX:
            text += "\n\n" + _outcome_table(outcomes)
        self._print(f"[Iter {iteration}] {text.splitlines()[0]} ({elapsed:.1f} s)")

        self._screen_cache[key] = (iteration, text)
        self._journal.add(JournalEntry(
            iteration, "fault_screen", description, request=data,
            verdict=text.splitlines()[0], result_text=text,
            rows=[outcome_row(o, r.job.run_dir) for o, r in zip(outcomes, results)],
            elapsed_s=elapsed, llm_reasoning=str(data.get("reasoning", "")), run_dir=str(iter_dir),
        ))
        return text, None

    def _judge(self, res: RunResult, wmap: dict, study: FaultStudy, clear: float) -> FaultOutcome:
        checks = []
        if res.status.csv_found and res.status.csv_rows:
            series = read_results(res.csv_path, wmap)
            checks = run_checks(series, study.checks, clear)
        return FaultOutcome(res.job.bus, res.status, checks)

    def _subject(self, study: FaultStudy) -> str:
        if study.all_buses:
            return f"{self._case_name} (fault at each bus)"
        if study.poi is not None:
            return f"{self._case_name} around POI bus {study.poi} (up to {study.hops} bus(es) away)"
        return f"{self._case_name} (faults at buses {', '.join(str(b) for b in study.buses or [])})"

    def _handle_complete(self, iteration: int, data: dict) -> dict:
        answer = {k: data.get(k, "") for k in ("summary", "details", "reasoning")}
        self._journal.add(JournalEntry(iteration, "complete", "final answer",
                                       verdict=str(answer["summary"]), findings=answer))
        return answer

    def _finalize(self, session: GridkitSession) -> GridkitSession:
        session.journal_path = self._journal.export_json()
        print()
        print("=" * 60)
        print("  AgentiGrid (GridKit) study complete")
        print("=" * 60)
        print(f"  Goal:        {session.goal}")
        print(f"  Case:        {session.case_file}")
        if session.steady_state:
            print(f"  Start from:  {session.steady_state}")
        print(f"  Stopped:     {session.termination_reason}")
        if session.error:
            print(f"  Error:       {session.error[:300]}")
        if session.final_answer:
            print(f"  Summary:     {session.final_answer.get('summary', '')}")
            details = str(session.final_answer.get("details", "")).strip()
            if details:
                print("  Details:")
                for line in details.splitlines():
                    print(f"    {line}")
        print(f"  Tokens:      {session.total_prompt_tokens} prompt + "
              f"{session.total_completion_tokens} completion")
        print(f"  Journal:     {session.journal_path}")
        return session


def _exago_label(source: SteadyStateSource) -> str:
    """Short name of an ExaGO start for verdict lines."""
    return (f"the latest ExaGO steady state ({source.application.upper()} iteration "
            f"{source.iteration}, {source.journal.name})")


def _outcome_table(outcomes: list[FaultOutcome]) -> str:
    """Every fault, one row: result and the main check values."""
    def val(o: FaultOutcome, name: str, fmt: str) -> str:
        c = next((c for c in o.checks if c.name == name), None)
        if c is None or c.value is None:
            return "-" if c is None else c.status
        return fmt.format(c.value)

    lines = ["ALL BUS FAULTS:",
             f"{'Bus':>6} | {'Result':<10} | {'Angle spread':>12} | {'Vmin 2.5s':>9} | {'Damping':>8}",
             "-" * 58]
    for o in sorted(outcomes, key=lambda o: o.bus):
        lines.append(
            f"{o.bus:>6} | {o.status:<10} | {val(o, 'angle_stability', '{:.1f} deg'):>12} | "
            f"{val(o, 'voltage_recovery', '{:.3f}'):>9} | {val(o, 'damping', '{:.1%}'):>8}"
        )
    return "\n".join(lines)
