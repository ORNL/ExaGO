"""HIL agent loop: goal -> LLM -> close_tie studies on the HIL -> answer.

Mirrors the GridKit loop: the LLM picks one JSON action per turn; AgentiGrid
operates the HIL, reads the meters, compares with ExaGO and writes the verdict.
Actions: ``close_tie`` and ``complete``.

A close_tie study always starts from a clean radial start (all ties open, set
before the start), so every study sees the same network before its tie is closed.
With ``start_from: latest_steady_state`` the readings are compared with the most
recent ExaGO solution (see from_exago.py), the same lookup GridKit uses.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from agentigrid.config import AppConfig
from agentigrid.gridkit_engine.from_exago import goal_wants_steady_state
from agentigrid.hil_engine.check import study_result
from agentigrid.hil_engine.commands import (
    OPEN_REFUSED, REFUSED_ACTIONS, CloseTie, parse_close_tie, validate_close_tie,
)
from agentigrid.hil_engine.connection import BusReading, HilDevice, TyphoonHil
from agentigrid.hil_engine.from_exago import (
    ExagoState, SteadyStateError, latest_state, radial_reference,
)
from agentigrid.hil_engine.journal import HilJournal, JournalEntry
from agentigrid.hil_engine.network import HilNetwork, describe, load_network
from agentigrid.hil_engine.schema_description import command_schema_text
from agentigrid.hil_engine.settings import HilSettings
from agentigrid.logging_setup import log_scope
from agentigrid.prompts.hil import build_system_prompt
from agentigrid.token_usage import TokenLedger

logger = logging.getLogger("agentigrid.hil_engine.agent_loop")

_MAX_CONSECUTIVE_PARSE_FAILURES = 3
_BACKEND_ERROR_RE = re.compile(r"^(\w+ API error:|Cannot connect to Ollama)")


@dataclass
class HilSession:
    goal: str
    model: Path
    session_dir: Path
    termination_reason: str = ""
    error: Optional[str] = None
    final_answer: Optional[dict] = None
    journal_path: Optional[Path] = None
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    token_ledger: TokenLedger = field(default_factory=TokenLedger)
    studies: list[str] = field(default_factory=list)     # verdict lines, in order
    steady_state: Optional[str] = None                    # ExaGO solution used


def _reading_dict(r: BusReading) -> dict:
    return {"sim_time": r.sim_time, "volts": r.volts, "degrees": r.degrees}


class HilAgentLoop:
    def __init__(self, cfg: AppConfig, settings: HilSettings | None = None, backend=None,
                 device: HilDevice | None = None, quiet: bool = False):
        self._cfg = cfg
        self._settings = settings or HilSettings.from_config(cfg)
        if backend is None:
            from agentigrid.backends import create_backend
            backend = create_backend(cfg.llm)
        self._backend = backend
        self._device = device          # built in run() when not given (needs the network)
        self._quiet = quiet
        self._cache: dict[tuple, str] = {}
        self._exago: Optional[tuple[ExagoState, Optional[ExagoState]]] = None

    def _print(self, msg: str) -> None:
        if not self._quiet:
            print(msg, flush=True)
        else:
            logger.info(msg)

    # ------------------------------------------------------------------
    @log_scope(logger, "HIL session")
    def run(self, model: Path | str, goal: str, from_exago: bool = False,
            network: HilNetwork | None = None) -> HilSession:
        """Run the session on the Typhoon model *model* (.tse). With *from_exago*, every
        study is compared with the most recent ExaGO steady state (--from-exago).
        *network* skips reading the model (tests)."""
        model = Path(model)
        self._net: HilNetwork = network or load_network(model)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_dir = self._settings.workdir / f"session_{stamp}"
        self._session_dir.mkdir(parents=True, exist_ok=True)
        self._goal = goal
        self._from_exago = from_exago
        self._journal = HilJournal(goal, model, self._session_dir)
        session = HilSession(goal=goal, model=model, session_dir=self._session_dir)
        self._session = session
        if self._device is None:
            self._device = TyphoonHil(self._settings, self._net)
        self._print(f"[Session] {model.name}: {len(self._net.buses)} buses, ties "
                    f"{self._net.tie_names()} -> {self._session_dir}")
        try:
            return self._run(session)
        except Exception as exc:
            session.termination_reason = "hil_error"
            session.error = f"{type(exc).__name__}: {exc}"
            logger.exception("HIL session failed")
            self._journal.add(JournalEntry(0, "error", "HIL session failed", result_text=session.error))
            return session
        finally:
            try:
                self._device.shutdown()
            finally:
                self._finalize(session)

    def _run(self, session: HilSession) -> HilSession:
        with log_scope(logger, "HIL start"):
            self._print(f"[Session] Opening the HIL ({self._device.describe()})...")
            self._device.open()
            self._device.restart(set())
            base = self._device.read_buses()
        base_text = self._base_text(base)
        self._journal.add(JournalEntry(0, "base_state", "radial start, all ties open",
                                       result_text=base_text, readings={"radial": _reading_dict(base)}))
        self._print(f"[Session] Radial start read at simulation time {base.sim_time:.2f} s")

        system_prompt = build_system_prompt(
            command_schema_text(self._settings.tol_pct, self._settings.tol_deg),
            describe(self._net), base_text)
        latest: Optional[str] = None
        feedback: Optional[str] = None
        parse_failures = 0
        for iteration in range(1, self._settings.max_iterations + 1):
            with log_scope(logger, "HIL iteration", iteration=iteration):
                user_prompt = self._user_prompt(latest, feedback, iteration)
                feedback = None
                self._print(f"[Iter {iteration}] Asking {self._backend.name()} ({self._cfg.llm.model})...")
                response = self._backend.complete(system_prompt, user_prompt)
                session.token_ledger.record("search", response, iteration,
                                            prompt_chars=len(system_prompt) + len(user_prompt))
                session.total_prompt_tokens = session.token_ledger.total_prompt
                session.total_completion_tokens = session.token_ledger.total_completion
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
                                                   result_text=(response.raw_text or "")[:2000]))
                    if parse_failures >= _MAX_CONSECUTIVE_PARSE_FAILURES:
                        session.termination_reason = "parse_failures"
                        break
                    feedback = "Your reply was not a valid JSON object. Reply with exactly one JSON object."
                    continue
                parse_failures = 0
                action = str(data.get("action", "")).lower()
                if action == "close_tie":
                    latest, feedback = self._handle_close_tie(iteration, data)
                    if latest:
                        session.studies.append(latest.splitlines()[0])
                elif action in REFUSED_ACTIONS:
                    latest = f"REFUSED: {action}. {OPEN_REFUSED}"
                    self._journal.add(JournalEntry(iteration, "refused", action, request=data,
                                                   verdict=latest, result_text=latest))
                    self._print(f"[Iter {iteration}] {latest}")
                elif action == "complete":
                    session.final_answer = self._handle_complete(iteration, data)
                    session.termination_reason = "completed"
                    break
                else:
                    feedback = f"Unknown action '{action}'. Valid actions: close_tie, complete."
                    self._journal.add(JournalEntry(iteration, "error", feedback, request=data))
        else:
            session.termination_reason = "max_iterations"
        return session

    # ------------------------------------------------------------------
    def _base_text(self, r: BusReading) -> str:
        vmin = min(self._net.buses, key=lambda b: r.volts[b])
        vmax = max(self._net.buses, key=lambda b: r.volts[b])
        rows = "  ".join(f"{b}:{r.volts[b]:.1f}" for b in self._net.buses)
        return (f"All ties open, feeder closed, simulation time {r.sim_time:.2f} s.\n"
                f"Lowest voltage {r.volts[vmin]:.2f} V at bus {vmin}; highest {r.volts[vmax]:.2f} V at bus {vmax}.\n"
                f"Bus voltages (V, line-to-neutral): {rows}")

    def _user_prompt(self, latest: Optional[str], feedback: Optional[str], iteration: int) -> str:
        parts = [f"GOAL: {self._goal}", "", "HISTORY", self._journal.digest() or "(none)"]
        if latest:
            parts += ["", "LATEST RESULT", latest]
        if feedback:
            parts += ["", "ERROR IN YOUR LAST REPLY", feedback]
        parts += ["", f"Iteration {iteration} of {self._settings.max_iterations}. Reply with one JSON action."]
        return "\n".join(parts)

    def _exago_states(self, iteration: int) -> tuple[Optional[tuple[ExagoState, Optional[ExagoState]]], Optional[str]]:
        """(closed state, radial reference) of the latest ExaGO steady state, or the reason it
        cannot be used. Looked up once per session."""
        if self._exago is not None:
            return self._exago, None
        try:
            source, state = latest_state(self._settings.exago_workdir, self._net)
        except SteadyStateError as exc:
            text = f"The most recent ExaGO steady state cannot be used: {exc}"
            self._journal.add(JournalEntry(iteration, "steady_state", "latest ExaGO steady state",
                                           verdict=text, result_text=text))
            return None, text
        radial = radial_reference(source, self._net)
        self._exago = (state, radial)
        self._session.steady_state = state.label
        text = (f"Steady state: {state.label}, solution {state.solved_file}; ties closed: "
                f"{', '.join(f'{a}-{b}' for a, b in sorted(state.closed_ties)) or 'none'}. "
                f"Radial reference: {radial.label if radial else 'none in this ExaGO session'}.")
        self._journal.add(JournalEntry(iteration, "steady_state", "latest ExaGO steady state",
                                       result_text=text,
                                       exago={"closed": str(state.solved_file),
                                              "radial": str(radial.solved_file) if radial else None}))
        self._print(f"[Iter {iteration}] {text}")
        return self._exago, None

    def _handle_close_tie(self, iteration: int, data: dict) -> tuple[Optional[str], Optional[str]]:
        """Returns (result text for the LLM, error feedback)."""
        with log_scope(logger, "HIL action", iteration=iteration, action="close_tie"):
            try:
                study = parse_close_tie(data)
            except ValueError as exc:
                self._journal.add(JournalEntry(iteration, "error", str(exc), request=data))
                return None, str(exc)
            if self._from_exago:
                study = CloseTie(study.tie, "latest_steady_state", study.description)
            if study.start_from == "radial" and goal_wants_steady_state(self._goal):
                msg = ('close_tie rejected: the goal asks for the most recent steady state, '
                       'so set "start_from": "latest_steady_state".')
                self._journal.add(JournalEntry(iteration, "error", msg, request=data))
                return None, msg
            reason = validate_close_tie(study, self._net)
            name = f"{study.tie[0]}-{study.tie[1]}"
            if reason:
                text = f"close_tie NOT RUN: {reason}."
                self._journal.add(JournalEntry(iteration, "close_tie", f"tie {name} (not run)",
                                               request=data, verdict=text, result_text=text))
                return text, None

            key = (study.tie, study.start_from)
            if key in self._cache:
                text = f"IDENTICAL to an earlier study; result repeated.\n{self._cache[key]}"
                self._journal.add(JournalEntry(iteration, "close_tie", f"tie {name} (repeat)",
                                               request=data, verdict=text.splitlines()[1], result_text=text))
                return text, None

            ex_closed = ex_radial = None
            notes: list[str] = []
            if study.start_from == "latest_steady_state":
                states, why = self._exago_states(iteration)
                if states is None:
                    text = f"close_tie NOT RUN: {why}"
                    self._journal.add(JournalEntry(iteration, "close_tie", f"tie {name} (not run)",
                                                   request=data, verdict=text, result_text=text))
                    return text, None
                ex_closed, ex_radial = states
                if not ex_closed.usable:
                    text = ("close_tie SKIPPED: the HIL cannot build the network of the latest ExaGO "
                            f"steady state ({ex_closed.label}): " + "; ".join(ex_closed.problems) + ".")
                    self._journal.add(JournalEntry(iteration, "close_tie", f"tie {name} (skipped)",
                                                   request=data, verdict=text, result_text=text))
                    return text, None
                if ex_closed.closed_ties != {study.tie}:
                    got = ", ".join(f"{a}-{b}" for a, b in sorted(ex_closed.closed_ties)) or "none"
                    text = (f"close_tie NOT RUN: the latest ExaGO steady state ({ex_closed.label}) has "
                            f"ties closed: {got}; this study closes {name} only. The comparison is only "
                            "valid for the same network.")
                    self._journal.add(JournalEntry(iteration, "close_tie", f"tie {name} (not run)",
                                                   request=data, verdict=text, result_text=text))
                    return text, None
                notes.append(f"ExaGO solution with the tie closed: {ex_closed.solved_file}")
                if ex_radial:
                    notes.append(f"ExaGO radial solution: {ex_radial.solved_file}")

            t0 = time.monotonic()
            self._print(f"[Iter {iteration}] HIL: radial start, then close tie {name} live...")
            tie = self._net.tie(*study.tie)
            self._device.restart(set())
            radial = self._device.read_buses()
            ok = self._device.close_tie(study.tie)
            closed = self._device.read_buses()
            commands = [
                {"command": "restart", "ties_closed": [], "feeder": "closed", "sim_time": radial.sim_time},
                {"command": "close_tie", "tie": name, "switch": tie.switch, "scada_input": tie.scada_input,
                 "value": 1.0, "accepted": ok, "sim_time_read": closed.sim_time},
            ]
            res = study_result(self._net, study.tie, radial, closed, ok, ex_closed, ex_radial,
                               self._settings.tol_pct, self._settings.tol_deg, notes)
            self._cache[key] = res.text
            self._journal.add(JournalEntry(
                iteration, "close_tie", study.description or f"close tie {name}", request=data,
                verdict=res.verdict, result_text=res.text, commands=commands,
                readings={"radial": _reading_dict(radial), "closed": _reading_dict(closed)},
                comparisons=[{k: v for k, v in asdict(c).items()} for c in res.comparisons],
                exago={"closed": str(ex_closed.solved_file) if ex_closed else None,
                       "radial": str(ex_radial.solved_file) if ex_radial else None},
                elapsed_s=time.monotonic() - t0))
            self._print(f"[Iter {iteration}] {res.verdict}")
            return res.text, None

    def _handle_complete(self, iteration: int, data: dict) -> dict:
        answer = {k: data.get(k, "") for k in ("summary", "details", "reasoning")}
        self._journal.add(JournalEntry(iteration, "complete", "final answer",
                                       verdict=str(answer["summary"]), findings=answer))
        return answer

    @log_scope(logger, "HIL finalization")
    def _finalize(self, session: HilSession) -> HilSession:
        self._journal.token_usage = session.token_ledger.to_dict()
        session.journal_path = self._journal.export_json()
        print()
        print("=" * 60)
        print("  AgentiGrid (HIL) study complete")
        print("=" * 60)
        print(f"  Goal:        {session.goal}")
        print(f"  Model:       {session.model}")
        if session.steady_state:
            print(f"  Compared to: {session.steady_state}")
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
        print(f"  {session.token_ledger.summary_text()}")
        print(f"  Journal:     {session.journal_path}")
        return session
