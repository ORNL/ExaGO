"""Router agent loop: one goal -> the tools it needs, one action per turn (``--tool router``).

The router knows no tool. Each tool comes as an adapter (``router_engine/adapters``)
that gives a card for the system prompt, handlers for its actions, the checks those
actions need, and the work it still has pending. The LLM picks one JSON action per
turn; the router sends it to the adapter that owns it. Tools exchange nothing but
steady states (``state.py``): ExaGO makes them, every tool works from the latest or
the best one. "Best" follows the goal's objectives, extracted once at the start the
same way the ExaGO agent does it.

Every action after the first carries the agent's "check" of the latest result and its
"decision" (proceed or stop). The code holds the gate: after a failed check no further
tool action is accepted unless the goal says to continue after a failure.

Outputs, one folder per session: ``journal.json`` (every turn, every result),
``log.txt`` (the same as text, with the agent's check, decision and reasoning and the
final answer in full), ``llm_transcript.txt`` / ``.jsonl`` (every LLM call: the prompt
it was given and its reply, word for word), ``analysis.txt`` (the final engineering
analysis), ``summary.txt`` (one row per action and the overall verdict),
``system_prompt.txt`` and each tool's own files.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from agentigrid.config import AppConfig
from agentigrid.exago_engine.objective_parser import build_objective_extraction_prompt, parse_objective_extraction
from agentigrid.logging_setup import log_scope
from agentigrid.prompts.router import build_analysis_prompt, build_system_prompt
from agentigrid.router_engine.adapters.base import ActionResult, Context, ToolAdapter
from agentigrid.router_engine.state import METRICS, Objective, StateStore
from agentigrid.token_usage import TokenLedger

logger = logging.getLogger("agentigrid.router_engine.agent_loop")

DEFAULT_MAX_TURNS = 20
_MAX_CONSECUTIVE_PARSE_FAILURES = 3
_MAX_CONSECUTIVE_REFUSALS = 3       # replies refused in a row before the session stops
_BACKEND_ERROR_RE = re.compile(r"^(\w+ API error:|Cannot connect to Ollama)")
_DECISIONS = ("proceed", "stop")
# A goal that asks to go on after a failed comparison or check.
_CONTINUE_ON_FAIL_RE = re.compile(
    r"\b(even if|regardless of|despite|whether or not|continue (?:after|on|if|when|even))\b"
    r"[^.]{0,80}\b(fail\w*|mismatch\w*|differ\w*|disagree\w*|match\w*)", re.IGNORECASE)


@dataclass
class RouterSession:
    goal: str
    session_dir: Path
    tools: list[str] = field(default_factory=list)
    termination_reason: str = ""
    error: Optional[str] = None
    final_answer: Optional[dict] = None
    overall_verdict: str = ""
    objectives: list[dict] = field(default_factory=list)
    analysis: Optional[str] = None
    journal_path: Optional[Path] = None
    log_path: Optional[Path] = None
    summary_path: Optional[Path] = None
    analysis_path: Optional[Path] = None
    transcript_path: Optional[Path] = None
    token_ledger: TokenLedger = field(default_factory=TokenLedger)
    rows: list[dict] = field(default_factory=list)          # summary.txt
    checks: list[tuple[str, bool]] = field(default_factory=list)   # (what, passed)


class RouterAgentLoop:
    def __init__(self, cfg: AppConfig, adapters: list[ToolAdapter], max_turns: int = DEFAULT_MAX_TURNS,
                 workdir: Path | None = None, backend=None, quiet: bool = False):
        self._cfg = cfg
        self._offered = adapters
        self._max_turns = max_turns
        self._workdir = Path(workdir) if workdir else Path(cfg.output.workdir) / "router"
        if backend is None:
            from agentigrid.backends import create_backend
            backend = create_backend(cfg.llm)
        self._backend = backend
        self._quiet = quiet

    def _print(self, msg: str) -> None:
        if not self._quiet:
            print(msg, flush=True)
        logger.info(msg)

    # ------------------------------------------------------------------
    @log_scope(logger, "Router session")
    def run(self, goal: str) -> RouterSession:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._dir = self._workdir / f"session_{stamp}"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._goal = goal
        self._continue_on_fail = bool(_CONTINUE_ON_FAIL_RE.search(goal))
        self._entries: list[dict] = []
        self._log = self._dir / "log.txt"
        self._transcript = self._dir / "llm_transcript.txt"
        self._transcript_jsonl = self._dir / "llm_transcript.jsonl"
        self._calls = 0
        self._store = StateStore()
        self._stopped: Optional[str] = None
        self._complete_refused = False
        self._has_result = False
        self._watch = False
        self._attention: list[str] = []
        self._tools: list[ToolAdapter] = []
        session = RouterSession(goal=goal, session_dir=self._dir, log_path=self._log,
                                transcript_path=self._transcript)
        self._session = session
        self._write_log(f"AgentiGrid router session {stamp}\nGoal: {goal}\n"
                        f"LLM: {self._cfg.llm.backend} ({self._cfg.llm.model})\nTurn limit: {self._max_turns}")
        try:
            self._run(session)
        except Exception as exc:
            session.termination_reason = "error"
            session.error = f"{type(exc).__name__}: {exc}"
            logger.exception("Router session failed")
            self._add(0, "error", "session failed", result_text=session.error)
        finally:
            for t in self._tools:
                try:
                    t.close()
                except Exception:                    # noqa: BLE001 - keep closing the others
                    logger.exception("Closing %s failed", t.name)
            try:
                self._analysis(session)
            except Exception as exc:                 # noqa: BLE001 - the records still get written
                logger.exception("Final analysis failed")
                session.analysis = f"(final analysis failed: {type(exc).__name__}: {exc})"
            self._finalize(session)
        return session

    def _call(self, phase: str, turn: int, system_prompt: str, user_prompt: str, system_name: str):
        """One LLM call, recorded word for word in the transcript."""
        response = self._backend.complete(system_prompt, user_prompt)
        self._session.token_ledger.record(phase, response, turn, prompt_chars=len(system_prompt) + len(user_prompt))
        self._calls += 1
        rec = {"call": self._calls, "phase": phase, "turn": turn, "system_prompt": system_name,
               "user_prompt": user_prompt, "reply": response.raw_text,
               "prompt_tokens": getattr(response, "prompt_tokens", None),
               "completion_tokens": getattr(response, "completion_tokens", None),
               "timestamp": datetime.now().isoformat(timespec="seconds")}
        with self._transcript_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        with self._transcript.open("a", encoding="utf-8") as f:
            f.write(f"\n{'#' * 78}\n# LLM call {self._calls}: {phase}, turn {turn} "
                    f"(system prompt: {system_name})\n{'#' * 78}\n"
                    f"--- PROMPT ---\n{user_prompt}\n--- REPLY ---\n{response.raw_text}\n")
        return response

    def _run(self, session: RouterSession) -> None:
        for t in self._offered:
            why = t.ready()
            if why:
                self._print(f"[Session] {t.title}: not available ({why})")
                self._add(0, "tool", f"{t.name} not available", result_text=why)
                continue
            t.start(self._dir, self._goal, self._quiet)
            self._tools.append(t)
        session.tools = [t.name for t in self._tools]
        self._watch = not any(t.name == "exago" for t in self._tools)
        if self._watch and not any(getattr(t, "standalone", False) for t in self._tools):
            session.termination_reason = "not_run"
            session.overall_verdict = "OVERALL VERDICT: NOT RUN — no ExaGO case: the tools work from ExaGO steady states."
            return
        self._owner = {a: t for t in self._tools for a in t.actions}
        self._print(f"[Session] Tools: {', '.join(t.title for t in self._tools)} -> {self._dir}")
        self._add(0, "tools", ", ".join(session.tools),
                  result_text="\n".join(f"{t.name}: {t.title}" for t in self._tools))
        if not self._watch and not self._objectives(session):
            return
        system_prompt = build_system_prompt([(t.title, t.synonyms, t.card()) for t in self._tools],
                                            watch=self._watch)
        (self._dir / "system_prompt.txt").write_text(system_prompt)
        latest: Optional[str] = None
        feedback: Optional[str] = None
        parse_failures = 0
        refusals = 0
        for turn in range(1, self._max_turns + 1):
            with log_scope(logger, "Router turn", turn=turn):
                user_prompt = self._user_prompt(latest, feedback, turn)
                feedback = None
                self._print(f"[Turn {turn}] Asking {self._backend.name()} ({self._cfg.llm.model})...")
                response = self._call("search", turn, system_prompt, user_prompt, "system_prompt.txt")
                data = response.json_data
                if data is None and _BACKEND_ERROR_RE.match(response.raw_text or ""):
                    session.termination_reason = "llm_error"
                    session.error = response.raw_text.strip()
                    self._add(turn, "error", "LLM call failed", result_text=session.error)
                    break
                if data is None:
                    parse_failures += 1
                    self._add(turn, "error", "reply was not JSON", result_text=(response.raw_text or "")[:2000])
                    if parse_failures >= _MAX_CONSECUTIVE_PARSE_FAILURES:
                        session.termination_reason = "parse_failures"
                        break
                    feedback = "Your reply was not a valid JSON object. Reply with exactly one JSON object."
                    continue
                parse_failures = 0
                action = str(data.get("action", "")).lower()
                feedback = self._gate(action, data)
                if feedback:
                    refusals += 1
                    self._add(turn, "error", feedback, request=data)
                    self._print(f"[Turn {turn}] {feedback}")
                    if refusals >= _MAX_CONSECUTIVE_REFUSALS:
                        session.termination_reason = "format_errors"
                        session.error = (f"{refusals} replies in a row were refused; the last reason: {feedback}")
                        break
                    continue
                if action == "complete":
                    done, feedback = self._complete(turn, data)
                    if done:
                        session.termination_reason = "completed"
                        break
                    refusals += 1
                    continue
                refusals = 0
                tool = self._owner[action]
                res = tool.handle(action, data, Context(self._store, self._dir, turn, self._print))
                latest, feedback = self._record(turn, tool, action, data, res)
        else:
            session.termination_reason = "max_turns"

    # ------------------------------------------------------------------
    def _gate(self, action: str, data: dict) -> Optional[str]:
        """The reason an action is refused before it runs, or None."""
        if not action:
            return ('Your reply has no "action", so nothing was run. Send the same reply again with the '
                    f'action added, for example: {self._template()}')
        if action != "complete" and action not in self._owner:
            return (f"Unknown action '{action}'. Valid actions: {', '.join(sorted(self._owner))}, complete. "
                    f"For example: {self._template()}")
        if action != "complete" and self._stopped:
            return f"{action} refused: the study was stopped ({self._stopped}). Finish with complete."
        if self._has_result:
            check = str(data.get("check") or "").strip()
            decision = str(data.get("decision") or "").strip().lower()
            if not check:
                return (f'{action} refused: "check" is missing, so nothing was run. Send the same {action} '
                        'reply again with "check" (what the latest result shows, and whether it passed) and '
                        f'"decision" added, for example: {self._template(action)}')
            if decision not in _DECISIONS:
                return (f'{action} refused: "decision" must be "proceed" or "stop", not '
                        f'{decision or "missing"!r}. For example: {self._template(action)}')
            if action != "complete" and decision == "stop":
                return f'{action} refused: your decision is "stop"; finish with complete instead.'
            last = self._session.checks[-1] if self._session.checks else None
            if action != "complete" and last is not None and not last[1] and not self._continue_on_fail:
                return (f"{action} refused: the last check FAILED ({last[0]}). The goal does not ask to continue "
                        "after a failure: set \"decision\": \"stop\" and finish with complete, reporting the "
                        "failing buses.")
        return None

    def _template(self, action: Optional[str] = None) -> str:
        """A reply skeleton for *action* (or the action the session most likely needs next)."""
        if action is None and "wait" in getattr(self, "_owner", {}):
            action = "wait" if self._has_result else "hil_start"
        if action is None:
            pend = self._pending()
            if any("HIL" in p for p in pend):
                action = "hil_step"
            elif "exago_solve" in self._owner:
                action = "exago_solve"
            else:
                action = "complete"
        extra = {"exago_solve": ', "start_from": "latest", "commands": [...]',
                 "hil_step": ', "start_from": "latest"',
                 "close_tie": ', "tie": [<bus>, <bus>]',
                 "gridkit_fault_screen": ', "start_from": "latest", ...fault_screen fields...',
                 "complete": ', "summary": "<the OVERALL VERDICT line>", "details": "..."'}.get(action, "")
        check = ', "check": "<what the latest result shows>", "decision": "proceed"' if self._has_result else ""
        return f'{{"action": "{action}"{check}{extra}, "reasoning": "<why>"}}'

    def _objectives(self, session: RouterSession) -> bool:
        """Extract the goal's objectives (for "best"). False if the LLM cannot be reached."""
        sys_p, user_p = build_objective_extraction_prompt(self._goal, list(METRICS))
        (self._dir / "objectives_system_prompt.txt").write_text(sys_p)
        resp = self._call("objective_extraction", 0, sys_p, user_p, "objectives_system_prompt.txt")
        if _BACKEND_ERROR_RE.match(resp.raw_text or ""):
            session.termination_reason = "llm_error"
            session.error = resp.raw_text.strip()
            self._add(0, "error", "LLM call failed", result_text=session.error)
            return False
        parsed = parse_objective_extraction(resp.raw_text or "") or []
        for o in parsed:
            self._store.objectives.append(Objective(str(o.get("name", "")), str(o.get("direction", "")),
                                                    o.get("threshold"), str(o.get("priority") or "primary")))
        session.objectives = [vars(o) for o in self._store.objectives]
        text = ("; ".join(f"{o.direction} {o.name}" + (f" (limit {o.threshold})" if o.threshold is not None else "")
                          for o in self._store.objectives) or "none (\"best\" is not available; use \"latest\")")
        self._add(0, "objectives", "objectives from the goal", result_text=text)
        self._print(f"[Session] Objectives: {text}")
        return True

    def _record(self, turn: int, tool: ToolAdapter, action: str, data: dict,
                res: ActionResult) -> tuple[Optional[str], Optional[str]]:
        if res.feedback:
            self._add(turn, "error", res.feedback, request=data, tool=tool.name)
            self._print(f"[Turn {turn}] {res.feedback}")
            return None, res.feedback
        self._has_result = True
        self._attention = list(res.attention)
        if res.stop:
            self._stopped = res.stop
        if res.passed is not None:
            self._session.checks.append((res.verdict or action, res.passed))
        if res.row:
            self._session.rows.append({"turn": turn, **res.row})
        self._add(turn, action, res.verdict or "", request=data, tool=tool.name, verdict=res.verdict,
                  result_text=res.text, attention=res.attention, **res.record)
        if data.get("check"):
            self._print(f"[Turn {turn}] Agent check: {data.get('check')} -> {data.get('decision')}")
        if res.verdict:
            self._print(f"[Turn {turn}] {res.verdict}")
        return res.text, None

    # ------------------------------------------------------------------ prompts
    def _pending(self) -> list[str]:
        return [p for t in self._tools if (p := t.pending(self._store))]

    def _overall(self) -> str:
        if self._watch:
            return " ".join(t.overall() for t in self._tools if hasattr(t, "overall"))
        checks = self._session.checks
        tail = []
        if self._stopped:
            tail.append(f"study stopped: {self._stopped}")
        tail += self._pending()
        tail = ("; " + "; ".join(tail)) if tail else ""
        n_states = len(self._store.states)
        if not checks:
            return (f"OVERALL VERDICT: NO CHECKS — {n_states} steady state(s) solved, no tool compared or "
                    f"judged anything yet{tail}.")
        failed = [w for w, ok in checks if not ok]
        if failed:
            return f"OVERALL VERDICT: FAIL — {len(failed)} of {len(checks)} checks failed{tail}."
        return f"OVERALL VERDICT: PASS — all {len(checks)} checks passed ({n_states} steady states){tail}."

    def _state_text(self) -> str:
        if self._watch:
            lines = []
            if self._stopped:
                lines.append(f"The study was stopped: {self._stopped}. Finish with complete.")
            pend = self._pending()
            if pend:
                lines.append("Not done yet: " + "; ".join(pend))
            lines.append(self._overall())
            return "\n".join(lines)
        best, why = self._store.best()
        lines = ["Steady states:", self._store.describe(),
                 f"Latest: {self._store.latest().name if self._store.states else 'none'}",
                 f"Best: {best.name + ' (' + why + ')' if best else 'not available (' + why + ')'}"]
        last = self._session.checks[-1] if self._session.checks else None
        if last is not None:
            lines.append(f"Last check: {'PASS' if last[1] else 'FAIL'} — {last[0]}")
            if not last[1]:
                lines.append("Continue after a failure: " + ("yes, the goal asks for it" if self._continue_on_fail
                                                             else "no (the goal does not ask for it)"))
        if self._attention:
            lines.append("Near the limit in the latest result: " + "; ".join(self._attention))
        pend = self._pending()
        if self._stopped:
            lines.append(f"The study was stopped: {self._stopped}. Finish with complete.")
        if pend:
            lines.append("Not done yet: " + "; ".join(pend))
        lines.append(self._overall())
        return "\n".join(lines)

    def _user_prompt(self, latest: Optional[str], feedback: Optional[str], turn: int) -> str:
        parts = [f"GOAL: {self._goal}", "", "STATE", self._state_text(), "", "HISTORY", self._digest() or "(none)"]
        if latest:
            parts += ["", "LATEST RESULT", latest]
        if feedback:
            parts += ["", "ERROR IN YOUR LAST REPLY", feedback]
        fields = '"action"' + (', "check", "decision"' if self._has_result else "") + \
            ', "reasoning" and the action\'s own fields'
        parts += ["", f"Turn {turn} of {self._max_turns}. Reply with ONE JSON object containing {fields}. "
                      f"Actions: {', '.join(sorted(self._owner))}, complete."]
        if self._has_result:
            parts.append(f"Shape of the reply (any action): {self._template()}")
        return "\n".join(parts)

    def _digest(self) -> str:
        out = []
        for e in self._entries:
            if e["turn"] > 0 and e.get("verdict"):
                req = e.get("request") or {}
                line = f"[Turn {e['turn']}] {e['action']}: {e['verdict']}"
                if req.get("check"):
                    line += f"\n    your check before it: {req['check']} -> {req.get('decision')}"
                out.append(line)
        return "\n".join(out)

    def _complete(self, turn: int, data: dict) -> tuple[bool, Optional[str]]:
        block = [] if self._stopped else [b for t in self._tools
                                           if hasattr(t, "blocks_complete") and (b := t.blocks_complete())]
        if block:
            msg = "complete refused: " + "; ".join(block) + ". Reply with wait or close_tie."
            self._add(turn, "error", msg, request=data)
            return False, msg
        pend = [] if self._stopped else self._pending()
        if pend and not self._complete_refused:
            self._complete_refused = True
            msg = ("complete refused: " + "; ".join(pend) + ". Do it first (complete is accepted after it, "
                   "or if you send complete again).")
            self._add(turn, "error", msg, request=data)
            return False, msg
        answer = {k: data.get(k, "") for k in ("check", "decision", "summary", "details", "reasoning")}
        self._session.final_answer = answer
        self._add(turn, "complete", "final answer", request=data, verdict=str(answer["summary"]), findings=answer)
        return True, None

    # ------------------------------------------------------------------ analysis
    def _analysis(self, session: RouterSession) -> None:
        """One LLM call after the session: an engineering analysis of everything that ran."""
        if session.termination_reason in ("llm_error", "not_run", ""):
            return
        if not self._store.states and not (self._watch and self._session.rows):
            return
        material = self._analysis_material(session)
        sys_p, user_p = build_analysis_prompt(self._goal, material, watch=self._watch)
        (self._dir / "analysis_system_prompt.txt").write_text(sys_p)
        self._print("[Session] Writing the final analysis...")
        resp = self._call("analysis", 0, sys_p, user_p, "analysis_system_prompt.txt")
        text = (resp.raw_text or "").strip()
        if _BACKEND_ERROR_RE.match(text):
            session.analysis = f"(final analysis not written: {text})"
            return
        session.analysis = text
        session.analysis_path = self._dir / "analysis.txt"
        session.analysis_path.write_text(text + "\n", encoding="utf-8")

    def _analysis_material(self, session: RouterSession) -> str:
        parts = ["STEADY STATES (made by ExaGO; metrics from the solved files)"] if self._store.states else []
        for s in self._store.states:
            m = s.metrics
            parts.append(f"{s.name}: {s.label}; built on {('S' + str(s.parent)) if s.parent is not None else 'the input case'}; "
                         f"commands {json.dumps(s.commands)}; " + ", ".join(f"{k} {v:.6g}" for k, v in m.items()))
        parts += ["", "EVERY ACTION, IN ORDER (with the agent's check and decision, and the full result)"]
        for e in self._entries:
            if e["turn"] == 0 or e["action"] == "complete":
                continue
            req = e.get("request") or {}
            head = f"--- Turn {e['turn']}: {e['action']}" + (f" ({e.get('tool')})" if e.get("tool") else "")
            parts.append(head)
            if req.get("check"):
                parts.append(f"Agent check: {req['check']} -> {req.get('decision')}")
            if e["action"] == "error":
                parts.append(f"Refused: {e['description']}")
            elif e.get("result_text"):
                parts.append(str(e["result_text"]))
        events = self._dir / "events.csv"
        if self._watch and events.exists():
            parts += ["", "SCENARIO AND SWITCHING EVENTS (time after the start; the agent was not shown the "
                          "scenario events)", events.read_text().strip()]
        parts += ["", f"{self._overall()}", f"Session ended: {session.termination_reason}"]
        if session.final_answer:
            parts += ["", "THE AGENT'S FINAL ANSWER", json.dumps(session.final_answer, indent=1)]
        return "\n".join(parts)

    # ------------------------------------------------------------------ records
    def _write_log(self, text: str) -> None:
        with self._log.open("a", encoding="utf-8") as f:
            f.write(text.rstrip("\n") + "\n")

    def _add(self, turn: int, action: str, description: str = "", **fields) -> None:
        entry = {"turn": turn, "action": action, "description": description,
                 "timestamp": datetime.now().isoformat(timespec="seconds"), **fields}
        self._entries.append(entry)
        req = fields.get("request") or {}
        parts = [f"\n=== Turn {turn}: {action}" + (f" ({fields['tool']})" if fields.get("tool") else "")
                 + (f" - {description}" if description else "")]
        if req.get("check"):
            parts.append(f"Agent check: {req['check']}")
            parts.append(f"Agent decision: {req.get('decision')}")
        if req.get("reasoning"):
            parts.append(f"LLM reasoning: {req['reasoning']}")
        if req:
            parts.append("LLM action: " + json.dumps({k: v for k, v in req.items()
                                                      if k not in ("reasoning", "check", "decision")}))
        if fields.get("result_text"):
            parts.append("Result:\n" + str(fields["result_text"]))
        if fields.get("findings"):
            f = fields["findings"]
            parts.append(f"Final summary: {f.get('summary', '')}\nFinal details:\n{f.get('details', '')}")
        self._write_log("\n".join(parts))

    def _summary_table(self) -> str:
        s = self._session
        lines = [f"Goal: {s.goal}", f"Tools: {', '.join(s.tools) or 'none'}",
                 "Objectives: " + ("; ".join(f"{o['direction']} {o['name']}" for o in s.objectives) or "none"), "",
                 f"{'Turn':>4} | {'Tool':<7} | {'State':<5} | {'What':<40} | Result", "-" * 110]
        for r in s.rows:
            lines.append(f"{r['turn']:>4} | {r['tool']:<7} | {r.get('state', ''):<5} | "
                         f"{str(r.get('what', ''))[:40]:<40} | {r.get('result', '')}")
        lines += ["", s.overall_verdict, f"Stopped: {s.termination_reason}"]
        return "\n".join(lines)

    @log_scope(logger, "Router finalization")
    def _finalize(self, session: RouterSession) -> RouterSession:
        if not session.overall_verdict:
            session.overall_verdict = self._overall()
        journal = {"tool": "router", "goal": session.goal, "session_dir": str(self._dir), "tools": session.tools,
                   "llm": {"backend": self._cfg.llm.backend, "model": self._cfg.llm.model},
                   "objectives": session.objectives,
                   "steady_states": [{"name": s.name, "label": s.label, "tool": s.tool, "application": s.application,
                                      "parent": s.parent, "commands": s.commands, "network": str(s.network_file),
                                      "solved": str(s.solved_file), "metrics": s.metrics}
                                     for s in self._store.states],
                   "termination_reason": session.termination_reason, "error": session.error,
                   "overall_verdict": session.overall_verdict, "summary_rows": session.rows,
                   "final_answer": session.final_answer, "analysis": session.analysis,
                   "entries": self._entries, "token_usage": session.token_ledger.to_dict()}
        session.journal_path = self._dir / "journal.json"
        session.journal_path.write_text(json.dumps(journal, indent=1, default=str), encoding="utf-8")
        session.summary_path = self._dir / "summary.txt"
        session.summary_path.write_text(self._summary_table() + "\n", encoding="utf-8")
        self._write_log(f"\n=== End of session\n{session.overall_verdict}\nStopped: {session.termination_reason}"
                        + (f"\nError: {session.error}" if session.error else "")
                        + f"\n{session.token_ledger.summary_text()}")
        if session.analysis:
            self._write_log(f"\n=== Final analysis\n{session.analysis}")
        print()
        print("=" * 60)
        print("  AgentiGrid router session complete")
        print("=" * 60)
        print(f"  Goal:       {session.goal}")
        print(f"  Tools:      {', '.join(session.tools) or 'none'}")
        print(f"  Stopped:    {session.termination_reason}")
        if session.error:
            print(f"  Error:      {session.error[:300]}")
        print(f"  {session.overall_verdict}")
        if session.final_answer:
            print(f"  Summary:    {session.final_answer.get('summary', '')}")
        print(f"  {session.token_ledger.summary_text()}")
        print(f"  Log:        {self._log}")
        if session.analysis_path:
            print(f"  Analysis:   {session.analysis_path}")
        print(f"  Transcript: {self._transcript}")
        print(f"  Summary:    {session.summary_path}")
        print(f"  Journal:    {session.journal_path}")
        return session
