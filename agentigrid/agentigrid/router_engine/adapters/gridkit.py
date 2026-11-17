"""GridKit in a router session: a transient-stability fault screen from a steady state.

``gridkit_fault_screen`` takes the latest or the best ExaGO steady state, applies it
to the GridKit case's dynamic models (the same conversion as ``--tool gridkit
--from-exago``), checks it with a no-fault run, and runs the fault screen with the
GridKit agent's own study code. The verdict comes from that code.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from agentigrid.gridkit_engine.from_exago import SteadyStateSource
from agentigrid.gridkit_engine.schema_description import command_schema_text
from agentigrid.gridkit_engine.settings import GridkitSettings
from agentigrid.logging_setup import log_scope
from agentigrid.router_engine.adapters.base import ActionResult, Context, ToolAdapter

logger = logging.getLogger("agentigrid.router_engine.adapters.gridkit")


class GridkitAdapter(ToolAdapter):
    name = "gridkit"
    title = "GridKit (transient stability, dynamic simulation)"
    synonyms = ("transient stability, dynamic simulation, time-domain simulation, fault screen, "
                "short circuit, GridKit, DynamicSimulation, ContingencyAnalysis")
    actions = ("gridkit_fault_screen",)

    def __init__(self, case: Path, settings: GridkitSettings, cfg=None, backend=None, loop=None):
        self._case = Path(case)
        self._settings = settings
        self._cfg = cfg
        self._backend = backend
        self._loop = loop                      # a GridkitAgentLoop (tests may pass a fake)
        self._current: Optional[int] = None    # steady state the loop starts from now
        self._opened = False

    def ready(self) -> Optional[str]:
        if not self._case.exists():
            return f"GridKit case {self._case} does not exist"
        if self._loop is None and not self._settings.dynamic_simulation.exists():
            return f"GridKit DynamicSimulation not found at {self._settings.dynamic_simulation}"
        return None

    def start(self, session_dir: Path, goal: str, quiet: bool) -> None:
        if self._loop is None:
            from agentigrid.gridkit_engine.agent_loop import GridkitAgentLoop
            self._loop = GridkitAgentLoop(self._cfg, settings=self._settings, backend=self._backend, quiet=quiet)
        self._loop.open_external(self._case, goal, session_dir / "gridkit")
        self._opened = True

    def card(self) -> str:
        s = self._settings
        return f"""\
TOOL: {self.title}
{self._loop.network_text()}
Action: a fault screen, written exactly like the GridKit "fault_screen" below but
with "action": "gridkit_fault_screen" and "start_from": "latest" | "best":
{{"action": "gridkit_fault_screen", "check": "<what the latest result shows>", "decision": "proceed",
 "start_from": "latest" | "best", ...fault_screen fields..., "reasoning": "<why, from your check>"}}
- The screen starts from the chosen ExaGO steady state, applied to this case's
  dynamic models and checked with a no-fault run first. The GridKit case's own
  operating point is not used, so "start_from" here is "latest" or "best".
- One screen that answers the goal; do not re-run an identical screen.
- Use the defaults unless the goal states a value. "Each bus" / "every bus" ->
  "all_buses": true. A bus (POI) -> "buses": [<bus>]; add "poi"/"hops" only if
  the goal asks for nearby buses or the PJM POI screen.
- Never report a SKIPPED PJM test as passed and never suggest an approximation for
  it; a new generator or load is NOT added to the model (skipped test NEW).
- Contingency analysis -> "application": "ContingencyAnalysis".

GridKit fault_screen fields:
{command_schema_text(s.fault_start_s, s.tmax_s)}"""

    def handle(self, action: str, data: dict, ctx: Context) -> ActionResult:
        with log_scope(logger, "Router GridKit action", turn=ctx.turn):
            s, why = ctx.store.choose(data.get("start_from"))
            if s is None:
                return ActionResult(feedback=f"gridkit_fault_screen: no steady state to start from: {why}. "
                                             "Call exago_solve first.")
            if s.sid != self._current:
                source = SteadyStateSource(ctx.session_dir / "journal.json", s.sid, s.label, s.application,
                                           s.run_dir, s.solved_file)
                ok, text = self._loop.use_steady_state(source, ctx.turn, s.name)
                self._current = s.sid if ok else None
                if not ok:
                    msg = f"gridkit_fault_screen NOT RUN: {s.name} cannot be used as GridKit's operating point.\n{text}"
                    return ActionResult(text=msg, verdict=msg.splitlines()[0], record={"state": s.name},
                                        row={"tool": "GridKit", "state": s.name, "what": s.label,
                                             "result": msg.splitlines()[0]})
            study = {k: v for k, v in data.items() if k not in ("start_from",)}
            study["action"] = "fault_screen"
            study["start_from"] = "latest_steady_state"
            text, feedback = self._loop.fault_screen(ctx.turn, study)
            if feedback:
                return ActionResult(feedback=f"gridkit_fault_screen: {feedback}")
            first = text.splitlines()[0] if text else ""
            up = first.upper()
            passed = False if "FAIL" in up else (True if "PASS" in up else None)
            text = f"Steady state used: {s.name} ({why}).\n{text}"
            return ActionResult(text=text, verdict=f"GridKit from {s.name}: {first}", passed=passed,
                                record={"state": s.name},
                                row={"tool": "GridKit", "state": s.name, "what": str(data.get("description") or "fault screen"),
                                     "passed": passed, "result": first})

    def close(self) -> None:
        if self._opened and self._loop is not None:
            try:
                self._loop.export_journal()
            except Exception:                  # noqa: BLE001 - the router journal is the record
                logger.exception("GridKit journal export failed")
