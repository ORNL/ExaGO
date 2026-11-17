"""ExaGO in a router session: apply ExaGO commands on a steady state and solve.

The first ``exago_solve`` solves the input case (with any commands given). Every
later one starts from the latest or the best steady state, applies the commands
with the ExaGO agent's own modifier (the same command set) and solves again. The
next solve builds on the steady state's network file, not on ExaGO's saved
solution, because ExaGO's saved file can swap branch ends; the power flow solution
is the same, only the solver's starting point differs.
"""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from typing import Optional

from agentigrid.exago_engine.commands import parse_command
from agentigrid.exago_engine.modifier import apply_modifications
from agentigrid.exago_engine.schema_description import command_schema_text
from agentigrid.exago_parsers.matpower_parser import parse_matpower
from agentigrid.exago_parsers.matpower_writer import write_matpower
from agentigrid.gridkit_engine.from_exago import STEADY_STATE_APPS
from agentigrid.logging_setup import log_scope
from agentigrid.router_engine.adapters.base import ActionResult, Context, ToolAdapter
from agentigrid.router_engine.state import SteadyState, solved_metrics

logger = logging.getLogger("agentigrid.router_engine.adapters.exago")

APPLICATIONS = ("pflow", "opflow")
# Commands that need TCOPFLOW / SOPFLOW inputs a single steady state does not have.
_NOT_FOR_STEADY_STATE = {"scale_load_profile", "scale_wind_scenario"}


class ExagoAdapter(ToolAdapter):
    name = "exago"
    title = "ExaGO (steady-state AC power flow and optimal power flow)"
    synonyms = ("steady-state AC power flow, ACPF, AC power flow, power flow, load flow, PFLOW, "
                "optimal power flow, OPF, ACOPF, OPFLOW, ExaGO")
    actions = ("exago_solve",)

    def __init__(self, case: Path, binaries: dict[str, Path], timeout: float = 600):
        self._case = Path(case)
        self._binaries = binaries          # application -> binary
        self._timeout = timeout

    def ready(self) -> Optional[str]:
        if not self._case.exists():
            return f"ExaGO case {self._case} does not exist"
        if "pflow" not in self._binaries:
            return "ExaGO PFLOW binary not found"
        return None

    def card(self) -> str:
        apps = ", ".join(a for a in APPLICATIONS if a in self._binaries)
        return f"""\
TOOL: {self.title}
Input case: {self._case.name} (MATPOWER). Applications available: {apps}.
Action:
{{"action": "exago_solve",
 "check": "<what the latest result shows>",
 "decision": "proceed",
 "start_from": "latest" | "best",
 "commands": [<ExaGO modification commands, may be empty>],
 "application": "pflow" (default) or "opflow",
 "description": "<short label>", "reasoning": "<why, from your check>"}}
- The FIRST exago_solve solves the input case; give "commands": [] to solve it as
  it is (the base case). Every later exago_solve applies "commands" to the steady
  state chosen by "start_from" and solves again; the result is a new steady state.
- Switching a branch, line or tie into operation / in service means
  {{"action": "set_branch_status", "fbus": <bus>, "tbus": <bus>, "status": 1}};
  taking it out of service means status 0.
- Changes accumulate: a steady state keeps every change of the states it was built on.
- Do not solve the same state again with no commands.

ExaGO modification commands:
{command_schema_text()}"""

    def handle(self, action: str, data: dict, ctx: Context) -> ActionResult:
        with log_scope(logger, "Router ExaGO action", turn=ctx.turn):
            return self._solve(data, ctx)

    def _solve(self, data: dict, ctx: Context) -> ActionResult:
        app = str(data.get("application") or "pflow").lower()
        if app not in APPLICATIONS or app not in self._binaries:
            return ActionResult(feedback=f"exago_solve: application must be one of "
                                         f"{[a for a in APPLICATIONS if a in self._binaries]}, not {app!r}")
        raw_cmds = data.get("commands") or []
        if not isinstance(raw_cmds, list):
            return ActionResult(feedback='exago_solve: "commands" must be a list of command objects')
        cmds = []
        for c in raw_cmds:
            if not isinstance(c, dict):
                return ActionResult(feedback=f"exago_solve: each command must be an object, not {c!r}")
            if str(c.get("action", "")).lower() in _NOT_FOR_STEADY_STATE:
                return ActionResult(feedback=f"exago_solve: {c.get('action')} is for multi-period or "
                                             "scenario studies, not for a single steady state")
            try:
                cmds.append(parse_command(c))
            except ValueError as exc:
                return ActionResult(feedback=f"exago_solve: {exc}")
        store = ctx.store
        parent: Optional[SteadyState] = None
        if store.states:
            parent, why = store.choose(data.get("start_from"))
            if parent is None:
                return ActionResult(feedback=f"exago_solve: cannot start from {data.get('start_from')!r}: {why}")
            if not cmds and app == parent.application:
                return ActionResult(feedback=f"exago_solve: no commands; {parent.name} is already solved "
                                             f"with {app.upper()}. Give the change to make.")
            base_net = parse_matpower(parent.network_file)
            what_from = f"{parent.name} ({why})"
        else:
            base_net = parse_matpower(self._case)
            what_from = "the input case"
        sid = len(store.states)
        run_dir = ctx.session_dir / "exago" / f"S{sid}"
        run_dir.mkdir(parents=True, exist_ok=True)
        net, report = apply_modifications(base_net, cmds, application=app) if cmds else (base_net, None)
        if report and report.skipped:
            reasons = "; ".join(f"{type(c).__name__}: {', '.join(r)}" for c, r in report.skipped)
            return ActionResult(feedback=f"exago_solve: commands not applied ({reasons}); nothing was solved")
        netfile = run_dir / self._case.name
        write_matpower(net, netfile)
        label = data.get("description") or (
            "base case" if not cmds else "; ".join(r for _, r in report.applied))
        ctx.print(f"[Turn {ctx.turn}] ExaGO {app.upper()} {('S' + str(sid))}: {label} (from {what_from})...")
        solved = run_dir / STEADY_STATE_APPS[app]
        solved.unlink(missing_ok=True)
        t0 = time.monotonic()
        res = subprocess.run([str(self._binaries[app]), "-netfile", str(netfile), "-save_output", "-print_output"],
                             cwd=run_dir, capture_output=True, text=True, timeout=self._timeout)
        stdout = run_dir / f"{app}_stdout.txt"
        stdout.write_text(res.stdout + res.stderr)
        out = res.stdout
        converged = ("CONVERGED" in out and "DID NOT CONVERGE" not in out.upper()
                     and res.returncode == 0 and solved.exists())
        iters = next((l.split()[-1] for l in out.splitlines() if l.startswith("Number of iterations")), None)
        iters = int(iters) if iters and iters.isdigit() else None
        cmd_list = [c for c in raw_cmds]
        record = {"exago": {"application": app, "network": str(netfile), "solved": str(solved),
                            "stdout": str(stdout), "converged": converged, "iterations": iters,
                            "from": what_from, "commands": cmd_list,
                            "applied": [r for _, r in report.applied] if report else []},
                  "elapsed_s": time.monotonic() - t0}
        if not converged:
            text = (f"ExaGO {app.upper()} on {label} (from {what_from}) DID NOT CONVERGE "
                    f"(console output: {stdout}). The study stops here.")
            return ActionResult(text=text, verdict=f"ExaGO: DID NOT CONVERGE ({label})",
                                stop=f"ExaGO did not converge ({label})", record=record)
        state = store.add(SteadyState(sid, "exago", app, str(label), netfile, solved, run_dir,
                                      parent.sid if parent else None, cmd_list, solved_metrics(solved), iters))
        m = state.metrics
        vmin = min(parse_matpower(solved).buses, key=lambda b: b.Vm)
        lines = [f"ExaGO {app.upper()} CONVERGED in {iters} iterations: steady state {state.name} = {label} "
                 f"(from {what_from}). Solution: {solved}",
                 f"Lowest voltage {m['voltage_min']:.6f} pu at bus {vmin.bus_i}; highest {m['voltage_max']:.6f} pu.",
                 f"Generation {m['total_generation_mw']:.4f} MW; load {m['total_load_mw']:.4f} MW (includes loads "
                 f"modelled as shunts, Gs x Vm^2); losses {m['active_losses_mw']:.4f} MW; reactive generation "
                 f"{m['total_reactive_gen_mvar']:.4f} MVAr."]
        if report:
            lines.append("Applied: " + "; ".join(r for _, r in report.applied))
            lines += [f"Warning: {w}" for w in report.warnings]
        record["state"] = state.name
        return ActionResult(text="\n".join(lines),
                            verdict=f"ExaGO {app.upper()} {state.name} ({label}): CONVERGED, {iters} iterations",
                            record=record,
                            row={"tool": "ExaGO", "state": state.name, "what": str(label),
                                 "result": f"CONVERGED ({iters} it.), Vmin {m['voltage_min']:.4f} pu, "
                                           f"losses {m['active_losses_mw']:.4f} MW"})
