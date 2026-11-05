"""Start a GridKit study from the most recent ExaGO steady state.

GridKit needs dynamic data (machines, governors, exciters) that a MATPOWER
case does not have, so the GridKit case still supplies the dynamic models; the
ExaGO solution supplies the operating point and the network:

  buses      init Vr/Vi from the solved Vm/Va
  machines   p0/q0 from the solved Pg/Qg; machines of generators that are off
             in ExaGO are removed together with their controls (governor,
             exciter, stabilizer: every device on the machine's signals)
  loads      one constant-impedance LoadZIP per bus from Pd/Qd
  shunts     separate constant-impedance LoadZIP from Gs/Bs
  branches   rebuilt from the online ExaGO branches (R, X, B, tap, phase)

An online ExaGO generator without a machine model is an error: GridKit could
only treat it as a fixed injection, which is not a dynamic study.

"Most recent" means the last simulation in the newest ExaGO journal, not the
best one: GridKit is meant to run on top of the latest steady state.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from agentigrid.logging_setup import log_scope
from agentigrid.exago_parsers.matpower_model import MATNetwork
from agentigrid.exago_parsers.matpower_parser import parse_matpower
from agentigrid.gridkit_parsers.case_parser import MACHINE_CLASSES, _is_bus_port

logger = logging.getLogger("agentigrid.gridkit_engine.from_exago")

# AC steady states that ExaGO saves as one solved MATPOWER case (-save_output).
STEADY_STATE_APPS = {"opflow": "opflowout.m", "pflow": "pflowout.m"}

_GOAL_RE = re.compile(r"\b(most\s+recent|latest|last)\s+steady[\s-]*state\b", re.IGNORECASE)


class SteadyStateError(ValueError):
    """The ExaGO steady state cannot be found or used for GridKit."""


def goal_wants_steady_state(goal: str) -> bool:
    """True if the goal asks to use the most recent steady-state results."""
    return bool(_GOAL_RE.search(goal or ""))


@dataclass(frozen=True)
class SteadyStateSource:
    journal: Path
    iteration: int
    description: str
    application: str
    run_dir: Path
    solved_file: Path

    def describe(self) -> str:
        return (f"ExaGO {self.application.upper()} iteration {self.iteration} "
                f"(\"{self.description}\") from {self.journal.name}, solution {self.solved_file}")


def _netfile(argv: list[str]) -> Optional[Path]:
    for flag, value in zip(argv, argv[1:]):
        if flag == "-netfile":
            return Path(value)
    return None


def find_latest_steady_state(exago_workdir: Path | str) -> SteadyStateSource:
    """The last simulation of the newest ExaGO journal in *exago_workdir*.

    Raises SteadyStateError when that simulation is not a single converged
    AC steady state with a saved solution (no silent fallback to an older one).
    """
    with log_scope(logger, "ExaGO steady state lookup", workdir=exago_workdir):
        exago_workdir = Path(exago_workdir)
        journals = sorted(exago_workdir.glob("journal_*.json"))
        if not journals:
            raise SteadyStateError(f"No ExaGO journal in {exago_workdir}; run an ExaGO study first.")
        journal = journals[-1]          # names carry the timestamp, so sorting by name is by time
        entries = json.loads(journal.read_text()).get("entries", [])
        simulated = [e for e in entries if e.get("exago_command")]
        if not simulated:
            raise SteadyStateError(f"{journal.name} has no simulation.")
        entry = simulated[-1]
        cmd = entry["exago_command"]
        it = entry.get("iteration")
        app = cmd.get("application", "")
        where = f"The most recent ExaGO iteration ({it}, {journal.name})"
        if cmd.get("mode") != "single":
            raise SteadyStateError(f"{where} is a '{cmd.get('mode')}' run of many solves, "
                                   "not a single steady state.")
        if app not in STEADY_STATE_APPS:
            raise SteadyStateError(f"{where} ran {app}; only {sorted(STEADY_STATE_APPS)} give an "
                                   "AC steady state GridKit can start from.")
        # PFLOW's saved file always has converged = 0 (ExaGO writes the OPFLOW
        # flag), so the journal's status is the one to trust.
        status = str(entry.get("convergence_status", ""))
        if not status.upper().startswith("CONVERGED"):
            raise SteadyStateError(f"{where} did not converge (status: {status or 'unknown'}).")
        netfile = _netfile(cmd.get("argv") or [])
        if netfile is None:
            raise SteadyStateError(f"{where}: no -netfile in the recorded command.")
        solved = netfile.parent / STEADY_STATE_APPS[app]
        if not solved.exists():
            raise SteadyStateError(f"{where}: solution file {solved} is missing "
                                   "(runs before ExaGO saved its output cannot be used).")
        return SteadyStateSource(journal, int(it), str(entry.get("description", "")), app,
                                 netfile.parent, solved)


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def _signal_ports(device: dict) -> set:
    return {v for k, v in device.get("ports", {}).items() if not _is_bus_port(k)}


# GridKit cases whose renewables are "<load>_static_gen_<n>" LoadZIP injections
# (no dynamic model). The derived ExaGO case carries them as negative load.
STATIC_GEN_CASES = {"ACTIVSg10k"}

# A dispatch this close to Pmin/Pmax is taken as at the limit (ExaGO -save_output
# writes Pg with 2 decimals).
PG_LIMIT_SNAP_MW = 0.01


def static_gen_loads(case: dict, loads: list[dict]) -> list[dict]:
    """The renewable injections to keep unchanged; empty unless *case* is in
    STATIC_GEN_CASES. They are fixed at the case's own output, which the ExaGO
    case netted from its load, so only the ordinary load is rebuilt from ExaGO
    and each keeps its own load model (constant current vs. impedance)."""
    if case.get("header", {}).get("case_name") not in STATIC_GEN_CASES:
        return []
    return [d for d in loads if "_static_gen_" in d["id"]]


def _snap_to_valve_limits(devices: list[dict], base: float) -> list[str]:
    """Machines at a Tgov1 valve limit, set exactly inside it.

    Tgov1 starts from valve position p0 * base / Trate and rejects it outside
    [Pvmin, Pvmax] with a 4-epsilon tolerance. Its stored limits differ from the
    MATPOWER Pmin/Pmax in the last digits, so a unit dispatched at Pmax can
    fail; within PG_LIMIT_SNAP_MW of a limit it is set 1e-10 (relative) inside.
    """
    machines = {d["ports"].get("pmech"): d for d in devices if d["class"] in MACHINE_CLASSES}
    snapped = []
    for gov in devices:
        if gov["class"] != "Tgov1" or gov["ports"].get("pmech") not in machines:
            continue
        m, p = machines[gov["ports"]["pmech"]], gov["params"]
        pg = m["params"]["p0"] * base
        for limit, inward in ((p["Pvmax"], -1e-10), (p["Pvmin"], 1e-10)):
            mw = limit * p["Trate"]
            if abs(pg - mw) < PG_LIMIT_SNAP_MW:
                m["params"]["p0"] = mw * (1.0 + inward) / base
                snapped.append(m["id"])
                break
    return snapped


def apply_steady_state(case: dict, net: MATNetwork) -> tuple[dict, list[str]]:
    """GridKit case with *net*'s solved operating point and network.

    Returns (new case, notes). Raises SteadyStateError when the two cases
    cannot be matched.
    """
    base = float(net.baseMVA)
    out = copy.deepcopy(case)
    notes: list[str] = []
    buses = {b.bus_i: b for b in net.buses}
    gk_buses = {b["number"]: b for b in out["buses"]}

    # --- buses ---------------------------------------------------------
    missing = sorted(set(buses) - set(gk_buses))
    extra = sorted(set(gk_buses) - set(buses))
    if missing or extra:
        raise SteadyStateError(
            "Bus numbers differ between the ExaGO and GridKit cases"
            + (f"; only in ExaGO: {missing[:10]}" if missing else "")
            + (f"; only in GridKit: {extra[:10]}" if extra else "")
        )
    isolated = sorted(n for n, b in buses.items() if b.type == 4)
    if isolated:
        raise SteadyStateError(f"ExaGO case has isolated buses {isolated[:10]}; "
                               "GridKit cannot start from an islanded network.")
    for n, b in buses.items():
        va = math.radians(b.Va)
        gk_buses[n]["init"] = {"Vr": b.Vm * math.cos(va), "Vi": b.Vm * math.sin(va)}

    devices = out["devices"]

    # --- generators -> machines ------------------------------------------
    gens_at: dict[int, list] = defaultdict(list)
    for g in net.generators:
        gens_at[g.bus].append(g)
    machines_at: dict[int, list[dict]] = defaultdict(list)
    for d in devices:
        if d["class"] in MACHINE_CLASSES:
            machines_at[d["ports"]["bus"]].append(d)
    no_model, ambiguous, removed_machines, at_limit = [], [], [], []
    for bus in sorted(set(gens_at) | set(machines_at)):
        gens = gens_at.get(bus, [])
        online = [g for g in gens if g.status]
        # Machines pair with the MATPOWER rows in file order: both files list a
        # bus's units in the source order (ACTIVSg10k: 1, 10, 2, ... 9), which a
        # numeric sort of the unit ids would break at buses with 10+ units.
        machines = machines_at.get(bus, [])
        if len(online) > len(machines):
            no_model.append(bus)
            continue
        if len(gens) == len(machines):
            pairs = list(zip(gens, machines))
        elif len(online) == len(machines):
            pairs = list(zip(online, machines))
        elif not online:
            pairs = [(None, m) for m in machines]
        else:
            ambiguous.append(bus)
            continue
        for g, m in pairs:
            if g is None or not g.status:
                removed_machines.append(m)
            else:
                # ExaGO writes Pg to 2 decimals; a unit dispatched at Pmin/Pmax
                # can land just outside it, which GridKit's governors reject.
                pg = g.Pg
                for limit in (g.Pmax, g.Pmin):
                    if pg != limit and abs(pg - limit) < PG_LIMIT_SNAP_MW:
                        at_limit.append(m["id"])
                        pg = limit
                        break
                m["params"]["p0"] = pg / base
                m["params"]["q0"] = g.Qg / base
    if no_model:
        raise SteadyStateError(
            f"Online ExaGO generator(s) at bus(es) {no_model} have no dynamic model "
            "in the GridKit case; GridKit cannot simulate them."
        )
    if at_limit:
        notes.append(f"{len(at_limit)} machine(s) within {PG_LIMIT_SNAP_MW} MW of Pmin/Pmax set "
                     f"to the limit (ExaGO's output rounding): {', '.join(at_limit[:10])}")
    at_valve = _snap_to_valve_limits(devices, base)
    if at_valve:
        notes.append(f"{len(at_valve)} machine(s) within {PG_LIMIT_SNAP_MW} MW of their Tgov1 valve "
                     f"limit set just inside it: {', '.join(at_valve[:10])}")
    if ambiguous:
        raise SteadyStateError(
            f"Cannot match ExaGO generators to GridKit machines at bus(es) {ambiguous} "
            "(different unit counts)."
        )
    if removed_machines:
        signals = set().union(*(_signal_ports(m) for m in removed_machines))
        drop = {id(m) for m in removed_machines}
        controls = [d for d in devices if id(d) not in drop and d["class"] not in MACHINE_CLASSES
                    and _signal_ports(d) & signals]
        drop |= {id(d) for d in controls}
        devices[:] = [d for d in devices if id(d) not in drop]
        still_used = set().union(*(_signal_ports(d) for d in devices)) if devices else set()
        out["signals"] = [s for s in out.get("signals", []) if s["signal_id"] in still_used]
        notes.append(
            "Removed machines of generators that are off in ExaGO: "
            + ", ".join(m["id"] for m in removed_machines)
            + (f" (with {len(controls)} control device(s))" if controls else "")
        )

    # --- loads and shunts --------------------------------------------------
    old_loads = [d for d in devices if d["class"] == "LoadZIP"]
    static = static_gen_loads(case, old_loads)
    injection: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])   # MW, MVAr
    for d in static:
        injection[d["ports"]["bus"]][0] += d["params"]["Pnom"] * base
        injection[d["ports"]["bus"]][1] += d["params"]["Qnom"] * base
    if static:
        static_ids = {id(d) for d in static}
        old_loads = [d for d in old_loads if id(d) not in static_ids]
        notes.append(f"{len(static)} static_gen injection(s) at {len(injection)} bus(es) kept "
                     "unchanged; ExaGO load at those buses rebuilt net of them.")
    nonz = sorted({d["ports"]["bus"] for d in old_loads
                   if d["params"].get("alphaI", 0) or d["params"].get("alphaP", 0)})
    if nonz:
        # Pnom applies at the initial voltage either way, so the steady state
        # holds; only the dynamic voltage dependence differs.
        notes.append(f"Loads at buses {nonz[:10]} keep their ZIP fractions.")
    alphas = {d["ports"]["bus"]: (d["params"].get("alphaI", 0.0), d["params"].get("alphaP", 0.0))
              for d in old_loads}
    load_ids = {d["ports"]["bus"]: d["id"] for d in old_loads}
    devices[:] = [d for d in devices if d["class"] != "LoadZIP"] + static
    n_shunts = 0
    for n, b in sorted(buses.items()):
        # ExaGO's Pd/Qd is net of any kept injection, so add it back.
        pd = b.Pd - injection[n][0] if n in injection else b.Pd
        qd = b.Qd - injection[n][1] if n in injection else b.Qd
        if n in injection and n not in load_ids and abs(pd) < 0.05 and abs(qd) < 0.05:
            pd = qd = 0.0     # rounding left over from netting; the bus has no load
        if pd or qd:
            ai, ap = alphas.get(n, (0.0, 0.0))
            devices.append({"class": "LoadZIP", "id": load_ids.get(n, f"load_{n}"),
                            "ports": {"bus": n},
                            "params": {"Pnom": pd / base, "Qnom": qd / base,
                                       "alphaI": ai, "alphaP": ap}})
        if b.Gs or b.Bs:
            vm2 = b.Vm * b.Vm      # Pnom/Qnom are taken at the initial voltage
            devices.append({"class": "LoadZIP", "id": f"shunt_{n}", "ports": {"bus": n},
                            "params": {"Pnom": b.Gs * vm2 / base, "Qnom": -b.Bs * vm2 / base,
                                       "alphaI": 0.0, "alphaP": 0.0}})
            n_shunts += 1
    if n_shunts:
        notes.append(f"{n_shunts} ExaGO shunt(s) added as constant-impedance loads.")

    # --- branches ------------------------------------------------------------
    old_ids: dict[tuple, list[str]] = defaultdict(list)
    for d in devices:
        if d["class"] == "Branch":
            old_ids[tuple(sorted((d["ports"]["bus1"], d["ports"]["bus2"])))].append(d["id"])
    devices[:] = [d for d in devices if d["class"] != "Branch"]
    used: dict[tuple, int] = defaultdict(int)
    n_off = 0
    for br in net.branches:
        key = tuple(sorted((br.fbus, br.tbus)))
        k = used[key]
        used[key] += 1
        if not br.status:
            n_off += 1
            continue
        ids = old_ids.get(key, [])
        devices.append({
            "class": "Branch",
            "id": ids[k] if k < len(ids) else f"branch_{br.fbus}_{br.tbus}_{k + 1}",
            "ports": {"bus1": br.fbus, "bus2": br.tbus},
            "params": {"R": br.r, "X": br.x, "B": br.b,
                       "tap": br.ratio if br.ratio else 1.0,
                       "phase": math.radians(br.angle)},
        })
    only_gk = sorted(f"{key[0]}-{key[1]}" for key, ids in old_ids.items()
                     if used.get(key, 0) < len(ids))
    if only_gk:
        notes.append(f"Branches only in the GridKit case were left out (ExaGO's network is "
                     f"the steady state): {', '.join(only_gk[:10])}")
    if n_off:
        notes.append(f"{n_off} branch(es) out of service in ExaGO left out.")

    out.setdefault("header", {})["case_comments"] = "operating point from ExaGO"
    return out, notes


def exago_shifts_to_degrees(net: MATNetwork) -> list[str]:
    """Convert the branch phase shifts of an ExaGO saved solution to degrees.

    ExaGO reads MATPOWER shift angles in degrees and converts them to radians
    (psreaddata.cpp), but -save_output writes them back still in radians
    (psoutput.cpp), so opflowout.m/pflowout.m break the MATPOWER convention.
    Bus angles (Va) are written in degrees and are left as they are.
    """
    shifted = [br for br in net.branches if br.angle]
    for br in shifted:
        br.angle = math.degrees(br.angle)
    if not shifted:
        return []
    return [f"{len(shifted)} phase shift(s) in ExaGO's solution converted from radians to degrees: "
            + ", ".join(f"{br.fbus}-{br.tbus} {br.angle:.3f} deg [{br.angle!r} deg]" for br in shifted[:10])]


def _end_flows(br, vf: complex, vt: complex, base: float) -> tuple[complex, complex]:
    """MVA entering *br* at fbus with the tap and phase shift on the fbus side,
    and at fbus if they were on the tbus side instead."""
    ys = 1.0 / complex(br.r, br.x)
    ysh = 0.5j * br.b
    tap = (br.ratio or 1.0) * complex(math.cos(math.radians(br.angle)), math.sin(math.radians(br.angle)))
    i_tap_here = (ys + ysh) / abs(tap) ** 2 * vf - ys / tap.conjugate() * vt
    i_tap_there = (ys + ysh) * vf - ys / tap * vt
    return vf * i_tap_here.conjugate() * base, vf * i_tap_there.conjugate() * base


def exago_branch_directions(net: MATNetwork) -> list[str]:
    """Put back the from/to buses of transformers that ExaGO wrote reversed.

    ExaGO's -save_output (psoutput.cpp) writes some branches with fbus and tbus
    swapped but keeps the tap ratio and phase shift, which belong to the
    original from bus, so a reader puts them on the wrong winding (ACTIVSg10k:
    3562 of 12706 branches, 476 of them off-nominal transformers). Its reported
    flows are right for the real direction, so for each transformer the
    direction whose computed flow matches the reported Pf/Qf at fbus is kept.
    Run after exago_shifts_to_degrees. Lines (tap 1, no shift) are the same
    either way and are left as written.
    """
    buses = {b.bus_i: b.Vm * complex(math.cos(math.radians(b.Va)), math.sin(math.radians(b.Va)))
             for b in net.buses}
    base = net.baseMVA
    flipped = []
    for br in net.branches:
        if not br.status or ((br.ratio in (0.0, 1.0)) and not br.angle) or len(br.extra) < 2:
            continue
        reported = complex(br.extra[0], br.extra[1])
        as_written, reversed_ = (abs(s - reported) for s in
                                 _end_flows(br, buses[br.fbus], buses[br.tbus], base))
        if reversed_ < as_written:
            br.fbus, br.tbus = br.tbus, br.fbus
            br.extra[:4] = br.extra[2:4] + br.extra[:2]
            flipped.append(br)
    if not flipped:
        return []
    return [f"{len(flipped)} transformer(s) written by ExaGO with from/to reversed put back in "
            "their real direction (tap on the right side): "
            + ", ".join(f"{br.fbus}-{br.tbus} tap {br.ratio:g}" for br in flipped[:10])]


def build_from_exago(case: dict, source: SteadyStateSource) -> tuple[dict, list[str]]:
    """Read *source*'s solved case and apply it to the GridKit *case*."""
    with log_scope(
        logger, "ExaGO to GridKit conversion",
        application=source.application,
        run_dir=source.run_dir,
    ):
        net = parse_matpower(source.solved_file)
        notes = exago_shifts_to_degrees(net)
        notes += exago_branch_directions(net)
        out, more = apply_steady_state(case, net)
        return out, notes + more
