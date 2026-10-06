#!/usr/bin/env python3
"""Schema exemplars — the AgentiGrid action/command structure as worked examples.

Many invalid proposals are API-structure mistakes, not power-systems mistakes:
e.g. emitting {"action": "set_all_bus_vlimits", ...} as the TOP-LEVEL action
instead of nesting it inside a "modify" action's "commands" list. The command
list is in the system prompt, but only as bare command objects; these exemplars
show complete, correct responses.

Every exemplar is CERTIFIED by AgentiGrid's own code before it is written:
  1. the top-level action is one the agent loop dispatches;
  2. every nested command passes `parse_command` (the loop's parser);
  3. where a held-out reference case can host it, the commands pass the
     validator and are applied by `apply_modifications` with nothing skipped.
Exemplars that cannot be applied to a held-out case (e.g. phase-shifter commands,
since no held-out case has a phase shifter) are labelled "schema-validated only".

Reference case: case118 (kept out of the evaluation tasks the frozen corpus was
audited against). Bus and branch numbers therefore refer to case118; the
exemplars teach structure.

    python rag/tools/rag_schema_exemplars.py                      # write rag/corpus/
    python rag/tools/rag_schema_exemplars.py --inspect            # print, write nothing
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

OUT_NAME = "agentigrid_schema_exemplars.txt"
DEFAULT_REF_CASE = "../datafiles/case118.m"
# Top-level actions dispatched by AgentiGrid's agent loop (agent_loop._run_iteration).
TOP_LEVEL_ACTIONS = ("modify", "complete", "analyze", "sweep", "explore", "select")

# Each exemplar: request, apps it applies to, the response the agent should emit,
# and an optional note. Requests are generic single-step instructions, never the
# evaluated goals (corpus_guard.py audits this against an evaluation spec).
EXEMPLARS: list[dict] = [
    {"request": "Set voltage limits on every bus to 0.97-1.03 pu.",
     "apps": ["opflow", "scopflow"],
     "response": {"action": "modify", "mode": "accumulative",
                  "description": "All-bus voltage band 0.97-1.03 pu",
                  "reasoning": "A system-wide band is one set_all_bus_vlimits command, nested in modify.",
                  "commands": [{"action": "set_all_bus_vlimits", "Vmin": 0.97, "Vmax": 1.03}]},
     "note": "set_all_bus_vlimits is a COMMAND, not an action. Wrong: "
             "{\"action\": \"set_all_bus_vlimits\", \"Vmin\": 0.97, \"Vmax\": 1.03} as the whole "
             "response. Right: nest it in the \"commands\" list of a \"modify\" action."},
    {"request": "Tighten the voltage limits of bus 12 only to 0.98-1.02 pu.",
     "apps": ["opflow", "scopflow"],
     "response": {"action": "modify", "mode": "accumulative",
                  "description": "Bus 12 voltage band 0.98-1.02 pu",
                  "reasoning": "Single-bus limits use set_bus_vlimits.",
                  "commands": [{"action": "set_bus_vlimits", "bus": 12, "Vmin": 0.98, "Vmax": 1.02}]},
     "note": "In OPFLOW/SCOPFLOW, bus voltages are decision variables: constrain them with "
             "set_bus_vlimits / set_all_bus_vlimits, not with set_gen_voltage."},
    {"request": "Increase all loads in the network by 8%.",
     "apps": ["opflow", "scopflow", "pflow"],
     "response": {"action": "modify", "mode": "fresh",
                  "description": "Uniform load scaling x1.08",
                  "reasoning": "Uniform scaling of every load is scale_all_loads with factor 1.08.",
                  "commands": [{"action": "scale_all_loads", "factor": 1.08}]},
     "note": "factor is a multiplier: 1.08 = +8%, 0.9 = -10%. mode \"fresh\" applies the change to "
             "the base case; \"accumulative\" stacks it on the current state."},
    {"request": "Set the load at bus 15 to 95 MW and 30 MVAr.",
     "apps": ["opflow", "scopflow", "pflow"],
     "response": {"action": "modify", "mode": "accumulative",
                  "description": "Bus 15 load 95 MW / 30 MVAr",
                  "reasoning": "Absolute load values use set_load.",
                  "commands": [{"action": "set_load", "bus": 15, "Pd": 95.0, "Qd": 30.0}]}},
    {"request": "Add a new 40 MW load at bus 20 on top of its existing load.",
     "apps": ["opflow", "scopflow", "pflow"],
     "response": {"action": "modify", "mode": "accumulative",
                  "description": "Add 40 MW at bus 20",
                  "reasoning": "add_load_at_bus adds to the existing load; set_load would replace it.",
                  "commands": [{"action": "add_load_at_bus", "bus": 20, "Pd": 40.0, "Qd": 0.0}]}},
    {"request": "Take the generator at bus 10 out of service.",
     "apps": ["opflow", "scopflow", "pflow"],
     "response": {"action": "modify", "mode": "fresh",
                  "description": "Generator at bus 10 offline",
                  "reasoning": "Generator outage is set_gen_status with status 0.",
                  "commands": [{"action": "set_gen_status", "bus": 10, "status": 0}]}},
    {"request": "Set the output of the generator at bus 12 to 150 MW.",
     "apps": ["pflow"],
     "response": {"action": "modify", "mode": "accumulative",
                  "description": "Bus 12 generator Pg = 150 MW",
                  "reasoning": "In PFLOW the dispatch is fixed by set_gen_dispatch; Pg must lie in [Pmin, Pmax].",
                  "commands": [{"action": "set_gen_dispatch", "bus": 12, "Pg": 150.0}]},
     "note": "In OPFLOW/SCOPFLOW the optimizer re-dispatches generators; to force an output there, "
             "bound it instead (this command sets Pg as the starting value)."},
    {"request": "Raise the voltage setpoint of the generator at bus 10 to 1.04 pu.",
     "apps": ["pflow"],
     "response": {"action": "modify", "mode": "accumulative",
                  "description": "Bus 10 Vg = 1.04 pu",
                  "reasoning": "In PFLOW the generator voltage setpoint is enforced, so it is the primary voltage control.",
                  "commands": [{"action": "set_gen_voltage", "bus": 10, "Vg": 1.04}]},
     "note": "Only meaningful in PFLOW. In OPFLOW/SCOPFLOW the solver overrides Vg."},
    {"request": "Add 25 MVAr of capacitive shunt support at bus 44.",
     "apps": ["pflow"],
     "response": {"action": "modify", "mode": "accumulative",
                  "description": "Bus 44 shunt Bs = 25",
                  "reasoning": "Positive Bs is capacitive and raises the local voltage.",
                  "commands": [{"action": "set_shunt_susceptance", "bus": 44, "Bs": 25.0}]}},
    {"request": "Change the tap ratio of the transformer between buses 8 and 5 to 1.0.",
     "apps": ["pflow"],
     "response": {"action": "modify", "mode": "accumulative",
                  "description": "Transformer 8-5 tap = 1.0",
                  "reasoning": "Tap changes use set_tap_ratio on a transformer branch.",
                  "commands": [{"action": "set_tap_ratio", "fbus": 8, "tbus": 5, "ratio": 1.0}]},
     "note": "Only transformer branches (ratio != 0 in the case) accept a tap change."},
    {"request": "Set the phase shifter between buses 30 and 17 to 4 degrees.",
     "apps": ["pflow"],
     "response": {"action": "modify", "mode": "accumulative",
                  "description": "Phase shift 30-17 = 4 deg",
                  "reasoning": "Phase-shifter angle changes use set_phase_shift_angle.",
                  "commands": [{"action": "set_phase_shift_angle", "fbus": 30, "tbus": 17, "angle": 4.0}]},
     "note": "Only branches that are phase shifters (angle != 0 in the case) accept this command."},
    {"request": "Take the line between buses 23 and 24 out of service.",
     "apps": ["opflow", "scopflow", "pflow"],
     "response": {"action": "modify", "mode": "fresh",
                  "description": "Line 23-24 out of service",
                  "reasoning": "A branch outage is set_branch_status with status 0.",
                  "commands": [{"action": "set_branch_status", "fbus": 23, "tbus": 24, "status": 0}]},
     "note": "In SCOPFLOW this permanently changes the topology; contingencies come from the "
             "contingency file, not from set_branch_status."},
    {"request": "Raise the thermal rating of the line between buses 30 and 38 to 250 MVA.",
     "apps": ["opflow", "scopflow", "pflow"],
     "response": {"action": "modify", "mode": "accumulative",
                  "description": "Line 30-38 rateA = 250 MVA",
                  "reasoning": "Branch ratings use set_branch_rate (rateA in MVA).",
                  "commands": [{"action": "set_branch_rate", "fbus": 30, "tbus": 38, "rateA": 250.0}]}},
    {"request": "Scale all loads by 1.05 and set the voltage band on every bus to 0.96-1.04 pu in one step.",
     "apps": ["opflow", "scopflow"],
     "response": {"action": "modify", "mode": "fresh",
                  "description": "Loads x1.05 with 0.96-1.04 pu band",
                  "reasoning": "Several commands go in one commands list, applied in order.",
                  "commands": [{"action": "scale_all_loads", "factor": 1.05},
                               {"action": "set_all_bus_vlimits", "Vmin": 0.96, "Vmax": 1.04}]}},
    {"request": "List the 3 buses nearest to bus 49 before deciding.",
     "apps": ["opflow", "scopflow", "pflow"],
     "response": {"action": "analyze", "query_type": "nearest_neighbors", "bus": 49, "k": 3,
                  "reasoning": "Topology questions use a structured query_type, not free text."}},
    {"request": "The target is met; stop and report.",
     "apps": ["opflow", "scopflow", "pflow"],
     "response": {"action": "complete",
                  "reasoning": "The last feasible solve satisfies the stated target.",
                  "findings": {"summary": "One-sentence answer to the goal, with the key number.",
                               "details": "Iteration, objective value and constraint status supporting it."}},
     "note": "complete carries findings; it does not carry commands."},
]


def certify(ex: dict, ref_net) -> str:
    """Run AgentiGrid's own parser/validator/modifier on an exemplar.

    Returns the certification label; raises ValueError if the exemplar is invalid.
    """
    from agentigrid.engine.commands import parse_command
    from agentigrid.engine.modifier import apply_modifications

    resp = ex["response"]
    action = resp.get("action")
    if action not in TOP_LEVEL_ACTIONS:
        raise ValueError(f"top-level action {action!r} is not dispatched by the agent loop")
    raws = resp.get("commands", [])
    if action != "modify":
        if raws:
            raise ValueError(f"{action!r} must not carry commands")
        return "structure-validated"
    if not raws:
        raise ValueError("modify with no commands")
    cmds = [parse_command(r) for r in raws]  # raises ValueError on a bad command
    if ref_net is None:
        return "schema-validated only"
    apps = ex.get("apps", [])
    app = "pflow" if apps == ["pflow"] else "opflow"
    _net, report = apply_modifications(ref_net, cmds, application=app)
    if report.skipped:
        reasons = "; ".join("; ".join(r) for _c, r in report.skipped)
        reasons = reasons.split(".")[0][:90]
        return f"schema-validated only (not applicable to reference case: {reasons})"
    return "validated: parsed and applied on held-out case118"


def render(ex: dict, label: str) -> str:
    tag = (f"[source: schema exemplar | generated from AgentiGrid's command set | {label} | "
           f"apps: {', '.join(ex.get('apps', []))}]")
    lines = [tag, f"Request: {ex['request']}",
             f"Correct response (JSON): {json.dumps(ex['response'], separators=(', ', ': '))}"]
    if ex.get("note"):
        lines.append(f"Note: {ex['note']}")
    return "\n".join(lines)


def build(ref_case: Path | None) -> list[str]:
    ref_net = None
    if ref_case is not None and ref_case.exists():
        from agentigrid.parsers.matpower_parser import parse_matpower
        ref_net = parse_matpower(ref_case)
    return [render(ex, certify(ex, ref_net)) for ex in EXEMPLARS]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref-case", default=DEFAULT_REF_CASE,
                    help="held-out MATPOWER case used to certify applicability (default: case118)")
    ap.add_argument("--out", default="rag/corpus")
    ap.add_argument("--inspect", action="store_true")
    args = ap.parse_args(argv)

    ref = Path(args.ref_case)
    if not ref.exists():
        print(f"warning: reference case {ref} not found; exemplars will be schema-validated only",
              file=sys.stderr)
    chunks = build(ref)
    if args.inspect:
        print("\n\n".join(chunks))
        return 0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / OUT_NAME).write_text("\n\n".join(chunks) + "\n", encoding="utf-8")
    print(f"Wrote {len(chunks)} schema exemplar(s) -> {out / OUT_NAME}")
    print("Re-ingest with:  rm -rf rag/store && python -m agentigrid.rag.ingest rag/corpus")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
