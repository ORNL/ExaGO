"""Apply GridKit case commands to a COPY of the case (overlay).

The original case file is never changed: commands work on a deep copy that
is written into the run folder.
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from agentigrid.gridkit_engine.commands import AddBusFault, CaseCommand, FaultStudy, RecordVariables
from agentigrid.gridkit_engine.sweep_metrics import required_variables
from agentigrid.gridkit_engine.validation import resolve_fault_buses, validate_command
from agentigrid.gridkit_parsers.case_parser import FAULT_CLASS, MACHINE_CLASSES, build_element_map

logger = logging.getLogger("agentigrid.gridkit_engine.modifier")

ADDED_FAULT_ID = "agentigrid_fault_{bus}"


@dataclass
class ModificationReport:
    applied: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _apply_one(cmd: CaseCommand, case: dict) -> str:
    if isinstance(cmd, AddBusFault):
        for dev in case["devices"]:
            if dev.get("class") == FAULT_CLASS and (dev.get("ports") or {}).get("bus") == cmd.bus:
                params = dev.setdefault("params", {})
                params["R"], params["X"] = cmd.R, cmd.X
                return f"bus {cmd.bus}: existing fault {dev.get('id')} set to R={cmd.R}, X={cmd.X}"
        case["devices"].append({
            "class": FAULT_CLASS,
            "id": ADDED_FAULT_ID.format(bus=cmd.bus),
            "ports": {"bus": cmd.bus},
            "params": {"state0": False, "R": cmd.R, "X": cmd.X},
        })
        return f"bus {cmd.bus}: fault device added (R={cmd.R}, X={cmd.X})"
    if isinstance(cmd, RecordVariables):
        n_m = 0
        for dev in case["devices"]:
            if dev.get("class") in MACHINE_CLASSES:
                dev["mon"] = list(cmd.machines)
                n_m += 1
        for bus in case["buses"]:
            bus["mon"] = list(cmd.buses)
        return (f"recording {cmd.machines} for {n_m} machines and "
                f"{cmd.buses} for {len(case['buses'])} buses")
    raise ValueError(f"Unknown command type {type(cmd).__name__}")


def apply_modifications(
    case: dict, commands: list[CaseCommand], element_map: dict,
) -> tuple[dict, ModificationReport]:
    """Validate and apply *commands* to a deep copy of *case*.

    Invalid commands are skipped and listed in the report. The copy also loses
    any case-level ``monitors`` entry, so GridKit writes the output file the
    solver file names.
    """
    work = copy.deepcopy(case)
    report = ModificationReport()
    if work.pop("monitors", None) is not None:
        report.applied.append("case-level monitors removed (output goes to the solver output_file)")
    for cmd in commands:
        v = validate_command(cmd, element_map)
        report.warnings += v.warnings
        if not v.valid:
            report.errors += v.errors
            continue
        report.applied.append(_apply_one(cmd, work))
    return work, report


def prepare_fault_case(
    case: dict, study: FaultStudy, element_map: dict, only_study_faults: bool = False,
) -> tuple[dict, dict, dict[int, int], ModificationReport]:
    """Case copy for a fault study: a fault device at every study bus and the
    variables the requested checks read.

    With *only_study_faults*, fault devices at other buses (such as a demo
    fault shipped with the case) are left out of the copy. ContingencyAnalysis
    runs every fault device in the file, so they would add runs nobody asked
    for. DynamicSimulation only switches on the fault its events name, so
    other fault devices stay in the copy, inactive.

    Returns (case copy, its element map, {bus: GridKit element_id}, report).
    """
    need = required_variables(study.checks)
    record = RecordVariables(
        machines=sorted(need["machines"] | {"delta", "omega"}),
        buses=sorted(need["buses"] | {"Vm"}),
    )
    buses = resolve_fault_buses(study, element_map)
    commands: list[CaseCommand] = [AddBusFault(bus=b, R=study.R, X=study.X) for b in buses]
    commands.append(record)
    work, report = apply_modifications(case, commands, element_map)
    if only_study_faults:
        keep = set(buses)
        dropped = [d.get("id") for d in work["devices"]
                   if d.get("class") == FAULT_CLASS and (d.get("ports") or {}).get("bus") not in keep]
        if dropped:
            work["devices"] = [d for d in work["devices"]
                               if not (d.get("class") == FAULT_CLASS
                                       and (d.get("ports") or {}).get("bus") not in keep)]
            report.applied.append(f"fault devices at other buses left out: {', '.join(map(str, dropped))}")
    work_map = build_element_map(work)
    fault_ids = {f["bus"]: f["element_id"] for f in reversed(work_map["bus_faults"])}
    return work, work_map, {b: fault_ids[b] for b in buses if b in fault_ids}, report


def write_case(case: dict, path: Path | str) -> Path:
    """Write a case copy as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(case), encoding="utf-8")
    return path
