"""Short text description of a GridKit case for the LLM, from the element map."""

from __future__ import annotations

from collections import Counter

from agentigrid.gridkit_parsers.case_parser import MACHINE_CLASSES


def network_summary(element_map: dict, max_list: int = 40) -> str:
    counts = element_map["counts"]
    by_class = counts["by_class"]
    machines = element_map["machines"]
    machine_models = Counter(m["class"] for m in machines)
    fault_buses = sorted({f["bus"] for f in element_map["bus_faults"] if f["bus"] is not None})
    machine_buses = sorted({m["bus"] for m in machines if m["bus"] is not None})

    def _short(values: list[int]) -> str:
        text = ", ".join(str(v) for v in values[:max_list])
        return text + (f", ... ({len(values)} total)" if len(values) > max_list else "")

    other = {k: v for k, v in by_class.items()
             if k not in MACHINE_CLASSES and k not in ("Branch", "BusFault")}
    lines = [
        f"Case: {element_map.get('case_name') or '(unnamed)'}",
        f"Buses: {counts['buses']}   Branches (lines and transformers): {by_class.get('Branch', 0)}",
        f"Synchronous machines: {len(machines)} ("
        + ", ".join(f"{n} {c}" for c, n in sorted(machine_models.items())) + ")",
        f"Machine buses: {_short(machine_buses)}",
        "Other devices: " + (", ".join(f"{n} {c}" for c, n in sorted(other.items())) or "none"),
        f"Buses with a fault device in the file: {len(fault_buses)} of {counts['buses']}"
        + (f" ({_short(fault_buses)})" if fault_buses else "")
        + ". AgentiGrid adds fault devices to its case copy for any other bus it faults.",
    ]
    return "\n".join(lines)
