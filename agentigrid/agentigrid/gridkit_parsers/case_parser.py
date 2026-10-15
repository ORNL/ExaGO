"""Reader for GridKit phasor-dynamics case files and the element map.

A GridKit case (``*.case.json``) lists buses and devices. Devices refer to
buses through ports named ``bus``, ``bus1``, ``bus2``; governors, stabilizers
and signal sources have no bus port and connect to other devices only through
signal ports. GridKit's fault events refer to a bus fault by its position
among the ``BusFault`` devices in the file (``element_id``), not by bus number.

The element map ties all of this to bus numbers once, when a case is loaded,
and is saved next to the run folders so every later step uses the same map.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger("agentigrid.gridkit_parsers.case")

ELEMENT_MAP_FILE = "element_map.json"
ELEMENT_MAP_FORMAT = 1

MACHINE_CLASSES = frozenset({"Genrou", "Gensal", "GenClassical"})
FAULT_CLASS = "BusFault"


class CaseFileError(ValueError):
    """The case file is missing, not JSON, or lacks buses/devices."""


def load_case(case_file: Path | str) -> dict:
    """Read a GridKit case file and check its top-level structure."""
    path = Path(case_file)
    if not path.exists():
        raise CaseFileError(f"Case file not found: {path}")
    try:
        case = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CaseFileError(f"Case file is not valid JSON: {path} ({exc})") from exc
    if not isinstance(case, dict):
        raise CaseFileError(f"Case file root is not an object: {path}")
    for key in ("buses", "devices"):
        if not isinstance(case.get(key), list):
            raise CaseFileError(f"Case file has no '{key}' list: {path}")
    return case


def _is_bus_port(port: str) -> bool:
    return port.startswith("bus")


def build_element_map(case: dict, case_file: Path | str | None = None) -> dict:
    """Build the element map for a loaded case.

    Returns a JSON-ready dict with:
      buses       one entry per bus: number, class, name, CSV label, device ids
      devices     one entry per device in file order: class, id, position among
                  devices of its class, buses, how the bus was found
                  ("port", "signal", or None), signal ids, CSV label
      bus_faults  element_id (GridKit's fault index), device id, bus
      machines    synchronous machines: id, class, bus
      counts      number of buses, devices, and devices per class
    """
    buses: list[dict] = []
    bus_by_number: dict[int, dict] = {}
    for raw in case["buses"]:
        number = int(raw["number"])
        cls = raw.get("class", "Bus")
        name = str(raw.get("name", number))
        entry = {
            "bus": number,
            "class": cls,
            "name": name,
            "label": f"{cls}_{name}",
            "devices": [],
        }
        buses.append(entry)
        bus_by_number[number] = entry

    devices: list[dict] = []
    class_counts: Counter = Counter()
    id_counts: Counter = Counter()
    devices_by_signal: dict[int, list[int]] = defaultdict(list)
    for index, raw in enumerate(case["devices"]):
        cls = raw.get("class", "")
        dev_id = str(raw.get("id", ""))
        ports = raw.get("ports") or {}
        dev_buses = sorted({int(v) for k, v in ports.items() if _is_bus_port(k)})
        signals = sorted({int(v) for k, v in ports.items() if not _is_bus_port(k)})
        entry = {
            "index": index,
            "class": cls,
            "id": dev_id,
            "class_index": class_counts[cls],
            "buses": dev_buses,
            "via": "port" if dev_buses else None,
            "signals": signals,
            "label": f"{cls}_{dev_id}",
        }
        class_counts[cls] += 1
        id_counts[dev_id] += 1
        devices.append(entry)
        for sig in signals:
            devices_by_signal[sig].append(index)

    # Devices with only signal ports take the buses of the devices they share
    # a signal with. Repeat until nothing changes, so chains such as
    # stabilizer -> exciter -> machine bus resolve.
    changed = True
    while changed:
        changed = False
        for dev in devices:
            if dev["buses"] or not dev["signals"]:
                continue
            found = {
                b
                for sig in dev["signals"]
                for other in devices_by_signal[sig]
                if other != dev["index"]
                for b in devices[other]["buses"]
            }
            if found:
                dev["buses"] = sorted(found)
                dev["via"] = "signal"
                changed = True

    for dev in devices:
        for b in dev["buses"]:
            if b in bus_by_number:
                bus_by_number[b]["devices"].append(dev["id"])
            else:
                logger.warning(
                    "Device %s (%s) refers to bus %s, which is not in the case",
                    dev["id"], dev["class"], b,
                )

    duplicate_ids = sorted(k for k, v in id_counts.items() if v > 1)
    if duplicate_ids:
        logger.warning("Case has repeated device ids: %s", duplicate_ids[:10])

    bus_faults = [
        {
            "element_id": d["class_index"],
            "id": d["id"],
            "bus": d["buses"][0] if d["buses"] else None,
        }
        for d in devices
        if d["class"] == FAULT_CLASS
    ]
    machines = [
        {"id": d["id"], "class": d["class"], "bus": d["buses"][0] if d["buses"] else None}
        for d in devices
        if d["class"] in MACHINE_CLASSES
    ]

    header = case.get("header") or {}
    return {
        "format": ELEMENT_MAP_FORMAT,
        "case_name": header.get("case_name", ""),
        "case_file": str(case_file) if case_file is not None else None,
        "counts": {
            "buses": len(buses),
            "devices": len(devices),
            "by_class": dict(sorted(class_counts.items())),
        },
        "buses": buses,
        "devices": devices,
        "bus_faults": bus_faults,
        "machines": machines,
        "duplicate_ids": duplicate_ids,
    }


def save_element_map(element_map: dict, out_dir: Path | str) -> Path:
    """Write the element map as ``element_map.json`` in *out_dir*."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ELEMENT_MAP_FILE
    path.write_text(json.dumps(element_map, indent=1), encoding="utf-8")
    return path


def load_element_map(path: Path | str) -> dict:
    """Read an element map written by :func:`save_element_map`.

    *path* may be the file itself or the folder that holds it.
    """
    path = Path(path)
    if path.is_dir():
        path = path / ELEMENT_MAP_FILE
    return json.loads(path.read_text(encoding="utf-8"))


def create_element_map(case_file: Path | str, out_dir: Path | str) -> tuple[dict, Path]:
    """Read *case_file*, build its element map and save it in *out_dir*.

    Returns the map and the path of the saved file.
    """
    case = load_case(case_file)
    element_map = build_element_map(case, case_file)
    path = save_element_map(element_map, out_dir)
    counts = element_map["counts"]
    logger.info(
        "Element map for %s: %d buses, %d devices, %d bus faults -> %s",
        element_map["case_name"] or case_file, counts["buses"], counts["devices"],
        len(element_map["bus_faults"]), path,
    )
    return element_map, path


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------

def fault_element_id(element_map: dict, bus: int) -> Optional[int]:
    """GridKit ``element_id`` of the bus fault at *bus*, or None if it has none.

    If a bus has more than one fault device, the first one in the file is used.
    """
    for fault in element_map["bus_faults"]:
        if fault["bus"] == bus:
            return fault["element_id"]
    return None


def devices_at_bus(element_map: dict, bus: int) -> list[dict]:
    """All devices at *bus*, including those tied to it only through signals."""
    return [d for d in element_map["devices"] if bus in d["buses"]]


def _label_index(element_map: dict) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for b in element_map["buses"]:
        index[b["label"]] = {"kind": "bus", "bus": b["bus"], "class": b["class"], "id": None}
    for d in element_map["devices"]:
        index[d["label"]] = {
            "kind": "device",
            "bus": d["buses"][0] if d["buses"] else None,
            "class": d["class"],
            "id": d["id"],
        }
    return index


def _match_column(index: dict[str, dict], column: str) -> Optional[dict]:
    # Labels may contain "_", so try the longest prefix first.
    pos = len(column)
    while (pos := column.rfind("_", 0, pos)) > 0:
        owner = index.get(column[:pos])
        if owner is not None:
            return {**owner, "variable": column[pos + 1:]}
    return None


def column_owners(element_map: dict, columns: list[str]) -> dict[str, Optional[dict]]:
    """Match every CSV column name to its bus or device (see :func:`column_owner`)."""
    index = _label_index(element_map)
    return {c: _match_column(index, c) for c in columns}


def column_owner(element_map: dict, column: str) -> Optional[dict]:
    """Match a GridKit CSV column name to its bus or device.

    GridKit names columns ``<class>_<name>_<variable>`` for buses and
    ``<class>_<id>_<variable>`` for devices. Returns a dict with ``kind``
    ("bus" or "device"), ``bus``, ``class``, ``id`` (devices only) and
    ``variable``; None for the time column or an unknown label.
    """
    return _match_column(_label_index(element_map), column)
