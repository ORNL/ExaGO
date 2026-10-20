"""Reader for GridKit output CSV files.

Turns the CSV (``t`` plus one column per recorded variable) into time series
grouped by bus and by device, using the element map to know which bus or
device each column belongs to.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path

from agentigrid.gridkit_parsers.case_parser import column_owners

logger = logging.getLogger("agentigrid.gridkit_parsers.results")


@dataclass
class TimeSeries:
    """Recorded variables of one GridKit run."""

    t: list[float]
    buses: dict[int, dict[str, list[float]]] = field(default_factory=dict)
    devices: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    device_class: dict[str, str] = field(default_factory=dict)
    device_bus: dict[str, int | None] = field(default_factory=dict)
    unmatched: list[str] = field(default_factory=list)

    def bus_variable(self, name: str) -> dict[int, list[float]]:
        """``{bus: series}`` for one bus variable (e.g. "Vm")."""
        return {b: v[name] for b, v in self.buses.items() if name in v}

    def device_variable(self, name: str, classes: frozenset[str] | None = None) -> dict[str, list[float]]:
        """``{device id: series}`` for one device variable (e.g. "delta")."""
        return {
            d: v[name] for d, v in self.devices.items()
            if name in v and (classes is None or self.device_class[d] in classes)
        }


def read_results(csv_path: Path | str, element_map: dict) -> TimeSeries:
    """Read a GridKit CSV into a :class:`TimeSeries`."""
    with Path(csv_path).open(encoding="utf-8") as fh:
        rows = [r for r in csv.reader(fh) if r]
    if not rows:
        return TimeSeries(t=[])
    header, data = rows[0], rows[1:]
    columns = [[float(r[i]) for r in data] for i in range(len(header))]
    series = TimeSeries(t=columns[0])
    owners = column_owners(element_map, header[1:])
    for i, name in enumerate(header[1:], start=1):
        owner = owners[name]
        if owner is None:
            series.unmatched.append(name)
        elif owner["kind"] == "bus":
            series.buses.setdefault(owner["bus"], {})[owner["variable"]] = columns[i]
        else:
            dev = owner["id"]
            series.devices.setdefault(dev, {})[owner["variable"]] = columns[i]
            series.device_class[dev] = owner["class"]
            series.device_bus[dev] = owner["bus"]
    if series.unmatched:
        logger.warning("CSV columns not in the element map: %s", series.unmatched[:10])
    return series
