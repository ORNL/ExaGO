"""Data model for MATPOWER network representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Bus:
    """Single bus in a MATPOWER network."""

    bus_i: int
    type: int
    Pd: float
    Qd: float
    Gs: float
    Bs: float
    area: int
    Vm: float
    Va: float
    baseKV: float
    zone: int
    Vmax: float
    Vmin: float
    # Optional extended fields (present in some files, e.g. ACTIVSg200)
    lam_P: Optional[float] = None
    lam_Q: Optional[float] = None
    mu_Vmax: Optional[float] = None
    mu_Vmin: Optional[float] = None


@dataclass
class Generator:
    """Single generator in a MATPOWER network."""

    bus: int
    Pg: float
    Qg: float
    Qmax: float
    Qmin: float
    Vg: float
    mBase: float
    status: int
    Pmax: float
    Pmin: float
    # Additional standard MATPOWER gen columns (Pc1, Pc2, Qc1min, etc.)
    extra: list[float] = field(default_factory=list)


@dataclass
class Branch:
    """Single branch (line or transformer) in a MATPOWER network."""

    fbus: int
    tbus: int
    r: float
    x: float
    b: float
    rateA: float
    rateB: float
    rateC: float
    ratio: float
    angle: float
    status: int
    angmin: float
    angmax: float
    # Optional extended fields
    extra: list[float] = field(default_factory=list)


@dataclass
class GenCost:
    """Generator cost data."""

    model: int
    startup: float
    shutdown: float
    ncost: int
    coeffs: list[float] = field(default_factory=list)


@dataclass
class MATNetwork:
    """Complete MATPOWER network representation."""

    casename: str
    version: str
    baseMVA: float
    buses: list[Bus]
    generators: list[Generator]
    branches: list[Branch]
    gencost: list[GenCost]
    # Preserve raw comment header for faithful round-tripping
    header_comments: str
    # Any additional data sections not explicitly parsed
    # (e.g., bus_name, gentype, genfuel, areas) stored as raw text
    extra_sections: dict[str, str] = field(default_factory=dict)
