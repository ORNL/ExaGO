"""A small HIL network, ExaGO solutions for it, and a fake HIL device (no Typhoon needed).

Network: buses 1-4, lines 1-2, 2-3, 3-4 and a tie 2-4 (Tie_sw_1 / Tie_line_1).
"""

from __future__ import annotations

import json
from pathlib import Path

from agentigrid.hil_engine.connection import BusReading, HilDevice
from agentigrid.hil_engine.network import HilNetwork, Tie

V_LN = 12.66e3 / 3 ** 0.5

RADIAL_VM = {1: 1.0, 2: 0.99, 3: 0.98, 4: 0.97}
RADIAL_VA = {1: 0.0, 2: -0.1, 3: -0.2, 4: -0.3}
TIE_VM = {1: 1.0, 2: 0.991, 3: 0.985, 4: 0.983}
TIE_VA = {1: 0.0, 2: -0.09, 3: -0.15, 4: -0.18}


def network() -> HilNetwork:
    return HilNetwork(
        model=Path("small.tse"), buses=[1, 2, 3, 4], meter_offset=700,
        ties=[Tie((2, 4), "Tie_sw_1", "Tie_line_1", "Line 704")], feeder_switch="Feeder sw",
        v_ll_kv=12.66,
        lines={"Line 701": (1, 2), "Line 702": (2, 3), "Line 703": (3, 4), "Line 704": (2, 4)},
    )


def write_m(path: Path, vm: dict, va: dict, tie_status: int, line_23_status: int = 1) -> Path:
    """A solved MATPOWER case for the small network."""
    path.parent.mkdir(parents=True, exist_ok=True)
    buses = "\n".join(f"\t{b}\t{3 if b == 1 else 1}\t0\t0\t0.1\t-0.05\t1\t{vm[b]}\t{va[b]}\t12.66\t1\t1.1\t0.9;"
                      for b in (1, 2, 3, 4))
    path.write_text(f"""function mpc = {path.stem}
mpc.version = '2';
mpc.baseMVA = 100;
mpc.bus = [
{buses}
];
mpc.gen = [
\t1\t0.3\t0.1\t100\t-100\t1.0\t100\t1\t100\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0;
];
mpc.branch = [
\t1\t2\t0.01\t0.01\t0\t9999\t9999\t9999\t0\t0\t1\t-360\t360;
\t3\t2\t0.01\t0.01\t0\t9999\t9999\t9999\t0\t0\t{line_23_status}\t-360\t360;
\t3\t4\t0.01\t0.01\t0\t9999\t9999\t9999\t0\t0\t1\t-360\t360;
\t4\t2\t0.02\t0.02\t0\t9999\t9999\t9999\t0\t0\t{tie_status}\t-360\t360;
];
mpc.gencost = [
\t2\t0\t0\t3\t0\t1\t0;
];
""")
    return path


def exago_workdir(tmp_path: Path, tie_status: int = 1, line_23_status: int = 1,
                  with_radial: bool = True) -> Path:
    """An ExaGO work folder: journal with iteration 0 (radial) and 1 (tie closed), as PFLOW."""
    wd = tmp_path / "exago"
    entries = []
    if with_radial:
        write_m(wd / "iter0" / "pflowout.m", RADIAL_VM, RADIAL_VA, 0)
        entries.append(_entry(0, wd / "iter0", "Base case"))
    write_m(wd / "iter1" / "pflowout.m", TIE_VM, TIE_VA, tie_status, line_23_status)
    entries.append(_entry(1, wd / "iter1", "close 2-4"))
    (wd / "journal_20261104_120000.json").write_text(json.dumps({"entries": entries}))
    return wd


def _entry(it: int, run_dir: Path, desc: str) -> dict:
    return {"iteration": it, "description": desc, "convergence_status": "CONVERGED",
            "exago_command": {"mode": "single", "application": "pflow",
                              "argv": ["/bin/pflow", "-netfile", str(run_dir / "case.m"), "-print_output"]}}


class FakeHil(HilDevice):
    """Answers with the ExaGO voltages (times *scale*) for the current tie state."""

    def __init__(self, scale: float = 1.0, accept: bool = True):
        self.scale = scale
        self.accept = accept
        self.closed: set = set()
        self.log: list = []
        self.t = 0.0

    def open(self):
        self.log.append("open")

    def restart(self, closed_ties):
        self.closed = set(closed_ties)
        self.log.append(("restart", sorted(closed_ties)))

    def close_tie(self, tie):
        self.log.append(("close_tie", tie))
        if self.accept:
            self.closed.add(tuple(tie))
        return self.accept

    def read_buses(self):
        self.t += 2.0
        vm, va = (TIE_VM, TIE_VA) if self.closed else (RADIAL_VM, RADIAL_VA)
        return BusReading(self.t, {b: vm[b] * V_LN * self.scale for b in vm}, dict(va))

    def shutdown(self):
        self.log.append("shutdown")
