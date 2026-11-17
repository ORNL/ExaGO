"""The connection to Typhoon HIL Control Center (THCC) and the virtual HIL.

The only module that calls the Typhoon API. The engine uses :class:`HilDevice`;
tests replace it with a fake.

THCC is started headless (``typhoon_hil.exe -hapi``) when it is not running, with
its output detached so it does not hold the caller's terminal or pipe. A THCC this
session started is closed at the end (``hil.close_started_thcc``); one that was
already running is left as it was.

Switching rule (study plan, step 4): ties are only CLOSED while the simulation
runs. Any other switch state is set before a start (``restart``): opening a tie
live leaves the virtual HIL in a wrong state near the tie.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from agentigrid.hil_engine.network import HilNetwork
from agentigrid.hil_engine.settings import HilSettings
from agentigrid.logging_setup import log_scope

logger = logging.getLogger("agentigrid.hil_engine.connection")


class HilError(RuntimeError):
    """The HIL could not do what was asked."""


@dataclass
class BusReading:
    sim_time: float
    volts: dict[int, float]          # bus -> V rms line-to-neutral (phase a)
    degrees: dict[int, float]        # bus -> angle, degrees


class HilDevice:
    """Interface the engine uses (and tests fake)."""

    def open(self) -> None: ...
    def restart(self, closed_ties: set[tuple[int, int]]) -> None: ...
    def close_tie(self, tie: tuple[int, int]) -> bool: ...
    def read_buses(self) -> BusReading: ...
    def set_input(self, name: str, value: float) -> bool: ...
    def describe(self) -> str: return "HIL"
    def shutdown(self) -> None: ...


class TyphoonHil(HilDevice):
    """Typhoon virtual HIL through ``typhoon.api``."""

    def __init__(self, settings: HilSettings, network: HilNetwork):
        self._s = settings
        self._net = network
        self._hil = None
        self._started_pids: set[int] = set()
        self._model_loaded = False

    # -- THCC ---------------------------------------------------------------
    def _thcc_pids(self) -> set[int]:
        exe = str(self._s.thcc_executable)
        pids = set()
        for proc in Path("/proc").iterdir():
            if not proc.name.isdigit():
                continue
            try:
                argv = (proc / "cmdline").read_bytes().split(b"\0")
            except OSError:
                continue
            if argv and argv[0].decode(errors="replace") == exe:
                pids.add(int(proc.name))
        return pids

    def _start_thcc(self) -> None:
        exe = self._s.thcc_executable
        if not exe.exists():
            raise HilError(f"Typhoon HIL Control Center not found at {exe} (hil.typhoon_dir)")
        before = self._thcc_pids()
        logger.info("Starting THCC headless: %s -hapi", exe)
        subprocess.Popen([str(exe), "-hapi"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True, cwd=str(exe.parent))
        time.sleep(2)
        self._started_pids = self._thcc_pids() - before

    @staticmethod
    def _thcc_answering(timeout_ms: int = 3000) -> bool:
        """True when THCC announces its services on the API announcement ports.

        Checked before the first API call: if the API finds no THCC it starts one
        itself, attached to this process's output.
        """
        import zmq
        from typhoon.api.common import SERVICE_REGISTRY_HEADER, get_announcement_range

        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.LINGER, 0)
        try:
            start, end = get_announcement_range()
            for port in range(start, end + 1):
                sock.connect(f"tcp://localhost:{port}")
            sock.setsockopt_string(zmq.SUBSCRIBE, "")
            deadline = time.monotonic() + timeout_ms / 1000
            while time.monotonic() < deadline:
                if sock.poll(200):
                    msg = sock.recv_json()
                    result = (msg or {}).get("result") or [None]
                    if result[0] == SERVICE_REGISTRY_HEADER:
                        return True
            return False
        finally:
            sock.close()

    def open(self) -> None:
        with log_scope(logger, "HIL open", model=self._net.model):
            os.environ["TYPHOONPATH"] = str(self._s.typhoon_dir)
            try:
                import typhoon.api.common  # noqa: F401
            except ImportError as exc:
                raise HilError("typhoon-hil-api is not installed in this Python environment") from exc
            if not self._thcc_answering():
                if not self._thcc_pids():
                    self._start_thcc()
                deadline = time.monotonic() + self._s.startup_timeout
                while not self._thcc_answering(5000):
                    if time.monotonic() > deadline:
                        raise HilError(f"THCC did not answer within {self._s.startup_timeout:.0f} s")
                logger.info("THCC is answering")
            from typhoon.api import hil
            self._hil = hil
            cpd = self._compiled_model()
            t0 = time.monotonic()
            if hil.is_simulation_running():
                hil.stop_simulation()
            if not hil.load_model(str(cpd), vhil_device=True):
                raise HilError(f"Loading {cpd} on the virtual HIL failed")
            self._model_loaded = True
            logger.info("Model %s loaded on the virtual HIL in %.1f s", cpd.name, time.monotonic() - t0)

    def _compiled_model(self) -> Path:
        """The .cpd, compiled for the virtual device if missing or older than the schematic."""
        cpd = self._s.compiled_model_file()
        model = self._net.model
        if cpd.exists() and cpd.stat().st_mtime >= model.stat().st_mtime:
            return cpd
        from typhoon.api.schematic_editor import model as mdl
        with log_scope(logger, "HIL compile", model=model):
            mdl.load(str(model))
            mdl.set_model_property_value("hil_device", self._s.device)
            mdl.set_model_property_value("hil_configuration_id", self._s.configuration_id)
            mdl.set_ns_var("t_exec", self._s.t_exec)
            mdl.save()
            if not mdl.compile():
                raise HilError(f"Compiling {model} failed")
            cpd = Path(mdl.get_compiled_model_file(str(model)))
        return cpd

    # -- switching and readings ---------------------------------------------
    def _need(self):
        if self._hil is None or not self._model_loaded:
            raise HilError("HIL not opened")
        return self._hil

    def restart(self, closed_ties: set[tuple[int, int]]) -> None:
        """Stop, set every tie (closed if listed, else open) and the feeder breaker, start, settle."""
        hil = self._need()
        if hil.is_simulation_running():
            hil.stop_simulation()
        for t in self._net.ties:
            if not hil.set_scada_input_value(t.scada_input, 1.0 if t.buses in closed_ties else 0.0):
                raise HilError(f"Setting {t.scada_input} failed")
        if self._net.feeder_switch and not hil.set_contactor(self._net.feeder_switch,
                                                             swControl=True, swState=True):
            raise HilError(f"Closing {self._net.feeder_switch} failed")
        hil.start_simulation()
        hil.wait_sec(self._s.settle_s)

    def close_tie(self, tie: tuple[int, int]) -> bool:
        hil = self._need()
        t = self._net.tie(*tie)
        if t is None:
            raise HilError(f"No tie switch between buses {tie[0]} and {tie[1]}")
        ok = bool(hil.set_scada_input_value(t.scada_input, 1.0))
        hil.wait_sec(self._s.settle_s)
        return ok

    def set_input(self, name: str, value: float) -> bool:
        """Set a SCADA input (e.g. a variable load's reference) now, before or during a run."""
        return bool(self._need().set_scada_input_value(name, float(value)))

    def read_buses(self) -> BusReading:
        hil = self._need()
        vsig, asig = self._net.signals()
        n = max(1, self._s.read_samples)
        sums_v = [0.0] * len(vsig)
        sums_a = [0.0] * len(asig)
        t = hil.get_sim_time()
        for k in range(n):
            vals = hil.read_analog_signals(signals=vsig + asig)
            if vals is None or any(v is None for v in vals):
                raise HilError("Reading the bus meters failed")
            for i in range(len(vsig)):
                sums_v[i] += vals[i]
                sums_a[i] += vals[len(vsig) + i]
            if k < n - 1:
                time.sleep(self._s.read_interval_s)
        buses = self._net.buses
        return BusReading(sim_time=t,
                          volts={b: sums_v[i] / n for i, b in enumerate(buses)},
                          degrees={b: sums_a[i] / n for i, b in enumerate(buses)})

    def describe(self) -> str:
        return f"virtual {self._s.device} (configuration {self._s.configuration_id})"

    def shutdown(self) -> None:
        """Leave the simulation in the rest state; close THCC if this session started it."""
        if self._hil is not None and self._model_loaded:
            try:
                if self._s.rest_state == "stopped":
                    self._hil.stop_simulation()
                else:
                    closed = {t.buses for t in self._net.ties} if self._s.rest_state == "meshed" else set()
                    self.restart(closed)
            except Exception as exc:     # noqa: BLE001 - clean-up must not hide the result
                logger.warning("Could not leave the HIL in the %s state: %s", self._s.rest_state, exc)
        if self._started_pids and self._s.close_started_thcc:
            for pid in self._started_pids:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            logger.info("Closed THCC processes started by this session: %s", sorted(self._started_pids))
