"""Registry of named checks for GridKit fault screens, and the screen verdict.

Same idea as the ExaGO sweep metrics: the LLM chooses checks BY NAME from this
registry; it never writes the logic. Each check says which variables must be
recorded, so the case copy records exactly what the checks read.

Check signature:
    check(series, context) -> CheckResult
``series`` is a gridkit_parsers.results_parser.TimeSeries; ``context`` holds
``clear_time_s`` (time the fault is cleared).

A check can be made informational (gridkit.damping_in_verdict: false for damping):
it is still computed and reported, with status "info", and never counts in a verdict.

CheckOptions (gridkit config, all off by default) change HOW two checks are computed,
never which checks exist, so the LLM sees the same check names; their descriptions
change only when an option is switched on:
  damping_method "modal"     robust matrix pencil instead of the peak method
  final_voltage_window_s > 0 final voltage as the mean of the last window_s seconds
  final_voltage_margin_pu > 0 final-voltage band widened by this margin
  final_voltage_pre_fault_delta_pu > 0  final voltage judged relative to the pre-fault value:
                             no new violation, and no violation (or value at a limit) worsened
                             by more than this
  short_circuit_info true    short-circuit strength of the faulted buses reported as
                             information (gridkit_engine/ssc.py); never part of a verdict
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Optional

from agentigrid.gridkit_engine import criteria
from agentigrid.gridkit_engine.criteria import CheckResult
from agentigrid.gridkit_parsers.case_parser import MACHINE_CLASSES
from agentigrid.gridkit_parsers.status_parser import RunStatus


def _angles(series):
    return series.device_variable("delta", MACHINE_CLASSES)


def _vm(series):
    return series.bus_variable("Vm")


@dataclass(frozen=True)
class CheckOptions:
    """How the damping and final-voltage checks are computed (gridkit config)."""

    damping_method: str = "peak"            # "peak" or "modal" (criteria.DAMPING_METHODS)
    final_voltage_window_s: float = 0.0     # 0 = last sample
    final_voltage_margin_pu: float = 0.0    # 0 = band exactly FINAL_V_MIN_PU-FINAL_V_MAX_PU
    short_circuit_info: bool = False        # report Ssc of the faulted buses (information only)
    weak_bus_percentile: float = 5.0        # weak = weakest p % of the buses at the same voltage
    final_voltage_pre_fault_delta_pu: float = 0.0   # > 0: final voltage relative to the pre-fault value

    def is_default(self) -> bool:
        return self == CheckOptions()

    def to_dict(self) -> dict:
        return {"damping_method": self.damping_method,
                "final_voltage_window_s": self.final_voltage_window_s,
                "final_voltage_margin_pu": self.final_voltage_margin_pu,
                "short_circuit_info": self.short_circuit_info,
                "weak_bus_percentile": self.weak_bus_percentile,
                "final_voltage_pre_fault_delta_pu": self.final_voltage_pre_fault_delta_pu}


DEFAULT_OPTIONS = CheckOptions()


def _opts(ctx) -> CheckOptions:
    return ctx.get("options", DEFAULT_OPTIONS)


CHECKS: dict[str, tuple[Callable, dict[str, set[str]], str]] = {
    "angle_stability": (
        lambda s, ctx: criteria.check_angle_stability(s.t, _angles(s)),
        {"machines": {"delta"}},
        "machines stay in synchronism (largest change of the rotor-angle difference between any two "
        "machines, relative to the pre-fault steady state, <= 180 deg)",
    ),
    "voltage_recovery": (
        lambda s, ctx: criteria.check_voltage_recovery(s.t, _vm(s), ctx["clear_time_s"]),
        {"buses": {"Vm"}},
        "all buses >= 0.70 pu from 2.5 s after clearing (PJM M-14B G.2.2)",
    ),
    "comed_voltage_envelope": (
        lambda s, ctx: criteria.check_voltage_envelope(s.t, _vm(s), ctx["clear_time_s"]),
        {"buses": {"Vm"}},
        "ComEd zone recovery envelope (PJM M-14B G.9.4)",
    ),
    "damping": (
        lambda s, ctx: criteria.check_damping(s.t, _angles(s), ctx["clear_time_s"],
                                              method=_opts(ctx).damping_method),
        {"machines": {"delta"}},
        "damping ratio >= 3% (PJM M-14B G.2.2); run >= 10 s",
    ),
    "final_voltage": (
        lambda s, ctx: criteria.check_final_voltage(
            _vm(s), t=s.t, window_s=_opts(ctx).final_voltage_window_s,
            margin_pu=_opts(ctx).final_voltage_margin_pu,
            pre_fault_delta_pu=_opts(ctx).final_voltage_pre_fault_delta_pu),
        {"buses": {"Vm"}},
        "bus voltages 0.90-1.10 pu at end of run",
    ),
}


# Why a check is informational only: shown to the LLM and under the per-fault table.
INFO_NOTES: dict[str, str] = {
    "damping": "informational only: the peak-decay estimate is not reliable on multi-mode "
               "responses; not part of the verdict",
}


def info_label(name: str, options: CheckOptions = DEFAULT_OPTIONS) -> str:
    """'damping (informational only: ...)'."""
    if name == "damping" and options.damping_method == "modal":
        return f"{name} (informational only (gridkit.damping_in_verdict: false); matrix pencil; " \
               "not part of the verdict)"
    return f"{name} ({INFO_NOTES.get(name, 'informational only; not part of the verdict')})"


def describe(name: str, options: CheckOptions = DEFAULT_OPTIONS) -> str:
    """Description of a check for the LLM; changes only when an option is on."""
    desc = CHECKS[name][2]
    if name == "damping" and options.damping_method == "modal":
        return ("lowest damping ratio >= 3% (PJM M-14B G.2.2), by matrix pencil: a mode counts only if "
                "found in two fitting windows (clearing + 1 s and + 2 s); modes and swings below "
                f"{criteria.DAMPING_MIN_AMPLITUDE_DEG:g} deg count as settled; run >= 10 s")
    if name == "final_voltage" and (options.final_voltage_window_s or options.final_voltage_margin_pu
                                    or options.final_voltage_pre_fault_delta_pu):
        where = (f"mean of the last {options.final_voltage_window_s:g} s" if options.final_voltage_window_s
                 else "end of run")
        band = f"{criteria.FINAL_V_MIN_PU:.2f}-{criteria.FINAL_V_MAX_PU:.2f} pu"
        if options.final_voltage_margin_pu:
            band += f" with margin {options.final_voltage_margin_pu:g} pu"
        text = f"bus voltages {band}, {where}"
        if options.final_voltage_pre_fault_delta_pu:
            text += (f"; judged relative to the pre-fault value: the fault must not create a violation, "
                     f"nor worsen a voltage already at or beyond a limit before the fault by more than "
                     f"{options.final_voltage_pre_fault_delta_pu:g} pu")
        return text
    return desc


def check_lines(info_only: frozenset[str] = frozenset(),
                options: CheckOptions = DEFAULT_OPTIONS) -> list[str]:
    """One line per registered check for the LLM, informational ones marked as such."""
    return [info_label(name, options) if name in info_only else f"{name}: {describe(name, options)}"
            for name in CHECKS]


def get_check(name: str) -> Callable:
    """The check function registered under *name*."""
    if name not in CHECKS:
        raise ValueError(f"Unknown check '{name}'. Available: {sorted(CHECKS)}")
    return CHECKS[name][0]


def required_variables(names: list[str]) -> dict[str, set[str]]:
    """Union of the variables the named checks read: ``{"machines": {...}, "buses": {...}}``."""
    need: dict[str, set[str]] = {"machines": set(), "buses": set()}
    for n in names:
        for kind, vars_ in CHECKS[n][1].items():
            need[kind] |= vars_
    return need


def register_check(name: str, fn: Callable, needs: dict[str, set[str]], description: str) -> None:
    CHECKS[name] = (fn, needs, description)


def as_info(result: CheckResult) -> CheckResult:
    """*result* reported only: status "info", the status it would have had kept in its data."""
    return replace(result, status="info", data={**result.data, "status_if_judged": result.status})


def run_checks(series, names: list[str], clear_time_s: float,
               info_only: frozenset[str] = frozenset(),
               options: CheckOptions = DEFAULT_OPTIONS) -> list[CheckResult]:
    """Run the named checks; those in *info_only* come back with status "info"."""
    ctx = {"clear_time_s": clear_time_s, "options": options}
    results = [get_check(n)(series, ctx) for n in names]
    return [as_info(r) if r.name in info_only else r for r in results]


# ---------------------------------------------------------------------------
# Screen verdict
# ---------------------------------------------------------------------------

@dataclass
class FaultOutcome:
    """Result of one faulted-bus run."""

    bus: int
    run: RunStatus
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def status(self) -> str:
        """"fail" if the run failed or any check failed, "not_tested" if a
        check could not be judged, otherwise "pass". "info" checks never count."""
        if not self.run.solved or any(c.status == "fail" for c in self.checks):
            return "fail"
        if any(c.status == "not_tested" for c in self.checks):
            return "not_tested"
        return "pass"


# Most informational lines in a verdict (one per fault and check); the rest are in the journal.
MAX_INFO_LINES = 50


def screen_verdict(subject: str, outcomes: list[FaultOutcome], study_note: str,
                   options: CheckOptions = DEFAULT_OPTIONS) -> str:
    """Verdict line, table of every failure (never cut short), untested checks,
    informational checks, and the PJM tests skipped because GridKit lacks their elements."""
    n = len(outcomes)
    failed = [o for o in outcomes if o.status == "fail"]
    untested = [o for o in outcomes if o.status == "not_tested"]
    if failed:
        head = f"VERDICT: {subject} does NOT pass: {len(failed)} of {n} bus faults failed ({study_note})."
    elif untested:
        head = (f"VERDICT: {subject} is NOT CONFIRMED: no failures, but {len(untested)} of {n} "
                f"bus faults have checks that could not be judged ({study_note}).")
    else:
        head = f"VERDICT: {subject} passes all {n} bus faults ({study_note})."
    lines = [head]

    if failed:
        lines += ["", f"FAILED bus faults ({len(failed)} of {n}):",
                  f"{'Bus':>6} | {'Failed':<22} | Detail", "-" * 72]
        for o in sorted(failed, key=lambda o: o.bus):
            if not o.run.solved:
                lines.append(f"{o.bus:>6} | {'solver':<22} | {o.run.reason}")
            for c in o.checks:
                if c.status == "fail":
                    lines.append(f"{o.bus:>6} | {c.name:<22} | {c.detail}")

    if untested:
        lines += ["", f"NOT JUDGED ({len(untested)} of {n}):"]
        for o in sorted(untested, key=lambda o: o.bus):
            for c in o.checks:
                if c.status == "not_tested":
                    lines.append(f"  bus {o.bus}: {c.name} - {c.detail}")

    info = [(o.bus, c) for o in sorted(outcomes, key=lambda o: o.bus)
            for c in o.checks if c.status == "info"]
    if info:
        names = sorted({c.name for _, c in info})
        lines += ["", "INFORMATIONAL, not part of the verdict: " + "; ".join(info_label(n, options) for n in names)]
        lines += [f"  bus {bus}: {c.name} - {c.detail}" for bus, c in info[:MAX_INFO_LINES]]
        if len(info) > MAX_INFO_LINES:
            lines.append(f"  ... {len(info) - MAX_INFO_LINES} more (values in the journal)")

    lines += ["", "PJM tests SKIPPED (GridKit lacks the element):"]
    lines += [f"  {tid}: {name} - {why}" for tid, name, why in criteria.SKIPPED_TESTS]
    return "\n".join(lines)
