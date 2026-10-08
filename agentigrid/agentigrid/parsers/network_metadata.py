"""Static network metadata for LLM context.

Computes structural facts about a MATPOWER network that are constant for
the duration of a search session: which buses are slack, which generators
are must-run (Pmin == Pmax) or offline, and the diversity of cost curves
across online generators. Surfacing these up-front prevents the LLM from
wasting iterations on commands that can have no effect (e.g. dispatching
the slack bus, redispatching among generators that share identical cost
curves).
"""

from __future__ import annotations

from agentigrid.parsers.matpower_model import MATNetwork


_UNIFORM_COST_WARNING_TEMPLATE = (
    "WARNING: All {n} online generators share identical quadratic cost "
    "coefficients (c2={c2}, c1={c1}, c0={c0}). Under PFLOW with default "
    "dispatch, total generation cost is essentially fixed by the load "
    "level — redispatch among these generators cannot reduce cost. If the "
    "goal is cost minimization, consider reporting this finding rather "
    "than searching further."
)


def _gencost_tuple(coeffs: list[float]) -> tuple[float, ...]:
    """Return the (c2, c1, c0) tuple for a polynomial cost entry.

    Pads to 3 values with zeros if the coefficients list is shorter.
    """
    if not coeffs:
        return (0.0, 0.0, 0.0)
    padded = list(coeffs) + [0.0] * (3 - len(coeffs))
    return tuple(padded[:3])


def _format_bus_list(buses: list[int], max_inline: int = 12) -> str:
    """Format a bus list for the prompt: inline if short, summarised if long."""
    if not buses:
        return "(none)"
    if len(buses) <= max_inline:
        return ", ".join(str(b) for b in buses)
    head = ", ".join(str(b) for b in buses[:max_inline])
    return f"{head}, ... ({len(buses)} total)"


def network_metadata(net: MATNetwork) -> str:
    """Build the Network Metadata section for the LLM system prompt.

    Args:
        net: Parsed MATPOWER network.

    Returns:
        A multi-line string describing structural facts about the network.
        Aim for under ~25 lines for typical 200-bus networks.
    """
    lines: list[str] = []
    lines.append("=== Network Metadata (static facts, computed once) ===")
    lines.append("")

    # --- Slack / reference bus ---
    slack_buses = [b.bus_i for b in net.buses if b.type == 3]
    if slack_buses:
        lines.append(
            f"Slack/reference bus: {', '.join(str(b) for b in slack_buses)}"
        )
        lines.append(
            "  Note: set_gen_dispatch on the slack bus has no effect — the "
            "slack bus's Pg is determined by power balance."
        )
    else:
        lines.append("Slack/reference bus: (none defined; type=3 not found)")
    lines.append("")

    # --- Must-run generators (Pmin == Pmax) ---
    must_run = [
        g.bus for g in net.generators
        if g.status == 1 and g.Pmax == g.Pmin
    ]
    if must_run:
        lines.append(
            f"Must-run generators (Pmin == Pmax, dispatch fixed): "
            f"{_format_bus_list(must_run)}"
        )
        lines.append(
            "  Note: set_gen_dispatch to a different value on these "
            "generators will be a no-op."
        )
    else:
        lines.append("Must-run generators (Pmin == Pmax): (none)")

    # --- Offline generators ---
    offline = [
        (g.bus, g.Pmax) for g in net.generators if g.status == 0
    ]
    if offline:
        offline_strs = [f"{b}(Pmax={pm:g})" for b, pm in offline[:10]]
        suffix = f", ... ({len(offline)} total)" if len(offline) > 10 else ""
        lines.append(
            f"Offline generators (status=0): {', '.join(offline_strs)}{suffix}"
        )
        lines.append(
            "  Note: candidates for set_gen_status: 1 if more capacity is needed."
        )
    else:
        lines.append("Offline generators: (none)")
    lines.append("")

    # --- Cost coefficient diversity ---
    online_with_cost: list[tuple[int, tuple[float, ...]]] = []
    for i, g in enumerate(net.generators):
        if g.status != 1:
            continue
        if i >= len(net.gencost):
            continue
        gc = net.gencost[i]
        if gc.model != 2:
            continue
        coeffs = _gencost_tuple(gc.coeffs)
        # Skip generators with all-zero cost (placeholder rows for non-cost
        # generators like wind/PV that have no fuel cost).
        if all(c == 0.0 for c in coeffs):
            continue
        online_with_cost.append((g.bus, coeffs))

    unique_tuples: dict[tuple[float, ...], list[int]] = {}
    for bus, coeffs in online_with_cost:
        unique_tuples.setdefault(coeffs, []).append(bus)

    n_unique = len(unique_tuples)
    n_priced = len(online_with_cost)

    if n_priced == 0:
        lines.append("Generator cost curves: (no priced generators)")
    elif n_unique == 1:
        c2, c1, c0 = next(iter(unique_tuples))
        lines.append(
            _UNIFORM_COST_WARNING_TEMPLATE.format(
                n=n_priced, c2=c2, c1=c1, c0=c0,
            )
        )
    elif n_unique <= 4:
        lines.append(
            f"Generator cost curves: {n_unique} unique tuples across "
            f"{n_priced} priced online generators:"
        )
        for coeffs, buses in unique_tuples.items():
            c2, c1, c0 = coeffs
            lines.append(
                f"  c2={c2}, c1={c1}, c0={c0}: "
                f"buses {_format_bus_list(buses)}"
            )
    else:
        lines.append(
            f"Generator cost curves: {n_unique} distinct cost curves across "
            f"{n_priced} priced online generators; see analyze action for "
            "full listing."
        )
        # Summarise marginal-cost diversity since c0 is just a constant.
        marginal: dict[tuple[float, float], int] = {}
        for _, (c2, c1, _c0) in online_with_cost:
            marginal[(c2, c1)] = marginal.get((c2, c1), 0) + 1
        if len(marginal) <= 4:
            lines.append("  Distinct marginal-cost (c2, c1) curves:")
            for (c2, c1), count in sorted(marginal.items()):
                lines.append(
                    f"    c2={c2}, c1={c1}: {count} generator(s)"
                )

    return "\n".join(lines)
