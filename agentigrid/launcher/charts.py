"""Plotly chart builders for the AgentiGrid Launcher.

All chart functions return plotly.graph_objects.Figure objects,
usable both for st.plotly_chart() display and fig.write_image() PNG export.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import plotly.graph_objects as go

from agentigrid.engine.journal import SearchJournal
from agentigrid.parsers.opflow_results import OPFLOWResult

# ── Color Palette ────────────────────────────────────────────────────────────

COLORS = {
    "feasible": "#2ecc71",
    "infeasible": "#e74c3c",
    "failed": "#95a5a6",
    "base_case": "#3498db",
    "best_solution": "#2ecc71",
    "voltage_band": "rgba(52, 152, 219, 0.15)",
    "voltage_line": "#3498db",
    "limit_line": "#e74c3c",
    "gen_base": "#3498db",
    "gen_best": "#2ecc71",
    "gen_bounds": "rgba(0,0,0,0.1)",
    "loading_base": "#3498db",
    "loading_best": "#2ecc71",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _empty_figure(message: str, height: int = 400) -> go.Figure:
    """Return an empty figure with a centered text annotation."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color="gray"),
    )
    fig.update_layout(
        height=height,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


# ── Chart 1: Convergence ────────────────────────────────────────────────────

def convergence_chart(
    journal: SearchJournal,
    highlight_best: bool = True,
    height: int = 400,
    best_iteration: int | None = None,
) -> go.Figure:
    """Line+scatter chart of objective value across iterations.

    Args:
        journal: SearchJournal with iteration entries.
        highlight_best: Annotate the best feasible solution.
        height: Chart height in pixels.
        best_iteration: If provided, annotate this iteration as "best"
            instead of the lowest-cost feasible iteration.

    Returns:
        Plotly Figure.
    """
    entries = journal.entries
    if not entries:
        return _empty_figure("No data available", height)

    # Separate valid and failed iterations
    iters_valid, vals_valid, colors_valid, hover_valid = [], [], [], []
    iters_failed = []
    for e in entries:
        if e.objective_value is not None:
            iters_valid.append(e.iteration)
            vals_valid.append(e.objective_value)
            feasible = e.feasible
            colors_valid.append(
                COLORS["feasible"] if feasible else COLORS["infeasible"]
            )
            status = "Feasible" if feasible else "Infeasible"
            hover_valid.append(
                f"Iter {e.iteration}<br>${e.objective_value:,.2f}<br>{status}"
            )
        elif e.convergence_status == "FAILED":
            iters_failed.append(e.iteration)

    if not iters_valid and not iters_failed:
        return _empty_figure("No data available", height)

    fig = go.Figure()

    # Line + colored scatter for valid iterations
    if iters_valid:
        fig.add_trace(go.Scatter(
            x=iters_valid, y=vals_valid,
            mode="lines",
            line=dict(color="rgba(100,100,100,0.4)"),
            showlegend=False,
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=iters_valid, y=vals_valid,
            mode="markers",
            marker=dict(color=colors_valid, size=9),
            hovertext=hover_valid,
            hoverinfo="text",
            showlegend=False,
        ))

    # Failed iterations as gray X markers on a y=0 baseline
    if iters_failed:
        fig.add_trace(go.Scatter(
            x=iters_failed,
            y=[min(vals_valid) if vals_valid else 0] * len(iters_failed),
            mode="markers",
            marker=dict(
                color=COLORS["failed"], size=10,
                symbol="x",
            ),
            name="Failed",
            hovertext=[f"Iter {i}<br>FAILED" for i in iters_failed],
            hoverinfo="text",
        ))

    # Annotate best solution
    if highlight_best and iters_valid:
        best_iter_found = None
        best_val_found = None

        if best_iteration is not None:
            # Use the provided override
            for it, val in zip(iters_valid, vals_valid):
                if it == best_iteration:
                    best_iter_found = it
                    best_val_found = val
                    break

        if best_iter_found is None:
            # Fall back to lowest-cost feasible
            feasible_pairs = [
                (it, val) for it, val, c in zip(iters_valid, vals_valid, colors_valid)
                if c == COLORS["feasible"]
            ]
            if feasible_pairs:
                best_iter_found, best_val_found = min(feasible_pairs, key=lambda p: p[1])

        if best_iter_found is not None and best_val_found is not None:
            fig.add_annotation(
                x=best_iter_found, y=best_val_found,
                text=f"Best: ${best_val_found:,.2f} (iter {best_iter_found})",
                showarrow=True, arrowhead=2,
                ax=40, ay=-40,
                font=dict(size=11, color=COLORS["best_solution"]),
            )

    fig.update_layout(
        title="Convergence — Objective Value",
        xaxis_title="Iteration",
        yaxis_title="Objective Value ($)",
        height=height,
        margin=dict(l=60, r=20, t=50, b=40),
    )
    return fig


# ── Chart 2: Voltage Range ──────────────────────────────────────────────────

def voltage_range_chart(
    journal: SearchJournal,
    v_min_limit: float = 0.95,
    v_max_limit: float = 1.05,
    height: int = 400,
) -> go.Figure:
    """Area chart showing voltage min/max envelope across iterations.

    Args:
        journal: SearchJournal with iteration entries.
        v_min_limit: Lower voltage limit reference line (p.u.).
        v_max_limit: Upper voltage limit reference line (p.u.).
        height: Chart height in pixels.

    Returns:
        Plotly Figure.
    """
    entries = journal.entries
    if not entries:
        return _empty_figure("No data available", height)

    # Filter out failed iterations (voltage_min == 0)
    iters, v_mins, v_maxs = [], [], []
    for e in entries:
        if e.voltage_min > 0:
            iters.append(e.iteration)
            v_mins.append(e.voltage_min)
            v_maxs.append(e.voltage_max)

    if not iters:
        return _empty_figure("No voltage data available", height)

    fig = go.Figure()

    # Filled area between min and max
    fig.add_trace(go.Scatter(
        x=iters + iters[::-1],
        y=v_maxs + v_mins[::-1],
        fill="toself",
        fillcolor=COLORS["voltage_band"],
        line=dict(color="rgba(0,0,0,0)"),
        name="V range",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=iters, y=v_maxs, mode="lines",
        line=dict(color=COLORS["voltage_line"]),
        name="V_max",
        hovertemplate="Iter %{x}<br>V_max: %{y:.4f} p.u.<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=iters, y=v_mins, mode="lines",
        line=dict(color=COLORS["voltage_line"], dash="dot"),
        name="V_min",
        hovertemplate="Iter %{x}<br>V_min: %{y:.4f} p.u.<extra></extra>",
    ))

    # Reference lines
    fig.add_hline(y=v_min_limit, line_dash="dash", line_color=COLORS["limit_line"],
                  annotation_text=f"{v_min_limit} p.u.")
    fig.add_hline(y=v_max_limit, line_dash="dash", line_color=COLORS["limit_line"],
                  annotation_text=f"{v_max_limit} p.u.")

    fig.update_layout(
        title="Voltage Range Across Iterations",
        xaxis_title="Iteration",
        yaxis_title="Voltage Magnitude (p.u.)",
        height=height,
        margin=dict(l=60, r=20, t=50, b=40),
        showlegend=False,
    )
    return fig


# ── Chart 3: Voltage Profile ────────────────────────────────────────────────

def voltage_profile_chart(
    base_result: OPFLOWResult | None,
    best_result: OPFLOWResult | None,
    v_min_limit: float = 0.95,
    v_max_limit: float = 1.05,
    height: int = 450,
) -> go.Figure | None:
    """Scatter chart comparing bus voltage magnitudes between base and best.

    Args:
        base_result: Base case OPFLOW results.
        best_result: Best feasible OPFLOW results.
        v_min_limit: Lower voltage limit.
        v_max_limit: Upper voltage limit.
        height: Chart height in pixels.

    Returns:
        Plotly Figure, or None if both results are missing.
    """
    if base_result is None and best_result is None:
        return None

    fig = go.Figure()

    # Acceptable voltage band
    fig.add_hrect(
        y0=v_min_limit, y1=v_max_limit,
        fillcolor="rgba(46, 204, 113, 0.08)",
        line_width=0,
        annotation_text="Acceptable range",
        annotation_position="top left",
    )
    fig.add_hline(y=v_min_limit, line_dash="dash",
                  line_color=COLORS["limit_line"], line_width=1)
    fig.add_hline(y=v_max_limit, line_dash="dash",
                  line_color=COLORS["limit_line"], line_width=1)

    if base_result is not None:
        buses_base = sorted(base_result.buses, key=lambda b: b.bus_id)
        fig.add_trace(go.Scatter(
            x=[b.bus_id for b in buses_base],
            y=[b.Vm for b in buses_base],
            mode="lines+markers",
            marker=dict(size=5, color=COLORS["base_case"]),
            line=dict(color=COLORS["base_case"]),
            name="Base Case",
            hovertemplate="Bus %{x}<br>Vm: %{y:.4f} p.u.<extra>Base</extra>",
        ))

    if best_result is not None:
        buses_best = sorted(best_result.buses, key=lambda b: b.bus_id)
        fig.add_trace(go.Scatter(
            x=[b.bus_id for b in buses_best],
            y=[b.Vm for b in buses_best],
            mode="lines+markers",
            marker=dict(size=5, color=COLORS["best_solution"]),
            line=dict(color=COLORS["best_solution"]),
            name="Best Solution",
            hovertemplate="Bus %{x}<br>Vm: %{y:.4f} p.u.<extra>Best</extra>",
        ))

    fig.update_layout(
        title="Voltage Profile — Base Case vs Best Solution",
        xaxis_title="Bus Number",
        yaxis_title="Voltage Magnitude (p.u.)",
        height=height,
        margin=dict(l=60, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ── Chart 4: Generator Dispatch ──────────────────────────────────────────────

def generator_dispatch_chart(
    base_result: OPFLOWResult | None,
    best_result: OPFLOWResult | None,
    height: int = 450,
) -> go.Figure | None:
    """Grouped bar chart comparing generator active power outputs.

    Args:
        base_result: Base case OPFLOW results.
        best_result: Best feasible OPFLOW results.
        height: Chart height in pixels.

    Returns:
        Plotly Figure, or None if both results are missing.
    """
    if base_result is None and best_result is None:
        return None

    # Build generator labels handling duplicates at the same bus
    def _gen_data(result: OPFLOWResult) -> dict[str, dict]:
        bus_counts: dict[int, int] = defaultdict(int)
        data: dict[str, dict] = {}
        for g in result.generators:
            bus_counts[g.bus] += 1
            idx = bus_counts[g.bus]
            label = f"Gen@Bus {g.bus}" if idx == 1 else f"Gen@Bus {g.bus} #{idx}"
            data[label] = {
                "Pg": g.Pg, "Pmin": g.Pmin, "Pmax": g.Pmax, "status": g.status,
            }
        return data

    base_data = _gen_data(base_result) if base_result else {}
    best_data = _gen_data(best_result) if best_result else {}

    # Merge labels, keep only generators online in at least one case
    all_labels = list(dict.fromkeys(list(base_data.keys()) + list(best_data.keys())))
    labels = [
        lbl for lbl in all_labels
        if base_data.get(lbl, {}).get("status", 0) == 1
        or best_data.get(lbl, {}).get("status", 0) == 1
    ]

    if not labels:
        return _empty_figure("No online generators", height)

    # Use horizontal bars if many generators
    horizontal = len(labels) > 20

    fig = go.Figure()

    if base_data:
        pg_base = [base_data.get(l, {}).get("Pg", 0) for l in labels]
        pmin = [base_data.get(l, {}).get("Pmin", 0) for l in labels]
        pmax = [base_data.get(l, {}).get("Pmax", 0) for l in labels]
        # Error bars showing Pmin-Pmax range relative to Pg
        err_minus = [max(pg - pm, 0) for pg, pm in zip(pg_base, pmin)]
        err_plus = [max(px - pg, 0) for pg, px in zip(pg_base, pmax)]

        bar_kw = dict(
            name="Base Case",
            marker_color=COLORS["gen_base"],
            error_y=dict(
                type="data", symmetric=False,
                array=err_plus, arrayminus=err_minus,
                color=COLORS["gen_bounds"], thickness=1, width=3,
            ) if not horizontal else None,
            error_x=dict(
                type="data", symmetric=False,
                array=err_plus, arrayminus=err_minus,
                color=COLORS["gen_bounds"], thickness=1, width=3,
            ) if horizontal else None,
        )
        if horizontal:
            fig.add_trace(go.Bar(y=labels, x=pg_base, orientation="h", **bar_kw))
        else:
            fig.add_trace(go.Bar(x=labels, y=pg_base, **bar_kw))

    if best_data:
        pg_best = [best_data.get(l, {}).get("Pg", 0) for l in labels]
        bar_kw2 = dict(
            name="Best Solution",
            marker_color=COLORS["gen_best"],
        )
        if horizontal:
            fig.add_trace(go.Bar(y=labels, x=pg_best, orientation="h", **bar_kw2))
        else:
            fig.add_trace(go.Bar(x=labels, y=pg_best, **bar_kw2))

    fig.update_layout(
        title="Generator Dispatch — Base Case vs Best Solution",
        barmode="group",
        height=height,
        margin=dict(l=60, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    if horizontal:
        fig.update_layout(xaxis_title="Active Power (MW)", yaxis_title="")
    else:
        fig.update_layout(xaxis_title="", yaxis_title="Active Power (MW)")

    return fig


# ── Chart 5: Line Loading ────────────────────────────────────────────────────

def line_loading_chart(
    base_result: OPFLOWResult | None,
    best_result: OPFLOWResult | None,
    top_n: int = 15,
    height: int = 450,
) -> go.Figure | None:
    """Horizontal bar chart of most loaded lines, comparing base and best.

    Args:
        base_result: Base case OPFLOW results.
        best_result: Best feasible OPFLOW results.
        top_n: Number of most-loaded lines to show.
        height: Chart height in pixels.

    Returns:
        Plotly Figure, or None if both results are missing.
    """
    if base_result is None and best_result is None:
        return None

    def _loading_map(result: OPFLOWResult) -> dict[tuple[int, int], float]:
        lmap: dict[tuple[int, int], float] = {}
        for br in result.branches:
            if br.Slim == 0:
                continue
            loading = max(br.Sf, br.St) / br.Slim * 100
            lmap[(br.from_bus, br.to_bus)] = loading
        return lmap

    base_load = _loading_map(base_result) if base_result else {}
    best_load = _loading_map(best_result) if best_result else {}

    # Merge keys, sort by base loading descending
    all_keys = list(dict.fromkeys(list(base_load.keys()) + list(best_load.keys())))
    all_keys.sort(key=lambda k: base_load.get(k, best_load.get(k, 0)), reverse=True)
    top_keys = all_keys[:top_n]

    if not top_keys:
        return _empty_figure("No branch loading data", height)

    # Reverse so highest-loaded appears at top of horizontal bar chart
    top_keys = top_keys[::-1]
    labels = [f"{f}\u2192{t}" for f, t in top_keys]

    fig = go.Figure()

    if base_load:
        fig.add_trace(go.Bar(
            y=labels,
            x=[base_load.get(k, 0) for k in top_keys],
            orientation="h",
            name="Base Case",
            marker_color=COLORS["loading_base"],
        ))

    if best_load:
        fig.add_trace(go.Bar(
            y=labels,
            x=[best_load.get(k, 0) for k in top_keys],
            orientation="h",
            name="Best Solution",
            marker_color=COLORS["loading_best"],
        ))

    # 100% loading reference line
    fig.add_vline(x=100, line_dash="dash", line_color=COLORS["limit_line"],
                  annotation_text="100%")

    fig.update_layout(
        title=f"Top {min(top_n, len(all_keys))} Most Loaded Lines",
        xaxis_title="Loading (%)",
        yaxis_title="",
        barmode="group",
        height=height,
        margin=dict(l=80, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ── Chart 6: Multi-Objective Trend ──────────────────────────────────────────

def multi_objective_trend_chart(
    journal: SearchJournal,
    height: int = 450,
) -> go.Figure | None:
    """Line chart showing how each tracked objective evolves across iterations.

    Uses multiple y-axes when metrics have different orders of magnitude,
    so all traces are readable regardless of scale differences.

    Args:
        journal: SearchJournal with iteration entries.
        height: Chart height in pixels.

    Returns:
        Plotly Figure, or None if not enough data.
    """
    import math
    from plotly.subplots import make_subplots

    registry = journal.objective_registry
    if not registry.objectives:
        return None

    tracked_entries = [e for e in journal.entries if e.tracked_metrics]
    if len(tracked_entries) < 2:
        return None

    obj_names = [o.name for o in registry.objectives]

    colors_list = [
        "#3498db", "#2ecc71", "#e74c3c", "#f39c12",
        "#9b59b6", "#1abc9c", "#e67e22", "#34495e",
    ]

    # Collect data per objective
    obj_data: dict[str, dict] = {}
    for idx, name in enumerate(obj_names):
        iters = []
        values = []
        for e in tracked_entries:
            v = e.tracked_metrics.get(name)
            if v is not None:
                iters.append(e.iteration)
                values.append(v)
        if values:
            obj = next(o for o in registry.objectives if o.name == name)
            obj_data[name] = {
                "iters": iters,
                "values": values,
                "obj": obj,
                "color": colors_list[idx % len(colors_list)],
                "max_val": max(abs(v) for v in values) if values else 0,
            }

    if not obj_data:
        return None

    # Group objectives by order of magnitude for axis assignment
    def _order(v: float) -> int:
        return int(math.floor(math.log10(max(v, 1e-10))))

    sorted_names = sorted(obj_data.keys(), key=lambda n: obj_data[n]["max_val"], reverse=True)
    axis_groups: list[list[str]] = []
    for name in sorted_names:
        order = _order(obj_data[name]["max_val"])
        placed = False
        for group in axis_groups:
            ref_order = _order(obj_data[group[0]]["max_val"])
            if abs(order - ref_order) <= 1:
                group.append(name)
                placed = True
                break
        if not placed:
            axis_groups.append([name])

    # Limit to 3 axes maximum
    while len(axis_groups) > 3:
        axis_groups[-2].extend(axis_groups[-1])
        axis_groups.pop()

    n_axes = len(axis_groups)

    # Build figure with secondary y-axes if needed
    if n_axes == 1:
        fig = go.Figure()
    else:
        specs = [[{"secondary_y": True}]] if n_axes >= 2 else [[{}]]
        fig = make_subplots(specs=specs)

    # Map each objective to its axis index
    name_to_axis: dict[str, int] = {}
    for ax_idx, group in enumerate(axis_groups):
        for name in group:
            name_to_axis[name] = ax_idx

    # Add traces
    for name, data in obj_data.items():
        obj = data["obj"]
        color = data["color"]
        dash = "solid" if obj.priority == "primary" else "dash" if obj.priority == "secondary" else "dot"
        label = f"{name} [{obj.direction[0]}]"
        ax_idx = name_to_axis[name]

        trace_kwargs = dict(
            x=data["iters"],
            y=data["values"],
            mode="lines+markers",
            name=label,
            line=dict(color=color, dash=dash),
            marker=dict(size=6),
            hovertemplate=f"{name}<br>Iter %{{x}}<br>Value: %{{y:.4f}}<extra></extra>",
        )

        if n_axes == 1:
            fig.add_trace(go.Scatter(**trace_kwargs))
        elif n_axes == 2:
            secondary = (ax_idx == 1)
            fig.add_trace(go.Scatter(**trace_kwargs), secondary_y=secondary)
        else:
            if ax_idx == 0:
                fig.add_trace(go.Scatter(**trace_kwargs))
            elif ax_idx == 1:
                fig.add_trace(go.Scatter(yaxis="y2", **trace_kwargs))
            else:
                fig.add_trace(go.Scatter(yaxis="y3", **trace_kwargs))

        # Add constraint threshold line
        if obj.direction == "constraint" and obj.threshold is not None:
            if n_axes == 1:
                fig.add_hline(
                    y=obj.threshold, line_dash="dot", line_color=color,
                    annotation_text=f"{name} limit", opacity=0.5,
                )
            elif n_axes == 2:
                secondary = (ax_idx == 1)
                fig.add_hline(
                    y=obj.threshold, line_dash="dot", line_color=color,
                    annotation_text=f"{name} limit", opacity=0.5,
                    secondary_y=secondary,
                )
            else:
                x_range = [min(data["iters"]), max(data["iters"])]
                axis_ref = "" if ax_idx == 0 else f"y{ax_idx + 1}"
                fig.add_trace(go.Scatter(
                    x=x_range, y=[obj.threshold, obj.threshold],
                    mode="lines", line=dict(color=color, dash="dot", width=1),
                    showlegend=False, hoverinfo="skip",
                    yaxis=axis_ref if axis_ref else None,
                ))

    # Layout
    layout_kwargs: dict = dict(
        title="Multi-Objective Trends",
        xaxis_title="Iteration",
        height=height,
        margin=dict(l=60, r=80 if n_axes >= 2 else 20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    if n_axes == 1:
        layout_kwargs["yaxis_title"] = "Metric Value"
    elif n_axes == 2:
        left_names = ", ".join(n[:15] for n in axis_groups[0])
        right_names = ", ".join(n[:15] for n in axis_groups[1])
        fig.update_yaxes(title_text=left_names, secondary_y=False)
        fig.update_yaxes(title_text=right_names, secondary_y=True)
    else:
        layout_kwargs["yaxis"] = dict(title=", ".join(n[:12] for n in axis_groups[0]))
        layout_kwargs["yaxis2"] = dict(
            title=", ".join(n[:12] for n in axis_groups[1]),
            overlaying="y", side="right",
        )
        layout_kwargs["yaxis3"] = dict(
            title=", ".join(n[:12] for n in axis_groups[2]),
            overlaying="y", side="right", anchor="free", position=0.95,
        )

    fig.update_layout(**layout_kwargs)
    return fig


# ── Chart 7: Reserve Trajectory ──────────────────────────────────────────────

def reserve_trajectory_chart(
    trajectory: list,
    *,
    height: int = 350,
) -> go.Figure:
    """Step/line chart of hot reserve across greedy de-commitment steps.

    Args:
        trajectory: List of dicts from ``ReserveMinResult.trajectory``.
        height: Chart height in pixels.

    Returns:
        Plotly Figure.
    """
    if not trajectory:
        return _empty_figure("No trajectory data available", height)

    steps = [t["step"] for t in trajectory]
    reserves = [t["reserve"] for t in trajectory]
    hover = []
    for t in trajectory:
        bus = t.get("decommitted_bus")
        gid = t.get("decommitted_gen_id")
        if bus is None:
            hover.append(f"Step 0 — Full commitment<br>Reserve: {t['reserve']:,.1f} MW")
        else:
            hover.append(
                f"Step {t['step']} — De-commit gen@{bus}#{gid}<br>"
                f"Reserve: {t['reserve']:,.1f} MW<br>"
                f"Committed units: {t.get('on_count', '?')}"
            )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps,
        y=reserves,
        mode="lines+markers",
        line=dict(color=COLORS["base_case"], width=2),
        marker=dict(size=9, color=COLORS["base_case"]),
        hovertext=hover,
        hoverinfo="text",
        name="Hot Reserve",
        showlegend=False,
    ))

    # Annotate the final minimum
    if reserves:
        min_val = reserves[-1]
        min_step = steps[-1]
        fig.add_annotation(
            x=min_step, y=min_val,
            text=f"Min: {min_val:,.1f} MW",
            showarrow=True, arrowhead=2,
            ax=50, ay=-35,
            font=dict(size=11, color=COLORS["infeasible"]),
        )

    fig.update_layout(
        title="Hot Reserve Reduction (greedy de-commitment)",
        xaxis_title="Step (accepted de-commitments)",
        yaxis_title="Hot Reserve (MW)",
        height=height,
        margin=dict(l=60, r=20, t=50, b=40),
    )
    return fig
