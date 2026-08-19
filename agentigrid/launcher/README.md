# AgentiGrid Launcher

A Streamlit-based GUI for configuring, running, and monitoring AgentiGrid search sessions. The launcher provides a web interface for setting up simulations, watching live progress, exploring results with interactive charts, and generating PDF reports.

## Prerequisites

- Python 3.10+
- AgentiGrid installed from the project root: `pip install -e .`
- ExaGO binaries configured in `applications/` (see main project README)
- An LLM API key (Anthropic or OpenAI) set as an environment variable

## Installation

```bash
pip install -r launcher/requirements.txt
```

This installs Streamlit, Plotly, kaleido (for PDF chart export), ReportLab (for PDF generation), and PyYAML.

## Running

The launcher must be run from the **project root** directory so that config paths (`./applications`, `./data`, `./workdir`) resolve correctly.

```bash
# Recommended: use the launch script
./launcher/run.sh

# Or manually from project root
cd /path/to/agentigrid
streamlit run launcher/app.py
```

Do **not** run `streamlit run app.py` from inside `launcher/` — paths will not resolve.

## Features

### Configuration Panel (Sidebar)
- Select MATPOWER base case files from the `data/` directory
- Choose LLM backend (Anthropic, OpenAI, Ollama, Ollama-Cloud) with auto-populated model defaults
- Adjust temperature, iteration mode (accumulative/fresh), and max iterations
- Search mode selector: **Standard** (goal-directed search) or **Stress Test** (adversarial contingency exploration)
- Application selector: choose between supported ExaGO applications (OPFLOW for full AC OPF, DCOPFLOW for fast DC approximation, SCOPFLOW for security-constrained OPF, TCOPFLOW for multi-period OPF, SOPFLOW for stochastic OPF, PFLOW for LLM-driven power flow analysis)
- Contingency file selector: appears when SCOPFLOW is selected, showing available `.cont` files from the `data/` directory
- Load profile selectors: appear when TCOPFLOW is selected, auto-matching profile CSV files to the selected base case (layered fallback: exact prefix → stripped suffix → all profiles). Includes active load (P), reactive load (Q), and optional wind generation profile dropdowns
- Temporal parameters: appear when TCOPFLOW is selected — Duration (hours), Time-step (minutes), and Generator ramp coupling toggle
- Scenario file selector: appears when SOPFLOW is selected, auto-matching wind scenario CSV files to the selected base case (layered fallback: exact prefix → stripped suffix → all scenarios). Supports both single-period and multi-period scenario formats
- SOPFLOW parameters: appear when SOPFLOW is selected — Solver (IPOPT or EMPAR) and First/second stage coupling toggle. MPI core count (`--np`) is available for EMPAR
- PFLOW info: when PFLOW is selected, a note explains that PFLOW is analysis (not optimization) and the LLM drives the search directly. No additional configuration files are needed
- **Sweep concurrency controls** (OPFLOW only): a "Run sweep in parallel" checkbox and a "Concurrent solves" number input (1–128, default = `min(cpu_count, 16)`) control how many OPFLOW subprocesses run at once during a sweep action. When more than one worker is used, each worker's BLAS thread count is pinned to 1 to avoid oversubscription. Set "Concurrent solves" to 1 to force sequential execution (useful for debugging or on resource-constrained hosts). These controls are disabled for non-OPFLOW applications
- **Concurrent explore/select** (PFLOW only): checkbox enables the LLM to propose multiple simulation variants per iteration. Each explore action runs 2–8 configurations concurrently, computes a Pareto front, and presents non-dominated variants for selection. The "Max variants per explore" input controls the parallelism level (2–16, default 8). When enabled, the live monitor shows an explore status panel with variant feasibility, Pareto markers (★), and key metrics for each variant. The iteration log shows explored variants for `select` entries
- Preset goal library with common optimization tasks (minimize cost, fix voltage violations, stress testing, multi-objective, PFLOW-specific goals like loadability search and voltage improvement, etc.)
- Custom goal input via free-text area

### Live Search Monitor
- Two-column layout: iteration timeline (left) and live charts (right)
- Expandable iteration cards showing LLM reasoning, commands, and key metrics
- Real-time convergence chart (objective value vs iteration, color-coded by feasibility; not shown for PFLOW which has no optimization objective — a note directs to the voltage range chart instead)
- Live voltage range chart with limit reference lines
- Progress stats: iteration count, feasible count, best cost found
- Phase status indicator (sending prompt, running simulation, parsing results, etc.). When concurrent PFLOW is active, the phase indicator shows "Running 5 variants..." during parallel simulation and "Computing Pareto front..." during result analysis
- Stop button for graceful search termination

### Results & Summary View (Three Tabs)
- **Overview**: Summary metrics, base-vs-best comparison table, convergence chart, voltage range chart
- **Detailed Results**: Voltage profile comparison, generator dispatch chart, line loading chart, full iteration history table

> **Note:** When using DCOPFLOW, voltage profile and voltage range charts show flat lines at 1.0 pu (expected — DC approximation fixes all voltages). Line loading and generator dispatch charts remain informative.
> **Note:** When using PFLOW, the convergence chart is not displayed (PFLOW has no objective value). The "Best Cost" metric is replaced with a feasibility-based "Best Solution" indicator. Cost columns in iteration tables show "N/A (no optimization)" instead of "$0.00".
- **Analysis & Report**: On-demand LLM-generated analytical summary, auto-generated search narrative, PDF report download

### Multi-Objective Tracking

When a search involves multiple objectives (e.g., minimize cost while constraining voltage), the results view displays:

- **Multi-objective trend chart** — shows how each tracked metric evolves across iterations, with separate y-axes for metrics at different scales (e.g., cost in thousands vs voltage deviation in hundredths), color-coded traces by priority (solid for primary, dashed for secondary, dotted for watch-only), and constraint threshold lines
- **Tradeoff analysis** — the post-search LLM analysis identifies key tradeoffs and can recommend multiple solutions
- **Preference evolution history** — expandable section showing when objectives were registered, reprioritized, or proposed by the LLM

Objectives can be added mid-search via the steering panel (e.g., "also track line loading"). When new objectives are added, metrics are backfilled for all previous iterations automatically.

### Interactive Steering Panel

The live search monitor includes a steering panel (right column, below the progress stats) that lets you guide the LLM mid-search without stopping it.

**Controls:**
- **Directive input** — free-text field for the steering instruction
- **Augment** — injects the directive alongside the current goal; the LLM considers it as an additional constraint or preference
- **Replace** — injects the directive as a full goal replacement; previous directives are cleared
- **Pause / Resume** — pauses the search at the next iteration boundary, or resumes it
- **Steering history expander** — shows all directives injected so far (iteration, mode, text)

**Semantics:**
- Multiple augment directives accumulate; a replace directive clears all previous ones.
- Injecting any directive while paused automatically resumes the search.
- The steering history is included in the PDF report.

### PDF Reports
- Professional multi-page PDF with title page, executive summary, convergence charts, results comparison tables, full iteration log, steering directive history, and multi-objective tracking section (when applicable)
- Uses DejaVu Sans font for diacritics support
- Chart images exported via Plotly/kaleido
- **Sweep result reporting — solver certification semantics**: the "Status" column in sweep result tables (both in the UI and in the PDF) is derived exclusively from the solver's `convergence_status` field. A candidate is shown as "Did not converge" for any non-CONVERGED status, and "Constraint violation" only when the solver explicitly converged to an infeasible operating point (PFLOW-style post-solve check). Uncertified last-iterate metrics (V_min, V_max, max line loading, violations) are shown in a separately labelled group ("Last iterate — uncertified") and must not be read as the certified cause of infeasibility
- **Cost-minimization sweep ranking**: when the search goal is cost minimization, the sweep overview shows a ranked table of the top-K cheapest feasible buses (default K=10) with Δ-from-best cost. When the gap between the first- and second-ranked candidate is below `report.near_optimal_abs_tol` (default $5/h), a caveat is displayed noting that the candidates are effectively equivalent within solver tolerance
- **Boundary (hosting-capacity) sweep table**: when a sweep is run in boundary mode (`"mode": "boundary"`), the overview and PDF show a **hosting-capacity table** — per bus: maximum feasible MW, binding constraint, boundary-point Vmin/Vmax, max line loading, and probe count — sorted by capacity (highest first). A footnote states that the reported boundary is the OPFLOW convergence boundary and that non-convergence is treated as the infeasible signal that caps the bisection. The header metrics show the highest hosting capacity and which bus achieves it
- **Adaptive sweep columns (C2/C3)**: the feasible-bus table adds columns to match the sweep type. A **dispatchable** generator-siting sweep (`entity_dispatchable: true`) adds a "Dispatched Pg (MW)" column and a mode caption; the cost ranking still applies. A **`max_delta_v`** metric sweep replaces the cost column with "Max ΔV (pu)" and ranks buses by largest voltage step. A **`reactive_adequacy`** predicate sweep relabels the table "Reactive-adequate buses" (a headroom test at forced P=Pmax, Q=Qmax) with the limiting quantity shown in the infeasible table's reason, and adds a **"Dispatched Q (MVAr)"** column auditing the Q-forcing (equals the Qmax target at every adequate bus)
- **Certified metric gating & corrected summary (C.2/C.3 fixes)**: custom metrics are shown only for converged candidates (a non-converged bus never displays a trusted metric value), and the "lowest-cost feasible" summary line descends into sweep results, so the reported optimum is the best sweep candidate (bus + cost), not the base case

### Session History
- Completed sessions are tracked in the sidebar for reference during a browser session

### Session Save/Resume

The sidebar includes a dedicated save/resume section:

- **Save Session** — saves the current or completed search state to a timestamped directory under `workdir/`. The saved state includes the full journal, objective registry, steering history, and the current modified network
- **Resume from** — dropdown listing previously saved sessions. Select one and click "Resume Search" to continue from where it left off. The LLM backend and configuration settings are taken from the current sidebar values, so you can resume with a different model or temperature

Saved sessions are stored as a directory containing `session.json` (metadata, journal, objectives) and optionally `current_network.m` (the MATPOWER network state at the save point).

## Configuration

The GUI widget values override defaults from `configs/default_config.yaml`. The override mechanism uses dot-notation keys (e.g., `llm.backend`, `search.max_iterations`) passed to `agentigrid.config.load_config(cli_overrides=...)`.

Key configuration paths:
- **Base config**: `configs/default_config.yaml`
- **Data files**: `data/*.m` (MATPOWER format)
- **Applications**: `applications/` (ExaGO binaries)
- **Working directory**: `workdir/` (created at runtime)

Key configuration fields added in prompt-#14 reporting fixes:

| Field | Default | Description |
|---|---|---|
| `search.sweep_max_workers` | `0` | OPFLOW sweep concurrency; 0 = auto (`min(cpu_count, 16)`), 1 = sequential |
| `search.sweep_llm_top_n` | `25` | Top-N rows in the token-bounded LLM-facing sweep view |
| `search.sweep_full_table_threshold` | `250` | Candidate count above which the LLM view switches to summary; 0 = always summarize |
| `search.boundary_initial_mw` | `50.0` | Boundary sweep: initial probe / exponential-bracketing start (MW) |
| `search.boundary_max_mw` | `2000.0` | Boundary sweep: bracketing cap; if still feasible, report ≥ cap (MW) |
| `search.boundary_tol_mw` | `1.0` | Boundary sweep: bisection stop gap (MW) |
| `search.boundary_max_probes` | `24` | Boundary sweep: max OPFLOW solves per candidate |
| `search.boundary_gen_q_frac` | `0.4` | Boundary sweep: generator reactive band as a fraction of ΔP |
| `search.boundary_power_factor_default` | `system_average` | Boundary sweep load PF: `system_average` \| `0..1` \| `unity` |
| `search.added_gen_cost_strategy` | `median_existing` | Cost curve for a dispatchable added unit: `median_existing` (mid-merit) or `explicit` |
| `search.added_gen_dispatchable_default` | `false` | Default mode for add_generator_at_bus sweeps (false = fixed injection) |
| `search.switched_load_mw` | `100.0` | MW load block switched in at a candidate bus for the `max_delta_v` metric |
| `report.cost_min_top_k` | `10` | Number of cheapest feasible buses shown in cost-min sweep ranking |
| `report.near_optimal_abs_tol` | `5.0` | $/h gap below which top candidates are flagged as equivalently optimal |

## Troubleshooting

### "No .m files found in data/ directory"
Ensure MATPOWER `.m` files are present in the `data/` directory at the project root.

### "ANTHROPIC_API_KEY not set" / "OPENAI_API_KEY not set"
Export the relevant API key before launching:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```
Or configure it in your `env_setup.sh` script.

### "DejaVu Sans not found" (PDF generation warning)
The PDF generator uses DejaVu Sans for diacritics support. Install the font package:
- **openSUSE**: `sudo zypper install dejavu-sans-fonts`
- **Debian/Ubuntu**: `sudo apt install fonts-dejavu-core`
- **Fedora**: `sudo dnf install dejavu-sans-fonts`

The generator falls back to Helvetica if DejaVu Sans is not found.

### "Failed to export chart image" (PDF charts missing)
Install kaleido for Plotly image export:
```bash
pip install kaleido
```

### Config paths not resolving / import errors
Ensure you run the launcher from the **project root**, not from inside `launcher/`. Use `./launcher/run.sh` which handles this automatically.

### Streamlit port conflict
If port 8501 is in use, specify an alternative:
```bash
streamlit run launcher/app.py --server.port 8502
```
