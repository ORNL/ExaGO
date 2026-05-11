# AgentiGrid Launcher — Design Document

**Version:** 1.0
**Date:** 2026-03-27
**Authors:** Samim / Claude (collaborative design)

---

## 1. Overview and Goals

### 1.1 Purpose

The AgentiGrid Launcher is a Streamlit-based GUI application that provides an interactive frontend for the AgentiGrid search engine. It allows users to configure and execute LLM-driven power grid optimization searches, monitor iteration progress in real time, visualize results, and generate PDF reports — all without touching the command line.

### 1.2 Motivation

The ExaGO launcher demonstrated that a well-designed GUI dramatically improves accessibility and adoption. Terminal output from AgentiGrid is detailed but opaque to non-specialists. For project support applications, the ability to show the search process visually — watching the LLM iterate toward a solution — is far more compelling than console logs.

### 1.3 Design Principles

1. **Self-contained in `launcher/`**: All launcher code lives in `agentigrid/launcher/`. It imports from `agentigrid.*` but never modifies files outside its own folder. Existing project structure (`agentigrid/`, `configs/`, `data/`, `prompts/`, `tests/`, etc.) remains untouched.

2. **Thin presentation layer**: The launcher is a visualization and interaction layer over the existing AgentiGrid engine. Business logic stays in `agentigrid.engine`. The launcher consumes the engine's data structures (`SearchSession`, `SearchJournal`, `JournalEntry`, `OPFLOWResult`) directly.

3. **ExaGO-integration ready**: Design decisions should survive eventual integration into the ExaGO ecosystem. Invest in data contracts and analysis logic (reusable), keep Streamlit-specific code straightforward (replaceable).

4. **Self-contained visualization**: Charts and plots use Plotly within Streamlit. No dependency on the ChatGrid/Node.js visualization server.

---

## 2. Architecture

### 2.1 High-Level Component Diagram

```
┌──────────────────────────────────────────────────────────┐
│                   Streamlit App (launcher/)               │
│                                                          │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │ Config     │  │ Live Search  │  │ Results &        │ │
│  │ Panel      │  │ Monitor      │  │ Summary View     │ │
│  └─────┬──────┘  └──────┬───────┘  └────────┬─────────┘ │
│        │                │                    │           │
│  ┌─────▼────────────────▼────────────────────▼─────────┐ │
│  │              SessionManager                         │ │
│  │  (bridges Streamlit ↔ AgentiGrid engine)               │ │
│  └─────────────────────┬───────────────────────────────┘ │
│                        │                                 │
│  ┌─────────────────────▼───────────────────────────────┐ │
│  │              ReportGenerator                        │ │
│  │  (PDF export with Plotly charts + LLM summary)      │ │
│  └─────────────────────────────────────────────────────┘ │
└────────────────────────┬─────────────────────────────────┘
                         │ imports
┌────────────────────────▼─────────────────────────────────┐
│                  AgentiGrid Core (agentigrid/)                 │
│                                                          │
│  AgentLoopController  ──►  SearchSession                 │
│  SimulationExecutor   ──►  SearchJournal / JournalEntry  │
│  LLM Backends         ──►  OPFLOWResult                  │
│  Parsers              ──►  MATNetwork                    │
└──────────────────────────────────────────────────────────┘
```

### 2.2 Integration Strategy: Callback-Based Progress Reporting

The current `AgentLoopController.run()` executes the entire search loop synchronously and reports progress via `self._print()`. For the GUI we need real-time iteration-by-iteration updates.

**Approach**: Add an optional **callback mechanism** to `AgentLoopController`. The callback is invoked after each iteration with the latest `JournalEntry` and current session state. The Streamlit app registers a callback that updates `st.session_state`, which triggers UI re-renders.

**Required change to `AgentLoopController`** (minimal, non-breaking):

```python
# In AgentLoopController.__init__:
def __init__(self, config: AppConfig, quiet: bool = False,
             on_iteration: Callable[[int, JournalEntry, str], None] | None = None,
             on_phase: Callable[[int, str], None] | None = None) -> None:
    ...
    self._on_iteration = on_iteration  # Called after each iteration completes
    self._on_phase = on_phase          # Called on phase transitions within iteration
```

The `on_iteration(iteration, entry, phase)` callback receives:
- `iteration`: iteration number (0 = base case)
- `entry`: the `JournalEntry` just recorded
- `phase`: status string ("completed", "failed", "parse_error")

The `on_phase(iteration, phase_name)` callback receives:
- `iteration`: current iteration number
- `phase_name`: one of "llm_request", "applying_commands", "running_simulation", "parsing_results"

This is the **only change** to existing `agentigrid/` code. The callback parameters are optional, defaulting to `None`, so CLI usage is completely unaffected.

**Note**: `AgentLoopController` was extended beyond callbacks to also support interactive steering (steering queue, pause/resume via `threading.Event`, `inject_steering()`, `pause()`, `resume()`, `is_paused()`, `steering_history` property). The `SearchJournal` gained `add_analysis()`, `summary_stats(best_iteration_override, goal_type)`, `format_multi_objective_summary()`, and an `objective_registry` attribute (`ObjectiveRegistry` instance). `JournalEntry` gained a `tracked_metrics` field for per-iteration multi-objective metric values. The engine also gained `metric_extractor.py` (14 deterministic OPFLOW metric extractors) and `objective_parser.py` (LLM-based objective extraction from natural language). These additions are non-breaking — all new parameters default to `None` and CLI behaviour is unchanged.

### 2.3 Threading Model

Streamlit reruns the entire script on each interaction, which conflicts with a long-running search loop. Solution:

- The search runs in a **background thread** started from the Streamlit app.
- The `on_iteration` callback writes updates to a **thread-safe queue** (`queue.Queue`) or directly to `st.session_state` (which is thread-safe for writes).
- The Streamlit main loop polls for updates using `st.empty()` containers and `time.sleep()` with periodic reruns.
- A `st.session_state.search_running` flag controls UI state (disable config inputs during search, show stop button, etc.).

### 2.4 Session State Management

Streamlit's `st.session_state` holds all runtime state:

```python
# Key session state variables:
st.session_state.search_running: bool        # Is a search currently executing?
st.session_state.search_thread: Thread       # Background thread reference
st.session_state.search_session: SearchSession  # Completed session (after search ends)
st.session_state.iteration_log: list[dict]   # Live iteration updates from callback
st.session_state.current_phase: str          # Current phase within active iteration
st.session_state.current_iteration: int      # Current iteration number
st.session_state.stop_requested: bool        # User clicked "Stop Search"
st.session_state.completed_sessions: list    # History of completed sessions (current app run)
```

---

## 3. Data Flow

### 3.1 Search Lifecycle from GUI Perspective

```
User fills config ──► "Start Search" button clicked
        │
        ▼
SessionManager.start_search(config, goal)
        │
        ├── Builds AppConfig from GUI inputs
        ├── Creates AgentLoopController with callbacks
        ├── Launches controller.run() in background thread
        │
        ▼
    [Background Thread]
        │
        ├── Iteration 0: Base case
        │   └── on_iteration(0, entry, "completed") ──► UI shows base case card
        │
        ├── Iteration 1..N:
        │   ├── on_phase(i, "llm_request")  ──► UI shows "Waiting for LLM..."
        │   ├── on_phase(i, "applying_commands") ──► UI shows "Applying modifications..."
        │   ├── on_phase(i, "running_simulation") ──► UI shows "Running OPFLOW..."
        │   ├── on_phase(i, "parsing_results") ──► UI shows "Analyzing results..."
        │   └── on_iteration(i, entry, status) ──► UI adds iteration card, updates chart
        │
        └── Search ends (completed / max_iterations / error)
            └── on_iteration signals completion ──► UI transitions to Results view
        │
        ▼
Results & Summary View activates
        │
        ├── Convergence chart (from journal objective_trend)
        ├── Base case vs. best solution comparison
        ├── Detailed metrics tables
        └── "Generate PDF Report" button ──► ReportGenerator
```

### 3.2 Data Structures Used by the Launcher

The launcher works entirely with existing data structures from `agentigrid`:

| Data Structure | Source Module | Used For |
|---|---|---|
| `AppConfig` | `agentigrid.config` | Building configuration from GUI inputs |
| `SearchSession` | `agentigrid.engine.agent_loop` | Complete session record after search |
| `SearchJournal` | `agentigrid.engine.journal` | Iteration history, summary stats |
| `JournalEntry` | `agentigrid.engine.journal` | Individual iteration data for cards |
| `OPFLOWResult` | `agentigrid.parsers.opflow_results` | Detailed simulation results for charts |
| `SimulationResult` | `agentigrid.engine.executor` | Raw simulation output |
| `ObjectiveEntry` | `agentigrid.engine.journal` | Single tracked objective definition |
| `ObjectiveRegistry` | `agentigrid.engine.journal` | Registry of all tracked objectives with preference history |
| `LLMConfig`, etc. | `agentigrid.config` | Config section dataclasses |

No new data models are needed in the launcher. The `SessionManager` bridges between GUI inputs and these existing structures.

### 3.3 Iteration Update Record

For real-time display, the callback writes concise update dicts to `st.session_state.iteration_log`:

```python
{
    "iteration": 3,
    "timestamp": "2026-03-27T14:23:15",
    "description": "Reduce gen at bus 5 by 20%",
    "action": "modify",
    "status": "completed",           # completed | failed | parse_error
    "objective_value": 5734.21,      # None if simulation failed
    "feasible": True,
    "convergence_status": "CONVERGED",
    "voltage_range": (0.953, 1.047),
    "max_line_loading_pct": 78.3,
    "total_gen_mw": 312.5,
    "sim_elapsed": 1.23,
    "llm_reasoning": "The previous iteration showed...",
    "commands_count": 2,
    "commands_summary": ["set_gen_dispatch bus=5 Pg=160", "set_gen_voltage bus=5 Vg=1.03"],
    "mode": "accumulative",
    "prompt_tokens": 3200,
    "completion_tokens": 450,
    "tracked_metrics": {"generation_cost": 5734.21, "voltage_deviation": 0.047},
}
```

This is derived directly from `JournalEntry` fields in the `on_iteration` callback — no new data model, just a dict reformatting for display convenience.

---

## 4. UI Specification

The app uses a **single-page layout** with a sidebar for configuration and a main area that transitions between states: Configuration → Running → Results.

### 4.1 Sidebar — Configuration Panel

Always visible. Contains all search parameters.

**Section: Base Case**
- File selector (`st.selectbox`) listing `.m` files found in the `data/` directory
- Path display showing the resolved file path
- Small network summary after selection (number of buses, generators, branches — from `network_summary()`)

**Section: LLM Backend**
- Backend selector: `anthropic` | `openai` | `ollama` | `ollama-cloud`
- Model name text input (pre-filled with defaults per backend: `claude-sonnet-4-20250514` for Anthropic, `gpt-4o` for OpenAI, etc.)
- Temperature slider (0.0 – 1.0, default 0.3)

**Section: Search Parameters**
- Application selector: `opflow` (initially only, grayed-out placeholders for others)
- Mode selector: `accumulative` | `fresh`
- Max iterations slider (1–50, default 20)

**Section: Search Goal**
- Large text area (`st.text_area`) for the natural-language goal
- Optional "Example Goals" dropdown that populates the text area with preset prompts:
  - "Minimize total generation cost while maintaining all bus voltages within 0.95–1.05 p.u."
  - "Find the maximum load the network can serve while keeping all line loadings below 90%"
  - "Reduce generation cost by at least 10% compared to the base case"
  - "Identify and resolve voltage violations (buses outside 0.95–1.05 p.u. range)"

**Section: Actions**
- **"Start Search"** button (disabled during active search)
- **"Stop Search"** button (visible only during active search)

**Section: Session History** (below actions)
- List of completed sessions in current app run (clickable to review past results)

### 4.2 Main Area — States

#### State A: Welcome / Pre-Search

Shown before any search has been started.

- AgentiGrid logo/title and brief description
- Quick-start instructions: "Select a base case, choose your LLM backend, write a search goal, and click Start Search."
- If base case is selected, show the network summary card (bus count, gen count, branch count, total load, total generation capacity)

#### State B: Live Search Monitor

Shown while a search is running. Two-column layout:

**Left column (wider, ~65%)** — Iteration Timeline:
- Each iteration displayed as an expandable card (`st.expander`)
- **Collapsed view** shows:
  - Iteration number and status icon (✓ green for feasible, ✗ red for failed, ⚠ yellow for infeasible but converged)
  - One-line description from LLM
  - Objective value (or "FAILED")
  - Simulation time
- **Expanded view** adds:
  - LLM reasoning text
  - Commands applied (formatted list)
  - Key metrics: voltage range, max line loading, gen/load totals
  - Mode used (fresh/accumulative)
- Current iteration shows a live status indicator:
  - Spinner with phase text: "Sending prompt to Claude (claude-sonnet-4-20250514)..." → "Applying 3 modifications..." → "Running OPFLOW simulation..." → "Parsing results..."

**Right column (~35%)** — Live Charts:
- **Convergence chart** (Plotly line chart): Objective value vs. iteration number
  - Points colored by feasibility (green = feasible, red = infeasible/failed)
  - Updates after each iteration
- **Voltage range chart** (Plotly): Min and max voltage across iterations (area between them shaded)
  - Horizontal reference lines at typical limits (0.95 and 1.05 p.u.)
- **Progress indicator**: "Iteration 5 of 20" with a progress bar
- **Token usage**: Cumulative prompt + completion tokens
- **Elapsed time**: Total search duration so far

**Bottom bar** — Status ribbon:
- Current status text: "Iteration 5: Running OPFLOW simulation..."
- Stop Search button (secondary position)

#### State C: Results & Summary View

Shown after search completes. Tabbed layout with three tabs:

**Tab 1: Overview**

- **Search Summary Card** at top:
  - Goal, application, backend/model, total iterations, duration, termination reason
  - Token usage summary
  - Best objective value and which iteration found it

- **Base Case vs. Best Solution Comparison** (side-by-side metrics):
  | Metric | Base Case | Best Solution | Change |
  |---|---|---|---|
  | Objective (cost) | $6,291.23 | $5,734.21 | -8.9% |
  | Total Generation | 315.2 MW | 312.5 MW | -0.9% |
  | Voltage Min | 0.942 p.u. | 0.953 p.u. | +0.011 |
  | Voltage Max | 1.058 p.u. | 1.047 p.u. | -0.011 |
  | Max Line Loading | 87.3% | 78.3% | -9.0 pp |
  | Violations | 2 | 0 | -2 |

- **Convergence Chart** (full-width Plotly chart, same as live but final):
  - Objective value trend with annotations at key points (best solution, any failures)
  - Interactive (hover shows iteration details)

**Tab 2: Detailed Results**

- **Voltage Profile Chart** (Plotly bar or scatter chart):
  - Voltage magnitude at each bus, base case vs. best solution overlaid
  - Horizontal bands showing voltage limits
  - Sorted by bus number or by voltage deviation

- **Generator Dispatch Comparison** (Plotly grouped bar chart):
  - Active power output per generator, base case vs. best solution
  - Shows Pmin/Pmax bounds as error bars or shading

- **Line Loading Chart** (Plotly horizontal bar):
  - Top 10–15 most loaded lines
  - Showing loading percentage, base case vs. best solution

- **Iteration History Table** (`st.dataframe`):
  - Full journal data in a sortable, filterable table
  - Columns: Iteration, Description, Cost, Feasible, V_min, V_max, Max Loading, Sim Time

**Tab 3: LLM Analysis & Report**

- **LLM-Generated Summary Analysis**:
  - After search completes, make one final LLM call asking for a structured analytical summary of the search
  - Displayed in a formatted text block
  - Covers: strategy assessment, key findings, convergence behavior, recommendations
  - The LLM sees the complete journal history and final results

- **Search Narrative**:
  - Chronological summary of what happened: "Iteration 1: The LLM began by... Iteration 2: Building on this..."
  - Auto-generated from journal entries' descriptions and reasoning

- **"Generate PDF Report" button**:
  - Produces a downloadable PDF with all the above content
  - `st.download_button` for immediate download

---

## 5. PDF Report Specification

### 5.1 Report Structure

The PDF report is a self-contained document suitable for inclusion in project proposals or sharing with collaborators.

**Page 1: Title Page**
- Title: "AgentiGrid Search Report"
- Subtitle: The search goal (truncated if very long)
- Date and time of search
- Application and backend/model used
- Generated by AgentiGrid v0.1.0

**Page 2: Executive Summary**
- Search goal (full text)
- Key results: best objective, improvement vs. base case, iteration count, duration
- LLM-generated summary analysis (from Tab 3)

**Page 3: Convergence Analysis**
- Convergence chart (objective value over iterations) — exported from Plotly as static image
- Voltage range trend chart
- Brief narrative of convergence behavior

**Page 4: Results Comparison**
- Base case vs. best solution comparison table
- Generator dispatch comparison chart
- Voltage profile chart

**Page 5+: Detailed Iteration Log**
- Complete iteration history table
- For each iteration: description, commands, key metrics, LLM reasoning (condensed)

### 5.2 Technical Implementation

- **Library**: ReportLab (Platypus for layout)
- **Font**: DejaVu Sans (supports diacritics — important for names like Pejić, etc.)
- **Charts**: Plotly figures exported as PNG images via `plotly.io.write_image()` (requires kaleido package), then embedded in ReportLab
- **Page size**: A4 (more common in European/academic context)
- **Color scheme**: Consistent with Streamlit app charts

---

## 6. Project Structure

```
agentigrid/
├── launcher/                        # ◄── All new code goes here
│   ├── ARCHITECTURE.md              # This design document
│   ├── app.py                       # Main Streamlit application entry point
│   ├── requirements.txt             # Launcher-specific dependencies
│   ├── run.sh                       # Convenience script to start the launcher
│   ├── README.md                    # Launcher documentation
│   │
│   ├── session_manager.py           # Bridges Streamlit ↔ AgentLoopController
│   ├── report_generator.py          # PDF report generation (ReportLab)
│   ├── charts.py                    # Plotly chart builders (reused in app + PDF)
│   ├── config_builder.py            # Builds AppConfig from GUI widget values
│   │
│   └── assets/                      # Static assets
│       ├── logo.png                 # AgentiGrid logo (optional)
│       └── example_goals.yaml       # Preset goal prompts
│
├── agentigrid/                         # Existing — NOT MODIFIED (except callback hooks)
│   ├── engine/
│   │   ├── agent_loop.py            # ◄── Minor addition: callback parameters
│   │   ├── journal.py               # Used as-is
│   │   ├── executor.py              # Used as-is
│   │   └── ...
│   ├── backends/                    # Used as-is
│   ├── parsers/                     # Used as-is
│   ├── config.py                    # Used as-is
│   └── ...
│
├── configs/                         # Existing — NOT MODIFIED
├── data/                            # Existing — NOT MODIFIED (read from launcher)
├── prompts/                         # Existing — NOT MODIFIED
├── tests/                           # Existing — NOT MODIFIED
├── workdir/                         # Existing — used by executor during search
└── logs/                            # Existing — used by logging
```

### 6.1 File Responsibilities

| File | Responsibility |
|---|---|
| `app.py` | Streamlit page layout, widget rendering, session state management, main UI flow |
| `session_manager.py` | Creates `AppConfig`, instantiates `AgentLoopController` with callbacks, manages background thread, provides iteration data to the UI |
| `config_builder.py` | Translates GUI widget values into the dict/override format that `load_config()` expects; scans `data/` directory for available `.m` files |
| `charts.py` | Plotly figure builders: `convergence_chart()`, `voltage_profile_chart()`, `generator_dispatch_chart()`, `line_loading_chart()`, `voltage_range_trend_chart()`, `multi_objective_trend_chart()`. Used both for live display and for PDF image export |
| `report_generator.py` | Builds a ReportLab PDF document from a completed `SearchSession`, embedding chart images and formatted tables |
| `run.sh` | Shell script: `cd` to project root, then `streamlit run launcher/app.py` |
| `example_goals.yaml` | YAML list of preset goal strings for the dropdown |

### 6.2 Dependencies (launcher/requirements.txt)

```
streamlit>=1.30.0
plotly>=5.18.0
kaleido>=0.2.1        # For Plotly static image export (used in PDF)
reportlab>=4.0
pyyaml>=6.0
```

Note: The launcher also depends on the `agentigrid` package (installed via `pip install -e .` from the project root). This is not listed in `requirements.txt` since it's the parent project.

---

## 7. Module Specifications

### 7.1 `session_manager.py`

```python
class SessionManager:
    """Bridges Streamlit UI with the AgentiGrid AgentLoopController."""

    def __init__(self):
        self._thread: Optional[Thread] = None
        self._controller: Optional[AgentLoopController] = None
        self._session: Optional[SearchSession] = None
        self._update_queue: queue.Queue = queue.Queue()
        self._goal_classification: Optional[dict] = None
        self._opflow_by_iteration: dict[int, OPFLOWResult] = {}

    def start_search(self, config_overrides: dict, goal: str,
                     config_path: str | Path | None = None) -> None:
        """Build config, create controller with callbacks, launch in thread."""
        ...

    def stop_search(self) -> None:
        """Request graceful stop of the running search."""
        ...

    def poll_updates(self) -> list[dict]:
        """Non-blocking drain of the update queue. Called by Streamlit main loop."""
        ...

    def is_running(self) -> bool:
        """Check if search thread is still alive."""
        ...

    def get_session(self) -> Optional[SearchSession]:
        """Get the completed SearchSession after search ends."""
        ...

    # --- Steering and pause/resume ---

    def inject_steering(self, directive: str, mode: str = "augment") -> None:
        """Forward a steering directive to the running controller."""
        ...

    def pause_search(self) -> None:
        """Pause the search at the next iteration boundary."""
        ...

    def resume_search(self) -> None:
        """Resume a paused search."""
        ...

    def is_paused(self) -> bool:
        """Check if the search is currently paused."""
        ...

    def get_steering_history(self) -> list[dict]:
        """Return the list of steering directives injected so far."""
        ...

    # --- Multi-objective tracking ---

    def get_objective_registry(self) -> list[dict] | None:
        """Get the objective registry data from the running or completed session."""
        ...

    def get_preference_history(self) -> list[dict] | None:
        """Get the preference evolution history."""
        ...

    # --- Analysis and goal classification ---

    def get_summary_analysis(self, session: SearchSession) -> str:
        """Make a final LLM call to generate analytical summary.

        Also requests a structured JSON goal classification block from the LLM:
          {"goal_type": "...", "best_iteration": N, "best_iteration_rationale": "..."}
        Parsed and stored in self._goal_classification.
        """
        ...

    def get_goal_classification(self) -> dict | None:
        """Get the LLM-determined goal classification, or None if not yet computed."""
        ...

    def get_best_opflow(self) -> OPFLOWResult | None:
        """Best feasible OPFLOW result — uses LLM classification when available."""
        ...

    def get_opflow_by_iteration(self, iteration: int) -> OPFLOWResult | None:
        """Get the OPFLOW result for a specific iteration."""
        ...
```

**Key design decisions:**

- Uses `queue.Queue` for thread-safe communication between background search thread and Streamlit's main loop.
- `inject_steering()`, `pause_search()`, `resume_search()` are thin forwarding wrappers over `AgentLoopController` methods — the controller owns the actual queue and threading primitives.
- `get_summary_analysis()` creates a separate one-shot LLM call with the complete journal as context, requesting both a structured narrative and a JSON goal-classification block. The classification drives which iteration is highlighted as "best" in the GUI and PDF report.
- `_on_pause_state_callback(paused)` puts a `{"type": "pause_state", "paused": bool}` message on the update queue so the GUI can update the Pause/Resume button label in the next rerun.
- `get_best_opflow()` returns the LLM-classified best iteration's `OPFLOWResult` when available, falling back to the lowest-cost feasible result. This ensures the comparison charts reflect the correct "best" for non-cost-minimization goals.

### 7.2 `charts.py`

All chart functions return `plotly.graph_objects.Figure` objects, usable both for `st.plotly_chart()` display and `fig.write_image()` PNG export.

```python
def convergence_chart(journal: SearchJournal,
                      highlight_best: bool = True) -> go.Figure:
    """Line chart of objective value across iterations."""
    ...

def voltage_range_chart(journal: SearchJournal) -> go.Figure:
    """Area chart showing voltage min/max range across iterations."""
    ...

def voltage_profile_chart(base_result: OPFLOWResult,
                          best_result: OPFLOWResult) -> go.Figure:
    """Bar/scatter chart comparing bus voltages between base and best."""
    ...

def generator_dispatch_chart(base_result: OPFLOWResult,
                             best_result: OPFLOWResult) -> go.Figure:
    """Grouped bar chart comparing generator outputs."""
    ...

def line_loading_chart(base_result: OPFLOWResult,
                       best_result: OPFLOWResult,
                       top_n: int = 15) -> go.Figure:
    """Horizontal bar chart of most loaded lines."""
    ...

def multi_objective_trend_chart(journal: SearchJournal,
                               height: int = 450) -> go.Figure | None:
    """Line chart showing how each tracked objective evolves across iterations.

    Each objective gets its own trace, color-coded with line style by priority
    (solid=primary, dashed=secondary, dotted=watch). Constraint thresholds
    are shown as horizontal reference lines. Returns None if fewer than 2
    iterations have tracked_metrics data."""
    ...
```

### 7.3 `report_generator.py`

```python
class ReportGenerator:
    """Generates PDF reports from completed search sessions."""

    def __init__(self, font_name: str = "DejaVuSans"):
        """Initialize with font configuration."""
        ...

    def generate(self, session: SearchSession,
                 summary_text: str,
                 output_path: Path,
                 base_result: OPFLOWResult | None = None,
                 best_result: OPFLOWResult | None = None) -> Path:
        """Generate a complete PDF report.

        Args:
            session: Completed search session with journal
            summary_text: LLM-generated summary analysis text
            output_path: Where to save the PDF
            base_result: Base case OPFLOW results (for comparison charts)
            best_result: Best solution OPFLOW results

        Returns:
            Path to the generated PDF file.
        """
        ...
```

### 7.4 `config_builder.py`

```python
def scan_data_files(data_dir: Path = Path("../data")) -> list[Path]:
    """Find all .m (MATPOWER) files in the data directory."""
    ...

def scan_config_files(configs_dir: Path = Path("../configs")) -> list[Path]:
    """Find all .yaml config files in the configs directory."""
    ...

def build_config_overrides(
    base_case: str,
    backend: str,
    model: str,
    temperature: float,
    application: str,
    default_mode: str,
    max_iterations: int,
    **kwargs,
) -> dict[str, Any]:
    """Build a CLI-style overrides dict from GUI widget values.

    Returns a dict suitable for passing to load_config(cli_overrides=...).
    """
    ...
```

---

## 8. Callback Integration — Required Change to `agent_loop.py`

This is the **only modification** to existing `agentigrid/` code. It adds optional callback parameters to `AgentLoopController` without changing any existing behavior.

### 8.1 Changes to `AgentLoopController.__init__`

```python
def __init__(self, config: AppConfig, quiet: bool = False,
             on_iteration: Callable[[int, JournalEntry, str, OPFLOWResult | None], None] | None = None,
             on_phase: Callable[[int, str], None] | None = None) -> None:
    ...
    self._on_iteration = on_iteration
    self._on_phase = on_phase
```

### 8.2 Callback Invocations in `_iteration()`

```python
def _iteration(self, iteration: int, goal: str) -> tuple[str, bool]:
    # Before LLM call:
    if self._on_phase:
        self._on_phase(iteration, "llm_request")

    # ... existing LLM call code ...

    # Before applying modifications (in _handle_modify):
    if self._on_phase:
        self._on_phase(iteration, "applying_commands")

    # Before simulation run (in _handle_modify):
    if self._on_phase:
        self._on_phase(iteration, "running_simulation")

    # After simulation parsing (in _handle_modify):
    if self._on_phase:
        self._on_phase(iteration, "parsing_results")
```

### 8.3 Callback Invocations After Each Iteration

In the main `run()` method, after `_iteration()` returns:

```python
# After _iteration returns and journal is updated:
if self._on_iteration:
    latest_entry = self._journal.latest
    if latest_entry:
        self._on_iteration(iteration, latest_entry, action_type, self._latest_opflow)
```

And after the base case (iteration 0):

```python
# After base case journal entry is added:
if self._on_iteration:
    latest_entry = self._journal.latest
    if latest_entry:
        self._on_iteration(0, latest_entry, "base_case", self._latest_opflow)
```

### 8.4 Graceful Stop Support

Add a `request_stop()` method and check in the loop:

```python
def request_stop(self) -> None:
    """Request graceful termination of the search loop."""
    self._stop_requested = True

# In __init__:
self._stop_requested = False

# In run(), inside the for loop, before each iteration:
if self._stop_requested:
    session.termination_reason = "user_stopped"
    self._print("\nSearch stopped by user.")
    break
```

---

## 9. Storing OPFLOWResult for Visualization

Currently, the `AgentLoopController` keeps `self._latest_opflow` (the most recent result) but does not store per-iteration `OPFLOWResult` objects in the journal — only the extracted summary metrics. For the detailed comparison charts (voltage profile, generator dispatch, line loading), we need the full `OPFLOWResult` for at least the base case and the best iteration.

### 9.1 Approach: Store Key Results in Session

Rather than modifying the journal (which is designed for compact textual representation), we'll have the `SessionManager` capture and store the full `OPFLOWResult` for:
1. **Iteration 0** (base case) — always
2. **The best feasible iteration** — updated whenever a new best is found

This is done via the callback mechanism: the `on_iteration` callback in `SessionManager` can also receive and store the `OPFLOWResult` reference.

**Required extension**: The `on_iteration` callback signature is:

```python
on_iteration: Callable[[int, JournalEntry, str, OPFLOWResult | None], None]
```

Where the fourth parameter is the parsed `OPFLOWResult` from that iteration (or `None` if simulation failed).

To pass this, the `AgentLoopController` will provide `self._latest_opflow` to the callback.

---

## 10. Implementation Plan

### Phase 1: Foundation (Claude Code tasks 1–3)

**Task 1: Callback integration into `agent_loop.py`**
- Add `on_iteration`, `on_phase` callback parameters
- Add `request_stop()` method and stop-check in loop
- Add `OPFLOWResult` passing to callback
- Verify CLI still works identically (all callbacks default to None)

**Task 2: Project scaffolding and `config_builder.py`**
- Create `launcher/` directory structure
- Create `requirements.txt`, `run.sh`, `README.md`
- Implement `config_builder.py` (scan data files, build config overrides)
- Create `assets/example_goals.yaml`

**Task 3: `session_manager.py`**
- Implement `SessionManager` class with thread management
- Queue-based communication
- Config building → controller creation → thread launch
- Stop search support
- Summary analysis LLM call

### Phase 2: Core UI (Claude Code tasks 4–6)

**Task 4: `app.py` — Configuration panel and basic layout**
- Sidebar with all config widgets
- Welcome state in main area
- Network summary display when base case selected
- Session state initialization

**Task 5: `app.py` — Live search monitor**
- Start/stop search integration with SessionManager
- Iteration timeline with expandable cards
- Phase status indicator
- Polling loop with periodic rerun

**Task 6: `charts.py` — All Plotly chart builders**
- Convergence chart
- Voltage range trend chart
- Voltage profile comparison chart
- Generator dispatch comparison chart
- Line loading comparison chart

### Phase 3: Results and Reporting (Claude Code tasks 7–9)

**Task 7: `app.py` — Results & Summary view**
- Three-tab results layout
- Overview tab with comparison table and convergence chart
- Detailed results tab with all comparison charts
- LLM analysis tab with summary text

**Task 8: `report_generator.py` — PDF report**
- ReportLab setup with DejaVu Sans
- Title page, executive summary, charts, iteration log
- Chart image export via Plotly/kaleido
- Download button integration in app.py

**Task 9: Polish and testing**
- Error handling for all failure modes (missing base case, LLM errors, simulation failures)
- Edge cases (0 feasible iterations, single iteration, max iterations reached)
- UI refinements based on testing
- README documentation

### Phase 4: Future Enhancements (not in initial scope)

- Replay mode (load completed journal and step through)
- Session persistence (save/load sessions across app restarts)
- ChatGrid integration for detailed network visualization
- Support for SCOPFLOW/TCOPFLOW/SOPFLOW/DCOPFLOW/PFLOW applications
- Multi-run comparison (compare results across different goals or configurations)

---

## 11. Key Technical Considerations

### 11.1 Streamlit Rerun Behavior

Streamlit reruns the full script on every interaction. All persistent state must live in `st.session_state`. The background thread and queue survive reruns because they're stored in session state. The polling mechanism uses `st.empty()` containers and `time.sleep(1)` with `st.rerun()` to check for updates during active search.

### 11.2 Working Directory

The launcher runs with `cwd = launcher/`. Relative paths in the default config (`./applications`, `./data`, `./workdir`) are relative to where the user invokes AgentiGrid from. The `config_builder` needs to resolve paths relative to the project root (parent of `launcher/`), not relative to `launcher/` itself. The `run.sh` script should `cd` to the project root before launching Streamlit, or the `config_builder` should use `Path(__file__).parent.parent` as the base for relative paths.

### 11.3 Preserving OPFLOWResult for Charts

The voltage profile, generator dispatch, and line loading charts require the full `OPFLOWResult` (with per-bus, per-generator, per-branch data). The `SessionManager` stores these for the base case and best iteration. If memory becomes a concern for large networks (thousands of buses), we could serialize to disk, but for the proof-of-concept scale this is not an issue.

### 11.4 LLM Summary Analysis Prompt

The final analytical summary is generated by a one-shot LLM call after the search completes. The prompt includes:
- The original goal
- The complete journal (formatted via `journal.format_detailed()`)
- The base case and best solution summary metrics
- An instruction to produce a structured analysis covering:
  - Overall assessment (was the goal achieved?)
  - Search strategy analysis (what approach did the LLM take?)
  - Convergence behavior (monotonic improvement, exploration, plateaus?)
  - Key modifications that had the most impact
  - Potential further improvements
  - Recommendations

### 11.5 Error Handling Strategy

- **LLM API errors**: Display in the iteration card, search continues (existing retry logic handles transient failures)
- **Simulation failures**: Display in the iteration card as failed, search continues (LLM adapts)
- **Configuration errors** (missing binary, invalid paths): Show error before search starts, don't launch
- **Thread crashes**: Detect via `thread.is_alive()`, display error message, allow restart
- **No feasible solutions found**: Results view still shows, comparison table shows "N/A" for best solution, convergence chart shows all points red

---

## 12. Summary

The AgentiGrid Launcher transforms the CLI-only search tool into an interactive, visually rich application suitable for demonstrations and project proposals. By keeping the launcher self-contained in `launcher/` and building on the existing engine's data structures, we minimize code duplication and maintain a clear path toward ExaGO integration.

The only modification to existing code is the addition of optional callback hooks in `AgentLoopController` — a clean, non-breaking change that enables real-time GUI updates without affecting CLI operation.

The implementation is structured as 9 Claude Code tasks across 3 phases, each building on the previous, with clear module boundaries and testable milestones.
