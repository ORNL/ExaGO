# ExaGO Launcher

A Streamlit-based GUI for configuring, running, and visualizing ExaGO power grid simulation applications.

## What is ExaGO Launcher?

ExaGO Launcher provides a web-based interface for the ExaGO suite of power grid optimization tools. Instead of manually constructing command-line invocations, users can select parameters through a form-based UI, preview the resulting command, run simulations, and inspect output — all from a browser.

The launcher supports six ExaGO applications:

| Application | Description |
|-------------|-------------|
| **OPFLOW**  | AC Optimal Power Flow |
| **PFLOW**   | AC Power Flow |
| **DCOPFLOW**| DC Optimal Power Flow |
| **SCOPFLOW**| Security-Constrained OPF |
| **SOPFLOW** | Stochastic Security-Constrained OPF |
| **TCOPFLOW**| Multi-Period OPF |

## Quick Start

### Prerequisites

- Python 3.10+
- ExaGO compiled and installed (binaries available in `build/applications/`)
    - Building ExaGO will generate `launcher/config.yaml` file.
- Required Python packages: `streamlit`, `pyyaml`

### Install and Run

```bash
cd launcher/
pip install -r requirements.txt
./run.sh
```

Or manually:

```bash
streamlit run app.py -- --config config.yaml
```

The application opens in your browser at `http://localhost:8501`.

## Configuration

All paths are loaded from a single **`config.yaml`** file. No paths are
hardcoded in the application code, making it portable across machines.
Configuration file `launcher/config.yaml` is generated during ExaGO build.
Edit it only if you know what you are doing.

### Config File Location

The app resolves the config file path in this order:

1. **CLI argument**: `streamlit run app.py -- --config /path/to/config.yaml`
2. **Environment variable**: `EXAGO_LAUNCHER_CONFIG=/path/to/config.yaml`
3. **Default**: `config.yaml` in the same directory as `app.py`

### Config File Structure

```yaml
# Root ExaGO directory — all other paths can reference this via ${exago_dir}
exago_dir: /path/to/ExaGO

# Path to compiled application binaries (opflow, pflow, scopflow, etc.)
applications_dir: ${exago_dir}/build/applications

# Path to data files (network .m files, contingency .cont files, scenarios, etc.)
datafiles_dir: ${exago_dir}/datafiles

# Path to the visualization frontend directory
viz_dir: ${exago_dir}/viz

# Path to GIC data files (.gic files and JSON output used by the viz engine)
gic_data_dir: ${exago_dir}/viz/data

# Directory where simulation output files are saved
output_dir: ${exago_dir}/../LLMSim/launcher/output

# Environment variables injected into the subprocess when running simulations
environment:
  ld_library_path:
    - /path/to/extra/libs         # Appended to LD_LIBRARY_PATH
  path:
    - /path/to/petsc/bin          # Prepended to PATH
  pythonpath:
    - /path/to/exago/python/api   # Prepended to PYTHONPATH
```

### Variable Substitution

The string `${exago_dir}` in any value is replaced with the value of the `exago_dir` key. This means you typically only need to change one line (`exago_dir`) when moving to a new machine.

### Directory Validation

On startup, the sidebar shows the status of every configured directory with green (exists) or yellow warning (missing) indicators. Click **Reload Config** in the sidebar to re-read `config.yaml` after editing it.

## Application Tabs

### Tab 1: Run Simulation

The main simulation interface, organized in steps:

**Step 1 — Select Application**: Choose from the six ExaGO applications via a dropdown.

**Step 2 — Common Parameters**:
- **Network file** (`-netfile`): Upload a `.m` (MATPOWER format) network file using the file uploader.
- **MPI processes**: Number of parallel processes (uses `mpiexec -n` when > 1).
- **Print output** (`-print_output`): Display results to stdout (default: on).
- **Save output** (`-save_output`): Write results to disk (default: off).
- **Output format** (`-opflow_output_format`): MATPOWER or JSON (shown when save is enabled).

**Step 3 — Application-Specific Parameters**: The UI dynamically shows relevant options based on the selected application:

- **OPFLOW**: Model, solver, initialization method, line flow constraints, load loss/power imbalance penalties, generator bus voltage mode, objective function, tolerance, optional GIC file.
- **SCOPFLOW**: Contingency file, number of contingencies, solver, mode (preventive/corrective), optional multi-period settings with load and wind profiles.
- **SOPFLOW**: Scenario file, contingency file, number of scenarios/contingencies, solver, mode, multi-contingency toggle.
- **TCOPFLOW**: Active/reactive power load profiles, optional wind generation profile, time step, duration.
- **PFLOW / DCOPFLOW**: No additional parameters beyond the common ones.

Advanced or optional parameters are collapsed under expanders to keep the interface clean.

**Step 4 — Command Preview**: The fully constructed command line is displayed as a code block so you can verify it before running.

**Step 5 — Run & Results**: Click the Run button to execute. Output streams in real time. After completion, the UI shows:
- Execution time
- Convergence status (color-coded: green for converged, red for failed)
- Full simulation output in an expandable panel
- Download button for output files (if save was enabled)

### Tab 2: Visualization

Generates JSON output from OPFLOW and feeds it to the ExaGO visualization frontend.

Two modes:

- **Run OPFLOW & Visualize**: Upload a network file (and optionally a GIC file), run OPFLOW with JSON output, copy the result to `viz/data/`, then launch the viz server.
- **Use Existing JSON**: Upload a previously generated JSON file and configure the viz engine directly.

The visualization server runs on `http://localhost:5173` (Vite dev server). Use the Launch/Stop buttons to control it.

### Tab 3: Python API

Generates equivalent Python code using the `exago` Python module that mirrors the current configuration from Tab 1. This helps users transition from the GUI to programmatic usage.

- View the generated code
- Download it as a `.py` file
- Run it directly from the browser

## Output Files

### Output Directory

All simulation output is written to the directory configured as `output_dir` in `config.yaml`. By default:

```
launcher/output/
```

### Output Structure

The output structure depends on the application and output format:

**Single-file output** (OPFLOW, PFLOW, DCOPFLOW):

```
output/
├── opflowout.m          # MATPOWER format output
├── opflowout.json       # JSON format output (if JSON selected)
└── staging/             # Uploaded input files (managed by the launcher)
```

**Directory output** (SCOPFLOW, SOPFLOW, TCOPFLOW):

These applications produce a directory containing multiple output files, one per contingency, scenario, or time period:

```
output/
├── scopflowout/         # SCOPFLOW output directory
│   ├── cont_0.m         # Base case solution
│   ├── cont_1.m         # Contingency 1 solution
│   ├── cont_2.m         # Contingency 2 solution
│   └── ...
├── sopflowout/          # SOPFLOW output directory
│   ├── scen_0/          # Scenario 0
│   │   ├── cont_0.m
│   │   └── ...
│   └── ...
├── tcopflowout/         # TCOPFLOW output directory
│   ├── step_0.m         # Time step 0
│   ├── step_1.m         # Time step 1
│   └── ...
└── staging/
```

### Downloading Output

The launcher's download button handles both cases automatically:

- **Single files**: Downloaded directly.
- **Output directories**: Zipped into a `.zip` archive and offered for download.

### Staging Directory

Uploaded input files are temporarily saved to `output/staging/`. This directory is managed by the launcher and is excluded from the output file scan. You can safely delete its contents between sessions.

## File Types

| Extension | Description | Used By |
|-----------|-------------|---------|
| `.m`      | MATPOWER network case files | All applications (`-netfile`) |
| `.cont`   | Contingency definition files | SCOPFLOW, SOPFLOW (`-ctgcfile`) |
| `.csv`    | Scenario files, load profiles | SOPFLOW (`-windgen`), TCOPFLOW/SCOPFLOW (load profiles) |
| `.gic`    | Geomagnetically Induced Current data | OPFLOW (`-gicfile`) |
| `.json`   | JSON output for visualization | Visualization tab |

## Troubleshooting

**"Binary not found"**: Verify that `applications_dir` in `config.yaml` points to the directory containing the compiled ExaGO binaries (`opflow`, `pflow`, etc.). Check the sidebar directory status indicators.

**Solver failures**: Some solvers (HIOP, HIOPSPARSE) require ExaGO to be compiled with HiOp support. Fall back to IPOPT if unsure.

**"npm not found"** (Visualization tab): The viz server requires Node.js. Install Node.js and run `npm install` in the `viz/` directory first.

**Missing libraries at runtime**: Add any required shared library paths to the `environment.ld_library_path` list in `config.yaml`.
