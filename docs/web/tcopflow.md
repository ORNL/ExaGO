# Multi-period optimal power flow (TCOPFLOW)

TCOPFLOW solves a multi-period alternating current optimal power flow ([ACOPF](./opflow.md)) with the objective of minimizing the total dispatch cost  over the given time horizon while satisfying steady-state ACOPF constraints at each time step and ramping constraints across all time steps. 


## Formulation

Over a horizon of $N_t$ time steps, TCOPFLOW minimizes the total production cost subject to network physics, engineering limits, and inter-temporal ramping constraints:

```math
\begin{aligned}
\text{min}&~\sum_{t \in N_t} f(x_t)& \\
&\text{s.t.}& \\
&~g(x_t) = 0~~~t \in \{0,\ldots, N_t\}& \\
&~h(x_t) \le 0~~t \in \{0,\ldots, N_t\}& \\
&x^- \le x_t \le x^+~~t\in \{0,\ldots, N_t\}& \\
-\Delta{x}_t & \le x_t - x_{t-\Delta{t}} \le \Delta{x}_t~~t \in \{1,\ldots, N_t\}&
\end{aligned}
```

Here $x_t$ contains the ACOPF state at time $t$, $f$ is the total generation cost, $g$ enforces AC power balance, $h$ covers inequality limits (thermal flows, generator capability curves, etc.), while the final set of inequalities enforces generator real-power ramping limits between consecutive steps. The time horizon is defined by the duration (hours) and the user-specified time step size (minutes). When the ramping constraints are disabled the problem decouples into $N_t$ independent ACOPFs.

## Build and Solver Dependencies

- TCOPFLOW depends on the PETSc-based ExaGO infrastructure and currently supports only Ipopt optimization solver. Ensure ExaGO is configured with Ipopt before building the applications target so that `tcopflow` is generated alongside the other solvers.
- The executable is produced by the CMake target defined in [`applications/tcopflow_main.cpp`](../../applications/tcopflow_main.cpp), and will appear in `bin/tcopflow` after running `cmake --build` on the main project.

## Key Include and Source Files

| Component | Purpose |
| --- | --- |
| [`include/tcopflow.h`](../../include/tcopflow.h) | Public C API for creating TCOPFLOW objects, setting data, solving, and retrieving solutions. |
| [`src/tcopflow/interface/tcopflow.cpp`](../../src/tcopflow/interface/tcopflow.cpp) | High-level application logic: option parsing, PS/OPFLOW object creation, model/solver registration, and coupling setup. |
| [`src/tcopflow/interface/tcopflowreadprofiles.cpp`](../../src/tcopflow/interface/tcopflowreadprofiles.cpp) | CSV profile ingestion for time-varying inputs (e.g. load and wind data). |
| [`src/tcopflow/interface/tcopflowoutput.cpp`](../../src/tcopflow/interface/tcopflowoutput.cpp) | Routines that format per-time-step solutions and write MATPOWER files. |
| [`src/tcopflow/model/genramp/genramp.cpp`](../../src/tcopflow/model/genramp/genramp.cpp) | Default temporal coupling model that enforces generator ramping constraints. |
| [`src/tcopflow/solver/ipopt/tcopflow_ipopt.cpp`](../../src/tcopflow/solver/ipopt/tcopflow_ipopt.cpp) | Ipopt bindings (objective, gradient, Jacobian, and Hessian callbacks). |

Use these entry points when extending the formulation (e.g., adding new coupling models or solvers) or when embedding TCOPFLOW via the public API.

## Required Input Files

1. **Network case** (`-netfile`) — MATPOWER `.m` or `.mat` describing buses, branches, and generators. Example: `datafiles/case9/case9mod.m`.
2. **Active load profile** (`-tcopflow_ploadprofile`) — CSV with one column per bus and one row per time step representing power demand in MW. Example: [`datafiles/case9/load_P.csv`](../../datafiles/case9/load_P.csv).
3. **Reactive load profile** (`-tcopflow_qloadprofile`) — CSV analogous to the active profile but for reactive power in MVAr. Example: [`datafiles/case9/load_Q.csv`](../../datafiles/case9/load_Q.csv).
4. **Wind generation profile** (`-tcopflow_windgenprofile`, optional) — CSV with time-series injections per wind unit. If omitted, wind injection stays at the base-case level.

If any profile is omitted, TCOPFLOW assumes a flat profile (the base-case value is replicated across all time steps). Time-series files must provide at least `duration / (dT/60)` rows; extra rows are ignored.

## Running TCOPFLOW

1. Build ExaGO with Ipopt optimization solver enabled.
2. Prepare or copy the MATPOWER network file and CSV profile files into an accessible directory (the project ships reusable examples under [`datafiles/`](../../datafiles).
3. Launch the executable:

```bash
mpiexec -n 1 ./bin/tcopflow \
  -netfile $EXAGO_DIR/datafiles/case9/case9mod.m \
  -tcopflow_ploadprofile $EXAGO_DIR/datafiles/case9/load_P.csv \
  -tcopflow_qloadprofile $EXAGO_DIR/datafiles/case9/load_Q.csv \
  -tcopflow_dT 5 -tcopflow_duration 0.5 \
  -print_output -save_output
```

Currently Ipopt runs on a single MPI rank, so `-n 1` is sufficient to run the example (running with e.g. `-n 2` will just run the same computation twice). The command will allow for MPI parallel optimization solvers to be used with ExaGO.

### Output Artifacts

- When `-save_output` is set, MATPOWER snapshots for each time step are written under `tcopflowout/`.
- `-print_output` streams solver statistics, nodal injections, line flows, and generator set-points to stdout (mirrors the sample log in [`docs/manual/tcopflow.tex`](../manual/tcopflow.tex)).

## Runtime Options

All options can be passed on the command line or via an options file (see [`options/tcopflowoptions`](../../options/tcopflowoptions)).

| Option | Description | Default / Notes |
| --- | --- | --- |
| `-netfile <path>` | MATPOWER network file. | Defaults to `datafiles/case9/case9mod.m` if unset in the sample driver. |
| `-tcopflow_ploadprofile <csv>` | Active power demand profile. | Flat profile if omitted. |
| `-tcopflow_qloadprofile <csv>` | Reactive power demand profile. | Flat profile if omitted. |
| `-tcopflow_windgenprofile <csv>` | Wind generation profile. | Flat profile (base dispatch) if omitted. |
| `-tcopflow_dT <minutes>` | Time-step length in minutes. | Sample options file uses `5.0`. |
| `-tcopflow_duration <hours>` | Simulation horizon in hours. | Sample options file uses `0.5` (30 minutes). |
| `-tcopflow_iscoupling <0/1>` | Enable/disable inter-temporal coupling (generator ramping). | `1` (enabled). |
| `-tcopflow_tolerance <value>` | Optimality tolerance passed to the solver. | Inherits IPOPT default if not set. |
| `-tcopflow_model <name>` | Temporal model. | `GENRAMP` (see [`src/tcopflow/model/genramp`](../../src/tcopflow/model/genramp)). |
| `-tcopflow_solver <name>` | Nonlinear solver backend. | `IPOPT` (see [`src/tcopflow/solver/ipopt`](../../src/tcopflow/solver/ipopt)). |
| `-print_output` | Dump solution summary to stdout. | Disabled by default. |
| `-save_output` | Write MATPOWER files for each step into `tcopflowout/`. | Disabled by default. |
| `-opflow_ignore_lineflow_constraints <0/1>` | Forwarded to each embedded OPFLOW instance to toggle thermal limits. | Defaults to `0`. |
| `-options_left no` | PETSc utility flag to suppress unused-option warnings (see sample options file). | Optional.

You can keep commonly used settings in `options/tcopflowoptions` and load them by adding `-options_file options/tcopflowoptions` to the command line.

## Usage Examples

### 1. Quick sanity check on the IEEE 9-bus case

```
mpiexec -n 1 ./bin/tcopflow \
  -netfile $EXAGO_DIR/datafiles/case9/case9mod.m \
  -tcopflow_ploadprofile $EXAGO_DIR/datafiles/case9/load_P.csv \
  -tcopflow_qloadprofile $EXAGO_DIR/datafiles/case9/load_Q.csv \
  -tcopflow_dT 15 -tcopflow_duration 1.0 \
  -print_output
```

This runs 5 time steps (1 hour / 15 minutes) and prints the solver summary so you can verify convergence before scripting larger studies.

### 2. Scenario with wind profile and saved outputs

```
mpiexec -n 1 ./bin/tcopflow \
  -netfile $HOME/cases/activesg2000.m \
  -tcopflow_ploadprofile $HOME/cases/load_P.csv \
  -tcopflow_qloadprofile $HOME/cases/load_Q.csv \
  -tcopflow_windgenprofile $HOME/cases/wind.csv \
  -tcopflow_dT 10 -tcopflow_duration 2.0 \
  -tcopflow_tolerance 1e-5 \
  -save_output -print_output
```

The command spans 13 time steps (2 hours / 10 minutes) and exports every snapshot to `tcopflowout/` for downstream analysis.

## Notes and Tips

- Profiles should be pre-aligned to the desired time step; TCOPFLOW does not resample CSV data.
- Use PETSc logging (`-log_summary`) alongside the provided stage markers to diagnose read vs. solve time.
- The TCOPFLOW API (see [`include/tcopflow.h`](../../include/tcopflow.h)) enables embedding the solver into custom drivers; reuse `TCOPFLOWCreate`, `TCOPFLOWSetLoadProfiles`, `TCOPFLOWSolve`, and `TCOPFLOWSaveSolutionAll` as shown in [`applications/tcopflow_main.cpp`](../../applications/tcopflow_main.cpp).
