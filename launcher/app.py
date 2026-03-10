"""ExaGO Streamlit Launcher Application."""

import argparse
import io
import os
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import streamlit as st
import yaml

# ── Configuration Loading ────────────────────────────────────────────────────

def resolve_variables(config):
    """Resolve ${exago_dir} references in config values."""
    exago_dir = config.get("exago_dir", "")

    def _resolve(value):
        if isinstance(value, str):
            return value.replace("${exago_dir}", exago_dir)
        if isinstance(value, dict):
            return {k: _resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_resolve(v) for v in value]
        return value

    return {k: _resolve(v) for k, v in config.items()}


def load_config(config_path: str) -> dict:
    """Load and resolve the YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        st.error(f"Configuration file not found: {config_path}")
        st.stop()
    with open(path) as f:
        raw = yaml.safe_load(f)
    return resolve_variables(raw)


def get_config_path() -> str:
    """Determine config file path from CLI args or environment."""
    # Check for --config argument (passed after -- in streamlit run app.py -- --config ...)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=None)
    args, _ = parser.parse_known_args()
    if args.config:
        return args.config
    # Check environment variable
    env_path = os.environ.get("EXAGO_LAUNCHER_CONFIG")
    if env_path:
        return env_path
    # Default: config.yaml next to this script
    return str(Path(__file__).parent / "config.yaml")


def validate_directories(cfg: dict) -> list[tuple[str, str, bool]]:
    """Return list of (label, path, exists) for configured directories."""
    keys = [
        ("exago_dir", "ExaGO Root"),
        ("applications_dir", "Applications"),
        ("datafiles_dir", "Data Files"),
        ("viz_dir", "Visualization"),
        ("gic_data_dir", "GIC Data"),
        ("output_dir", "Output"),
    ]
    results = []
    for key, label in keys:
        p = cfg.get(key, "")
        results.append((label, p, Path(p).is_dir()))
    return results


def _normalize_paths(items):
    """Convert config items into a list of path strings."""
    paths = []
    for item in items:
        if isinstance(item, str):
            paths.append(item)
        elif isinstance(item, dict):
            p = item.get("path")
            if p:
                paths.append(p)
    return paths

def _merge_env(env, key, values):
    values = _normalize_paths(values or [])
    if not values:
        return
    existing = env.get(key)
    env[key] = ":".join([*values, existing] if existing else values)


def build_env(cfg: dict) -> dict:
    env = os.environ.copy()
    env_cfg = cfg.get("environment", {}) or {}

    _merge_env(env, "LD_LIBRARY_PATH", env_cfg.get("ld_library_path"))
    _merge_env(env, "PATH", env_cfg.get("path"))
    _merge_env(env, "PYTHONPATH", env_cfg.get("pythonpath"))

    return env



# ── Application Definitions ─────────────────────────────────────────────────

APPLICATIONS = {
    "OPFLOW": {"desc": "AC Optimal Power Flow", "binary": "opflow"},
    "PFLOW": {"desc": "AC Power Flow", "binary": "pflow"},
    "DCOPFLOW": {"desc": "DC Optimal Power Flow", "binary": "dcopflow"},
    "SCOPFLOW": {"desc": "Security-Constrained OPF", "binary": "scopflow"},
    "SOPFLOW": {"desc": "Stochastic Security-Constrained OPF", "binary": "sopflow"},
    "TCOPFLOW": {"desc": "Multi-Period OPF", "binary": "tcopflow"},
}

# ── Streamlit Page Config ────────────────────────────────────────────────────

st.set_page_config(page_title="ExaGO Launcher", page_icon="⚡", layout="wide")

# ── Load Configuration ───────────────────────────────────────────────────────

config_path = get_config_path()
if "cfg" not in st.session_state or st.session_state.get("_config_path") != config_path:
    st.session_state.cfg = load_config(config_path)
    st.session_state._config_path = config_path

cfg = st.session_state.cfg

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("figs/ornl-logo.png", width='stretch')

    st.title("⚡ ExaGO Launcher")
    st.caption(f"Config: `{config_path}`")
    if st.button("Reload Config"):
        st.session_state.cfg = load_config(config_path)
        cfg = st.session_state.cfg
        st.rerun()

    st.subheader("Directory Status")
    for label, path, exists in validate_directories(cfg):
        if exists:
            st.success(f"{label}", icon="✅")
        else:
            st.warning(f"{label}: missing\n`{path}`", icon="⚠️")

datafiles_dir = cfg.get("datafiles_dir", "")
gic_data_dir = cfg.get("gic_data_dir", "")

# Staging directory for uploaded files
_staging_dir = Path(cfg.get("output_dir", "/tmp")) / "staging"
_staging_dir.mkdir(parents=True, exist_ok=True)


def file_upload(label: str, extensions: list[str], key: str,
                help_text: str | None = None) -> str | None:
    """Render a file uploader, save uploaded file to staging, return the on-disk path.

    Returns the absolute path to the staged file, or None if nothing uploaded.
    """
    uploaded = st.file_uploader(label, type=extensions, key=key, help=help_text)
    if uploaded is not None:
        dest = _staging_dir / uploaded.name
        dest.write_bytes(uploaded.getvalue())
        st.caption(f":green[Staged: `{dest}`]")
        return str(dest)
    return None

# ── Tabs ─────────────────────────────────────────────────────────────────────

tab_run, tab_viz, tab_python = st.tabs(["Run Simulation", "Visualization", "Python API"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Run Simulation
# ══════════════════════════════════════════════════════════════════════════════

with tab_run:
    st.header("Run Simulation")

    # Step 1 – Application Selection
    app_labels = [f"{k} - {v['desc']}" for k, v in APPLICATIONS.items()]
    selected_label = st.selectbox("Application", app_labels, key="app_select")
    app_key = selected_label.split(" - ")[0]
    app_info = APPLICATIONS[app_key]

    st.divider()

    # Step 2 – Common Parameters
    st.subheader("Common Parameters")
    col1, col2 = st.columns(2)
    with col1:
        netfile = file_upload("Network file (`-netfile`)", ["m"], key="run_netfile",
                              help_text="Select a .m network file")
    with col2:
        n_mpi = st.number_input("MPI processes", min_value=1, value=1, key="run_mpi")

    col3, col4 = st.columns(2)
    with col3:
        print_output = st.checkbox("Print output (`-print_output`)", value=True, key="run_print")
    with col4:
        save_output = st.checkbox("Save output (`-save_output`)", value=False, key="run_save")

    output_format = None
    if save_output:
        output_format = st.selectbox("Output format (`-opflow_output_format`)", ["MATPOWER", "JSON"], key="run_fmt")

    st.divider()

    # Step 3 – Application-Specific Parameters
    st.subheader(f"{app_key} Parameters")
    extra_args = []

    # ── OPFLOW ──────────────────────────────────────────────────────────────
    if app_key == "OPFLOW":
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            model = st.selectbox("Model", ["POWER_BALANCE_POLAR", "POWER_BALANCE_HIOP", "PBPOLRAJAHIOP"], key="opf_model")
            extra_args += ["-opflow_model", model]
        with col_b:
            solver = st.selectbox("Solver", ["IPOPT", "HIOP", "HIOPSPARSE"], key="opf_solver")
            extra_args += ["-opflow_solver", solver]
        with col_c:
            init = st.selectbox("Initialization", ["MIDPOINT", "FROMFILE", "ACPF", "FLATSTART", "DCOPF"], key="opf_init")
            extra_args += ["-opflow_initialization", init]

        with st.expander("Advanced OPFLOW Options"):
            ignore_lf = st.checkbox("Ignore line flow constraints", key="opf_ignore_lf")
            if ignore_lf:
                extra_args.append("-opflow_ignore_lineflow_constraints")

            col_ll, col_pi = st.columns(2)
            with col_ll:
                load_loss = st.checkbox("Include load loss variables", key="opf_loadloss")
                if load_loss:
                    extra_args.append("-opflow_include_loadloss_variables")
                    ll_penalty = st.number_input("Load loss penalty", value=1000.0, key="opf_ll_pen")
                    extra_args += ["-opflow_loadloss_penalty", str(ll_penalty)]
            with col_pi:
                pwr_imb = st.checkbox("Include power imbalance variables", key="opf_pwrimb")
                if pwr_imb:
                    extra_args.append("-opflow_include_powerimbalance_variables")
                    pi_penalty = st.number_input("Power imbalance penalty", value=10000.0, key="opf_pi_pen")
                    extra_args += ["-opflow_powerimbalance_penalty", str(pi_penalty)]

            genbus = st.selectbox("Generator bus voltage", ["VARIABLE_WITHIN_BOUNDS", "FIXED_WITHIN_QBOUNDS"], key="opf_genbus")
            extra_args += ["-opflow_genbusvoltage", genbus]

            objective = st.selectbox("Objective", ["MIN_GEN_COST", "MIN_GENSETPOINT_DEVIATION", "NO_OBJ"], key="opf_obj")
            extra_args += ["-opflow_objective", objective]

            tol = st.number_input("Tolerance", value=1e-6, format="%.1e", key="opf_tol")
            extra_args += ["-opflow_tolerance", str(tol)]

            use_gic = st.checkbox("Use GIC file", key="opf_use_gic")
            if use_gic:
                gic_file = file_upload("GIC file (`-gicfile`)", ["gic"], key="opf_gic",
                                       help_text="Select a .gic file")
                if gic_file:
                    extra_args += ["-gicfile", gic_file]

    # ── SCOPFLOW ────────────────────────────────────────────────────────────
    elif app_key == "SCOPFLOW":
        col_a, col_b = st.columns(2)
        with col_a:
            ctg_file = file_upload("Contingency file (`-ctgcfile`)", ["cont"], key="scop_ctg",
                                   help_text="Select a .cont file")
            if ctg_file:
                extra_args += ["-ctgcfile", ctg_file]
            n_ctg = st.number_input("Number of contingencies (-1=all)", value=-1, min_value=-1, key="scop_nc")
            extra_args += ["-scopflow_Nc", str(n_ctg)]
        with col_b:
            solver = st.selectbox("Solver", ["IPOPT", "HIOP", "EMPAR"], key="scop_solver")
            extra_args += ["-scopflow_solver", solver]
            mode = st.selectbox("Mode", ["0 - Preventive", "1 - Corrective"], key="scop_mode")
            extra_args += ["-scopflow_mode", mode[0]]

        with st.expander("Multi-Period Options"):
            enable_mp = st.checkbox("Enable multi-period", key="scop_mp")
            if enable_mp:
                extra_args.append("-scopflow_enable_multiperiod")
                col_mp1, col_mp2 = st.columns(2)
                with col_mp1:
                    duration = st.number_input("Duration (hours)", value=0.5, key="scop_dur")
                    extra_args += ["-scopflow_duration", str(duration)]
                    dt = st.number_input("Time step (minutes)", value=5.0, key="scop_dt")
                    extra_args += ["-scopflow_dT", str(dt)]
                with col_mp2:
                    lp = file_upload("Active load profile", ["csv"], key="scop_lp",
                                     help_text="Select a load_P .csv file")
                    if lp:
                        extra_args += ["-scopflow_ploadprofile", lp]
                    lq = file_upload("Reactive load profile", ["csv"], key="scop_lq",
                                     help_text="Select a load_Q .csv file")
                    if lq:
                        extra_args += ["-scopflow_qloadprofile", lq]
                    wp = file_upload("Wind profile", ["m"], key="scop_wind",
                                     help_text="Select a wind profile .m file")
                    if wp:
                        extra_args += ["-scopflow_windgenprofile", wp]

    # ── SOPFLOW ─────────────────────────────────────────────────────────────
    elif app_key == "SOPFLOW":
        col_a, col_b = st.columns(2)
        with col_a:
            scen = file_upload("Scenario file (`-windgen`)", ["csv"], key="sop_scen",
                               help_text="Select a scenario .csv file")
            if scen:
                extra_args += ["-windgen", scen]
            n_scen = st.number_input("Number of scenarios (-1=all)", value=-1, min_value=-1, key="sop_ns")
            extra_args += ["-sopflow_Ns", str(n_scen)]
        with col_b:
            ctg_file = file_upload("Contingency file (`-ctgcfile`)", ["cont"], key="sop_ctg",
                                   help_text="Select a .cont file")
            if ctg_file:
                extra_args += ["-ctgcfile", ctg_file]
            n_ctg = st.number_input("Number of contingencies (-1=all)", value=-1, min_value=-1, key="sop_nc")
            extra_args += ["-sopflow_Nc", str(n_ctg)]

        col_c, col_d = st.columns(2)
        with col_c:
            solver = st.selectbox("Solver", ["IPOPT", "HIOP", "EMPAR"], key="sop_solver")
            extra_args += ["-sopflow_solver", solver]
        with col_d:
            mode = st.selectbox("Mode", ["0 - Preventive", "1 - Corrective"], key="sop_mode")
            extra_args += ["-sopflow_mode", mode[0]]

        enable_mc = st.checkbox("Enable multi-contingency", key="sop_mc")
        if enable_mc:
            extra_args.append("-sopflow_enable_multicontingency")

    # ── TCOPFLOW ────────────────────────────────────────────────────────────
    elif app_key == "TCOPFLOW":
        col_a, col_b = st.columns(2)
        with col_a:
            lp = file_upload("Active power load profile", ["csv"], key="tc_lp",
                             help_text="Select a load_P .csv file")
            if lp:
                extra_args += ["-tcopflow_ploadprofile", lp]
            lq = file_upload("Reactive power load profile", ["csv"], key="tc_lq",
                             help_text="Select a load_Q .csv file")
            if lq:
                extra_args += ["-tcopflow_qloadprofile", lq]
        with col_b:
            use_wind = st.checkbox("Use wind generation profile", key="tc_use_wind")
            if use_wind:
                wp = file_upload("Wind generation profile", ["m"], key="tc_wind",
                                 help_text="Select a wind profile .m file")
                if wp:
                    extra_args += ["-tcopflow_windgenprofile", wp]
            dt = st.number_input("Time step (minutes)", value=5.0, min_value=0.1, key="tc_dt")
            extra_args += ["-tcopflow_dT", str(dt)]
            dur = st.number_input("Duration (hours)", value=0.5, min_value=0.01, key="tc_dur")
            extra_args += ["-tcopflow_duration", str(dur)]

    # ── PFLOW / DCOPFLOW: no extra options ──────────────────────────────────
    elif app_key in ("PFLOW", "DCOPFLOW"):
        st.info("No additional parameters for this application.")

    st.divider()

    # Step 4 – Build & Preview Command
    binary_path = str(Path(cfg["applications_dir"]) / app_info["binary"])

    cmd_parts = []
    if n_mpi > 1:
        cmd_parts += ["mpiexec", "-n", str(n_mpi)]
    cmd_parts.append(binary_path)
    cmd_parts += ["-netfile", netfile or "<upload network file>"]
    if print_output:
        cmd_parts.append("-print_output")
    if save_output:
        cmd_parts.append("-save_output")
        if output_format:
            cmd_parts += ["-opflow_output_format", output_format]
    cmd_parts += extra_args

    st.subheader("Command Preview")
    st.code(" \\\n  ".join(cmd_parts), language="bash")

    # Step 5 – Run
    if st.button("▶ Run Simulation", type="primary", key="run_btn", disabled=netfile is None):
        if netfile is None:
            st.error("Please upload a network file first.")
        else:
            st.session_state.run_output = ""
            st.session_state.run_status = "running"

    if st.session_state.get("run_status") == "running":
        output_dir = cfg.get("output_dir", "")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        env = build_env(cfg)
        t0 = time.time()
        output_placeholder = st.empty()
        status_placeholder = st.empty()

        try:
            proc = subprocess.Popen(
                cmd_parts,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=output_dir,
                env=env,
            )
            full_output = []
            for line in proc.stdout:
                full_output.append(line)
                output_placeholder.code("".join(full_output[-200:]), language="text")

            proc.wait()
            elapsed = time.time() - t0
            all_output = "".join(full_output)
            st.session_state.run_output = all_output
            st.session_state.run_status = "done"
            st.session_state.run_elapsed = elapsed
            st.session_state.run_returncode = proc.returncode

        except FileNotFoundError:
            st.error(f"Binary not found: `{binary_path}`")
            st.session_state.run_status = "error"
        except Exception as e:
            st.error(f"Execution error: {e}")
            st.session_state.run_status = "error"

    if st.session_state.get("run_status") == "done":
        elapsed = st.session_state.get("run_elapsed", 0)
        rc = st.session_state.get("run_returncode", -1)
        all_output = st.session_state.get("run_output", "")

        st.subheader("Results")
        st.metric("Execution Time", f"{elapsed:.2f}s")

        # Parse convergence
        if "Optimal Solution Found" in all_output or "Converged" in all_output:
            st.success("Converged — Optimal Solution Found", icon="✅")
        elif rc != 0:
            st.error(f"Simulation failed (exit code {rc})", icon="❌")
        else:
            st.info("Simulation completed (check output for details)")

        with st.expander("Full Output", expanded=True):
            st.code(all_output, language="text")

        # Download output if saved
        if save_output:
            output_dir = cfg.get("output_dir", "")
            out_entries = [p for p in Path(output_dir).glob("*") if p.name != "staging"]
            if out_entries:
                latest = max(out_entries, key=lambda p: p.stat().st_mtime)
                if latest.is_file():
                    st.download_button(
                        "Download Output File",
                        data=latest.read_bytes(),
                        file_name=latest.name,
                        key="dl_output",
                    )
                elif latest.is_dir():
                    # Zip directory contents for download
                    buf = io.BytesIO()
                    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                        for file in sorted(latest.rglob("*")):
                            if file.is_file():
                                zf.write(file, file.relative_to(latest))
                    buf.seek(0)
                    st.download_button(
                        f"Download Output ({latest.name}.zip)",
                        data=buf.getvalue(),
                        file_name=f"{latest.name}.zip",
                        mime="application/zip",
                        key="dl_output",
                    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: Visualization
# ══════════════════════════════════════════════════════════════════════════════

with tab_viz:
    st.header("Visualization")

    viz_dir = cfg.get("viz_dir", "")
    gic_data_dir_v = cfg.get("gic_data_dir", "")

    mode = st.radio("Mode", ["Run OPFLOW & Visualize", "Use Existing JSON"], horizontal=True, key="viz_mode")

    if mode == "Run OPFLOW & Visualize":
        st.subheader("Step 1: Select Files")
        col1, col2 = st.columns(2)
        with col1:
            viz_netfile = file_upload("Network file", ["m"], key="viz_netfile",
                                      help_text="Select a .m network file")
        with col2:
            use_gic = st.checkbox("Include GIC data", key="viz_gic")
            viz_gic_file = None
            if use_gic:
                viz_gic_file = file_upload("GIC file", ["gic"], key="viz_gic_file",
                                           help_text="Select a .gic file")

        # Build OPFLOW command for viz
        opflow_bin = str(Path(cfg["applications_dir"]) / "opflow")
        viz_cmd = [
            opflow_bin,
            "-netfile", viz_netfile or "<upload network file>",
            "-save_output",
            "-opflow_output_format", "JSON",
            "-opflow_initialization", "FROMFILE",
            "-opflow_ignore_lineflow_constraints",
        ]
        if use_gic and viz_gic_file:
            viz_cmd += ["-gicfile", viz_gic_file]

        st.subheader("Step 2: Run OPFLOW")
        st.code(" \\\n  ".join(viz_cmd), language="bash")

        if st.button("▶ Run OPFLOW for Visualization", type="primary", key="viz_run_btn",
                     disabled=viz_netfile is None):
            env = build_env(cfg)
            output_dir = cfg.get("output_dir", "")
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            with st.spinner("Running OPFLOW..."):
                try:
                    result = subprocess.run(
                        viz_cmd,
                        capture_output=True,
                        text=True,
                        cwd=output_dir,
                        env=env,
                        timeout=300,
                    )
                    if result.returncode == 0:
                        st.success("OPFLOW completed successfully")
                        st.session_state.viz_opflow_output = result.stdout
                    else:
                        st.error("OPFLOW failed")
                        st.code(result.stdout + "\n" + result.stderr, language="text")
                        st.session_state.viz_opflow_output = None
                except FileNotFoundError:
                    st.error(f"Binary not found: `{opflow_bin}`")
                    st.session_state.viz_opflow_output = None

        if st.session_state.get("viz_opflow_output"):
            st.subheader("Step 3: Copy Output to Viz")
            # Find the JSON output file
            output_dir = cfg.get("output_dir", "")
            json_outputs = list(Path(output_dir).glob("*.json"))
            if json_outputs:
                latest_json = max(json_outputs, key=lambda p: p.stat().st_mtime)
                st.info(f"JSON output: `{latest_json.name}`")

                if st.button("Copy to viz/data & Configure", key="viz_copy_btn"):
                    # Run geninputfile.py
                    geninput_script = str(Path(viz_dir) / "geninputfile.py")
                    try:
                        result = subprocess.run(
                            [sys.executable, geninput_script, str(latest_json)],
                            capture_output=True,
                            text=True,
                            cwd=viz_dir,
                        )
                        if result.returncode == 0:
                            st.success("File copied and viz configured")
                            st.session_state.viz_ready = True
                        else:
                            st.error(f"geninputfile.py failed: {result.stderr}")
                    except FileNotFoundError:
                        st.error(f"Script not found: `{geninput_script}`")
            else:
                st.warning("No JSON output found. Run OPFLOW first.")

    else:
        # Use existing JSON
        st.subheader("Select Existing JSON File")
        json_path = file_upload("JSON file", ["json"], key="viz_sel_json",
                                help_text="Select a .json output file")

        if st.button("Configure Viz with Selected File", key="viz_existing_btn",
                     disabled=json_path is None):
            geninput_script = str(Path(viz_dir) / "geninputfile.py")
            try:
                result = subprocess.run(
                    [sys.executable, geninput_script, json_path],
                    capture_output=True,
                    text=True,
                    cwd=viz_dir,
                )
                if result.returncode == 0:
                    st.success("Viz configured successfully")
                    st.session_state.viz_ready = True
                else:
                    st.error(f"geninputfile.py failed: {result.stderr}")
            except FileNotFoundError:
                st.error(f"Script not found: `{geninput_script}`")

    # Launch viz server
    st.divider()
    st.subheader("Visualization Server")

    col_launch, col_status = st.columns(2)
    with col_launch:
        if st.button("🚀 Launch Viz Server", key="viz_launch_btn"):
            try:
                proc = subprocess.Popen(
                    ["npm", "start"],
                    cwd=viz_dir,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                st.session_state.viz_pid = proc.pid
                st.success(f"Viz server started (PID: {proc.pid})")
            except FileNotFoundError:
                st.error("npm not found. Install Node.js to use the visualization server.")

    with col_status:
        if st.session_state.get("viz_pid"):
            st.info(f"Server PID: {st.session_state.viz_pid}")
            if st.button("Stop Server", key="viz_stop_btn"):
                try:
                    os.killpg(os.getpgid(st.session_state.viz_pid), signal.SIGTERM)
                    st.success("Server stopped")
                    del st.session_state.viz_pid
                except ProcessLookupError:
                    st.info("Server already stopped")
                    del st.session_state.viz_pid

    st.markdown("**Open visualization:** [http://localhost:5173](http://localhost:5173)")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: Python API
# ══════════════════════════════════════════════════════════════════════════════

with tab_python:
    st.header("Python API")
    st.info("Generate equivalent Python code using the `exago` module for the current simulation configuration.")

    # Generate Python code based on Tab 1 settings
    py_app = app_key
    py_netfile = netfile or "<upload network file>"

    # Map UI initialization names to Python API enum names
    _init_to_enum = {
        "MIDPOINT": "OPFLOWINIT_MIDPOINT",
        "FROMFILE": "OPFLOWINIT_FROMFILE",
        "ACPF": "OPFLOWINIT_ACPF",
        "FLATSTART": "OPFLOWINIT_FLATSTART",
        "DCOPF": "OPFLOWINIT_DCOPF",
    }
    # Map UI output format names to Python API constants
    _fmt_to_enum = {
        "MATPOWER": "exago.MATPOWER",
        "JSON": "exago.JSON",
        "CSV": "exago.CSV",
    }

    # ── Build environment comment block from config ──────────────────────
    env_cfg = cfg.get("environment", {}) or {}
    env_comment_lines = [
        "#!/usr/bin/env python3",
        "# ExaGO Python API Script",
        "# Generated by ExaGO Launcher",
        "#",
        "# Required environment variables (set these before running):",
    ]
    py_paths = env_cfg.get("pythonpath", []) or []
    if py_paths:
        env_comment_lines.append(f"#   export PYTHONPATH={':'.join(py_paths)}:$PYTHONPATH")
    ld_paths = env_cfg.get("ld_library_path", []) or []
    if ld_paths:
        env_comment_lines.append(f"#   export LD_LIBRARY_PATH={':'.join(ld_paths)}:$LD_LIBRARY_PATH")
    bin_paths = env_cfg.get("path", []) or []
    if bin_paths:
        env_comment_lines.append(f"#   export PATH={':'.join(bin_paths)}:$PATH")

    # ── Common preamble ──────────────────────────────────────────────────
    lines = env_comment_lines + [
        "",
        "import exago",
        "",
        f'# Initialize ExaGO (sets up PETSc/MPI)',
        f'exago.initialize("{py_app.lower()}")',
        "",
    ]

    # ── Body lines (filled per application, finalize appended after) ─────
    body = []

    if py_app == "OPFLOW":
        body += [
            "# Create OPFLOW instance",
            "opf = exago.OPFLOW()",
            "",
            f'opf.read_mat_power_data("{py_netfile}")',
            "",
        ]
        model_val = st.session_state.get("opf_model", "POWER_BALANCE_POLAR")
        solver_val = st.session_state.get("opf_solver", "IPOPT")
        init_val = st.session_state.get("opf_init", "MIDPOINT")
        init_enum = _init_to_enum.get(init_val, f"OPFLOWINIT_{init_val}")

        body.append(f'opf.set_model("{model_val}")')
        body.append(f'opf.set_solver("{solver_val}")')
        body.append(f'opf.set_initialization_type("{init_enum}")')
        body.append("")

        if st.session_state.get("opf_ignore_lf"):
            body.append("opf.set_ignore_lineflow_constraints(True)")
        if st.session_state.get("opf_loadloss"):
            body.append("opf.set_has_loadloss(True)")
            body.append(f"opf.set_loadloss_penalty({st.session_state.get('opf_ll_pen', 1000.0)})")
        if st.session_state.get("opf_pwrimb"):
            body.append("opf.set_has_bus_power_imbalance(True)")
            body.append(f"opf.set_bus_power_imbalance_penalty({st.session_state.get('opf_pi_pen', 10000.0)})")

        genbus_val = st.session_state.get("opf_genbus", "VARIABLE_WITHIN_BOUNDS")
        body.append(f'opf.set_gen_bus_voltage_type("{genbus_val}")')

        obj_val = st.session_state.get("opf_obj", "MIN_GEN_COST")
        body.append(f'opf.set_objective_type("{obj_val}")')

        tol_val = st.session_state.get("opf_tol", 1e-6)
        body.append(f"opf.set_tolerance({tol_val})")
        body.append("")
        body.append("# Solve")
        body.append("opf.solve()")
        body.append("")
        if print_output:
            body.append("# Print solution")
            body.append("opf.print_solution()")
        if save_output:
            fmt = output_format or "MATPOWER"
            fmt_enum = _fmt_to_enum.get(fmt, f"exago.{fmt}")
            body.append(f"opf.save_solution({fmt_enum})")

    elif py_app == "SCOPFLOW":
        body += [
            "# Create SCOPFLOW instance",
            "scopf = exago.SCOPFLOW()",
            "",
            f'scopf.set_network_data("{py_netfile}")',
            "",
        ]
        if st.session_state.get("scop_ctg"):
            body.append(f'scopf.set_contingency_data("{st.session_state["scop_ctg"]}", exago.ContingencyFileInputFormat.NATIVE)')
        nc = st.session_state.get("scop_nc", -1)
        body.append(f"scopf.set_num_contingencies({nc})")
        solver_val = st.session_state.get("scop_solver", "IPOPT")
        body.append(f'scopf.set_solver("{solver_val}")')
        mode_val = st.session_state.get("scop_mode", "0 - Preventive")
        body.append(f"scopf.set_mode({mode_val[0]})")
        body.append("")
        body.append("# Solve")
        body.append("scopf.solve()")
        if print_output:
            body.append("")
            body.append("# Print solution")
            body.append("scopf.print_solution(0)")

    elif py_app == "SOPFLOW":
        body += [
            "# Create SOPFLOW instance",
            "sopf = exago.SOPFLOW()",
            "",
            f'sopf.set_network_data("{py_netfile}")',
            "",
        ]
        if st.session_state.get("sop_scen"):
            body.append(f'sopf.set_scenario_data("{st.session_state["sop_scen"]}", exago.ScenarioFileInputFormat.NATIVE_SINGLEPERIOD, exago.ScenarioUncertaintyType.WIND)')
        ns = st.session_state.get("sop_ns", -1)
        body.append(f"sopf.set_num_scenarios({ns})")
        if st.session_state.get("sop_ctg"):
            body.append(f'sopf.set_contingency_data("{st.session_state["sop_ctg"]}", exago.ContingencyFileInputFormat.NATIVE)')
        nc = st.session_state.get("sop_nc", -1)
        body.append(f"sopf.set_num_contingencies({nc})")
        solver_val = st.session_state.get("sop_solver", "IPOPT")
        body.append(f'sopf.set_solver("{solver_val}")')
        body.append("")
        body.append("# Solve")
        body.append("sopf.solve()")
        if print_output:
            body.append("")
            body.append("# Print solution")
            body.append("sopf.print_solution(0)")

    elif py_app == "TCOPFLOW":
        body += [
            "# Create TCOPFLOW instance",
            "tcopf = exago.TCOPFLOW()",
            "",
            f'tcopf.set_network_data("{py_netfile}")',
            "",
        ]
        tc_lp = st.session_state.get("tc_lp")
        tc_lq = st.session_state.get("tc_lq")
        if tc_lp and tc_lq:
            body.append(f'tcopf.set_load_profiles("{tc_lp}", "{tc_lq}")')
        dt_val = st.session_state.get("tc_dt", 5.0)
        dur_val = st.session_state.get("tc_dur", 0.5)
        body.append(f"tcopf.set_time_step_and_duration({dt_val}, {dur_val})")
        body.append("")
        body.append("# Solve")
        body.append("tcopf.solve()")
        if print_output:
            body.append("")
            body.append("# Print solution")
            body.append("tcopf.print_solution(0)")

    elif py_app == "PFLOW":
        body += [
            "# Create PFLOW instance",
            "pf = exago.PFLOW()",
            "",
            f'pf.read_mat_power_data("{py_netfile}")',
            "",
            "# Solve",
            "pf.solve()",
        ]
        if print_output:
            body.append("")
            body.append("# Print solution")
            body.append("pf.print_solution()")

    elif py_app == "DCOPFLOW":
        # DCOPFLOW has no Python bindings — skip init/finalize wrapper
        lines = [
            "# NOTE: DCOPFLOW does not have Python API bindings.",
            "# Use the command-line interface instead (see Command Preview in the Run Simulation tab).",
        ]
        body = []

    # Assemble: preamble + body + finalize
    if body:
        lines += body
        lines += [
            "",
            "# Clean up",
            "try:",
            "    exago.finalize()",
            "except Exception:",
            "    pass  # MPI finalization may throw during cleanup; results are valid",
        ]

    python_code = "\n".join(lines)
    st.code(python_code, language="python")

    col_copy, col_run = st.columns(2)
    with col_copy:
        st.download_button(
            "📋 Download as .py",
            data=python_code,
            file_name=f"exago_{py_app.lower()}.py",
            mime="text/x-python",
            key="py_dl",
        )
    with col_run:
        if st.button("▶ Run with Python API", key="py_run_btn"):
            env = build_env(cfg)
            with st.spinner("Running Python script..."):
                try:
                    result = subprocess.run(
                        [sys.executable, "-c", python_code],
                        capture_output=True,
                        text=True,
                        env=env,
                        timeout=300,
                    )
                    if result.returncode == 0:
                        st.success("Script completed successfully")
                    else:
                        st.error(f"Script failed (exit code {result.returncode})")
                    if result.stdout:
                        with st.expander("Output", expanded=True):
                            st.code(result.stdout, language="text")
                    if result.stderr:
                        with st.expander("Errors"):
                            st.code(result.stderr, language="text")
                except subprocess.TimeoutExpired:
                    st.error("Script timed out after 5 minutes")
