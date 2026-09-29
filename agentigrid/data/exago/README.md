# ExaGO Simulation Data

Place your own ExaGO input files in `datafiles/`. The `examples/` subdirectory
holds symlinks to the data files shipped with ExaGO.

## Supported file types

- **MATPOWER `.m` files** — Network case definitions (bus, generator, branch data)
- **`.gic` files** — Geomagnetically Induced Current data
- **Contingency files** (`.cont`) — For security-constrained analysis (SCOPFLOW)
- **Load profiles** (`*_load_P.csv`, `*_load_Q.csv`) — Time-series load data for TCOPFLOW
- **Wind profiles** (`*_wind.csv`) — Wind generation profiles for TCOPFLOW (optional)
- **Scenario files** (`*_scenarios.csv`, `*_10_scenarios.csv`) — Stochastic wind scenarios (SOPFLOW)

## File naming conventions

### Contingency files (SCOPFLOW)
Name the file to match the base case: `<casename>.cont`
- `case9mod.m` → `case9.cont`
- `case_ACTIVSg200.m` → `case_ACTIVSg200.cont`

### Load profiles (TCOPFLOW)
Name profile files using the case prefix: `<casename>_load_P.csv` and `<casename>_load_Q.csv`

The launcher auto-selects profiles matching the base case using layered fallback:
1. Exact prefix: `case9mod_load_P.csv`
2. Strip known suffixes (e.g., "mod"): `case9_load_P.csv`
3. Fallback: all available profiles

Example:
- `case9mod.m` → `case9_load_P.csv` + `case9_load_Q.csv`

### Wind profiles (TCOPFLOW, optional)
- `<casename>_wind.csv` — e.g., `case9_wind.csv`

### Scenario files (SOPFLOW)
Name scenario files using the case prefix:
- `<casename>_scenarios.csv` — Multi-period scenario file (with timestamps)
- `<casename>_10_scenarios.csv` — Single-period scenario file (with weights)

Example:
- `case9mod_gen3_wind.m` → `case9_scenarios.csv` + `case9_10_scenarios.csv`

Note: SOPFLOW requires a network file with wind generators (`gentype='W2'`, `genfuel='wind'`).

## Example

The ACTIVSg200 synthetic test case is a good starting point:

```
data/exago/datafiles/
├── case_ACTIVSg200.m       # Network definition
├── case_ACTIVSg200.gic     # GIC data
└── case_ACTIVSg200.cont   # Contingency file (SCOPFLOW)
```

For TCOPFLOW, add load profiles:

```
data/exago/datafiles/
├── case9mod.m              # Network definition
├── case9_load_P.csv        # Active load profile
├── case9_load_Q.csv        # Reactive load profile
└── case9_wind.csv          # Wind profile (optional)
```

For SOPFLOW (stochastic scenarios):

```
data/exago/datafiles/
├── case9mod_gen3_wind.m    # Network with wind generator
├── case9_scenarios.csv     # Multi-period wind scenarios
└── case9_10_scenarios.csv  # Single-period wind scenarios with weights
```

## Example data symlinks

`examples/` links to ExaGO's own `datafiles/` folder (not the `datafiles/`
folder here). The case9 files live in ExaGO's `datafiles/case9/`
subdirectory under different names, so they are linked individually under
the names the conventions above expect. The IEEE 118-bus
case used by several tests comes from ExaGO's `tests/data/`. Run from the
AgentiGrid root, with `EXAGO` set to your ExaGO checkout:

```bash
EXAGO=/path/to/ExaGO
mkdir -p data/exago/examples
cd data/exago/examples
for f in "$EXAGO"/datafiles/*; do
  case "$(basename "$f")" in test_validation|unit) ;; *) ln -s "$f" . ;; esac
done
ln -s "$EXAGO/datafiles/case9/case9.cont"             case9.cont
ln -s "$EXAGO/datafiles/case9/case9_dcline.m"         case9_dcline.m
ln -s "$EXAGO/datafiles/case9/case9mod.m"             case9mod.m
ln -s "$EXAGO/datafiles/case9/case9mod_gen3_wind.m"   case9mod_gen3_wind.m
ln -s "$EXAGO/datafiles/case9/case9mod_gen3_wind2.m"  case9mod_gen3_wind2.m
ln -s "$EXAGO/datafiles/case9/case9mod_loadloss.m"    case9mod_loadloss.m
ln -s "$EXAGO/datafiles/case9/load_P.csv"             case9_load_P.csv
ln -s "$EXAGO/datafiles/case9/load_Q.csv"             case9_load_Q.csv
ln -s "$EXAGO/datafiles/case9/scenarios_9bus.csv"     case9_scenarios.csv
ln -s "$EXAGO/datafiles/case9/10_scenarios_9bus.csv"  case9_10_scenarios.csv
ln -s "$EXAGO/tests/data/ieee_118_bus_v10.m"          ieee_118_bus_v10.m
```
