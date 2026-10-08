# GridKit Application Binaries

Place or symlink GridKit phasor-dynamics binaries in this directory.

AgentiGrid does not run GridKit yet. These links prepare for transient
studies that will be added later.

## Supported binaries

- `DynamicSimulation` — Time-domain simulation with fault events
- `ContingencyAnalysis` — Repeats the simulation for every bus fault in the case

Each binary takes a single solver JSON file: `DynamicSimulation study.solver.json`.

## Example

Run from the AgentiGrid root, with `GRIDKIT` set to your GridKit checkout:

```bash
GRIDKIT=/path/to/GridKit
for p in DynamicSimulation ContingencyAnalysis; do
  ln -s "$GRIDKIT/build/bin/$p" ./applications/gridkit/$p
done
```
