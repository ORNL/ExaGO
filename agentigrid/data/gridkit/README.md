# GridKit Simulation Data

Place your own GridKit case files (`*.case.json`) in `datafiles/`. The `examples/`
subdirectory holds symlinks to the phasor-dynamics cases shipped with GridKit.

AgentiGrid does not run GridKit yet. These files prepare for transient
studies that will be added later.

## Supported file types

- **Case files** (`*.case.json`) — Buses, devices (machines, governors,
  exciters, loads, branches, bus faults) and the starting operating point.
  See GridKit's `GridKit/Model/PhasorDynamics/INPUT_FORMAT.md`.

## Example data symlinks

GridKit keeps each case in its own subdirectory; `examples/` links every case
file into one folder. Run from the AgentiGrid root, with `GRIDKIT` set to your
GridKit checkout:

```bash
GRIDKIT=/path/to/GridKit
cd data/gridkit/examples
for f in "$GRIDKIT"/cases/PhasorDynamics/*/*.case.json; do
  ln -s "$f" .
done
```
