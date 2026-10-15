# GridKit Simulation Data

Place your own GridKit case files (`*.case.json`) in `datafiles/`. The `examples/`
subdirectory holds symlinks to the phasor-dynamics cases shipped with GridKit.

AgentiGrid reads these case files for transient-stability studies
(`agentigrid --tool gridkit <case.json> "goal"`). It never changes them: faults
and recorded variables are added to a copy in the run folder.

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
mkdir -p data/gridkit/examples
cd data/gridkit/examples
for f in "$GRIDKIT"/cases/PhasorDynamics/*/*.case.json; do
  ln -s "$f" .
done
```
