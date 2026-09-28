# ExaGO Application Binaries

Place or symlink ExaGO application binaries in this directory.

## Supported binaries

- `opflow` — Optimal Power Flow
- `scopflow` — Security-Constrained Optimal Power Flow
- `tcopflow` — Time-Coupled Optimal Power Flow
- `sopflow` — Stochastic Optimal Power Flow
- `dcopflow` — DC Optimal Power Flow
- `pflow` — Power Flow

## Example

Run from the AgentiGrid root, with `EXAGO` set to your ExaGO checkout:

```bash
EXAGO=/path/to/ExaGO
for p in opflow scopflow tcopflow sopflow dcopflow pflow; do
  ln -s "$EXAGO/build/bin/$p" ./applications/exago/$p
done
```

Alternatively, set the `exago.binary_dir` config option to point to an external directory containing the binaries (e.g., the ExaGO build `bin/` directory).
