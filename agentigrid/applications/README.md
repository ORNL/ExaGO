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

```bash
# Symlink from an ExaGO build
ln -s /path/to/ExaGO/build/bin/opflow ./applications/opflow
ln -s /path/to/ExaGO/build/bin/scopflow ./applications/scopflow
ln -s /path/to/ExaGO/build/bin/tcopflow ./applications/tcopflow
ln -s /path/to/ExaGO/build/bin/sopflow ./applications/sopflow
ln -s /path/to/ExaGO/build/bin/dcopflow ./applications/dcopflow
ln -s /path/to/ExaGO/build/bin/pflow ./applications/pflow
```

Alternatively, set the `exago.binary_dir` config option to point to an external directory containing the binaries (e.g., the ExaGO build `bin/` directory).
