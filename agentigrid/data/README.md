# Simulation Data

Input files are grouped by the tool that reads them:

```
data/
├── exago/
│   ├── datafiles/  your own ExaGO files (.m, .cont, profiles, scenarios)
│   └── examples/   symlinks to ExaGO's example data files
└── gridkit/
    ├── datafiles/  your own GridKit files (case .json)
    └── examples/   symlinks to GridKit's example case files
```

Put your own files in `exago/datafiles/` or `gridkit/datafiles/`. The `examples/`
folders only hold symlinks to the data shipped with each tool, so AgentiGrid
does not depend on them once you add your own files.

See the README in each subdirectory for supported file types and the symlink
commands.

The symlinks and your own data are not tracked by git; only the READMEs are.
