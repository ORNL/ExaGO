# Application Binaries

AgentiGrid runs external simulation tools. Each tool has its own
subdirectory here holding symlinks to that tool's binaries.

```
applications/
├── exago/     ExaGO binaries (opflow, dcopflow, pflow, scopflow, sopflow, tcopflow)
└── gridkit/   GridKit binaries (DynamicSimulation, ContingencyAnalysis)
```

The tools are built and installed separately. See the README in each
subdirectory for the symlink commands.

The symlinks hold machine-specific paths and are not tracked by git; only
the READMEs are.
