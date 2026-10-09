# AgentiGrid Tests

Tests are grouped by the part of AgentiGrid they cover:

| Folder | Covers |
|---|---|
| `llm/` | LLM backends and JSON extraction (`agentigrid/backends/`) |
| `grid_files/` | MATPOWER reader, writer and network summary (`parsers/matpower_*`, `network_*`) |
| `commands/` | Grid-change commands, validation and the modifier (`engine/commands.py`, `modifier.py`) |
| `runner/` | Running simulation binaries and logging their invocations (`engine/executor.py`) |
| `results/` | Result readers and summaries for the LLM (`parsers/*_parser.py`, `*_summary.py`) |
| `exago/` | Support for individual ExaGO applications (DCOPFLOW, PFLOW, SCOPFLOW, TCOPFLOW, SOPFLOW) |
| `agent_loop/` | Agent loop controller and interactive steering (`engine/agent_loop.py`) |
| `journal/` | Search journal, objectives, Pareto front, session save and report regeneration |
| `analyses/` | Explore, sweeps, boundary search, contingency, relief, reserve and topology |
| `e2e/` | End-to-end runs through the agent loop |
| `fixtures/` | Shared sample files (e.g. `sample_opflow_output.txt`) |

`fixtures/sample_opflow_output.txt` is OPFLOW output for `case_ACTIVSg200.m` as
rewritten by AgentiGrid; `results/test_opflow_parser.py` checks values from it.
To regenerate it (needs the `opflow` binary), run the live executor test with
`AGENTIGRID_REGENERATE_FIXTURES=1 python -m pytest tests/runner/test_executor.py -k test_opflow_run`
and update the expected values in `results/test_opflow_parser.py`.

Run the fast tests from the AgentiGrid root:

```bash
python -m pytest -m "not slow"
```

Tests marked `slow` call a real LLM API and need an API key.
