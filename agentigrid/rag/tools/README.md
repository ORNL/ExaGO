# RAG corpus tools

Scripts that build and check the AgentiGrid RAG knowledge base (`rag/corpus/`)
from **ground truth**: the ExaGO binaries, the case files and AgentiGrid's own
parser and validator, rather than hand-written prose. Every chunk carries a
`[source: ...]` tag, so a retrieved reference traces back to where it came from.

Run everything **from the AgentiGrid root** (`agentigrid/`, where you launch
AgentiGrid). The defaults are relative paths: `applications/exago`, `data/exago`
and `rag/corpus`.

## 1. Tool help and case metadata

    python rag/tools/rag_harvest.py

This writes two kinds of file:

- `rag/corpus/exago_<app>_help.txt`: one per application, from `--help` / `--version`;
- `rag/corpus/exago_cases.txt`: bus, generator and branch counts for each `.m` file.

Options:

    --bin-dir applications/exago     # where the binaries are
    --data-dir data/exago            # repeatable
    --apps opflow scopflow ...       # subset
    --no-cases / --no-help

## 2. Schema exemplars

    python rag/tools/rag_schema_exemplars.py             # write rag/corpus/
    python rag/tools/rag_schema_exemplars.py --inspect   # print, write nothing

This writes `rag/corpus/agentigrid_schema_exemplars.txt`: complete, correct
responses, for example `set_all_bus_vlimits` nested inside a `modify` action and
never used as the top-level action.

AgentiGrid's own parser, validator and modifier certify each exemplar on
`case118`. The exception is the phase-shifter exemplar, which no reference case
can host, so it is labelled "schema-validated only".

## 3. Audit and freeze

    python rag/tools/corpus_guard.py rag/corpus                                 # personal paths only
    python rag/tools/corpus_guard.py --spec my_eval_spec.json rag/corpus          # + hold-out audit
    python rag/tools/corpus_guard.py --spec my_eval_spec.json rag/corpus --freeze

The audit flags three things:

- the text of an evaluated goal;
- a worked example on an evaluated network;
- an absolute personal path.

`--freeze` writes `corpus_manifest.json` with the corpus SHA-256. It refuses while any findings remain. A run's journal records `rag_config`, so the corpus a run used can be traced.

The shipped corpus (v1) was audited and frozen this way. The hold-out spec it was audited against belongs to a separate evaluation and is not part of this repository. `exago_exemplars_from_runs.txt` holds solver-certified worked examples, scraped from successful AgentiGrid runs on networks and goals outside that evaluation. The scraper and the batch runner that produced them are not included here.

Copyright: do not paste paper or textbook text into the corpus. State facts in
your own words and cite them, or keep such notes in a local, gitignored corpus.

## 4. Re-ingest after any corpus change

    rm -rf rag/store && python -m agentigrid.rag.ingest rag/corpus

## Notes

- **Idempotent.** Re-running overwrites only the generated files. Hand-written corpus files are left alone.
- **Offline.** None of these scripts calls an LLM or the network. Only `ingest` does, to get Ollama embeddings.
- **Review before trusting.** Generated chunks carry an "auto-generated, review before trusting" tag.
