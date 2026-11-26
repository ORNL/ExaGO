# RAG in AgentiGrid: optional retrieval grounding

## What it is

Retrieval-augmented generation (RAG) adds a short block of reference material to the LLM's prompt. The material comes from a frozen corpus that is indexed into a vector store: tool help, case metadata, methodology notes, and certified examples of correct AgentiGrid responses. The aim is fewer proposals that are executable but wrong for the goal. RAG is wired into the ExaGO agent loop; the GridKit loop does not use it yet.

- **Off by default.** With RAG off, AgentiGrid behaves exactly as before and needs none of the extra dependencies.
- **Generation stage only.** Retrieval changes what the LLM sees. It never reaches the deterministic validator (`engine/validation.py`), so verification stays reproducible.
- **Fails before the run, not silently during it.** When a mode other than `off` is set, AgentiGrid checks before the first iteration that the store opens, is not empty and the embedding server answers. If not, it stops with a clear message (CLI exit code 1; the UI shows the error) instead of running without references.
- **Defensive during the run.** If a single retrieval fails mid-run (for example the embedding server drops out), that iteration runs without references and the failure is counted in the journal (`rag_config.stats`), so a degraded run is visible rather than mislabelled.

## Modes

The environment variable `AGENTIGRID_RAG_MODE` selects the mode. The UI sets it from the "Reference knowledge (RAG)" selector under **Advanced**.

| Mode | What reaches the prompt |
|---|---|
| `off` (default) | Nothing |
| `basic` | Up to `k=3` chunks with cosine similarity of at least `min_score=0.35` |
| `corrective` | Retrieved chunks are graded CORRECT, AMBIGUOUS or INCORRECT, then refined to the relevant sentences. On INCORRECT the query is reformulated once; if still INCORRECT, the context is withheld. |

The older switch `AGENTIGRID_RAG=1` / `0` still works and means `basic` / `off`.

**Query** (`AGENTIGRID_RAG_QUERY`, any mode):

| Value | What is embedded and searched |
|---|---|
| `goal` (default) | The goal text. Every iteration sends the same query, so a fixed store returns the same references each time. |
| `state` | The goal text plus a short summary of the latest solve (convergence, feasibility, number of violations, voltage range, maximum line loading, the worst violations) and the previous iteration's command errors. Capped at 1,500 characters. |
| `error` | The previous iteration's command errors first, then the worst violations, then the state line, without the goal text. Falls back to the goal text when there are no errors or violations. |

`state` and `error` let retrieval follow what is going wrong; with `goal`, a corrective grader sees the same query, and gives the same verdict, in every iteration. The query is built only from what the agent itself sees. Unknown values stop the run at start-up.

Retrieved references are preceded by one caution line: they come from other networks, so bus, branch and generator numbers must be taken from the current network.

**Corrective mode options:**

- `AGENTIGRID_CRAG_GRADER`: `cosine` (default; deterministic) or `reranker`. The reranker is a local cross-encoder. It needs `sentence-transformers`; the model is set with `AGENTIGRID_RERANKER_MODEL`, default `BAAI/bge-reranker-base`.
- `AGENTIGRID_CRAG_TAU_LOWER` / `AGENTIGRID_CRAG_TAU_UPPER`: grading thresholds, default 0.30 / 0.50. These are starting points, not calibrated values: with the cosine grader they rate almost everything CORRECT.
- `AGENTIGRID_CRAG_REFINE`: `strips` (default; kept chunks are split into sentences, re-scored and recombined) or `whole` (a kept chunk is injected intact if its own grade reaches the lower threshold). Use `whole` when the corpus holds structured worked examples, which `strips` cuts apart. Unknown values stop the run at start-up.

**Store selection:** `AGENTIGRID_RAG_STORE` and `AGENTIGRID_RAG_COLLECTION` point at a different index, for example one built from another corpus.

## The bundled corpus is a demo

`rag/corpus/` (frozen hash `da4b77a5…`, see `corpus_manifest.json`) is a small **demo and test corpus**. It was built from small networks (case9mod, case118, ACTIVSg500) and exists so a fresh checkout can check that retrieval works end to end. It is not an experiment corpus:

- its worked examples come from networks of 500 buses or fewer, so they say little about large networks;
- any experiment that evaluates on case9mod, case118 or ACTIVSg500 would retrieve worked answers for its own test network (leakage).

For experiments, build and freeze a dedicated corpus (`rag/tools/corpus_guard.py --freeze`), ingest it into its own store, and set `AGENTIGRID_RAG_STORE` to that store. The journal records which corpus a run used (`rag_config.store_manifest.corpus_sha256`, see below), so runs on the demo store are easy to spot.

## Code layout

- `agentigrid/rag/`, the importable package `agentigrid.rag`:
  - `embed.py`: Ollama embeddings (`nomic-embed-text`);
  - `store.py`: persistent Chroma index, cosine distance;
  - `retriever.py`: basic mode;
  - `corrective.py`: corrective mode;
  - `grader_reranker.py`;
  - `ingest.py`: builds the index;
  - `corpus_hash.py`: corpus identity and chunking scheme.
- `rag/corpus/`: the source `.txt` / `.md` files and `corpus_manifest.json` (the frozen corpus hash).
- `rag/store/`: the built index. It's gitignored; build it locally.
- `rag/tools/`: corpus builders and the audit tool (see `rag/tools/README.md`).

## Setup

All commands run from the AgentiGrid root (`agentigrid/`).

1. Install the optional dependencies:

   ```bash
   pip install -r requirements-rag.txt
   ```

2. Run Ollama with the embedding model, and point AgentiGrid at it if it isn't on localhost:

   ```bash
   ollama pull nomic-embed-text
   export OLLAMA_HOST="http://<ollama-host>:11434"
   ```

3. Build the index. Re-run this after any corpus change.

   ```bash
   rm -rf rag/store && python -m agentigrid.rag.ingest rag/corpus
   ```

   Ingest first checks that the embedding server answers; if not, it stops with a clear message and leaves an existing store untouched. It prints the number of chunks and records the corpus hash and chunking scheme in `rag/store/ingest_manifest.json`. Chunking keeps each paragraph whole up to 3000 characters, so worked examples are never cut in half.

## Running

**CLI:**

```bash
GOAL="Reduce total generation cost by 10%"
agentigrid ./data/exago/examples/case39.m "$GOAL" --backend ollama --model qwen2.5:7b --max-iter 4                          # off
AGENTIGRID_RAG_MODE=basic agentigrid ./data/exago/examples/case39.m "$GOAL" --backend ollama --model qwen2.5:7b --max-iter 4  # basic
```

**UI:** `./launcher/run.sh`, then go to **Advanced**, then **Reference knowledge (RAG)**. During a search, the live monitor shows a grounding caption with the mode, the number of references and the top score. The results page states whether the run used RAG.

**Check retrieval on its own:**

```bash
python -c "import os; from agentigrid.rag import Retriever; print(Retriever(enabled=True, host=os.environ.get('OLLAMA_HOST', 'http://localhost:11434')).retrieve('Assess security under single-element outages'))"
```

Lines like `[ref N | score 0.xx]` mean it works. Empty output means see Troubleshooting.

## What a run records

`journal.json` gets three fields:

- `rag_enabled`;
- `rag_config`: the mode, `k`, `min_score`, the thresholds and grader for corrective mode, the counts of retrieval calls, errors and empty results, and the store's identity:
  - `store_path`, `collection`, `embed_host`, `embed_model`;
  - `store_manifest`, copied from the store's `ingest_manifest.json`: `corpus_sha256`, `files`, `chunks`, `collection`, `embed_model`, `chunking`, `ingested_at`. It is `null` for a store built by an older ingest; rebuild such a store.

- `rag_log`: one record per retrieval (RAG on only): iteration, query mode, the query and its SHA-256, the scored hits (source file, similarity, length, a 16-hex content hash, and the grader's score in corrective mode), how many were kept, the verdict and any rewrite (corrective), the action (`used`, `empty`, `withheld`, `error`), and the length and SHA-256 of the injected block. It shows exactly what each iteration's model saw from retrieval.

`rag_config` also records `query_mode`, and `refine` in corrective mode.

Token use is unaffected: RAG makes no LLM calls (corrective reformulation is a fixed keyword expansion), so `token_usage` covers every call of a RAG run.

## Troubleshooting

- **Empty retrieval with RAG on.** There are three likely causes:
  - `OLLAMA_HOST` is unset or wrong;
  - the index is empty or was built with another chunking scheme; rebuild it with `rm -rf rag/store && python -m agentigrid.rag.ingest rag/corpus`;
  - every hit is below `min_score`.
- **"RAG is on but the vector store could not be opened / is empty".** `chromadb` is missing, `AGENTIGRID_RAG_STORE` points at the wrong folder, or the store was never built. Install `requirements-rag.txt` and run the ingest command above.
- **"RAG is on but the embedding server does not answer".** Ollama isn't running, `OLLAMA_HOST` is wrong, or the model isn't pulled.
- **`model ... not found (404)` from embeddings.** Run `ollama pull nomic-embed-text`.
- **Reranker not offered in the UI.** `sentence-transformers` isn't installed.
