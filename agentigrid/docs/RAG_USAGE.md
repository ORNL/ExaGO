# RAG in AgentiGrid: optional retrieval grounding

## What it is

Retrieval-augmented generation (RAG) adds a short block of reference material to the LLM's prompt. The material comes from a curated corpus in `rag/corpus/`: ExaGO tool help, case metadata, methodology notes, and certified examples of correct AgentiGrid responses. The aim is fewer proposals that are executable but wrong for the goal.

- **Off by default.** With RAG off, AgentiGrid behaves exactly as before and needs none of the extra dependencies.
- **Generation stage only.** Retrieval changes what the LLM sees. It never reaches the deterministic validator (`engine/validation.py`), so verification stays reproducible.
- **Defensive.** Any retrieval failure (embedding server down, empty index, missing `chromadb`) returns nothing, and the run behaves like the no-RAG baseline. Failures are counted in the journal (`rag_config.stats`), so a degraded run is visible rather than mislabelled.

## Modes

The environment variable `AGENTIGRID_RAG_MODE` selects the mode. The UI sets it from the "Reference knowledge (RAG)" selector under **Advanced**.

| Mode | What reaches the prompt |
|---|---|
| `off` (default) | Nothing |
| `basic` | Up to `k=3` chunks with cosine similarity of at least `min_score=0.35` |
| `corrective` | Retrieved chunks are graded CORRECT, AMBIGUOUS or INCORRECT, then refined to the relevant sentences. On INCORRECT the query is reformulated once; if still INCORRECT, the context is withheld. |

The older switch `AGENTIGRID_RAG=1` / `0` still works and means `basic` / `off`.

**Corrective mode options:**

- `AGENTIGRID_CRAG_GRADER`: `cosine` (default; deterministic) or `reranker`. The reranker is a local cross-encoder. It needs `sentence-transformers`; the model is set with `AGENTIGRID_RERANKER_MODEL`, default `BAAI/bge-reranker-base`.
- `AGENTIGRID_CRAG_TAU_LOWER` / `AGENTIGRID_CRAG_TAU_UPPER`: grading thresholds, default 0.30 / 0.50. These are starting points, not calibrated values.

**Store selection:** `AGENTIGRID_RAG_STORE` and `AGENTIGRID_RAG_COLLECTION` point at a different index, for example one built from another corpus.

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

   It prints the number of chunks and records the corpus hash and chunking scheme in `rag/store/ingest_manifest.json`. Chunking keeps each paragraph whole up to 3000 characters, so worked examples are never cut in half.

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

`journal.json` gets two fields:

- `rag_enabled`;
- `rag_config`: the mode, `k`, `min_score`, the thresholds and grader for corrective mode, and the counts of retrieval calls, errors and empty results.

## Troubleshooting

- **Empty retrieval with RAG on.** There are three likely causes:
  - `OLLAMA_HOST` is unset or wrong;
  - the index is empty or was built with another chunking scheme; rebuild it with `rm -rf rag/store && python -m agentigrid.rag.ingest rag/corpus`;
  - every hit is below `min_score`.
- **`enabled: false` in `rag_config` although a mode was set.** The retriever disabled itself at start-up: either `chromadb` is missing or the store path is wrong.
- **`model ... not found (404)` from embeddings.** Run `ollama pull nomic-embed-text`.
- **Reranker not offered in the UI.** `sentence-transformers` isn't installed.
