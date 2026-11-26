"""ExaGO documentation corpus builder; one store from several frozen corpora."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from agentigrid.rag.corpus_hash import corpus_files, corpus_hash

ROOT = Path(__file__).resolve().parents[2]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bd = _load("build_docs_corpus", "rag/tools/build_docs_corpus.py")

MD = """<!-- comment -->
## SCOPFLOW
SCOPFLOW solves a contingency-constrained OPF. See [OPFLOW](./opflow.md).
![diagram](img.png)

### Usage

#### Options
|-netfile| Name of network file|
|-ctgcfile| contingency list|

```
# not a heading inside code
./scopflow -netfile x.m
```
"""


def test_sections_fold_heading_only_and_ignore_code_hashes():
    secs = bd.sections(bd.clean(MD))
    paths = [p for p, _ in secs]
    assert paths == ["SCOPFLOW", "SCOPFLOW > Usage > Options"]
    body = "\n".join(secs[1][1])
    assert "# not a heading inside code" in body


def test_blocks_carry_provenance_and_clean_markdown():
    blocks = bd.blocks_for("scopflow.md", MD, 2800)
    assert blocks[0].startswith("[source: ExaGO documentation | file: scopflow.md | section: SCOPFLOW]\n")
    assert "See OPFLOW." in blocks[0] and "](" not in blocks[0] and "img.png" not in blocks[0]
    assert all("\n\n" not in b for b in blocks)          # ingest keeps each block one chunk


def test_long_sections_are_split_under_the_cap():
    md = "## Big\n" + "\n\n".join("p" * 900 for _ in range(6))
    blocks = bd.blocks_for("big.md", md, 2800)
    assert len(blocks) >= 2 and all(len(b) <= 2800 for b in blocks)
    assert all(b.startswith("[source: ExaGO documentation | file: big.md | section: Big]") for b in blocks)


def test_build_is_deterministic_and_notice_is_not_indexed(tmp_path):
    src = tmp_path / "repo" / "docs" / "web"
    src.mkdir(parents=True)
    (src / "scopflow.md").write_text(MD)
    (tmp_path / "repo" / "LICENSE").write_text("licence text")
    out = tmp_path / "corpus_docs"
    bd.build(src, out)
    h1 = corpus_hash(out)[0]
    bd.build(src, out)
    assert corpus_hash(out)[0] == h1
    names = sorted(p.name for p in corpus_files(out))
    assert names == ["exago_docs_scopflow.txt"]           # SOURCE and LICENSE.ExaGO are not indexed
    assert (out / "LICENSE.ExaGO").read_text() == "licence text"
