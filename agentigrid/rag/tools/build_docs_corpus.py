#!/usr/bin/env python3
"""Build the documentation corpus from ExaGO's own user documentation.

Turns ``docs/web/*.md`` of the ExaGO repository into retrieval-ready text: one
block per document section (heading + body), each prefixed with its provenance,
so a retrieved block is self-identifying like the core corpus chunks:

    [source: ExaGO documentation | file: scopflow.md | section: Usage > Options]
    <section text>

Blocks are separated by a blank line (ingest's paragraph-first chunking keeps
each block whole); a section longer than --max-chars is split at its own
paragraph breaks, each part keeping the header. Markdown links become their
text, images and HTML comments are dropped. Output is deterministic.

Much of the documentation (installation, TCOPFLOW, SOPFLOW, testing) is
irrelevant to typical AgentiGrid goals, so it is also useful as realistic noise
for a corrective grader.

    python rag/tools/build_docs_corpus.py --src ../docs/web --out rag/corpus_docs

Then audit and freeze it (corpus_guard.py --freeze) and ingest it into its own
store (python -m agentigrid.rag.ingest rag/corpus_docs --store rag/store_docs),
or build it as a subfolder of a larger corpus folder: ingest and the corpus hash
include subfolders.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
LINK = re.compile(r"(?<!!)\[([^\]]+)\]\([^)]*\)")
IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
COMMENT = re.compile(r"<!--.*?-->", re.S)
HTML_TAG = re.compile(r"</?(?:img|br|p|div|span|a)\b[^>]*>", re.I)


def clean(text: str) -> str:
    text = COMMENT.sub("", text)
    text = IMAGE.sub("", text)
    text = LINK.sub(r"\1", text)
    text = HTML_TAG.sub("", text)
    return text.replace("\r\n", "\n")


def sections(md: str) -> list[tuple[str, list[str]]]:
    """[(heading path, body lines)] in document order; heading-only sections are
    folded into the path of the next section that has a body."""
    out: list[tuple[str, list[str]]] = []
    stack: list[tuple[int, str]] = []
    body: list[str] = []
    in_code = False

    def flush():
        if any(l.strip() for l in body):
            path = " > ".join(t for _l, t in stack) or "(top)"
            out.append((path, list(body)))
        body.clear()

    for line in md.split("\n"):
        if line.strip().startswith("```"):
            in_code = not in_code
            body.append(line)
            continue
        m = None if in_code else HEADING.match(line)
        if m:
            flush()
            level, title = len(m.group(1)), m.group(2).strip()
            stack[:] = [(l, t) for l, t in stack if l < level] + [(level, title)]
        else:
            body.append(line)
    flush()
    return out


def blocks_for(fname: str, md: str, max_chars: int) -> list[str]:
    out = []
    for path, lines in sections(clean(md)):
        header = f"[source: ExaGO documentation | file: {fname} | section: {path}]"
        # paragraphs inside the section (blank-line separated), each without inner blank lines
        paras, cur = [], []
        for l in lines:
            if l.strip():
                cur.append(l.rstrip())
            elif cur:
                paras.append("\n".join(cur))
                cur = []
        if cur:
            paras.append("\n".join(cur))
        part: list[str] = []
        size = len(header)
        for p in paras:
            if part and size + len(p) + 1 > max_chars:
                out.append(header + "\n" + "\n".join(part))
                part, size = [], len(header)
            if len(p) + len(header) + 1 > max_chars:      # a single huge paragraph: hard wrap
                for i in range(0, len(p), max_chars - len(header) - 1):
                    out.append(header + "\n" + p[i:i + max_chars - len(header) - 1])
                continue
            part.append(p)
            size += len(p) + 1
        if part:
            out.append(header + "\n" + "\n".join(part))
    return out


def build(src: Path, out: Path, max_chars: int = 2800) -> dict:
    files = sorted(p for p in src.glob("*.md") if p.is_file())
    if not files:
        raise SystemExit(f"no .md files in {src}")
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("exago_docs_*.txt"):
        old.unlink()
    stats = {}
    for p in files:
        blocks = blocks_for(p.name, p.read_text(encoding="utf-8", errors="replace"), max_chars)
        if not blocks:
            continue
        (out / f"exago_docs_{p.stem}.txt").write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
        stats[p.name] = len(blocks)
    try:
        rev = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
    except Exception:
        rev = "unknown"
    # provenance + licence notice; no .txt/.md suffix, so it is not indexed
    lic = src.resolve().parents[1] / "LICENSE"
    (out / "SOURCE").write_text(
        "Built by rag/tools/build_docs_corpus.py from the ExaGO repository's docs/web/*.md\n"
        f"ExaGO commit: {rev}\nFiles (blocks): " + ", ".join(f"{k} ({v})" for k, v in stats.items()) + "\n"
        "ExaGO is distributed under the licence in LICENSE.ExaGO (copied alongside); this\n"
        "corpus is a reformatted copy of its documentation and keeps that notice.\n")
    if lic.exists():
        (out / "LICENSE.ExaGO").write_text(lic.read_text(encoding="utf-8", errors="replace"))
    return {"commit": rev, "files": stats, "blocks": sum(stats.values())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="../docs/web", help="ExaGO docs/web directory")
    ap.add_argument("--out", default="rag/corpus_docs")
    ap.add_argument("--max-chars", type=int, default=2800,
                    help="block size cap (below ingest's 3000-character paragraph limit)")
    a = ap.parse_args(argv)
    res = build(Path(a.src), Path(a.out), a.max_chars)
    print(f"Wrote {res['blocks']} blocks from {len(res['files'])} files to {a.out} (ExaGO commit {res['commit'][:12]}).")
    for k, v in res["files"].items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
