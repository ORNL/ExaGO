#!/usr/bin/env python3
"""Corpus guard: hold-out leakage audit + freeze manifest for the RAG corpus.

When retrieval is evaluated on specific (network, goal) tasks, a corpus that
contains worked solutions to those same tasks measures lookup of the answer
rather than grounding. This tool audits every corpus chunk against one or more
evaluation specs (JSON with "goals": [{"text": ...}] and "cases":
[{"path"|"network": ...}]) and flags:

  * goal-leak      — a chunk contains the text of an evaluated goal;
  * eval-exemplar  — a worked example (proposal / correct response / correct
                     specification) that names an evaluated network;
  * personal-path  — an absolute home / Windows user path (machine-specific
                     noise, and it must not reach a public repo).

Plain facts about an evaluated network (e.g. its bus count in the case metadata)
are allowed: they are documentation, not a solution.

`--freeze` writes corpus_manifest.json (per-file SHA-256 + an overall corpus
hash) so every run can record exactly which corpus it was grounded on.

    python rag/tools/corpus_guard.py rag/corpus                         # personal paths only
    python rag/tools/corpus_guard.py --spec my_eval_spec.json rag/corpus
    python rag/tools/corpus_guard.py --spec my_eval_spec.json rag/corpus --freeze
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

CORPUS_SUFFIXES = (".txt", ".md")
MANIFEST_NAME = "corpus_manifest.json"
_EXEMPLAR_MARKERS = ("proposal", "correct response", "correct specification",
                     "correct action", "commands actually run")
_PERSONAL_PATH = re.compile(r"(/home/[A-Za-z0-9._-]+/|/Users/[A-Za-z0-9._-]+/|[A-Za-z]:\\Users\\|/mnt/c/Users/)")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def load_holdout(spec_paths: list[Path]) -> dict:
    """Evaluated goal texts and network stems from evaluation spec(s)."""
    goals, networks = set(), set()
    for sp in spec_paths:
        spec = json.loads(Path(sp).read_text())
        for g in spec.get("goals", []):
            if g.get("text"):
                goals.add(_norm(g["text"]))
        for c in spec.get("cases", []):
            if c.get("path"):
                networks.add(Path(c["path"]).stem.lower())
            if c.get("network"):
                networks.add(str(c["network"]).lower())
    return {"goals": goals, "networks": networks}


def _network_pattern(stem: str) -> re.Pattern:
    """Match a network name as a whole token (case39 must not match case39x/case390).

    'case_ACTIVSg200' and the short form 'ACTIVSg200' both count."""
    names = {stem, stem.removeprefix("case_"), stem.removeprefix("case")}
    names = {n for n in names if len(n) >= 4}
    alt = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
    return re.compile(rf"(?<![A-Za-z0-9])(?:case_?)?(?:{alt})(?![A-Za-z0-9])", re.I)


def split_chunks(text: str) -> list[str]:
    """Paragraph-level chunks (blank-line separated), mirroring ingest's granularity."""
    return [c.strip() for c in re.split(r"\n\s*\n", text) if c.strip()]


def audit_text(text: str, holdout: dict) -> list[dict]:
    findings = []
    net_pats = {n: _network_pattern(n) for n in holdout["networks"]}
    for i, chunk in enumerate(split_chunks(text)):
        low = _norm(chunk)
        for g in holdout["goals"]:
            if g and g in low:
                findings.append({"chunk": i, "kind": "goal-leak", "detail": g})
        if any(m in low for m in _EXEMPLAR_MARKERS):
            for n, pat in net_pats.items():
                if pat.search(chunk):
                    findings.append({"chunk": i, "kind": "eval-exemplar", "detail": n})
        m = _PERSONAL_PATH.search(chunk)
        if m:
            findings.append({"chunk": i, "kind": "personal-path", "detail": m.group(0)})
    return findings


def corpus_files(corpus_dir: Path) -> list[Path]:
    from agentigrid.rag.corpus_hash import corpus_files as _cf
    return _cf(corpus_dir)


def corpus_hash(corpus_dir: Path) -> tuple[str, dict[str, str]]:
    """Overall SHA-256 over (relative path, content hash) pairs (shared definition)."""
    from agentigrid.rag.corpus_hash import corpus_hash as _ch
    return _ch(corpus_dir)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus", nargs="?", default="rag/corpus")
    ap.add_argument("--spec", action="append", default=[],
                    help="evaluation spec whose goals/networks are held out (repeatable)")
    ap.add_argument("--freeze", action="store_true",
                    help="write corpus_manifest.json (refused if the audit finds problems)")
    args = ap.parse_args(argv)

    corpus_dir = Path(args.corpus)
    if not corpus_dir.is_dir():
        print(f"corpus dir not found: {corpus_dir}", file=sys.stderr)
        return 2
    holdout = load_holdout([Path(s) for s in args.spec]) if args.spec else {"goals": set(), "networks": set()}
    if not args.spec:
        print("warning: no --spec given; only personal paths are checked", file=sys.stderr)

    total = 0
    for p in corpus_files(corpus_dir):
        for f in audit_text(p.read_text(encoding="utf-8", errors="replace"), holdout):
            total += 1
            print(f"{p.relative_to(corpus_dir)}  chunk {f['chunk']:>3}  {f['kind']:<14} {f['detail']}")
    digest, per_file = corpus_hash(corpus_dir)
    print(f"\n{len(per_file)} file(s); {total} finding(s); corpus sha256 {digest}")

    if args.freeze:
        if total:
            print("Refusing to freeze: resolve the findings first.", file=sys.stderr)
            return 1
        (corpus_dir / MANIFEST_NAME).write_text(json.dumps({
            "corpus_sha256": digest, "files": per_file,
            "held_out_specs": args.spec, "frozen_at": datetime.now().isoformat(),
        }, indent=2) + "\n")
        print(f"Frozen: {corpus_dir / MANIFEST_NAME}")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
