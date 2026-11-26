#!/usr/bin/env python3
"""Tier-1 RAG corpus harvester for AgentiGrid.

Auto-generates provenance-tagged corpus chunks from GROUND TRUTH — the ExaGO
binaries themselves and the MATPOWER case files — so nothing is hand-authored
or invented. Every chunk is prefixed with a [source] tag, so when the retriever
surfaces it the provenance travels with the text (the corpus stays auditable).

Two harvesters:
  * app --help / --version  -> rag/corpus/exago_<app>_help.txt
  * .m case-file metadata    -> rag/corpus/exago_cases.txt

Run from the AgentiGrid project root, then re-ingest:
    python rag_tools/rag_harvest.py
    rm -rf rag/store && python -m agentigrid.rag.ingest rag/corpus

Nothing here calls the LLM or the network. Safe to run repeatedly (idempotent:
it overwrites its own output files, leaving hand-written corpus files alone).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import re
import subprocess
import sys
from pathlib import Path

DEFAULT_APPS = ["opflow", "scopflow", "tcopflow", "sopflow", "dcopflow", "pflow"]

# Files this script writes. Used so a re-run only clobbers its own output and
# never a curated / hand-written corpus file.
_GENERATED_PREFIX = "exago_"

_TODAY = _dt.date.today().isoformat()


def _tag(src: str) -> str:
    """Compact provenance tag placed at the head of every chunk."""
    return f"[source: {src} | harvested: {_TODAY} | auto-generated, review before trusting]"


def _paragraphs(text: str) -> list[str]:
    """Split raw text into non-trivial paragraphs (blank-line separated)."""
    blocks = re.split(r"\n\s*\n", text)
    out = []
    for b in blocks:
        b = b.rstrip()
        # keep anything with at least a couple of word characters
        if len(re.sub(r"\W", "", b)) >= 3:
            out.append(b)
    return out


# ---------------------------------------------------------------------------
# Binary discovery + --help harvest
# ---------------------------------------------------------------------------

def _resolve_binary(bin_dir: Path, app: str) -> Path | None:
    cand = bin_dir / app
    if cand.exists():
        return cand.resolve()
    return None


def _run(cmd: list[str], timeout: int) -> tuple[int, str]:
    """Run a command, return (returncode, combined stdout+stderr). Never raises."""
    try:
        p = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except FileNotFoundError:
        return 127, "binary not found"
    except subprocess.TimeoutExpired:
        return 124, "timed out"
    except Exception as e:  # pragma: no cover - defensive
        return 1, f"error: {e}"


def harvest_help(bin_dir: Path, out_dir: Path, apps: list[str], timeout: int) -> list[str]:
    written = []
    for app in apps:
        binary = _resolve_binary(bin_dir, app)
        if binary is None:
            print(f"  [skip] {app}: no binary at {bin_dir/app}", file=sys.stderr)
            continue

        # ExaGO help usually prints on --help; some options print to stderr.
        rc, help_text = _run([str(binary), "--help"], timeout)
        if not help_text.strip() or rc == 127:
            print(f"  [skip] {app}: --help produced nothing (rc={rc})", file=sys.stderr)
            continue

        vrc, vtext = _run([str(binary), "--version"], timeout)
        version = vtext.strip().splitlines()[0] if (vrc == 0 and vtext.strip()) else "unknown"

        # basename only: the absolute path is machine-specific and must not enter the corpus
        src = f"exago {app} --help | binary: {Path(str(binary)).name} | version: {version}"
        chunks = [f"{_tag(src)}\n{para}" for para in _paragraphs(help_text)]
        if not chunks:
            continue

        # A leading one-line summary chunk so a query like "how do I run scopflow"
        # retrieves the app's identity even before matching a specific option.
        summary = (
            f"{_tag(src)}\n{app}: ExaGO application. Version {version}. "
            f"Invoked as `{app} <options>`. See the option chunks below for flags."
        )
        chunks.insert(0, summary)

        out_path = out_dir / f"{_GENERATED_PREFIX}{app}_help.txt"
        out_path.write_text("\n\n".join(chunks) + "\n", encoding="utf-8")
        written.append(out_path.name)
        print(f"  [ok]   {app}: {len(chunks)} chunks -> {out_path.name}")
    return written


# ---------------------------------------------------------------------------
# MATPOWER .m case-file metadata harvest
# ---------------------------------------------------------------------------

def _count_matpower_block(text: str, name: str) -> int | None:
    """Count data rows in `mpc.<name> = [ ... ];`. Returns None if not present."""
    m = re.search(rf"mpc\.{name}\s*=\s*\[(.*?)\]\s*;", text, re.DOTALL)
    if not m:
        return None
    rows = 0
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        if re.search(r"\d", line):
            rows += 1
    return rows


def harvest_cases(data_dirs: list[Path], out_dir: Path) -> list[str]:
    entries = []
    seen = set()
    for d in data_dirs:
        if not d.exists():
            continue
        for m_file in sorted(d.rglob("*.m")):
            if m_file.name in seen:
                continue
            try:
                text = m_file.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            buses = _count_matpower_block(text, "bus")
            gens = _count_matpower_block(text, "gen")
            branches = _count_matpower_block(text, "branch")
            if buses is None and gens is None and branches is None:
                continue  # not a MATPOWER case
            seen.add(m_file.name)
            parts = []
            if buses is not None:
                parts.append(f"{buses} buses")
            if gens is not None:
                parts.append(f"{gens} generators")
            if branches is not None:
                parts.append(f"{branches} branches")
            src = f"case file: {m_file}"
            entries.append(
                f"{_tag(src)}\n"
                f"{m_file.name}: {', '.join(parts)}. "
                f"MATPOWER-format network case usable with ExaGO OPF apps "
                f"(pass via the app's network/case option)."
            )
    if not entries:
        print("  [skip] cases: no MATPOWER .m files found", file=sys.stderr)
        return []
    out_path = out_dir / f"{_GENERATED_PREFIX}cases.txt"
    out_path.write_text("\n\n".join(entries) + "\n", encoding="utf-8")
    print(f"  [ok]   cases: {len(entries)} case files -> {out_path.name}")
    return [out_path.name]


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin-dir", default="applications/exago",
                    help="Directory holding the ExaGO app binaries/symlinks (default: ./applications/exago)")
    ap.add_argument("--data-dir", action="append", default=None,
                    help="Directory to scan for .m case files (repeatable). "
                         "Default: ./data/exago/datafiles and ./data/exago/examples")
    ap.add_argument("--out", default="rag/corpus",
                    help="Corpus output directory (default: rag/corpus)")
    ap.add_argument("--apps", nargs="*", default=DEFAULT_APPS,
                    help=f"Apps to harvest help from (default: {' '.join(DEFAULT_APPS)})")
    ap.add_argument("--timeout", type=int, default=20, help="Per-command timeout seconds")
    ap.add_argument("--no-help", action="store_true", help="Skip the --help harvest")
    ap.add_argument("--no-cases", action="store_true", help="Skip the case-file harvest")
    args = ap.parse_args()

    bin_dir = Path(args.bin_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dirs = [Path(p) for p in (args.data_dir or ["data/exago/datafiles", "data/exago/examples"])]

    print(f"Harvesting into {out_dir}/ (provenance-tagged, safe to re-run)")
    written = []
    if not args.no_help:
        # Fallback: if ./applications is empty, hint at the build dir.
        if not any(bin_dir.glob("*")):
            print(f"  note: {bin_dir}/ looks empty. Point --bin-dir at your "
                  f"ExaGO build/applications, or run from the project root.",
                  file=sys.stderr)
        written += harvest_help(bin_dir, out_dir, args.apps, args.timeout)
    if not args.no_cases:
        written += harvest_cases(data_dirs, out_dir)

    if not written:
        print("Nothing harvested. Check --bin-dir / --data-dir.", file=sys.stderr)
        return 1
    print(f"\nDone. {len(written)} corpus file(s) written. Re-ingest with:")
    print("  rm -rf rag/store && python -m agentigrid.rag.ingest rag/corpus")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
