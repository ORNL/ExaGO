"""Corpus tools: schema exemplars, hold-out / personal-path guard, frozen-corpus identity."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "rag" / "corpus"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "rag" / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


schema = _load("rag_schema_exemplars")
guard = _load("corpus_guard")


# --- schema exemplars -----------------------------------------------------------

def test_schema_exemplars_certify_on_case118():
    chunks = schema.build(ROOT.parent / "datafiles" / "case118.m")
    assert len(chunks) == len(schema.EXEMPLARS)
    applied = sum("applied on held-out case118" in c for c in chunks)
    assert applied >= 12
    # the one command without a host branch in any held-out case is labelled honestly
    assert any("set_phase_shift_angle" in c and "schema-validated only" in c for c in chunks)
    for ex in schema.EXEMPLARS:
        assert ex["response"]["action"] in schema.TOP_LEVEL_ACTIONS


def test_schema_rejects_top_level_command():
    bad = {"request": "x", "apps": ["opflow"],
           "response": {"action": "set_all_bus_vlimits", "Vmin": 0.9, "Vmax": 1.1}}
    with pytest.raises(ValueError):
        schema.certify(bad, None)


# --- guard ------------------------------------------------------------------------

def test_guard_flags():
    hold = {"goals": {guard._norm("Reduce total generation cost by 10%")},
            "networks": {"case_activsg200", "case39"}}
    text = ("Request: Reduce total generation cost by 10%\n\n"
            "Correct specification: application scopflow on case_ACTIVSg200\n\n"
            "Case facts: case_ACTIVSg200 has 200 buses\n\n"
            "Proposal ran /home/someone/x.m\n\n"
            "Correct response on case390 is fine")
    kinds = [f["kind"] for f in guard.audit_text(text, hold)]
    assert kinds.count("goal-leak") == 1
    assert kinds.count("eval-exemplar") == 1        # facts-only chunk and case390 are not flagged
    assert kinds.count("personal-path") == 1


def test_freeze_refuses_on_findings_and_hashes(tmp_path):
    (tmp_path / "a.txt").write_text("clean fact\n")
    assert guard.main([str(tmp_path), "--freeze"]) == 0
    m = json.loads((tmp_path / guard.MANIFEST_NAME).read_text())
    assert len(m["corpus_sha256"]) == 64 and list(m["files"]) == ["a.txt"]
    (tmp_path / "b.txt").write_text("path /home/someone/x\n")
    assert guard.main([str(tmp_path), "--freeze"]) == 1


def test_journal_roundtrips_skipped_commands(tmp_path):
    from agentigrid.engine.journal import JournalEntry, SearchJournal
    j = SearchJournal()
    e = JournalEntry(iteration=1, description="d", commands=[], objective_value=1.0, feasible=True,
                     convergence_status="CONVERGED", violations_count=0, voltage_min=1.0, voltage_max=1.0,
                     max_line_loading_pct=0.0, total_gen_mw=0.0, total_load_mw=0.0, llm_reasoning="",
                     mode="fresh", elapsed_seconds=0.0)
    assert e.skipped_commands is None
    e.skipped_commands = ["Skipped x"]
    j.add_entry(e)
    out = tmp_path / "j.json"
    j.export_json(out)
    assert json.loads(out.read_text())["entries"][0]["skipped_commands"] == ["Skipped x"]


# --- frozen-corpus identity -----------------------------------------------------

def test_corpus_status_tracks_freeze_and_ingest(tmp_path):
    from agentigrid.rag.corpus_hash import CHUNKING, INGEST_MANIFEST_NAME, corpus_hash, corpus_status
    corpus, store = tmp_path / "corpus", tmp_path / "store"
    corpus.mkdir(); store.mkdir()
    (corpus / "a.txt").write_text("alpha")
    assert "not frozen" in corpus_status(corpus, store)["reason"]
    assert guard.main([str(corpus), "--freeze"]) == 0
    assert "store not built" in corpus_status(corpus, store)["reason"]
    digest, _ = corpus_hash(corpus)
    (store / INGEST_MANIFEST_NAME).write_text(json.dumps({"corpus_sha256": digest, "chunking": dict(CHUNKING)}))
    st = corpus_status(corpus, store)
    assert st["ok"] and st["corpus_sha256"] == digest
    (corpus / "a.txt").write_text("alpha changed")
    assert "changed since it was frozen" in corpus_status(corpus, store)["reason"]



def test_shipped_corpus_matches_its_manifest_and_is_clean():
    """The committed corpus is exactly the frozen one, with no personal paths."""
    from agentigrid.rag.corpus_hash import MANIFEST_NAME, corpus_hash
    manifest = json.loads((CORPUS / MANIFEST_NAME).read_text())
    digest, per_file = corpus_hash(CORPUS)
    assert digest == manifest["corpus_sha256"]
    assert per_file == manifest["files"]
    no_holdout = {"goals": set(), "networks": set()}
    for f in guard.corpus_files(CORPUS):
        assert guard.audit_text(f.read_text(encoding="utf-8"), no_holdout) == [], f.name
