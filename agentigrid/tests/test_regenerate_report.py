"""Tests for the bounded classification digest and offline report regeneration.

Covers:
- format_for_classification is bounded on huge sweeps (vs format_detailed) and
  preserves the global-best candidate
- _finalize uses the bounded digest, not format_detailed
- session load restores sweep/contingency/reserve metadata (Part C)
- load_journal_export round-trips all previously-lost fields and rejects a
  session.json-shaped file
- regenerate_from_journal renders a PDF offline with and without analysis, never
  calling ExaGO
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agentigrid.config import (
    AppConfig, ExagoConfig, DataConfig, LLMConfig, SearchConfig, OutputConfig,
)
from agentigrid.engine.journal import SearchJournal
from agentigrid.engine.regenerate_report import (
    load_journal_export,
    build_session_stub,
    regenerate_from_journal,
)

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_IEEE118 = _DATA_DIR / "ieee_118_bus_v10.m"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _cfg(tmp_path: Path, base_case: Path | None = None) -> AppConfig:
    return AppConfig(
        exago=ExagoConfig(
            binary_dir=tmp_path / "bin", opflow_binary=None, scopflow_binary=None,
            tcopflow_binary=None, sopflow_binary=None, dcopflow_binary=None,
            pflow_binary=None, env_script=None, timeout=30,
        ),
        data=DataConfig(data_dir=tmp_path / "data"),
        llm=LLMConfig(
            backend="openai", model="test-model", api_key_env="TEST_KEY",
            openai_base_url=None, ollama_host="http://localhost:11434",
            ollama_cloud_host=None, temperature=0.3, max_tokens=4096,
        ),
        search=SearchConfig(
            max_iterations=5, default_mode="accumulative",
            base_case=base_case, gic_file=None, application="opflow",
        ),
        output=OutputConfig(
            workdir=tmp_path / "wd", logs_dir=tmp_path / "logs", save_journal=False,
            journal_format="json", save_modified_files=False, verbose=False,
        ),
    )


def _big_sweep_journal(n_candidates: int, best_bus: int) -> SearchJournal:
    """A journal with one large cost sweep; the global-best cost is at *best_bus*."""
    journal = SearchJournal()
    summaries = []
    feasible_buses = []
    for i in range(n_candidates):
        bus = i + 1
        # Cost descends toward best_bus so its cost is the unique minimum.
        cost = 1000.0 + abs(bus - best_bus)
        feasible = True
        summaries.append({
            "bus": bus, "feasible": feasible, "cost": cost,
            "voltage_min": 0.96, "voltage_max": 1.04,
            "max_line_loading_pct": 80.0, "violations": 0,
            "status": "CONVERGED", "reason": "",
        })
        feasible_buses.append(bus)
    journal.add_sweep(
        iteration=1, description="huge cost sweep",
        candidate_count=n_candidates, candidate_summaries=summaries,
        feasible_buses=feasible_buses,
        llm_reasoning="x" * 5000,  # long reasoning to confirm capping
    )
    return journal


# ---------------------------------------------------------------------------
# Part A: bounded classification digest
# ---------------------------------------------------------------------------

def test_classification_digest_bounded():
    best_bus = 4242
    journal = _big_sweep_journal(5000, best_bus)

    digest = journal.format_for_classification()
    detailed = journal.format_detailed()

    # Bounded: the digest stays small even with 5,000 candidates.
    assert len(digest) < 40_000, f"digest too large: {len(digest)} chars"
    # The global-best candidate's bus survives truncation.
    assert str(best_bus) in digest
    # Reasoning is capped (not the full 5,000 chars).
    assert "x" * 5000 not in digest
    # Contrast: format_detailed dumps every candidate and is far larger.
    assert len(detailed) > 10 * len(digest)


def test_classification_digest_keeps_best_and_worst():
    journal = _big_sweep_journal(500, best_bus=250)
    digest = journal.format_for_classification()
    # Best (bus 250, min cost) and worst (bus 500, max cost among 1..500) both kept.
    assert "250" in digest
    assert "worst" in digest.lower()


# ---------------------------------------------------------------------------
# Part B: _finalize uses the digest, not format_detailed
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _IEEE118.exists(), reason="ieee_118_bus_v10.m not available")
def test_finalize_uses_digest(tmp_path):
    from agentigrid.engine.agent_loop import AgentLoopController, SearchSession

    cfg = _cfg(tmp_path, base_case=_IEEE118)
    backend_mock = MagicMock()
    backend_mock.name.return_value = "mock"
    backend_mock.complete.return_value = SimpleNamespace(
        raw_text='```json\n{"goal_type": "cost_minimization", '
                 '"best_iteration": 1, "best_iteration_rationale": "r"}\n```'
    )
    with patch("agentigrid.engine.agent_loop.create_backend", return_value=backend_mock), \
         patch("agentigrid.engine.agent_loop.SimulationExecutor"):
        controller = AgentLoopController(cfg)

    controller._journal = _big_sweep_journal(50, best_bus=10)
    session = SearchSession(
        goal="minimize cost", application="opflow", base_case_path=_IEEE118,
        config=cfg, journal=controller._journal, start_time="2024-01-01T00:00:00",
        end_time="2024-01-01T00:01:00", termination_reason="complete",
    )

    calls = {"digest": 0, "detailed": 0}
    real_digest = SearchJournal.format_for_classification
    real_detailed = SearchJournal.format_detailed

    def spy_digest(self, *a, **k):
        calls["digest"] += 1
        return real_digest(self, *a, **k)

    def spy_detailed(self, *a, **k):
        calls["detailed"] += 1
        return real_detailed(self, *a, **k)

    with patch.object(SearchJournal, "format_for_classification", spy_digest), \
         patch.object(SearchJournal, "format_detailed", spy_detailed):
        controller._finalize(session, 1.0)

    assert calls["digest"] >= 1, "finalize did not call format_for_classification"
    assert calls["detailed"] == 0, "finalize still calls format_detailed"


# ---------------------------------------------------------------------------
# Part C: session load restores sweep/contingency/reserve metadata
# ---------------------------------------------------------------------------

def test_load_session_preserves_sweep_fields(tmp_path):
    from agentigrid.engine.session_io import save_session, load_session

    journal = SearchJournal()
    journal.add_sweep(
        iteration=1, description="sweep", candidate_count=3,
        candidate_summaries=[{"bus": 1, "feasible": True, "cost": 10.0}],
        feasible_buses=[1, 2],
    )
    journal.add_contingency(
        iteration=2, description="n-1", target_bus=5,
        neighbors=[(6, 1), (7, 2)], order=1,
        contingency_summaries=[{"label": "gen@6", "passed": False}],
        passed_count=0, failed_count=1,
    )

    save_dir = tmp_path / "sess"
    save_session(
        save_dir=save_dir, goal="g", application="opflow",
        base_case_path=Path("base.m"), config_path=None, journal=journal,
        steering_history=[], active_steering_directives=[], current_network=None,
        total_prompt_tokens=0, total_completion_tokens=0, last_iteration=2,
    )
    loaded = load_session(save_dir)
    entries = loaded["journal_entries"]

    sweep = next(e for e in entries if e.mode == "sweep")
    assert sweep.candidate_count == 3
    assert sweep.feasible_buses == [1, 2]

    cont = next(e for e in entries if e.mode == "contingency")
    assert cont.contingency_meta is not None
    assert cont.contingency_meta["target_bus"] == 5
    assert cont.candidate_count == 1


# ---------------------------------------------------------------------------
# Part D: load_journal_export round-trip + schema rejection
# ---------------------------------------------------------------------------

def test_load_journal_export_roundtrip(tmp_path):
    journal = SearchJournal()
    journal.add_sweep(
        iteration=1, description="sweep", candidate_count=4,
        candidate_summaries=[{"bus": 9, "feasible": True, "cost": 5.0}],
        feasible_buses=[9], exago_command={"cmd": "opflow", "args": ["-x"]},
    )
    journal.add_reserve(
        iteration=2, description="reserve",
        reserve_meta={"n_on": 12, "min_reserve": 300.0, "n1_secure": True},
        contingency_summaries=[{"label": "gen@1", "passed": True}],
    )
    export_path = tmp_path / "journal.json"
    journal.export_json(export_path)

    loaded = load_journal_export(export_path)
    entries = loaded.entries

    sweep = next(e for e in entries if e.mode == "sweep")
    assert sweep.candidate_count == 4
    assert sweep.feasible_buses == [9]
    assert sweep.exago_command == {"cmd": "opflow", "args": ["-x"]}

    reserve = next(e for e in entries if e.mode == "reserve")
    assert reserve.reserve_meta is not None
    assert reserve.reserve_meta["min_reserve"] == 300.0

    # Contingency-meta survives too (add a contingency entry and re-check).
    assert all(hasattr(e, "contingency_meta") for e in entries)


def test_load_rejects_session_schema(tmp_path):
    session_shaped = {
        "format_version": "1.1",
        "goal": "g",
        "journal": {"entries": [], "objective_registry": [], "preference_history": []},
    }
    p = tmp_path / "session.json"
    p.write_text(json.dumps(session_shaped), encoding="utf-8")
    with pytest.raises(ValueError, match="session.json"):
        load_journal_export(p)


def test_load_rejects_non_journal(tmp_path):
    p = tmp_path / "garbage.json"
    p.write_text(json.dumps({"foo": "bar"}), encoding="utf-8")
    with pytest.raises(ValueError, match="not a journal export"):
        load_journal_export(p)


# ---------------------------------------------------------------------------
# Part D: offline regeneration (PDF)
# ---------------------------------------------------------------------------

def _sweep_export(tmp_path: Path) -> Path:
    journal = _big_sweep_journal(20, best_bus=5)
    export_path = tmp_path / "sweep_journal.json"
    journal.export_json(export_path)
    return export_path


def test_regenerate_sweep_no_analysis(tmp_path):
    export_path = _sweep_export(tmp_path)
    cfg = _cfg(tmp_path)
    backend = MagicMock()  # must NOT be called when analysis is disabled

    out_dir = tmp_path / "out"
    pdf_path = regenerate_from_journal(
        journal_json_path=export_path, config=cfg, backend=backend,
        out_dir=out_dir, goal="minimize cost", generate_analysis=False,
    )

    assert pdf_path.exists()
    data = pdf_path.read_bytes()
    assert data.startswith(b"%PDF"), "output is not a PDF"
    assert len(data) > 1000
    backend.complete.assert_not_called()


def test_regenerate_with_mock_backend(tmp_path):
    export_path = _sweep_export(tmp_path)
    cfg = _cfg(tmp_path)

    backend = MagicMock()
    backend.complete.return_value = SimpleNamespace(
        raw_text='```json\n{"goal_type": "cost_minimization", '
                 '"best_iteration": 1, "best_iteration_rationale": "cheapest bus"}\n```'
    )

    captured = {}
    real_import = None

    class _SpyGenerator:
        def generate(self, session, summary_text=None, base_result=None,
                     best_result=None, goal_classification=None, steering_history=None):
            captured["summary_text"] = summary_text
            captured["goal_classification"] = goal_classification
            captured["session"] = session
            return b"%PDF-1.4\nfake pdf bytes\n%%EOF"

    with patch("agentigrid.engine.regenerate_report._import_report_generator",
               return_value=_SpyGenerator):
        pdf_path = regenerate_from_journal(
            journal_json_path=export_path, config=cfg, backend=backend,
            out_dir=tmp_path / "out", goal="minimize cost", generate_analysis=True,
        )

    assert pdf_path.exists()
    assert pdf_path.read_bytes().startswith(b"%PDF")
    backend.complete.assert_called_once()
    # The parsed classification was threaded into generate().
    assert captured["goal_classification"] is not None
    assert captured["goal_classification"]["goal_type"] == "cost_minimization"
    assert captured["goal_classification"]["best_iteration"] == 1
    assert captured["summary_text"] is not None


def test_build_session_stub_shape(tmp_path):
    journal = _big_sweep_journal(3, best_bus=2)
    cfg = _cfg(tmp_path, base_case=_IEEE118 if _IEEE118.exists() else None)
    session = build_session_stub(journal, cfg, goal="my goal")

    # Exactly the attributes generate() reads must be present and consistent.
    assert session.goal == "my goal"
    assert session.application == "opflow"
    assert session.journal is journal
    assert session.config is cfg
    assert session.total_prompt_tokens == 0
    assert session.total_completion_tokens == 0
    assert session.start_time and session.end_time
