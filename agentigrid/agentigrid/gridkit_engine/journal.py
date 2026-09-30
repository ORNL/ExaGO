"""Journal of a GridKit session: one entry per iteration, saved as JSON."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from agentigrid.gridkit_engine.sweep_metrics import FaultOutcome


@dataclass
class JournalEntry:
    iteration: int
    action: str                          # base_case | fault_screen | complete | error
    description: str = ""
    request: dict = field(default_factory=dict)
    verdict: Optional[str] = None        # first line of the result (VERDICT: ...)
    result_text: str = ""                # what the LLM was shown
    rows: list[dict] = field(default_factory=list)   # one per run
    elapsed_s: float = 0.0
    llm_reasoning: str = ""
    run_dir: Optional[str] = None
    findings: Optional[dict] = None      # complete: the LLM's full answer
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


def outcome_row(o: FaultOutcome, run_dir: Path | None = None) -> dict:
    return {
        "bus": o.bus,
        "status": o.status,
        "solved": o.run.solved,
        "run": o.run.reason,
        "checks": {
            c.name: {"status": c.status, "value": c.value, "limit": c.limit, "detail": c.detail}
            for c in o.checks
        },
        "run_dir": str(run_dir) if run_dir else None,
    }


class GridkitJournal:
    def __init__(self, goal: str, case_file: Path, session_dir: Path):
        self.goal = goal
        self.case_file = case_file
        self.session_dir = session_dir
        self.entries: list[JournalEntry] = []

    def add(self, entry: JournalEntry) -> JournalEntry:
        self.entries.append(entry)
        return entry

    def digest(self) -> str:
        """Earlier iterations, one or two lines each, for the next prompt."""
        lines = []
        for e in self.entries:
            head = f"[Iter {e.iteration}] {e.action}"
            if e.description:
                head += f": {e.description}"
            lines.append(head)
            if e.verdict:
                lines.append(f"    {e.verdict}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "tool": "gridkit",
            "goal": self.goal,
            "case_file": str(self.case_file),
            "session_dir": str(self.session_dir),
            "element_map": str(self.session_dir / "element_map.json"),
            "entries": [asdict(e) for e in self.entries],
        }

    def export_json(self, path: Path | None = None) -> Path:
        path = path or self.session_dir / "journal.json"
        path.write_text(json.dumps(self.to_dict(), indent=1, default=str), encoding="utf-8")
        return path
