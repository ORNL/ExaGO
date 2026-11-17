"""Journal of a HIL session: one entry per iteration, saved as JSON."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class JournalEntry:
    iteration: int
    action: str                          # base_state | steady_state | close_tie | complete | error
    description: str = ""
    request: dict = field(default_factory=dict)
    verdict: Optional[str] = None
    result_text: str = ""                # what the LLM was shown
    commands: list[dict] = field(default_factory=list)   # every HIL switch command
    readings: dict = field(default_factory=dict)          # radial / closed bus readings
    comparisons: list[dict] = field(default_factory=list)
    exago: dict = field(default_factory=dict)             # ExaGO solutions used
    elapsed_s: float = 0.0
    findings: Optional[dict] = None      # complete: the LLM's full answer
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


class HilJournal:
    def __init__(self, goal: str, model: Path, session_dir: Path):
        self.goal = goal
        self.model = model
        self.session_dir = session_dir
        self.entries: list[JournalEntry] = []
        self.token_usage: dict | None = None

    def add(self, entry: JournalEntry) -> JournalEntry:
        self.entries.append(entry)
        return entry

    def digest(self) -> str:
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
            "tool": "hil",
            "goal": self.goal,
            "model": str(self.model),
            "session_dir": str(self.session_dir),
            "entries": [asdict(e) for e in self.entries],
            **({"token_usage": self.token_usage} if self.token_usage is not None else {}),
        }

    def export_json(self, path: Path | None = None) -> Path:
        path = path or self.session_dir / "journal.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1, default=str), encoding="utf-8")
        return path
