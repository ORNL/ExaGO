"""What every tool gives the router: a card for the system prompt, action handlers, checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from agentigrid.router_engine.state import StateStore


@dataclass
class Context:
    """What the router lends an adapter for one action."""
    store: StateStore
    session_dir: Path
    turn: int
    print: Callable[[str], None]


@dataclass
class ActionResult:
    text: Optional[str] = None            # shown to the LLM as LATEST RESULT
    verdict: Optional[str] = None         # one line, for the history and the summary
    feedback: Optional[str] = None        # error in the LLM's request (shown as ERROR)
    stop: Optional[str] = None            # the study must stop: reason
    passed: Optional[bool] = None         # for the overall verdict (None: not a check)
    record: dict = field(default_factory=dict)    # extra fields for the journal
    row: Optional[dict] = None            # one line of summary.txt
    attention: list[str] = field(default_factory=list)   # e.g. a difference near its limit


class ToolAdapter:
    name: str = ""                        # "exago", "hil", "gridkit"
    title: str = ""                       # e.g. "ExaGO (steady-state AC power flow)"
    synonyms: str = ""                    # words in a goal that mean this tool
    actions: tuple[str, ...] = ()

    def ready(self) -> Optional[str]:
        """None if the tool can be used, else the reason (inputs, binaries)."""
        return None

    def start(self, session_dir: Path, goal: str, quiet: bool) -> None:
        """Called once when the session starts, before the card is read."""

    def card(self) -> str:
        """The tool's part of the system prompt: what it does, its actions, its rules."""
        raise NotImplementedError

    def handle(self, action: str, data: dict, ctx: Context) -> ActionResult:
        raise NotImplementedError

    def pending(self, store: StateStore) -> Optional[str]:
        """Work this tool has been asked for in this session's pattern but not done yet
        (complete is refused once while there is any)."""
        return None

    def close(self) -> None:
        """Release devices and processes; called once at the end."""
