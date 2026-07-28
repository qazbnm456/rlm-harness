"""Phase C (part 1) — reconstruct and replay a recorded run.

Replay reads the JSONL trace and rebuilds an ordered timeline. For deterministic
replay it serves *recorded* tool outputs rather than re-executing tools (which
may be non-deterministic or have side effects). This makes a past run inspectable
and step-through-able without touching the outside world.

Pure stdlib; no dspy import.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .trace import (
    EVENT_MAIN_STEP,
    EVENT_SUB_CALL,
    EVENT_TOOL_CALL,
    load_events,
)


@dataclass
class Timeline:
    run_id: str
    events: list[dict]

    @property
    def main_steps(self) -> list[dict]:
        return [e for e in self.events if e["type"] == EVENT_MAIN_STEP]

    @property
    def sub_calls(self) -> list[dict]:
        return [e for e in self.events if e["type"] == EVENT_SUB_CALL]

    @property
    def tool_calls(self) -> list[dict]:
        return [e for e in self.events if e["type"] == EVENT_TOOL_CALL]

    def summary(self) -> str:
        return (
            f"run {self.run_id}: {len(self.main_steps)} main steps, "
            f"{len(self.sub_calls)} sub calls, {len(self.tool_calls)} tool calls"
        )


def reconstruct(events: list[dict]) -> Timeline:
    """Build a :class:`Timeline` from an ordered event list (single run)."""
    if not events:
        return Timeline(run_id="", events=[])
    run_id = events[0].get("run_id", "")
    # Events are already in step order within a run; sort defensively by step_id.
    ordered = sorted(events, key=lambda e: e.get("step_id", 0))
    return Timeline(run_id=run_id, events=ordered)


def load_timeline(path: str, run_id: str) -> Timeline:
    """Load and reconstruct a single run's timeline from a trace file."""
    return reconstruct(load_events(path, run_id=run_id))


@dataclass
class RecordedToolProvider:
    """Serve recorded tool outputs in order, for deterministic replay.

    Matches each ``replay(tool, args)`` to the next recorded ``tool_call`` for
    that tool name. Raises if the recording is exhausted, so a replay that drifts
    from the original path fails loudly instead of silently re-executing.
    """

    timeline: Timeline
    _cursor: dict[str, int] = field(default_factory=dict)

    def replay(self, tool: str, args: dict | None = None) -> Any:
        calls = [e for e in self.timeline.tool_calls if e["payload"].get("tool") == tool]
        idx = self._cursor.get(tool, 0)
        if idx >= len(calls):
            raise LookupError(
                f"No recorded output #{idx} for tool {tool!r} (replay drifted "
                f"from the recording)."
            )
        self._cursor[tool] = idx + 1
        payload = calls[idx]["payload"]
        return payload.get("result", payload.get("result_len"))
