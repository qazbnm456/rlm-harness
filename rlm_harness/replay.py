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

    ``tool`` is matched against the trace's ``payload["tool"]`` — the RAW name (for MCP, the
    server's own name, e.g. ``get-weather``). A caller holding the sanitised REPL name the model
    typed (``get_weather``) will match nothing; the ``repl_name`` payload field carries that
    mapping when the two differ.
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
        # A tool_call carries its output under one of several keys — `record_tool_call` pins
        # none, and the kit's own tools disagree: MCP and read_skill use `preview`, web_search
        # uses `results`, the make_model_tool convention uses `raw`, list_skills uses `result`.
        # Reading only `result` meant THREE of the four shipped tool families replayed as `None`,
        # silently — measured, not theorised — while `dataset.py:_action_record` already read the
        # fallback. Two readers of one trace disagreeing is the bug; this aligns them.
        for key in ("raw", "result", "results"):
            if payload.get(key) is not None:
                return payload[key]
        # `preview` is deliberately NOT in that list. It is a TRUNCATED head of the output, so
        # serving it would hand a replay silently-wrong bytes — worse than failing. Raise the
        # same loud error this class already uses for drift, since a replay that cannot be
        # served faithfully should stop, not improvise.
        if payload.get("preview") is not None:
            raise LookupError(
                f"Recorded output #{idx} for tool {tool!r} exists only as `preview`, which is a "
                f"TRUNCATED head of the real output (MCP tools and read_skill record this way). "
                f"Replaying it would serve wrong bytes; re-record with the full output under "
                f"`raw`/`result` if this call needs to be replayable."
            )
        return payload.get("result_len")
