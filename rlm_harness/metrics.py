"""Trace utilization metrics — how a run's activity was distributed across the root LM's own
turns, tool calls, and sub-LM escalations. A sibling to ``rubric.py``'s "derive facts from a trace"
shape, but structurally different: ``rubric.criteria_facts`` slices a CALLER-SUPPLIED facts dict
against a caller-supplied lens; this module COMPUTES fixed counts/rates directly from the raw
``trace/v1`` event stream, with no caller input beyond the events themselves. Reward-free, like
every other trace-derived module here: raw counts and rates, never a score.

Reads ONLY already-frozen ``trace/v1`` fields (``event["type"]``, ``event["payload"]["tool"]``) —
no new event type, no new payload field, nothing the trace contract needs to change for. dspy-free,
stdlib only.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from .trace import EVENT_MAIN_STEP, EVENT_SUB_CALL, EVENT_TOOL_CALL, group_by_run


@dataclass(frozen=True)
class RunUtilization:
    """One run's activity, counted and rated. Both rates are denominated over ``main_steps`` (root
    LM turns) — "how many tool calls / sub-LM escalations happened per root-LM turn taken," the
    framing closest to how much of the agent's own activity routed through a given channel. This is
    a judgment call, not a uniquely correct answer: the raw counts are exposed alongside the rates,
    so a consumer wanting a different denominator can recompute one from the same fields.

    ``None`` (not ``0.0``) when ``main_steps == 0``: a run that never took a root-LM turn has no
    denominator, and ``0.0`` would misleadingly read as "measured and found to be zero usage"
    rather than "undefined." A crashed/cancelled run that failed before its first ``Prediction``
    ever returned is a real, reachable example — it can carry live-recorded ``tool_call``/
    ``sub_call`` events with zero ``main_step`` events (``RLMTask.arun()`` only records the main
    trajectory `if "prediction" in captured`), which is exactly the case this ``None`` distinguishes
    from a genuinely idle run.
    """

    main_steps: int
    tool_calls_total: int
    tool_calls_by_name: dict[str, int] = field(default_factory=dict)
    sub_calls_total: int = 0
    tool_call_rate: float | None = None
    sub_call_rate: float | None = None


def compute_run_utilization(events: Iterable[dict]) -> RunUtilization:
    """Pure function over ONE run's events (e.g. ``group_by_run(load_events(path))[run_id]``)."""
    main_steps = 0
    tool_calls_by_name: dict[str, int] = {}
    sub_calls_total = 0

    for event in events:
        etype = event.get("type")
        if etype == EVENT_MAIN_STEP:
            main_steps += 1
        elif etype == EVENT_TOOL_CALL:
            name = event.get("payload", {}).get("tool", "?")
            tool_calls_by_name[name] = tool_calls_by_name.get(name, 0) + 1
        elif etype == EVENT_SUB_CALL:
            sub_calls_total += 1

    tool_calls_total = sum(tool_calls_by_name.values())
    tool_call_rate = (tool_calls_total / main_steps) if main_steps else None
    sub_call_rate = (sub_calls_total / main_steps) if main_steps else None

    return RunUtilization(
        main_steps=main_steps,
        tool_calls_total=tool_calls_total,
        tool_calls_by_name=tool_calls_by_name,
        sub_calls_total=sub_calls_total,
        tool_call_rate=tool_call_rate,
        sub_call_rate=sub_call_rate,
    )


def compute_utilization_by_run(events: Iterable[dict]) -> dict[str, RunUtilization]:
    """Convenience: group ``events`` by ``run_id`` (reusing ``trace.group_by_run``) and compute
    each run's :class:`RunUtilization` in one call — a batch/dataset-level view across many runs."""
    return {
        run_id: compute_run_utilization(run_events)
        for run_id, run_events in group_by_run(events).items()
    }
