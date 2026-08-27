"""Trace utilization metrics — how a run's activity was distributed across the root LM's own
turns, tool calls, and sub-LM escalations. A sibling to ``rubric.py``'s "derive facts from a trace"
shape, but structurally different: ``rubric.criteria_facts`` slices a CALLER-SUPPLIED facts dict
against a caller-supplied lens; this module COMPUTES fixed counts/rates directly from the raw
``trace/v1`` event stream, with no caller input beyond the events themselves. Reward-free, like
every other trace-derived module here: raw counts and rates, never a score.

Reads only already-frozen ``trace/v1`` fields — ``event["type"]``, ``payload["tool"]``, the
optional ``payload["duration_s"]``, and (through :func:`rlm_harness.trace.payload_cause`, never
directly) ``circuit_broken`` / ``endpoint_error`` / ``error`` / ``ok``. No new event type, nothing
the trace contract needs to change for. dspy-free, stdlib only.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from .trace import (
    CAUSE_CIRCUIT_BROKEN,
    CAUSE_ENDPOINT,
    CAUSE_INVALID,
    CAUSE_OK,
    EVENT_MAIN_STEP,
    EVENT_SUB_CALL,
    EVENT_TOOL_CALL,
    group_by_run,
    payload_cause,
)


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


@dataclass(frozen=True)
class ToolWaste:
    """One tool's calls, split by OUTCOME and by the wall-clock each outcome cost.

    The question this answers: how much of a run's time went into tool calls that produced nothing
    usable? On the corpus that motivated it, 57% of all tool wall-clock produced output the
    consumer's own validator rejected — a number nobody could see, because a trace carried no
    durations and ``ok`` alone cannot tell a rejection from an endpoint failure.

    Outcomes come from :func:`rlm_harness.trace.payload_cause`, never from raw ``ok``: ``ok`` is
    frequently ABSENT on an endpoint-failure payload, so ``payload.get("ok")`` reads ``None`` and
    every naive counter absorbs infrastructure failures as content declines. That mistake has
    shipped four times.

    ``*_seconds`` are ``None``, not ``0.0``, when the events carry no ``duration_s`` — i.e. a trace
    written before 1.6.0, or a tool that does not measure. ``None`` means "not recorded"; ``0.0``
    would read as "measured and found to be free". Deliberately NOT inferred from the gaps between
    events: that charges a whole turn's model generation to the turn's first tool call, which is
    the exact error this class exists to stop people making.

    Reward-free, like every other module here: counts, seconds and rates — never a score.
    """

    tool: str
    calls: int
    invalid: int = 0            # the validator ran and rejected
    endpoint_errors: int = 0    # the call itself failed; the validator never ran
    circuit_broken: int = 0     # short-circuited; nothing was called at all
    ok: int = 0
    total_seconds: float | None = None
    wasted_seconds: float | None = None   # spent on invalid + endpoint outcomes
    measured_calls: int = 0     # how many of `calls` carried a duration

    @property
    def invalid_rate(self) -> float | None:
        """Declines as a share of calls that actually reached the validator.

        Denominated over ``invalid + ok`` rather than ``calls``: a circuit-broken call never ran a
        validator and an endpoint failure never got an output to judge, so counting either in the
        denominator understates how often the model's output was actually rejected. ``None`` when
        the validator never ran — undefined, not zero.
        """
        judged = self.invalid + self.ok
        return (self.invalid / judged) if judged else None

    @property
    def wasted_share(self) -> float | None:
        """``wasted_seconds`` over ``total_seconds``.

        ``None`` when nothing was measured — and also when everything measured summed to exactly
        zero, which is indistinguishable from unmeasured here and would otherwise divide by zero.
        """
        if self.total_seconds is None or not self.total_seconds:
            return None
        return (self.wasted_seconds or 0.0) / self.total_seconds


def compute_tool_waste(events: Iterable[dict]) -> dict[str, ToolWaste]:
    """Per-tool outcome and cost breakdown for one run's events, keyed by tool name.

    Pure function over ``trace/v1`` events, same shape as :func:`compute_run_utilization` — pass
    one run's events, or use :func:`compute_tool_waste_by_run` across many.
    """
    calls: dict[str, int] = {}
    by_cause: dict[str, dict[str, int]] = {}
    secs: dict[str, float] = {}
    wasted: dict[str, float] = {}
    measured: dict[str, int] = {}

    for event in events:
        if event.get("type") != EVENT_TOOL_CALL:
            continue
        payload = event.get("payload") or {}
        name = payload.get("tool", "?")
        calls[name] = calls.get(name, 0) + 1
        cause = payload_cause(payload)
        by_cause.setdefault(name, {})[cause] = by_cause.setdefault(name, {}).get(cause, 0) + 1

        duration = payload.get("duration_s")
        if isinstance(duration, (int, float)) and not isinstance(duration, bool):
            secs[name] = secs.get(name, 0.0) + float(duration)
            measured[name] = measured.get(name, 0) + 1
            if cause in (CAUSE_INVALID, CAUSE_ENDPOINT):
                wasted[name] = wasted.get(name, 0.0) + float(duration)

    out: dict[str, ToolWaste] = {}
    for name, n in calls.items():
        causes = by_cause.get(name, {})
        has_time = name in measured
        out[name] = ToolWaste(
            tool=name,
            calls=n,
            invalid=causes.get(CAUSE_INVALID, 0),
            endpoint_errors=causes.get(CAUSE_ENDPOINT, 0),
            circuit_broken=causes.get(CAUSE_CIRCUIT_BROKEN, 0),
            ok=causes.get(CAUSE_OK, 0),
            total_seconds=secs.get(name) if has_time else None,
            wasted_seconds=wasted.get(name, 0.0) if has_time else None,
            measured_calls=measured.get(name, 0),
        )
    return out


def compute_tool_waste_by_run(events: Iterable[dict]) -> dict[str, dict[str, ToolWaste]]:
    """Convenience: group ``events`` by ``run_id`` and compute each run's tool breakdown."""
    return {
        run_id: compute_tool_waste(run_events)
        for run_id, run_events in group_by_run(events).items()
    }
