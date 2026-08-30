"""Trace utilization metrics — how a run's activity was distributed across the root LM's own
turns, tool calls, and sub-LM escalations. A sibling to ``rubric.py``'s "derive facts from a trace"
shape, but structurally different: ``rubric.criteria_facts`` slices a CALLER-SUPPLIED facts dict
against a caller-supplied lens; this module COMPUTES fixed counts/rates directly from the raw
``trace/v1`` event stream, with no caller input beyond the events themselves. Reward-free, like
every other trace-derived module here: raw counts and rates, never a score.

Reads only already-frozen ``trace/v1`` fields — ``event["type"]``, ``payload["tool"]``, the
optional ``payload["duration_s"]``, ``payload["code"]`` and ``payload["final_reasoning"]`` (both
1.8.0, for :func:`compute_run_facts`), and (through :func:`rlm_harness.trace.payload_cause`, never
directly) ``circuit_broken`` / ``endpoint_error`` / ``error`` / ``ok``. No new event type, nothing
the trace contract needs to change for. dspy-free at module top — :func:`compute_run_facts` reaches
``_dspy_compat`` for two dspy behavioural facts, and that module keeps its own dspy imports lazy.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from .trace import (
    CAUSE_CIRCUIT_BROKEN,
    CAUSE_ENDPOINT,
    CAUSE_INVALID,
    CAUSE_OK,
    EVENT_FINAL,
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


#: The exact key set :func:`compute_run_facts` emits — a CLOSED, public constant, not a docstring
#: promise. The dict is BUILT against this tuple, `tests/test_contract.py` pins its contents
#: exactly, and a consumer's rubric lens imports it instead of hand-copying names. Adding a key
#: therefore means editing a SemVer-governed public name that shows up in a diff, which is the
#: mechanism keeping a reward-shaped scalar from drifting into the source of truth — the kit writes
#: these keys itself, so `run_label_bundle`'s "refuse a caller-supplied name" analogue would have
#: been dead code here.
#:
#: Every value is a SCALAR. `tool_calls_by_name` is deliberately ABSENT: an open key space cannot
#: belong to a closed set, and a `ToolWaste` dataclass reaching `record()`'s `json.dumps(...,
#: default=str)` would be silently stringified into the trace as `"ToolWaste(tool='g', calls=1…)"`.
#: A consumer wanting per-tool detail calls `compute_run_utilization` / `compute_tool_waste`.
RUN_FACT_KEYS: tuple[str, ...] = (
    "main_steps", "tool_calls", "sub_calls", "tool_call_rate", "sub_call_rate",
    "tool_declines", "tool_endpoint_errors", "tool_circuit_breaks",
    "tool_wasted_seconds", "tool_total_seconds", "tool_ok", "tool_measured_calls",
    "fence_refused_turns", "budget_exhausted",
)


def _sum_or_none(values: Iterable[float | None]) -> float | None:
    """Sum, preserving "nothing was measured" as ``None`` rather than collapsing it to ``0.0``.

    ``ToolWaste`` is explicit that its ``*_seconds`` are ``None``, never ``0.0``, when no call
    carried a duration, and a test pins that. Summing with ``or 0.0`` would regress the distinction
    inside the very dict that feeds a rubric — so: ``None`` when nothing measured anything, a
    PARTIAL sum otherwise, with ``tool_measured_calls`` riding alongside to show the partiality
    (one measured call in fifty is otherwise indistinguishable from fifty in fifty).
    """
    seen = [v for v in values if v is not None]
    return sum(seen) if seen else None


def compute_run_facts(events: Iterable[dict]) -> dict:
    """The GENERIC half of a rubric's facts for ONE run — everything the kit can observe without
    knowing the consumer's domain.

    Feeds :func:`rlm_harness.rubric.criteria_facts`, which is documented pure and stays that way:
    the consumer supplies its criteria, its category lens, and its own domain facts, and merges
    this dict in. Nine consumers each hand-derive most of what is here, because `metrics.py` landed
    five weeks after their rubrics did and nobody went back.

    Reward-free by construction: counts, rates and one boolean. Keys are exactly
    :data:`RUN_FACT_KEYS`.

    **Single-run**, like :func:`compute_run_utilization` — pass one run's events, or use
    :func:`compute_run_facts_by_run` for a file holding several. A multi-run list silently
    conflates, which is why both existing computers already ship a ``_by_run`` sibling.

    Two readings that need care, both stated rather than implied:

    * ``fence_refused_turns`` is ``0`` on a run with no ``main_step`` events at all — UNMEASURED,
      not measured-zero. ``main_steps`` rides in the same dict to disambiguate.
    * ``budget_exhausted`` is ``None`` whenever the answer is unknown (see below), never ``False``.
    """
    events = list(events)
    util = compute_run_utilization(events)
    waste = compute_tool_waste(events).values()
    facts = {
        "main_steps": util.main_steps,
        "tool_calls": util.tool_calls_total,
        "sub_calls": util.sub_calls_total,
        "tool_call_rate": util.tool_call_rate,
        "sub_call_rate": util.sub_call_rate,
        "tool_declines": sum(w.invalid for w in waste),
        "tool_endpoint_errors": sum(w.endpoint_errors for w in waste),
        "tool_circuit_breaks": sum(w.circuit_broken for w in waste),
        "tool_wasted_seconds": _sum_or_none(w.wasted_seconds for w in waste),
        "tool_total_seconds": _sum_or_none(w.total_seconds for w in waste),
        "tool_ok": sum(w.ok for w in waste),
        "tool_measured_calls": sum(w.measured_calls for w in waste),
        "fence_refused_turns": _count_fence_refused(events),
        "budget_exhausted": _budget_exhausted(events),
    }
    assert set(facts) == set(RUN_FACT_KEYS)   # the closed set is built, not merely documented
    return facts


def compute_run_facts_by_run(events: Iterable[dict]) -> dict[str, dict]:
    """:func:`compute_run_facts` per ``run_id`` — the sibling both other computers already have."""
    return {rid: compute_run_facts(evs) for rid, evs in group_by_run(events).items()}


def _count_fence_refused(events: list[dict]) -> int:
    """Turns dspy refused to execute because of a markdown fence tag in the cell.

    **Named and documented for the MECHANISM, never a cause**, because the obvious cause is wrong.
    Running dspy's own stripper over three real corpora — 1,406 + 137 + 252 turns — found 60
    refusals and **zero** that START with a fence; 55 of the 60 are valid Python assigning a
    documentation page whose TEXT contains a fenced example::

        markdown = \"\"\"# Overview … ```bash … \"\"\"

    dspy's ``_strip_code_fences`` scans the whole cell including string literals, reads the inner
    tag, and refuses the turn. So this counts **dspy fence-stripper friction**, and the tag
    distribution (the documented repository's own language, plus ``bash`` for install steps) is a
    documentation generator behaving correctly. A consumer read it as format non-compliance and
    spent two prompt generations suppressing the blocks its own pages needed.

    The decision is `_dspy_compat.dspy_refuses_fence` — one place, mirroring dspy's private parser
    verbatim, because a shortcut disagrees with it on thousands of real cells.
    """
    from . import _dspy_compat

    return sum(
        1 for e in events
        if e.get("type") == EVENT_MAIN_STEP
        and _dspy_compat.dspy_refuses_fence((e.get("payload") or {}).get("code"))
    )


def _budget_exhausted(events: list[dict]) -> bool | None:
    """Did the run stop because its ITERATION budget ran out? ``None`` when unknowable.

    dspy marks that branch itself: falling out of the turn loop without a ``FINAL`` sets
    ``final_reasoning`` to a fixed marker, which the kit has always recorded on the ``final`` event.
    So this needs no configured cap staged into the trace — it works on every trace ever written,
    and it avoids the ``main_steps >= cap`` formula's false positive on a run that SUBMITs
    successfully on its last allowed turn.

    ``None``, never ``False``, when: there is no ``final`` event (a run whose ``aforward`` raised
    records none — 31 of 503 real runs), or ``final_reasoning`` is absent from shape drift.

    **Validated against the REAL deno/pyodide interpreter, not only the scripted one.** The kit's
    own test drives the fall-through through ``ScriptedInterpreter``, which proves dspy writes the
    marker and the kit reads it back but says nothing about whether the real sandbox path reaches
    that branch the same way. A consumer ran all three states against `dspy.PythonInterpreter` on
    deno 2.8.2 — scripting only the LM, since the interpreter is the seam that matters — and the
    field discriminated: ``True`` on a run that never submits, ``False`` on one that does, ``None``
    on a run that raises before the flush. **Two traps that fake a negative result:** the forced
    -final path makes a SECOND LM call for the task's own output field, so a scripted LM that runs
    out of turns dies in ``extract``, ``aforward`` raises, no ``final`` is recorded, and the marker
    looks lost when it is not; and a ``True`` without a submitting control run is not a measurement,
    since a field that is always ``True`` produces it too.

    **What it can and cannot answer.** Only a run that FINISHED inside its budget — the trajectory
    is recorded after ``aforward()`` returns, so a SIGKILLed job is exactly the case an operator
    asks about and exactly the case this reports ``None`` for. Exact
    EQUALITY, never a substring test: the success path writes the model's own reasoning, so ``in``
    would let a model quoting the phrase flip the fact. Last ``final`` wins if a ``run_id`` somehow
    carries two. Detects the ITERATION cap only — ``max_llm_calls`` exhaustion raises inside the
    sandbox and comes back as a turn, so it reads ``False`` here.
    """
    from . import _dspy_compat

    finals = [e for e in events if e.get("type") == EVENT_FINAL]
    if not finals:
        return None
    reasoning = (finals[-1].get("payload") or {}).get("final_reasoning")
    if reasoning is None:
        return None
    return reasoning == _dspy_compat.forced_final_marker()
