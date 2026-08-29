"""Phase B — unified, replayable trajectory recording.

Two sources must be merged to get a complete picture of an RLM-as-harness run:

1. The main LM's REPL trajectory, which ``dspy.RLM`` already returns on the
   ``Prediction`` object as ``trajectory`` (a list of ``{reasoning, code,
   output}`` dicts, verified against dspy 3.3.0) plus ``final_reasoning``.
2. The intercepted sub-LM pipeline and any LM-decided tool calls — which live
   *inside* the intercepted ``sub_lm`` / tool wrappers and are therefore invisible
   to the RLM trajectory. These are exactly the steps most valuable for Agentic RL.

``TraceRecorder`` collects both into a single append-only JSONL event stream,
keyed by ``run_id`` + a monotonically increasing ``step_id``. The active recorder
is published via a ``contextvar`` so the intercepted ``sub_lm`` and tools find it without
threading it through every call. An optional Langfuse sink mirrors events for
observability; the JSONL is the source of truth for the RL dataset (it must not
depend on Langfuse's export format).

This module is dependency-light: stdlib ``json`` only. No dspy import.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from contextvars import ContextVar, Token
from typing import Any, Self

SCHEMA = "rlm-harness/trace/v1"

# Event types written to the stream.
EVENT_RUN_START = "run_start"
EVENT_MAIN_STEP = "main_step"
EVENT_SUB_CALL = "sub_call"
EVENT_TOOL_CALL = "tool_call"
EVENT_FINAL = "final"
EVENT_RESULT = "result"
EVENT_RUN_END = "run_end"

_active: ContextVar[TraceRecorder | None] = ContextVar("rlm_harness_recorder", default=None)


def current_recorder() -> TraceRecorder | None:
    """Return the recorder active in the current context, or ``None``."""
    return _active.get()


@contextlib.contextmanager
def recorder_scope(recorder: TraceRecorder | None) -> Iterator[None]:
    """Make ``recorder`` the active recorder for the CURRENT context (thread), restoring on exit.

    A ``ContextVar`` is NOT inherited by threads a ``ThreadPoolExecutor`` spawns, so when
    ``dspy.RLM.llm_query_batched`` fans the sub-LM across executor workers, those workers see
    ``current_recorder() is None`` and the batched escalations record NO ``sub_call`` (under-counting
    the lifeline). Re-establishing the recorder per call inside the worker thread fixes that. Used by
    the per-run sub-LM binding in ``rlm_harness.sub_lm`` (kept here because ``_active`` is module-private)."""
    token = _active.set(recorder)
    try:
        yield
    finally:
        _active.reset(token)


#: What produced a model-backed tool's outcome. `ok=False` means THREE different things and they
#: are not interchangeable — see `payload_cause`. Defined HERE, in the lowest layer, because the
#: same four names have to mean the same thing for a live `ModelToolResult` (which re-exports them)
#: and for a recorded payload a dataset/replay reader gets back off disk. Two vocabularies for one
#: distinction is how it gets collapsed again.
CAUSE_OK = "ok"                          # the validator ran and accepted
CAUSE_INVALID = "invalid"                # the validator ran and rejected
CAUSE_ENDPOINT = "endpoint"              # the model call failed after retries; the validator never ran
CAUSE_CIRCUIT_BROKEN = "circuit_broken"  # short-circuited; no model call, no validator


def payload_cause(payload: dict) -> str:
    """Which of the four outcomes a recorded ``tool_call`` payload is — the read-side mirror of
    :attr:`rlm_harness.tools.ModelToolResult.cause`.

    A consumer reading a trace has only the payload, and ``ok`` alone cannot tell a validator
    rejection from an endpoint failure or a circuit break. Reading it as one thing has shipped in
    four separate consumers, into training labels, a scored rubric criterion, and delivered report
    text — in the worst measured case a run whose validator ran ZERO times was reported as 113
    format-quality failures, and scored against the planner's spec quality for them.

    Reads three keys, in the order that cannot disagree with itself: ``circuit_broken`` (nothing was
    called), then the endpoint string (the model call failed, so nothing was validated), then
    ``ok``. The endpoint string is looked up under BOTH ``endpoint_error`` and ``error`` because the
    consumer convention has used each; the ``ok`` key is often ABSENT on an endpoint payload, and
    ``payload.get("ok")`` returning ``None`` is exactly how it silently reads as a decline.

    **The endpoint keys are tested for PRESENCE, not truthiness**, which is what makes this an actual
    mirror of :attr:`rlm_harness.tools.ModelToolResult.cause` rather than a near-copy that disagrees on
    one case. The write side has always been ``self.endpoint_error is not None``; this side shipped
    as ``payload.get("endpoint_error") or ...``, and the two differ on the EMPTY STRING — which is
    not a corner case but the COMMON one, because the field is filled with ``str(exc)`` and that is
    ``''`` for ``httpx.ConnectTimeout`` / ``ReadTimeout`` / ``ConnectError``, ``TimeoutError``,
    ``OSError`` and ``http.client.RemoteDisconnected``. Under truthiness every one of those fell
    through to ``CAUSE_INVALID``: a dropped connection labelled a content decline, which is exactly
    the misclassification this function exists to prevent. Found by a downstream consumer that
    declined to adopt this function for that reason and pinned the divergence in its own tests.
    """
    if payload.get("circuit_broken"):
        return CAUSE_CIRCUIT_BROKEN
    if payload.get("endpoint_error") is not None or payload.get("error") is not None:
        return CAUSE_ENDPOINT
    return CAUSE_OK if payload.get("ok") else CAUSE_INVALID


def record_tool_call(
    tool: str, *, args: dict | None = None, duration_s: float | None = None, **fields: Any
) -> dict | None:
    """Record a ``tool_call`` event on the active recorder; return it, or ``None``.

    Every tool wrapper otherwise repeats the same three lines — look up the active
    recorder, guard against ``None``, then ``record("tool_call", {...})`` — and in
    doing so re-derives by hand the canonical payload shape the replay/dataset
    readers consume (``payload["tool"]`` to match a call, ``payload.get("args")``,
    ``payload.get("result")`` / ``"ok"`` / ``"raw"`` / ``"reasoning"`` / ``"errors"``
    as the outcome). Centralising emission here keeps that format — the replay/RL
    source of truth — owned in ONE place instead of copied across every tool.

    ``args`` (when given) and any extra keyword fields are merged into the payload
    verbatim, so a caller stays free to attach tool-specific fields (``note``,
    ``bytes``, ``results``, ``template_id`` …). No-ops and returns ``None`` when no
    recorder is active, so a tool can call it unconditionally.

    **For a model-backed tool, record the outcome ONCE, AFTER the branch, and record all three
    fields.** ``ok=False`` has three causes (see :func:`payload_cause`) and a reader can only split
    them if the payload carries ``circuit_broken`` and the endpoint string alongside ``ok``. Two
    hazards, both observed in shipped consumers:

    - Recording BEFORE the endpoint check destroys the distinction at write time, and no read-side
      fix can recover it. One consumer emits its delegation event with a bare ``ok=`` and then
      raises on the endpoint error two lines later; 226 events in its corpus are indistinguishable
      between "the harness was unreachable" and "the harness returned nothing usable".
    - Omitting ``ok`` on the endpoint path (recording only ``error=``) is worse than it looks:
      ``payload.get("ok")`` then returns ``None``, which is falsy, so every ``not payload.get("ok")``
      counter downstream silently absorbs infrastructure failures as content declines.

    Passing ``cause=result.cause`` explicitly is the cheapest way to be sure — the derivation is
    then done once, by the code that knows, rather than re-derived by every reader.

    **``duration_s`` — how long the tool actually took, in seconds.** An explicit parameter rather
    than one more ``**fields`` entry so the name and the unit are documented in ONE place; written
    only when given, like ``args``, so a caller that does not measure adds no key. Measure with
    a MONOTONIC clock around the work itself — ``time.perf_counter()``, or ``time.monotonic()`` as
    ``make_command_tool`` already uses — never with wall-clock.

    Worth the two lines at every call site: without it a trace's only clock is the envelope ``ts``,
    stamped when the event is RECORDED, so the sole way to attribute wall-clock is the gap between
    consecutive events — which charges a whole turn's model generation to that turn's first tool
    call. Every attribution made against this kit's own corpus before 1.6.0 had that error in it.
    """
    recorder = current_recorder()
    if recorder is None:
        return None
    payload: dict[str, Any] = {"tool": tool}
    if args is not None:
        payload["args"] = args
    if duration_s is not None:
        payload["duration_s"] = round(float(duration_s), 6)
    payload.update(fields)
    return recorder.record(EVENT_TOOL_CALL, payload)


class TraceRecorder:
    """Append-only JSONL recorder for one or more runs.

    Use as a context manager so it becomes the active recorder for the duration
    of a run::

        with TraceRecorder("trace.jsonl", run_id="r1") as rec:
            result = await task.arun(...)   # main_step/sub_call/tool_call land here
    """

    def __init__(
        self,
        path: str,
        run_id: str,
        *,
        langfuse: Any = None,
        meta: dict | None = None,
        clock=time.time,
        on_event: Callable[[dict], None] | None = None,
    ) -> None:
        self.path = path
        self.run_id = run_id
        self._langfuse = langfuse
        self._meta = meta or {}
        self._clock = clock
        # Optional LIVE observer: every recorded event is also handed to this callback as it happens
        # (best-effort). Lets a consumer stream the trajectory in real time — a streaming UI uses
        # it for tool_calls/sub_calls, which the planner's REPL invokes INSIDE the sandbox (so dspy's
        # on_tool callback never sees them, but the recorder does). Never mutates the persisted trace.
        self._on_event = on_event
        self._step = 0
        self._token: Token | None = None
        self._fh = None
        # LIVE per-turn timestamps for the main LM's REPL turns. dspy.RLM only exposes its trajectory
        # on the FINAL Prediction, so record_main_trajectory() would otherwise stamp every main_step
        # at finalize time (all identical). A per-turn callback (rlm_harness.task) feeds note_main_step()
        # AS each turn is parsed; record_main_trajectory() then matches by reasoning and backfills the
        # real ts — keeping the full {reasoning,code,output} payload, only correcting the timestamp.
        # Empty (no callback wired, or replay) → record_main_trajectory falls back to clock().
        self._main_ts: list[tuple[Any, float]] = []
        # Sandbox execute() durations, staged by the interpreter wrappers the kit owns and matched
        # onto turns in `record_main_trajectory`. Same lifecycle as `_main_ts`: reset per attempt.
        self._exec_s: list[tuple[Any, float]] = []
        # llm_query_batched fans sub_lm calls across threads; a wrapped sub_lm
        # records a sub_call per thread. Serialise step assignment + the JSONL
        # write so concurrent escalations can't race step_ids or interleave lines
        # (the JSONL is the replay/RL source of truth — it must stay intact).
        self._lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> Self:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")
        self._token = _active.set(self)
        # `rlm_harness` sits BESIDE `meta`, never inside it: `meta` is the caller's namespace
        # (`rubric_to_meta` writes there) and the kit must not squat in it. Deferred import
        # because `trace.py` is imported BY `__init__.py` — at module top this is circular, and
        # by the time a recorder is entered the package is fully imported.
        #
        # Why at all: `schema` is the FORMAT version (`trace/v1`), which says nothing about which
        # kit wrote the file. A corpus spanning several releases is then un-attributable — you
        # cannot tell a behaviour change from a version change, which is exactly the wall this
        # release's own measurement work hit.
        from . import __version__

        self.record(EVENT_RUN_START, {"meta": self._meta, "rlm_harness": __version__})
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self.record(
                EVENT_RUN_END,
                {"ok": exc_type is None, "error": repr(exc) if exc else None},
            )
        finally:
            if self._token is not None:
                _active.reset(self._token)
                self._token = None
            if self._fh is not None:
                self._fh.close()
                self._fh = None

    # -- recording ---------------------------------------------------------

    def record(self, event_type: str, payload: dict, *, ts: float | None = None) -> dict:
        """Append one event and return it. Steps are assigned monotonically.

        ``ts`` overrides the event timestamp; default ``None`` stamps ``clock()`` (now). The override
        exists so ``record_main_trajectory`` can backfill a main_step's LIVE per-turn time (captured
        while the run was in flight) instead of the finalize time it would otherwise get.

        Thread-safe: the step-assignment + file write run under a lock so
        concurrent ``sub_call`` records (e.g. from ``llm_query_batched`` fanning
        the wrapped sub_lm across threads) can't race ``step_id`` or interleave
        JSONL lines. The optional Langfuse mirror runs outside the lock (best
        effort, never blocks the source-of-truth write on the network).
        """
        with self._lock:
            event = {
                "schema": SCHEMA,
                "run_id": self.run_id,
                "step_id": self._step,
                "ts": self._clock() if ts is None else ts,
                "type": event_type,
                "payload": payload,
            }
            self._step += 1
            if self._fh is not None:
                self._fh.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
                self._fh.flush()
        self._mirror_langfuse(event)
        if self._on_event is not None:
            try:
                self._on_event(event)   # live observer (best-effort, outside the lock)
            except Exception:
                pass
        return event

    # -- live per-turn timing (fed by a dspy parse callback; see rlm_harness.task) -------------

    def begin_main_capture(self) -> None:
        """Reset the live per-turn timestamp buffer at the start of a run attempt.

        ``run_with_retry`` may re-run the RLM; only the FINAL attempt's turns end up in the recorded
        trajectory, so the buffer is cleared per attempt to keep it aligned with what will be recorded.
        """
        with self._lock:
            self._main_ts = []
            self._exec_s = []

    def note_main_step(self, reasoning: Any, ts: float | None = None) -> None:
        """Buffer that a ROOT planner turn was parsed LIVE at ``ts`` (default: now).

        Matched back to the post-hoc trajectory (by ``reasoning``) in ``record_main_trajectory`` to
        backfill the event ts. Thread-safe (a dspy callback may fire from a worker thread). Never
        touches the JSONL — it only stages a timestamp for later reconciliation.

        **Callers must stage exactly ONE entry per turn.** The match is by ``reasoning``, and a
        model that repeats a reasoning string across turns (a retry loop does) makes any surplus
        entry claimable by the wrong turn. ``task.py:_MainStepTimer`` is the in-kit caller and
        deduplicates dspy's nested parse callbacks for precisely this reason; see its docstring for
        why there were two, and what a -338.7s rendered duration looked like when there were.
        """
        stamp = self._clock() if ts is None else ts
        with self._lock:
            self._main_ts.append((reasoning, stamp))

    def note_exec_duration(self, seconds: float, code: Any = None) -> None:
        """Buffer how long ONE sandbox ``execute()`` took, keyed by the CODE it ran.

        The gap before a ``main_step`` is the single biggest bucket in a real trace, and it mixes
        two things with completely different fixes: the root LM GENERATING the turn, and the
        sandbox EXECUTING it. Nothing in ``trace/v1`` separated them, so "99.8% of wall-clock is
        the root LM turn" was as far as any analysis could get. Measured on a real consumer once
        this existed: **execution is ~1% of a turn's wall-clock**, the rest is generation.

        **What this measures is ``execute()`` WALL-CLOCK, which is not the same as time the sandbox
        spent running Python.** dspy's interpreter dispatches a tool call synchronously from inside
        ``execute()`` (``PythonInterpreter._handle_tool_call``), and ``llm_query`` / the sub-LM are
        injected as tools, so a cell whose code calls one BLOCKS here for the whole round trip —
        host-side network time, a subprocess, another model's generation — and every second of it
        lands in ``exec_duration_s``. There is no hook to subtract it (the same limitation
        ``RLMConfig.sandbox_turn_timeout_s`` carries, and for the same reason). A one-line cell that
        calls one slow tool is therefore indistinguishable here from four minutes of real compute.
        **Read a large outlier as "the turn blocked", not as "the sandbox was busy".** Observed in the
        wild and confirmed: a 235.5s value on a 143-character single-line cell that was
        ``print(llm_query(...))`` — the whole of it a sub-LM round trip, with the generated text
        recorded as the turn's ``output``.

        Cross-checking it against the same run's ``tool_call`` / ``sub_call`` events is worth trying
        but **often will not resolve it, and a reader must not take the absence as the field lying**:

        * ``llm_query`` produces a ``sub_call`` only if the caller passed an
          :func:`~rlm_harness.sub_lm.intercept_sub_lm`-wrapped ``sub_lm``. A plain ``dspy.LM`` is
          called by dspy directly and records NOTHING, so the single largest block of time in a run
          can have no event of its own. (``RLMTask`` does wrap the sub-LM with
          ``bind_recorder_to_sub_lm``, but that only makes the recorder visible across dspy's worker
          threads — it does not emit an event.)
        * a ``tool_call``'s ``duration_s`` is OPTIONAL and is set only by the tools whose cost is a
          wait outside this process. The local read/grep/edit tools record the call with no duration
          at all, so a slow local tool is invisible here too.

        When neither is available the honest read is "this turn blocked on something unrecorded",
        not a number to reconcile.

        Called by the interpreter wrappers the kit owns (``sandbox.py``'s guarded ``execute`` and
        ``ContainerInterpreter.execute``). An interpreter a caller injects directly is NOT wrapped,
        so its turns carry no ``exec_duration_s`` at all — absent, rather than a wrong zero.

        **Matched by ``code``, never by position.** A turn does NOT always reach the sandbox: dspy
        raises ``SyntaxError`` out of ``_strip_code_fences`` for an explicitly non-Python fence
        (```` ```json ````, ```` ```bash ````) and records that turn from the *unstripped* text
        without calling ``execute()`` at all. Zipping positionally therefore shifts every later
        turn's duration by one and silently drops the last — a confidently WRONG attribution with
        nothing to signal it, which is worse than having no attribution. Matching on the code that
        actually ran makes the skipped turn match nothing, so its key is simply absent. Same
        earliest-unused-at-or-after-the-cursor rule ``note_main_step``'s ``reasoning`` match uses.

        Unlike ``note_main_step`` this stages ONE entry per ``execute()`` call from dspy's strictly
        sequential loop, so the staged list is already 1:1 with the turns that ran and in their
        order — a duplicated code cell is NOT a defect here. The cursor is symmetry plus one real
        if unlikely guard: dspy runs a setup ``execute()`` before the loop, whose staged entry a
        turn with a colliding code string could otherwise claim.

        Thread-safe and never touches the JSONL; like ``note_main_step`` it only stages a value for
        reconciliation in ``record_main_trajectory``.
        """
        with self._lock:
            self._exec_s.append((code, float(seconds)))

    def record_main_trajectory(self, prediction: Any) -> None:
        """Extract the RLM ``Prediction`` trajectory into ``main_step`` events.

        Each turn's ``ts`` is the LIVE time it was parsed (from ``note_main_step``), matched by
        ``reasoning`` — so a re-rendered trace reflects when turns actually happened, not when the
        trajectory was flushed. The match consumes the earliest unused live stamp with the same
        reasoning AT OR AFTER the previous match (see the cursor comment below); a turn with no live
        stamp (no callback wired, or replay) falls back to ``clock()`` — unchanged from before.
        Payload shape, ``step_id`` and file order are identical either way; only the ts value
        of a main_step improves, which leaves step_id-ordered readers (RL dataset, replay) and the
        ``max(ts)-min(ts)`` elapsed metric untouched.

        **How a reader must order these events** — a downstream consumer got this wrong, so it is
        written down here rather than left to be inferred:

        * ``main_step`` events are emitted in ONE BLOCK once ``aforward()`` has returned, so a
          ``tool_call`` recorded mid-run precedes them in FILE order while being chronologically
          LATER. Measured at 70 of 76 real traces. This is by design, not a defect.
        * ``payload["turn"]`` is AUTHORITATIVE for ordering, and file order among ``main_step``
          events already matches it (72 of 72 traces). **Never sort main_steps by ``ts`` to
          "recover" their order** — that reorders turns.
        * ``ts`` is for placing a turn against the tool calls around it, and nothing else.

        Tolerant of shape drift: a missing/oddly-typed ``trajectory`` is recorded
        as empty rather than raising, so a dspy minor-version change degrades to a
        thinner trace instead of a crash.
        """
        trajectory = getattr(prediction, "trajectory", None) or []
        if not isinstance(trajectory, Iterable) or isinstance(trajectory, (str, bytes)):
            trajectory = []
        with self._lock:
            live = list(self._main_ts)
            execs = list(self._exec_s)
        # Both matches scan FORWARD ONLY, from a cursor parked just past the previous match.
        # The cursor alone is what makes each entry single-use: it only ever advances to `i + 1`
        # after consuming index `i`, and every scan starts AT the cursor, so no index can be
        # revisited. (A separate `used` flag list lived here until 1.6.1 and became provably dead
        # the moment the cursor arrived — every index the loop can reach is, by construction,
        # past everything already consumed.)
        # Trajectory order IS chronological order, so consuming staged entries in non-decreasing
        # index order is true by construction — and enforcing it means a later turn can never be
        # handed a stamp/duration that belongs to an earlier one. Each cursor is a single-element
        # list so the closures can advance it. Note the cursor advances ONLY on a match: a turn
        # that matches nothing must not push the cursor past a LATER turn's entry (that would
        # break `test_a_turn_dspy_never_executed_does_not_steal_the_next_turns_duration`).
        #
        # This is defence in depth, NOT the fix for the ts inversions — those are fixed at source
        # by `task.py:_MainStepTimer` staging one entry per turn instead of two. A cursor alone
        # cannot repair ADJACENT duplicate keys (turn 1 would still take turn 0's spare entry),
        # which merely hides the symptom while leaving the value wrong.
        ts_cursor = [0]
        exec_cursor = [0]

        def _match_exec(code: Any) -> float | None:
            """The earliest unused duration staged for exactly this code AT OR AFTER the cursor."""
            if not isinstance(code, str):
                return None
            for i in range(exec_cursor[0], len(execs)):
                ran, secs = execs[i]
                if ran == code:
                    exec_cursor[0] = i + 1
                    return secs
            return None

        def _match_ts(reasoning: Any) -> float | None:
            for i in range(ts_cursor[0], len(live)):
                r, t = live[i]
                if r == reasoning:
                    ts_cursor[0] = i + 1
                    return t
            return None

        for turn, entry in enumerate(trajectory):
            entry = entry if isinstance(entry, dict) else {"raw": entry}
            reasoning = entry.get("reasoning")
            payload = {
                "turn": turn,
                "reasoning": reasoning,
                "code": entry.get("code"),
                "output": entry.get("output"),
            }
            # Matched on the CODE that ran, never on position — see `note_exec_duration`. A turn
            # dspy recorded without executing (a non-Python fence tag) matches nothing and the key
            # is ABSENT, instead of stealing the next turn's duration and shifting every one after
            # it. Conditional for the same reason `args` is: an unconditional null would land on
            # every main_step of every trace forever.
            ran_for = _match_exec(entry.get("code"))
            if ran_for is not None:
                payload["exec_duration_s"] = round(ran_for, 6)
            self.record(
                EVENT_MAIN_STEP,
                payload,
                ts=_match_ts(reasoning),   # live per-turn ts, or None → clock() fallback
            )
        self.record(
            EVENT_FINAL,
            {"final_reasoning": getattr(prediction, "final_reasoning", None)},
        )

    def record_result(self, output: Any) -> None:
        """Record the task's final validated output (after coercion)."""
        try:
            serialised = (
                output.model_dump() if hasattr(output, "model_dump") else output
            )
        except Exception:
            serialised = repr(output)
        self.record(EVENT_RESULT, {"output": serialised})

    # -- optional observability sink --------------------------------------

    def _mirror_langfuse(self, event: dict) -> None:
        if self._langfuse is None:
            return
        try:  # pragma: no cover - exercised only with a real client
            self._langfuse.event(
                name=event["type"],
                metadata={"run_id": event["run_id"], "step_id": event["step_id"]},
                input=event["payload"],
            )
        except Exception:
            # Observability must never break the run or the JSONL source of truth.
            pass


def load_events(path: str, run_id: str | None = None) -> list[dict]:
    """Read a JSONL trace file, optionally filtering to one ``run_id``.

    Events are returned in file order (which is also step order per run).
    """
    events: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            if run_id is None or event.get("run_id") == run_id:
                events.append(event)
    return events


def group_by_run(events: Iterable[dict]) -> dict[str, list[dict]]:
    """Group a flat event list into ``{run_id: [events...]}`` preserving order."""
    runs: dict[str, list[dict]] = {}
    for event in events:
        runs.setdefault(event.get("run_id"), []).append(event)
    return runs
