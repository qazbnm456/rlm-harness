"""Phase B — unified, replayable trajectory recording.

Two sources must be merged to get a complete picture of an RLM-as-harness run:

1. The main LM's REPL trajectory, which ``dspy.RLM`` already returns on the
   ``Prediction`` object as ``trajectory`` (a list of ``{reasoning, code,
   output}`` dicts, verified against dspy 3.2.1) plus ``final_reasoning``.
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

SCHEMA = "rlm-kit/trace/v1"

# Event types written to the stream.
EVENT_RUN_START = "run_start"
EVENT_MAIN_STEP = "main_step"
EVENT_SUB_CALL = "sub_call"
EVENT_TOOL_CALL = "tool_call"
EVENT_FINAL = "final"
EVENT_RESULT = "result"
EVENT_RUN_END = "run_end"

_active: ContextVar[TraceRecorder | None] = ContextVar("rlm_kit_recorder", default=None)


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
    the per-run sub-LM binding in ``rlm_kit.sub_lm`` (kept here because ``_active`` is module-private)."""
    token = _active.set(recorder)
    try:
        yield
    finally:
        _active.reset(token)


def record_tool_call(
    tool: str, *, args: dict | None = None, **fields: Any
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
    """
    recorder = current_recorder()
    if recorder is None:
        return None
    payload: dict[str, Any] = {"tool": tool}
    if args is not None:
        payload["args"] = args
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
        # at finalize time (all identical). A per-turn callback (rlm_kit.task) feeds note_main_step()
        # AS each turn is parsed; record_main_trajectory() then matches by reasoning and backfills the
        # real ts — keeping the full {reasoning,code,output} payload, only correcting the timestamp.
        # Empty (no callback wired, or replay) → record_main_trajectory falls back to clock().
        self._main_ts: list[tuple[Any, float]] = []
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
        self.record(EVENT_RUN_START, {"meta": self._meta})
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

    # -- live per-turn timing (fed by a dspy parse callback; see rlm_kit.task) -------------

    def begin_main_capture(self) -> None:
        """Reset the live per-turn timestamp buffer at the start of a run attempt.

        ``run_with_retry`` may re-run the RLM; only the FINAL attempt's turns end up in the recorded
        trajectory, so the buffer is cleared per attempt to keep it aligned with what will be recorded.
        """
        with self._lock:
            self._main_ts = []

    def note_main_step(self, reasoning: Any, ts: float | None = None) -> None:
        """Buffer that a ROOT planner turn was parsed LIVE at ``ts`` (default: now).

        Matched back to the post-hoc trajectory (by ``reasoning``) in ``record_main_trajectory`` to
        backfill the event ts. Thread-safe (a dspy callback may fire from a worker thread). Never
        touches the JSONL — it only stages a timestamp for later reconciliation.
        """
        stamp = self._clock() if ts is None else ts
        with self._lock:
            self._main_ts.append((reasoning, stamp))

    def record_main_trajectory(self, prediction: Any) -> None:
        """Extract the RLM ``Prediction`` trajectory into ``main_step`` events.

        Each turn's ``ts`` is the LIVE time it was parsed (from ``note_main_step``), matched by
        ``reasoning`` — so a re-rendered trace reflects when turns actually happened, not when the
        trajectory was flushed. The match consumes the earliest unused live stamp with the same
        reasoning (so dspy's double parse-callback per turn resolves to the first/true time); a turn
        with no live stamp (no callback wired, or replay) falls back to ``clock()`` — unchanged from
        before. Payload shape, ``step_id`` and file order are identical either way; only the ts value
        of a main_step improves, which leaves step_id-ordered readers (RL dataset, replay) and the
        ``max(ts)-min(ts)`` elapsed metric untouched.

        Tolerant of shape drift: a missing/oddly-typed ``trajectory`` is recorded
        as empty rather than raising, so a dspy minor-version change degrades to a
        thinner trace instead of a crash.
        """
        trajectory = getattr(prediction, "trajectory", None) or []
        if not isinstance(trajectory, Iterable) or isinstance(trajectory, (str, bytes)):
            trajectory = []
        with self._lock:
            live = list(self._main_ts)
        used = [False] * len(live)

        def _match_ts(reasoning: Any) -> float | None:
            for i, (r, t) in enumerate(live):
                if not used[i] and r == reasoning:
                    used[i] = True
                    return t
            return None

        for turn, entry in enumerate(trajectory):
            entry = entry if isinstance(entry, dict) else {"raw": entry}
            reasoning = entry.get("reasoning")
            self.record(
                EVENT_MAIN_STEP,
                {
                    "turn": turn,
                    "reasoning": reasoning,
                    "code": entry.get("code"),
                    "output": entry.get("output"),
                },
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
