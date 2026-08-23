"""``run_isolated`` — bridge a coroutine into a sync call site that may already own a running
event loop (mirrors the bridging problem ``mcp.py`` solves for its own, different reason).

A consumer building their OWN transport for :func:`rlm_harness.tools.harness_from_endpoint` (e.g.
an in-process ``call_endpoint`` that awaits a child ``RLMTask.arun()`` directly, instead of
spawning a subprocess or making an HTTP call) needs to call ``async`` code from a plain sync
function — a `dspy.RLM` tool is invoked with a bare ``()``, never awaited (see
``rlm_harness/tools/command.py``'s module docstring for the same constraint on ``run_command``).
The obvious ``asyncio.run(coro)`` fails with ``RuntimeError: asyncio.run() cannot be called from a
running event loop`` the moment the calling thread already has one — which is not a hypothetical
edge case here: :meth:`rlm_harness.task.RLMTask.run` calls ``asyncio.run(self.arun(**inputs))`` for
its own sync entry point, so a tool dispatched from *inside* that call genuinely executes on a
thread with an actively-running loop. This is the everyday shape of this codebase's sync path.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


def run_isolated(coro_factory: Callable[[], Awaitable[T]]) -> T:
    """Run ``coro_factory()`` to completion on a DEDICATED new thread with its own fresh event
    loop, and block the calling thread for the result.

    Always isolates — regardless of whether the CALLING thread already has a running loop — rather
    than inspecting ``asyncio.get_running_loop()`` and reusing the current thread when "safe": a
    fresh thread is cheap, and a presence/running check has to correctly distinguish "a loop
    object exists" from "a loop is actually running," which is an easy place to get subtly wrong
    for no real benefit. One thread per call, by design: this is NOT a persistent bridge like
    ``mcp.py``'s dedicated long-lived loop for a stateful session — there is no persistent
    connection state here to keep alive, so a fresh thread per call is the simplest correct answer.

    The child thread's exception (if ``coro_factory()`` raises, or the coroutine itself raises) is
    re-raised on the CALLING thread, unchanged — never swallowed, so a caller-side retry loop (e.g.
    :func:`rlm_harness.tools.make_harness_tool`'s transient-retry handling) still sees it.

    **Contextvar scoping — read before wrapping a traced call.** A fresh ``threading.Thread``
    starts with an EMPTY ``contextvars.Context``; contextvars are NOT inherited into it. This is
    the identical hazard :func:`rlm_harness.trace.recorder_scope`'s docstring already documents for
    ``dspy.RLM.llm_query_batched``'s ``ThreadPoolExecutor`` workers ("those workers see
    ``current_recorder() is None``... under-counting the lifeline") — ``run_isolated`` has the same
    exposure, arguably more starkly (no partial context-copying at all). Concretely: if you wrap
    ``run_isolated(...)`` INSIDE an outer ``with TraceRecorder(...):`` block, ``current_recorder()``
    returns ``None`` *inside* ``coro_factory`` — any ``record_tool_call``/``intercept_sub_lm``
    activity a delegated child triggers is silently unrecorded. Establish any contextvar-scoped
    state you need active inside ``coro_factory`` — notably a delegated child's OWN
    ``TraceRecorder``, for its own separate rollout — **inside** ``coro_factory`` itself, never
    around the call to ``run_isolated``.
    """
    result: dict[str, T] = {}
    error: dict[str, BaseException] = {}

    def _target() -> None:
        try:
            result["value"] = asyncio.run(coro_factory())
        except BaseException as exc:  # re-raised on the calling thread below; never swallowed here
            error["exc"] = exc

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join()

    if "exc" in error:
        raise error["exc"]
    return result["value"]
