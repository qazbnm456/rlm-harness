"""`run_isolated` — the async-bridge primitive for a consumer's own harness-delegation transport.

All offline, no dspy: `run_isolated` is pure `threading`/`asyncio`, dspy-free by construction."""
import asyncio

import pytest

from rlm_harness.tools import run_isolated
from rlm_harness.trace import TraceRecorder, current_recorder


def test_run_isolated_returns_the_coroutines_result():
    async def _coro():
        return "hello"

    assert run_isolated(lambda: _coro()) == "hello"


def test_run_isolated_propagates_exception_from_the_coroutine():
    async def _boom():
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        run_isolated(lambda: _boom())


def test_run_isolated_works_from_inside_a_running_event_loop():
    # The hazard this function exists to survive: RLMTask.run() calls
    # asyncio.run(self.arun(**inputs)) for its own sync entry point, so a tool dispatched from
    # inside that call genuinely executes on a thread with an ACTIVELY-RUNNING loop. `outer()`
    # reproduces exactly that shape — a naive `asyncio.run(coro)` called synchronously from here
    # would raise "cannot be called from a running event loop"; run_isolated must not.
    async def _inner():
        return 42

    async def outer():
        return run_isolated(lambda: _inner())

    assert asyncio.run(outer()) == 42


def test_run_isolated_does_not_see_an_outer_recorder(tmp_path):
    # A fresh threading.Thread starts with an EMPTY contextvars.Context — contextvars are NOT
    # inherited into it (the same non-inheritance rlm_harness.trace.recorder_scope's docstring
    # already documents for dspy.RLM's ThreadPoolExecutor sub-LM workers). A TraceRecorder entered
    # AROUND the run_isolated() call must therefore be invisible INSIDE coro_factory.
    path = str(tmp_path / "outer.jsonl")
    seen = {}

    def coro_factory():
        async def _inner():
            seen["recorder"] = current_recorder()
            return "ok"

        return _inner()

    with TraceRecorder(path, run_id="outer"):
        result = run_isolated(coro_factory)

    assert result == "ok"
    assert seen["recorder"] is None


def test_run_isolated_recorder_entered_inside_is_scoped_and_detaches(tmp_path):
    # The documented, correct usage: establish a TraceRecorder (for the delegated child's OWN
    # rollout) INSIDE coro_factory. It must be the active recorder for code running within that
    # scope, and must be gone again once the `with` block exits — even though all of this runs on
    # run_isolated's dedicated thread, not the caller's.
    path = str(tmp_path / "inner.jsonl")
    seen = {}

    def coro_factory():
        async def _inner():
            with TraceRecorder(path, run_id="child"):
                seen["during"] = current_recorder()
            seen["after"] = current_recorder()
            return "ok"

        return _inner()

    result = run_isolated(coro_factory)

    assert result == "ok"
    assert seen["during"] is not None and seen["during"].run_id == "child"
    assert seen["after"] is None


def test_run_isolated_sequential_calls_do_not_leak_recorder_state(tmp_path):
    # Regression-proofs the "one thread per call, by design" claim itself, not just the bridging
    # mechanism: a future "simplification" that reused a thread/pool across calls could leave a
    # recorder's _active.set() unmatched by a reset() on a persistent Context, letting one call's
    # recorder leak forward into the next. The current fresh-thread-per-call design is structurally
    # immune (a new Context every time) — this asserts that stays true across two calls in a row.
    path = str(tmp_path / "seq.jsonl")
    seen = []

    def first():
        async def _inner():
            with TraceRecorder(path, run_id="first"):
                pass
            return "first-done"

        return _inner()

    def second():
        async def _inner():
            seen.append(current_recorder())
            return "second-done"

        return _inner()

    assert run_isolated(first) == "first-done"
    assert run_isolated(second) == "second-done"
    assert seen == [None]
