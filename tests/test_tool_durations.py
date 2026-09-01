"""Which shipped tools must record `duration_s`, enforced the way REPL safety already is.

`metrics.compute_tool_waste` can only attribute wall-clock to the calls that report it, and a tool
that silently stops reporting is invisible rather than loud. Two factories shipped without it in
1.6.0's first draft — `make_git_clone_tool` (a network clone) and `model_as_tool` (an actual LM
call) — which is precisely the failure mode `tests/test_repl_safety.py`'s `_REPL_FACTORIES` table
exists for: six factories once shipped with no REPL-safety coverage because each author had to
remember. Same cure, same shape: every `make_*` is either listed as outbound-and-timed, or
exempt WITH a written reason, and a new one that is neither fails here at the moment it ships.
"""

import types

import pytest

pytest.importorskip("dspy")

import rlm_harness.tools as tools_pkg
from rlm_harness.tools import make_command_tool, make_fetch_tool, make_web_search_tool
from rlm_harness.trace import TraceRecorder


def _durations(tmp_path, call):
    """Run `call` under a real recorder; return the duration_s of each tool_call it emitted."""
    import json

    p = tmp_path / "t.jsonl"
    with TraceRecorder(str(p), run_id="r"):
        call()
    return [
        (json.loads(x)["payload"].get("tool"), json.loads(x)["payload"].get("duration_s"))
        for x in p.read_text().splitlines()
        if '"tool_call"' in x
    ]


# Factories whose cost is a WAIT on something outside this process. Each MUST record duration_s.
_OUTBOUND = {
    "make_fetch_tool": lambda: make_fetch_tool(lambda url: "body")("https://example.com/x"),
    "make_web_search_tool": lambda: make_web_search_tool(lambda q: [])("a query"),
    "make_command_tool": lambda: make_command_tool(
        lambda cmd: types.SimpleNamespace(
            exit_code=0, stdout="", stderr="", duration_ms=None
        )
    )("ls"),
    "make_git_clone_tool": None,   # built per-test: it needs a root dir. See _build_git_clone.
}


def _build_git_clone(tmp_path):
    """`make_git_clone_tool` needs a root, so it cannot be a zero-arg lambda like the others."""
    from rlm_harness.tools import make_git_clone_tool

    def _runner(argv, **kw):
        return types.SimpleNamespace(exit_code=0, stdout="", stderr="", duration_ms=None)

    tool = make_git_clone_tool(str(tmp_path), _runner)
    return lambda: tool("https://example.com/r.git", "dest")

# ...and every other `make_*`, with the reason it does not time ITSELF. Named one at a time rather
# than pattern-matched: the guard is only worth having if each entry has to be argued for.
#
# WHAT THIS TABLE MEANS CHANGED IN 1.8.3. It used to list tools that carry NO duration at all. They
# all carry one now — `RLMTask` wraps every tool it hands the model and `record_tool_call` fills the
# field from that wrapper. So an entry here says only "does not scope its own window", which is the
# right answer for a tool with no inner boundary worth timing separately.
#
# The filesystem entries used to be argued as "sub-millisecond, and their refusal paths never touch
# anything". Both halves are retired: the refusal argument is reversed in README's "Which shipped
# tools carry a duration" (absent means unmeasured, not instant), and `make_grep_files_tool`'s
# measurement-based exemption was reopened by its own terms — it named "a pathological regex over a
# large tree" as what would reopen it, and a consumer's re-measurement across nine real repositories
# found median 744ms and max 6.3s on a 2,110-file repo against the n=146/median-0.029s this table
# used to cite. That tool alone is ~40% of sandbox execution time on that corpus.
_NOT_OUTBOUND = {
    "make_read_file_tool": "local fs read; timed by the task seam",
    "make_write_file_tool": "local fs write; timed by the task seam",
    "make_edit_file_tool": "local fs edit; timed by the task seam",
    "make_grep_files_tool": "local fs scan; timed by the task seam, and NOT cheap -- see above",
    "make_extract_archive_tool": "local archive extraction; timed by the task seam",
    # Host-side validators, never placed in a `tools=[...]` list and doing no I/O at all.
    "make_json_schema_validator": "host-side validator, no I/O",
    "make_schema_validator": "host-side validator, no I/O",
    # Side-effect-free BASES that deliberately record nothing at all — the consumer's own wrapper
    # owns the `record_tool_call`, and therefore owns passing `duration_s`.
    "make_model_tool": "base factory; records nothing (consumer's wrapper does)",
    "make_harness_tool": "base factory; records nothing (consumer's wrapper does)",
}


def test_every_shipped_factory_is_classified():
    """The guard that makes this table self-maintaining: a `make_*` reaching
    `rlm_harness.tools.__all__` with no entry in either table fails HERE, when it ships."""
    shipped = {n for n in tools_pkg.__all__ if n.startswith("make_")}
    unclassified = shipped - set(_OUTBOUND) - set(_NOT_OUTBOUND)
    assert not unclassified, (
        f"shipped but unclassified: {sorted(unclassified)} — add it to _OUTBOUND (and make it "
        f"record duration_s), or to _NOT_OUTBOUND with the reason it does not."
    )


@pytest.mark.parametrize("factory", sorted(_OUTBOUND))
def test_an_outbound_tool_records_its_duration(factory, tmp_path):
    build = _OUTBOUND[factory] or _build_git_clone(tmp_path)
    recorded = _durations(tmp_path, build)
    assert recorded, f"{factory} recorded no tool_call at all"
    for tool, duration in recorded:
        assert duration is not None, f"{factory} recorded {tool!r} with no duration_s"
        assert duration >= 0.0


def test_model_as_tool_records_its_duration(tmp_path):
    """The one model-backed tool the KIT records, so the kit owns its duration. It shipped
    without one; this is the pin."""
    from rlm_harness.sub_lm import model_as_tool

    tool = model_as_tool("openai/gpt-4o-mini", lambda prompt=None, **kw: ["answer"])
    recorded = _durations(tmp_path, lambda: tool("hi"))
    assert recorded and recorded[0][1] is not None


# --- the task seam: duration_s is filled without the tool asking ------------------------------
#
# Everything above tests a tool that measures ITSELF. Since 1.8.3 a tool that does not is timed
# anyway, by `_ensure_tool_timing` applied once at `RLMTask._build_rlm`. These drive the REAL
# `dspy.RLM.aforward` loop offline so the wrapper is exercised where it actually runs, not called
# by hand -- a hand-call would pass with the seam unwired.

import asyncio
import inspect
import json
import pathlib
import tempfile

from pydantic import BaseModel

import rlm_harness.runtime as rt
from rlm_harness import RLMConfig, RLMTask
from rlm_harness.testing import ScriptedInterpreter, assert_repl_safe, call, scripted_lm, submit
from rlm_harness.trace import _ensure_tool_timing, record_tool_call


class _Out(BaseModel):
    x: int


def _run_with_tool(tmp_path, tool, *, steps=None, turns=None):
    """Drive one offline run whose script dispatches `tool`; return the recorded tool_call events."""
    dummy = scripted_lm(turns or [{"reasoning": "r", "code": "c"}, {"reasoning": "s", "code": "S"}])
    rt.configure(RLMConfig(main_model="x", sub_model="x", interpreter="mock", observe=False),
                 main_lm=dummy, sub_lm=dummy)

    class T(RLMTask):
        signature = "q: str -> answer: _Out"
        output_field = "answer"
        output_model = _Out

    task = T(tools=[tool], interpreter=ScriptedInterpreter(
        list(steps or [call(tool.__name__)]) + [submit({"answer": {"x": 1}})]
    ))
    p = tmp_path / "t.jsonl"
    with TraceRecorder(str(p), run_id="r"):
        asyncio.run(task.arun(q="hi"))
    return [json.loads(x)["payload"] for x in p.read_text().splitlines()
            if json.loads(x)["type"] == "tool_call"]


def _untimed_tool():
    def slow_tool(n: int = 1) -> str:
        import time as _t
        _t.sleep(0.02)
        record_tool_call("slow_tool", args={"n": n}, ok=True)   # no duration_s of its own
        return "done"
    return slow_tool


def test_a_tool_that_never_measures_itself_is_timed_at_the_task_seam(tmp_path):
    """The 1.7.0 lesson one field over: a value every author must remember is missing for someone.

    Asserted `> 0.0`, never `is not None` -- 0.0 satisfies the weaker check while proving the clock
    was never read."""
    calls = _run_with_tool(tmp_path, _untimed_tool())
    assert len(calls) == 1
    assert calls[0]["duration_s"] > 0.0
    assert calls[0]["duration_s"] >= 0.02          # it really slept


def test_the_seam_records_no_event_of_its_own(tmp_path):
    """The wrapper publishes a start time and NOTHING else. A consumer whose tools delegate to this
    kit's factories refused to add durations for exactly this reason: a wrapper that RECORDED would
    emit a second `tool_call` and double `tool_calls`, `tool_ok` and everything derived from them."""
    calls = _run_with_tool(tmp_path, _untimed_tool())
    assert len(calls) == 1 and calls[0]["tool"] == "slow_tool"


def test_a_tool_that_measures_itself_keeps_its_own_number(tmp_path):
    """Explicit wins. A self-timing tool scopes the window more precisely than the seam can --
    `make_fetch_tool` starts its clock AFTER the SSRF check, `make_command_tool` keeps a
    runner-reported figure. The seam must never overwrite either."""
    def picky(n: int = 1) -> str:
        import time as _t
        _t.sleep(0.02)
        record_tool_call("picky", args={"n": n}, ok=True, duration_s=0.001)
        return "done"

    calls = _run_with_tool(tmp_path, picky)
    assert calls[0]["duration_s"] == 0.001          # NOT the ~0.02 the seam would have filled


def test_a_tool_called_outside_a_task_is_not_timed(tmp_path):
    """Unchanged behaviour, and the reason `tests/test_trace.py`'s direct-call assertions still
    hold: the wrapper exists only on what a task hands the model."""
    p = tmp_path / "t.jsonl"
    with TraceRecorder(str(p), run_id="r"):
        _untimed_tool()()
    # splitlines()[0] is `run_start`, not the tool_call -- reading it made an earlier version of
    # this test pass against a `record_tool_call` that stamped duration_s=99.0 unconditionally.
    events = [json.loads(x) for x in p.read_text().splitlines()]
    payloads = [e["payload"] for e in events if e["type"] == "tool_call"]
    assert len(payloads) == 1
    assert "duration_s" not in payloads[0]


def test_two_calls_in_one_run_get_independent_durations(tmp_path):
    """Each call re-publishes its own start, so the second is not charged for the first.

    The two sleeps are deliberately LOPSIDED and the bar is a ratio. An earlier version slept the
    same 0.02s twice and asserted `d1 < d0 + 0.015`, which failed 25% of the time under CPU
    contention -- `time.sleep` only ever overruns, and a preempted second call blew the slack by
    8x. A cumulative implementation reports d1 ~= d0 + d1, so a long-then-short pair separates the
    two by a factor, not by a constant a loaded runner can eat.

    NOTE this does NOT exercise the token RESET -- sequential calls each `set()` on entry, so
    dropping the reset keeps them green. The reset is pinned by the nesting test below."""
    naps = iter([0.30, 0.01])

    def lopsided(n: int = 1) -> str:
        import time as _t
        _t.sleep(next(naps))
        record_tool_call("lopsided", args={"n": n}, ok=True)
        return "done"

    calls = _run_with_tool(
        tmp_path, lopsided,
        steps=[call("lopsided"), call("lopsided")],
        turns=[{"reasoning": "a", "code": "1"}, {"reasoning": "b", "code": "2"},
               {"reasoning": "c", "code": "3"}],
    )
    assert len(calls) == 2
    assert all(c["duration_s"] > 0.0 for c in calls)
    # 0.30s then 0.01s. Correct: d1 is ~30x SMALLER than d0. Cumulative: d1 >= d0. Half of d0 is a
    # 0.15s margin -- the second call would have to be preempted for that long to false-fail.
    assert calls[1]["duration_s"] < calls[0]["duration_s"] / 2


def test_a_nested_tool_call_restores_the_outer_start(tmp_path):
    """What the token-based reset is actually for. When one tool calls another, the inner call
    publishes its own start; on the way out it must restore the OUTER one. Without the reset the
    outer tool's `duration_s` goes ABSENT, not short — the published name stays the inner tool's,
    so the outer's own record fails the name match and fills nothing.

    Sequential calls cannot show this -- they each `set()` on entry regardless."""
    def inner(n: int = 1) -> str:
        import time as _t
        _t.sleep(0.03)
        record_tool_call("inner", args={"n": n}, ok=True)
        return "in"

    wrapped_inner = _ensure_tool_timing(inner)

    def outer(n: int = 1) -> str:
        import time as _t
        _t.sleep(0.03)
        wrapped_inner()                   # nested: sets and (must) reset the start time
        _t.sleep(0.03)
        record_tool_call("outer", args={"n": n}, ok=True)
        return "out"

    calls = {c["tool"]: c for c in _run_with_tool(tmp_path, outer)}
    assert calls["inner"]["duration_s"] > 0.0
    # The outer window spans BOTH its own sleeps plus the inner call: ~0.09s. With the reset
    # dropped this KeyErrors -- the name match fails and no duration is written at all.
    assert calls["outer"]["duration_s"] >= 0.08


def test_double_wrapping_is_harmless(tmp_path):
    """A careful consumer may wrap its own tool and the task then wraps again.

    What is claimed is only that this does not break or double-record. WHICH start wins is not
    asserted, because it is not observable: the two wrappers nest immediately, so their start times
    differ by microseconds. An earlier version of this test named the inner one and could not tell
    the two policies apart."""
    calls = _run_with_tool(tmp_path, _ensure_tool_timing(_untimed_tool()))
    assert len(calls) == 1 and calls[0]["duration_s"] > 0.0


def test_the_seam_keeps_a_tool_repl_safe():
    """dspy builds the sandbox proxy from `inspect.signature(tool.func)` and its Tool metadata from
    `typing.get_type_hints`; both follow `__wrapped__`, the latter to reach the ORIGINAL module's
    globals -- which is load-bearing because every `tools/*.py` uses
    `from __future__ import annotations`, so the annotations `functools.wraps` copies are strings
    that resolve nowhere else."""
    tool = make_fetch_tool(lambda u: "body")
    wrapped = _ensure_tool_timing(tool)
    assert_repl_safe(wrapped)
    assert wrapped.__name__ == tool.__name__
    import dspy
    dspy.Tool(wrapped)                              # must not raise: aborts ALL registration if it does


def test_a_dspy_Tool_object_is_left_alone(tmp_path):
    """`mcp._make_tool` returns a `dspy.Tool`, a pydantic model with no `__name__`. Wrapping one
    leaves the wrapper called `timed` while `functools.wraps` copies the model's field dict onto it,
    so TWO abort the whole task with "Duplicate tool name 'timed'" and ONE registers as `timed` with
    its args collapsed to `{"kwargs": {}}` -- the `**kwargs` proxy bug `assert_repl_safe` exists to
    stop, which it cannot see here because it reads the copied pydantic fields.

    MCP tools already record their own duration, so passing them through loses nothing."""
    import inspect as _i

    import dspy

    def call(**kwargs):
        return "x"

    call.__signature__ = _i.Signature(
        [_i.Parameter("city", _i.Parameter.KEYWORD_ONLY, annotation=str)]
    )
    tools = [dspy.Tool(call, name=n, desc="d", args={"city": {"type": "string"}})
             for n in ("get_weather", "db_query")]
    wrapped = [_ensure_tool_timing(t) for t in tools]
    assert all(w is t for w, t in zip(wrapped, tools))       # untouched, not merely working

    rlm = dspy.RLM("q->a", tools=wrapped)                    # would raise ValueError if wrapped
    registered = rlm.tools if isinstance(rlm.tools, list) else list(rlm.tools.values())
    assert {getattr(x, "name", None) for x in registered} >= {"get_weather", "db_query"}


def test_a_nested_unwrapped_tool_is_not_charged_the_outer_clock(tmp_path):
    """Only the tools a task hands the model are wrapped. A COMPOSITE tool -- a consumer's tool
    that calls a kit tool inside itself -- would otherwise charge its whole window to every event
    recorded beneath it. Measured before the name check existed: two zero-cost `read_file` calls
    inside a 0.05s tool each reported 0.05s, tripling `compute_tool_waste.total_seconds`.

    That is WORSE than the `None` it replaced -- `None` is an honest unknown, this was a confident
    wrong answer -- so a name mismatch fills nothing and fails back to the unknown."""
    import tempfile

    from rlm_harness.tools import make_read_file_tool

    d = tempfile.mkdtemp()
    (pathlib.Path(d) / "a.py").write_text("x = 1\n")
    inner = make_read_file_tool(d)

    def composite(n: int = 1) -> str:
        import time as _t
        _t.sleep(0.05)
        inner("a.py")
        record_tool_call("composite", ok=True)
        return "ok"

    calls = {c["tool"]: c for c in _run_with_tool(tmp_path, composite)}
    assert "duration_s" not in calls["read_file"]            # honest unknown, not 0.05
    assert calls["composite"]["duration_s"] >= 0.05


def test_the_start_time_does_not_leak_when_a_tool_raises(tmp_path):
    """The reset is in a `finally`. Without it a raising tool leaves its start published and the
    NEXT tool inherits it."""
    def boom(n: int = 1) -> str:
        raise RuntimeError("x")

    _ensure_tool_timing(boom)
    try:
        _ensure_tool_timing(boom)()
    except RuntimeError:
        pass
    from rlm_harness.trace import _tool_started
    assert _tool_started.get() is None


def test_the_start_time_is_per_thread_not_shared():
    """A ContextVar, not a module global: two `RLMTask`s driven from two threads each own an
    interpreter and can call tools concurrently, and a global start would interleave them.

    The two threads are SYNCHRONISED so that A reads its start while B is still inside its own
    call. Without that, a module global emulating token set/reset passes too -- B restores A's
    value on the way out, and a version of this test that read after B had finished could not tell
    the two apart."""
    import threading
    import time as _t

    b_inside = threading.Event()
    a_read = threading.Event()
    seen = {}

    def probe_a(n: int = 1) -> str:
        from rlm_harness.trace import _tool_started
        b_inside.wait(2.0)                 # let B publish its start first
        seen["a"] = _tool_started.get()    # ...and read while B is STILL inside
        a_read.set()
        return "ok"

    def probe_b(n: int = 1) -> str:
        from rlm_harness.trace import _tool_started
        b_inside.set()
        a_read.wait(2.0)                   # stay inside until A has read
        seen["b"] = _tool_started.get()
        return "ok"

    threads = [threading.Thread(target=_ensure_tool_timing(probe_a)),
               threading.Thread(target=_ensure_tool_timing(probe_b))]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=5.0)

    assert seen["a"] is not None and seen["b"] is not None
    assert seen["a"][0] == "probe_a"       # a module global would read "probe_b" here
    assert seen["b"][0] == "probe_b"
    _t.sleep(0)


def test_a_coroutine_function_is_passed_through(tmp_path):
    """Half of the registration fix, and deleting it survives every OTHER test in the suite.

    dspy branches on `inspect.iscoroutinefunction(tool.func)`, which does NOT follow `__wrapped__`.
    Wrapping an async tool therefore makes dspy take the SYNC path and raise "You are calling
    __call__ on an async tool" -- turning a run that completes into an `RLMTaskError`. Async tools
    are forbidden by CLAUDE.md and degrade to the model seeing `<coroutine object ...>`; the point
    here is that the seam must not make that worse."""
    async def async_tool(n: int = 1) -> str:
        return "never awaited"

    assert _ensure_tool_timing(async_tool) is async_tool


def test_an_async_tool_still_completes_its_run(tmp_path):
    """The end-to-end half: with the passthrough gone this raises instead of returning.

    The "coroutine was never awaited" warning this provokes IS the documented degradation -- dspy's
    interpreter calls a tool synchronously on both the forward and aforward paths, so an `async def`
    tool's body never runs and the model receives the literal `<coroutine object ...>`. Filtered
    rather than avoided, because reproducing it is the point."""
    async def async_tool(n: int = 1) -> str:
        return "never awaited"

    import gc
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calls = _run_with_tool(tmp_path, async_tool)   # must not raise RLMTaskError
        gc.collect()          # retire the un-awaited coroutine INSIDE the suppression, so the
                              # documented degradation does not leave a warning for a later reader
                              # to chase. Collecting it at teardown instead is what produced one.
    assert calls == []                                 # records nothing; the run still finishes


def test_shapes_functools_wraps_cannot_carry_are_passed_through():
    """A callable instance and a `functools.partial` have no `__name__` for `wraps` to copy and no
    `__globals__` for `get_type_hints` to reach, so wrapping changes what dspy registers. Passing
    them through loses a duration; wrapping them would lose the tool."""
    import functools as _f

    class Callable_:
        def __call__(self, n: int = 1) -> str:
            return "x"

    def plain(a: int, n: int = 1) -> str:
        return "x"

    inst = Callable_()
    part = _f.partial(plain, 1)
    assert _ensure_tool_timing(inst) is inst
    assert _ensure_tool_timing(part) is part


def test_a_bound_method_is_timed():
    """`inspect.ismethod` is in the allowlist on purpose -- a consumer may hand over `self.tool`."""
    class Holder:
        def probe(self, n: int = 1) -> str:
            import time as _t
            _t.sleep(0.02)
            record_tool_call("probe", ok=True)
            return "x"

    h = Holder()
    wrapped = _ensure_tool_timing(h.probe)
    assert wrapped is not h.probe
    p = pathlib.Path(tempfile.mkdtemp()) / "t.jsonl"
    with TraceRecorder(str(p), run_id="r"):
        wrapped()
    payload = next(json.loads(x)["payload"] for x in p.read_text().splitlines()
                   if json.loads(x)["type"] == "tool_call")
    assert payload["duration_s"] >= 0.02


def test_an_explicit_zero_duration_is_kept_not_overwritten(tmp_path):
    """`0.0` is a MEASUREMENT; only `None` means "not measured". The fill therefore tests `is None`,
    not falsiness -- with `if not duration_s` a tool reporting a real zero silently gets the seam's
    number instead, which is the exact None-vs-measured confusion this release exists to end."""
    def instant(n: int = 1) -> str:
        import time as _t
        _t.sleep(0.02)
        record_tool_call("instant", ok=True, duration_s=0.0)
        return "x"

    calls = _run_with_tool(tmp_path, instant)
    assert calls[0]["duration_s"] == 0.0             # NOT the ~0.02 the seam would have filled


def test_a_refusal_carries_a_duration_through_a_task(tmp_path):
    """The release's headline reversal, which had no test until a mutation proved it.

    A refusal used to record NO duration, on the argument that a blocked URL never touched the
    network. That rule is reversed: `None` means "nobody measured", so spending it on "measured,
    and it was instant" makes the two indistinguishable. Skipping the fill on `ok=False` kept all
    887 tests green, so the reversal was stated in five places and enforced in none."""
    from rlm_harness.tools import make_fetch_tool

    tool = make_fetch_tool(lambda u: "body")

    def refuse(n: int = 1) -> str:
        return tool("http://127.0.0.1/admin")        # SSRF-refused: never leaves the process

    refuse.__name__ = "fetch_url"                    # record name == wrapped name, as the kit's is
    calls = _run_with_tool(tmp_path, refuse)
    assert calls[0]["ok"] is False
    assert "duration_s" in calls[0]                  # present...
    assert calls[0]["duration_s"] >= 0.0             # ...and a true ~0, not an absent field


def test_the_fill_keeps_sub_millisecond_resolution(tmp_path):
    """Every other seam test sleeps >= 0.02s, so rounding the fill to milliseconds stayed green
    while making every local filesystem tool report 0.0 -- "measured, and it was free", which is
    the None-vs-measured-zero confusion this release exists to end, one level down."""
    def instant(n: int = 1) -> str:
        # A small, RELIABLE amount of work rather than nothing at all. With no work the measured
        # value came in at 1e-06 -- one unit of the payload's 6dp rounding -- so the lower bound
        # held only because of incidental plumbing overhead and would go red on a faster host.
        sum(range(20000))
        record_tool_call("instant", ok=True)
        return "x"

    calls = _run_with_tool(tmp_path, instant)
    assert 0.0 < calls[0]["duration_s"] < 0.001


def test_the_name_match_compares_by_value_not_identity(tmp_path):
    """A tool name built at RUNTIME is a different string object each time. Comparing with `is`
    passes every test here -- each is a compile-time constant and therefore interned -- and
    silently stops filling for a consumer that names its tools from data."""
    parts = ["py", "thon"]
    lang = "".join(parts)                             # built at runtime, not interned
    name = f"read_{lang}"

    def probe(n: int = 1) -> str:
        import time as _t
        _t.sleep(0.02)
        record_tool_call(f"read_{lang}", ok=True)     # equal value, distinct object
        return "x"

    probe.__name__ = name
    calls = _run_with_tool(tmp_path, probe, steps=[call(name)])
    assert calls[0]["duration_s"] >= 0.02


def test_the_published_name_is_dunder_name_not_qualname(tmp_path):
    """dspy reads `__name__`; a consumer renaming a tool sets that and often not `__qualname__`.
    Taking the qualname's tail instead happens to agree for every tool in this repo."""
    def probe(n: int = 1) -> str:
        import time as _t
        _t.sleep(0.02)
        record_tool_call("renamed_tool", ok=True)
        return "x"

    probe.__name__ = "renamed_tool"                   # __qualname__ still ends in "probe"
    assert probe.__qualname__.rsplit(".", 1)[-1] != probe.__name__
    calls = _run_with_tool(tmp_path, probe, steps=[call("renamed_tool")])
    assert calls[0]["duration_s"] >= 0.02


def test_every_tool_on_a_task_is_timed_not_just_the_first(tmp_path):
    """The release's headline is "every tool a task hands the model", and until this test nothing
    exercised the seam at more than ONE tool -- `_run_with_tool` always passes a single-element
    list, so timing only `resolved_tools[0]` kept all 891 tests green. The shipping configuration
    is several tools on one task."""
    def alpha(n: int = 1) -> str:
        import time as _t
        _t.sleep(0.02)
        record_tool_call("alpha", ok=True)
        return "a"

    def beta(n: int = 1) -> str:
        import time as _t
        _t.sleep(0.02)
        record_tool_call("beta", ok=True)
        return "b"

    dummy = scripted_lm([{"reasoning": "a", "code": "1"}, {"reasoning": "b", "code": "2"},
                         {"reasoning": "c", "code": "3"}])
    rt.configure(RLMConfig(main_model="x", sub_model="x", interpreter="mock", observe=False),
                 main_lm=dummy, sub_lm=dummy)

    class T(RLMTask):
        signature = "q: str -> answer: _Out"
        output_field = "answer"
        output_model = _Out

    task = T(tools=[alpha, beta], interpreter=ScriptedInterpreter(
        [call("alpha"), call("beta"), submit({"answer": {"x": 1}})]))
    p = tmp_path / "t.jsonl"
    with TraceRecorder(str(p), run_id="r"):
        asyncio.run(task.arun(q="hi"))
    got = {json.loads(x)["payload"]["tool"]: json.loads(x)["payload"]
           for x in p.read_text().splitlines() if json.loads(x)["type"] == "tool_call"}
    assert set(got) == {"alpha", "beta"}
    assert got["alpha"]["duration_s"] >= 0.02
    assert got["beta"]["duration_s"] >= 0.02        # the SECOND tool, which a [0]-only seam misses


def test_the_whole_task_guard_inspects_the_list_dspy_receives(tmp_path, monkeypatch):
    """`assert_task_repl_safe`'s first listed job is catching duplicate tool names, which abort
    registration for every tool. It reads `resolved_tools`, and since 1.8.3 `_build_rlm` hands dspy
    a WRAPPED derivative -- so a seam wrapper that lost `__name__` would abort construction while
    the guard still passed. That is this change's own first-round defect, invisible to the kit's own
    whole-task check.

    Asserting the guard merely does not raise proves nothing (it passes either way). Instead the
    seam is replaced with one that DOES collide, and the guard must then object -- which it can only
    do if it is looking at the wrapped list."""
    import rlm_harness.trace as trace_mod
    from rlm_harness.testing import assert_task_repl_safe

    def alpha(n: int = 1) -> str:
        return "a"

    def beta(n: int = 1) -> str:
        return "b"

    dummy = scripted_lm([{"reasoning": "a", "code": "1"}])
    rt.configure(RLMConfig(main_model="x", sub_model="x", interpreter="mock", observe=False),
                 main_lm=dummy, sub_lm=dummy)

    class T(RLMTask):
        signature = "q: str -> answer: _Out"
        output_field = "answer"
        output_model = _Out

    task = T(tools=[alpha, beta])
    assert_task_repl_safe(task)                      # the real seam keeps both names distinct

    def colliding(tool):
        def timed(n: int = 1) -> str:
            return tool(n)
        timed.__name__ = "collide"                   # what a wrapper that drops __name__ produces
        return timed

    monkeypatch.setattr(trace_mod, "_ensure_tool_timing", colliding)
    with pytest.raises(AssertionError, match="[Dd]uplicate"):
        assert_task_repl_safe(task)


def test_a_filled_duration_has_an_upper_bound_too(tmp_path):
    """Every other assertion here is a lower bound, so a systematic inflation -- multiplying the
    fill by two, say -- shipped green. A confident wrong answer is the failure this whole release
    argues is worse than an honest unknown, so the number is bounded from above as well."""
    def naps(n: int = 1) -> str:
        import time as _t
        _t.sleep(0.10)
        record_tool_call("naps", ok=True)
        return "x"

    calls = _run_with_tool(tmp_path, naps)
    assert 0.10 <= calls[0]["duration_s"] < 0.10 * 1.8


def test_the_name_match_is_equality_not_a_prefix(tmp_path):
    """`tool.startswith(published)` survives every other test here. A tool recording under a name
    that merely EXTENDS the wrapped one would then be charged that tool's window."""
    def read(n: int = 1) -> str:
        import time as _t
        _t.sleep(0.02)
        record_tool_call("read_file_extended", ok=True)   # starts with the wrapped name "read"
        return "x"

    calls = _run_with_tool(tmp_path, read, steps=[call("read")])
    assert "duration_s" not in calls[0]


def test_a_generator_function_records_no_duration(tmp_path):
    """The shape that survived four review rounds, because nothing pinned it and the guide
    described it wrongly.

    A generator function IS wrapped -- `inspect.isfunction` is True -- but calling it only builds
    the generator object. The wrapper releases the start time on the way out, and the body (with
    its `record_tool_call`) runs later, finding nothing published. So this fails to an ABSENT
    field, like the shapes that are never wrapped, rather than to a small number. A plain function
    that merely RETURNS a generator expression is a different shape and is timed normally."""
    def gen_tool(n: int = 1):
        import time as _t
        _t.sleep(0.03)
        record_tool_call("gen_tool", ok=True)
        yield "x"

    def returns_genexp(n: int = 1):
        import time as _t
        _t.sleep(0.03)
        record_tool_call("returns_genexp", ok=True)
        return (i for i in range(3))          # a generator EXPRESSION, from a plain function

    assert not inspect.isgeneratorfunction(returns_genexp)
    # It IS wrapped -- skipping generator functions outright would be observationally identical
    # for `duration_s`, so without this the docstring's first claim has no witness.
    assert _ensure_tool_timing(gen_tool) is not gen_tool

    p = tmp_path / "t.jsonl"
    with TraceRecorder(str(p), run_id="r"):
        list(_ensure_tool_timing(gen_tool)())        # consume it, so the body really runs
        _ensure_tool_timing(returns_genexp)()
    got = {json.loads(x)["payload"]["tool"]: json.loads(x)["payload"]
           for x in p.read_text().splitlines() if json.loads(x)["type"] == "tool_call"}
    assert "duration_s" not in got["gen_tool"]       # absent, NOT "real but small"
    assert got["returns_genexp"]["duration_s"] >= 0.03
