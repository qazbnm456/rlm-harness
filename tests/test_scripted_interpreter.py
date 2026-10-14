"""The scripted-interpreter forward-path test seam (`rlm_harness.testing`).

Unlike `test_integration_dspy.py` (which builds the RLM but never forwards), this drives the REAL
`dspy.RLM.aforward` loop OFFLINE (no live model, no Deno) via a scripted DummyLM + `ScriptedInterpreter`
injected through `RLMTask(interpreter=...)`. This is the layer that catches wiring bugs a construction
test can't (e.g. a prompt naming a tool the interpreter can't resolve). Skipped if dspy is absent.
"""

import asyncio

import pytest

dspy = pytest.importorskip("dspy")

from pydantic import BaseModel

import rlm_harness.runtime as rt
from rlm_harness import RLMConfig, RLMTask
from rlm_harness.testing import ScriptedInterpreter, call, scripted_lm, submit
from rlm_harness.trace import (
    EVENT_RESULT,
    EVENT_TOOL_CALL,
    TraceRecorder,
    load_events,
    record_tool_call,
)


class _Out(BaseModel):
    x: int


def _configure(turns):
    dummy = scripted_lm(turns)
    rt.configure(RLMConfig(main_model="x", sub_model="x", interpreter="mock", observe=False),
                 main_lm=dummy, sub_lm=dummy)
    return dummy


def test_interpreter_override_is_stored():
    _configure([{"reasoning": "r", "answer": "{}"}])

    class T(RLMTask):
        signature = "q: str -> answer: _Out"
        output_field = "answer"
        output_model = _Out

    si = ScriptedInterpreter()
    assert T(interpreter=si)._interpreter is si


def test_scripted_forward_dispatches_a_real_tool_and_submits(tmp_path):
    """The whole planner -> tool -> SUBMIT -> validated result chain runs offline; the injected tool
    ACTUALLY executes inside the loop (its tracing records a real tool_call), and the SUBMIT coerces."""
    _configure([
        {"reasoning": "call the tool", "code": "print(mark(x=1))"},
        {"reasoning": "submit the answer", "code": "SUBMIT(answer={'x': 5})"},
    ])
    seen = []

    def mark(x: int) -> str:
        """Record a mark."""
        record_tool_call("mark", args={"x": x}, ok=True)
        seen.append(x)
        return f"marked {x}"

    class T(RLMTask):
        signature = "q: str -> answer: _Out"
        output_field = "answer"
        output_model = _Out
        instructions = "Call mark, then SUBMIT."
        tools = [mark]

    si = ScriptedInterpreter([call("mark", x=1), submit({"answer": {"x": 5}})])
    task = T(interpreter=si)

    path = str(tmp_path / "t.jsonl")
    with TraceRecorder(path, run_id="r"):
        result = asyncio.run(task.arun(q="hi"))

    assert isinstance(result, _Out) and result.x == 5     # SUBMIT coerced into the output_model
    assert seen == [1]                                    # the REAL tool ran inside the loop
    assert si.calls                                       # execute() was driven on the injected double
    ev = load_events(path)
    assert any(e["type"] == EVENT_TOOL_CALL and e["payload"].get("tool") == "mark" for e in ev)
    assert any(e["type"] == EVENT_RESULT for e in ev)


def test_dict_step_submits_without_a_tool(tmp_path):
    """A bare dict step is a SUBMIT: a one-turn run that finalizes immediately."""
    _configure([{"reasoning": "submit", "code": "SUBMIT(answer={'x': 9})"}])

    class T(RLMTask):
        signature = "q: str -> answer: _Out"
        output_field = "answer"
        output_model = _Out

    task = T(interpreter=ScriptedInterpreter([{"answer": {"x": 9}}]))
    with TraceRecorder(str(tmp_path / "t.jsonl"), run_id="r"):
        result = asyncio.run(task.arun(q="hi"))
    assert result.x == 9


def test_dspys_own_setup_executions_consume_no_step_and_stay_out_of_calls(tmp_path):
    """`.calls` is the MODEL's turns, on every dspy, and a script's length is its turn count.

    Drives the real `aforward` loop rather than calling `execute` by hand, because the thing
    under test is whether DSPY's own pre-loop execution is accounted separately: dspy 3.4.1 added
    one (`RLM._setup_facade` installing the sandbox `dspy` shim) and it silently ate step 0 of
    every script written against 3.4.0. A hand-rolled `execute(SHIM_SETUP, ...)` call would assert
    the kit against its own idea of what dspy does; this asserts it against what dspy DOES.

    The facade count is asserted against dspy's CAPABILITY, not a version number, so this stays a
    measurement on a dspy that renames the module or backports the behaviour.
    """
    _configure([{"reasoning": "submit", "code": "SUBMIT(answer={'x': 3})"}])

    class T(RLMTask):
        signature = "q: str -> answer: _Out"
        output_field = "answer"
        output_model = _Out

    si = ScriptedInterpreter([submit({"answer": {"x": 3}})])
    with TraceRecorder(str(tmp_path / "t.jsonl"), run_id="r"):
        result = asyncio.run(T(interpreter=si).arun(q="hi"))

    # A ONE-step script must finalize on its one step: this is the assertion that went red.
    assert result.x == 3
    # `.calls` holds the model's code and nothing else. A consumer reads `.calls[0]`.
    assert si.calls == ["SUBMIT(answer={'x': 3})"]

    try:
        import dspy.primitives.facade  # noqa: F401
    except ImportError:
        installs_a_facade = False
    else:
        installs_a_facade = True
    if installs_a_facade:
        assert si.facade_installs == 1, (
            "dspy exposes its sandbox facade but installed none on this double: either the "
            "install stopped happening or `is_facade_install` stopped recognising it, and the "
            "step accounting above is now green for the wrong reason"
        )
    else:
        assert si.facade_installs == 0
