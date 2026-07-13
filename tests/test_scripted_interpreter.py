"""The scripted-interpreter forward-path test seam (`rlm_kit.testing`).

Unlike `test_integration_dspy.py` (which builds the RLM but never forwards), this drives the REAL
`dspy.RLM.aforward` loop OFFLINE — no live model, no Deno — via a scripted DummyLM + `ScriptedInterpreter`
injected through `RLMTask(interpreter=...)`. This is the layer that catches wiring bugs a construction
test can't (e.g. a prompt naming a tool the interpreter can't resolve). Skipped if dspy is absent.
"""

import asyncio

import pytest

dspy = pytest.importorskip("dspy")

from pydantic import BaseModel  # noqa: E402

import rlm_kit.runtime as rt  # noqa: E402
from rlm_kit import RLMConfig, RLMTask  # noqa: E402
from rlm_kit.testing import ScriptedInterpreter, call, scripted_lm, submit  # noqa: E402
from rlm_kit.trace import (  # noqa: E402
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
    """A bare dict step is a SUBMIT — a one-turn run that finalizes immediately."""
    _configure([{"reasoning": "submit", "code": "SUBMIT(answer={'x': 9})"}])

    class T(RLMTask):
        signature = "q: str -> answer: _Out"
        output_field = "answer"
        output_model = _Out

    task = T(interpreter=ScriptedInterpreter([{"answer": {"x": 9}}]))
    with TraceRecorder(str(tmp_path / "t.jsonl"), run_id="r"):
        result = asyncio.run(task.arun(q="hi"))
    assert result.x == 9
