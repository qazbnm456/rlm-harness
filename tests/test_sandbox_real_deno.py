"""The two sandbox-death fixes, against a REAL Deno/Pyodide sandbox.

Every other sandbox test monkeypatches ``PythonInterpreter.execute``, which is exactly why both
bugs below shipped: a fake ``execute`` cannot exit, and a fake kill cannot end a dspy session.
These need Deno and skip without it (CI has none; see ``docs/VERIFY.md``), so run them locally
on any change to ``sandbox.py`` or the runner shim in ``_dspy_compat.py``.

1. A cell whose exception carries an argument JSON cannot encode (``{}[:5]`` raises
   ``KeyError(slice(...))``) used to kill Deno while the error was being REPORTED. It must come
   back as the ordinary recoverable error, with the session alive.
2. A turn timeout used to promise a restart that never happened: the next cell got
   ``PythonInterpreter session has ended``, which is terminal.
"""

from __future__ import annotations

import importlib.util
import shutil

import pytest

pytestmark = [
    pytest.mark.skipif(importlib.util.find_spec("dspy") is None, reason="dspy not installed"),
    pytest.mark.skipif(shutil.which("deno") is None, reason="needs a real Deno on PATH"),
]


@pytest.fixture
def interp():
    from rlm_harness.sandbox import build_interpreter

    made = []

    def make(**kw):
        i = build_interpreter("pyodide", **kw)
        made.append(i)
        return i

    yield make
    for i in made:
        i.shutdown()


@pytest.mark.parametrize(
    ("cell", "shows"),
    [
        ("d = {'a': 1}; d[:500]", "slice(None, 500, None)"),
        ("import json; json.loads('{\"t\": 1}')[0:3]", "slice(0, 3, None)"),
        ("b'\\xff'.decode('utf-8')", "UnicodeDecodeError"),
        ("{}[b'x']", "b'x'"),
        ("raise ValueError(object())", "object object at"),
        ("import math; raise ValueError(math.nan)", "nan"),
        ("raise ValueError(float('inf'))", "inf"),
        ("x = []; x.append(x); raise ValueError(x)", "[[...]]"),
    ],
)
def test_an_unencodable_exception_is_a_recoverable_error_and_the_session_lives(interp, cell, shows):
    from dspy.primitives.code_interpreter import CodeExecutionError

    i = interp()
    i.execute("kept = 41")
    with pytest.raises(CodeExecutionError) as caught:
        i.execute(cell)
    assert shows in str(caught.value)
    # Same process, same namespace: nothing was lost.
    assert i.execute("print(kept + 1)").strip() == "42"


def test_ordinary_errors_report_exactly_as_before(interp):
    from dspy.primitives.code_interpreter import CodeExecutionError

    i = interp()
    with pytest.raises(CodeExecutionError) as caught:
        i.execute("1/0")
    assert str(caught.value) == "ZeroDivisionError: ['division by zero']"


def test_a_turn_timeout_hands_the_model_a_working_sandbox_next_turn(interp):
    from dspy.primitives.code_interpreter import CodeExecutionError

    i = interp(turn_timeout_s=3)
    calls = []
    i.tools["echo"] = lambda text: calls.append(text) or f"echo:{text}"
    i.execute("kept = 1")
    with pytest.raises(CodeExecutionError, match="per-turn"):
        i.execute("while True: pass")
    # Fresh state, as the message promises, and tools registered again.
    assert i.execute("print('kept' in globals())").strip() == "False"
    assert i.execute("print(echo(text='hi'))").strip() == "echo:hi"
    assert calls == ["hi"]
    # And a second timeout recovers the same way: the reset is not one-shot.
    with pytest.raises(CodeExecutionError, match="per-turn"):
        i.execute("while True: pass")
    assert i.execute("print(2 + 2)").strip() == "4"


@pytest.mark.parametrize("payload", ["{1, 2}", "b'raw'", "float('nan')"])
def test_a_submit_that_cannot_serialize_is_refused_not_stringified(payload):
    """A SUBMIT payload is the ANSWER: a repr must never be accepted as the value."""
    from dspy.primitives.code_interpreter import FinalOutput

    from rlm_harness.sandbox import build_interpreter

    i = build_interpreter("pyodide")
    i.output_fields = [{"name": "answer", "type": "str"}]
    try:
        out = i.execute(f"SUBMIT(answer={payload})")
        assert isinstance(out, FinalOutput)
        # Not a dict: dspy's RLM loop answers a recoverable "FINAL returned str" error.
        assert isinstance(out.output, str) and "not JSON-serializable" in out.output, out.output
        assert i.execute("print('alive')").strip() == "alive"
        ok = i.execute("SUBMIT(answer='fine')")
        assert ok.output == {"answer": "fine"}
    finally:
        i.shutdown()


def test_a_deleted_runner_copy_is_rewritten_for_the_next_interpreter(interp):
    import os

    first = interp()
    path = first._get_runner_path()
    assert path.endswith("runner.js") and os.path.isfile(path)
    os.remove(path)
    second = interp()
    assert os.path.isfile(second._get_runner_path())
    assert second.execute("print(1)").strip() == "1"


def test_a_cancel_stays_terminal(interp):
    import threading

    from rlm_harness.sandbox import SandboxCancelled

    ev = threading.Event()
    i = interp(cancel_event=ev)
    assert i.execute("print(1)").strip() == "1"
    timer = threading.Timer(1.0, ev.set)
    timer.start()
    with pytest.raises(SandboxCancelled):
        i.execute("while True: pass")
    assert i._session_ended is True


def test_a_turn_timeout_restart_brings_back_dspys_sandbox_session(interp):
    """A restart must bring back dspy's sandbox session, because the PROMPT still offers sub-agents.

    `RLM` bootstraps a session of dspy in the sandbox before the turn loop and tells the model it may
    build sub-agents. A turn timeout resets the session: the fresh Deno re-mounts files and
    re-registers tools, because those are things dspy redoes per session, but the bootstrap was an
    `execute` and dspy never runs it again. The probe is `__dspy_guest_spec`, which only the
    bootstrap sets. `import dspy` is not a probe: the sandbox restores a snapshot in which dspy is
    already imported, so it succeeds after a restart that lost the session.

    Installed through dspy's own `SandboxSession`, not a hand-written bootstrap, so this asserts the
    replay against what dspy actually does.
    """
    from dspy.primitives import facade
    from dspy.primitives.code_interpreter import CodeExecutionError

    i = interp(turn_timeout_s=10)
    session = facade.SandboxSession({}, lm=None, reserve=lambda n=1: None)
    session.install(i, import_dspy=True)
    probe = "print('__dspy_guest_spec' in globals())"
    assert i.execute(probe).strip() == "True"
    i.execute("leftover = 1")

    with pytest.raises(CodeExecutionError, match="per-turn"):
        i.execute("while True: pass")

    # The session really did restart (the control: without it the assertion below is vacuous).
    assert i.execute("print('leftover' in globals())").strip() == "False"
    assert i.execute(probe).strip() == "True"
    session.close()


# ---- a whole run on the real sandbox, through the fake OpenAI-compatible server ------------------


def _deno_run(fake_openai, tmp_path, **config):
    import asyncio

    import rlm_harness
    from rlm_harness import RLMConfig, RLMTask, TraceRecorder, load_events

    class _Task(RLMTask):
        signature = "q: str -> answer: str"
        output_field = "answer"

    rlm_harness.configure(RLMConfig(main_model="qwen/qwen3-next", sub_model="qwen/qwen3-next",
                                    base_url=fake_openai.base_url, api_key="k", interpreter="pyodide",
                                    max_iterations=3, **config))
    fake_openai.script = [{"plan": "look", "code": "print(len(q))"},
                          {"plan": "done", "code": "SUBMIT('42')"}]
    path = str(tmp_path / "t.jsonl")
    with TraceRecorder(path, run_id="r"):
        out = asyncio.run(_Task().arun(q="hello"))
    steps = [e["payload"] for e in load_events(path) if e["type"] == "main_step"]
    system = fake_openai.requests[0]["body"]["messages"][0]["content"]
    return out, steps, system


def test_every_turn_on_the_sandbox_records_how_long_it_ran(fake_openai, tmp_path):
    """dspy runs each turn with its rebind prelude in front; the duration is matched to the turn the
    trajectory records, which has none, so both turns carry one."""
    out, steps, system = _deno_run(fake_openai, tmp_path)
    assert out == "42"
    assert "import dspy" in system, "the control: sub-agents are on, so the prelude is in play"
    assert [s["output"].strip() for s in steps[:1]] == ["5"]
    assert all(isinstance(s.get("exec_duration_s"), float) for s in steps), steps


def test_sub_agents_off_builds_no_bundle_and_offers_none(fake_openai, tmp_path, monkeypatch):
    """With sub-agents off dspy's `provide_dspy` is never reached: no bundle is built and the sandbox
    is granted nothing for one. The prompt offers no sub-agents, and the run is otherwise the same."""
    from dspy.primitives import sandbox_bundle

    def refuse(*args, **kwargs):
        raise AssertionError("the sandbox bundle was built with sub-agents off")

    monkeypatch.setattr(sandbox_bundle, "prepare_bundle", refuse)
    out, steps, system = _deno_run(fake_openai, tmp_path, sandbox_sub_agents=False)
    assert out == "42"
    assert "import dspy" not in system
    assert all(isinstance(s.get("exec_duration_s"), float) for s in steps), steps


def test_a_sub_agent_built_in_the_sandbox_is_recorded_as_a_sub_call(fake_openai, tmp_path):
    """Model code that imports dspy and runs a `Predict` calls the run's sub-LM on the host through
    dspy's sandbox session. That escalation goes through the kit's sub-LM wrapper like `llm_query`,
    so it is recorded."""
    import asyncio

    import rlm_harness
    from rlm_harness import RLMConfig, RLMTask, TraceRecorder, load_events

    class _Task(RLMTask):
        signature = "q: str -> answer: str"
        output_field = "answer"

    rlm_harness.configure(RLMConfig(main_model="qwen/qwen3-next", sub_model="qwen/qwen3-next",
                                    base_url=fake_openai.base_url, api_key="k", interpreter="pyodide",
                                    max_iterations=3))
    fake_openai.script = [
        {"plan": "ask", "code": 'import dspy\nprint(dspy.Predict("q -> a")(q="hi").a)'},
        {"a": "from the sub-agent"},
        {"plan": "done", "code": "SUBMIT('42')"},
    ]
    path = str(tmp_path / "t.jsonl")
    with TraceRecorder(path, run_id="r"):
        assert asyncio.run(_Task().arun(q="hello")) == "42"
    events = load_events(path)
    step = next(e["payload"] for e in events if e["type"] == "main_step")
    assert "from the sub-agent" in step["output"], step["output"]
    calls = [e["payload"] for e in events if e["type"] == "sub_call"]
    assert len(calls) == 1 and "from the sub-agent" in calls[0]["raw"], calls
