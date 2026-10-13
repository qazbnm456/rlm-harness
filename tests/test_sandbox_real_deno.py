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
