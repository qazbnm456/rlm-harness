"""Test support for driving the RLM forward path OFFLINE — no live model, no Deno, no network.

``dspy.RLM`` normally runs the model's Python inside a sandboxed interpreter (pyodide/deno). That makes
the *forward* path (planner turn -> tool call -> SUBMIT -> validated result) expensive to test: it needs
a paid model and a Deno subprocess, so the kit's own tests and every consumer stop at ``_build_rlm()``
(construction) and never exercise the loop. But the loop is exactly where wiring bugs hide — a prompt
that names a tool ``foo`` while the tool registered as ``foo_tool`` is a ``NameError`` no construction
test can see.

``ScriptedInterpreter`` closes that gap. It is a ``dspy`` ``CodeInterpreter`` test double that runs a
fixed SCRIPT instead of executing model-written code: ``dspy.RLM`` injects the REAL tools onto its
``.tools`` dict, and each ``execute()`` runs the next scripted STEP — which may DISPATCH a real tool (so
its tracing runs for real) or SUBMIT a final result (terminating the loop). Paired with ``scripted_lm``
(a ``DummyLM`` whose canned turns parse under the kit's JSON adapter) and injected via
``RLMTask(interpreter=...)``, it drives the whole ``planner -> tools -> result`` chain with zero cost.

This module imports ``dspy`` LAZILY (inside functions), so ``import rlm_kit.testing`` stays cheap and the
``import rlm_kit`` / dspy-free-module invariants are untouched. It is a TEST seam: the injected
interpreter bypasses ``sandbox.build_interpreter`` (and therefore the insecure-interpreter guard) exactly
like an injected ``DummyLM`` bypasses the real model — the caller supplies the double explicitly and owns
it. The default string path (``RLMConfig(interpreter=...)`` -> ``build_interpreter``) is unchanged and
keeps the guard.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Union

# A step is one execute() worth of behaviour. It is one of:
#   - a ``dict``     -> SUBMIT it as the run's final output (``{output_field: value}``); ends the loop.
#   - a ``str``      -> the REPL output for that turn (non-terminal — the next planner turn sees it).
#   - a ``callable`` -> called ``step(tools, variables)``; its return is interpreted by the SAME rules
#                       (dict -> submit, str -> output), or a dspy ``FinalOutput`` is passed through.
Step = Union[dict, str, Callable[[dict, dict], Any]]


class ScriptedInterpreter:
    """A scripted ``dspy`` ``CodeInterpreter`` double for offline forward-path tests.

    Build it with a list of STEPS (see ``Step``); one step is consumed per ``execute()`` call, in order.
    When the script is exhausted it returns ``""`` forever (a non-terminal no-op) so a loop that never
    reaches a SUBMIT step runs to its iteration cap — useful for budget-exhaustion tests.

    ``.calls`` records the code strings ``dspy`` asked to execute, in order, for assertions. ``.tools``
    is populated by ``dspy.RLM`` with the run's execution tools (the consumer's tools + ``SUBMIT`` /
    ``llm_query`` / ...), so a callable step can dispatch a REAL tool: ``lambda tools, v: tools["scan"](x=1)``.
    """

    def __init__(self, steps: Sequence[Step] = ()) -> None:
        self.tools: dict[str, Callable[..., Any]] = {}
        self.steps: list[Step] = list(steps)
        self.calls: list[str] = []
        self._i = 0

    def execute(self, code: str, variables: Optional[dict] = None) -> Any:
        self.calls.append(code)
        step: Step = self.steps[self._i] if self._i < len(self.steps) else ""
        self._i += 1
        return self._interpret(step, variables or {})

    def _interpret(self, step: Step, variables: dict) -> Any:
        from dspy.primitives.code_interpreter import FinalOutput

        if callable(step) and not isinstance(step, (str, dict)):
            step = step(self.tools, variables)
        if isinstance(step, FinalOutput):
            return step
        if isinstance(step, dict):
            return FinalOutput(step)          # SUBMIT: {output_field: value}
        return "" if step is None else str(step)

    def shutdown(self) -> None:
        return None


def submit(output: dict) -> dict:
    """A readable alias for a SUBMIT step: ``submit({"verdict": {...}})`` == ``{"verdict": {...}}``.
    Returns the dict unchanged; ``ScriptedInterpreter`` wraps a dict step in ``FinalOutput``."""
    return dict(output)


def call(tool_name: str, **kwargs: Any) -> Callable[[dict, dict], str]:
    """A step that dispatches a REAL injected tool and returns its (stringified) output as the REPL
    output for that turn. ``call("scan_indicators", region="...")``. The tool must be one dspy injected
    onto the interpreter's ``.tools`` (a consumer tool, or a built-in like ``llm_query``)."""

    def _step(tools: dict, _variables: dict) -> str:
        return str(tools[tool_name](**kwargs))

    return _step


def scripted_lm(turns: Sequence[dict]) -> Any:
    """A ``DummyLM`` whose canned ``{"reasoning", "code"}`` turns parse under the kit's JSON adapter —
    the planner side of an offline scripted forward run. One turn is consumed per RLM iteration, so
    provide at least as many turns as ``ScriptedInterpreter`` steps up to (and including) the SUBMIT.

    The ``code`` string is what lands in the recorded trajectory (a ``main_step``); it should MATCH what
    the paired interpreter step does, since the scripted interpreter runs the step, not the code.
    """
    import dspy
    from dspy.utils.dummies import DummyLM

    return DummyLM(list(turns), adapter=dspy.JSONAdapter())
