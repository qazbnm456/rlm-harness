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

This module imports ``dspy`` LAZILY (inside functions), so ``import rlm_harness.testing`` stays cheap and the
``import rlm_harness`` / dspy-free-module invariants are untouched. It is a TEST seam: the injected
interpreter bypasses ``sandbox.build_interpreter`` (and therefore the insecure-interpreter guard) exactly
like an injected ``DummyLM`` bypasses the real model — the caller supplies the double explicitly and owns
it. The default string path (``RLMConfig(interpreter=...)`` -> ``build_interpreter``) is unchanged and
keeps the guard.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from typing import Any

from ._dspy_compat import reserved_result_names, reserved_tool_names
from ._toolname import is_valid_tool_name

# A step is one execute() worth of behaviour. It is one of:
#   - a ``dict``     -> SUBMIT it as the run's final output (``{output_field: value}``); ends the loop.
#   - a ``str``      -> the REPL output for that turn (non-terminal — the next planner turn sees it).
#   - a ``callable`` -> called ``step(tools, variables)``; its return is interpreted by the SAME rules
#                       (dict -> submit, str -> output), or a dspy ``FinalOutput`` is passed through.
Step = dict | str | Callable[[dict, dict], Any]


def _resolve_tool_name(tool: Any) -> str:
    """The name DSPY will validate and register — not the one Python reports.

    `dspy.Tool` takes an explicit `name=` and only falls back to `func.__name__` when it is
    omitted, so for a tool built as `dspy.Tool(f, name="get-weather")` the function's own name
    is a string dspy never looks at. Checking it would validate the wrong value and pass a tool
    dspy refuses. (`mcp.py` builds its tools exactly that way, and an early draft of the 1.0.2
    fix sanitised `__name__` alone — a placebo this resolution catches.)
    """
    fn = getattr(tool, "func", tool)
    return getattr(tool, "name", None) or getattr(fn, "__name__", None) or repr(fn)


def assert_repl_safe(tool: Any) -> None:
    """Assert ``tool`` is safe to inject into the RLM's REPL (``RLMTask(tools=[...])``).

    dspy.RLM builds the in-sandbox tool proxy from ``inspect.signature(tool.func)`` — NOT from
    ``dspy.Tool.args`` — and this holds for BOTH the Deno ``PythonInterpreter`` and rlm-harness's
    ``ContainerInterpreter`` (each reads the wrapped func's signature). Two consequences no
    CONSTRUCTION test can see, only a real REPL call:

    * a ``*args``/``**kwargs`` param is flattened into a single proxy param literally named
      ``args``/``kwargs`` — the model can only pass the value under that meaningless name, which a
      strict MCP server rejects and a plain tool mis-binds (this is the ``_make_tool`` kwargs bug);
    * a required (no-default) param placed AFTER a defaulted one makes the generated Deno ``def`` a
      ``SyntaxError`` that aborts the ENTIRE tool registration.

    Call this on every callable a consumer exposes to the planner — it turns the
    "explicit-params-only" convention (documented but historically un-tested) into an enforced
    invariant, so a future factory can't silently reintroduce the hazard. Accepts a ``dspy.Tool``
    (checks its ``.func``) or a bare callable.
    """
    fn = getattr(tool, "func", tool)

    # Resolve the name the way DSPY does, not the way Python does. `dspy.Tool` takes an
    # explicit `name=` and only falls back to `func.__name__` when it is omitted — so for a
    # tool built as `dspy.Tool(f, name="get-weather")`, `f.__name__` is a string dspy never
    # looks at. Checking it would validate the wrong value and pass a tool that dspy refuses.
    # (This is not hypothetical: `mcp.py` builds its tools exactly that way, and an earlier
    # draft of the 1.0.2 fix sanitised `__name__` alone — a placebo this check catches.)
    label = _resolve_tool_name(tool)

    # dspy validates the name at `RLM(...)` construction and a failure aborts the ENTIRE
    # tool registration, so one bad name silently takes every other tool down with it.
    if not is_valid_tool_name(label):
        raise AssertionError(
            f"REPL tool name {label!r} is not a valid Python identifier (or is a keyword): "
            f"dspy refuses it at RLM construction, which aborts registration for EVERY tool "
            f"in the task. Derive the REPL name with `rlm_harness.sanitize_tool_name` (or "
            f"`unique_tool_names` for a whole set) and keep the raw name for the wire/trace "
            f"identity."
        )
    if label in reserved_tool_names():
        raise AssertionError(
            f"REPL tool name {label!r} is reserved by dspy's sandbox "
            f"({sorted(reserved_tool_names())}) — registering it would shadow a built-in."
        )

    seen_default = False
    for pname, p in inspect.signature(fn).parameters.items():
        if p.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL):
            # A tool may carry `__repl_degraded__` explaining WHY it could not be given a real
            # signature (mcp.py sets it when a schema property name cannot be a Python
            # parameter). Surface it: without the reason this reads as a fixable mistake, when
            # for that tool it is a known, documented dead end.
            why = getattr(fn, "__repl_degraded__", None)
            raise AssertionError(
                f"REPL tool {label!r} exposes a {p.kind.name} param {pname!r}: dspy flattens it into a "
                f"proxy param literally named {pname!r}, so the model cannot call it correctly. "
                + (f"Known cause — {why}" if why else
                   "Give the tool EXPLICIT named params (build one from a JSON Schema with "
                   "`rlm_harness.signature_from_json_schema`).")
            )
        if p.default is not inspect.Parameter.empty:
            seen_default = True
        elif seen_default:
            raise AssertionError(
                f"REPL tool {label!r}: required param {pname!r} follows a defaulted one, so the "
                f"generated Deno `def {label}(…)` is a SyntaxError. Order required params first."
            )


def _signature_field_names(signature: str) -> tuple[list[str], list[str]]:
    """Split a dspy signature string into (input names, output names).

    A VERBATIM MIRROR of dspy's own parser, not a reimplementation — and the deliberate
    exception to "every dspy fact goes through `_dspy_compat`". dspy exposes no name-only
    parser, and the obvious alternative, `dspy.Signature(sig)`, is a trap: it resolves the
    output TYPE by walking the call stack's globals, so it raises `Unknown name: …` for a
    dynamically-built model and would make this helper pass or fail depending on which frame
    happened to hold the name — the exact behaviour `_build_rlm`'s `custom_types=` exists to
    avoid. So we mirror the two lines that matter, from `dspy/signatures/signature.py`:

        :616   if signature.count("->") != 1: raise ValueError(...)
        :649   ast.parse(f"def f({field_string}): pass").body[0].args.args

    `.args.args` ONLY — dspy ignores `posonlyargs` / `kwonlyargs` / `vararg` / `kwarg`, so
    reading more would report fields dspy never registers. The arrow-count guard is dspy's
    own first check and also disposes of an arrow inside a string literal
    (`doc: Literal['a->b'] -> answer: str`), which a naive split silently corrupts.
    """
    import ast

    if not isinstance(signature, str):
        # AssertionError, not TypeError, on purpose: EVERY failure from this helper family is an
        # assertion about the task under test, so a caller writes one `pytest.raises`. Raising a
        # different type here for a wrong-typed field would split that.
        raise AssertionError(  # noqa: TRY004
            f"signature must be a str, got {type(signature).__name__}"
        )
    if signature.count("->") != 1:
        raise AssertionError(
            f"signature {signature!r} must contain exactly one '->' (dspy raises on this too)"
        )
    parts = []
    for side in signature.split("->"):
        try:
            tree = ast.parse(f"def f({side.strip()}): pass")
        except SyntaxError as exc:
            # RAISE, never skip: a field named `class` / `in` is one dspy also rejects, and a
            # helper that silently no-ops on exactly the broken tasks is worse than absent.
            raise AssertionError(
                f"signature {signature!r} does not parse ({exc.msg}) — dspy rejects it too; a "
                f"field name must be a valid Python identifier and not a keyword"
            ) from exc
        parts.append([a.arg for a in tree.body[0].args.args])  # type: ignore[attr-defined]
    return parts[0], parts[1]


def assert_task_repl_safe(task: Any) -> None:
    """Assert a WHOLE ``RLMTask`` is safe to construct — the checks no per-tool test can make.

    :func:`assert_repl_safe` validates one tool's shape and name. Three of dspy's
    construction-time rules are properties of the whole task, and each aborts registration for
    EVERY tool:

    * duplicate tool names — dspy keys its tool dict by name;
    * a signature INPUT field colliding with a tool name, or with a reserved sandbox name;
    * a signature OUTPUT field dspy's own Prediction already owns (``trajectory`` /
      ``final_reasoning``).

    Accepts a task INSTANCE or a subclass. Prefer an instance: tools assembled at runtime (the MCP
    case, and anything set in ``__init__`` or passed as ``tools=``) exist only there, and those are
    exactly the tool sets these rules bite on — a class-level check cannot see them.

    Note ``RLMTask.__init__`` needs a configured runtime, so constructing an instance in a test
    means either a prior ``configure(...)`` or passing BOTH ``config=`` and ``sub_lm=``.

    This is a TEST-time helper, deliberately NOT wired into ``RLMTask.__init__``: dspy enforces
    all four at construction, and duplicating an upstream check in the hot path invites the two
    to drift.

    **What it is still FOR, now that dspy enforces them too.** It moves the failure from run time
    to test time, names the rule instead of surfacing a dspy-worded ``ValueError``, and — the part
    dspy has no equivalent for at all — it runs ``assert_repl_safe`` over every tool, which checks
    the SHAPE rules (no ``*args``/``**kwargs``, no required param after a defaulted one) that dspy
    does not validate anywhere. Those two shape hazards are invisible until a real REPL call.
    """
    # `resolved_tools` is `RLMTask`'s ONE derivation of the effective list (a `tools=` kwarg
    # replaces the declaration), so read it when present rather than re-deriving the rule here.
    #
    # The `isclass` guard is load-bearing: on a CLASS, `getattr(cls, "resolved_tools")` returns
    # the `property` DESCRIPTOR, which is truthy and not iterable — so a naive
    # `getattr(...) or getattr(...)` swallows the fallback and raises
    # `TypeError: 'property' object is not iterable` on the documented class path.
    if inspect.isclass(task):
        tools_attr = getattr(task, "tools", ())
    else:
        resolved = getattr(task, "resolved_tools", None)
        tools_attr = resolved if resolved is not None else getattr(task, "tools", ())
    tools = list(tools_attr or ())
    for tool in tools:
        assert_repl_safe(tool)

    names = [_resolve_tool_name(t) for t in tools]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise AssertionError(
            f"duplicate REPL tool name(s) {duplicates}: dspy keys its tool dict by name and "
            f"refuses the collision, so registration aborts for EVERY tool on the task. "
            f"Give each tool a distinct name."
        )

    signature = getattr(task, "signature", "")
    if not signature:
        return
    inputs, outputs = _signature_field_names(signature)

    reserved = reserved_tool_names()
    if bad := sorted(set(inputs) & reserved):
        raise AssertionError(
            f"signature input field(s) {bad} collide with dspy's built-in sandbox functions "
            f"({sorted(reserved)})."
        )
    if bad := sorted(set(inputs) & set(names)):
        raise AssertionError(
            f"signature input field(s) {bad} collide with a tool name. dspy binds both into the "
            f"same REPL namespace, so one would shadow the other."
        )
    if bad := sorted(set(outputs) & reserved_result_names()):
        raise AssertionError(
            f"signature output field(s) {bad} collide with the names dspy's own RLM Prediction "
            f"carries ({sorted(reserved_result_names())})."
        )


class ScriptedInterpreter:
    """A scripted ``dspy`` ``CodeInterpreter`` double for offline forward-path tests.

    Build it with a list of STEPS (see ``Step``); one step is consumed per ``execute()`` call, in order.
    When the script is exhausted it returns ``""`` forever (a non-terminal no-op) so a loop that never
    reaches a SUBMIT step runs to its iteration cap — useful for budget-exhaustion tests.

    ``.calls`` records the code strings ``dspy`` asked to execute, in order, for assertions. ``.tools``
    is populated by ``dspy.RLM`` with the run's execution tools (the consumer's tools + ``SUBMIT`` /
    ``llm_query`` / ...), so a callable step can dispatch a REAL tool: ``lambda tools, v: tools["scan"](x=1)``.

    It implements dspy's full ``CodeInterpreter`` surface — ``tools`` / ``start`` / ``execute`` /
    ``shutdown`` — and must keep doing so: from dspy 3.3.0 that protocol is ``@runtime_checkable`` and
    a caller-supplied interpreter is ``isinstance``-checked against it before every forward pass, so a
    missing method turns into a ``TypeError`` at run time rather than anything a type checker catches.
    """

    def __init__(self, steps: Sequence[Step] = ()) -> None:
        self.tools: dict[str, Callable[..., Any]] = {}
        self.steps: list[Step] = list(steps)
        self.calls: list[str] = []
        self._i = 0

    def start(self) -> None:
        """No-op: there is no process to spin up. Present because dspy's ``CodeInterpreter``
        protocol requires it (see the class docstring), not because the double needs it."""

    def execute(self, code: str, variables: dict | None = None) -> Any:
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
