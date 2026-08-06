"""Cross-version shims for ``dspy``'s ``RLM`` / interpreter API renames.

PRIVATE (``_``-prefixed): not part of the public surface, may change without notice.

The kit declares ``dspy>=3.2.1`` and consumers pin the KIT, not dspy — so a fresh
install resolves whatever dspy is current, and one build has to work across the
renames dspy has made to ``RLM``'s constructor and to the interpreter error
hierarchy. Every such difference is resolved HERE, by introspection, so the rest of
the kit reads as if there were only one dspy.

Known differences:

===================  =========================  ==========================
what                 dspy 3.2.x                 dspy 3.3.x
===================  =========================  ==========================
caller's interpreter ``RLM(interpreter=obj)``   1st POSITIONAL arg of
                                                ``forward``/``aforward``
iteration cap        ``max_iterations=``        ``max_iters=``
recoverable REPL     ``CodeInterpreterError``   ``CodeExecutionError``
error                                           (``CodeInterpreterError``
                                                 became TERMINAL)
===================  =========================  ==========================

Interpreter OWNERSHIP is the same on both and is what the kit relies on: dspy shuts
down only an interpreter it created itself, never one the caller supplied. 3.3.0
states this explicitly ("Pass an existing interpreter as the first positional
argument when calling the module"), so ``RLMTask._teardown_interpreter`` stays
correct either way. Do NOT switch to 3.3.0's ``interpreter_factory=``: dspy shuts
down whatever that factory returns, which would double-shutdown the kit's sandbox.

This module must stay importable without dspy (its module top is dspy-free); every
lookup imports dspy lazily and is cached, since the installed dspy cannot change
mid-process.
"""

from __future__ import annotations

import inspect
from functools import lru_cache
from typing import Any

# Canonical kit name -> the dspy kwarg names that have carried it, NEWEST FIRST.
# Probing in this order means a future rename only needs a new entry at the front.
_BUDGET_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("max_iterations", ("max_iters", "max_iterations")),
    ("max_llm_calls", ("max_llm_calls",)),
    ("max_output_chars", ("max_output_chars",)),
)


@lru_cache(maxsize=1)
def _rlm_init_signature() -> inspect.Signature | None:
    """``dspy.RLM.__init__``'s signature, or ``None`` if it cannot be read."""
    try:
        import dspy

        return inspect.signature(dspy.RLM.__init__)
    except Exception:  # pragma: no cover - defensive: a future dspy may not introspect
        return None


@lru_cache(maxsize=1)
def _rlm_init_params() -> frozenset[str]:
    sig = _rlm_init_signature()
    return frozenset(sig.parameters) if sig is not None else frozenset()


@lru_cache(maxsize=1)
def _rlm_init_takes_var_keyword() -> bool:
    """True if ``RLM.__init__`` has a ``**kwargs``, i.e. name probing proves nothing."""
    sig = _rlm_init_signature()
    if sig is None:
        return False
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


@lru_cache(maxsize=1)
def rlm_accepts_interpreter_kwarg() -> bool:
    """True on dspy 3.2.x, where a caller-owned interpreter goes to ``RLM(...)``.

    False on 3.3.x, where it goes to ``forward``/``aforward`` as the first positional
    argument instead — see :func:`forward_interpreter_args`.
    """
    if _rlm_init_takes_var_keyword():
        return True  # can't disprove it; the legacy path is the safer guess
    return "interpreter" in _rlm_init_params()


def forward_interpreter_args(interpreter: Any) -> tuple:
    """The positional args to prepend to ``rlm.aforward(...)`` for ``interpreter``.

    Empty when the interpreter was already handed to the constructor (3.2.x) or when
    there is no caller-owned interpreter at all.
    """
    if interpreter is None or rlm_accepts_interpreter_kwarg():
        return ()
    return (interpreter,)


def rlm_budget_kwargs(
    *, max_iterations: int, max_llm_calls: int, max_output_chars: int
) -> dict[str, int]:
    """Map the kit's budget caps onto the names the installed dspy actually accepts.

    Silently dropping a cap is the failure mode to avoid here: before this shim the
    kit passed all three under their 3.2.x names inside one all-or-nothing ``try``,
    so on 3.3.x the single renamed ``max_iterations`` made dspy reject the call and
    the fallback dropped ALL THREE caps back to dspy's defaults — unbounded relative
    to what ``RLMConfig`` asked for, with nothing logged at the caller's level.
    """
    values = {
        "max_iterations": max_iterations,
        "max_llm_calls": max_llm_calls,
        "max_output_chars": max_output_chars,
    }
    params = _rlm_init_params()
    permissive = _rlm_init_takes_var_keyword() or not params

    resolved: dict[str, int] = {}
    for canonical, candidates in _BUDGET_ALIASES:
        if permissive:
            # Can't probe: use the name this kit has always sent (oldest = last).
            resolved[candidates[-1]] = values[canonical]
            continue
        for name in candidates:
            if name in params:
                resolved[name] = values[canonical]
                break
    return resolved


#: Names dspy owns inside the sandbox. The hardcoded floor for
#: :func:`reserved_tool_names`; unioned with whatever the installed dspy exposes.
_RESERVED_FALLBACK = frozenset({"llm_query", "llm_query_batched", "SUBMIT", "print"})

#: dspy's attribute for that set, NEWEST FIRST — 3.3.0 renamed `_RESERVED_TOOL_NAMES`
#: to `_RESERVED_SANDBOX_NAMES`. Same probing order as `_BUDGET_ALIASES` above.
_RESERVED_ATTRS = ("_RESERVED_SANDBOX_NAMES", "_RESERVED_TOOL_NAMES")


@lru_cache(maxsize=1)
def reserved_tool_names() -> frozenset[str]:
    """Tool names dspy refuses because it owns them inside the sandbox.

    Returns the UNION of what the installed dspy exposes and `_RESERVED_FALLBACK`, not
    one or the other. The asymmetry is deliberate: a stale fallback that over-rejects
    fails LOUDLY and locally (the kit renames a tool nobody had a problem with), while
    one that under-rejects passes here and resurfaces as a dspy ``ValueError`` in a
    consumer's rollout. Over-rejecting is the cheap direction.

    These are ``_``-private dspy attributes, so a rename or removal must be a non-event:
    anything unreadable falls through to the fallback rather than raising.
    """
    names = set(_RESERVED_FALLBACK)
    try:
        import dspy

        for attr in _RESERVED_ATTRS:
            found = getattr(dspy.RLM, attr, None)
            if found:
                names |= {str(n) for n in found}
                break
    except Exception:  # pragma: no cover - defensive: never let this raise
        pass
    return frozenset(names)


@lru_cache(maxsize=1)
def recoverable_interpreter_error() -> type[Exception]:
    """The interpreter-error class dspy's RLM loop CATCHES and feeds back to the model.

    Load-bearing, and the reason this is resolved rather than hardcoded: dspy 3.3.0
    added ``CodeExecutionError`` and INVERTED the meaning of the base class. Raising a
    bare ``CodeInterpreterError`` was recoverable on 3.2.x and is TERMINAL on 3.3.x —
    so a sandbox turn-timeout that used to hand the model another turn would instead
    end the whole run, with no test failure to reveal it.

    Falls back to ``CodeInterpreterError`` on a dspy too old to have the split, which
    is exactly right there: on 3.2.x that base class IS the recoverable one.
    """
    from dspy.primitives import code_interpreter

    return getattr(
        code_interpreter, "CodeExecutionError", code_interpreter.CodeInterpreterError
    )


@lru_cache(maxsize=1)
def terminal_interpreter_error() -> type[Exception]:
    """The interpreter-error class dspy's RLM loop does NOT catch — a run-ending failure.

    ``CodeInterpreterError`` on both: on 3.3.x it is terminal by design, and on 3.2.x
    there is nothing more terminal to reach for (dspy catches it, but the kit has no
    stronger interpreter-level signal there — a genuinely run-ending condition should
    use a non-``CodeInterpreterError`` exception such as ``SandboxCancelled``).
    """
    from dspy.primitives.code_interpreter import CodeInterpreterError

    return CodeInterpreterError
