"""Shims for ``dspy``'s ``RLM`` / interpreter API, resolved by introspection in ONE place.

PRIVATE (``_``-prefixed): not part of the public surface, may change without notice.

**Why this module exists, and why it survives the 3.3.0 floor.** rlm-harness declares only a
FLOOR on dspy and consumers pin the KIT, so a consumer's fresh install resolves whatever dspy
is current. dspy 3.3.0 renamed three things at once — the caller-owned interpreter moved from
``RLM(interpreter=…)`` to ``forward``/``aforward``'s first positional arg, ``max_iterations``
became ``max_iters``, and ``CodeInterpreterError`` stopped being the RECOVERABLE interpreter
error (the new ``CodeExecutionError`` took that role) — and only the first failed loudly. The
kit was completely unrunnable on a fresh install while its whole suite stayed green
(CHANGELOG 1.0.1).

Since 1.2.0 the floor is ``dspy>=3.3.0`` and the 3.2.x branches are gone, so most of these
now resolve a single answer. **They are kept anyway**: the module's value was never "supports
two versions", it is that every dspy fact lives at ONE introspected call site, so the NEXT
rename is a one-line change here plus a red test in ``tests/test_dspy_compat.py`` — instead of
a silent behaviour change in someone's rollout. Do not collapse a shim into its call site just
because it currently has one branch.

Note one consequence of the floor: the interpreter seam is now HARDCODED to the
``forward()``-positional form. A future dspy that moved it back to the constructor would fail
LOUDLY (a ``TypeError`` from ``aforward``) rather than auto-adapting — which is the right
trade, and ``.github/workflows/dspy-latest.yml`` is what catches it.

This module must stay importable without dspy (its module top is dspy-free); every lookup
imports dspy lazily and is cached, since the installed dspy cannot change mid-process.
"""

from __future__ import annotations

import inspect
from functools import lru_cache
from typing import Any

# Canonical kit name -> the dspy kwarg names that have carried it, NEWEST FIRST.
# Probing in this order means a future rename only needs a new entry at the front.
_BUDGET_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("max_iterations", ("max_iters",)),
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


def forward_interpreter_args(interpreter: Any) -> tuple:
    """The positional args to prepend to ``rlm.aforward(...)`` for ``interpreter``.

    From dspy 3.3.0 a caller-owned interpreter is the FIRST POSITIONAL argument of
    ``forward``/``aforward``, not a constructor kwarg. Ownership is what makes this the right
    seam: dspy shuts down only an interpreter it created itself, never one the caller supplied
    ("Pass an existing interpreter as the first positional argument when calling the module"),
    so ``RLMTask._teardown_interpreter`` stays correct. Do NOT switch to ``interpreter_factory=``
    — dspy DOES shut down whatever that factory returns, which would double-shutdown the kit's
    sandbox.

    Empty when there is no caller-owned interpreter at all.
    """
    return () if interpreter is None else (interpreter,)


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
            # Can't probe: send the name this kit currently targets (newest = last, since
            # the alias tuples are newest-first and 1.2.0 dropped the legacy spellings).
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

#: dspy's attribute for that set, NEWEST FIRST — a tuple, not a constant, so the next rename
#: is one entry here. Same probing shape as `_BUDGET_ALIASES` above.
_RESERVED_ATTRS = ("_RESERVED_SANDBOX_NAMES",)


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


#: Output-field names dspy's RLM owns on its own Prediction. Hardcoded floor for
#: :func:`reserved_result_names`, unioned with whatever the installed dspy exposes.
_RESERVED_RESULT_FALLBACK = frozenset({"trajectory", "final_reasoning"})


@lru_cache(maxsize=1)
def reserved_result_names() -> frozenset[str]:
    """Output-field names dspy refuses because its own Prediction already carries them.

    Union with the fallback, like :func:`reserved_tool_names` — but for a DIFFERENT reason, so
    don't read the two rationales as one. Over-rejecting a *tool* name is cheap: the kit renames
    a tool nobody minded. Over-rejecting an *output field* is not auto-fixable — the field is the
    consumer's signature, and a false positive fails a task that runs fine on their dspy. The
    union is still right, because the failure mode it prevents (a task that constructs here and
    raises in their rollout) is worse than a loud, local, one-line-to-fix rename.
    """
    names = set(_RESERVED_RESULT_FALLBACK)
    try:
        import dspy

        found = getattr(dspy.RLM, "_RESERVED_RESULT_NAMES", None)
        if found:
            names |= {str(n) for n in found}
    except Exception:  # pragma: no cover - defensive: never let this raise
        pass
    return frozenset(names)


@lru_cache(maxsize=1)
def recoverable_interpreter_error() -> type[Exception]:
    """The interpreter-error class dspy's RLM loop CATCHES and feeds back to the model.

    Load-bearing, and the reason it is resolved rather than hardcoded at the raise sites: dspy
    3.3.0 added ``CodeExecutionError`` and INVERTED the meaning of the base class. Raising a bare
    ``CodeInterpreterError`` was recoverable on 3.2.x and is TERMINAL from 3.3.0 — so a sandbox
    turn-timeout that used to hand the model another turn would instead end the whole run, with
    no test failure to reveal it. That inversion is why every raise site asks here instead of
    naming a class (see ``sandbox.py``'s watchdog and ``container_interpreter.py``'s execute path).
    """
    from dspy.primitives.code_interpreter import CodeExecutionError

    return CodeExecutionError


@lru_cache(maxsize=1)
def terminal_interpreter_error() -> type[Exception]:
    """The interpreter-error class dspy's RLM loop does NOT catch — a run-ending failure.

    ``CodeInterpreterError``: terminal by design from dspy 3.3.0, since the recoverable role
    moved to its ``CodeExecutionError`` subclass. A condition that must end the run REGARDLESS
    of dspy's handling should still use an exception outside dspy's hierarchy entirely — that is
    what ``SandboxCancelled`` is, and why it needs no shim.
    """
    from dspy.primitives.code_interpreter import CodeInterpreterError

    return CodeInterpreterError


@lru_cache(maxsize=1)
def _lm_error_classes() -> tuple[type[Exception] | None, type[Exception] | None]:
    """``(dspy.LMError, dspy.ContextWindowExceededError)``, or ``None`` for either that a
    future dspy no longer exposes under that name. Split out of :func:`is_fast_fail_lm_error`
    so the two lookups are cached once instead of on every classified exception."""
    import dspy

    return getattr(dspy, "LMError", None), getattr(dspy, "ContextWindowExceededError", None)


def is_fast_fail_lm_error(exc: BaseException) -> bool:
    """True for an LM failure worth failing the whole task on immediately, not retrying.

    dspy's own ``is_retryable_lm_error`` classifies an auth/billing/configuration failure, an
    invalid request, or an unsupported model/feature as NOT retryable — a raw retry re-sends the
    exact same doomed call ``max_retries`` times for no benefit. ``run_with_retry`` did not honor
    that classification before this; every LM error consumed the full retry budget and was then
    wrapped in ``RLMTaskError``, indistinguishable from a genuine validation failure. This mirrors
    dspy's classification, with ONE deliberate carve-out.

    **The carve-out:** ``ContextWindowExceededError`` is a ``LMInvalidRequestError`` and dspy
    calls it non-retryable — correct for dspy's own LM-level retry, which resends the identical
    request. It is NOT correct here: ``run_with_retry`` retries by re-running the WHOLE
    trajectory, which can genuinely produce a shorter prompt on the next attempt (a different
    turn sequence, a truncated tool result). So it is excluded and keeps retrying like any other
    exception — this was the one contested part of the design (CHANGELOG 1.2.0) and is resolved
    HERE, not left to whoever reads the CHANGELOG note next.

    Deliberately reached through the PUBLIC ``dspy.is_retryable_lm_error`` rather than the
    private ``dspy.utils.exceptions._RETRYABLE_LM_ERRORS`` tuple it is built from — the same
    "introspect the public seam, never a private one" rule as every other shim here. Returns
    ``False`` (never fast-fails) for anything that is not a ``dspy.LMError`` at all, and for
    every case where the installed dspy is missing the classes/helper this needs — conservative
    by construction, so a future dspy renaming these degrades to "always retry", the behaviour
    before this existed, rather than to over-eager fast-failing.
    """
    lm_error, context_window_exceeded = _lm_error_classes()
    if lm_error is None or not isinstance(exc, lm_error):
        return False
    if context_window_exceeded is not None and isinstance(exc, context_window_exceeded):
        return False
    import dspy

    is_retryable = getattr(dspy, "is_retryable_lm_error", None)
    if is_retryable is None:
        return False
    return not is_retryable(exc)
