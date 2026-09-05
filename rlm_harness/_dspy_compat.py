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

Since 1.2.0 the floor is ``dspy>=3.3.0`` (``>=3.3.1`` since 1.5.0, which needs
``interpreter_factory.execution_instructions``) and the 3.2.x branches are gone, so most of these
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

import contextlib
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
    as the way to SUPPLY an interpreter — dspy DOES shut down whatever that factory returns,
    which would double-shutdown the kit's sandbox. (``interpreter_instructions_kwargs`` below
    does pass an ``interpreter_factory``, but only as a metadata CARRIER that dspy never calls;
    see its docstring for why that is not the same thing.)

    Empty when there is no caller-owned interpreter at all.
    """
    return () if interpreter is None else (interpreter,)


def _raise_carrier_invoked() -> None:
    raise RuntimeError(
        "rlm-harness passed dspy an interpreter_factory as a metadata carrier only — it must "
        "never be INVOKED, because dspy would then own and shut down what it returns while "
        "RLMTask._teardown_interpreter also shuts down its own sandbox. Reaching here means the "
        "caller-owned interpreter stopped being passed positionally to forward()/aforward()."
    )


@lru_cache(maxsize=1)
def _dspy_reads_execution_instructions() -> bool:
    """True if the installed dspy renders ``interpreter_factory.execution_instructions``.

    Probed off ``PythonInterpreter``, which grew the attribute in the same release that taught
    ``RLM._build_signatures`` to read it. Introspected rather than version-gated, like every
    other answer in this module.
    """
    try:
        from dspy.primitives.python_interpreter import PythonInterpreter

        return isinstance(getattr(PythonInterpreter, "execution_instructions", None), str)
    except Exception:  # pragma: no cover - defensive
        return False


def interpreter_instructions_kwargs(interpreter: Any) -> dict[str, Any]:
    """``RLM(...)`` kwargs that describe ``interpreter``'s runtime to the model, or ``{}``.

    THE PROBLEM. From dspy 3.3.1 the action prompt carries an "Execution environment:" section,
    and dspy sources it from ``self._interpreter_factory.execution_instructions``. The kit does
    not set ``interpreter_factory`` — it supplies the interpreter POSITIONALLY, which is what
    keeps ownership (see ``forward_interpreter_args``) — so the attribute is read off dspy's
    DEFAULT factory, ``PythonInterpreter``. Every run is therefore told "Python runs in
    Pyodide/WebAssembly … subprocesses and native extensions are unavailable" no matter what is
    actually executing the code. For the ``container`` interpreter that is false in the one way
    that matters: spawning subprocesses is the entire reason it exists. Nothing goes red; the
    model simply stops trying.

    THE FIX, and why it does not reopen the ownership hole. The returned factory is a metadata
    CARRIER: dspy reads an attribute off it and never calls it. That is not an assumption —
    ``_validate_interpreter_factory`` validates without invoking, ``_interpreter_context``
    returns the caller-owned interpreter and returns early, and ``RLMTask._build_rlm`` always
    resolves a non-``None`` interpreter, so the factory is unreachable by construction. It still
    raises if invoked, so a future dspy that changes the positional seam fails loudly here
    instead of silently double-shutting-down the sandbox.

    Returns ``{}`` — changing nothing — unless ALL of:

    * the installed dspy actually renders the text (``_dspy_reads_execution_instructions``);
    * ``RLM.__init__`` really accepts ``interpreter_factory``. Load-bearing, and NOT redundant
      with the check above: ``_build_rlm``'s ``except TypeError`` fallback re-passes the same
      kwargs, so an unknown kwarg raises on BOTH constructions and takes the run down rather
      than degrading. It is also what "never hardcode a dspy kwarg name" requires;
    * ``interpreter`` exposes a non-empty ``execution_instructions`` string;
    * ``interpreter`` is not a dspy ``PythonInterpreter`` — dspy's own default already describes
      those correctly, so carrying its text back to it would be pure noise.
    """
    if not _dspy_reads_execution_instructions():
        return {}
    if "interpreter_factory" not in _rlm_init_params() and not _rlm_init_takes_var_keyword():
        return {}
    text = getattr(interpreter, "execution_instructions", None)
    if not isinstance(text, str) or not text.strip():
        return {}
    try:
        from dspy.primitives.python_interpreter import PythonInterpreter

        if isinstance(interpreter, PythonInterpreter):
            return {}
    except Exception:  # pragma: no cover - defensive
        pass

    def carrier():
        _raise_carrier_invoked()

    carrier.execution_instructions = text
    return {"interpreter_factory": carrier}


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


@lru_cache(maxsize=1)
def _lm_response_cls() -> Any:
    """dspy's typed sub-LM response class, or ``None`` on a dspy that has none."""
    import dspy

    return getattr(dspy, "LMResponse", None)


def sub_lm_response_text(response: Any) -> str | None:
    """The completion TEXT out of whatever shape a sub-LM returned, or ``None``.

    dspy's ``RLM._query_lm`` accepts TWO shapes from ``sub_lm`` and this mirrors that read:
    a typed ``dspy.LMResponse`` (take ``.text``), or the legacy ``list[str | dict]`` (take the
    first element, and its ``"text"`` key when it is a dict). Anything else yields ``None``.

    **Why this is a shim and not three lines at the call site.** ``sub_lm.py`` used to assume the
    legacy list unconditionally — ``[outputs]`` for anything non-list — which turned an
    ``LMResponse`` into ``[LMResponse]`` and made dspy raise ``Sub-LM response must contain text,
    got LMResponse``. That break was invisible on the default path and fired only under
    ``dspy.context(experimental=True)``, and dspy's own source dates the legacy shape: *"In DSPy
    3.3 and 3.4, ordinary calls preserve the legacy public return value"*. So the assumption had a
    two-minor shelf life and no test could see it expire. Resolving it HERE is what makes
    ``tests/test_dspy_compat.py`` the place a 3.5 contract change goes red.
    """
    lm_response = _lm_response_cls()
    if lm_response is not None and isinstance(response, lm_response):
        text = getattr(response, "text", None)
        return text if isinstance(text, str) else None
    if isinstance(response, (list, tuple)) and response:
        first = response[0]
        text = first.get("text") if isinstance(first, dict) else first
        return text if isinstance(text, str) else None
    return None


def sub_lm_response_with_text(response: Any, text: str) -> Any:
    """``response`` with its completion text replaced by ``text``, in the SAME shape.

    Shape preservation is the point: a sub-LM wrapper must hand dspy back what dspy handed it,
    or it silently narrows what the installed dspy supports. For a typed ``LMResponse`` the first
    ``text`` part of the first output is replaced and **every LATER text part of that output is
    dropped** — because ``LMOutput.text`` JOINS all of them, so replacing only the first would
    leave the rest appended to the substituted text (``"AB"`` round-tripping to ``"ABB"``).
    Multi-part text is not exotic: dspy emits one ``LMTextPart`` per content item, so any provider
    returning a content ARRAY (citation-interleaved text, Responses-API output blocks) produces
    several. Thinking, tool-call, citation and refusal parts, every sibling output, and every
    response-level field (``model``, ``usage``, ``cost``, ``cache_hit``, …) survive untouched, via
    ``model_copy`` so the ORIGINAL is not mutated. For the legacy list only element 0 is replaced;
    additional completions pass through.

    Falls back to a one-element list for a shape it does not recognise — the legacy shape dspy
    has always accepted — so an unknown future return type degrades to something dspy can read
    rather than to a crash.
    """
    lm_response = _lm_response_cls()
    if lm_response is not None and isinstance(response, lm_response):
        outputs = list(getattr(response, "outputs", None) or ())
        if not outputs:
            return [text]
        parts = list(getattr(outputs[0], "parts", None) or ())
        replaced = False
        new_parts = []
        for part in parts:
            if getattr(part, "type", None) != "text":
                new_parts.append(part)          # thinking / tool_call / citation / refusal
            elif not replaced:
                new_parts.append(part.model_copy(update={"text": text}))
                replaced = True
            # ...and every LATER text part is DROPPED: `LMOutput.text` joins them, so keeping one
            # would append the old tail to the substituted text.
        if not replaced:
            return [text]
        new_first = outputs[0].model_copy(update={"parts": new_parts})
        return response.model_copy(update={"outputs": [new_first, *outputs[1:]]})
    if isinstance(response, (list, tuple)) and response:
        rest = list(response[1:])
        return [text, *rest]
    return [text]


@lru_cache(maxsize=1)
def python_fence_langs() -> frozenset[str]:
    """The markdown fence tags dspy's ``_strip_code_fences`` accepts as Python.

    Introspected from ``dspy.predict.rlm._PYTHON_FENCE_LANGS``, with a hardcoded fallback so a
    consumer computing metrics in a dspy-free report renderer never hits an ``ImportError``. The
    fallback makes a RENAME silent, which is why ``tests/test_dspy_compat.py`` asserts the
    INTROSPECTION PATH resolves rather than merely that the value equals the fallback — and it is
    not silent in a safe direction either: a stale set counts executed turns as refused ones the
    day dspy adds a lang.
    """
    try:
        from dspy.predict.rlm import _PYTHON_FENCE_LANGS

        return frozenset(_PYTHON_FENCE_LANGS)
    except Exception:
        return frozenset({"", "python", "py", "python3", "py3"})


@lru_cache(maxsize=1)
def forced_final_marker() -> str:
    """The ``final_reasoning`` dspy writes when its turn loop falls through without a ``FINAL``.

    **NOT introspectable** — unlike :func:`python_fence_langs`, this is a bare string literal at two
    sites in dspy (``rlm.py``'s ``_extract_fallback`` / ``_aextract_fallback``) with no constant
    behind it, so there is nothing to look up. Asserting this function's return value against the
    same literal in a test would assert the kit against itself and stay green through any dspy
    rename — the exact failure this module exists to prevent. Its test therefore DRIVES a real
    forced-final run and compares the resulting ``Prediction.final_reasoning`` to this value.

    Reached only by the iteration budget running out; ``max_llm_calls`` exhaustion raises inside the
    sandbox and comes back as a turn instead, so a reader of this marker is measuring the ITERATION
    cap specifically.
    """
    return "Extract forced final output"


def dspy_refuses_fence(code: Any) -> bool:
    """Would dspy refuse to execute this cell because of a markdown fence tag?

    **A VERBATIM MIRROR of the `_`-private ``dspy.predict.rlm._strip_code_fences``** — the same
    declared exception ``testing.py``'s ``_signature_field_names`` carries. The accepted-tag SET is
    introspectable (:func:`python_fence_langs`) but the DECISION is not: it lives in that function's
    prelude, and every step below is load-bearing.

    A shortcut does not work, measured against the real function over tens of thousands of cells:
    a ``re.search(r"```([^\\n`]*)")`` + ``split()[0]`` form produced **1,764 disagreements and 3,855
    IndexError crashes**, crashing on a BARE ``` fence — the commonest shape. A prose paraphrase
    that drops the ``.strip()``, the empty-tag guard, or either early return still crashed 673
    times. This port was verified at **0 disagreements and 0 crashes** over tens of thousands of real and
    fuzzed cells across several independent runs, including
    CRLF, ``\\r``-only, tabs, four and five backticks, ``~~~``, unbalanced decorative pairs, unicode
    whitespace in tags, and the ``\\x85``/``\\x0b``/``\\x0c`` characters ``splitlines()`` splits on
    but ``partition("\\n")`` does not — it survives those only BECAUSE it is verbatim.

    Deliberately NOT ``lru_cache``d. That is this module's norm for an argument-taking shim (only
    the nullary ones are cached); here the key would be an unbounded set of full code cells.

    A non-``str`` ``code`` — shape drift, a hand-built event, a future dspy — counts as NOT refused,
    so a caller's count stays an ``int`` rather than becoming unmeasurable.
    """
    if not isinstance(code, str):
        return False
    code = code.strip()
    if "```" not in code:
        return False
    lines = code.splitlines()          # dspy pops DECORATIVE outer ``` / ``` line pairs
    while len(lines) >= 2 and lines[0].strip() == "```" and lines[-1].strip() == "```":
        lines.pop(0)
        lines.pop()
    code = "\n".join(lines).strip()
    if "```" not in code:
        return False
    lang_line, sep, _rest = code[code.find("```") + 3:].partition("\n")
    if not sep:
        # dspy RETURNS the code and RUNS it. This is the ONLY accept path whose recorded `code`
        # still contains a fence, and its output is a fixpoint (re-parsing re-accepts).
        return False
    stripped = lang_line.strip()
    lang = (stripped.split(maxsplit=1)[0] if stripped else "").lower()
    return lang not in python_fence_langs()


# --- token budgets and usage ---------------------------------------------------------------
#
# Why this lives here at all: a truncated completion and a malformed one raise the SAME exception
# type, so a consumer diagnosing `AdapterParseError` cannot tell which it had. dspy DOES detect
# truncation -- `LM._check_truncation` tests `finish_reason == "length"` -- and then only
# `logger.warning`s it, discarding the datum before any caller can see it. What survives is the
# token COUNT, and `completion_tokens == max_tokens` is the same fact with an extra property: it
# also shows a turn APPROACHING the cap, where a boolean fires only after death.
#
# MEASURED SINCE, and the approaching half is CONDITIONAL on the cap: on the first production corpus
# (one model, cap 32768; 385 runs reaching run_end, of which 379 succeeded) the ratio distribution
# has a HOLE. The BINS are the 379 successes -- 363 below 0.6, ZERO between 0.6 and 1.0, 16 at the
# cap; adding the 6 failures gives 364 / 0 / 21 across all 385. But transposing those bins onto a 16384 cap moves 64 of 379 runs
# (16.9%) into the empty band, so the hole is an artifact of a cap set at ~2x what the model needs,
# not a property of the model. Do not promise early warning unconditionally, and do not deny it
# either -- it depends on the caller's cap. What holds regardless: the count separates a truncation
# from a malformed reply, and 76% of truncations self-heal because dspy's own SyntaxError feedback
# repairs a truncated CODE cell while a truncated FINAL answer has no handler.

_LM_BUDGET_KEYS = ("max_tokens", "max_completion_tokens")


def applied_lm_budget(lm: Any) -> dict[str, Any] | None:
    """The generation cap actually APPLIED by ``lm``, as ``{"cap": int, "key": str}`` or ``None``.

    Read off the LM, never from ``RLMConfig``: ``runtime`` builds an LM from config ONLY for a role
    that is still ``None``, and an injected ``main_lm``/``sub_lm`` is used verbatim -- so the
    configured cap can be one the call never used, which is exactly the consumer whose run died.
    This mirrors dspy's own ``_check_truncation``, which reads ``self.kwargs['max_tokens']``.

    **Both key names, and the found one is reported.** dspy rewrites ``max_tokens`` to
    ``max_completion_tokens`` in ``LM._get_initial_kwargs`` for OpenAI reasoning models, so
    ``dspy.LM("openai/o3", max_tokens=16384).kwargs`` carries ONLY the latter. Reading the former
    alone returns ``None`` for precisely the thinking-model case this exists to explain.

    **NAMED KEYS ONLY -- never serialise ``lm.kwargs``.** It carries ``api_key`` for every LM the
    kit builds, and a trace is a shipped artifact that reaches replay, the dataset exporters and
    every consumer's corpus. One dict-dump here is a credential in every file.

    ``None`` when no cap is set. Note the key is PRESENT with value ``None`` in that case, so this
    cannot distinguish "never set" from "explicitly None" -- both are absent, which is the honest
    reading either way.
    """
    kwargs = getattr(lm, "kwargs", None)
    if not isinstance(kwargs, dict):
        return None
    for key in _LM_BUDGET_KEYS:
        cap = kwargs.get(key)
        if isinstance(cap, int) and not isinstance(cap, bool):
            return {"cap": cap, "key": key}
    return None


def current_usage_tracker() -> Any:
    """dspy's active ``UsageTracker``, or ``None`` -- through the PUBLIC ``dspy.settings``.

    Never `module.py`'s ``thread_local_overrides.get().get("usage_tracker")``: that private read is
    the EVIDENCE for the behaviour below, not the API to code against. Verified equivalent with no
    tracker, inside one, and inside a ``copy_context()`` worker.
    """
    try:
        import dspy

        return getattr(dspy.settings, "usage_tracker", None)
    except Exception:
        return None


@contextlib.contextmanager
def usage_tracking() -> Any:
    """Yield a ``UsageTracker`` for the enclosed block, REUSING one already installed.

    Installing unconditionally would SHADOW a consumer's own tracker: ``dspy.Module`` creates one
    only when none is set, so a consumer's ``with dspy.track_usage(): await task.arun(...)`` would
    collect ZERO entries for everything inside -- this kit writing a structural zero into someone
    else's measurement. Reuse instead, and read a SLICE (see :func:`usage_since`) so the
    consumer's own calls are not counted as ours.

    Yields ``None`` when the installed dspy has no usage tracking at all.
    """
    existing = current_usage_tracker()
    if existing is not None:
        yield existing
        return
    try:
        from dspy.utils.usage_tracker import track_usage
    except Exception:
        yield None
        return
    with track_usage() as tracker:
        yield tracker


def usage_baseline(tracker: Any) -> dict[str, int]:
    """Per-model call counts to slice from. ``add_usage`` only ever APPENDS, which is what makes a
    length snapshot a valid cursor."""
    data = getattr(tracker, "usage_data", None)
    return {k: len(v) for k, v in data.items()} if isinstance(data, dict) else {}


def usage_since(tracker: Any, baseline: dict[str, int]) -> dict[str, list]:
    """The calls recorded on ``tracker`` since ``baseline`` -- keyed by the MODEL-NAME STRING.

    ``usage_data`` is keyed by ``lm.model`` (a string) and is a ``defaultdict(list)``, so indexing
    it with an LM OBJECT returns ``[]`` silently AND inserts a bogus key. ``base.get(k, 0)``, never
    ``base[k]``: a model that first appears mid-scope (a tool-LM on another model) has no baseline.
    """
    data = getattr(tracker, "usage_data", None)
    if not isinstance(data, dict):
        return {}
    out = {}
    for model, entries in data.items():
        fresh = list(entries[baseline.get(model, 0):])
        if fresh:
            out[str(model)] = fresh
    return out
