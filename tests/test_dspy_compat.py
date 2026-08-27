"""``_dspy_compat`` — the cross-dspy-version shims.

WHY THIS FILE EXISTS. dspy 3.3.0 renamed three things at once and every one of them
failed in a way the previous suite could not see:

1. ``RLM(interpreter=…)`` moved to ``forward()``'s first positional arg → a HARD
   ``TypeError`` on every run, including the default pyodide path. Loud, at least.
2. ``max_iterations`` → ``max_iters`` → the old all-or-nothing ``try`` around the
   budget kwargs swallowed it and dropped ALL THREE caps to dspy's defaults. Silent.
3. ``CodeInterpreterError`` stopped being the recoverable interpreter error (the new
   ``CodeExecutionError`` subclass took that role) → a per-turn sandbox timeout, and
   any exception the model's own code raised in the container, went from "retry next
   turn" to "kill the run". Silent, and only visible under load.

The kit declares only a FLOOR on dspy while consumers pin the KIT, so a consumer's fresh
install picks whatever dspy is current — none of them can be expected to notice any of the
above. These tests assert the shim's CONTRACT against the installed dspy, so the next rename
lands here as a red test instead of in someone's rollout. That is why they survived the 1.2.0
floor bump: the shims now resolve a single answer each, but these are what make the NEXT
rename loud.
"""

from __future__ import annotations

import inspect

import pytest

dspy = pytest.importorskip("dspy", reason="dspy not installed")

from rlm_harness import _dspy_compat


@pytest.fixture(autouse=True)
def _clear_caches():
    """Every shim is ``lru_cache``d (the installed dspy can't change mid-process), so a
    test that monkeypatches dspy must not leak its answer into the next one."""
    for fn in (
        _dspy_compat._rlm_init_signature,
        _dspy_compat._rlm_init_params,
        _dspy_compat._rlm_init_takes_var_keyword,
        _dspy_compat.reserved_tool_names,
        _dspy_compat.reserved_result_names,
        _dspy_compat.recoverable_interpreter_error,
        _dspy_compat.terminal_interpreter_error,
        _dspy_compat._lm_error_classes,
        _dspy_compat._dspy_reads_execution_instructions,
    ):
        fn.cache_clear()
    yield
    for fn in (
        _dspy_compat._rlm_init_signature,
        _dspy_compat._rlm_init_params,
        _dspy_compat._rlm_init_takes_var_keyword,
        _dspy_compat.reserved_tool_names,
        _dspy_compat.reserved_result_names,
        _dspy_compat.recoverable_interpreter_error,
        _dspy_compat.terminal_interpreter_error,
        _dspy_compat._lm_error_classes,
        _dspy_compat._dspy_reads_execution_instructions,
    ):
        fn.cache_clear()


# ---- budget caps ---------------------------------------------------------------


def test_budget_kwargs_are_all_accepted_by_the_installed_dspy():
    """THE regression for silent cap loss: every name the shim emits must be a real
    parameter of the installed ``dspy.RLM.__init__``, or the constructor raises and
    `_build_rlm`'s fallback quietly drops the caller's budget entirely."""
    resolved = _dspy_compat.rlm_budget_kwargs(
        max_iterations=7, max_llm_calls=11, max_output_chars=13
    )
    accepted = set(inspect.signature(dspy.RLM.__init__).parameters)
    assert set(resolved) <= accepted, (
        f"{sorted(set(resolved) - accepted)} is not a dspy.RLM kwarg on dspy "
        f"{dspy.__version__}; update _BUDGET_ALIASES"
    )


def test_all_three_budget_caps_are_mapped():
    """No cap may be silently dropped — the values must survive the renaming."""
    resolved = _dspy_compat.rlm_budget_kwargs(
        max_iterations=7, max_llm_calls=11, max_output_chars=13
    )
    assert len(resolved) == 3, f"a cap went missing: {resolved}"
    assert sorted(resolved.values()) == [7, 11, 13]


def test_iteration_cap_uses_whichever_name_this_dspy_has():
    resolved = _dspy_compat.rlm_budget_kwargs(
        max_iterations=7, max_llm_calls=11, max_output_chars=13
    )
    name = "max_iters" if "max_iters" in resolved else "max_iterations"
    assert resolved[name] == 7


def test_budget_prefers_the_newest_alias(monkeypatch):
    """When a dspy accepts BOTH spellings, send the newer one — the older is on its way out
    and a deprecation shim can vanish in a patch release.

    `_BUDGET_ALIASES` is monkeypatched to hold two names. Without that this test would be a
    TAUTOLOGY after 1.2.0 dropped the legacy spellings: the shim would have only one name to
    choose from, so the assertion could not fail however the preference logic behaved. What is
    under test is the ORDERING RULE, which has to outlive any particular alias list."""

    def _both(self, signature, max_iters=20, max_iterations=20,
              max_llm_calls=50, max_output_chars=10_000):
        ...

    monkeypatch.setattr(dspy.RLM, "__init__", _both, raising=True)
    monkeypatch.setattr(
        _dspy_compat, "_BUDGET_ALIASES",
        (("max_iterations", ("max_iters", "max_iterations")),
         ("max_llm_calls", ("max_llm_calls",)),
         ("max_output_chars", ("max_output_chars",))),
        raising=True,
    )
    for fn in (_dspy_compat._rlm_init_signature, _dspy_compat._rlm_init_params,
               _dspy_compat._rlm_init_takes_var_keyword):
        fn.cache_clear()

    resolved = _dspy_compat.rlm_budget_kwargs(
        max_iterations=7, max_llm_calls=11, max_output_chars=13
    )
    assert "max_iters" in resolved and "max_iterations" not in resolved


# ---- the caller-owned interpreter ------------------------------------------------


def test_the_interpreter_seam_still_exists_on_this_dspy():
    """THE tripwire for the seam. `forward_interpreter_args` has no dspy contact left after the
    3.3.0 floor — it is a one-liner — so this assertion is the only thing that would notice dspy
    moving the interpreter again. It must stay UNCONDITIONAL: making it mirror the shim would
    make it a tautology, and the seam would then be untestable by construction."""
    for method in (dspy.RLM.forward, dspy.RLM.aforward):
        params = list(inspect.signature(method).parameters.values())
        positional = [
            p for p in params
            if p.name != "self"
            and p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                           inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        assert positional, f"{method.__name__} takes no positional interpreter"
        assert positional[0].name == "interpreter", (
            f"{method.__name__}'s first positional arg is {positional[0].name!r}, not "
            f"'interpreter' — dspy moved the seam; update `forward_interpreter_args`."
        )


def test_no_interpreter_still_means_no_forward_args():
    assert _dspy_compat.forward_interpreter_args(None) == ()


# ---- recoverable vs terminal interpreter errors ----------------------------------


def test_recoverable_error_is_the_one_dspy_actually_catches():
    """The heart of failure mode 3. dspy's RLM catches its recoverable interpreter
    error and hands the model another turn; anything else ends the run. The shim must
    name the class that is on the CAUGHT side for the installed dspy."""
    from dspy.primitives import code_interpreter

    assert _dspy_compat.recoverable_interpreter_error() is code_interpreter.CodeExecutionError


def test_recoverable_error_is_catchable_as_the_base_class():
    """Consumers and the kit's own `except CodeInterpreterError` sites must keep
    working: the recoverable class has to remain a SUBCLASS of the base, not a sibling."""
    from dspy.primitives.code_interpreter import CodeInterpreterError

    assert issubclass(_dspy_compat.recoverable_interpreter_error(), CodeInterpreterError)


def test_sandbox_cancelled_is_never_caught_as_an_interpreter_error():
    """The `SandboxCancelled` invariant (CLAUDE.md), asserted against the REAL classes
    rather than by reading the code: a caller-driven cancel must not be absorbed by
    dspy's recoverable-error handling on ANY supported dspy, or it degrades into a
    retried turn and the cancel is silently ignored."""
    from dspy.primitives.code_interpreter import CodeInterpreterError

    from rlm_harness import SandboxCancelled

    assert not issubclass(SandboxCancelled, CodeInterpreterError)
    assert not issubclass(SandboxCancelled, _dspy_compat.recoverable_interpreter_error())
    assert not issubclass(SandboxCancelled, _dspy_compat.terminal_interpreter_error())
    # dspy 3.3.1 re-parented CodeInterpreterError under DSPyError. `SandboxCancelled` stands
    # OUTSIDE dspy's hierarchy entirely, and that is what makes it non-recoverable on every
    # version — so pin the new root too, not just the leaf that moved.
    assert not issubclass(SandboxCancelled, dspy.DSPyError)


def test_terminal_error_is_not_the_recoverable_one_when_dspy_splits_them():
    from dspy.primitives import code_interpreter

    terminal = _dspy_compat.terminal_interpreter_error()
    assert terminal is code_interpreter.CodeInterpreterError
    assert terminal is not _dspy_compat.recoverable_interpreter_error()


# ---- the interpreter's execution instructions -------------------------------------


class _Descriptive:
    """A caller-owned interpreter that describes its own runtime, like the kit's own do."""

    execution_instructions = "Runs in a container. Subprocesses ARE available."
    tools: dict = {}

    def start(self): ...
    def execute(self, code, variables=None): return ""
    def shutdown(self): ...


def test_the_carrier_puts_our_text_in_the_prompt_not_dspys_pyodide_default():
    """THE regression. Without this the action prompt tells a container run that subprocesses
    are unavailable — dspy reads the text off `_interpreter_factory`, which defaults to
    PythonInterpreter no matter what is actually executing the code."""
    kwargs = _dspy_compat.interpreter_instructions_kwargs(_Descriptive())
    assert set(kwargs) == {"interpreter_factory"}

    rlm = dspy.RLM(dspy.Signature("doc: str -> answer: str"), **kwargs)
    instructions = rlm.generate_action.signature.instructions
    assert "Subprocesses ARE available" in instructions
    assert "Pyodide" not in instructions


def test_the_carrier_raises_if_dspy_ever_invokes_it():
    """It is a metadata carrier, never a constructor: dspy shuts down whatever a factory
    RETURNS, which would double-shutdown the kit's sandbox. Fail loudly instead."""
    factory = _dspy_compat.interpreter_instructions_kwargs(_Descriptive())["interpreter_factory"]
    with pytest.raises(RuntimeError, match="never be INVOKED"):
        factory()


def test_a_module_carrying_the_factory_still_copies_and_serialises():
    """dspy copies and dumps modules (optimizers do it constantly). A factory object that broke
    `deepcopy` or `dump_state` would turn this shim into a landmine far from its call site.

    (That the factory is never INVOKED rests on the positional seam, which
    `test_the_interpreter_seam_still_exists_on_this_dspy` above pins unconditionally.)"""
    import copy

    rlm = dspy.RLM(
        dspy.Signature("doc: str -> answer: str"),
        **_dspy_compat.interpreter_instructions_kwargs(_Descriptive()),
    )
    assert copy.deepcopy(rlm) is not rlm
    assert isinstance(rlm.dump_state(), dict)


def test_a_pyodide_interpreter_gets_no_carrier():
    """dspy's own default already describes those correctly; carrying its text back would be
    noise, and would mean passing a factory for no reason."""
    from dspy.primitives.python_interpreter import PythonInterpreter

    interp = PythonInterpreter.__new__(PythonInterpreter)   # no Deno subprocess spawned
    assert _dspy_compat.interpreter_instructions_kwargs(interp) == {}


@pytest.mark.parametrize("interp", [object(), None], ids=["no-attribute", "none"])
def test_an_interpreter_that_describes_nothing_changes_nothing(interp):
    assert _dspy_compat.interpreter_instructions_kwargs(interp) == {}


def test_blank_instructions_are_treated_as_absent():
    class _Blank:
        execution_instructions = "   "

    assert _dspy_compat.interpreter_instructions_kwargs(_Blank()) == {}


def test_no_carrier_when_dspy_does_not_accept_the_kwarg(monkeypatch):
    """Load-bearing, and NOT redundant with the render probe: `_build_rlm`'s `except TypeError`
    fallback re-passes the same kwargs, so an unknown kwarg raises on BOTH constructions and
    takes the whole run down instead of degrading to dspy's defaults."""
    def _init(self, signature, sub_lm=None, tools=None):   # no interpreter_factory, no **kwargs
        ...

    monkeypatch.setattr(dspy.RLM, "__init__", _init)
    _dspy_compat._rlm_init_signature.cache_clear()
    _dspy_compat._rlm_init_params.cache_clear()
    _dspy_compat._rlm_init_takes_var_keyword.cache_clear()
    assert _dspy_compat.interpreter_instructions_kwargs(_Descriptive()) == {}


def test_every_interpreter_the_kit_ships_describes_itself():
    """A sweep, so a NEW kit interpreter cannot silently inherit dspy's Pyodide description —
    and so deleting one of these attributes goes red. `_JsonLiteralInterpreter` is deliberately
    absent: it subclasses PythonInterpreter, whose own text is already correct."""
    from rlm_harness.sandbox import build_interpreter
    from rlm_harness.testing import ScriptedInterpreter

    for interp in (build_interpreter("mock"), ScriptedInterpreter()):
        text = getattr(interp, "execution_instructions", "")
        assert isinstance(text, str) and text.strip(), f"{type(interp).__name__} describes nothing"
        assert "Pyodide" not in text
        # ...and the shim actually carries it, rather than the attribute being decorative.
        kwargs = _dspy_compat.interpreter_instructions_kwargs(interp)
        assert kwargs["interpreter_factory"].execution_instructions == text


def test_no_carrier_when_dspy_does_not_render_the_text(monkeypatch):
    """A dspy that never renders it gets nothing — the shim resolves the answer by
    introspection, so an older dspy degrades to exactly today's behaviour."""
    monkeypatch.setattr(_dspy_compat, "_dspy_reads_execution_instructions", lambda: False)
    assert _dspy_compat.interpreter_instructions_kwargs(_Descriptive()) == {}


# ---- fast-failing non-retryable LM errors -----------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        dspy.LMAuthError("bad key"),
        dspy.LMBillingError("out of credit"),
        dspy.LMConfigurationError("no model configured"),
        dspy.LMUnsupportedModelError("unknown model"),
        dspy.LMUnsupportedFeatureError("no such feature"),
    ],
)
def test_lm_errors_dspy_calls_non_retryable_fail_fast(exc):
    """The core contract: an LM error dspy's OWN `is_retryable_lm_error` says not to retry
    must not burn the retry budget re-running the same doomed trajectory."""
    assert _dspy_compat.is_fast_fail_lm_error(exc) is True


@pytest.mark.parametrize(
    "exc",
    [
        dspy.LMRateLimitError("slow down"),
        dspy.LMTimeoutError("timed out"),
        dspy.LMServerError("500"),
        dspy.LMTransportError("connection reset"),
    ],
)
def test_lm_errors_dspy_calls_retryable_keep_retrying(exc):
    """The mirror: anything dspy's own helper calls retryable must not fast-fail here — this
    predicate must never be STRICTER than dspy's own classification."""
    assert _dspy_compat.is_fast_fail_lm_error(exc) is False


def test_context_window_exceeded_is_the_one_carve_out():
    """THE regression for the contested part of the design (CHANGELOG 1.2.0).
    `ContextWindowExceededError` is a non-retryable `LMInvalidRequestError` by dspy's own
    classification, but `run_with_retry` re-runs the WHOLE trajectory rather than resending the
    identical request — a later attempt can genuinely produce a shorter prompt that fits. It must
    keep retrying here even though dspy calls it non-retryable."""
    exc = dspy.ContextWindowExceededError()
    assert dspy.is_retryable_lm_error(exc) is False  # confirms the premise: dspy says no
    assert _dspy_compat.is_fast_fail_lm_error(exc) is False  # this shim disagrees, on purpose


def test_non_lm_exception_never_fast_fails():
    assert _dspy_compat.is_fast_fail_lm_error(ValueError("not an LM error at all")) is False


def test_unclassifiable_lm_error_still_gets_a_verdict():
    """`LMUnexpectedError` is dspy's own catch-all bucket for a failure it could not classify
    more precisely. It is still an `LMError` that `is_retryable_lm_error` calls non-retryable, so
    it fast-fails too — trusting dspy's classification rather than second-guessing it here."""
    assert _dspy_compat.is_fast_fail_lm_error(dspy.LMUnexpectedError("???")) is True


def test_missing_is_retryable_helper_degrades_to_never_fast_fail(monkeypatch):
    """A future/older dspy without `is_retryable_lm_error` must not be treated as
    'everything fast-fails' — that would be MORE aggressive than today's behavior with no
    classification to back it. Degrade to the pre-existing behavior: always retry."""
    monkeypatch.delattr(dspy, "is_retryable_lm_error", raising=True)
    assert _dspy_compat.is_fast_fail_lm_error(dspy.LMAuthError("bad key")) is False


def test_missing_lm_error_class_degrades_to_never_fast_fail(monkeypatch):
    """A dspy without `LMError` at all can't be classified — never fast-fail rather than guess."""
    monkeypatch.delattr(dspy, "LMError", raising=True)
    assert _dspy_compat.is_fast_fail_lm_error(dspy.LMAuthError("bad key")) is False


# ---- import hygiene --------------------------------------------------------------


def test_module_top_is_dspy_free():
    """`_dspy_compat` is on the dspy-free list (CLAUDE.md): importing it must NOT drag
    dspy in. It is imported at `task.py`'s module top and from inside `sandbox.py`, so a
    stray top-level `import dspy` here would quietly make `sandbox.py` dspy-bearing.
    Checked in a SUBPROCESS because this test session has already imported dspy."""
    import subprocess
    import sys

    code = (
        "import sys; import rlm_harness._dspy_compat as m; "
        "assert 'dspy' not in sys.modules, 'importing _dspy_compat pulled in dspy'; "
        "assert callable(m.recoverable_interpreter_error); print('ok')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


def test_importing_the_package_still_does_not_import_dspy():
    """The companion invariant: `import rlm_harness` stays cheap. `task.py` imports
    `_dspy_compat` at ITS module top, which is fine only because `task.py` itself is a
    lazy `__getattr__` re-export — this pins that it stayed that way."""
    import subprocess
    import sys

    code = (
        "import sys, rlm_harness; "
        "assert 'dspy' not in sys.modules, 'import rlm_harness pulled in dspy'; print('ok')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


# ---- degraded introspection ------------------------------------------------------


def test_var_keyword_signature_falls_back_to_the_current_names(monkeypatch):
    """A dspy whose ``RLM.__init__`` is ``(*args, **kwargs)`` tells us nothing — probing names
    proves nothing, so send the names this kit currently targets and let the constructor judge.

    This branch is STILL LIVE after the 3.3.0 floor (`_rlm_init_takes_var_keyword` has a real
    caller), and it reads `candidates[-1]`. Shrinking `_BUDGET_ALIASES` therefore CHANGED what
    it sends — this test is the only coverage of that, which is why it was updated rather than
    deleted with the other two-version cases."""

    def _opaque(self, *args, **kwargs):
        ...

    monkeypatch.setattr(dspy.RLM, "__init__", _opaque, raising=True)
    for fn in (_dspy_compat._rlm_init_signature, _dspy_compat._rlm_init_params,
               _dspy_compat._rlm_init_takes_var_keyword):
        fn.cache_clear()

    resolved = _dspy_compat.rlm_budget_kwargs(
        max_iterations=7, max_llm_calls=11, max_output_chars=13
    )
    assert resolved == {"max_iters": 7, "max_llm_calls": 11, "max_output_chars": 13}
