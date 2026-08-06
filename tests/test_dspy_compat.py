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

The kit declares ``dspy>=3.2.1`` while consumers pin the KIT, so a consumer's fresh
install picks whatever dspy is current — none of them can be expected to notice any
of the above. These tests assert the shim's CONTRACT against the installed dspy, so
the next rename lands here as a red test instead of in someone's rollout.
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
        _dspy_compat.rlm_accepts_interpreter_kwarg,
        _dspy_compat.recoverable_interpreter_error,
        _dspy_compat.terminal_interpreter_error,
    ):
        fn.cache_clear()
    yield
    for fn in (
        _dspy_compat._rlm_init_signature,
        _dspy_compat._rlm_init_params,
        _dspy_compat._rlm_init_takes_var_keyword,
        _dspy_compat.rlm_accepts_interpreter_kwarg,
        _dspy_compat.recoverable_interpreter_error,
        _dspy_compat.terminal_interpreter_error,
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
    """When a dspy accepts BOTH spellings, send the newer one — the older is the one
    on its way out, and a deprecation shim can disappear in a patch release."""

    def _both(self, signature, max_iters=20, max_iterations=20,
              max_llm_calls=50, max_output_chars=10_000):
        ...

    monkeypatch.setattr(dspy.RLM, "__init__", _both, raising=True)
    for fn in (_dspy_compat._rlm_init_signature, _dspy_compat._rlm_init_params,
               _dspy_compat._rlm_init_takes_var_keyword):
        fn.cache_clear()

    resolved = _dspy_compat.rlm_budget_kwargs(
        max_iterations=7, max_llm_calls=11, max_output_chars=13
    )
    assert "max_iters" in resolved and "max_iterations" not in resolved


# ---- the caller-owned interpreter ------------------------------------------------


def test_interpreter_goes_to_exactly_one_seam():
    """It is passed to the constructor XOR to forward() — never both (dspy would then
    hold one interpreter and be handed another), never neither (the sandbox the kit
    built, with its JSON-literal aliases and watchdog, would be silently ignored)."""
    sentinel = object()
    via_ctor = _dspy_compat.rlm_accepts_interpreter_kwarg()
    via_forward = bool(_dspy_compat.forward_interpreter_args(sentinel))
    assert via_ctor != via_forward


def test_the_chosen_seam_actually_exists_on_this_dspy():
    if _dspy_compat.rlm_accepts_interpreter_kwarg():
        assert "interpreter" in inspect.signature(dspy.RLM.__init__).parameters
    else:
        # 3.3.x: forward()/aforward() take it positionally, before **input_args.
        for method in (dspy.RLM.forward, dspy.RLM.aforward):
            params = list(inspect.signature(method).parameters.values())
            positional = [
                p for p in params
                if p.name != "self"
                and p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                               inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
            assert positional, f"{method.__name__} takes no positional interpreter"
            assert positional[0].name == "interpreter"


def test_no_interpreter_means_no_forward_args():
    assert _dspy_compat.forward_interpreter_args(None) == ()


# ---- recoverable vs terminal interpreter errors ----------------------------------


def test_recoverable_error_is_the_one_dspy_actually_catches():
    """The heart of failure mode 3. dspy's RLM catches its recoverable interpreter
    error and hands the model another turn; anything else ends the run. The shim must
    name the class that is on the CAUGHT side for the installed dspy."""
    from dspy.primitives import code_interpreter

    recoverable = _dspy_compat.recoverable_interpreter_error()
    expected = getattr(
        code_interpreter, "CodeExecutionError", code_interpreter.CodeInterpreterError
    )
    assert recoverable is expected


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


def test_terminal_error_is_not_the_recoverable_one_when_dspy_splits_them():
    from dspy.primitives import code_interpreter

    terminal = _dspy_compat.terminal_interpreter_error()
    assert terminal is code_interpreter.CodeInterpreterError
    if hasattr(code_interpreter, "CodeExecutionError"):
        assert terminal is not _dspy_compat.recoverable_interpreter_error()


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


def test_var_keyword_signature_falls_back_to_the_legacy_names(monkeypatch):
    """A dspy whose ``RLM.__init__`` is ``(*args, **kwargs)`` tells us nothing. Probing
    names then proves nothing either, so fall back to what the kit has always sent and
    let the constructor be the judge."""

    def _opaque(self, *args, **kwargs):
        ...

    monkeypatch.setattr(dspy.RLM, "__init__", _opaque, raising=True)
    for fn in (_dspy_compat._rlm_init_signature, _dspy_compat._rlm_init_params,
               _dspy_compat._rlm_init_takes_var_keyword,
               _dspy_compat.rlm_accepts_interpreter_kwarg):
        fn.cache_clear()

    assert _dspy_compat.rlm_accepts_interpreter_kwarg() is True
    resolved = _dspy_compat.rlm_budget_kwargs(
        max_iterations=7, max_llm_calls=11, max_output_chars=13
    )
    assert resolved == {"max_iterations": 7, "max_llm_calls": 11, "max_output_chars": 13}
