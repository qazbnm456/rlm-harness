import importlib.util
import threading
import time

import pytest

from rlm_kit.sandbox import (
    _JSON_LITERAL_ALIASES,
    SandboxCancelled,
    SandboxSecurityError,
    build_interpreter,
)

# The pyodide/deno path now constructs dspy's PythonInterpreter (to inject the
# JSON-literal aliases), so it needs dspy at call time. Construction stays lazy —
# Deno is not spawned — so these run without a sandbox, just not without dspy.
_HAS_DSPY = importlib.util.find_spec("dspy") is not None
_needs_dspy = pytest.mark.skipif(not _HAS_DSPY, reason="dspy not installed")


@_needs_dspy
def test_pyodide_returns_alias_injecting_sandbox():
    interp = build_interpreter("pyodide")
    assert interp is not None
    # Same aliases, and constructing it did NOT spawn the Deno subprocess.
    assert interp._JSON_ALIASES == _JSON_LITERAL_ALIASES
    assert getattr(interp, "deno_process", None) is None
    interp.shutdown()


@_needs_dspy
def test_deno_returns_alias_injecting_sandbox():
    interp = build_interpreter("deno")
    assert interp._JSON_ALIASES == {"true": True, "false": False, "null": None}
    interp.shutdown()


@_needs_dspy
def test_none_defaults_to_alias_injecting_sandbox():
    interp = build_interpreter(None)
    assert interp is not None
    assert interp._JSON_ALIASES == _JSON_LITERAL_ALIASES
    interp.shutdown()


@_needs_dspy
def test_sandbox_execute_merges_json_aliases_into_variables(monkeypatch):
    """The override merges true/false/null into the variables dspy passes to the
    parent execute, so a JSON-trained model's `SUBMIT({"x": true})` resolves."""
    from dspy.primitives.python_interpreter import PythonInterpreter

    captured = {}

    def fake_super_execute(self, code, variables=None):
        captured["code"] = code
        captured["variables"] = variables
        return "ok"

    monkeypatch.setattr(PythonInterpreter, "execute", fake_super_execute)
    interp = build_interpreter("deno")
    assert interp.execute("SUBMIT({'x': true})", {"source": "s"}) == "ok"
    assert captured["variables"] == {
        "true": True,
        "false": False,
        "null": None,
        "source": "s",
    }


def test_local_without_optin_is_refused():
    with pytest.raises(SandboxSecurityError):
        build_interpreter("local")


def test_local_without_optin_refused_even_case_insensitive():
    with pytest.raises(SandboxSecurityError):
        build_interpreter("LOCAL")


def test_unknown_interpreter_raises_value_error():
    with pytest.raises(ValueError):
        build_interpreter("rce-please")


def test_mock_interpreter_has_execute():
    interp = build_interpreter("mock")
    assert interp is not None
    assert interp.execute("1+1") == ""


# ---- watchdog: turn_timeout_s / cancel_event on the pyodide/deno interpreter ----------------


@_needs_dspy
def test_both_knobs_none_never_starts_a_watcher_thread(monkeypatch):
    """The disabled-by-default guarantee: `execute()`'s first check must call
    `super().execute(...)` directly with NO watcher thread — verified by asserting
    `threading.Thread`'s constructor is never called, not just by timing."""
    from dspy.primitives.python_interpreter import PythonInterpreter

    monkeypatch.setattr(PythonInterpreter, "execute", lambda self, code, variables=None: "fast")
    created = {"n": 0}
    real_thread = threading.Thread

    def counting_thread(*a, **kw):
        created["n"] += 1
        return real_thread(*a, **kw)

    monkeypatch.setattr(threading, "Thread", counting_thread)
    interp = build_interpreter("pyodide")
    assert interp.execute("1+1") == "fast"
    assert created["n"] == 0


@_needs_dspy
def test_a_knob_set_but_never_firing_returns_the_real_result_untouched(monkeypatch):
    """The guarded-but-never-fires path: a knob is set, but the underlying call finishes
    normally well before the deadline/cancel — the watchdog must be a pure no-op here, not
    just when disabled entirely (that's the OTHER test)."""
    from dspy.primitives.python_interpreter import PythonInterpreter

    monkeypatch.setattr(PythonInterpreter, "execute", lambda self, code, variables=None: "real result")
    interp = build_interpreter("pyodide", turn_timeout_s=5.0)
    interp.deno_process = type("P", (), {"kill": lambda self: pytest.fail("must not be killed")})()
    assert interp.execute("1+1") == "real result"

    ev = threading.Event()  # never set
    interp2 = build_interpreter("pyodide", cancel_event=ev)
    interp2.deno_process = type("P", (), {"kill": lambda self: pytest.fail("must not be killed")})()
    assert interp2.execute("1+1") == "real result"


@_needs_dspy
def test_turn_timeout_kills_and_raises_code_interpreter_error(monkeypatch):
    from dspy.primitives.code_interpreter import CodeInterpreterError
    from dspy.primitives.python_interpreter import PythonInterpreter

    def hangs(self, code, variables=None):
        time.sleep(2.0)
        return "too late"

    monkeypatch.setattr(PythonInterpreter, "execute", hangs)
    interp = build_interpreter("pyodide", turn_timeout_s=0.2)
    killed = {"n": 0}
    interp.deno_process = type("P", (), {"kill": lambda self: killed.__setitem__("n", killed["n"] + 1)})()

    with pytest.raises(CodeInterpreterError, match="exceeded"):
        interp.execute("while True: pass")
    assert killed["n"] >= 1


@_needs_dspy
def test_cancel_event_kills_and_raises_sandbox_cancelled_not_code_interpreter_error(monkeypatch):
    from dspy.primitives.python_interpreter import PythonInterpreter

    def hangs(self, code, variables=None):
        time.sleep(2.0)
        return "too late"

    monkeypatch.setattr(PythonInterpreter, "execute", hangs)
    ev = threading.Event()
    ev.set()  # already cancelled before the call even starts
    interp = build_interpreter("pyodide", cancel_event=ev)
    interp.deno_process = type("P", (), {"kill": lambda self: None})()

    with pytest.raises(SandboxCancelled):
        interp.execute("while True: pass")


@_needs_dspy
def test_mid_call_respawn_that_ultimately_succeeds_still_raises_when_cancelled(monkeypatch):
    """REQUIRED test (variant a) — catches the round-2 "absorbed cancel" regression: the
    ENTIRE super().execute() call completes to a normal, final result with NO exception
    raised anywhere inside it, simulating dspy's own BrokenPipeError -> respawn -> retry
    recovery finishing cleanly. The GUARDED call must still raise SandboxCancelled despite
    that clean success — a test that only asserts `.kill()` was called would pass against
    the broken (round-1) code just as readily as the fixed code; only checking the outcome
    of the GUARDED call actually distinguishes them."""
    from dspy.primitives.python_interpreter import PythonInterpreter

    def eventually_succeeds(self, code, variables=None):
        # The watchdog has already fired (cancel_event is set before the call starts) and
        # is re-killing on every 0.1s tick; this fake simulates dspy's own recovery
        # completing successfully anyway before returning.
        time.sleep(0.05)
        return "a normal, successful result"

    monkeypatch.setattr(PythonInterpreter, "execute", eventually_succeeds)
    ev = threading.Event()
    ev.set()
    interp = build_interpreter("pyodide", cancel_event=ev)
    interp.deno_process = type("P", (), {"kill": lambda self: None})()

    with pytest.raises(SandboxCancelled):
        interp.execute("print(1)")


@_needs_dspy
def test_mid_call_respawn_that_fails_again_raises_cleanly_not_a_raw_pipe_error(monkeypatch):
    """REQUIRED test (variant b): the retry-after-respawn ALSO fails, with a raw
    BrokenPipeError (dspy's own retry path has no try/except around its second write) —
    must still surface as SandboxCancelled, never a raw BrokenPipeError escaping
    ungracefully."""
    from dspy.primitives.python_interpreter import PythonInterpreter

    def raises_broken_pipe(self, code, variables=None):
        raise BrokenPipeError("simulated: the retried write also failed")

    monkeypatch.setattr(PythonInterpreter, "execute", raises_broken_pipe)
    ev = threading.Event()
    ev.set()
    interp = build_interpreter("pyodide", cancel_event=ev)
    interp.deno_process = type("P", (), {"kill": lambda self: None})()

    with pytest.raises(SandboxCancelled):
        interp.execute("print(1)")


@_needs_dspy
def test_a_plain_syntax_error_racing_a_fired_watchdog_is_still_mapped_cleanly(monkeypatch):
    """REQUIRED test (variant c, added per round 3): dspy's real execute() raises a plain
    SyntaxError (not CodeInterpreterError) on JSON-RPC error code -32000 (invalid Python) —
    must not escape the guarded call untouched when a watchdog reason is already set."""
    from dspy.primitives.python_interpreter import PythonInterpreter

    def raises_syntax_error(self, code, variables=None):
        raise SyntaxError("simulated: invalid Python (-32000)")

    monkeypatch.setattr(PythonInterpreter, "execute", raises_syntax_error)
    ev = threading.Event()
    ev.set()
    interp = build_interpreter("pyodide", cancel_event=ev)
    interp.deno_process = type("P", (), {"kill": lambda self: None})()

    with pytest.raises(SandboxCancelled):
        interp.execute("print(1")


@_needs_dspy
def test_an_unrelated_failure_with_no_watchdog_set_propagates_unchanged(monkeypatch):
    """The 'not fired' path: a genuine, unrelated interpreter failure with NEITHER knob
    set must propagate completely untouched — no watchdog wrapping at all is even entered
    (both knobs None), matching today's behavior exactly."""
    from dspy.primitives.code_interpreter import CodeInterpreterError
    from dspy.primitives.python_interpreter import PythonInterpreter

    def raises_normally(self, code, variables=None):
        raise CodeInterpreterError("a real, unrelated interpreter failure")

    monkeypatch.setattr(PythonInterpreter, "execute", raises_normally)
    interp = build_interpreter("pyodide")
    with pytest.raises(CodeInterpreterError, match="unrelated"):
        interp.execute("print(1)")


def test_build_interpreter_sets_watchdog_attributes_on_the_pyodide_interpreter():
    if not _HAS_DSPY:
        pytest.skip("dspy not installed")
    ev = threading.Event()
    interp = build_interpreter("pyodide", turn_timeout_s=5.0, cancel_event=ev)
    assert interp._turn_timeout_s == 5.0
    assert interp._cancel_event is ev


def test_build_interpreter_refuses_cancel_event_for_container():
    with pytest.raises(ValueError, match="cancel_event"):
        build_interpreter("container", cancel_event=threading.Event())


def test_build_interpreter_refuses_cancel_event_for_mock():
    with pytest.raises(ValueError, match="cancel_event"):
        build_interpreter("mock", cancel_event=threading.Event())


def test_build_interpreter_silently_accepts_turn_timeout_for_mock():
    # turn_timeout_s is silently irrelevant to a kind with no blocking call at all —
    # mirrors how `allow_insecure` is already silently irrelevant outside `local`.
    interp = build_interpreter("mock", turn_timeout_s=5.0)
    assert interp.execute("1+1") == ""
