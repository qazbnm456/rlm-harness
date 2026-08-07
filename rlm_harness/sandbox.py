"""Sandbox / code-interpreter selection — the security boundary of the scaffold.

RLM works by letting the model write and execute Python in a REPL. When that
REPL is fed half-trusted scraped content (a common case the moment a task pulls
from the web or any untrusted source), the interpreter choice *is* the attack surface.

Policy:

- ``pyodide`` / ``deno`` → dspy.RLM's default sandboxed (WASM / subprocess)
  interpreter, but constructed *here* as a thin subclass that pre-binds the JSON
  literals ``true``/``false``/``null`` in the REPL namespace (see
  ``_build_sandboxed_interpreter``). Same isolation as dspy's own default. Safe
  for untrusted content.
- ``mock`` → a no-op interpreter for tests.
- ``container`` → the environment interpreter (``container_interpreter.py``,
  opt-in): the REPL runs *inside* an isolated Docker container so model code can
  spawn subprocesses natively. A STRONGER boundary than the WASM sandbox
  (``--network=none``, LM creds host-side, caps dropped) and the OPPOSITE of
  ``local`` — handled *before* the insecure-interpreter check below, never routed
  through it. Needs the ``docker`` CLI (imported lazily; this module stays dspy-free).
- ``local`` → executes model-written code directly on the host. This is
  effectively arbitrary code execution and is refused unless the caller has
  *explicitly* opted in. The opt-in cannot be reached by accident.

``dspy`` is imported lazily inside the branches that need it so this module —
and the security guard in particular — stays importable and testable without a
full dspy install.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

# Interpreters that run model-written code on the host with no isolation.
INSECURE_INTERPRETERS = frozenset({"local"})


class SandboxSecurityError(RuntimeError):
    """Raised when an insecure interpreter is requested without explicit opt-in."""


#: Raised when `cancel_event` fires during a sandbox `execute()` call — deliberately NOT any
#: dspy interpreter-error subclass. dspy's ``RLM._execute_code`` catches only its own
#: recoverable interpreter error plus ``SyntaxError`` (``CodeExecutionError`` — see
#: ``_dspy_compat``), so this propagates
#: all the way up through ``RLMTask.arun()`` (and through ``run_with_retry``'s ``non_retryable``
#: allowlist, untouched) as a genuine run-ending failure on EVERY supported dspy. Standing
#: outside dspy's hierarchy entirely is what makes that true across the rename. A plain
#: ``RuntimeError`` subclass with zero dspy dependency, so it is defined here at module top
#: without touching this module's lazy-dspy-import discipline — the same posture
#: ``SandboxSecurityError`` above already has.
class SandboxCancelled(RuntimeError):
    """A caller explicitly cancelled a sandbox execution in progress."""


def build_interpreter(
    kind: str,
    *,
    allow_insecure: bool = False,
    container: Any = None,
    turn_timeout_s: float | None = None,
    cancel_event: threading.Event | None = None,
) -> Any | None:
    """Return a dspy ``CodeInterpreter`` for ``kind``, or ``None`` for the default.

    ``None`` means "let dspy.RLM construct its own default sandboxed interpreter",
    which is the secure path. A non-None return is produced for ``mock``, ``container``
    (needs the ``container`` options + the ``docker`` CLI), and (when explicitly allowed)
    ``local``.

    ``turn_timeout_s``/``cancel_event`` are watchdog knobs for the ``pyodide``/``deno``
    branch ONLY — see ``_build_sandboxed_interpreter``. ``cancel_event`` alongside any
    OTHER kind is a hard refusal (see below), not a silent no-op: ``container`` already has
    its own ``ContainerConfig.timeout_s`` and no ``cancel_event`` yet, and a caller wiring up
    an explicit Cancel control deserves a loud failure rather than a control that silently
    does nothing.
    """
    normalized = (kind or "pyodide").lower()

    if normalized in ("pyodide", "deno"):
        # dspy.RLM's default PythonInterpreter is the sandboxed WASM/subprocess
        # engine. We construct it ourselves (instead of returning None and letting
        # dspy build its own) only so we can pre-bind the JSON-literal aliases — the
        # isolation is identical. dspy.RLM merges its execution tools (SUBMIT,
        # llm_query, …) onto our instance, and ``RLMTask`` owns its teardown.
        return _build_sandboxed_interpreter(turn_timeout_s=turn_timeout_s, cancel_event=cancel_event)

    if cancel_event is not None:
        raise ValueError(
            f"cancel_event is only supported for interpreter kind 'pyodide'/'deno', got "
            f"{kind!r} — passing one to any other kind would silently do nothing, which is "
            f"worse than refusing loudly for a caller wiring up an explicit Cancel control."
        )

    if normalized == "mock":
        return _build_mock_interpreter()

    if normalized == "container":
        # The environment interpreter: the REPL runs inside an isolated container so model
        # code can spawn subprocesses natively. NOT routed through INSECURE_INTERPRETERS — it
        # is a STRONGER boundary than the WASM sandbox, the opposite of `local`. Lazily imported
        # (it is dspy-bearing) so this module and ``import rlm_harness`` stay dspy-free.
        return _build_container_interpreter(container)

    if normalized in INSECURE_INTERPRETERS:
        if not allow_insecure:
            raise SandboxSecurityError(
                "Interpreter 'local' executes model-written code directly on the "
                "host with no isolation. For a system that processes untrusted "
                "content this is remote code execution. Opt in explicitly via "
                "RLMConfig(allow_insecure_sandbox=True) or "
                "RLM_ALLOW_INSECURE_SANDBOX=1 if you understand the risk."
            )
        logger.warning(
            "INSECURE SANDBOX ACTIVE: 'local' interpreter runs model-written code "
            "on the host. Never enable this while processing untrusted input."
        )
        return _build_local_interpreter()

    # config.RLMConfig validates this earlier, but guard here too for direct callers.
    raise ValueError(f"Unknown interpreter kind: {kind!r}")


# JSON literals a model trained on JSON habitually emits inside the Python REPL —
# e.g. ``SUBMIT({"valid": true})`` — which raise ``NameError: name 'true' is not
# defined`` and make the model thrash on the identical call (a single run lost
# 14/25 REPL turns to exactly this). Pre-binding the three to their Python values
# makes the REPL tolerant of that one most-common JSON-in-Python slip.
_JSON_LITERAL_ALIASES = {"true": True, "false": False, "null": None}

# Built once, lazily — the class can only be defined after dspy is importable, and
# this module deliberately stays dspy-free at import time (see module docstring).
_sandboxed_interpreter_cls: type | None = None


def _build_sandboxed_interpreter(
    *,
    turn_timeout_s: float | None = None,
    cancel_event: threading.Event | None = None,
) -> Any:
    """dspy's default deno/pyodide sandbox, wrapped to pre-bind ``true``/``false``/
    ``null`` in the REPL namespace, and — when either knob is set — guarded by a
    watchdog that can kill a wedged sandbox turn from another thread.

    Construction spawns no subprocess (dspy's ``PythonInterpreter`` starts Deno
    lazily on first ``execute``), so this is cheap and an interpreter that is never
    run never starts Deno. The aliases are injected as REPL *variables*, which dspy
    serialises to ``true = True`` / ``false = False`` / ``null = None`` atop every
    executed cell; a real user variable of the same name still shadows them.

    The watchdog exists because ``PythonInterpreter.execute()`` blocks on a plain
    subprocess pipe read with NO timeout anywhere in dspy's own code — a wedged
    Deno subprocess, or a model-written REPL cell that spins forever, hangs the run
    with no recourse short of killing the whole process. `asyncio.Task.cancel()`
    cannot help: the blocking call has no `await` inside it, so the event loop never
    gets a chance to run cancellation machinery. This mirrors the identical
    timer-armed-before-blocking-read, kill-to-unblock idiom
    ``container_interpreter.py``'s ``ContainerInterpreter`` already uses for the
    ``container`` interpreter kind, ported here for ``pyodide``/``deno``.

    Two independent knobs, one mechanism:

    * ``turn_timeout_s`` — a per-``execute()`` safety-net deadline. Firing raises
      dspy's own RECOVERABLE interpreter error — whichever class that is on the
      installed dspy (``_dspy_compat.recoverable_interpreter_error``). Caught by
      ``RLM._execute_code`` and fed back to the model as an
      ``"[Error] ..."`` string — it gets to retry next turn against a
      freshly-respawned sandbox.
    * ``cancel_event`` — an externally-set ``threading.Event`` for a caller (e.g. a
      "Cancel" UI) that wants to stop an in-flight run NOW. Firing raises
      ``SandboxCancelled`` — NOT recoverable, not caught by
      ``RLM._execute_code``, propagates as a genuine run-ending failure.

    Both are ``None`` by default and cost nothing when unset: ``execute()``'s very
    first check calls ``super().execute(...)`` directly with no watcher thread
    started at all, so every existing caller (every downstream consumer today) is
    byte-identical to before this existed.
    """
    global _sandboxed_interpreter_cls
    if _sandboxed_interpreter_cls is None:
        from dspy.primitives.python_interpreter import PythonInterpreter

        from ._dspy_compat import recoverable_interpreter_error

        # The class dspy's RLM loop CATCHES, resolved for the installed dspy rather than
        # hardcoded — this IS the recoverable/terminal distinction the two raises below turn
        # on. dspy 3.3.0 moved it: `CodeInterpreterError` became TERMINAL and the recoverable
        # role passed to its `CodeExecutionError` subclass. Hardcoding the base class here
        # would silently turn the per-turn timeout — a SAFETY NET whose whole point is to hand
        # the model another turn — into a run-ending failure, with nothing going red to show it.
        _RecoverableExecError = recoverable_interpreter_error()

        class _JsonLiteralInterpreter(PythonInterpreter):
            _JSON_ALIASES = _JSON_LITERAL_ALIASES
            _turn_timeout_s: float | None = None
            _cancel_event: threading.Event | None = None

            def execute(self, code: str, variables: dict | None = None) -> Any:
                merged = {**self._JSON_ALIASES, **(variables or {})}
                # The ENTIRE disabled-by-default guarantee: when neither knob is
                # set, no watcher thread is ever created and this is byte-identical
                # to the pre-watchdog behavior. Keep this the FIRST thing this
                # method does — it was accidentally dropped once already during a
                # design revision, so it is deliberately isolated and commented.
                if self._turn_timeout_s is None and self._cancel_event is None:
                    return super().execute(code, merged)
                return self._execute_guarded(code, merged)

            def _execute_guarded(self, code: str, merged: dict) -> Any:
                import time

                fired: dict = {"reason": None}  # ONE field, ONE write per trigger
                stop = threading.Event()

                def _kill() -> None:
                    # A watchdog tick that fires in the razor-thin window between the
                    # call's own natural, successful completion and `stop.set()` below
                    # can still kill an already-idle, healthy process, discarding REPL
                    # state before the NEXT execute() call — the IDENTICAL race
                    # `container_interpreter.py`'s own `_recv_guarded`/`_fire` has, and
                    # accepted there without treating it as a defect. Inherited here
                    # rather than "fixed," for the same reason: closing it completely
                    # would need a lock this call site has no natural place to hold.
                    proc = getattr(self, "deno_process", None)
                    if proc is not None:
                        try:
                            proc.kill()
                        except Exception:
                            pass

                def _watch() -> None:
                    deadline = (
                        time.monotonic() + self._turn_timeout_s
                        if self._turn_timeout_s is not None else None
                    )
                    while not stop.is_set():
                        if self._cancel_event is not None and self._cancel_event.is_set():
                            fired["reason"] = "cancelled"
                            _kill()
                        elif deadline is not None and time.monotonic() >= deadline:
                            fired["reason"] = "timeout"
                            _kill()
                        stop.wait(timeout=0.1)

                watcher = threading.Thread(target=_watch, daemon=True)
                watcher.start()
                result = None
                error: BaseException | None = None
                try:
                    result = super().execute(code, merged)
                except Exception as e:
                    # Broad on purpose: dspy's execute() can raise CodeInterpreterError
                    # (the common case), a raw BrokenPipeError/OSError (a kill landing on
                    # a stdin.write() mid dspy's own respawn-and-retry), or a plain
                    # SyntaxError (JSON-RPC error code -32000, invalid Python) — any of
                    # these racing a fired watchdog must still map to a clean outcome
                    # below rather than escape untouched. BaseException is deliberately
                    # NOT used: SystemExit/KeyboardInterrupt must stay untouched. The
                    # `if error is not None: raise error` fallback below re-raises
                    # anything NOT fired completely transparently, so widening this catch
                    # has zero behavioral effect on a genuinely unrelated failure.
                    error = e
                finally:
                    stop.set()
                    watcher.join(timeout=1.0)

                # Checked HERE — unconditionally, after the try/except/finally above
                # has already run to completion, never only inside the except clause.
                # A fired watchdog always wins, even over an apparently SUCCESSFUL
                # result: dspy's own recovery (respawn -> re-mount -> re-register
                # tools -> retried write -> read) can complete inside one 0.1s poll
                # window, and a killed-and-respawned intermediate state cannot be
                # trusted to have produced a result untouched by the interruption —
                # the caller's intent to cancel/bound this turn must be honoured
                # regardless of whether the underlying work technically finished.
                reason = fired["reason"]
                if reason == "timeout":
                    raise _RecoverableExecError(
                        f"execution exceeded the {self._turn_timeout_s:g}s per-turn "
                        f"sandbox budget; the interpreter was killed and will restart "
                        f"with FRESH state on the next call"
                    ) from error
                if reason == "cancelled":
                    raise SandboxCancelled(
                        "sandbox execution was cancelled while in progress"
                    ) from error
                if error is not None:
                    raise error  # not fired: a genuine, unrelated failure — unchanged
                return result

        _sandboxed_interpreter_cls = _JsonLiteralInterpreter

    inst = _sandboxed_interpreter_cls()
    inst._turn_timeout_s = turn_timeout_s
    inst._cancel_event = cancel_event
    return inst


def _build_container_interpreter(container: Any) -> Any:
    """Construct the container-backed environment interpreter. Lazily imported because
    ``container_interpreter`` is dspy-bearing; ``sandbox.py`` itself stays dspy-free."""
    from .config import ContainerConfig
    from .container_interpreter import ContainerInterpreter

    return ContainerInterpreter(container or ContainerConfig())


def _build_mock_interpreter() -> Any:
    """A do-nothing interpreter usable in tests without a real sandbox.

    Implements the FULL ``CodeInterpreter`` surface — ``tools`` / ``start`` / ``execute`` /
    ``shutdown`` — and must keep doing so. From dspy 3.3.0 that protocol is
    ``@runtime_checkable`` and ``RLM._interpreter_context`` ``isinstance``-checks a
    caller-supplied interpreter on EVERY forward pass, so a missing member is a run-time
    ``TypeError``, not something a type checker or a construction test catches. ``tools`` and
    ``start`` were absent until 1.2.1 and `interpreter="mock"` could not run a task at all
    (it worked before only because dspy 3.2.x took the interpreter as a constructor kwarg and
    validated nothing). Same requirement, same reason, as
    ``rlm_harness.testing.ScriptedInterpreter``.

    Still dspy-free: the protocol is structural, so satisfying it needs no import.
    """

    class _MockInterpreter:
        def __init__(self) -> None:
            # RLM mutates this dict in place to inject the run's execution tools.
            self.tools: dict[str, Any] = {}

        def start(self) -> None:
            """No-op: there is no process to spin up."""

        def execute(self, code: str, variables: dict | None = None) -> str:
            return ""

        def shutdown(self) -> None:
            return None

    return _MockInterpreter()


def _build_local_interpreter() -> Any:  # pragma: no cover - never run in tests/CI
    """Construct dspy's local Python interpreter. Insecure by definition."""
    from dspy.primitives.python_interpreter import PythonInterpreter

    return PythonInterpreter()
