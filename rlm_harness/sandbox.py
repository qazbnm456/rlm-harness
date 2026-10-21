"""Sandbox / code-interpreter selection: the security boundary of the scaffold.

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
  ``local``: handled *before* the insecure-interpreter check below, never routed
  through it. Needs the ``docker`` CLI (imported lazily; this module stays dspy-free).
- ``local`` → executes model-written code directly on the host. This is
  effectively arbitrary code execution and is refused unless the caller has
  *explicitly* opted in. The opt-in cannot be reached by accident.

``dspy`` is imported lazily inside the branches that need it so this module,
and the security guard in particular. Stays importable and testable without a
full dspy install.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from typing import Any

from .trace import current_recorder  # dspy-free, like this module

logger = logging.getLogger(__name__)

# Interpreters that run model-written code on the host with no isolation.
INSECURE_INTERPRETERS = frozenset({"local"})


class SandboxSecurityError(RuntimeError):
    """Raised when an insecure interpreter is requested without explicit opt-in."""


#: Raised when `cancel_event` fires during a sandbox `execute()` call, deliberately NOT any
#: dspy interpreter-error subclass. dspy's ``RLM._execute_code`` catches only its own
#: recoverable interpreter error plus ``SyntaxError`` (``CodeExecutionError``, see
#: ``_dspy_compat``), so this propagates
#: all the way up through ``RLMTask.arun()`` (and through ``run_with_retry``'s ``non_retryable``
#: allowlist, untouched) as a genuine run-ending failure on EVERY supported dspy. Standing
#: outside dspy's hierarchy entirely is what makes that true across the rename. A plain
#: ``RuntimeError`` subclass with zero dspy dependency, so it is defined here at module top
#: without touching this module's lazy-dspy-import discipline, the same posture
#: ``SandboxSecurityError`` above already has.
class SandboxCancelled(RuntimeError):
    """A caller explicitly cancelled a sandbox execution in progress."""


class _CallerOwnedInterpreter:
    """A view of ``interp`` whose ``shutdown()`` is a no-op; every other attribute forwards.

    dspy >= 3.4.0 CREATES and SHUTS DOWN one interpreter per forward pass and deleted the
    caller-owned positional seam that used to keep a supplied interpreter the caller's. This view
    is what keeps ``RLMTask(interpreter=...)`` working the way its contract says: dspy shuts down
    the view, the real object survives every pass, and ``RLMTask._teardown_interpreter`` stays its
    single shutdown. Without it a caller's ``ScriptedInterpreter`` is destroyed after the first
    pass and a retry runs against a dead double.

    **The protocol members are declared STATICALLY, and that is not style.** dspy validates with
    ``isinstance(x, CodeInterpreter)`` on a ``@runtime_checkable`` Protocol, and from CPython 3.12
    (gh-102433) that resolves members through ``inspect.getattr_static``, which never consults
    ``__getattr__``. A purely dynamic proxy therefore passes on the 3.11 floor and FAILS on 3.12
    and 3.13, which is the matrix cell most local runs use. ``tests/test_sandbox.py`` pins this
    with a ``getattr_static`` sweep rather than a bare ``isinstance``, because an ``isinstance``
    assertion would itself be green on 3.11 and could not stop the regression returning.

    ``__setattr__`` forwarding is load-bearing too: ``RLM._inject_execution_context`` writes
    ``output_fields`` and ``_tools_registered`` onto whatever the factory returned, and a
    ``__getattr__``-only proxy would absorb both onto the wrapper, leaving the real interpreter
    with a previous pass's tools.
    """

    __slots__ = ("_interp",)

    def __init__(self, interp: Any) -> None:
        object.__setattr__(self, "_interp", interp)

    # --- the CodeInterpreter protocol, declared so `getattr_static` can see it ---------------
    @property
    def tools(self) -> Any:
        return object.__getattribute__(self, "_interp").tools

    def start(self) -> Any:
        return object.__getattribute__(self, "_interp").start()

    def execute(self, code: str, variables: dict | None = None) -> Any:
        # `variables=` BY KEYWORD, never positionally. dspy itself calls
        # `repl.execute(code, variables=dict(input_args))`, so it already requires the parameter to
        # be named that, and forwarding positionally is strictly narrower than what it accepts: a
        # caller-supplied interpreter declaring `def execute(self, code, *, variables=None)` works
        # when handed to dspy directly and would raise `TypeError` only through this view.
        return object.__getattribute__(self, "_interp").execute(code, variables=variables)

    def shutdown(self) -> None:
        """Deliberately nothing: the CALLER owns this interpreter's lifetime."""

    # --- everything else forwards -----------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_interp"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_interp"), name, value)


def caller_owned(interp: Any) -> Any:
    """Wrap a CALLER-SUPPLIED interpreter so dspy cannot shut it down. See the class above."""
    return _CallerOwnedInterpreter(interp)


class _NoSubAgentInterpreter:
    """A view of ``interp`` that refuses dspy's sandbox-facade install, and forwards everything else.

    **What it turns off.** From dspy 3.4.1 ``RLM`` installs a ``dspy`` shim inside the sandbox, so
    model code may build ``Predict``/``ChainOfThought``/``ReAct``/``RLM``/… sub-agents that run on
    the HOST, call the run's ``sub_lm``, and resolve the task's own tools BY NAME. That is a new
    capability crossing the boundary this module exists to defend, and for a consumer whose threat
    model is "the sandbox is the boundary" it has to be refusable. dspy has no flag for it, so the
    refusal is expressed where the kit does own a seam: the ``execute`` that installs the shim.

    **Raising ``SyntaxError`` is dspy's OWN degradation path, not a trick.**
    ``RLM._setup_facade`` catches ``(CodeExecutionError, SyntaxError)``, drops its two host tools
    and logs "RLM sub-agents are unavailable on %s", then runs the whole forward pass normally with
    the sub-agent paragraph left out of the prompt. So the model is never told about a capability
    it does not have, which is the half a bare refusal would get wrong.

    **The protocol members are declared STATICALLY**, for exactly the reason
    :class:`_CallerOwnedInterpreter` documents: dspy's ``isinstance`` against a
    ``@runtime_checkable`` Protocol resolves members through ``inspect.getattr_static`` from
    CPython 3.12, which never consults ``__getattr__``. ``shutdown`` FORWARDS here, unlike that
    class: this wraps an interpreter the kit built and handed to dspy, and dspy owns its shutdown.
    """

    __slots__ = ("_interp",)

    def __init__(self, interp: Any) -> None:
        object.__setattr__(self, "_interp", interp)

    @property
    def tools(self) -> Any:
        return object.__getattribute__(self, "_interp").tools

    def start(self) -> Any:
        return object.__getattribute__(self, "_interp").start()

    def execute(self, code: str, variables: dict | None = None) -> Any:
        from ._dspy_compat import is_facade_install

        if is_facade_install(variables):
            raise SyntaxError(
                "sandbox sub-agents are disabled for this run "
                "(RLMConfig(sandbox_sub_agents=False) / RLM_SANDBOX_SUB_AGENTS=0)"
            )
        return object.__getattribute__(self, "_interp").execute(code, variables=variables)

    def shutdown(self) -> None:
        return object.__getattribute__(self, "_interp").shutdown()

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_interp"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_interp"), name, value)


def build_interpreter(
    kind: str,
    *,
    allow_insecure: bool = False,
    container: Any = None,
    turn_timeout_s: float | None = None,
    cancel_event: threading.Event | None = None,
    sandbox_sub_agents: bool = True,
    read_paths: tuple[str, ...] = (),
) -> Any | None:
    """Return a dspy ``CodeInterpreter`` for ``kind``, or ``None`` for the default.

    ``None`` means "let dspy.RLM construct its own default sandboxed interpreter",
    which is the secure path. A non-None return is produced for ``mock``, ``container``
    (needs the ``container`` options + the ``docker`` CLI), and (when explicitly allowed)
    ``local``.

    ``turn_timeout_s``/``cancel_event`` are watchdog knobs for the ``pyodide``/``deno``
    branch ONLY: see ``_build_sandboxed_interpreter``. ``cancel_event`` alongside any
    OTHER kind is a hard refusal (see below), not a silent no-op: ``container`` already has
    its own ``ContainerConfig.timeout_s`` and no ``cancel_event`` yet, and a caller wiring up
    an explicit Cancel control deserves a loud failure rather than a control that silently
    does nothing.

    ``sandbox_sub_agents=False`` refuses dspy's in-sandbox ``dspy`` facade for EVERY kind, by
    wrapping the result in :class:`_NoSubAgentInterpreter`. The default wraps NOTHING, so the
    path every run takes today is byte-for-byte what it was: the view exists only for the
    consumer who turns the capability off.
    """
    normalized = (kind or "pyodide").lower()
    interp = _build_interpreter_for_kind(
        normalized,
        kind=kind,
        allow_insecure=allow_insecure,
        container=container,
        turn_timeout_s=turn_timeout_s,
        cancel_event=cancel_event,
        read_paths=read_paths,
    )
    if interp is not None and not sandbox_sub_agents:
        return _NoSubAgentInterpreter(interp)
    return interp


def _build_interpreter_for_kind(
    normalized: str,
    *,
    kind: str,
    allow_insecure: bool,
    container: Any,
    turn_timeout_s: float | None,
    cancel_event: threading.Event | None,
    read_paths: tuple[str, ...] = (),
) -> Any | None:
    """:func:`build_interpreter`'s body, split out so the sub-agent view wraps every kind once
    rather than at each ``return``."""

    if normalized in ("pyodide", "deno"):
        # dspy.RLM's default PythonInterpreter is the sandboxed WASM/subprocess
        # engine. We construct it ourselves (instead of returning None and letting
        # dspy build its own) only so we can pre-bind the JSON-literal aliases: the
        # isolation is identical. dspy.RLM merges its execution tools (SUBMIT,
        # llm_query, …) onto our instance, and ``RLMTask`` owns its teardown.
        return _build_sandboxed_interpreter(
            turn_timeout_s=turn_timeout_s, cancel_event=cancel_event, read_paths=read_paths
        )

    if read_paths:
        # The same reasoning as `cancel_event` below: only the pyodide/deno branch can honour
        # these, and a grant that silently does nothing would leave a caller believing the
        # sandbox can read a file it cannot. `container` runs with its own mounts and `mock`
        # has no filesystem at all, so neither is a place to quietly accept one.
        raise ValueError(
            f"read_paths is only supported for interpreter kind 'pyodide'/'deno', got {kind!r}: "
            f"accepting one for another kind would grant nothing while reporting success."
        )

    if cancel_event is not None:
        raise ValueError(
            f"cancel_event is only supported for interpreter kind 'pyodide'/'deno', got "
            f"{kind!r}: passing one to any other kind would silently do nothing, which is "
            f"worse than refusing loudly for a caller wiring up an explicit Cancel control."
        )

    if normalized == "mock":
        return _build_mock_interpreter()

    if normalized == "container":
        # The environment interpreter: the REPL runs inside an isolated container so model
        # code can spawn subprocesses natively. NOT routed through INSECURE_INTERPRETERS: it
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


# JSON literals a model trained on JSON habitually emits inside the Python REPL:
# e.g. ``SUBMIT({"valid": true})``, which raise ``NameError: name 'true' is not
# defined`` and make the model thrash on the identical call (a single run lost
# 14/25 REPL turns to exactly this). Pre-binding the three to their Python values
# makes the REPL tolerant of that one most-common JSON-in-Python slip.
_JSON_LITERAL_ALIASES = {"true": True, "false": False, "null": None}

# Built once, lazily. The class can only be defined after dspy is importable, and
# this module deliberately stays dspy-free at import time (see module docstring).
_sandboxed_interpreter_cls: type | None = None


def _build_sandboxed_interpreter(
    *,
    turn_timeout_s: float | None = None,
    cancel_event: threading.Event | None = None,
    read_paths: tuple[str, ...] = (),
) -> Any:
    """dspy's default deno/pyodide sandbox, wrapped to pre-bind ``true``/``false``/
    ``null`` in the REPL namespace, and, when either knob is set, guarded by a
    watchdog that can kill a wedged sandbox turn from another thread.

    Construction spawns no subprocess (dspy's ``PythonInterpreter`` starts Deno
    lazily on first ``execute``), so this is cheap and an interpreter that is never
    run never starts Deno. The aliases are injected as REPL *variables*, which dspy
    serialises to ``true = True`` / ``false = False`` / ``null = None`` atop every
    executed cell; a real user variable of the same name still shadows them.

    The watchdog exists because ``PythonInterpreter.execute()`` blocks on a plain
    subprocess pipe read with NO timeout anywhere in dspy's own code: a wedged
    Deno subprocess, or a model-written REPL cell that spins forever, hangs the run
    with no recourse short of killing the whole process. `asyncio.Task.cancel()`
    cannot help: the blocking call has no `await` inside it, so the event loop never
    gets a chance to run cancellation machinery. This mirrors the identical
    timer-armed-before-blocking-read, kill-to-unblock idiom
    ``container_interpreter.py``'s ``ContainerInterpreter`` already uses for the
    ``container`` interpreter kind, ported here for ``pyodide``/``deno``.

    Two independent knobs, one mechanism:

    * ``turn_timeout_s``: a per-``execute()`` safety-net deadline. Firing raises
      dspy's own RECOVERABLE interpreter error. Whichever class that is on the
      installed dspy (``_dspy_compat.recoverable_interpreter_error``). Caught by
      ``RLM._execute_code`` and fed back to the model as an
      ``"[Error] ..."`` string: it gets to retry next turn against a
      freshly-respawned sandbox.
    * ``cancel_event``: an externally-set ``threading.Event`` for a caller (e.g. a
      "Cancel" UI) that wants to stop an in-flight run NOW. Firing raises
      ``SandboxCancelled``: NOT recoverable, not caught by
      ``RLM._execute_code``, propagates as a genuine run-ending failure.

    Both are ``None`` by default and cost nothing when unset: ``execute()``'s very first check
    reaches ``super().execute(...)`` by the same direct path it took before either knob existed,
    with NO WATCHER THREAD started at all, which is the guarantee that matters for every
    downstream consumer today. (Since 1.6.0 both branches pass through a ``perf_counter()`` timing
    shell that stages the turn's duration for the trace, so the path is no longer byte-identical;
    it creates no thread, and timing wraps the BRANCHES rather than the method precisely so the
    disabled-by-default check stays first.)
    """
    global _sandboxed_interpreter_cls
    if _sandboxed_interpreter_cls is None:
        from dspy.primitives.python_interpreter import PythonInterpreter

        from ._dspy_compat import (
            is_facade_install,
            recoverable_interpreter_error,
            reset_interpreter_session,
            runner_path_with_safe_error_args,
        )

        # The class dspy's RLM loop CATCHES, resolved for the installed dspy rather than
        # hardcoded. This IS the recoverable/terminal distinction the two raises below turn
        # on. dspy 3.3.0 moved it: `CodeInterpreterError` became TERMINAL and the recoverable
        # role passed to its `CodeExecutionError` subclass. Hardcoding the base class here
        # would silently turn the per-turn timeout: a SAFETY NET whose whole point is to hand
        # the model another turn: into a run-ending failure, with nothing going red to show it.
        _RecoverableExecError = recoverable_interpreter_error()

        # Timing is applied to BOTH branches of `execute` below rather than wrapped around the
        # method, so the disabled-by-default guard stays literally the first statement it runs.
        # Observability must never break a run: a recorder that is absent, is a caller's own
        # duck-typed object, or raises, all degrade to "no duration recorded".
        def _timed(fn, code, merged):
            t0 = time.perf_counter()
            try:
                return fn(code, merged)
            finally:
                rec = current_recorder()
                note = getattr(rec, "note_exec_duration", None)
                if callable(note):
                    with contextlib.suppress(Exception):
                        note(time.perf_counter() - t0, code)

        class _JsonLiteralInterpreter(PythonInterpreter):
            _JSON_ALIASES = _JSON_LITERAL_ALIASES
            _turn_timeout_s: float | None = None
            _cancel_event: threading.Event | None = None
            #: The ``(code, variables)`` of dspy's sandbox-facade install, remembered so a
            #: restart can replay it. See ``_replay_facade_install``.
            _facade_install: tuple[str, dict] | None = None
            _facade_stale = False

            def _get_runner_path(self) -> str:
                # dspy's own runner, with the one helper that lets a model's exception kill the
                # sandbox replaced. Applies whether or not either watchdog knob is set: the
                # failure is in how ANY failed cell is reported. See the shim's docstring.
                return runner_path_with_safe_error_args(super()._get_runner_path())

            def execute(self, code: str, variables: dict | None = None) -> Any:
                if is_facade_install(variables):
                    # dspy installs its sandbox `dspy` shim with ONE execute before the turn
                    # loop (3.4.1+). Remember it: a turn-timeout kill resets the session, the
                    # next call spawns a FRESH Deno, and dspy never installs the shim again,
                    # while the action prompt still tells the model it may `import dspy`.
                    # Without the replay below the model follows that instruction into a
                    # NameError and spends its remaining turns on a capability the runtime
                    # silently lost.
                    self._facade_install = (code, dict(variables or {}))
                elif self._facade_stale:
                    self._replay_facade_install()
                merged = {**self._JSON_ALIASES, **(variables or {})}
                # The ENTIRE disabled-by-default guarantee: when neither knob is
                # set, NO WATCHER THREAD is ever created. Keep this the FIRST thing
                # this method does. It was accidentally dropped once already during
                # a design revision, so it is deliberately isolated and commented.
                # (Since 1.6.0 both branches go through `_timed`, which stages the
                # turn's duration for the trace. That is a `perf_counter()` pair and
                # one attribute lookup: it creates no thread, so the guarantee this
                # guard exists for is untouched. Timing wraps the BRANCHES rather
                # than the method precisely so this check stays first.)
                if self._turn_timeout_s is None and self._cancel_event is None:
                    return _timed(super().execute, code, merged)
                return _timed(self._execute_guarded, code, merged)

            def _replay_facade_install(self) -> None:
                """Re-install dspy's sandbox facade into the session that replaced a killed one.

                Replays the exact ``(code, variables)`` dspy itself ran, so nothing here has to
                know the shim's source or its host token: both are private to dspy, and a kit
                that imported them would be asserting a private seam it cannot see change.

                Best-effort by design, and the degradation is dspy's own: ``RLM._setup_facade``
                treats a sandbox that cannot host the shim as "no sub-agents this call" and logs
                it, so a failed replay must leave the run alive rather than turn a recovered turn
                timeout into a dead run. The flag is cleared FIRST, so a replay that raises is not
                retried on every later cell.
                """
                self._facade_stale = False
                install = self._facade_install
                if install is None:
                    return
                code, variables = install
                try:
                    self.execute(code, variables)
                except Exception as e:
                    logger.warning(
                        "could not re-install dspy's sandbox facade after a sandbox restart, so "
                        "`import dspy` in model code will fail for the rest of this run: %s", e
                    )

            def _execute_guarded(self, code: str, merged: dict) -> Any:
                fired: dict = {"reason": None}  # ONE field, ONE write per trigger
                stop = threading.Event()

                def _kill() -> None:
                    # A watchdog tick that fires in the razor-thin window between the
                    # call's own natural, successful completion and `stop.set()` below
                    # can still kill an already-idle, healthy process, discarding REPL
                    # state before the NEXT execute() call: the IDENTICAL race
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
                    # SyntaxError (JSON-RPC error code -32000, invalid Python): any of
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

                # Checked HERE: unconditionally, after the try/except/finally above
                # has already run to completion, never only inside the except clause.
                # A fired watchdog always wins, even over an apparently SUCCESSFUL
                # result: dspy's own recovery (respawn -> re-mount -> re-register
                # tools -> retried write -> read) can complete inside one 0.1s poll
                # window, and a killed-and-respawned intermediate state cannot be
                # trusted to have produced a result untouched by the interruption:
                # the caller's intent to cancel/bound this turn must be honoured
                # regardless of whether the underlying work technically finished.
                reason = fired["reason"]
                if reason == "timeout":
                    # The kill made dspy mark this session ENDED, so without a reset the
                    # "will restart" below was false: the model's next cell got
                    # `PythonInterpreter session has ended`, which dspy treats as terminal, and
                    # a turn timeout ended the whole run one turn late. Measured on a
                    # deployment, every turn-timeout death took that path.
                    # The watcher was told to stop and joined with a timeout; make sure it has
                    # really gone before the reset, or a late tick could kill the NEXT Deno.
                    if watcher.is_alive():
                        watcher.join()
                    reset_interpreter_session(self)
                    # The reset makes the next call spawn a fresh Deno, which re-mounts files and
                    # re-registers tools but does NOT bring back the sandbox-side `dspy` shim:
                    # that was an `execute`, not a tool registration. Mark it for replay.
                    self._facade_stale = True
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
                    raise error  # not fired: a genuine, unrelated failure, unchanged
                return result

        _sandboxed_interpreter_cls = _JsonLiteralInterpreter

    # `enable_read_paths` is a CONSTRUCTOR argument, not an attribute set afterwards like the two
    # watchdog knobs: dspy reads it when it builds the Deno `--allow-read` list, and the class
    # itself is cached and shared, so a grant stored on the class would outlive the run that
    # asked for it and reach the next one. Empty by default, which is every run that does not
    # stage a file: the list dspy receives is then exactly what it was before this existed.
    inst = _sandboxed_interpreter_cls(enable_read_paths=list(read_paths) or None)
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

    Implements the FULL ``CodeInterpreter`` surface: ``tools`` / ``start`` / ``execute`` /
    ``shutdown``, and must keep doing so. From dspy 3.3.0 that protocol is
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
        #: Honest about being a stub. Only reaches a prompt if a caller pairs interpreter="mock"
        #: with a real LM, which is not what the mock is for, but silently claiming Pyodide's
        #: capabilities would be worse than saying so.
        execution_instructions = (
            "This REPL is a stub: code is NOT executed and every execution returns an empty "
            "result. Do not rely on computing anything here."
        )

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
