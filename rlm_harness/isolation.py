"""``run_in_subprocess`` — a safe, isolated-subprocess primitive.

A small PRIMITIVE only: "safely run one picklable callable in an isolated OS process, get its
result or a clear error back, bounded by a timeout." Queue/scheduling logic (how a web server
actually schedules many of these — Celery, RQ, a plain thread/process pool) is explicitly the
CONSUMER's own concern, not shipped here — matching this kit's standing base/wrap posture
(``make_command_tool`` ships no executor, ``make_git_clone_tool`` ships no cloner) applied to
"run a whole task," not a REPL tool.

**Why this is a genuinely different gap than three things in this kit that already sound
similar**: ``interpreter="container"`` isolates the RLM's own REPL SANDBOX in a Docker
container, but the ROOT process still runs the RLM's own orchestration (LM calls, retries, tool
dispatch) directly — a hang/crash there is not covered. ``rlm_harness.tools.run_isolated``
bridges an async coroutine into a sync call site on a dedicated THREAD — same process, no OS-level
isolation, solves an event-loop-nesting problem, not a fault-isolation one. ``cancel_event``
stops an IN-FLIGHT run the calling code already owns and is watching — it doesn't hand a whole
task off to a separate process in the first place. The actual gap this closes: a web-facing
consumer whose request handler wants to run ONE task without that run's own crash, hang, or
resource usage taking down the request-handling process itself needs to run it in a SEPARATE OS
PROCESS.

**Fully generic — no `RLMTask` import, no dependency on this kit's own task machinery.** A
consumer uses it as ``run_in_subprocess(functools.partial(run_my_task, **inputs))`` where
``run_my_task`` is a plain, MODULE-LEVEL function (e.g. ``def run_my_task(**inputs): return
MyTask().run(**inputs)``) that itself constructs and runs the task — the kit's own task
machinery is entirely the consumer's business inside ``factory``, matching ``run_command``'s own
"the kit wraps, the consumer's callable does the real work" split.

**Why this lives at the top level, not under ``rlm_harness.tools``**: that package's own module
docstring scopes it as "Reusable tools that RLM tasks can expose to the model inside the REPL" —
nothing here is ever placed in a ``tools=[...]`` list or invoked by the model; "run a whole
separate task in an isolated process" is a host-level orchestration decision a consumer's own
request-handling code makes.
"""

from __future__ import annotations

import multiprocessing
import pickle
import queue as _queue_module
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

_RESULT_QUEUE_TIMEOUT_S = 5.0


def _child_entrypoint(
    factory: Callable[[], object],
    result_queue,
    max_memory_mb: int | None,
    cpu_time_limit_s: float | None,
) -> None:
    try:
        # Setting the resource caps runs INSIDE the same try/except as factory() itself: the
        # setrlimit() call can itself fail (e.g. a platform/OS rejecting a requested limit that's
        # already below the interpreter's own baseline virtual-memory footprint -- observed in
        # practice on macOS as `ValueError: current limit exceeds maximum limit`) -- without this,
        # such a failure would crash the child BEFORE anything reaches the relay queue, and the
        # parent would see only the generic "child exited without delivering a result" message,
        # obscuring the real, fixable cause (an unsatisfiable requested limit, not an OOM kill).
        if max_memory_mb is not None or cpu_time_limit_s is not None:
            try:
                import resource
            except ImportError:
                pass  # Windows: silently a no-op, documented as such -- never a crash on import.
            else:
                if max_memory_mb is not None:
                    limit = max_memory_mb * 1024 * 1024
                    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
                if cpu_time_limit_s is not None:
                    limit = int(cpu_time_limit_s)
                    resource.setrlimit(resource.RLIMIT_CPU, (limit, limit))

        result = factory()
    except BaseException as exc:  # never swallowed -- always relayed, same posture run_isolated
        # takes for its own thread-bridged call.
        payload = ("error", exc)
        try:
            pickle.dumps(payload)
        except BaseException:
            # The real exception isn't picklable (e.g. it holds an open file handle or a lock as
            # an attribute) -- proven SYNCHRONOUSLY, right here, by the same operation that will
            # actually pickle it, rather than discovering this asynchronously in Queue's feeder
            # thread (where a pickling failure is logged and silently DROPPED, never raised back
            # to the put() caller -- the parent would then hang on get() forever, having no way
            # to know the child "succeeded" at nothing). The fallback is built from plain strings,
            # always picklable.
            payload = ("error", RuntimeError(f"{type(exc).__name__}: {exc}"))
        result_queue.put(payload)
        return

    payload = ("ok", result)
    try:
        pickle.dumps(payload)
    except BaseException:
        payload = (
            "error",
            RuntimeError(
                f"factory() succeeded but its result of type {type(result).__name__!r} is not "
                "picklable, so it cannot be relayed back to the parent process"
            ),
        )
    result_queue.put(payload)


def run_in_subprocess(
    factory: Callable[[], T],
    *,
    timeout_s: float | None = None,
    grace_period_s: float = 5.0,
    max_memory_mb: int | None = None,
    cpu_time_limit_s: float | None = None,
) -> T:
    """Run ``factory()`` to completion in a fresh, isolated OS process and return its result —
    or raise the same exception ``factory()`` raised (or a clear synthesized one), never swallow.

    **``factory`` MUST be picklable** — a real, easy-to-get-wrong gotcha. A local closure or a
    ``lambda`` is NOT picklable across the ``"spawn"`` boundary this uses; ``functools.partial(
    module_level_function, **kwargs)`` (or a bare module-level function with no arguments) IS —
    but only if every value bound into ``args``/``kwargs`` is ALSO picklable, not merely the
    function reference itself: a live socket, open file handle, DB connection, or lock bound as
    one of the ``partial``'s own arguments hits the same class of pickling failure the
    "avoid closures" advice was supposed to prevent.

    Uses ``multiprocessing.get_context("spawn")`` — never ``"fork"``. On POSIX, ``spawn`` is
    itself still fork()+exec() internally, but exec runs BEFORE any user code executes in the
    freshly-forked child, which is what actually avoids the corruption class a bare ``fork``
    risks (a forked child inheriting a lock held by a parent thread that doesn't exist in the
    child, half-open sockets, etc.) — the parent calling this is very plausibly a web server
    already running an event loop / thread pool / open file descriptors / a live LM client.
    ``spawn`` also requires ``factory``'s entire import chain to be safely re-importable in the
    fresh interpreter (any module-level side effect re-runs in the child); if ``factory`` is
    defined in a script run as ``__main__``, guard it with ``if __name__ == "__main__":`` —
    ``multiprocessing``'s own spawn bootstrap has special, fragile handling for that case.

    ``timeout_s`` (default ``None`` = no limit): on expiry, ``process.terminate()`` (SIGTERM),
    then a ``grace_period_s`` (default ``5.0``) wait, then ``process.kill()`` (SIGKILL) if still
    alive, followed by a final reap so no zombie is left — raises ``TimeoutError``. SIGTERM does
    NOT reliably let the child's ``finally``/``atexit`` code run — that's only true if the child
    itself installs a ``signal.signal(SIGTERM, ...)`` handler; with none (the default), the OS's
    default disposition terminates it immediately.

    ``max_memory_mb``/``cpu_time_limit_s`` (default ``None`` = no cap): OPT-IN, POSIX-only,
    best-effort resource caps applied inside the child via ``resource.setrlimit`` before
    ``factory()`` runs — silent no-ops on a platform without ``resource`` (Windows), never a
    crash. ``cpu_time_limit_s`` (``RLIMIT_CPU``) works as expected on every POSIX platform tested,
    including macOS. ``max_memory_mb`` bounds VIRTUAL address space (``RLIMIT_AS``), NOT resident/
    physical memory — a Python process's baseline VSZ plus ordinary overcommitted allocations can
    overshoot a physical-memory intent by a real margin, and exceeding it (where the OS honors the
    limit at all) does not trigger an OS-level kill — the next allocation simply fails, typically
    surfacing as an ordinary ``MemoryError`` raised INSIDE the child's own code, relayed through
    the same exception path as any other error. **A real, empirically-confirmed platform gap,
    disclosed rather than silently discovered**: on macOS, the kernel refuses to LOWER
    ``RLIMIT_AS`` from its default of unlimited at all — every ``resource.setrlimit(RLIMIT_AS,
    ...)`` call, regardless of the requested value, raises ``ValueError: current limit exceeds
    maximum limit`` outright (confirmed directly: tried 50, 200, 500, and 1000 MB on macOS
    14.2.1/arm64, all failed identically; `RLIMIT_CPU` on the same platform succeeded normally).
    Setting the resource caps happens INSIDE the same try/except as ``factory()`` itself, so this
    failure is relayed to the caller as a clear ``ValueError`` (via the same test-pickle-then-
    relay path as any other error) — a caller passing ``max_memory_mb`` on macOS should expect
    this raised outright, NOT a silently-unenforced cap and NOT actual memory enforcement.
    Effectively, `max_memory_mb` is Linux-only in practice today; `cpu_time_limit_s` is not.
    """
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(
        target=_child_entrypoint,
        args=(factory, result_queue, max_memory_mb, cpu_time_limit_s),
    )
    process.start()
    process.join(timeout_s)

    if process.is_alive():
        process.terminate()
        process.join(grace_period_s)
        if process.is_alive():
            process.kill()
            process.join()
        raise TimeoutError(
            f"run_in_subprocess: factory() did not finish within {timeout_s}s "
            f"(terminated, then killed after a {grace_period_s}s grace period)"
        )

    try:
        status, payload = result_queue.get(timeout=_RESULT_QUEUE_TIMEOUT_S)
    except _queue_module.Empty:
        # The process already exited (we're past the join() above) but left nothing on the
        # queue -- an out-of-band kill (e.g. the host OOM-killing it) that bypassed this
        # function's own signal-based escalation entirely. Never hang; synthesize a clear error.
        raise RuntimeError(
            "run_in_subprocess: child process exited without delivering a result -- likely "
            "killed by an external signal (OOM) or crashed before the result could be relayed"
        ) from None

    if status == "error":
        raise payload
    return payload
