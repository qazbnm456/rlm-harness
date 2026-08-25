"""run_in_subprocess -- a safe, isolated-subprocess primitive. All offline, dspy-free.

Every `factory` used here MUST be a plain, MODULE-LEVEL function (never a local closure/lambda
inside a test) -- `multiprocessing`'s "spawn" start method needs to re-import it in the fresh
child interpreter, the exact constraint the function under test itself documents.
"""
import functools
import os
import signal
import time

import pytest

from rlm_harness.isolation import run_in_subprocess

# ---- module-level factories (picklable -- see the module docstring above) --------------------

def _add(a, b):
    return a + b


def _boom():
    raise ValueError("kaboom")


class _UnpicklableError(Exception):
    def __init__(self):
        super().__init__("weird")
        # A real reason pickling fails: an open file handle held as an attribute.
        self.handle = open(__file__)  # noqa: SIM115 -- deliberately left open, never used


def _boom_unpicklable():
    raise _UnpicklableError()


def _exit_without_relaying():
    # Simulates a child that dies WITHOUT ever going through the relay code -- e.g. os._exit,
    # a segfault, or an external OOM kill -- proving the parent's own bounded queue.get() fires
    # rather than hanging forever waiting for a payload that will never arrive.
    os._exit(1)


def _sleep(seconds):
    time.sleep(seconds)


def _sleep_ignoring_sigterm(seconds):
    # The handler MUST be installed INSIDE the factory's own body (the code that actually runs
    # as the child's entry point) -- installing it at module import time would also arm it in
    # the PARENT process and in any other spawned child importing this same module.
    signal.signal(signal.SIGTERM, lambda *a: None)
    time.sleep(seconds)


def _allocate_mb(mb):
    data = bytearray(mb * 1024 * 1024)
    return len(data)


def _busy_loop_cpu_bound():
    x = 0
    while True:
        x += 1


# ---- tests --------------------------------------------------------------------------------

def test_simple_factory_relays_the_result():
    assert run_in_subprocess(functools.partial(_add, 2, 3)) == 5


def test_picklable_exception_relays_unchanged():
    with pytest.raises(ValueError, match="kaboom"):
        run_in_subprocess(_boom)


def test_unpicklable_exception_falls_back_to_a_runtime_error():
    # Confirms the CHILD's synchronous test-pickle step catches this BEFORE ever calling
    # queue.put() -- not merely "doesn't hang forever" as a weaker, after-the-fact observation.
    with pytest.raises(RuntimeError, match="_UnpicklableError"):
        run_in_subprocess(_boom_unpicklable)


def test_child_exiting_without_relaying_raises_a_clear_error_not_a_hang():
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="without delivering a result"):
        run_in_subprocess(_exit_without_relaying)
    elapsed = time.monotonic() - started
    # Bounded by the internal queue-read timeout, not an indefinite hang.
    assert elapsed < 30.0


def test_timeout_raises_and_the_process_is_actually_gone(monkeypatch):
    processes = []
    _patch_process_capture(monkeypatch, processes)

    started = time.monotonic()
    with pytest.raises(TimeoutError):
        run_in_subprocess(functools.partial(_sleep, 60), timeout_s=1.0, grace_period_s=1.0)
    elapsed = time.monotonic() - started
    assert elapsed < 10.0  # terminated promptly, not left running the full 60s

    assert len(processes) == 1
    proc = processes[0]
    assert proc.exitcode is not None  # reaped -- join() was called after the kill, no zombie
    assert not proc.is_alive()


def test_sigterm_ignoring_factory_escalates_to_sigkill(monkeypatch):
    processes = []
    _patch_process_capture(monkeypatch, processes)

    started = time.monotonic()
    with pytest.raises(TimeoutError):
        run_in_subprocess(
            functools.partial(_sleep_ignoring_sigterm, 60), timeout_s=1.0, grace_period_s=1.0
        )
    elapsed = time.monotonic() - started
    # Proves the escalation path fired (terminate, wait out the grace period, THEN kill) --
    # not just the first-stage terminate, which this factory deliberately survives.
    assert 1.5 < elapsed < 10.0

    proc = processes[0]
    assert proc.exitcode is not None
    assert not proc.is_alive()


def test_local_lambda_fails_with_a_clear_pickling_error_not_a_hang():
    started = time.monotonic()
    with pytest.raises(Exception, match="[Pp]ickl"):
        run_in_subprocess(lambda: 1, timeout_s=10)
    assert time.monotonic() - started < 5.0


def test_max_memory_mb_never_silently_unenforced():
    # POSIX-only. Three observed outcomes, all acceptable -- the ONE thing that must never happen
    # is silently succeeding with the cap unenforced, or hanging:
    #   1. MemoryError, relayed cleanly -- RLIMIT_AS is genuinely lowerable and enforced (Linux).
    #   2. ValueError, relayed cleanly -- the OS refuses to lower RLIMIT_AS at all (confirmed on
    #      macOS: setrlimit itself fails regardless of the requested value).
    #   3. (Confirmed on real Linux CI, not just reasoned about) RLIMIT_AS IS enforced and the
    #      child correctly hits MemoryError -- but the child is now so memory-constrained that
    #      even multiprocessing.Queue's OWN internal feeder thread fails to start
    #      ("RuntimeError: can't start new thread"), which crashes the child before it can relay
    #      ANYTHING. There is no viable fallback here: a resource-exhausted process cannot be made
    #      to successfully report its own resource exhaustion through a mechanism (a new thread)
    #      that itself needs resources. This is indistinguishable, from the PARENT's side, from an
    #      external kill -- and IS handled by the exact same safety net: the parent's own bounded
    #      queue.get() times out and raises the generic "child exited without delivering a result"
    #      RuntimeError. Accepted and expected, not a bug -- the cap was still genuinely enforced.
    pytest.importorskip("resource")
    started = time.monotonic()
    with pytest.raises((MemoryError, ValueError, RuntimeError)):
        run_in_subprocess(functools.partial(_allocate_mb, 2000), max_memory_mb=50, timeout_s=30)
    assert time.monotonic() - started < 30.0


def test_cpu_time_limit_s_is_enforced():
    # Unlike max_memory_mb, RLIMIT_CPU is confirmed to actually enforce on every POSIX platform
    # tested (including macOS) -- the busy loop gets killed once it exceeds the CPU budget, well
    # before the generous timeout_s safety net.
    pytest.importorskip("resource")
    started = time.monotonic()
    with pytest.raises(Exception):
        run_in_subprocess(_busy_loop_cpu_bound, cpu_time_limit_s=1, timeout_s=30)
    assert time.monotonic() - started < 30.0


def test_two_sequential_calls_both_succeed_cleanly():
    assert run_in_subprocess(functools.partial(_add, 1, 1)) == 2
    assert run_in_subprocess(functools.partial(_add, 2, 2)) == 4


def _patch_process_capture(monkeypatch, out_list):
    """Wraps the multiprocessing context's own Process() constructor so the test can inspect the
    real Process object afterward (exitcode, is_alive()) without run_in_subprocess itself needing
    to expose it -- a black-box-preserving way to check "no zombie left" from outside."""
    import rlm_harness.isolation as isolation_module

    real_get_context = isolation_module.multiprocessing.get_context

    def get_context(method):
        ctx = real_get_context(method)
        real_process = ctx.Process

        def process(*args, **kwargs):
            p = real_process(*args, **kwargs)
            out_list.append(p)
            return p

        ctx.Process = process
        return ctx

    monkeypatch.setattr(isolation_module.multiprocessing, "get_context", get_context)
