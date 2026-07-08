"""Tests for runtime.configure observability bootstrap (no real LM/network)."""

import sys
import types

import pytest

dspy = pytest.importorskip("dspy")

import rlm_kit.runtime as rt  # noqa: E402
from rlm_kit import RLMConfig  # noqa: E402


def test_configure_without_observe_skips_instrumentation(monkeypatch):
    called = {"instr": False}
    monkeypatch.setattr(rt, "_try_instrument", lambda: called.__setitem__("instr", True))
    rt.configure(RLMConfig(main_model="x", sub_model="x", observe=False))
    assert called["instr"] is False
    assert rt.get_config().observe is False


def test_configure_with_observe_calls_instrument(monkeypatch):
    called = {"instr": False}
    monkeypatch.setattr(rt, "_try_instrument", lambda: called.__setitem__("instr", True))
    rt.configure(RLMConfig(main_model="x", sub_model="x", observe=True))
    assert called["instr"] is True


def test_configure_accepts_injected_lms():
    # The public seam for supplying a test double (or a pre-built client): pass main_lm/sub_lm and
    # configure uses them verbatim instead of building from config — so nothing reaches into _STATE.
    from dspy.utils.dummies import DummyLM

    d_main, d_sub = DummyLM([{"a": "1"}]), DummyLM([{"b": "2"}])
    rt.configure(
        RLMConfig(main_model="x", sub_model="x", observe=False), main_lm=d_main, sub_lm=d_sub
    )
    assert rt.get_sub_lm() is d_sub     # injected sub stored + handed back by the public accessor
    assert dspy.settings.lm is d_main   # injected main became the dspy global


def test_configure_tolerates_a_second_thread(monkeypatch):
    # dspy.configure is owner-locked to the first thread/task; a long-lived driver that runs each task
    # in a fresh worker thread (e.g. a server handling per-request live runs) would crash on
    # the 2nd run. configure must swallow that ownership RuntimeError and reuse the global config.
    import threading

    monkeypatch.setattr(rt, "_try_instrument", lambda: None)
    rt.configure(RLMConfig(main_model="x", sub_model="x", observe=False))  # owner = this (main) thread

    err = {}

    def worker():
        try:
            rt.configure(RLMConfig(main_model="x", sub_model="x", observe=False))  # a DIFFERENT thread
        except Exception as exc:  # noqa: BLE001
            err["exc"] = exc

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert "exc" not in err, f"configure from a 2nd thread must not raise; got {err.get('exc')!r}"


def test_try_instrument_bootstraps_langfuse(monkeypatch):
    """When langfuse is importable, _try_instrument calls get_client()."""
    got = {"client": False}
    fake = types.ModuleType("langfuse")
    fake.get_client = lambda: got.__setitem__("client", True)
    monkeypatch.setitem(sys.modules, "langfuse", fake)
    # OpenInference may or may not be installed; either path must not raise.
    rt._try_instrument()
    assert got["client"] is True


def test_try_instrument_never_fatal_without_langfuse(monkeypatch):
    monkeypatch.setitem(sys.modules, "langfuse", None)  # force ImportError
    rt._try_instrument()  # must not raise
