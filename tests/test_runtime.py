"""Tests for runtime.configure observability bootstrap (no real LM/network)."""

import sys
import types

import pytest

dspy = pytest.importorskip("dspy")

import rlm_harness.runtime as rt
from rlm_harness import RLMConfig


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
        except Exception as exc:
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


# ---- Claude-subscription auto-routing (configure() reads back claude_agent_lm's own sentinel) --

class _StubClaudeAgentLM:
    """A stand-in for ClaudeAgentLM that captures its own construction args, so these tests don't
    need the optional `claude-agent-sdk` dependency installed to verify the ROUTING logic."""

    calls: list = []

    def __init__(self, model, **kwargs):
        self.model = f"claude-agent-sdk/{model}"
        type(self).calls.append((model, kwargs))


def _stub_claude_agent_lm(monkeypatch):
    import rlm_harness.claude_agent_lm as cal

    _StubClaudeAgentLM.calls = []
    monkeypatch.setattr(cal, "ClaudeAgentLM", _StubClaudeAgentLM)
    return _StubClaudeAgentLM


def test_configure_routes_a_subscription_prefixed_main_model(monkeypatch):
    monkeypatch.setattr(rt, "_try_instrument", lambda: None)
    stub = _stub_claude_agent_lm(monkeypatch)
    rt.configure(RLMConfig(main_model="claude-agent-sdk/sonnet", sub_model="x", observe=False))
    assert stub.calls == [("sonnet", {})]
    assert dspy.settings.lm.model == "claude-agent-sdk/sonnet"


def test_configure_routes_only_the_prefixed_role_when_mixed(monkeypatch):
    monkeypatch.setattr(rt, "_try_instrument", lambda: None)
    stub = _stub_claude_agent_lm(monkeypatch)
    rt.configure(RLMConfig(main_model="x", sub_model="claude-agent-sdk/haiku", observe=False))
    assert stub.calls == [("haiku", {})]         # only the sub role routed
    assert dspy.settings.lm.model == "x"          # the main role built a plain dspy.LM as always
    assert rt.get_sub_lm().model == "claude-agent-sdk/haiku"


def test_configure_explicit_main_lm_override_wins_over_a_prefixed_model_string(monkeypatch):
    # An explicit main_lm/sub_lm kwarg wins outright regardless of the model string — the prefix is
    # only consulted for a role the caller left None.
    from dspy.utils.dummies import DummyLM

    monkeypatch.setattr(rt, "_try_instrument", lambda: None)
    stub = _stub_claude_agent_lm(monkeypatch)
    d_main = DummyLM([{"a": "1"}])
    rt.configure(
        RLMConfig(main_model="claude-agent-sdk/sonnet", sub_model="x", observe=False),
        main_lm=d_main,
    )
    assert stub.calls == []                       # routing never attempted for the overridden role
    assert dspy.settings.lm is d_main


def test_configure_lm_kwargs_never_forwarded_into_the_subscription_lm(monkeypatch):
    # ClaudeAgentLM.__init__ accepts **kwargs and forwards them into dspy.BaseLM without
    # validating they make sense for this adapter — api_key/base_url/custom_llm_provider must
    # never reach it (meaningless, or for base_url actively misleading).
    monkeypatch.setattr(rt, "_try_instrument", lambda: None)
    stub = _stub_claude_agent_lm(monkeypatch)
    rt.configure(
        RLMConfig(
            main_model="claude-agent-sdk/sonnet", sub_model="x",
            api_key="sk-should-not-leak", base_url="https://should-not-leak.example",
            observe=False,
        )
    )
    (name, kwargs) = stub.calls[0]
    assert name == "sonnet"
    assert kwargs == {}


def test_request_timeout_is_not_forwarded_into_the_subscription_lm(monkeypatch):
    """`request_timeout_s` bounds ONE HTTP request (dspy and litellm each retry around it);
    `ClaudeAgentLM.timeout_s` is an END-TO-END per-call deadline that includes time queued
    behind the SDK's semaphore. Mapping one onto the other would make queued sub-LM calls under
    `llm_query_batched` time out from waiting alone — so the knob stays litellm-only."""
    monkeypatch.setattr(rt, "_try_instrument", lambda: None)
    stub = _stub_claude_agent_lm(monkeypatch)
    rt.configure(
        RLMConfig(main_model="claude-agent-sdk/sonnet", sub_model="x", observe=False,
                  request_timeout_s=45.0)
    )
    assert stub.calls[0][1] == {}                          # not forwarded...
    assert rt.get_sub_lm().kwargs.get("timeout") == 45.0   # ...but still applied to the LM that can use it


def test_configure_warns_when_a_timeout_cannot_reach_an_auto_routed_role(monkeypatch, caplog):
    """The silent no-op is the actual defect: a consumer sets RLM_REQUEST_TIMEOUT, sees no
    error, and believes the route is bounded to that number."""
    monkeypatch.setattr(rt, "_try_instrument", lambda: None)
    _stub_claude_agent_lm(monkeypatch)
    with caplog.at_level("WARNING", logger="rlm_harness.runtime"):
        rt.configure(
            RLMConfig(main_model="claude-agent-sdk/sonnet", sub_model="x", observe=False,
                      request_timeout_s=45.0)
        )
    msg = "\n".join(r.getMessage() for r in caplog.records)
    assert "main_lm" in msg and "ClaudeAgentLM" in msg
    assert "timeout_s" in msg          # names the seam that DOES let you choose the number


def test_no_warning_when_no_role_was_auto_routed(monkeypatch, caplog):
    """A plain litellm setup consumes the knob fully — warning there would be noise, and noise
    is how a real warning gets ignored."""
    monkeypatch.setattr(rt, "_try_instrument", lambda: None)
    with caplog.at_level("WARNING", logger="rlm_harness.runtime"):
        rt.configure(RLMConfig(main_model="openai/x", sub_model="openai/x", observe=False,
                               request_timeout_s=45.0))
    assert not [r for r in caplog.records if "request_timeout_s" in r.getMessage()]


def test_no_warning_for_an_explicitly_injected_lm(monkeypatch, caplog):
    """A caller who built the LM themselves owns its deadline and does not need telling."""
    from dspy.utils.dummies import DummyLM

    monkeypatch.setattr(rt, "_try_instrument", lambda: None)
    injected = DummyLM([{"answer": "a"}])
    with caplog.at_level("WARNING", logger="rlm_harness.runtime"):
        rt.configure(
            RLMConfig(main_model="claude-agent-sdk/sonnet", sub_model="claude-agent-sdk/haiku",
                      observe=False, request_timeout_s=45.0),
            main_lm=injected, sub_lm=injected,
        )
    assert not [r for r in caplog.records if "request_timeout_s" in r.getMessage()]


def test_configure_bare_subscription_prefix_raises_value_error_not_system_exit(monkeypatch):
    monkeypatch.setattr(rt, "_try_instrument", lambda: None)
    _stub_claude_agent_lm(monkeypatch)
    with pytest.raises(ValueError, match=r"expected claude-agent-sdk/<id>"):
        rt.configure(RLMConfig(main_model="claude-agent-sdk/", sub_model="x", observe=False))


def test_configure_plain_model_never_routes(monkeypatch):
    # Regression guard: a plain (non-prefixed) model string is byte-for-byte unaffected by this
    # change — the routing branch is a pure addition to the `is None` path.
    monkeypatch.setattr(rt, "_try_instrument", lambda: None)
    stub = _stub_claude_agent_lm(monkeypatch)
    rt.configure(RLMConfig(main_model="x", sub_model="y", observe=False))
    assert stub.calls == []
    assert dspy.settings.lm.model == "x"


def test_configure_stale_anthropic_api_key_raises_runtime_error_uncollided(monkeypatch):
    # The REAL ClaudeAgentLM.__init__ logic (not the stub) runs here, to verify the actual
    # RuntimeError it raises for a stale ANTHROPIC_API_KEY propagates all the way out of
    # configure() — structurally, not merely because the message text happens to differ from
    # configure()'s own unrelated ownership-error swallow: the LM-construction step (where this
    # RuntimeError originates) runs entirely BEFORE the try/except that wraps dspy.configure(...),
    # so it cannot reach that handler regardless of message content.
    import rlm_harness.claude_agent_lm as cal

    monkeypatch.setattr(rt, "_try_instrument", lambda: None)
    monkeypatch.setattr(cal, "_require_claude_agent_sdk", lambda: None)  # bypass the SDK-install check
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-stale-key")
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        rt.configure(RLMConfig(main_model="claude-agent-sdk/sonnet", sub_model="x", observe=False))
