"""ClaudeAgentLM tests — the optional Claude-subscription adapter (`rlm-harness[subscription]`).

The heavy `claude-agent-sdk` is NOT a test dependency: the pure helpers run without it, the
lazy export is asserted without it, and construction is exercised against a FAKE
`claude_agent_sdk` injected into `sys.modules` — so the kit's CI never pulls the ~80MB SDK
wheel. dspy IS a hard dep, so the module imports; guard anyway for a dspy-less environment.
"""

import sys
import types

import pytest

pytest.importorskip("dspy")

import rlm_harness
from rlm_harness.claude_agent_lm import (
    _looks_rate_limited,
    _require_claude_agent_sdk,
    _split_messages,
    _translate_response_format,
)


def test_lazy_export_without_the_sdk():
    # In __all__ and gettable off the top-level package WITHOUT the SDK installed (the mcp_tools
    # pattern): the module imports clean, the SDK is only needed at construction.
    assert "ClaudeAgentLM" in rlm_harness.__all__
    assert rlm_harness.ClaudeAgentLM.__name__ == "ClaudeAgentLM"


def test_split_messages_bare_prompt():
    assert _split_messages("hello", None) == (None, "hello")


def test_split_messages_system_plus_single_user():
    system, user = _split_messages(
        None, [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    )
    assert system == "S"
    assert user == "U"


def test_split_messages_multi_turn_flatten():
    system, user = _split_messages(
        None, [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    )
    assert system is None
    assert "User: a" in user and "Assistant: b" in user
    assert user.rstrip().endswith("Assistant:")


def test_translate_response_format_pydantic_class():
    from pydantic import BaseModel

    class Out(BaseModel):
        x: int

    fmt = _translate_response_format(Out)
    assert fmt["type"] == "json_schema"
    assert "properties" in fmt["schema"]


def test_translate_response_format_none_and_dict_fallback():
    assert _translate_response_format(None) is None
    # a stock adapter's {"type": "json_object"} dict has no SDK equivalent → dropped
    assert _translate_response_format({"type": "json_object"}) is None


def test_looks_rate_limited_phrase_level():
    assert _looks_rate_limited("HTTP 429: rate limit exceeded")
    assert _looks_rate_limited("usage limit reached, try later")
    assert _looks_rate_limited("model overloaded (529)")
    # bare 'limit'/'rate' in ordinary error text must NOT trip the 30s backoff
    assert not _looks_rate_limited("failed to generate the delimiter")


def test_require_sdk_raises_friendly_install_hint(monkeypatch):
    # Setting sys.modules[name] = None makes `import name` raise, simulating the extra being absent
    # even if it happens to be installed in this env.
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)
    with pytest.raises(ImportError, match=r"rlm-harness\[subscription\]"):
        _require_claude_agent_sdk()


# -- construction against a FAKE SDK (kit CI never installs the real one) ----------------------


@pytest.fixture
def fake_sdk(monkeypatch):
    mod = types.ModuleType("claude_agent_sdk")
    mod.ClaudeAgentOptions = object
    mod.ResultMessage = object
    mod.query = lambda **kwargs: None
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", mod)
    return mod


def test_construction_sets_the_trace_label(fake_sdk, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    lm = rlm_harness.ClaudeAgentLM("opus")
    assert lm.model == "claude-agent-sdk/opus"
    assert lm._alias == "opus"


def test_the_call_deadline_is_a_constructor_choice_with_a_default(fake_sdk, monkeypatch):
    """This LM's bound on a model call, and it had NO test at all. It is deliberately not driven
    by `RLMConfig.request_timeout_s`: that knob is a per-HTTP-request bound which dspy and
    litellm each retry around, whereas this is END-TO-END for one call and includes time queued
    behind the module-level semaphore. `configure(main_lm=...)` is the seam for choosing it —
    see `runtime.configure`, which warns when the knob cannot reach an auto-routed role."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert rlm_harness.ClaudeAgentLM("opus")._timeout_s == 600.0
    assert rlm_harness.ClaudeAgentLM("opus", timeout_s=45.0)._timeout_s == 45.0


def test_construction_refuses_a_leftover_api_key(fake_sdk, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-bill")
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        rlm_harness.ClaudeAgentLM("sonnet")
    # explicit opt-in bypasses the guard
    lm = rlm_harness.ClaudeAgentLM("sonnet", allow_api_key=True)
    assert lm.model == "claude-agent-sdk/sonnet"


def test_construction_without_sdk_fails_fast(monkeypatch):
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ImportError, match=r"rlm-harness\[subscription\]"):
        rlm_harness.ClaudeAgentLM("sonnet")
