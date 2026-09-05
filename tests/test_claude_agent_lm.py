"""ClaudeAgentLM tests — the optional Claude-subscription adapter (`rlm-harness[subscription]`).

The heavy `claude-agent-sdk` is NOT a test dependency: the pure helpers run without it, the
lazy export is asserted without it, and construction is exercised against a FAKE
`claude_agent_sdk` injected into `sys.modules` — so the kit's CI never pulls the ~80MB SDK
wheel. dspy IS a hard dep, so the module imports; guard anyway for a dspy-less environment.
"""

import contextlib
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


# --- prompt tokens are split across three fields, not one ---------------------------------


def _usage_of(raw):
    """Drive the REAL helper. Re-implementing the sum here would test the test: a mutation of the
    production arithmetic would leave a duplicated version green."""
    from rlm_harness.claude_agent_lm import _prompt_tokens_from_sdk_usage

    return _prompt_tokens_from_sdk_usage(raw)


def test_a_cached_prompt_counts_the_cache_fields_not_just_the_remainder():
    """MEASURED on a live subscription call with a ~2k-token prompt. The Agent SDK caches the
    system prompt and tool definitions, so `input_tokens` is only what was neither written to
    nor read from the cache. Reading it alone recorded 2 where the prompt was 2049 — 0.1% — so
    every consumer reading this adapter's prompt size, out of the trace or out of `lm.history`,
    was three orders of magnitude low. (It is NOT 1.10.0's truncation ratio's denominator: that
    ratio is `completion_tokens / cap`.)"""
    first_sight = {"input_tokens": 2, "cache_creation_input_tokens": 2047,
                   "cache_read_input_tokens": 0, "output_tokens": 4}
    repeat = {"input_tokens": 2, "cache_creation_input_tokens": 0,
              "cache_read_input_tokens": 2047, "output_tokens": 4}
    assert _usage_of(first_sight) == 2049, "cache_creation_input_tokens was dropped"
    assert _usage_of(repeat) == 2049, "cache_read_input_tokens was dropped"


def test_a_provider_reporting_no_cache_fields_is_unaffected():
    """The fix must be inert where there is no caching: absent keys contribute nothing, so a
    plain `{input_tokens: N}` still maps to N rather than being changed by the new arithmetic."""
    assert _usage_of({"input_tokens": 731, "output_tokens": 12}) == 731
    assert _usage_of({}) == 0
    assert _usage_of(None) == 0


def test_a_bool_is_not_a_token_count():
    """Same guard the trace's own duration reader carries: `True` is an `int` in Python, and a
    provider returning one must not be summed as 1."""
    assert _usage_of({"input_tokens": True, "cache_read_input_tokens": 100}) == 100


# --- ...and the CALL SITE actually maps them that way -----------------------------------------
#
# The three tests above pin the arithmetic of a pure helper. They do NOT pin that `_acomplete`
# calls it: restoring the original `prompt_tokens = usage.get("input_tokens", 0)` at the call
# site, with the helper left intact but unused, leaves every one of them green. These drive the
# real coroutine end to end through `forward()` — the bridge loop, the semaphore, `_query_once`
# and the litellm mapping — against a fake SDK, so the defect's own location is covered.


@pytest.fixture
def sdk_returning(monkeypatch):
    """Build a fake `claude_agent_sdk` whose single call returns a `ResultMessage`.

    Richer than `fake_sdk` above, which only has to survive construction: this one is driven,
    so `query` is a real async generator and `ClaudeAgentOptions` accepts the adapter's kwargs.

    What it CANNOT catch is SDK drift: `ClaudeAgentOptions(**kwargs)` swallows anything, so a
    renamed kwarg stays green here. The eight the adapter passes were checked by hand against the
    real `claude-agent-sdk` 0.2.152, where `ResultMessage.usage` is a `dict[str, Any] | None`
    passed through verbatim from the CLI — which is why the shapes below are dicts, and why a
    `None` in one of their values is a case worth testing at all.
    """

    def build(usage, *, total_cost_usd=None, text="ok"):
        class ResultMessage:
            def __init__(self):
                self.usage = usage
                self.total_cost_usd = total_cost_usd
                self.is_error = False
                self.subtype = "success"
                self.result = text
                self.structured_output = None

        class ClaudeAgentOptions:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        async def query(*, prompt, options):
            yield ResultMessage()

        mod = types.ModuleType("claude_agent_sdk")
        mod.ClaudeAgentOptions = ClaudeAgentOptions
        mod.ResultMessage = ResultMessage
        mod.query = query
        monkeypatch.setitem(sys.modules, "claude_agent_sdk", mod)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        return mod

    return build


def test_the_call_site_reports_the_whole_prompt_not_the_uncached_remainder(sdk_returning):
    """The measured first-sight shape, driven through the real completion path. Before the fix
    this recorded prompt_tokens=2 and total_tokens=6."""
    sdk_returning({"input_tokens": 2, "cache_creation_input_tokens": 2047,
                   "cache_read_input_tokens": 0, "output_tokens": 4})
    response = rlm_harness.ClaudeAgentLM("sonnet").forward(prompt="hi")
    assert response.usage.prompt_tokens == 2049
    assert response.usage.completion_tokens == 4
    assert response.usage.total_tokens == 2053


def test_an_unreported_usage_stays_absent_instead_of_becoming_three_zeroes(sdk_returning):
    """`absent is not zero` is a promise the guide publishes and this adapter was breaking: it
    always passed a `usage=` kwarg, and litellm materialises `Usage(0, 0, 0)` from an empty one.
    A structural zero is indistinguishable from a real zero-token call, which is the exact
    failure class the kit treats as a defect elsewhere. Omitting the kwarg leaves the attribute
    off, so dspy's own read yields `{}` and its tracker skips the entry."""
    import dspy

    sdk_returning(None)
    lm = rlm_harness.ClaudeAgentLM("sonnet")
    assert not hasattr(lm.forward(prompt="hi"), "usage"), "an unreported usage became zeroes"
    # And drive DSPY, not a re-implementation of its read: the promise is that nothing lands in
    # the tracker, which is what a consumer's `run_end.payload.usage` is built from. Asserting
    # `dict(getattr(response, "usage", {}) or {}) == {}` here instead would be vacuous — it
    # follows from the line above and cannot fail if dspy changes what it does with an empty
    # entry. This can.
    with dspy.track_usage() as tracker:
        lm(prompt="hi")
    assert tracker.get_total_tokens() == {}, "an absent usage was recorded as a zero-token call"
    # Both dspy reads, not just the legacy one: the typed path resolves usage through
    # `usage_from_response` rather than `getattr(response, "usage", {})`, and `CLAUDE.md` flags
    # the typed/legacy split as exactly the kind of dspy difference that moves between versions.
    with dspy.context(experimental=True), dspy.track_usage() as typed_tracker:
        lm(prompt="hi")
    assert typed_tracker.get_total_tokens() == {}, "the typed dspy path recorded a zero instead"


def test_a_null_output_count_does_not_kill_the_call(sdk_returning):
    """NEITHER token field was read with an int guard — both were a bare `.get(..., 0)`, so a
    `None` from the SDK in either one reached `prompt_tokens + completion_tokens` and raised
    `TypeError`, failing the whole LM call over a missing count. `output_tokens` is the field
    exercised here because `input_tokens`'s replacement is already covered by the sum above."""
    sdk_returning({"input_tokens": 5, "output_tokens": None})
    response = rlm_harness.ClaudeAgentLM("sonnet").forward(prompt="hi")
    assert response.usage.prompt_tokens == 5
    assert response.usage.completion_tokens == 0
    assert response.usage.total_tokens == 5


def test_the_sdk_reported_cost_rides_along_untouched(sdk_returning):
    """`response_cost` is the SDK's OWN figure for the call, not derived from the tokens — the
    comment block guarding that distinction is the longest in the module and had no test. It is
    also the only numeric field the adapter reads outside `_token_count`, so pin that it survives
    and that an absent cost adds no key."""
    sdk_returning({"input_tokens": 5, "output_tokens": 2}, total_cost_usd=0.014149)
    priced = rlm_harness.ClaudeAgentLM("sonnet").forward(prompt="hi")
    assert priced._hidden_params["response_cost"] == 0.014149
    # ...and it must not be reconciled against the tokens: 7 tokens did not cost 1.4 cents.
    assert priced.usage.total_tokens == 7

    sdk_returning({"input_tokens": 5, "output_tokens": 2}, total_cost_usd=None)
    unpriced = rlm_harness.ClaudeAgentLM("sonnet").forward(prompt="hi")
    assert "response_cost" not in unpriced._hidden_params


def test_the_returned_text_is_the_sdk_result(sdk_returning):
    """The `text=` half of the fixture, exercised so the response body is not taken on trust
    while the usage half is being asserted."""
    sdk_returning({"input_tokens": 1, "output_tokens": 1}, text="the answer")
    response = rlm_harness.ClaudeAgentLM("sonnet").forward(prompt="hi")
    assert response.choices[0].message.content == "the answer"


# --- the totals give the context size conditionally; the rounds give it unconditionally -------
#
# MEASURED entry shape (consumer's machine, claude-agent-sdk 0.2.119 / CLI 2.1.261): the four
# token fields, `type`, and a nested `cache_creation` breakdown. `model` and further members
# appear in the SDK's schema, which is why the guard checks the CONTAINER and never an entry's
# keys — enumerating would freeze a set that demonstrably moves.

_ROUND = {
    "type": "message",
    "model": "claude-sonnet-5",
    "input_tokens": 2,
    "cache_creation_input_tokens": 1080,
    "cache_read_input_tokens": 0,
    "output_tokens": 125,
    # A DECOMPOSITION of cache_creation_input_tokens, not an addition to it — Anthropic documents
    # it as "Breakdown of cached tokens by TTL". Summing every int in this dict gives 2287.
    "cache_creation": {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 1080},
}


def _usage_with_rounds(rounds):
    return {"input_tokens": 2, "cache_creation_input_tokens": 1080,
            "cache_read_input_tokens": 0, "output_tokens": 125, "iterations": rounds}


def test_the_rounds_ride_through_verbatim_beside_the_aggregate(sdk_returning):
    """The whole point of carrying them: `prompt_tokens` is a per-CALL total, and a reader
    watching context growth cannot tell a growing prompt from a retried call without the
    breakdown. Entries stay in the SDK's own vocabulary — normalising them to
    prompt_tokens/completion_tokens would collapse the cache split, which is exactly what made
    1.10.2's bug invisible for as long as it was."""
    sdk_returning(_usage_with_rounds([_ROUND]))
    usage = rlm_harness.ClaudeAgentLM("sonnet").forward(prompt="hi").usage
    assert usage.api_rounds == {"rounds": [_ROUND]}, "the rounds were reshaped, not carried"
    # ...beside the aggregate, never instead of it, and the aggregate is untouched by them.
    assert usage.prompt_tokens == 1082
    assert usage.completion_tokens == 125


def test_the_aggregate_is_the_three_TOP_LEVEL_input_fields_not_every_int_in_a_round():
    """A pin on THIS adapter's arithmetic over a fixture — NOT a claim that the SDK's top-level
    usage equals its rounds' sum. That is not merely unobserved, it is structurally false in
    general: the totals accumulate across every API request a call made while the rounds cover
    only the last, and a `compaction` entry's tokens are excluded from the totals outright. What
    this DOES pin is the double-count the nested members invite: adding every integer in `_ROUND`
    gives 2287, counting `cache_creation`'s 1080 twice."""
    from rlm_harness.claude_agent_lm import _prompt_tokens_from_sdk_usage

    assert _prompt_tokens_from_sdk_usage(_usage_with_rounds([_ROUND])) == 1082

    def every_int(obj):
        if isinstance(obj, dict):
            return sum(every_int(v) for v in obj.values())
        return obj if isinstance(obj, int) and not isinstance(obj, bool) else 0

    assert every_int(_ROUND) == 2287, "the fixture must contain the trap this test exists to pin"


@pytest.mark.parametrize("reported", [
    pytest.param({}, id="absent"),
    pytest.param({"iterations": None}, id="null"),
    pytest.param({"iterations": []}, id="empty-list"),
    pytest.param({"iterations": ["a", "b"]}, id="list-of-strings"),
    pytest.param({"iterations": {"rounds": [_ROUND]}}, id="bare-dict"),
])
def test_only_a_non_empty_list_of_dicts_becomes_api_rounds(sdk_returning, reported):
    """Five inputs, ONE outcome: no key. The empty list is the one that needs saying — `all(...)`
    over an empty list is True, so a guard that only checked "list of dicts" would CARRY `[]`,
    and the CLI does construct empty ones. Collapsing all five keeps "no api_rounds" a single
    fact with a single meaning instead of four."""
    sdk_returning({"input_tokens": 5, "output_tokens": 2, **reported})
    usage = rlm_harness.ClaudeAgentLM("sonnet").forward(prompt="hi").usage
    assert not hasattr(usage, "api_rounds"), "an unusable shape reached the trace"
    assert usage.prompt_tokens == 5, "the aggregate must survive a rejected breakdown"


def test_a_round_with_junk_values_is_carried_and_does_not_touch_the_aggregate(sdk_returning):
    """Rounds get NONE of `_token_count`'s int/bool guards — that is what verbatim means. The
    three-field rule is an instruction to the READER, not something the kit applies per round."""
    junk = {"type": "message", "input_tokens": None, "output_tokens": True}
    sdk_returning(_usage_with_rounds([junk]))
    usage = rlm_harness.ClaudeAgentLM("sonnet").forward(prompt="hi").usage
    assert usage.api_rounds == {"rounds": [junk]}
    assert usage.prompt_tokens == 1082


def test_an_unreported_usage_cannot_be_resurrected_by_the_rounds(sdk_returning):
    """1.10.2's promise, unchanged: no usage means no `usage` kwarg at all, so there is no object
    for `api_rounds` to hang off and no response-level zero to manufacture."""
    sdk_returning(None)
    assert not hasattr(rlm_harness.ClaudeAgentLM("sonnet").forward(prompt="hi"), "usage")


def test_both_dspy_reads_carry_the_rounds_to_the_tracker(sdk_returning):
    """The typed path works only because dspy's `LMUsage` declares `extra="allow"` — a one-line
    upstream decision that could change and would silently drop the field for experimental-mode
    consumers. 1.10.2 learned to drive both reads; this drives both too."""
    import dspy

    sdk_returning(_usage_with_rounds([_ROUND]))
    lm = rlm_harness.ClaudeAgentLM("sonnet")
    for ctx in (contextlib.nullcontext(), dspy.context(experimental=True)):
        with ctx, dspy.track_usage() as tracker:
            lm(prompt="hi")
        entries = tracker.usage_data["claude-agent-sdk/sonnet"]
        assert [e["api_rounds"] for e in entries] == [{"rounds": [_ROUND]}]


def test_mixed_presence_reaches_the_kits_own_reader_intact(sdk_returning):
    """A run mixing calls that report rounds with calls that do not, asserted through
    `usage_since` — the shim the trace is actually built from, not the raw tracker one hop
    earlier. WITH-then-WITHOUT is the order used because it is the one a FLAT list would raise on
    through `get_total_tokens()`; nested it does not raise in any order, which is the point."""
    import dspy

    from rlm_harness._dspy_compat import usage_baseline, usage_since

    sdk_returning(_usage_with_rounds([_ROUND]))
    with dspy.track_usage() as tracker:
        base = usage_baseline(tracker)
        rlm_harness.ClaudeAgentLM("sonnet")(prompt="hi")
        sdk_returning({"input_tokens": 7, "output_tokens": 1})
        rlm_harness.ClaudeAgentLM("sonnet")(prompt="hi")
        fresh = usage_since(tracker, base)
    calls = fresh["claude-agent-sdk/sonnet"]
    assert [("api_rounds" in c) for c in calls] == [True, False], "order or presence was lost"
    assert calls[0]["api_rounds"] == {"rounds": [_ROUND]}
    assert calls[1]["prompt_tokens"] == 7


def test_a_rejected_breakdown_is_logged_and_an_absent_one_is_not(sdk_returning, caplog):
    """Without this line, "no run has `api_rounds`" is indistinguishable between the SDK never
    reporting it, reporting it unusably, and the key having been RENAMED upstream — the last being
    the failure this project has paid for repeatedly. A JSON `null` stays silent on purpose: that
    IS absence, not a rejection."""
    import logging

    cases = [({"iterations": "nope"}, 1), ({"iterations": [1, 2]}, 1),
             # `[]` is the shape most likely to be rejected in practice — it is what the CLI's
             # usage accumulator seeds itself with — so it must be one of the LOGGED ones.
             ({"iterations": []}, 1), ({"iterations": None}, 0), ({}, 0)]
    for reported, expected in cases:
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="rlm_harness.claude_agent_lm"):
            sdk_returning({"input_tokens": 5, "output_tokens": 2, **reported})
            rlm_harness.ClaudeAgentLM("sonnet").forward(prompt="hi")
        assert len(caplog.records) == expected, f"wrong DEBUG count for {reported}"


def test_a_multi_entry_breakdown_rides_through_in_order_and_unfiltered(sdk_returning):
    """Every other fixture here carries ONE round, so a guard that reordered or filtered entries
    would be invisible. This is the shape the field exists for — a server-side fallback puts the
    declined hop's `message` entry and the serving hop's `fallback_message` in the SAME request,
    which is exactly the case where "one `message` entry" is true and the totals still cover two
    hops. Order is load-bearing: the context reading is the LAST qualifying entry."""
    declined = {**_ROUND, "type": "message"}
    served = {"type": "fallback_message", "model": "claude-haiku-4-5-20251001",
              "input_tokens": 9, "cache_creation_input_tokens": 0,
              "cache_read_input_tokens": 1080, "output_tokens": 40}
    sdk_returning(_usage_with_rounds([declined, served]))
    usage = rlm_harness.ClaudeAgentLM("sonnet").forward(prompt="hi").usage
    assert usage.api_rounds == {"rounds": [declined, served]}, "entries were reordered or dropped"
