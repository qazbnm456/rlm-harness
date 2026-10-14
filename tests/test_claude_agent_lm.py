"""ClaudeAgentLM tests: the optional Claude-subscription adapter (`rlm-harness[subscription]`).

The heavy `claude-agent-sdk` is NOT a test dependency: the pure helpers run without it, the
lazy export is asserted without it, and construction is exercised against a FAKE
`claude_agent_sdk` injected into `sys.modules`, so the kit's CI never pulls the ~80MB SDK
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


def test_a_subscription_call_goes_through_THIS_adapter_and_not_litellm(fake_sdk, monkeypatch):
    """The property the old `_engine_spec` pin was a proxy for, asserted directly.

    Until 1.17.0 this class was a legacy `forward()` LM and the pin was the ABSENCE of
    `_engine_spec`, because `dspy.clients.execution.prepare` reads `hasattr(lm, "_engine_spec")`
    to decide between a legacy LM and a managed one, and a managed LM with no engine would have
    been routed to litellm. The class now IS engine-managed, legitimately, so that absence is
    gone and asserting it would fail for the right reason while protecting nothing. What must
    stay true is the thing it stood for: a subscription call reaches the Claude Agent SDK through
    this adapter, never litellm. Pinned by driving a call and demanding OUR engine ran.

    `_dspy_compat.copy_lm` is asserted alongside, because "never stamp `_engine_spec` on to
    satisfy `LM.copy`" was the other half: the stamp is now unnecessary rather than forbidden,
    and `copy_lm` must keep working either way.
    """
    import rlm_harness
    from rlm_harness._dspy_compat import copy_lm

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    lm = rlm_harness.ClaudeAgentLM("claude-sonnet-5")
    from rlm_harness.claude_agent_lm import _AsyncClaudeAgentEngine, _ClaudeAgentEngine

    assert isinstance(lm.engine, _ClaudeAgentEngine), "dspy would route this call to litellm"
    assert isinstance(lm._async_engine_spec, _AsyncClaudeAgentEngine), "the async seat is unrouted"
    assert type(copy_lm(lm, rollout_id=1)) is type(lm), "copy_lm stopped producing a variant"


def test_the_subscription_lm_does_not_default_to_a_live_cache(fake_sdk, monkeypatch):
    """An engine-backed LM is MANAGED, so dspy would consult its cache: before 1.17.0 a
    subscription call never touched one, because an unmanaged LM's `cache=True` is inert
    (1.15.1 measured exactly that and is why `lm_cache_is_live` is asymmetric). The migration
    would have switched caching ON for every consumer who builds this class directly and hands
    it in through `configure(main_lm=...)`, which keeps an injected LM's own flag. So the
    default moves here, and the caller can still ask for one.
    """
    import rlm_harness
    from rlm_harness._dspy_compat import lm_cache_is_live

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert lm_cache_is_live(rlm_harness.ClaudeAgentLM("sonnet")) is False
    assert lm_cache_is_live(rlm_harness.ClaudeAgentLM("sonnet", cache=True)) is True


def test_construction_sets_the_trace_label(fake_sdk, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert rlm_harness.ClaudeAgentLM("opus").model == "claude-agent-sdk/opus"


def test_the_alias_reaches_the_sdk_rather_than_the_prefixed_trace_label(sdk_returning):
    """The model the SDK is asked for is the ALIAS, not the trace label. Asserted on what the SDK
    receives: the LM used to keep a `_alias` copy of its own, and a mutation showed nothing read
    it, so a test asserting the copy pinned a vestige rather than the call."""
    sdk = sdk_returning({"input_tokens": 1, "output_tokens": 1})
    lm = rlm_harness.ClaudeAgentLM("opus")
    _complete(lm)
    assert lm.model == "claude-agent-sdk/opus", "the trace label lost its prefix"
    assert sdk.captured_options["model"] == "opus", "the SDK was asked for the prefixed label"


def test_the_call_deadline_is_a_constructor_choice_and_it_actually_fires(sdk_returning, monkeypatch):
    """This LM's bound on a model call, DRIVEN rather than read off an attribute.

    It is deliberately not `RLMConfig.request_timeout_s`: that knob is a per-HTTP-request bound
    which dspy and litellm each retry around, whereas this is END-TO-END for one call and includes
    time queued behind the module-level semaphore. `configure(main_lm=...)` is the seam for
    choosing it: see `runtime.configure`, which warns when the knob cannot reach an auto-routed
    role.

    The predecessor asserted `lm._timeout_s`, a copy the LM kept and nothing used. This hangs the
    fake SDK and demands the deadline fire, which also covers the cancel-on-timeout path that no
    test reached before.
    """
    import asyncio

    sdk = sdk_returning({"input_tokens": 1, "output_tokens": 1})

    async def hanging(*, prompt, options):
        await asyncio.sleep(30)
        yield None  # pragma: no cover - the deadline fires first

    monkeypatch.setattr(sdk, "query", hanging)
    with pytest.raises(TimeoutError, match="timed out after 0.2s"):
        _complete(rlm_harness.ClaudeAgentLM("opus", timeout_s=0.2))


def test_the_call_deadline_defaults_to_ten_minutes(fake_sdk, monkeypatch):
    """The DEFAULT, which cannot be driven in a test (it would take ten minutes), so it is read
    off the object that uses it rather than a copy: `_call` is what `complete` reads."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert rlm_harness.ClaudeAgentLM("opus")._call._timeout_s == 600.0
    assert rlm_harness.ClaudeAgentLM("opus", timeout_s=45.0)._call._timeout_s == 45.0


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
    nor read from the cache. Reading it alone recorded 2 where the prompt was 2049-0.1%, so
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
# real coroutine end to end through the ENGINE (`lm.engine.complete(request)`): the bridge loop,
# the semaphore, `_query_once` and the `lm15.Response` mapping, against a fake SDK, so the
# defect's own location is covered. It read `forward()` and "the litellm mapping" until 1.17.0
# migrated the class off both.


def _complete(lm, prompt="hi", *, response_format=None):
    """Drive the LM's own engine with a real `lm15.Request`, returning the `lm15.Response`.

    `lm.engine` is dspy's public accessor for what was passed as `engine=`. Driving it is the
    engine-era equivalent of the old `.forward(prompt=...)`: the same coroutine, bridge loop,
    semaphore and `_query_once` underneath. `lm("hi")` is used instead wherever the property under
    test is one DSPY mediates (usage tracking), since a re-implementation of that read would be
    vacuous.
    """
    import dspy

    T = dspy.lm15
    request = T.Request(
        model=lm.model,
        messages=[T.Message(role="user", parts=[T.TextPart(text=prompt)])],
        config=T.Config(response_format=response_format),
    )
    return lm.engine.complete(request)


@pytest.fixture
def sdk_returning(monkeypatch):
    """Build a fake `claude_agent_sdk` whose single call returns a `ResultMessage`.

    Richer than `fake_sdk` above, which only has to survive construction: this one is driven,
    so `query` is a real async generator and `ClaudeAgentOptions` accepts the adapter's kwargs.

    What it CANNOT catch is SDK drift: `ClaudeAgentOptions(**kwargs)` swallows anything, so a
    renamed kwarg stays green here. The eight the adapter passes were checked by hand against the
    real `claude-agent-sdk` 0.2.152, where `ResultMessage.usage` is a `dict[str, Any] | None`
    passed through verbatim from the CLI, which is why the shapes below are dicts, and why a
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

        captured: dict = {}

        class ClaudeAgentOptions:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                captured.clear()
                captured.update(kwargs)

        async def query(*, prompt, options):
            yield ResultMessage()

        mod = types.ModuleType("claude_agent_sdk")
        mod.ClaudeAgentOptions = ClaudeAgentOptions
        mod.ResultMessage = ResultMessage
        mod.query = query
        mod.captured_options = captured     # what the adapter actually passed, per call
        monkeypatch.setitem(sys.modules, "claude_agent_sdk", mod)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        return mod

    return build


def test_the_call_site_reports_the_whole_prompt_not_the_uncached_remainder(sdk_returning):
    """The measured first-sight shape, driven through the real completion path. Before the fix
    this recorded prompt_tokens=2 and total_tokens=6."""
    sdk_returning({"input_tokens": 2, "cache_creation_input_tokens": 2047,
                   "cache_read_input_tokens": 0, "output_tokens": 4})
    response = _complete(rlm_harness.ClaudeAgentLM("sonnet"))
    assert response.usage.input_tokens == 2049
    assert response.usage.output_tokens == 4
    assert response.usage.total_tokens == 2053


def test_an_unreported_usage_stays_absent_instead_of_becoming_three_zeroes(sdk_returning):
    """`absent is not zero` is a promise the guide publishes, kept by a DIFFERENT construction
    since the engine migration and worth re-pinning for that reason.

    `lm15.Response.usage` is a required field, so "omit the kwarg and leave the attribute off" is
    no longer available: absence is now an all-`None` `lm15.Usage`, which `usage_dict` reduces to
    `{}` before the tracker ever sees it. A structural zero would still be indistinguishable from
    a real zero-token call, which is the failure class this guards.
    """
    import dspy

    sdk_returning(None)
    lm = rlm_harness.ClaudeAgentLM("sonnet")
    usage = _complete(lm).usage
    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (None, None, None), (
        "an unreported usage became zeroes"
    )
    # And drive DSPY, not a re-implementation of its read: the promise is that nothing lands in
    # the tracker, which is what a consumer's `run_end.payload.usage` is built from. Asserting the
    # `None`s above alone would be half the property: they are only absence if dspy agrees.
    with dspy.track_usage() as tracker:
        lm(prompt="hi")
    assert tracker.get_total_tokens() == {}, "an absent usage was recorded as a zero-token call"
    # Both dspy reads, not just one: `docs/INVARIANTS.md` flags the typed/legacy split as exactly
    # the kind of dspy difference that moves between versions.
    with dspy.context(experimental=True), dspy.track_usage() as typed_tracker:
        lm(prompt="hi")
    assert typed_tracker.get_total_tokens() == {}, "the typed dspy path recorded a zero instead"


def test_a_real_zero_is_still_distinguishable_from_an_absent_usage(sdk_returning):
    """The other arm, and without it the test above is half a measurement: a construction that
    reported absence for EVERYTHING would satisfy it. A call the SDK says used zero tokens must
    still reach the tracker as a zero."""
    import dspy

    sdk_returning({"input_tokens": 0, "output_tokens": 0})
    with dspy.track_usage() as tracker:
        rlm_harness.ClaudeAgentLM("sonnet")(prompt="hi")
    assert tracker.get_total_tokens() == {"claude-agent-sdk/sonnet": {
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        "prompt_tokens_details": None, "completion_tokens_details": None,
    }}, "a genuine zero-token call was swallowed as absence"


def test_a_null_output_count_does_not_kill_the_call(sdk_returning):
    """NEITHER token field was read with an int guard. Both were a bare `.get(..., 0)`, so a
    `None` from the SDK in either one reached `prompt_tokens + completion_tokens` and raised
    `TypeError`, failing the whole LM call over a missing count. `output_tokens` is the field
    exercised here because `input_tokens`'s replacement is already covered by the sum above."""
    sdk_returning({"input_tokens": 5, "output_tokens": None})
    response = _complete(rlm_harness.ClaudeAgentLM("sonnet"))
    assert response.usage.input_tokens == 5
    assert response.usage.output_tokens == 0
    assert response.usage.total_tokens == 5


def test_the_sdk_reported_cost_rides_along_untouched(sdk_returning):
    """`response_cost` is the SDK's OWN figure for the call, not derived from the tokens: the
    comment block guarding that distinction is the longest in the module and had no test. It is
    also the only numeric field the adapter reads outside `_token_count`, so pin that it survives
    and that an absent cost adds no key."""
    sdk_returning({"input_tokens": 5, "output_tokens": 2}, total_cost_usd=0.014149)
    priced = _complete(rlm_harness.ClaudeAgentLM("sonnet"))
    assert priced.provider_data["total_cost_usd"] == 0.014149
    # ...and it must not be reconciled against the tokens: 7 tokens did not cost 1.4 cents.
    assert priced.usage.total_tokens == 7

    sdk_returning({"input_tokens": 5, "output_tokens": 2}, total_cost_usd=None)
    unpriced = _complete(rlm_harness.ClaudeAgentLM("sonnet"))
    assert unpriced.provider_data is None, "an absent cost invented a provider_data payload"


def test_the_returned_text_is_the_sdk_result(sdk_returning):
    """The `text=` half of the fixture, exercised so the response body is not taken on trust
    while the usage half is being asserted."""
    sdk_returning({"input_tokens": 1, "output_tokens": 1}, text="the answer")
    response = _complete(rlm_harness.ClaudeAgentLM("sonnet"))
    assert response.message.parts[0].text == "the answer"


# --- the totals give the context size conditionally; the rounds give it unconditionally -------
#
# MEASURED entry shape (consumer's machine, claude-agent-sdk 0.2.119 / CLI 2.1.261): the four
# token fields, `type`, and a nested `cache_creation` breakdown. `model` and further members
# appear in the SDK's schema, which is why the guard checks the CONTAINER and never an entry's
# keys: enumerating would freeze a set that demonstrably moves.

_ROUND = {
    "type": "message",
    "model": "claude-sonnet-5",
    "input_tokens": 2,
    "cache_creation_input_tokens": 1080,
    "cache_read_input_tokens": 0,
    "output_tokens": 125,
    # A DECOMPOSITION of cache_creation_input_tokens, not an addition to it: Anthropic documents
    # it as "Breakdown of cached tokens by TTL". Summing every int in this dict gives 2287.
    "cache_creation": {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 1080},
}


def _usage_with_rounds(rounds):
    return {"input_tokens": 2, "cache_creation_input_tokens": 1080,
            "cache_read_input_tokens": 0, "output_tokens": 125, "iterations": rounds}


def _rounds_events(lm, tmp_path, name="t.jsonl"):
    """Drive one call inside a recorder and return its `api_rounds` events.

    The breakdown is a trace EVENT since 1.17.0, not a key on the usage entry: `usage_dict`
    rebuilds a fixed key set off the typed `lm15.Usage` and passes nothing else through, so it
    could not ride along any more (measured). `step_id` is what orders these.
    """
    from rlm_harness.trace import EVENT_API_ROUNDS, TraceRecorder, load_events

    path = str(tmp_path / name)
    with TraceRecorder(path, run_id="r"):
        response = _complete(lm)
    events = [e for e in load_events(path) if e["type"] == EVENT_API_ROUNDS]
    return response, events


def test_the_rounds_ride_through_verbatim_as_their_own_event(sdk_returning, tmp_path):
    """The whole point of carrying them: `prompt_tokens` is a per-CALL total, and a reader
    watching context growth cannot tell a growing prompt from a retried call without the
    breakdown. Entries stay in the SDK's own vocabulary: normalising them to
    prompt_tokens/completion_tokens would collapse the cache split, which is exactly what made
    1.10.2's bug invisible for as long as it was. The nesting under `rounds` is kept so a reader
    that already parsed `api_rounds["rounds"]` parses this payload unchanged."""
    sdk_returning(_usage_with_rounds([_ROUND]))
    response, events = _rounds_events(rlm_harness.ClaudeAgentLM("sonnet"), tmp_path)
    assert [e["payload"]["rounds"] for e in events] == [[_ROUND]], "the rounds were reshaped"
    assert events[0]["payload"]["model"] == "claude-agent-sdk/sonnet"
    # ...beside the aggregate, never instead of it, and the aggregate is untouched by them.
    assert response.usage.input_tokens == 1082
    assert response.usage.output_tokens == 125


def test_a_call_outside_a_recorder_still_completes(sdk_returning):
    """Observability never breaks a run: with no recorder there is nowhere to put the breakdown,
    and the call must return anyway. This is the path a consumer takes when they build the LM and
    call it directly, outside any `TraceRecorder`."""
    sdk_returning(_usage_with_rounds([_ROUND]))
    response = _complete(rlm_harness.ClaudeAgentLM("sonnet"))
    assert response.usage.input_tokens == 1082
    assert response.message.parts[0].text == "ok"


@pytest.mark.parametrize("reported", [
    pytest.param({}, id="absent"),
    pytest.param({"iterations": None}, id="null"),
    pytest.param({"iterations": []}, id="empty-list"),
    pytest.param({"iterations": ["a", "b"]}, id="list-of-strings"),
    pytest.param({"iterations": {"rounds": [_ROUND]}}, id="bare-dict"),
])
def test_only_a_non_empty_list_of_dicts_becomes_an_api_rounds_event(sdk_returning, tmp_path, reported):
    """Five inputs, ONE outcome: no event. The empty list is the one that needs saying, `all(...)`
    over an empty list is True, so a guard that only checked "list of dicts" would CARRY `[]`,
    and the CLI does construct empty ones. Collapsing all five keeps "no api_rounds" a single
    fact with a single meaning instead of four.

    Asserted on the EVENT STREAM rather than on the response: an `lm15.Usage` has no
    `api_rounds` attribute to be missing, so `not hasattr(usage, "api_rounds")` would hold for
    every input including a good one, and could never go red.
    """
    sdk_returning({"input_tokens": 5, "output_tokens": 2, **reported})
    response, events = _rounds_events(rlm_harness.ClaudeAgentLM("sonnet"), tmp_path)
    assert events == [], "an unusable shape reached the trace"
    assert response.usage.input_tokens == 5, "the aggregate must survive a rejected breakdown"


def test_a_round_with_junk_values_is_carried_and_does_not_touch_the_aggregate(sdk_returning, tmp_path):
    """Rounds get NONE of `_token_count`'s int/bool guards: that is what verbatim means. The
    three-field rule is an instruction to the READER, not something the kit applies per round."""
    junk = {"type": "message", "input_tokens": None, "output_tokens": True}
    sdk_returning(_usage_with_rounds([junk]))
    response, events = _rounds_events(rlm_harness.ClaudeAgentLM("sonnet"), tmp_path)
    assert [e["payload"]["rounds"] for e in events] == [[junk]]
    assert response.usage.input_tokens == 1082


def test_an_unreported_usage_cannot_be_resurrected_by_the_rounds(sdk_returning, tmp_path):
    """1.10.2's promise, kept through the migration: a breakdown must not manufacture a usage.
    The SDK reports no usage at all here, so there are no rounds to find and the totals stay
    absent rather than becoming zeroes."""
    sdk_returning(None)
    response, events = _rounds_events(rlm_harness.ClaudeAgentLM("sonnet"), tmp_path)
    assert events == []
    assert response.usage.total_tokens is None


def test_the_rounds_reach_the_trace_under_both_dspy_reads(sdk_returning, tmp_path):
    """The predecessor of this test asserted the rounds reached dspy's USAGE TRACKER on both
    reads. That property is gone by construction, not by accident: on the engine path
    `usage_dict` rebuilds the entry from the typed `Usage` and drops everything else, measured on
    dspy 3.4.1. So the thing to pin is that the breakdown still arrives, by the route it takes
    now, and that neither dspy read interferes with it."""
    import dspy

    from rlm_harness.trace import EVENT_API_ROUNDS, TraceRecorder, load_events

    sdk_returning(_usage_with_rounds([_ROUND]))
    lm = rlm_harness.ClaudeAgentLM("sonnet")
    for i, ctx in enumerate((contextlib.nullcontext(), dspy.context(experimental=True))):
        path = str(tmp_path / f"both-{i}.jsonl")
        with TraceRecorder(path, run_id="r"), ctx:
            lm(prompt="hi")
        events = [e for e in load_events(path) if e["type"] == EVENT_API_ROUNDS]
        assert [e["payload"]["rounds"] for e in events] == [[_ROUND]], f"lost under read {i}"


def test_mixed_presence_keeps_its_order_through_the_event_stream(sdk_returning, tmp_path):
    """A run mixing calls that report rounds with calls that do not. `step_id` is what orders
    them, and that ordering is the reason this is an event at all: a tracker entry carries no
    per-call identity, so a kit-side list could only have been matched to one by POSITION, which
    is unsound once dspy owns retries and `llm_query_batched` fans calls across threads.
    WITHOUT-then-WITH is the order used, so a reader taking "the last breakdown" cannot pass by
    picking the first."""
    from rlm_harness.trace import EVENT_API_ROUNDS, TraceRecorder, load_events

    path = str(tmp_path / "mixed.jsonl")
    with TraceRecorder(path, run_id="r"):
        sdk_returning({"input_tokens": 7, "output_tokens": 1})
        rlm_harness.ClaudeAgentLM("sonnet")(prompt="hi")
        sdk_returning(_usage_with_rounds([_ROUND]))
        rlm_harness.ClaudeAgentLM("sonnet")(prompt="hi")
        sdk_returning({"input_tokens": 9, "output_tokens": 1})
        rlm_harness.ClaudeAgentLM("sonnet")(prompt="hi")
    events = [e for e in load_events(path) if e["type"] == EVENT_API_ROUNDS]
    assert len(events) == 1, "presence was lost: only the middle call reported a breakdown"
    assert events[0]["payload"]["rounds"] == [_ROUND]
    steps = [e["step_id"] for e in load_events(path)]
    assert steps == sorted(steps), "step_id stopped ordering the stream"


def test_a_multi_entry_breakdown_rides_through_in_order_and_unfiltered(sdk_returning, tmp_path):
    """Every other fixture here carries ONE round, so a guard that reordered or filtered entries
    would be invisible. This is the shape the field exists for: a server-side fallback puts the
    declined hop's `message` entry and the serving hop's `fallback_message` in the SAME request,
    which is exactly the case where "one `message` entry" is true and the totals still cover two
    hops. Order is load-bearing: the context reading is the LAST qualifying entry."""
    declined = {**_ROUND, "type": "message"}
    served = {"type": "fallback_message", "model": "claude-haiku-4-5-20251001",
              "input_tokens": 9, "cache_creation_input_tokens": 0,
              "cache_read_input_tokens": 1080, "output_tokens": 40}
    sdk_returning(_usage_with_rounds([declined, served]))
    _, events = _rounds_events(rlm_harness.ClaudeAgentLM("sonnet"), tmp_path)
    assert [e["payload"]["rounds"] for e in events] == [[declined, served]], "reordered or dropped"


def test_the_aggregate_is_the_three_TOP_LEVEL_input_fields_not_every_int_in_a_round():
    """A pin on THIS adapter's arithmetic over a fixture: NOT a claim that the SDK's top-level
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


def test_a_rejected_breakdown_is_logged_and_an_absent_one_is_not(sdk_returning, caplog):
    """Without this line, "no run has `api_rounds`" is indistinguishable between the SDK never
    reporting it, reporting it unusably, and the key having been RENAMED upstream: the last being
    the failure this project has paid for repeatedly. A JSON `null` stays silent on purpose: that
    IS absence, not a rejection."""
    import logging

    cases = [({"iterations": "nope"}, 1), ({"iterations": [1, 2]}, 1),
             # `[]` is the shape most likely to be rejected in practice. It is what the CLI's
             # usage accumulator seeds itself with, so it must be one of the LOGGED ones.
             ({"iterations": []}, 1), ({"iterations": None}, 0), ({}, 0)]
    for reported, expected in cases:
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="rlm_harness.claude_agent_lm"):
            sdk_returning({"input_tokens": 5, "output_tokens": 2, **reported})
            _complete(rlm_harness.ClaudeAgentLM("sonnet"))
        assert len(caplog.records) == expected, f"wrong DEBUG count for {reported}"




# --- what the adapter PASSES to the SDK, which nothing used to assert ------------------------
#
# The fixture's own note says a renamed kwarg stays green here, because `ClaudeAgentOptions`
# swallows anything. That understated the gap: no test asserted any VALUE either, so mutating
# `tools=[]` to a real toolset, or `setting_sources=[]` to the user's own settings, left the whole
# suite green -- and those two are the guarantee the module docstring and `docs/INVARIANTS.md`
# rest on ("rlm-harness's sandbox stays the only place code runs").


def test_a_completion_grants_the_sdk_no_tools_and_leaks_no_settings(sdk_returning):
    """The pure-completion guarantee, asserted on the VALUES the SDK receives.

    `tools=[]` empties the toolset (`allowed_tools` would merely auto-approve) and
    `setting_sources=[]` keeps the user's CLAUDE.md, settings and MCP servers out of a planner
    call. Both are load-bearing claims made in prose elsewhere; this is what makes them pins.
    """
    sdk = sdk_returning({"input_tokens": 1, "output_tokens": 1})
    _complete(rlm_harness.ClaudeAgentLM("sonnet"))
    assert sdk.captured_options["tools"] == [], "the SDK was handed a toolset"
    assert sdk.captured_options["setting_sources"] == [], "the user's settings could leak in"
    assert sdk.captured_options["env"] == {"CLAUDE_CODE_MAX_RETRIES": "2"}, "the CLI retry ladder moved"
    assert sdk.captured_options["model"] == "sonnet"


def test_a_structured_call_reaches_the_sdks_own_output_format(sdk_returning):
    """Structured output, end to end, in the shape the ENGINE receives rather than the one the
    adapter produces.

    1.17.0's migration broke this and nothing caught it: `_translate_response_format` read only a
    pydantic CLASS, which is what the kit's adapter injects, while dspy canonicalises that to a
    dict before the engine sees it. Every structured call silently got `output_format=None` and
    `max_turns=1`, which also made the SDK's validation round unreachable. The `response_format`
    knob on `_complete` existed for this test and was unused until it did.
    """
    sdk = sdk_returning({"input_tokens": 1, "output_tokens": 1})
    canonical = {"type": "json_schema", "name": "DSPyProgramOutputs", "strict": True,
                 "schema": {"type": "object", "properties": {"headline": {"type": "string"}}}}
    _complete(rlm_harness.ClaudeAgentLM("sonnet"), response_format=canonical)
    assert sdk.captured_options["output_format"] == {
        "type": "json_schema",
        "schema": {"type": "object", "properties": {"headline": {"type": "string"}}},
    }, "the schema did not reach the SDK"
    # dspy's own `name`/`strict` are not forwarded: they were never sent, and this is not the
    # place to guess what the SDK does with an unknown key.
    assert "name" not in sdk.captured_options["output_format"]
    # ...and the structured path gets the headroom its comment describes, which `max_turns=1`
    # silently removed.
    assert sdk.captured_options["max_turns"] == 8


def test_a_plain_completion_asks_for_no_output_format_and_one_turn(sdk_returning):
    """The other arm: without this the test above passes for a build that sets `max_turns=8`
    unconditionally, and the one-turn plain completion is half the politeness policy."""
    sdk = sdk_returning({"input_tokens": 1, "output_tokens": 1})
    _complete(rlm_harness.ClaudeAgentLM("sonnet"))
    assert sdk.captured_options["output_format"] is None
    assert sdk.captured_options["max_turns"] == 1


@pytest.mark.parametrize("response_format", [
    pytest.param({"type": "json_object"}, id="json-object-fallback"),
    pytest.param({"type": "json_schema", "schema": {}}, id="empty-schema"),
])
def test_a_response_format_with_no_usable_schema_asks_for_none(sdk_returning, response_format):
    """A stock adapter's `{"type": "json_object"}` has no SDK equivalent and must not become a
    schema-validated call with an empty schema. Two shapes, one outcome.

    A `json_schema` with NO `schema` key is deliberately not among them: `lm15.Config` refuses it
    outright (`response_format json_schema requires a 'schema' object`), so it cannot reach the
    engine and a kit-side guard for it would be unreachable code with a test to match.
    """
    sdk = sdk_returning({"input_tokens": 1, "output_tokens": 1})
    _complete(rlm_harness.ClaudeAgentLM("sonnet"), response_format=response_format)
    assert sdk.captured_options["output_format"] is None
    assert sdk.captured_options["max_turns"] == 1


# --- paths the new tests did not reach, each of which a mutation would have survived ----------


def test_the_async_engine_is_wired_and_not_just_constructed(sdk_returning):
    """The planner seat, i.e. the PRIMARY RLM route, and entirely new surface in 1.17.0.

    `dspy.LM` takes a sync `engine=` and an async `async_engine=`; the sync one alone would leave
    every `aforward` call going somewhere else. Coverage showed `acomplete` and
    `_AsyncClaudeAgentEngine.complete` unreached by any test, so nothing distinguished "wired" from
    "constructed".
    """
    import asyncio

    sdk_returning({"input_tokens": 1, "output_tokens": 1}, text="async ok")
    lm = rlm_harness.ClaudeAgentLM("sonnet")
    assert asyncio.run(lm.acall(prompt="hi")) == ["async ok"]


@pytest.mark.parametrize(("text", "sdk_calls"), [
    pytest.param("429 rate limit exceeded", 2, id="rate-limited-retries"),
    pytest.param("failed to generate the delimiter", 1, id="ordinary-error-does-not"),
])
def test_the_rate_limit_rung_fires_for_a_rate_limit_shape_only(sdk_returning, monkeypatch, text, sdk_calls):
    """The politeness ladder's own rung: one retry after a backoff, for a rate-limit-SHAPED error
    only. `_RATE_LIMIT_RE` is phrase-level precisely so ordinary error text ("failed to generate
    the delimiter") does not buy a sleep and a second call, and neither arm had a test."""
    import rlm_harness.claude_agent_lm as cal

    monkeypatch.setattr(cal, "_BACKOFF_S", 0.0)   # the ladder, not the wait, is under test
    sdk = sdk_returning({"input_tokens": 1, "output_tokens": 1})
    calls = {"n": 0}

    async def query(*, prompt, options):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError(text)
        yield sdk.ResultMessage()

    monkeypatch.setattr(sdk, "query", query)
    if sdk_calls == 1:
        with pytest.raises(RuntimeError, match="delimiter"):
            _complete(rlm_harness.ClaudeAgentLM("sonnet"))
    else:
        _complete(rlm_harness.ClaudeAgentLM("sonnet"))
    assert calls["n"] == sdk_calls


def test_a_non_text_part_is_refused_rather_than_stringified(sdk_returning):
    """The SDK takes ONE text prompt, so a non-text part has nowhere to go. The predecessor path
    ran `str(content)` over whatever it was handed, which sent a content array's repr to the model
    as if it were the prompt: a loud refusal names the part, a repr looks like a bad answer."""
    import dspy

    sdk_returning({"input_tokens": 1, "output_tokens": 1})
    T = dspy.lm15
    request = T.Request(
        model="claude-agent-sdk/sonnet",
        messages=[T.Message(role="user", parts=[T.BinaryPart(data="AA==", media_type="image/png")])],
        config=T.Config(),
    )
    with pytest.raises(TypeError, match="cannot send a .* part"):
        rlm_harness.ClaudeAgentLM("sonnet").engine.complete(request)


def test_a_recorder_that_raises_does_not_break_the_call(sdk_returning, monkeypatch, tmp_path):
    """Observability never breaks a run. The breakdown is recorded mid-call now, so a recorder
    that throws sits directly in the completion path."""
    from rlm_harness.trace import TraceRecorder

    sdk_returning(_usage_with_rounds([_ROUND]))
    monkeypatch.setattr(
        TraceRecorder, "record_api_rounds",
        lambda self, model, rounds: (_ for _ in ()).throw(RuntimeError("trace is broken")),
    )
    with TraceRecorder(str(tmp_path / "t.jsonl"), run_id="r"):
        response = _complete(rlm_harness.ClaudeAgentLM("sonnet"))
    assert response.usage.input_tokens == 1082, "a broken recorder took the call down with it"
