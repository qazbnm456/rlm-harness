import pytest

from rlm_harness.sub_lm import (
    SubLMValidationError,
    bind_recorder_to_sub_lm,
    intercept_sub_lm,
    model_as_tool,
)
from rlm_harness.trace import (
    EVENT_SUB_CALL,
    EVENT_TOOL_CALL,
    TraceRecorder,
    group_by_run,
    load_events,
)


class FakeLM:
    """Stands in for a dspy.LM: callable, returns a list of completions."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.model = "fake/local"
        self.kwargs = {}
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return [self._responses[min(self.calls - 1, len(self._responses) - 1)]]


# intercept_sub_lm builds a subclass of dspy.LM, so these tests need dspy.
dspy = pytest.importorskip("dspy")


def test_postprocess_applied():
    base = FakeLM(["  hello  "])
    mw = intercept_sub_lm(base, postprocessors=[str.strip])
    assert mw(prompt="x") == ["hello"]


def test_validation_failure_then_retry_succeeds():
    base = FakeLM(["bad", "good"])

    def must_be_good(text):
        return None if text == "good" else "not good"

    mw = intercept_sub_lm(base, validators=[must_be_good], max_retries=3)
    assert mw(prompt="x") == ["good"]
    assert base.calls == 2


def test_validation_exhausts_budget_raises():
    base = FakeLM(["bad"])
    mw = intercept_sub_lm(base, validators=[lambda t: "always bad"], max_retries=2)
    with pytest.raises(SubLMValidationError):
        mw(prompt="x")
    assert base.calls == 2


def test_sub_call_events_recorded(tmp_path):
    base = FakeLM(["  raw  "])
    mw = intercept_sub_lm(base, postprocessors=[str.strip], name="local")
    path = str(tmp_path / "t.jsonl")
    with TraceRecorder(path, run_id="r1"):
        mw(prompt="x")
    subs = [e for e in load_events(path) if e["type"] == EVENT_SUB_CALL]
    assert len(subs) == 1
    # sub_call payload labels the role explicitly (kind) + the wrapper name.
    assert subs[0]["payload"]["kind"] == "sub_lm"
    assert subs[0]["payload"]["name"] == "local"
    assert subs[0]["payload"]["raw"] == "  raw  "
    assert subs[0]["payload"]["processed"] == "raw"
    assert subs[0]["payload"]["input"] == "x"  # escalation input captured for RL


def test_model_as_tool_records_and_returns(tmp_path):
    base = FakeLM(["answer from B"])
    tool = model_as_tool("modelB", base, description="Ask model B.")
    assert tool.__name__ == "query_modelB"
    path = str(tmp_path / "t.jsonl")
    with TraceRecorder(path, run_id="r1"):
        out = tool("question")
    assert out == "answer from B"
    calls = [e for e in load_events(path) if e["type"] == EVENT_TOOL_CALL]
    assert calls[0]["payload"]["tool"] == "model:modelB"


def test_bind_recorder_records_batched_escalations_else_lost(tmp_path):
    # Mimics dspy.RLM.llm_query_batched: the intercepted sub-LM called from ThreadPoolExecutor workers.
    # Worker threads do NOT inherit the recorder ContextVar, so the UNBOUND sub-LM records nothing there
    # (the bug: lifeline under-counts); the per-run binding re-establishes the recorder per call.
    from concurrent.futures import ThreadPoolExecutor

    from rlm_harness.sub_lm import bind_recorder_to_sub_lm

    inner = intercept_sub_lm(FakeLM(["A"]), name="lifeline")
    path = str(tmp_path / "t.jsonl")
    with TraceRecorder(path, run_id="r1") as rec:
        with ThreadPoolExecutor(max_workers=3) as ex:          # CONTROL: unbound → recorded nothing
            list(ex.map(lambda p: inner(prompt=p), ["a", "b", "c"]))
        n_unbound = sum(1 for e in load_events(path) if e["type"] == EVENT_SUB_CALL)

        bound = bind_recorder_to_sub_lm(inner, rec)
        with ThreadPoolExecutor(max_workers=3) as ex:          # FIX: bound → each worker re-establishes it
            list(ex.map(lambda p: bound(prompt=p), ["d", "e", "f"]))
        n_total = sum(1 for e in load_events(path) if e["type"] == EVENT_SUB_CALL)

    assert n_unbound == 0                  # the bug: batched escalations from worker threads were lost
    assert n_total - n_unbound == 3        # the fix: all 3 recorded, with the right label
    subs = [e for e in load_events(path) if e["type"] == EVENT_SUB_CALL]
    assert {s["payload"]["name"] for s in subs} == {"lifeline"}


def test_bind_recorder_to_sub_lm_is_a_noop_without_a_recorder():
    from rlm_harness.sub_lm import bind_recorder_to_sub_lm

    inner = FakeLM(["x"])
    assert bind_recorder_to_sub_lm(inner, None) is inner   # passthrough, no wrapper allocated


# ---- shape preservation: hand dspy back whatever dspy handed us --------------------------------
#
# `RLM._query_lm` accepts dspy's TYPED response OR the legacy `list[str | dict]`. The wrapper used
# to collapse anything non-list into `[outputs]`, so a typed response became `[response]` and dspy
# raised "Sub-LM response must contain text": invisible on the default path and fatal under
# `dspy.context(experimental=True)`.
#
# The typed class is `dspy.lm15.Response`, with a single `.message`. These helpers build it from
# dspy's own types, so the test exercises the real class dspy branches on rather than a shape this
# file invented.

_TYPED_CLS = dspy.lm15.Response


def _typed(*texts, extra_parts=()):
    """dspy's typed response carrying `texts` as separate text parts."""
    from dspy.lm15 import Message, Response, TextPart, Usage

    parts = tuple(TextPart(text=t) for t in texts) + tuple(extra_parts)
    return Response(id=None, model="m", message=Message(role="assistant", parts=parts),
                    finish_reason="stop", usage=Usage())


def _thinking_part(text):
    """A non-text part, for asserting substitution leaves it alone.

    It carries a `.text` field of its OWN, which is exactly why the shim discriminates on the
    part's `type` tag and never on `hasattr(part, "text")`.
    """
    from dspy.lm15 import ThinkingPart

    return ThinkingPart(text=text)


def _as_dspy_reads_it(response):
    """dspy's `RLM._query_lm` return handling, mirrored: the contract the wrapper must satisfy."""
    if _TYPED_CLS is not None and isinstance(response, _TYPED_CLS):
        text = response.text
    elif isinstance(response, list) and response:
        first = response[0]
        text = first.get("text") if isinstance(first, dict) else first
    else:
        raise TypeError(f"Sub-LM must return a Response or a non-empty list, got {type(response).__name__}.")
    if not isinstance(text, str):
        raise TypeError(f"Sub-LM response must contain text, got {type(text).__name__}.")
    return text


class ShapedLM:
    """A sub-LM that returns exactly the object it was given."""

    def __init__(self, response):
        self._response = response
        self.model = "shaped/lm"
        self.kwargs = {}

    def __call__(self, *args, **kwargs):
        return self._response


def test_a_typed_response_survives_dspys_own_return_handling():
    """THE regression: a typed response wrapped in a list made dspy raise."""
    wrapped = intercept_sub_lm(ShapedLM(_typed("hi there")))
    assert _as_dspy_reads_it(wrapped(prompt="q")) == "hi there"


def test_a_no_op_pipeline_returns_the_base_objects_IDENTITY():
    """Not an equal reconstruction: the same object. This is what makes auto-wrapping safe: a
    sub-LM the kit wrapped on the caller's behalf must be indistinguishable from the bare one."""
    for response in (_typed("x"), ["x"], [{"text": "x"}], ["a", "b"]):
        base = ShapedLM(response)
        assert intercept_sub_lm(base)(prompt="q") is response


def test_a_shape_the_shim_does_not_recognise_is_returned_UNTOUCHED():
    """The loud-error-to-silent-empty regression, pinned. Rebuilding an unrecognised shape as
    `[""]` would hand the planner an empty completion that dspy would otherwise have rejected,
    and it would land in the RL data as a real escalation answer. dspy must get to raise."""
    unrecognised = [
        _typed(extra_parts=[_thinking_part("reasoning only, no answer")]),
        "a bare string",
        [],
        None,
    ]
    for response in unrecognised:
        base = ShapedLM(response)
        assert intercept_sub_lm(base)(prompt="q") is response


def test_substitution_replaces_ALL_text_of_a_multi_part_response():
    """A typed response's `.text` JOINS every text part, so replacing only the first leaves the
    rest appended: it round-tripped to "ABB". dspy emits one text part per content item, so any
    provider returning a content array produces several.

    The postprocessor has to CHANGE the text, and that is the whole reason this reads `str.lower`
    rather than `str.upper`. With `str.upper` over parts "A" and "B" the pipeline is a no-op, the
    shim correctly returns dspy's own object by identity, no rebuild happens at all, and the
    assertion passes without ever exercising the substitution it is named for.

    The JOINER is read off dspy rather than written down: hardcoding it would assert the kit against
    one dspy instead of against the property that matters, which is that no un-substituted tail
    survives."""
    joined = _typed("A", "B").text
    assert joined.lower() != joined, "the postprocessor must change the text or this proves nothing"
    base = ShapedLM(_typed("A", "B"))
    out = intercept_sub_lm(base, postprocessors=[str.lower])(prompt="q")
    assert out.text == joined.lower(), "an un-substituted text part survived the rebuild"
    assert len(_parts_of(out)) == 1, "the later text part was kept and will be appended to `.text`"


def _parts_of(response):
    """The parts of a typed response's message."""
    return list(response.message.parts)


def test_substitution_leaves_non_text_parts_and_sibling_fields_alone():
    original = _typed("answer", extra_parts=[_thinking_part("private")])
    out = intercept_sub_lm(ShapedLM(original), postprocessors=[str.upper])(prompt="q")
    assert out.text == "ANSWER"
    assert any(getattr(p, "type", None) == "thinking" for p in _parts_of(out))
    assert out.model == original.model
    assert original.text == "answer", "the caller's object was mutated"


def test_model_as_tool_reads_a_typed_response_too():
    """`outputs[0]` on a typed response would hand the model `str(response)`, the whole repr, and
    write it to the trace as the tool's result."""
    tool = model_as_tool("l", ShapedLM(_typed("THE ANSWER")))
    assert getattr(tool, "func", tool)(prompt="q") == "THE ANSWER"


# ---- automatic sub_call recording --------------------------------------------------------------


def test_auto_wrap_records_a_plain_lm_and_leaves_a_wrapped_one_alone():
    from rlm_harness.sub_lm import _ensure_sub_call_recording

    plain = FakeLM(["answer"])
    assert _ensure_sub_call_recording(plain) is not plain      # wrapped
    already = intercept_sub_lm(FakeLM(["answer"]), name="lifeline")
    assert _ensure_sub_call_recording(already) is already      # left alone, no double-recording
    assert _ensure_sub_call_recording(None) is None


def test_the_marker_probe_is_identity_not_truthiness():
    """`getattr` on a mock manufactures a truthy attribute for ANY name, so a truthiness probe
    would decide a mock "already records" and skip it: recreating, one layer up, the exact
    absent-event failure this feature exists to remove."""
    from unittest.mock import MagicMock

    from rlm_harness.sub_lm import _ensure_sub_call_recording

    mock = MagicMock()
    assert bool(getattr(mock, "records_sub_call", False)) is True   # truthiness would skip it
    assert _ensure_sub_call_recording(mock) is not mock             # identity does not


def test_auto_wrap_never_raises_and_degrades_to_the_bare_lm():
    """Auto-wrapping is an observability convenience the caller never asked for. It must never be
    the reason a run fails to start, and the probe alone is not enough, because `intercept_sub_lm`
    reads `.model`/`.kwargs` off the base and dies one line after a successful probe."""
    from unittest.mock import Mock

    from rlm_harness.sub_lm import _ensure_sub_call_recording

    class LazyProxy:
        def __getattr__(self, name):
            raise RuntimeError("connection not established")

    for hostile in (Mock(), LazyProxy()):
        assert _ensure_sub_call_recording(hostile) is hostile        # returned bare, no exception


def test_the_auto_payload_matches_an_explicit_no_argument_wrap():
    """"No second payload shape" is the property that made auto-wrapping worth doing; pin it.
    Holds against an explicit wrap with NO `name=`, which is what the kit calls."""
    from rlm_harness.sub_lm import _ensure_sub_call_recording

    def emit(make, tmp):
        with TraceRecorder(tmp, run_id="r") as rec:  # noqa: F841
            make(FakeLM(["answer"]))(prompt="q")
        return [e["payload"] for e in load_events(tmp) if e["type"] == EVENT_SUB_CALL]

    import tempfile
    from pathlib import Path

    d = Path(tempfile.mkdtemp())
    auto = emit(_ensure_sub_call_recording, str(d / "a.jsonl"))
    explicit = emit(intercept_sub_lm, str(d / "e.jsonl"))
    assert auto == explicit and len(auto) == 1

    # ...and the same holds one layer out, in the record a trainer actually consumes. This is the
    # half that matters: `export_actions` is the per-ACTION export an RL trainer does credit
    # assignment over, and before 1.7.0 an unwrapped consumer's escalations produced no `kind="sub"`
    # record at all.
    from rlm_harness.dataset import export_actions

    def actions(path):
        return [r for r in export_actions(group_by_run(load_events(path))) if r["kind"] == "sub"]

    assert actions(str(d / "a.jsonl")) == actions(str(d / "e.jsonl"))
    assert len(actions(str(d / "a.jsonl"))) == 1


def test_the_wrappers_copy_updates_the_object_that_MAKES_the_request():
    """The second half of the 1.14.0 copy fix, and it was wrong on EVERY dspy version.

    The inherited `copy()` updated the WRAPPER's decorative `kwargs` while `_base`, the object that
    actually makes the request, was shared by reference and never saw the update. So
    `copy(rollout_id=1)` produced an LM that reported a rollout id it would not send.
    """
    base = FakeLM(["hi"])
    dup = intercept_sub_lm(base).copy(rollout_id=1)
    assert dup.__dict__["_base"] is not base, "the base was shared by reference"
    assert dup.__dict__["_base"].kwargs["rollout_id"] == 1, "the update never reached the caller"
    assert base.kwargs.get("rollout_id") is None, "the original base was mutated"
    assert dup.kwargs["rollout_id"] == 1, "the wrapper's mirror disagrees with its base"
    assert dup.records_sub_call is True, "the copy stopped recording escalations"


def test_the_wrapper_survives_copy_deepcopy_and_dspys_own_copy():
    """`__getattr__` delegates through `_base`, and `copy`/`deepcopy`/`pickle` rebuild an instance
    WITHOUT calling `__init__`, so reading it as `self._base` re-enters the method forever.
    `dspy.BaseLM.copy()` does exactly that and is the documented way to get a rollout-id variant,
    so the recursion fires on a supported path. Since 1.7.0 the kit auto-wraps every sub-LM, which
    would have made it universal rather than opt-in."""
    import copy as copy_module

    base = FakeLM(["hi"])
    wrapped = intercept_sub_lm(base)
    assert type(copy_module.copy(wrapped)) is type(wrapped)
    assert type(copy_module.deepcopy(wrapped)) is type(wrapped)
    assert wrapped.copy() is not None          # dspy.BaseLM.copy: unguarded copy_module.copy
    assert wrapped.records_sub_call is True    # ...and the marker still resolves normally


def test_delegation_reaches_the_base_lms_own_attributes():
    """`__init__` copies only `model`/`kwargs`. Everything else a real dspy.LM carries has to come
    through delegation, or "observationally identical to the bare one" is false for attribute
    access, which matters now that the kit substitutes this object automatically."""
    from dspy.utils.dummies import DummyLM

    base = DummyLM([{"answer": "a"}] * 3)
    wrapped = intercept_sub_lm(base)
    for attr in ("history", "cache", "num_retries", "callbacks", "model_type"):
        assert hasattr(base, attr) and hasattr(wrapped, attr), attr
    with pytest.raises(AttributeError):
        _ = wrapped.definitely_not_an_attribute_on_either


def test_the_typed_shape_the_kit_supports_is_the_one_dspys_OWN_SOURCE_branches_on():
    """`_as_dspy_reads_it` above is a MIRROR of `RLM._query_lm`, and a mirror can drift.

    This pins the mirror against dspy's real source instead of against a shape this file invented:
    the class the kit's shims read is the class dspy's own `isinstance` names, and a string call's
    list is the other shape it accepts.
    """
    import inspect

    from dspy.predict import rlm as rlm_module

    source = inspect.getsource(rlm_module)
    assert "def _query_lm" in source
    body = source[source.index("def _query_lm"):]
    body = body[:body.index("if not isinstance(text, str)")]
    assert "lm15.Response" in body, (
        "dspy's _query_lm no longer branches on lm15.Response; re-point "
        "`_dspy_compat.sub_lm_response_text` and this mirror at whatever it accepts now."
    )
    assert _TYPED_CLS is not None, "the kit resolved no typed class from this dspy"
    assert _TYPED_CLS.__name__ in body, (
        f"the kit resolves {_TYPED_CLS.__name__} but dspy's _query_lm branches on something else"
    )


def test_a_typed_response_round_trips_through_the_wrapper_and_the_trace(tmp_path):
    """The regression itself, with the typed object built the way dspy's own types build one:
    the wrapper must hand dspy back the SHAPE it was given, and record the text either way."""
    from rlm_harness.sub_lm import _ensure_sub_call_recording

    base = ShapedLM(_typed("escalated"))
    path = str(tmp_path / "t.jsonl")
    with TraceRecorder(path, run_id="r"):
        out = _ensure_sub_call_recording(base)(prompt="q")
    assert isinstance(out, _TYPED_CLS)
    assert _as_dspy_reads_it(out) == "escalated"
    payload = [e["payload"] for e in load_events(path) if e["type"] == EVENT_SUB_CALL][0]
    assert isinstance(payload["raw"], str) and payload["raw"], "raw was not the completion text"


# --- a FAILED escalation is still an escalation (1.13.0) ------------------------------------

class _DeadLM:
    """A sub-LM whose provider is down, the shape that left no trace at all before 1.13.0."""

    model = "openai/sub-model"
    kwargs: dict = {}

    def __init__(self):
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        raise RuntimeError("litellm.BadRequestError: no healthy deployments")


def _events(path):
    import json
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_a_provider_failure_records_the_escalation_and_still_raises(tmp_path):
    """The record used to sit AFTER the call, so a raise skipped it and the run read as one where
    the planner never escalated. Observed on a real deployment whose sub model was pulled from the
    proxy: runs escalated repeatedly, every call failed, and the traces showed zero escalations."""
    dead = _DeadLM()
    path = tmp_path / "t.jsonl"
    with TraceRecorder(str(path), run_id="r"), pytest.raises(RuntimeError, match="no healthy"):
        intercept_sub_lm(dead)(prompt="summarise this")

    subs = [e for e in _events(path) if e["type"] == "sub_call"]
    assert len(subs) == 1, "a refused escalation must still be one recorded escalation"
    payload = subs[0]["payload"]
    assert payload["cause"] == "endpoint"
    assert "no healthy deployments" in payload["error"]
    assert payload["raw"] is None and payload["processed"] is None
    assert payload["input"] == "summarise this", "the escalation prompt is the RL-relevant half"


def test_the_provider_exception_is_never_replaced_by_a_recording_fault(tmp_path):
    """A bookkeeping bug must not become the error the caller sees, the same rule
    `TraceRecorder.__exit__` follows."""
    class _BrokenRecorder(TraceRecorder):
        def record(self, event_type, *a, **k):
            if event_type == EVENT_SUB_CALL:      # only the write this test is about
                raise ValueError("recorder is broken")
            return super().record(event_type, *a, **k)

    with _BrokenRecorder(str(tmp_path / "t.jsonl"), run_id="r"), \
            pytest.raises(RuntimeError, match="no healthy"):
        intercept_sub_lm(_DeadLM())(prompt="x")


def test_cause_separates_a_dead_provider_from_a_rejected_output(tmp_path):
    """`error` alone cannot tell them apart, and only one of the two is the model's doing."""
    path = tmp_path / "t.jsonl"
    lm = FakeLM(["fine"])
    with TraceRecorder(str(path), run_id="r"):
        intercept_sub_lm(lm)(prompt="q")
    assert [e["payload"]["cause"] for e in _events(path) if e["type"] == "sub_call"] == ["ok"]

    path2 = tmp_path / "t2.jsonl"
    rejecting = FakeLM(["bad", "bad"])
    with TraceRecorder(str(path2), run_id="r"), pytest.raises(SubLMValidationError):
        intercept_sub_lm(rejecting, validators=[lambda t: "nope" if t == "bad" else None],
                         max_retries=2)(prompt="q")
    causes = [e["payload"]["cause"] for e in _events(path2) if e["type"] == "sub_call"]
    assert causes == ["invalid", "invalid"]


# ---- dspy 3.5's call shapes, against a real dspy.LM on the fake OpenAI-compatible server ---------


def _served_lm(fake_openai):
    from rlm_harness import _dspy_compat

    return dspy.LM(_dspy_compat.endpoint_model("openai", "sub-model"), base_url=fake_openai.base_url,
                   api_key="k", cache=False, **_dspy_compat.endpoint_lm_kwargs())


def _sub_calls(path):
    return [e["payload"] for e in load_events(path) if e["type"] == "sub_call"]


def test_a_request_prompt_is_intercepted_and_recorded_as_its_text(fake_openai, tmp_path):
    """`llm_query` with a list prompt calls the sub-LM with an lm15 `Request`, built from the
    wrapper's own model and defaults. The wrapper hands it to the base unchanged, returns the typed
    response untouched, and records the prompt's text."""
    from dspy.adapters.base import Prompt
    from dspy.clients.requests import build_request
    from dspy.lm15 import Message, TextPart

    fake_openai.script = [{"role": "assistant", "content": "an answer"}]
    wrapped = intercept_sub_lm(_served_lm(fake_openai))
    request = build_request(wrapped, Prompt(messages=(Message.user((TextPart("part one "),
                                                                     TextPart("part two"))),)))
    path = str(tmp_path / "t.jsonl")
    with TraceRecorder(path, run_id="r"):
        response = wrapped(request)
    assert isinstance(response, dspy.lm15.Response) and response.text == "an answer"
    (call,) = _sub_calls(path)
    assert (call["input"], call["raw"], call["cause"]) == ("part one part two", "an answer", "ok")
    assert fake_openai.requests[0]["body"]["model"] == "sub-model"


def test_the_async_seat_is_intercepted_and_recorded_too(fake_openai, tmp_path):
    import asyncio

    fake_openai.script = [{"role": "assistant", "content": "async answer"}]
    wrapped = intercept_sub_lm(_served_lm(fake_openai), postprocessors=[str.upper])
    path = str(tmp_path / "t.jsonl")
    with TraceRecorder(path, run_id="r") as rec:
        bound = bind_recorder_to_sub_lm(wrapped, rec)
        out = asyncio.run(bound.acall("hi"))
    assert out == ["ASYNC ANSWER"]
    (call,) = _sub_calls(path)
    assert (call["input"], call["raw"], call["processed"]) == ("hi", "async answer", "ASYNC ANSWER")


def test_a_string_call_that_carried_reasoning_keeps_dspys_shape(fake_openai):
    """A string call to a reasoning model returns `[{"text", "reasoning_content"}]`; a post-processed
    answer replaces the text and keeps the dict, so the reasoning still reaches dspy."""
    fake_openai.script = [{"role": "assistant", "content": "answer", "reasoning_content": "why"}]
    out = intercept_sub_lm(_served_lm(fake_openai), postprocessors=[str.upper])("hi")
    assert out == [{"text": "ANSWER", "reasoning_content": "why"}]


def test_the_bound_async_seat_records_from_a_thread_with_no_recorder(fake_openai, tmp_path):
    """dspy may reach the sub-LM from a worker thread, which starts with an empty context: the
    recorder binding is what makes that call record, on the async seat as on the sync one."""
    import asyncio
    import threading

    fake_openai.script = [{"role": "assistant", "content": "from a worker"}]
    path = str(tmp_path / "t.jsonl")
    with TraceRecorder(path, run_id="r") as rec:
        bound = bind_recorder_to_sub_lm(intercept_sub_lm(_served_lm(fake_openai)), rec)
        worker = threading.Thread(target=lambda: asyncio.run(bound.acall("hi")))
        worker.start()
        worker.join()
    (call,) = _sub_calls(path)
    assert call["raw"] == "from a worker"
