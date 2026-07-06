"""Integration test against a real dspy.RLM (no LLM API, no Deno sandbox).

Unit tests elsewhere use fakes. This one wires our RLMTask through the *real*
dspy.RLM constructor to catch signature/kwarg/interpreter drift between rlm-kit
and the installed dspy. It does not call forward() (that needs a paid LLM and a
Deno sandbox), so it stays free and offline. Skipped if dspy is absent.
"""

import pytest

dspy = pytest.importorskip("dspy")

from pydantic import BaseModel  # noqa: E402

from rlm_kit import RLMConfig, RLMTask  # noqa: E402
import rlm_kit.runtime as rt  # noqa: E402
from rlm_kit.tools import make_schema_validator  # noqa: E402


class Finding(BaseModel):
    title: str
    severity: str


def _configure_with_dummy(interpreter="mock"):
    from dspy.utils.dummies import DummyLM

    dummy = DummyLM([{"reasoning": "r", "finding": "{}"}])
    cfg = RLMConfig(main_model="x", sub_model="x", interpreter=interpreter)
    rt._STATE.configured = True
    rt._STATE.config = cfg
    rt._STATE.sub_lm = dummy
    dspy.configure(lm=dummy)
    return dummy


def test_rlmtask_builds_real_dspy_rlm():
    dummy = _configure_with_dummy()

    class T(RLMTask):
        signature = "evidence: str -> finding: Finding"
        output_field = "finding"
        output_model = Finding
        instructions = "triage"
        tools = [make_schema_validator(Finding)]

    rlm = T()._build_rlm()
    assert isinstance(rlm, dspy.RLM)
    assert rlm.sub_lm is dummy
    assert "finding" in rlm.signature.output_fields
    # Budget kwargs were accepted by the real constructor (no TypeError fallback).
    assert rlm.max_iterations == 10
    assert rlm.max_llm_calls == 30
    assert rlm.max_output_chars == 10_000


def test_custom_output_type_resolves_without_frame_help():
    """_build_rlm must resolve the signature's custom output type via custom_types,
    not by dspy walking the call stack. We use a dynamically-built model whose NAME
    ('DynReportXYZ') is a bareword in no frame's globals or locals, so call-stack
    resolution cannot find it — only the explicit output_model binding can."""
    from pydantic import create_model

    _configure_with_dummy()
    DynModel = create_model("DynReportXYZ", x=(int, ...))

    # Contrast: dspy alone cannot resolve the name from the call stack here.
    with pytest.raises(Exception):
        dspy.Signature("q: str -> y: DynReportXYZ")

    class T(RLMTask):
        signature = "q: str -> y: DynReportXYZ"
        output_field = "y"
        output_model = DynModel
        instructions = "Produce the output."

    rlm = T()._build_rlm()  # passes custom_types={'DynReportXYZ': DynModel}
    assert isinstance(rlm, dspy.RLM)
    assert "y" in rlm.signature.output_fields


def test_custom_output_type_resolves_even_without_instructions():
    """dspy drops custom_types when instructions is None (it re-parses the signature
    without them); _build_rlm must defend against that so a task with an output_model
    but no instructions still resolves its type."""
    from pydantic import create_model

    _configure_with_dummy()
    DynModel = create_model("DynNoInstr", x=(int, ...))

    class T(RLMTask):
        signature = "q: str -> y: DynNoInstr"
        output_field = "y"
        output_model = DynModel
        # deliberately no instructions

    rlm = T()._build_rlm()
    assert isinstance(rlm, dspy.RLM)
    assert "y" in rlm.signature.output_fields


def test_intercepted_sub_lm_is_accepted_as_sub_lm():
    """An intercepted sub-LM must be a real dspy.LM usable as RLM.sub_lm."""
    _configure_with_dummy()
    from rlm_kit import intercept_sub_lm

    base = dspy.utils.dummies.DummyLM([{"text": "ok"}])
    mw = intercept_sub_lm(base, postprocessors=[str.strip], name="mw")
    assert isinstance(mw, dspy.LM)

    class T(RLMTask):
        signature = "q: str -> a: str"
        output_field = "a"

    rlm = T(sub_lm=mw)._build_rlm()
    assert rlm.sub_lm is mw


def test_build_adapter_chat_disables_json_fallback():
    """The portable "chat" adapter must NOT fall back to JSONAdapter — that fallback
    silently re-emits response_format=json_object, which strict endpoints (vLLM) reject."""
    a = rt._build_adapter("chat")
    assert isinstance(a, dspy.ChatAdapter) and not isinstance(a, dspy.JSONAdapter)
    assert a.use_json_adapter_fallback is False


def test_main_step_timer_captures_only_root_planner_turns():
    """The per-turn parse callback feeds the recorder ONLY for ROOT planner turns (parses carrying
    both `reasoning` and `code`); a lifeline parse (no code) or the extract-fallback parse (output
    fields, no code) must not be mistaken for a turn."""
    from rlm_kit.task import _MainStepTimer

    captured: list = []

    class _Rec:
        def note_main_step(self, reasoning, ts=None):
            captured.append(reasoning)

    timer = _MainStepTimer(_Rec())
    timer.on_adapter_parse_end("c1", {"reasoning": "plan A", "code": "x = 1"})   # planner turn ✓
    timer.on_adapter_parse_end("c2", {"answer": "42"})                          # extract/lifeline ✗
    timer.on_adapter_parse_end("c3", {"reasoning": "no code field"})            # not a turn ✗
    assert captured == ["plan A"]


class _Out(BaseModel):
    x: int


def _task_with_fake_rlm(pred):
    """An RLMTask whose RLM is a stub returning `pred` from aforward (no LLM / sandbox)."""

    class T(RLMTask):
        signature = "q: str -> answer: _Out"
        output_field = "answer"
        output_model = _Out

    task = T()

    class _FakeRLM:
        async def aforward(self, **kw):
            return pred

    task._build_rlm = lambda: _FakeRLM()
    return task


def test_arun_records_trajectory_on_failure(tmp_path):
    # A run that never produces a coercible result must STILL record the last attempt's trajectory,
    # so a FAILED run is navigable/debuggable — but no result event, so it stays correctly "failed".
    import asyncio
    import types

    from rlm_kit._retry import RLMTaskError
    from rlm_kit.trace import EVENT_MAIN_STEP, EVENT_RESULT, TraceRecorder, load_events

    _configure_with_dummy()
    pred = types.SimpleNamespace(
        trajectory=[{"reasoning": "t0", "code": "c0", "output": "o0"}],
        final_reasoning="gave up", answer="not-an-int")  # 'answer' can't coerce into _Out → retries fail
    task = _task_with_fake_rlm(pred)

    path = str(tmp_path / "trace.jsonl")
    with TraceRecorder(path, run_id="r1"):
        with pytest.raises(RLMTaskError):
            asyncio.run(task.arun(q="hi"))

    ev = load_events(path)
    assert any(e["type"] == EVENT_MAIN_STEP for e in ev)    # the failed run's turns ARE recorded now
    assert not any(e["type"] == EVENT_RESULT for e in ev)   # but NO result → still "failed" to readers


def test_arun_records_result_on_success(tmp_path):
    import asyncio
    import types

    from rlm_kit.trace import EVENT_MAIN_STEP, EVENT_RESULT, TraceRecorder, load_events

    _configure_with_dummy()
    pred = types.SimpleNamespace(
        trajectory=[{"reasoning": "t0", "code": "c0", "output": "o0"}],
        final_reasoning="done", answer={"x": 5})           # coerces into _Out → success
    task = _task_with_fake_rlm(pred)

    path = str(tmp_path / "trace.jsonl")
    with TraceRecorder(path, run_id="r1"):
        result = asyncio.run(task.arun(q="hi"))
    assert isinstance(result, _Out) and result.x == 5
    ev = load_events(path)
    assert any(e["type"] == EVENT_MAIN_STEP for e in ev)
    assert any(e["type"] == EVENT_RESULT for e in ev)       # success → result recorded as before


def test_build_adapter_json_and_default():
    assert isinstance(rt._build_adapter("json"), dspy.JSONAdapter)
    assert rt._build_adapter("default") is None  # leave dspy's stock adapter in place


def test_configure_chat_mode_disables_json_fallback():
    """`chat` mode must NOT fall back to JSONAdapter — that fallback silently re-emits
    response_format=json_object, which strict endpoints (vLLM) reject."""
    rt.configure(RLMConfig(main_model="openai/x", sub_model="openai/x",
                           interpreter="mock", adapter="chat"))
    assert isinstance(dspy.settings.adapter, dspy.ChatAdapter)
    assert dspy.settings.adapter.use_json_adapter_fallback is False


def test_lenient_json_adapter_recovers_braceless_object():
    """Schema-guided decoding (vLLM/NIM) sometimes drops the outer braces; the lenient
    adapter must still parse the object body, where stock JSONAdapter would raise."""
    sig = dspy.Signature("q: str -> reasoning: str, code: str")
    a = rt._LenientJSONAdapter()
    assert a.parse(sig, '"reasoning": "r", "code": "c"') == {"reasoning": "r", "code": "c"}
    # well-formed JSON still parses
    assert a.parse(sig, '{"reasoning": "r2", "code": "c2"}') == {"reasoning": "r2", "code": "c2"}


def test_lenient_adapter_promotes_reasoning_content_when_content_empty():
    """A REASONING root (qwen3 / deepseek / gpt-oss) sometimes emits the whole structured turn into
    `reasoning_content` and returns `content` (text) null. The adapter promotes it so the turn
    parses instead of dying on dspy's "empty or null response" check — this is what lets a reasoning
    model be the RLM root. Guarded: a normal output (text present) is NOT overridden, so a
    well-behaved model's native thinking stays discarded."""
    sig = dspy.Signature("q: str -> reasoning: str, code: str")
    a = rt._LenientJSONAdapter()
    # content empty + structured answer stuck in reasoning_content → recovered & parsed
    out = [{"text": None, "reasoning_content": '{"reasoning": "r", "code": "c"}'}]
    vals = a._call_postprocess(sig, sig, out, lm=None, lm_kwargs={})
    assert vals[0]["reasoning"] == "r" and vals[0]["code"] == "c"
    # a normal output (text present) wins — reasoning_content is ignored, native thinking discarded
    out2 = [{"text": '{"reasoning": "real", "code": "x"}',
             "reasoning_content": '{"reasoning": "IGNORED", "code": "y"}'}]
    vals2 = a._call_postprocess(sig, sig, out2, lm=None, lm_kwargs={})
    assert vals2[0]["reasoning"] == "real" and vals2[0]["code"] == "x"


def test_lenient_json_adapter_skips_wrap_for_braced_completion(monkeypatch):
    """The brace-wrap is only for a brace-LESS body. A completion that already starts with
    "{" but fails to parse (incomplete / missing a required field) must NOT be re-wrapped
    into "{{...}" — the original error stands. Otherwise we double the brace and obscure the
    real failure (a model emitting `{ "code": ... }` without the required reasoning field)."""
    from dspy.adapters.json_adapter import JSONAdapter
    from dspy.utils.exceptions import AdapterParseError

    seen = []

    def always_fail(self, signature, completion):
        seen.append(completion)
        raise AdapterParseError(adapter_name="JSONAdapter", signature=signature,
                                lm_response=completion, message="boom")

    monkeypatch.setattr(JSONAdapter, "parse", always_fail)
    sig = dspy.Signature("q: str -> reasoning: str, code: str")
    with pytest.raises(AdapterParseError):
        rt._LenientJSONAdapter().parse(sig, '{ "code": "x"')   # already starts with "{"
    assert seen == ['{ "code": "x"']   # only the original — no "{{"-wrapped retry
    # contrast: a brace-LESS body IS retried wrapped
    seen.clear()
    with pytest.raises(AdapterParseError):
        rt._LenientJSONAdapter().parse(sig, '"code": "x"')
    assert seen == ['"code": "x"', '{"code": "x"}']   # original, then brace-wrapped retry


def test_lenient_json_adapter_never_falls_back_to_bare_json_object():
    """Regression: when the json_schema call fails (e.g. a transient upstream 502), json mode
    must NOT degrade to bare ``json_object`` — vLLM/NIM reject it (400 "'json_object' requires a
    JSON schema"), which masks the real error and wastes the retry on a dead-on-arrival format.
    Stock JSONAdapter falls back; ``_LenientJSONAdapter`` must only ever send ``json_schema``
    — and it forces that form for ANY lm (here a plain dspy.LM whose
    ``supports_response_schema`` is False), so no special LM subclass is needed.
    Fails on the old code, which only overrode ``parse`` and inherited the fallback."""
    import asyncio

    class _RaisingLM(dspy.LM):
        def __init__(self):
            super().__init__("openai/x")
            self.seen = []

        async def acall(self, messages=None, **kw):  # noqa: ANN001
            self.seen.append(kw.get("response_format"))
            raise ConnectionError("simulated upstream 502")

    sig = dspy.Signature("q: str -> reasoning: str, code: str")
    lm = _RaisingLM()
    with pytest.raises(Exception):
        asyncio.run(rt._LenientJSONAdapter().acall(lm, {}, sig, [], {"q": "hi"}))
    assert lm.seen, "the adapter must attempt the json_schema call"
    assert all(
        not (isinstance(rf, dict) and rf.get("type") == "json_object") for rf in lm.seen
    ), "json mode must never degrade to bare json_object (vLLM/NIM reject it)"


def test_configure_defaults_to_json_mode():
    """Default adapter is "json": schema-guided structured output, which works on any
    structured-output endpoint (OpenAI-proper AND vLLM/NIM). No `adapter` passed → default."""
    rt.configure(RLMConfig(main_model="openai/x", sub_model="openai/x", interpreter="mock"))
    assert isinstance(dspy.settings.adapter, rt._LenientJSONAdapter)
    # the adapter forces json_schema, so the LM stays a plain dspy.LM (no special subclass)
    assert type(dspy.settings.lm) is dspy.LM


def test_configure_passes_max_tokens_to_lm():
    rt.configure(RLMConfig(main_model="openai/x", sub_model="openai/x",
                           interpreter="mock", max_tokens=2048))
    assert dspy.settings.lm.kwargs.get("max_tokens") == 2048


def test_configure_default_sends_generous_max_tokens():
    """Regression: the default config must SEND a generous max_tokens (not rely on the
    server's small default cap, which truncates a reasoning model's chain-of-thought before
    the answer → empty content). Fails on the old code, where max_tokens defaulted to None
    and nothing was sent."""
    rt.configure(RLMConfig(main_model="openai/x", sub_model="openai/x", interpreter="mock"))
    assert dspy.settings.lm.kwargs.get("max_tokens") == 8192


def test_configure_pins_openai_provider_when_base_url_set():
    """With a base_url (a custom OpenAI-compatible endpoint), the LM pins
    custom_llm_provider="openai" so a BARE model id ("qwen/qwen3-next") routes to base_url —
    litellm would otherwise read "qwen" as the provider and fail. No "openai/" prefix needed."""
    rt.configure(RLMConfig(main_model="qwen/qwen3-next", sub_model="qwen/qwen3-next",
                           interpreter="mock", base_url="https://endpoint.example/v1"))
    assert dspy.settings.lm.kwargs.get("custom_llm_provider") == "openai"


def test_configure_no_provider_pin_without_base_url():
    """Without a base_url (a direct provider, e.g. anthropic/claude), do NOT force the openai
    provider — let litellm parse the model's own provider prefix."""
    rt.configure(RLMConfig(main_model="openai/gpt-4o", sub_model="openai/gpt-4o", interpreter="mock"))
    assert "custom_llm_provider" not in dspy.settings.lm.kwargs
