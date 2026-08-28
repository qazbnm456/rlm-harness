"""Integration test against a real dspy.RLM (no LLM API, no Deno sandbox).

Unit tests elsewhere use fakes. This one wires our RLMTask through the *real*
dspy.RLM constructor to catch signature/kwarg/interpreter drift between rlm-harness
and the installed dspy. It does not call forward() (that needs a paid LLM and a
Deno sandbox), so it stays free and offline. Skipped if dspy is absent.
"""

import threading

import pytest

dspy = pytest.importorskip("dspy")

from pydantic import BaseModel

import rlm_harness.runtime as rt
from rlm_harness import RLMConfig, RLMTask, SandboxCancelled
from rlm_harness.tools import make_schema_validator


class Finding(BaseModel):
    title: str
    severity: str


def _configure_with_dummy(interpreter="mock"):
    from dspy.utils.dummies import DummyLM

    dummy = DummyLM([{"reasoning": "r", "finding": "{}"}])
    cfg = RLMConfig(main_model="x", sub_model="x", interpreter=interpreter, observe=False)
    rt.configure(cfg, main_lm=dummy, sub_lm=dummy)   # public injection seam — no _STATE poking
    return dummy


def _budget_attr(rlm, *names):
    """Read a budget cap off a built ``dspy.RLM`` under whichever name this dspy uses.

    Keeps these assertions about the CAP LANDING rather than about dspy's spelling of it.
    """
    for name in names:
        if hasattr(rlm, name):
            return getattr(rlm, name)
    raise AssertionError(
        f"dspy.RLM exposes none of {names!r} — it renamed a budget kwarg again; "
        f"add the new name to rlm_harness._dspy_compat._BUDGET_ALIASES."
    )


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
    # Read them by VALUE under whichever name the installed dspy uses — 3.3.0 renamed
    # `max_iterations` to `max_iters`. Asserting one hardcoded name would either fail on
    # a dspy that renamed it (a false alarm) or, worse, pass while the cap was silently
    # dropped. `_budget_attr` fails loudly if dspy exposes NEITHER name.
    assert _budget_attr(rlm, "max_iters", "max_iterations") == 10
    assert _budget_attr(rlm, "max_llm_calls") == 30
    assert _budget_attr(rlm, "max_output_chars") == 10_000


def test_build_rlm_describes_a_custom_interpreters_runtime_to_the_model(monkeypatch):
    """`_dspy_compat` tests prove the SHIM resolves a carrier; this proves `_build_rlm` actually
    EMITS it. Without this the wiring could be dropped and every dspy_compat test stays green
    while the model keeps being told it is running in Pyodide.

    Captured off the constructor rather than read back off the built RLM, because
    `rlm._interpreter_factory` is a dspy internal and asserting on those is exactly what
    `_dspy_compat` exists to avoid.
    """
    _configure_with_dummy()
    captured = {}
    real_init = dspy.RLM.__init__

    def _spy(self, signature, **kwargs):
        captured.update(kwargs)
        real_init(self, signature, **kwargs)

    monkeypatch.setattr(dspy.RLM, "__init__", _spy)

    class _Interp:
        execution_instructions = "Runs in a container. Subprocesses ARE available."
        tools: dict = {}

        def start(self): ...
        def execute(self, code, variables=None): return ""
        def shutdown(self): ...

    class T(RLMTask):
        signature = "doc: str -> answer: str"
        output_field = "answer"

    T(interpreter=_Interp())._build_rlm()
    factory = captured.get("interpreter_factory")
    assert factory is not None, (
        "_build_rlm dropped the execution-instructions carrier; the model will be told it is "
        "running in Pyodide regardless of the real interpreter"
    )
    assert factory.execution_instructions == _Interp.execution_instructions


def test_build_rlm_passes_no_factory_for_an_interpreter_that_describes_nothing(monkeypatch):
    """The mirror image, and the compatibility half: an interpreter that says nothing about
    itself — a consumer's own, predating this feature — must leave the constructor call exactly
    as it was before 1.5.0. Passing a factory at all is the risky side of this change."""
    _configure_with_dummy()
    captured = {}
    real_init = dspy.RLM.__init__

    def _spy(self, signature, **kwargs):
        captured.update(kwargs)
        real_init(self, signature, **kwargs)

    monkeypatch.setattr(dspy.RLM, "__init__", _spy)

    class _Silent:                      # no execution_instructions at all
        tools: dict = {}

        def start(self): ...
        def execute(self, code, variables=None): return ""
        def shutdown(self): ...

    class T(RLMTask):
        signature = "doc: str -> answer: str"
        output_field = "answer"

    T(interpreter=_Silent())._build_rlm()
    assert "interpreter_factory" not in captured


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
    from rlm_harness import intercept_sub_lm

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
    from rlm_harness.task import _MainStepTimer

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

    from rlm_harness._retry import RLMTaskError
    from rlm_harness.trace import EVENT_MAIN_STEP, EVENT_RESULT, TraceRecorder, load_events

    _configure_with_dummy()
    pred = types.SimpleNamespace(
        trajectory=[{"reasoning": "t0", "code": "c0", "output": "o0"}],
        final_reasoning="gave up", answer="not-an-int")  # 'answer' can't coerce into _Out → retries fail
    task = _task_with_fake_rlm(pred)

    path = str(tmp_path / "trace.jsonl")
    with TraceRecorder(path, run_id="r1"), pytest.raises(RLMTaskError):
        asyncio.run(task.arun(q="hi"))

    ev = load_events(path)
    assert any(e["type"] == EVENT_MAIN_STEP for e in ev)    # the failed run's turns ARE recorded now
    assert not any(e["type"] == EVENT_RESULT for e in ev)   # but NO result → still "failed" to readers


def test_arun_records_result_on_success(tmp_path):
    import asyncio
    import types

    from rlm_harness.trace import EVENT_MAIN_STEP, EVENT_RESULT, TraceRecorder, load_events

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


def test_cancel_event_reaches_the_built_interpreter_end_to_end():
    """Not just unit-tested in isolation on sandbox.py — confirms RLMTask(cancel_event=...)
    actually threads through `_build_rlm()` -> `build_interpreter(...)` and lands on the
    real, constructed interpreter instance's `_cancel_event` attribute."""
    _configure_with_dummy(interpreter="pyodide")

    class T(RLMTask):
        signature = "q: str -> answer: _Out"
        output_field = "answer"
        output_model = _Out

    ev = threading.Event()
    task = T(cancel_event=ev)
    task._build_rlm()   # builds the interpreter and queues it for the forward() seam

    # Assert on the KIT's own handle, not dspy's private `_interpreter` slot: dspy 3.3.0
    # stopped holding the interpreter on the module (it takes one per forward() call).
    built = task._built_interpreter
    assert built is not None and built._cancel_event is ev

    # ...and that the same instance is queued for delivery on whichever seam this dspy
    # queues for delivery on the forward() positional seam. Without this half, the test would
    # still pass if `_build_rlm` built the interpreter and then dropped it.
    delivered = task._forward_interpreter
    assert delivered is built


async def test_sandbox_cancelled_survives_the_real_retry_engine_end_to_end():
    """The integration-level counterpart to `test_retry.py`'s unit-level `non_retryable`
    test: drives the REAL `run_with_retry` + the REAL outer `except Exception:` block in
    `RLMTask.arun()`, with `max_retries=3` and an injected fake RLM that raises
    `SandboxCancelled` on its first call. Confirms the run ends after exactly ONE attempt
    with the ORIGINAL `SandboxCancelled` object escaping `arun()` — never retried, never
    wrapped in `RLMTaskError` — closing the gap between "the fix works in isolation" and
    "the fix works through the real call chain a consumer actually uses."""
    _configure_with_dummy()
    calls = {"n": 0}
    original = SandboxCancelled("cancelled by the caller")

    class T(RLMTask):
        signature = "q: str -> answer: _Out"
        output_field = "answer"
        output_model = _Out

    task = T(max_retries=3)

    class _FakeRLM:
        async def aforward(self, **kw):
            calls["n"] += 1
            raise original

    task._build_rlm = lambda: _FakeRLM()

    with pytest.raises(SandboxCancelled) as ei:
        await task.arun(q="hi")
    assert ei.value is original
    assert calls["n"] == 1


async def test_fast_fail_lm_error_survives_the_real_retry_engine_end_to_end():
    """The `is_fast_fail` counterpart to `test_sandbox_cancelled_survives_the_real_retry_engine_
    end_to_end`: an LM error dspy itself calls non-retryable (`LMAuthError`) must end the run
    after exactly ONE attempt through the REAL call chain, with the ORIGINAL exception object
    escaping `arun()` unwrapped — not `max_retries` attempts, not `RLMTaskError`."""
    _configure_with_dummy()
    calls = {"n": 0}
    original = dspy.LMAuthError("bad key")

    class T(RLMTask):
        signature = "q: str -> answer: _Out"
        output_field = "answer"
        output_model = _Out

    task = T(max_retries=3)

    class _FakeRLM:
        async def aforward(self, **kw):
            calls["n"] += 1
            raise original

    task._build_rlm = lambda: _FakeRLM()

    with pytest.raises(dspy.LMAuthError) as ei:
        await task.arun(q="hi")
    assert ei.value is original
    assert calls["n"] == 1


async def test_context_window_exceeded_still_retries_end_to_end():
    """The carve-out, driven through the same real call chain: `ContextWindowExceededError` is
    non-retryable by dspy's own classification, but must still consume the FULL retry budget
    here (a later attempt can produce a shorter trajectory that fits) rather than fast-failing
    on the first one."""
    _configure_with_dummy()
    calls = {"n": 0}

    class T(RLMTask):
        signature = "q: str -> answer: _Out"
        output_field = "answer"
        output_model = _Out

    task = T(max_retries=3)

    class _FakeRLM:
        async def aforward(self, **kw):
            calls["n"] += 1
            raise dspy.ContextWindowExceededError()

    task._build_rlm = lambda: _FakeRLM()

    from rlm_harness._retry import RLMTaskError

    with pytest.raises(RLMTaskError):
        await task.arun(q="hi")
    assert calls["n"] == 3  # burned the full retry budget, not fast-failed on attempt 1


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

        async def acall(self, messages=None, **kw):
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


def test_configure_passes_request_timeout_through_to_both_lms():
    """The knob is only worth anything if it reaches litellm. dspy.LM keeps kwargs it does not
    recognise and merges them into the call, and `litellm.completion` takes `timeout` — so the
    assertion is that the value lands in `lm.kwargs` under that exact name.

    BOTH roles, not just the main one. The sub-LM is the recursion seat and `dspy.RLM` fans it
    across a thread pool, where a wedged request is LESS visible — a batched worker's failure is
    swallowed into an `"[ERROR] ..."` string for the model rather than surfacing. A mutation that
    popped the key between the two `dspy.LM(...)` constructions left the main-LM-only version of
    this test green."""
    rt.configure(RLMConfig(main_model="openai/x", sub_model="openai/x", interpreter="mock",
                           request_timeout_s=600.0))
    assert dspy.settings.lm.kwargs.get("timeout") == 600.0
    assert rt.get_sub_lm().kwargs.get("timeout") == 600.0


def test_configure_sends_no_timeout_when_unset():
    """The default must behave EXACTLY as before this field existed. Sending `timeout=None`
    explicitly is not the same as sending nothing — clients differ on what an explicit null means
    — so the key must be absent, not present-and-None."""
    rt.configure(RLMConfig(main_model="openai/x", sub_model="openai/x", interpreter="mock"))
    assert "timeout" not in dspy.settings.lm.kwargs


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


# ---- the parse callback must stage ONE stamp per root turn -------------------------------------
#
# dspy wraps `parse` with `with_callbacks` once per class that DEFINES it, and the kit's own
# `_LenientJSONAdapter.parse` calls `super().parse(...)`. Under the kit's DEFAULT adapter
# (`config.adapter == "json"`) that made every root turn fire the callback twice with identical
# outputs. Two stamps per turn made the trace's earliest-unused match order-unsafe whenever a model
# repeated a `reasoning` string across turns, and a consumer rendered the result as a -338.7s turn.


def _timer_and_captured():
    from rlm_harness.task import _MainStepTimer

    captured: list = []

    class _Rec:
        def note_main_step(self, reasoning, ts=None):
            captured.append(reasoning)

    return _MainStepTimer(_Rec()), captured


def test_main_step_timer_stages_once_for_a_NESTED_parse():
    """The regression. A nested parse pair (the kit adapter delegating to super()) must stage the
    OUTERMOST frame only — one stamp, not two."""
    timer, captured = _timer_and_captured()
    outputs = {"reasoning": "plan A", "code": "x = 1"}
    timer.on_adapter_parse_start("outer", instance=None, inputs={})
    timer.on_adapter_parse_start("inner", instance=None, inputs={})
    timer.on_adapter_parse_end("inner", outputs)
    timer.on_adapter_parse_end("outer", outputs)
    assert captured == ["plan A"]


def test_main_step_timer_degrades_to_the_old_behaviour_without_parse_start():
    """Deliberate degrade path: if a future dspy stops firing `on_adapter_parse_start`, the depth
    never rises and every end is treated as outermost — i.e. exactly the pre-fix behaviour. It must
    NOT degrade to staging nothing, which would silently send every main_step ts back to the
    flush-time fallback with no test going red."""
    timer, captured = _timer_and_captured()
    outputs = {"reasoning": "plan A", "code": "x = 1"}
    timer.on_adapter_parse_end("c1", outputs)
    timer.on_adapter_parse_end("c2", outputs)
    assert captured == ["plan A", "plan A"]


def test_main_step_timer_depth_does_not_leak_between_turns():
    """An orphan END (no matching start) must not drive the counter NEGATIVE and desynchronise
    every later turn — that is the direction `max(0, ...)` guards.

    The opposite direction, an orphan START, is deliberately NOT guarded and would silence every
    later turn. It is unreachable through dspy: `dspy.utils.callback.with_callbacks` runs its end
    handlers from a `finally`, so an exception inside `parse` still fires the end. Stated here
    rather than left implied, because "the counter can only be wrong in one direction, and here is
    why" is the sort of claim this release exists to stop taking on trust."""
    timer, captured = _timer_and_captured()
    outputs = {"reasoning": "r", "code": "c"}
    # dspy's own call shape — BaseCallback declares (call_id, instance, inputs) with no defaults.
    timer.on_adapter_parse_end("orphan-end", outputs)          # end with no start
    timer.on_adapter_parse_start("t1", instance=None, inputs={})
    timer.on_adapter_parse_end("t1", outputs)
    timer.on_adapter_parse_start("t2", instance=None, inputs={})
    timer.on_adapter_parse_end("t2", outputs)
    assert captured == ["r", "r", "r"]


def test_main_step_timer_stages_once_through_the_REAL_kit_adapter():
    """End to end against the installed dspy and the kit's own default adapter — the path that
    actually double-fired. Pinned for the stock adapter too, so a future dspy that stops nesting
    does not silently halve the stamps."""
    import json as _json

    from rlm_harness.task import _MainStepTimer

    sig = dspy.Signature("q -> reasoning, code")
    completion = _json.dumps({"reasoning": "R1", "code": "c1"})
    class _Rec:
        def __init__(self):
            self.captured: list = []

        def note_main_step(self, reasoning, ts=None):
            self.captured.append(reasoning)

    for adapter in (rt._LenientJSONAdapter(), dspy.JSONAdapter()):
        rec = _Rec()
        with dspy.context(callbacks=[_MainStepTimer(rec)]):
            adapter.parse(sig, completion)
        assert rec.captured == ["R1"], f"{type(adapter).__name__} staged {len(rec.captured)}"


def test_main_step_timer_depth_is_per_thread():
    """dspy may parse on a worker thread. A shared integer counter would let one thread's nested
    parse suppress another thread's outermost one — losing a turn's stamp entirely, which sends its
    main_step ts back to the flush-time fallback. Only visible with real concurrency."""
    from rlm_harness.task import _MainStepTimer

    lock = threading.Lock()
    captured: list = []

    class _Rec:
        def note_main_step(self, reasoning, ts=None):
            with lock:
                captured.append(reasoning)

    timer = _MainStepTimer(_Rec())
    started = threading.Barrier(4)

    def one_turn(name):
        started.wait()
        outputs = {"reasoning": name, "code": "c"}
        timer.on_adapter_parse_start(name, instance=None, inputs={})   # outer
        timer.on_adapter_parse_start(name, instance=None, inputs={})   # the kit adapter's super()
        timer.on_adapter_parse_end(name, outputs)
        timer.on_adapter_parse_end(name, outputs)

    threads = [threading.Thread(target=one_turn, args=(f"t{i}",)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(captured) == ["t0", "t1", "t2", "t3"]


def test_adjacent_duplicate_turns_get_distinct_ts_end_to_end(tmp_path):
    """THE end-to-end regression, and the one that distinguishes the two candidate designs.

    A retry loop emits the SAME reasoning on consecutive turns. Driven through the real callback
    dispatch into a real recorder, both turns must keep their own stamp. Deduplicating in the
    callback achieves that; the trace-side forward-only cursor CANNOT (turn 1 would take turn 0's
    spare stamp — no longer negative, so the symptom hides while the value stays ~0.1s wrong).
    """
    import types as _types

    from rlm_harness.task import _MainStepTimer
    from rlm_harness.trace import EVENT_MAIN_STEP, TraceRecorder, load_events

    path = str(tmp_path / "trace.jsonl")
    # A clock that ticks 0.1s per READ, so a turn parsed twice produces two DIFFERENT stamps —
    # which is what made the surplus one claimable by a later turn.
    now = [0.0]

    def clock():
        now[0] = round(now[0] + 0.1, 3)
        return now[0]

    with TraceRecorder(path, run_id="r1", clock=clock) as rec:
        rec.begin_main_capture()
        timer = _MainStepTimer(rec)
        outputs = {"reasoning": "Retrying tool call - previous attempt failed", "code": "c"}
        for base in (5.0, 9.0):                               # two ADJACENT turns, same reasoning
            now[0] = base
            timer.on_adapter_parse_start("outer", instance=None, inputs={})
            timer.on_adapter_parse_start("inner", instance=None, inputs={})
            timer.on_adapter_parse_end("inner", outputs)
            timer.on_adapter_parse_end("outer", outputs)
        rec.record_main_trajectory(
            _types.SimpleNamespace(
                trajectory=[dict(outputs, output="o"), dict(outputs, output="o")],
                final_reasoning=None,
            )
        )

    ts = [e["ts"] for e in load_events(path) if e["type"] == EVENT_MAIN_STEP]
    assert ts == [5.1, 9.1], f"turn 1 inherited turn 0's spare stamp: {ts}"


def test_main_step_timer_stages_once_at_ANY_adapter_nesting_depth():
    """`Adapter.__init_subclass__` re-wraps `parse` with `with_callbacks` for every subclass,
    unconditionally — so the fire count is a property of the caller's adapter hierarchy, not a
    fixed double. A consumer subclassing the kit's adapter and overriding NOTHING already gets
    three fires. Pinned because a fix that divided by two, or that special-cased the kit's own
    adapter, would pass every other test in this file and still lose that consumer's stamps."""
    import json as _json

    from rlm_harness.task import _MainStepTimer

    class _ConsumerSubclass(rt._LenientJSONAdapter):
        """Overrides nothing — the subclassing alone adds a wrapper."""

    class _ConsumerSubclassCallingSuper(_ConsumerSubclass):
        def parse(self, signature, completion):
            return super().parse(signature, completion)

    class _CountFires(dspy.utils.callback.BaseCallback):
        n = 0

        def on_adapter_parse_end(self, call_id, outputs, exception=None):
            type(self).n += 1

    sig = dspy.Signature("q -> reasoning, code")
    completion = _json.dumps({"reasoning": "R1", "code": "c1"})

    class _Rec:
        def __init__(self):
            self.captured: list = []

        def note_main_step(self, reasoning, ts=None):
            self.captured.append(reasoning)

    expected_fires = {
        dspy.JSONAdapter: 1,
        rt._LenientJSONAdapter: 2,
        _ConsumerSubclass: 3,
        _ConsumerSubclassCallingSuper: 4,
    }
    for adapter_cls, fires in expected_fires.items():
        _CountFires.n = 0
        rec = _Rec()
        with dspy.context(callbacks=[_CountFires(), _MainStepTimer(rec)]):
            adapter_cls().parse(sig, completion)
        assert _CountFires.n == fires, f"{adapter_cls.__name__} fired {_CountFires.n}, not {fires}"
        assert rec.captured == ["R1"], f"{adapter_cls.__name__} staged {len(rec.captured)}"
