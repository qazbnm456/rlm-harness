import pytest

from rlm_harness.sub_lm import SubLMValidationError, intercept_sub_lm, model_as_tool
from rlm_harness.trace import EVENT_SUB_CALL, EVENT_TOOL_CALL, TraceRecorder, load_events


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
    # (the bug — lifeline under-counts); the per-run binding re-establishes the recorder per call.
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
