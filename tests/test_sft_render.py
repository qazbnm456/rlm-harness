"""`export_sft_turns(render="adapter")` reproduces the request the model was actually sent.

Each test drives the real `dspy.RLM.aforward` loop offline (scripted LM and `ScriptedInterpreter`),
captures every request at the LM boundary with its own callback, and compares the export's re-render
with it. The reference is never the kit's own capture, because a test comparing the export with the
kit's record passes when capture and render are wrong in the same way.

Coverage: both adapters the kit configures (json, chat), action turns and the extract fallback, a run
under optimizer-compiled instructions, typed outputs (flat pydantic, and nested with `custom_types`),
and the raw completion and `reasoning_content` path.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from pydantic import BaseModel, Field

dspy = pytest.importorskip("dspy")

from dspy.utils.callback import BaseCallback, with_callbacks
from dspy.utils.dummies import DummyLM

import rlm_harness
import rlm_harness.runtime as rt
from rlm_harness import RLMConfig, RLMTask, optimize
from rlm_harness.dataset import export_sft_turns
from rlm_harness.prompt_record import (
    SOURCE_COMPLETION,
    SOURCE_RENDERED,
    RenderError,
    messages_sha256,
)
from rlm_harness.testing import ScriptedInterpreter
from rlm_harness.trace import (
    EVENT_EXTRACT,
    EVENT_FINAL,
    EVENT_MAIN_STEP,
    EVENT_RLM_SETUP,
    TraceRecorder,
    group_by_run,
    load_events,
)


class _Sent(BaseCallback):
    """The reference: every message list handed to an LM, in call order, nothing else."""

    def __init__(self) -> None:
        self.requests: list[list[dict]] = []

    def on_lm_start(self, call_id, instance, inputs):
        self.requests.append(json.loads(json.dumps(inputs.get("messages"))))


class Flat(BaseModel):
    title: str = Field(description="A short title.")
    tags: list[str] = []


class Inner(BaseModel):
    x: int


class Nested(BaseModel):
    title: str
    inner: Inner


class _Plain(RLMTask):
    signature = "document: str, n: int -> answer: str"
    output_field = "answer"
    instructions = "Answer from the document."


class _Flat(RLMTask):
    signature = "document: str -> out: Flat"
    output_field = "out"
    output_model = Flat
    instructions = "Summarise."


class _Nested(RLMTask):
    signature = "document: str -> out: Nested"
    output_field = "out"
    output_model = Nested
    instructions = "Summarise."


def _configure(lm, adapter="json", max_iterations=3):
    rt.configure(
        RLMConfig(main_model="x", sub_model="x", interpreter="mock", observe=False,
                  adapter=adapter, max_iterations=max_iterations, max_retries=1),
        main_lm=lm, sub_lm=lm,
    )


def _dummy(turns, adapter):
    return DummyLM(list(turns), adapter=dspy.JSONAdapter() if adapter == "json" else dspy.ChatAdapter())


def _run(tmp_path, task, inputs, *, name="t.jsonl"):
    """Run `task` traced, returning (events, the requests the LM really received)."""
    sent = _Sent()
    path = str(tmp_path / name)
    with TraceRecorder(path, run_id="r"), dspy.context(callbacks=[sent]):
        asyncio.run(task.arun(**inputs))
    return load_events(path), sent.requests


def _assert_parity(records, requests):
    assert len(records) == len(requests), (len(records), len(requests))
    for rec, real in zip(records, requests, strict=True):
        # The record shape is a contract a trainer keys on (docs: `export_sft_turns`); the optional
        # keys are the ones a run may or may not produce.
        assert {"run_id", "kind", "turn", "adapter", "messages", "assistant", "assistant_source",
                "verified"} <= set(rec) <= {"run_id", "kind", "turn", "adapter", "messages",
                                            "assistant", "assistant_source", "verified",
                                            "reasoning_content", "usage", "run_tag", "attempt",
                                            "parse_wrapped"}
        assert rec["messages"] == real, f"{rec['kind']} turn {rec['turn']} differs from the request"
        assert rec["verified"] is True


_TWO_TURNS_THEN_EXTRACT = [
    {"reasoning": "look", "code": "```python\nprint(len(document))\n```"},
    {"reasoning": "count", "code": "print(n)"},
]


@pytest.mark.parametrize("adapter", ["json", "chat"])
def test_action_and_extract_turns_reproduce_the_request_byte_for_byte(tmp_path, adapter):
    """Two action turns use up the budget, so dspy's extract fallback answers. All three requests,
    the extract call included, come back exact."""
    lm = _dummy([*_TWO_TURNS_THEN_EXTRACT, {"answer": "42"}], adapter)
    _configure(lm, adapter=adapter, max_iterations=2)
    task = _Plain(interpreter=ScriptedInterpreter(["11", "2"]))
    events, requests = _run(tmp_path, task, {"document": "hello world", "n": 2})

    records = export_sft_turns(group_by_run(events), render="adapter")
    assert [r["kind"] for r in records] == ["action", "action", "extract"]
    assert [r["turn"] for r in records] == [0, 1, 2]
    assert {r["adapter"] for r in records} == {adapter}
    _assert_parity(records, requests)

    final = next(e for e in events if e["type"] == EVENT_FINAL)
    assert final["payload"]["extract"] is True


def test_a_run_that_submits_records_no_extract_and_says_so(tmp_path):
    from rlm_harness.testing import submit

    lm = _dummy([{"reasoning": "done", "code": "SUBMIT(answer='x')"}], "json")
    _configure(lm)
    task = _Plain(interpreter=ScriptedInterpreter([submit({"answer": "x"})]))
    events, requests = _run(tmp_path, task, {"document": "d", "n": 1})

    assert not [e for e in events if e["type"] == EVENT_EXTRACT]
    assert next(e for e in events if e["type"] == EVENT_FINAL)["payload"]["extract"] is False
    records = export_sft_turns(group_by_run(events), render="adapter")
    _assert_parity(records, requests)


def test_the_setup_records_what_the_model_was_told(tmp_path):
    lm = _dummy([*_TWO_TURNS_THEN_EXTRACT, {"answer": "42"}], "json")
    _configure(lm, max_iterations=2)

    def lookup(key: str) -> str:
        """Look a key up."""
        return key

    task = _Plain(interpreter=ScriptedInterpreter(["11", "2"]), tools=[lookup], run_tag="arm-a")
    events, _ = _run(tmp_path, task, {"document": "hello world", "n": 2})

    setup = next(e for e in events if e["type"] == EVENT_RLM_SETUP)["payload"]
    assert setup["adapter"] == "json"
    assert setup["dspy"] == dspy.__version__
    assert setup["max_iterations"] == 2
    assert setup["input_fields"] == ["document", "n"]
    assert setup["output_fields"] == ["answer"]
    assert [t["name"] for t in setup["tools"]] == ["lookup"]
    assert setup["run_tag"] == "arm-a"
    assert len(setup["variables_info"]) == 2
    assert "Total length: 11 characters" in setup["variables_info"][0]
    action = setup["predictors"]["action"]
    assert [f["name"] for f in action["fields"]] == [
        "variables_info", "repl_history", "iteration", "reasoning", "code"]
    assert "Answer from the document." in action["instructions"]
    extract = setup["predictors"]["extract"]
    assert [(f["name"], f["io"]) for f in extract["fields"]] == [
        ("variables_info", "input"), ("repl_history", "input"), ("answer", "output")]

    steps = [e["payload"] for e in events if e["type"] == EVENT_MAIN_STEP]
    assert [s["iteration"] for s in steps] == ["1/2", "2/2"]
    assert [s["history_len"] for s in steps] == [0, 1]
    assert all(s["run_tag"] == "arm-a" and s["attempt"] == 0 for s in steps)
    assert all("usage" in s for s in steps)
    # The model wrote a fenced cell on turn 0; the trajectory keeps dspy's stripped code.
    assert steps[0]["code"] == "print(len(document))"
    assert steps[0]["code_raw"] == "```python\nprint(len(document))\n```"
    assert "code_raw" not in steps[1]


def test_compiled_instructions_are_what_gets_recorded_and_rendered(tmp_path):
    """An optimizer's text replaces the shipped one; the record and the render follow the text
    the model actually saw, not the text this version of the kit would have built."""
    lm = _dummy([*_TWO_TURNS_THEN_EXTRACT, {"answer": "42"}], "json")
    _configure(lm, max_iterations=2)
    base = _Plain(interpreter=ScriptedInterpreter([]))
    path = tmp_path / "state.json"
    optimize.save_program(base._build_rlm(), str(path), task=base)
    state = json.loads(path.read_text())
    state["generate_action"]["signature"]["instructions"] = "COMPILED: work in two passes."
    state["extract"]["signature"]["instructions"] = "COMPILED: now answer."
    path.write_text(json.dumps(state))

    task = _Plain(interpreter=ScriptedInterpreter(["11", "2"]), compiled_instructions=str(path))
    events, requests = _run(tmp_path, task, {"document": "hello world", "n": 2})

    setup = next(e for e in events if e["type"] == EVENT_RLM_SETUP)["payload"]
    assert setup["predictors"]["action"]["instructions"] == "COMPILED: work in two passes."
    assert setup["predictors"]["extract"]["instructions"] == "COMPILED: now answer."
    records = export_sft_turns(group_by_run(events), render="adapter")
    _assert_parity(records, requests)
    assert "COMPILED: work in two passes." in records[0]["messages"][0]["content"]


@pytest.mark.parametrize("adapter", ["json", "chat"])
def test_a_flat_pydantic_output_renders_from_its_recorded_schema(tmp_path, adapter):
    """No class is passed: the extract signature's output type is rebuilt from the schema the
    trace recorded, which is everything an adapter reads off it."""
    lm = _dummy([{"reasoning": "r", "code": "print(1)"},
                 {"out": {"title": "t", "tags": ["a"]}}], adapter)
    _configure(lm, adapter=adapter, max_iterations=1)
    events, requests = _run(tmp_path, _Flat(interpreter=ScriptedInterpreter(["1"])),
                            {"document": "d"})
    records = export_sft_turns(group_by_run(events), render="adapter")
    _assert_parity(records, requests)
    extract = next(e for e in events if e["type"] == EVENT_EXTRACT)["payload"]
    assert extract["outputs"] == {"out": {"title": "t", "tags": ["a"]}}


def test_a_nested_output_needs_its_class_and_says_so(tmp_path):
    lm = _dummy([{"reasoning": "r", "code": "print(1)"},
                 {"out": {"title": "t", "inner": {"x": 1}}}], "json")
    _configure(lm, max_iterations=1)
    events, requests = _run(tmp_path, _Nested(interpreter=ScriptedInterpreter(["1"])),
                            {"document": "d"})
    runs = group_by_run(events)

    with pytest.raises(RenderError, match="custom_types"):
        export_sft_turns(runs, render="adapter")
    records = export_sft_turns(runs, render="adapter", custom_types={"Nested": Nested})
    _assert_parity(records, requests)


def test_a_wrong_render_is_reported_rather_than_passed_off(tmp_path):
    """The hash is what makes `verified` mean something: a render through the wrong type must
    come out False, and strict mode must refuse it."""
    lm = _dummy([{"reasoning": "r", "code": "print(1)"}, {"out": {"title": "t", "tags": []}}], "json")
    _configure(lm, max_iterations=1)
    events, _ = _run(tmp_path, _Flat(interpreter=ScriptedInterpreter(["1"])), {"document": "d"})
    runs = group_by_run(events)

    class Flat(BaseModel):  # same name, different schema
        title: int

    records = export_sft_turns(runs, render="adapter", custom_types={"Flat": Flat})
    assert [r["verified"] for r in records] == [True, False]
    with pytest.raises(RenderError, match="does not match"):
        export_sft_turns(runs, render="adapter", custom_types={"Flat": Flat}, strict=True)


def test_the_assistant_side_is_the_raw_completion(tmp_path):
    lm = _dummy([*_TWO_TURNS_THEN_EXTRACT, {"answer": "42"}], "json")
    _configure(lm, max_iterations=2)
    events, _ = _run(tmp_path, _Plain(interpreter=ScriptedInterpreter(["11", "2"])),
                     {"document": "hello world", "n": 2})
    records = export_sft_turns(group_by_run(events), render="adapter")
    assert {r["assistant_source"] for r in records} == {SOURCE_COMPLETION}
    assert json.loads(records[0]["assistant"]["content"]) == _TWO_TURNS_THEN_EXTRACT[0]
    assert json.loads(records[2]["assistant"]["content"]) == {"answer": "42"}

    # A trace without the raw text falls back to dspy's own formatter, and says it did.
    for e in events:
        e["payload"].pop("completion", None)
    fallback = export_sft_turns(group_by_run(events), render="adapter")
    assert {r["assistant_source"] for r in fallback} == {SOURCE_RENDERED}
    assert json.loads(fallback[0]["assistant"]["content"]) == _TWO_TURNS_THEN_EXTRACT[0]
    assert fallback[0]["assistant"]["content"].startswith('{\n  "reasoning"')


class _ReasoningLM(dspy.BaseLM):
    """Returns content plus a separate reasoning channel, as an OpenAI-compatible reasoning server."""

    def __init__(self, turns):
        super().__init__(model="reasoning-dummy")
        self.turns = list(turns)

    def _next(self):
        content, thinking = self.turns.pop(0)
        return [{"text": json.dumps(content), "reasoning_content": thinking}]

    @with_callbacks
    def __call__(self, prompt=None, messages=None, **kwargs):
        return self._next()

    @with_callbacks
    async def acall(self, prompt=None, messages=None, **kwargs):
        return self._next()


class _RawLM(dspy.BaseLM):
    """Returns each scripted reply verbatim, so a test can send text that is not valid JSON."""

    def __init__(self, replies):
        super().__init__(model="raw-dummy")
        self.replies = list(replies)

    @with_callbacks
    def __call__(self, prompt=None, messages=None, **kwargs):
        return [{"text": self.replies.pop(0)}]

    @with_callbacks
    async def acall(self, prompt=None, messages=None, **kwargs):
        return [{"text": self.replies.pop(0)}]


def test_a_reply_parsed_only_after_the_brace_wrap_is_marked(tmp_path):
    """The adapter wraps a brace-less reply in braces and parses it, so the run goes on, but the
    reply as written is not valid JSON. The step says so, and so does the record built from it."""
    replies = [
        '"reasoning": "look", "code": "print(1)"',          # wrapped
        json.dumps({"reasoning": "count", "code": "print(2)"}),  # parsed as is
        '"answer": "42"',                                    # wrapped, in the extract call
    ]
    _configure(_RawLM(replies), max_iterations=2)
    events, requests = _run(tmp_path, _Plain(interpreter=ScriptedInterpreter(["1", "2"])),
                            {"document": "hello world", "n": 2})
    steps = [e["payload"] for e in events if e["type"] == EVENT_MAIN_STEP]
    extract = next(e["payload"] for e in events if e["type"] == EVENT_EXTRACT)
    assert [s.get("parse_wrapped") for s in steps] == [True, None]
    assert "parse_wrapped" not in steps[1], "absent, not false, when the reply parsed as is"
    assert extract["parse_wrapped"] is True

    records = export_sft_turns(group_by_run(events), render="adapter")
    _assert_parity(records, requests)
    assert [r.get("parse_wrapped") for r in records] == [True, None, True]
    assert records[0]["assistant"]["content"] == replies[0]


def test_a_wrapped_reply_from_the_reasoning_channel_is_marked_and_rendered(tmp_path):
    """Content empty, the brace-less turn in `reasoning_content`: the adapter promotes it and wraps
    it. The flag still lands, and the record says its target was rendered, not written."""
    replies = [
        {"text": None, "reasoning_content": '"reasoning": "look", "code": "print(1)"'},
        {"text": json.dumps({"reasoning": "count", "code": "print(2)"})},
        {"text": json.dumps({"answer": "42"})},
    ]

    class _LM(dspy.BaseLM):
        def __init__(self):
            super().__init__(model="channel-dummy")

        @with_callbacks
        def __call__(self, prompt=None, messages=None, **kwargs):
            return [replies.pop(0)]

        @with_callbacks
        async def acall(self, prompt=None, messages=None, **kwargs):
            return [replies.pop(0)]

    _configure(_LM(), max_iterations=2)
    events, _ = _run(tmp_path, _Plain(interpreter=ScriptedInterpreter(["1", "2"])),
                     {"document": "hello world", "n": 2})
    records = export_sft_turns(group_by_run(events), render="adapter")
    assert [r.get("parse_wrapped") for r in records] == [True, None, None]
    assert records[0]["assistant_source"] == SOURCE_RENDERED
    assert json.loads(records[0]["assistant"]["content"]) == {"reasoning": "look", "code": "print(1)"}


def test_concurrent_runs_keep_their_own_wrap_marks(tmp_path):
    """Two runs in one event loop, one wrapping its first turn and the other its second. Each mark
    lands on its own run's step."""
    def plan(wrap_turn):
        turns = [{"reasoning": f"t{i}", "code": f"print({i})"} for i in range(2)]
        out = [json.dumps(t) if i != wrap_turn else json.dumps(t)[1:-1] for i, t in enumerate(turns)]
        return out + [json.dumps({"answer": "42"})]

    _configure(_RawLM([]), max_iterations=2)

    async def one(name, wrap_turn):
        path = str(tmp_path / f"{name}.jsonl")
        with TraceRecorder(path, run_id=name), dspy.context(lm=_RawLM(plan(wrap_turn))):
            await _Plain(interpreter=ScriptedInterpreter(["1", "2"])).arun(document="d", n=1)
        return [e["payload"].get("parse_wrapped") for e in load_events(path)
                if e["type"] == EVENT_MAIN_STEP]

    async def both():
        return await asyncio.gather(one("a", 0), one("b", 1))

    assert asyncio.run(both()) == [[True, None], [None, True]]


def test_reasoning_content_is_kept_as_its_own_field(tmp_path):
    lm = _ReasoningLM([({"reasoning": "r", "code": "print(1)"}, "thinking one"),
                       ({"answer": "a"}, "thinking two")])
    _configure(lm, max_iterations=1)
    events, requests = _run(tmp_path, _Plain(interpreter=ScriptedInterpreter(["1"])),
                            {"document": "d", "n": 1})
    records = export_sft_turns(group_by_run(events), render="adapter")
    _assert_parity(records, requests)
    assert [r["reasoning_content"] for r in records] == ["thinking one", "thinking two"]
    assert json.loads(records[0]["assistant"]["content"]) == {"reasoning": "r", "code": "print(1)"}


def test_the_default_export_is_unchanged_and_old_traces_are_skipped_loudly(tmp_path, caplog):
    lm = _dummy([*_TWO_TURNS_THEN_EXTRACT, {"answer": "42"}], "json")
    _configure(lm, max_iterations=2)
    events, _ = _run(tmp_path, _Plain(interpreter=ScriptedInterpreter(["11", "2"])),
                     {"document": "hello world", "n": 2})
    runs = group_by_run(events)

    default = export_sft_turns(runs)
    assert [sorted(r) for r in default] == [["input", "output", "run_id", "turn"]] * 2

    old = [e for e in events if e["type"] not in (EVENT_RLM_SETUP, EVENT_EXTRACT)]
    with caplog.at_level("WARNING", logger="rlm_harness.dataset"):
        assert export_sft_turns(group_by_run(old), render="adapter") == []
    assert "skipped 2 call(s)" in caplog.text
    with pytest.raises(ValueError, match="render"):
        export_sft_turns(runs, render="chatml")


def test_two_runs_under_one_recorder_render_against_their_own_history(tmp_path):
    lm = _dummy([*_TWO_TURNS_THEN_EXTRACT, {"answer": "1"}, *_TWO_TURNS_THEN_EXTRACT,
                 {"answer": "2"}], "json")
    _configure(lm, max_iterations=2)
    sent = _Sent()
    path = str(tmp_path / "t.jsonl")
    with TraceRecorder(path, run_id="r"), dspy.context(callbacks=[sent]):
        asyncio.run(_Plain(interpreter=ScriptedInterpreter(["11", "2"])).arun(document="a", n=1))
        asyncio.run(_Plain(interpreter=ScriptedInterpreter(["11", "2"])).arun(document="b", n=2))
    records = export_sft_turns(group_by_run(load_events(path)), render="adapter")
    _assert_parity(records, sent.requests)


def test_messages_sha256_survives_a_lone_surrogate():
    assert len(messages_sha256([{"role": "user", "content": "caf\udce9"}])) == 64


def test_capture_never_needs_rlm_harness_to_import_dspy_at_module_top():
    import importlib
    import sys

    module = importlib.import_module("rlm_harness.prompt_record")
    from pathlib import Path

    source = Path(module.__file__).read_text(encoding="utf-8")
    top = source.split("\ndef ", 1)[0]
    assert "import dspy" not in top and "from dspy" not in top
    assert "rlm_harness.prompt_record" in sys.modules
    assert rlm_harness.export_sft_turns is export_sft_turns


def test_a_retry_records_the_prompts_of_the_attempt_whose_turns_reached_the_trace(tmp_path):
    """Attempt 0 submits a value that fails validation; attempt 1 succeeds. The recorded turns are
    attempt 1's, so the recorded prompts, usage and `attempt` must be attempt 1's too."""
    from rlm_harness.testing import submit

    lm = _dummy([{"reasoning": "bad", "code": "SUBMIT(out='nope')"},
                 {"reasoning": "good", "code": "SUBMIT(out='{\"title\": \"t\"}')"}], "json")
    rt.configure(RLMConfig(main_model="x", sub_model="x", interpreter="mock", observe=False,
                           max_retries=2), main_lm=lm, sub_lm=lm)

    class Strict(BaseModel):
        title: str

    class _Strict(RLMTask):
        # The SIGNATURE says str, so dspy accepts either SUBMIT; only the kit's coercion into
        # `output_model` rejects the first, which is what makes it a whole-trajectory retry.
        signature = "document: str -> out: str"
        output_field = "out"
        output_model = Strict

    interp = ScriptedInterpreter([submit({"out": "nope"}), submit({"out": '{"title": "t"}'})])
    events, requests = _run(tmp_path, _Strict(interpreter=interp), {"document": "d"})
    steps = [e["payload"] for e in events if e["type"] == EVENT_MAIN_STEP]
    assert [s["reasoning"] for s in steps] == ["good"]
    assert steps[0]["attempt"] == 1
    records = export_sft_turns(group_by_run(events), render="adapter")
    assert len(records) == 1 and records[0]["messages"] == requests[-1]
    assert records[0]["verified"] is True


def test_format_inputs_are_read_from_either_callback_shape():
    from rlm_harness.task import _bind_format_inputs

    sig = dspy.Signature("a: str -> b: str")
    named = {"signature": sig, "demos": [], "inputs": {"a": "x"}}
    positional = {"args": (sig, [], {"a": "x"}), "kwargs": {}}
    assert _bind_format_inputs(named) == (sig, [], {"a": "x"})
    assert _bind_format_inputs(positional) == (sig, [], {"a": "x"})
    assert _bind_format_inputs({"args": ("not a signature",), "kwargs": {}}) is None
    assert _bind_format_inputs(None) is None


def test_one_unmatched_turn_is_skipped_not_fatal_to_the_corpus(tmp_path, caplog):
    """A turn the capture left without its fields (here: `history_len` removed, as when the
    reasoning check fails) must be skipped and counted, while every other call still exports."""
    lm = _dummy([*_TWO_TURNS_THEN_EXTRACT, {"answer": "42"}], "json")
    _configure(lm, max_iterations=2)
    events, requests = _run(tmp_path, _Plain(interpreter=ScriptedInterpreter(["11", "2"])),
                            {"document": "hello world", "n": 2})
    first = next(e for e in events if e["type"] == EVENT_MAIN_STEP)
    del first["payload"]["history_len"]

    with caplog.at_level("WARNING", logger="rlm_harness.dataset"):
        records = export_sft_turns(group_by_run(events), render="adapter")
    assert [(r["kind"], r["turn"]) for r in records] == [("action", 1), ("extract", 2)]
    _assert_parity(records, requests[1:])
    assert "skipped 1 call(s)" in caplog.text


def test_attempts_are_counted_even_without_a_usage_tracker(tmp_path, monkeypatch):
    """Attempts are numbered without a usage tracker too, so a retry's calls never overwrite the
    recorded attempt's."""
    import contextlib

    from rlm_harness import _dspy_compat
    from rlm_harness.testing import submit

    @contextlib.contextmanager
    def no_tracking():
        yield None

    monkeypatch.setattr(_dspy_compat, "usage_tracking", no_tracking)
    lm = _dummy([{"reasoning": "bad", "code": "SUBMIT(out='nope')"},
                 {"reasoning": "good", "code": "SUBMIT(out='{\"title\": \"t\"}')"}], "json")
    rt.configure(RLMConfig(main_model="x", sub_model="x", interpreter="mock", observe=False,
                           max_retries=2), main_lm=lm, sub_lm=lm)

    class Strict(BaseModel):
        title: str

    class _Strict(RLMTask):
        signature = "document: str -> out: str"
        output_field = "out"
        output_model = Strict

    interp = ScriptedInterpreter([submit({"out": "nope"}), submit({"out": '{"title": "t"}'})])
    events, requests = _run(tmp_path, _Strict(interpreter=interp), {"document": "d"})
    steps = [e["payload"] for e in events if e["type"] == EVENT_MAIN_STEP]
    assert [(s["reasoning"], s["attempt"]) for s in steps] == [("good", 1)]
    assert "usage" not in steps[0]
    records = export_sft_turns(group_by_run(events), render="adapter")
    assert records[0]["messages"] == requests[-1] and records[0]["verified"] is True


def test_an_empty_completion_is_never_the_training_target(tmp_path):
    """A reasoning server that puts the whole turn in `reasoning_content` and returns content ""
    is parsed by the kit's adapter from the reasoning channel. An assistant target of "" would teach
    an empty reply, so the record falls back to the rendered outputs and says so."""
    lm = _ReasoningLM([])
    lm.turns = [({"reasoning": "r", "code": "print(1)"}, None), ({"answer": "a"}, None)]
    lm._next = lambda: [{"text": "", "reasoning_content": json.dumps(lm.turns.pop(0)[0])}]
    _configure(lm, max_iterations=1)
    events, requests = _run(tmp_path, _Plain(interpreter=ScriptedInterpreter(["1"])),
                            {"document": "d", "n": 1})
    records = export_sft_turns(group_by_run(events), render="adapter")
    _assert_parity(records, requests)
    assert [r["assistant_source"] for r in records] == [SOURCE_RENDERED, SOURCE_RENDERED]
    assert json.loads(records[0]["assistant"]["content"]) == {"reasoning": "r", "code": "print(1)"}
    assert json.loads(records[0]["reasoning_content"]) == {"reasoning": "r", "code": "print(1)"}


def test_demos_are_written_once_not_on_every_step(tmp_path):
    lm = _dummy([*_TWO_TURNS_THEN_EXTRACT, {"answer": "42"}], "json")
    _configure(lm, max_iterations=2)

    class _WithDemos(_Plain):
        def _build_rlm(self):
            rlm = super()._build_rlm()
            rlm.generate_action.demos = [dspy.Example(
                variables_info="v", repl_history="h", iteration="1/1", reasoning="r", code="c")]
            return rlm

    events, requests = _run(tmp_path, _WithDemos(interpreter=ScriptedInterpreter(["11", "2"])),
                            {"document": "hello world", "n": 2})
    setup = next(e for e in events if e["type"] == EVENT_RLM_SETUP)["payload"]
    assert len(setup["predictors"]["action"]["demos"]) == 1
    assert all("demos" not in e["payload"] for e in events if e["type"] == EVENT_MAIN_STEP)
    _assert_parity(export_sft_turns(group_by_run(events), render="adapter"), requests)


def test_inputs_sha256_hashes_the_full_bound_values_in_signature_order(tmp_path):
    import hashlib

    lm = _dummy([*_TWO_TURNS_THEN_EXTRACT, {"answer": "42"}], "json")
    _configure(lm, max_iterations=2)
    long_doc = "長文 " * 5000                      # far past dspy's 1000-char preview
    events, _ = _run(tmp_path, _Plain(interpreter=ScriptedInterpreter(["11", "2"])),
                     {"n": 2, "document": long_doc})
    setup = next(e for e in events if e["type"] == EVENT_RLM_SETUP)["payload"]
    assert list(setup["inputs_sha256"]) == ["document", "n"]      # signature order, not call order
    assert setup["inputs_sha256"]["document"] == hashlib.sha256(long_doc.encode()).hexdigest()
    assert setup["inputs_sha256"]["n"] == hashlib.sha256(b"2").hexdigest()


def test_input_sha256_serialisation_is_the_documented_one():
    import hashlib

    from rlm_harness.large_text import LargeText
    from rlm_harness.prompt_record import input_sha256

    text = "line one\nline two 中文"
    with LargeText(text) as staged:
        assert input_sha256(staged) == input_sha256(text) == hashlib.sha256(
            text.encode()).hexdigest()
    value = {"b": [1, 2], "a": "é"}
    expected = hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    assert input_sha256(value) == expected
    assert input_sha256(Inner(x=1)) == hashlib.sha256(b'{"x": 1}').hexdigest()


def test_lm_settings_record_sampling_and_never_a_secret():
    from rlm_harness import _dspy_compat

    lm = dspy.LM("openai/teacher", api_key="sk-SECRET-123", api_base="http://internal:8000",
                 temperature=0.7, top_p=0.9, max_tokens=512,
                 extra_body={"chat_template_kwargs": {"enable_thinking": True},
                             "x_auth": "sk-SECRET-123"})
    got = _dspy_compat.lm_settings(lm)
    assert got["model"] == "openai/teacher"
    assert got["kwargs"]["temperature"] == 0.7 and got["kwargs"]["top_p"] == 0.9
    assert got["kwargs"]["max_tokens"] == 512
    assert got["kwargs"]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}
    assert {"api_key", "api_base", "extra_body.x_auth"} <= set(got["unrecorded_keys"])
    serialised = json.dumps(got)
    assert "SECRET" not in serialised and "internal:8000" not in serialised


def test_rlm_setup_records_the_main_and_sub_lm(tmp_path):
    main = _dummy([*_TWO_TURNS_THEN_EXTRACT, {"answer": "42"}], "json")
    _configure(main, max_iterations=2)
    sub = DummyLM([], adapter=dspy.JSONAdapter())
    sub.kwargs["temperature"] = 0.2
    events, _ = _run(tmp_path, _Plain(interpreter=ScriptedInterpreter(["11", "2"]), sub_lm=sub),
                     {"document": "d", "n": 1})
    lm = next(e for e in events if e["type"] == EVENT_RLM_SETUP)["payload"]["lm"]
    assert lm["main"]["model"] == "dummy"
    assert lm["sub_same_as_main"] is False and lm["sub"]["kwargs"]["temperature"] == 0.2

    same = _dummy([*_TWO_TURNS_THEN_EXTRACT, {"answer": "42"}], "json")
    _configure(same, max_iterations=2)
    events, _ = _run(tmp_path, _Plain(interpreter=ScriptedInterpreter(["11", "2"])),
                     {"document": "d", "n": 1}, name="same.jsonl")
    lm = next(e for e in events if e["type"] == EVENT_RLM_SETUP)["payload"]["lm"]
    assert lm["sub_same_as_main"] is True and "sub" not in lm


def test_a_secret_nested_under_an_allowlisted_key_is_never_written():
    from rlm_harness import _dspy_compat

    lm = dspy.LM("openai/teacher", api_key="k", thinking={"budget_tokens": 5, "api_key": "SECRET2"},
                 stop=["</a>", {"token": "SECRET4"}],
                 extra_body={"chat_template_kwargs": {"enable_thinking": True, "token": "SECRET3"}})
    got = _dspy_compat.lm_settings(lm)
    assert "SECRET" not in json.dumps(got)
    assert got["kwargs"]["thinking"] == {"budget_tokens": 5}
    assert got["kwargs"]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}
    assert "stop" not in got["kwargs"]
    assert {"thinking.api_key", "extra_body.chat_template_kwargs.token", "stop"} <= set(
        got["unrecorded_keys"])


def test_one_unhashable_input_loses_only_its_own_digest():
    from rlm_harness.large_text import LargeText
    from rlm_harness.task import _inputs_sha256

    class Sub(LargeText):
        pass

    with Sub("abc") as staged, LargeText("abc") as plain:
        got = _inputs_sha256({"bad": {1: "a", "b": 2}, "staged": staged, "plain": plain})
    assert "bad" not in got
    assert got["staged"] == got["plain"]
