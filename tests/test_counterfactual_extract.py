"""`export_counterfactual_extracts` builds the extract call a SUBMIT run never made.

The reference is a real extract: a second run with the same action turns and a smaller
`max_iterations` hits the budget, so dspy sends the extract request itself. A counterfactual cut
before the first run's SUBMIT has the same history, so it must be that request byte for byte, and
hash to the `messages_sha256` the second run recorded.
"""

from __future__ import annotations

import asyncio
import json
import os

import pytest
from pydantic import BaseModel

dspy = pytest.importorskip("dspy")

from dspy.utils.callback import BaseCallback
from dspy.utils.dummies import DummyLM

import rlm_harness
import rlm_harness.runtime as rt
from rlm_harness import RLMConfig, RLMTask, export_counterfactual_extracts
from rlm_harness.dataset import export_sft_turns
from rlm_harness.prompt_record import (
    SOURCE_COUNTERFACTUAL,
    TARGET_FINAL_OUTPUTS,
    TARGET_FINAL_REPR,
    TARGET_RESULT,
    messages_sha256,
)
from rlm_harness.testing import ScriptedInterpreter, submit
from rlm_harness.trace import EVENT_EXTRACT, EVENT_FINAL, TraceRecorder, group_by_run, load_events


class _Sent(BaseCallback):
    def __init__(self) -> None:
        self.requests: list[list[dict]] = []

    def on_lm_start(self, call_id, instance, inputs):
        self.requests.append(json.loads(json.dumps(inputs.get("messages"))))


class _Plain(RLMTask):
    signature = "document: str, n: int -> answer: str, score: int"
    output_field = "answer"
    instructions = "Answer from the document."


class Flat(BaseModel):
    title: str
    tags: list[str] = []


class _Flat(RLMTask):
    signature = "document: str -> out: Flat"
    output_field = "out"
    output_model = Flat
    instructions = "Summarise."


_TURNS = [
    {"reasoning": "look", "code": "```python\nprint(len(document))\n```"},
    {"reasoning": "count", "code": "print(n)"},
]
_INPUTS = {"document": "hello world", "n": 2}


def _run(tmp_path, name, task_cls, lm_turns, outputs, *, adapter="json", max_iterations,
         max_retries=1, inputs=_INPUTS):
    lm = DummyLM(list(lm_turns),
                 adapter=dspy.JSONAdapter() if adapter == "json" else dspy.ChatAdapter())
    rt.configure(
        RLMConfig(main_model="x", sub_model="x", interpreter="mock", observe=False,
                  adapter=adapter, max_iterations=max_iterations, max_retries=max_retries),
        main_lm=lm, sub_lm=lm,
    )
    sent = _Sent()
    path = str(tmp_path / name)
    with TraceRecorder(path, run_id=name), dspy.context(callbacks=[sent]):
        try:
            asyncio.run(task_cls(interpreter=ScriptedInterpreter(outputs)).arun(**inputs))
        except rlm_harness.RLMTaskError:
            pass
    return load_events(path), sent.requests


def _submit_run(tmp_path, adapter="json"):
    return _run(tmp_path, "submitted", _Plain,
                [*_TURNS, {"reasoning": "done", "code": "SUBMIT(answer='42', score=7)"}],
                ["11", "2", submit({"answer": "42", "score": 7})],
                adapter=adapter, max_iterations=3)


@pytest.mark.parametrize("adapter", ["json", "chat"])
def test_the_counterfactual_is_the_request_a_real_extract_sends(tmp_path, adapter):
    events, _ = _submit_run(tmp_path, adapter)
    cut_off, requests = _run(tmp_path, "cut_off", _Plain,
                             [*_TURNS, {"answer": "41", "score": 1}], ["11", "2"],
                             adapter=adapter, max_iterations=2)
    real = next(e["payload"] for e in cut_off if e["type"] == EVENT_EXTRACT)

    [rec] = export_counterfactual_extracts(group_by_run(events))
    assert rec["messages"] == requests[-1]
    assert messages_sha256(rec["messages"]) == real["messages_sha256"]
    assert rec["kind"] == "extract" and rec["turn"] == 2 and rec["adapter"] == adapter
    assert rec["assistant_source"] == SOURCE_COUNTERFACTUAL
    assert rec["verified"] is None
    assert rec["counterfactual"] == {"submit_turn": 2, "cut_back": 0,
                                     "target_source": TARGET_FINAL_OUTPUTS}
    assert rec["attempt"] == 0 and rec["run_id"] == "submitted"
    fmt = dspy.JSONAdapter() if adapter == "json" else dspy.ChatAdapter()
    sig = dspy.Signature("variables_info: str, repl_history: str -> answer: str, score: int")
    assert rec["assistant"] == {"role": "assistant", "content": fmt.format_assistant_message_content(
        sig, {"answer": "42", "score": 7})}
    if adapter == "json":
        assert json.loads(rec["assistant"]["content"]) == {"answer": "42", "score": 7}


def test_the_submitted_values_are_recorded_on_final(tmp_path):
    events, _ = _submit_run(tmp_path)
    final = next(e["payload"] for e in events if e["type"] == EVENT_FINAL)
    assert final["extract"] is False
    assert final["outputs"] == {"answer": "42", "score": 7}


def test_an_older_trace_reads_the_values_back_from_the_submit_turn(tmp_path):
    events, _ = _submit_run(tmp_path)
    [new] = export_counterfactual_extracts(group_by_run(events))
    for e in events:
        if e["type"] == EVENT_FINAL:
            del e["payload"]["outputs"]   # what a 1.25.0 trace looks like
    [old] = export_counterfactual_extracts(group_by_run(events))
    assert new["counterfactual"]["target_source"] == TARGET_FINAL_OUTPUTS
    assert old["counterfactual"]["target_source"] == TARGET_FINAL_REPR
    assert old["assistant"] == new["assistant"]
    assert old["messages"] == new["messages"]


def test_earlier_cuts_and_a_cut_with_no_history_left(tmp_path, caplog):
    events, _ = _submit_run(tmp_path)
    with caplog.at_level("WARNING", logger="rlm_harness.dataset"):
        recs = export_counterfactual_extracts(group_by_run(events), cut_back=(0, 1, 2))
    assert [(r["turn"], r["counterfactual"]["cut_back"]) for r in recs] == [(2, 0), (1, 1)]
    assert all(r["assistant"] == recs[0]["assistant"] for r in recs)
    assert "skipped 1 record(s)" in caplog.text
    with pytest.raises(ValueError):
        export_counterfactual_extracts(group_by_run(events), cut_back=(-1,))


def _without_final_outputs(events):
    """What a 1.25.0 trace looks like: no `final.outputs`."""
    events = json.loads(json.dumps(events))
    for e in events:
        if e["type"] == EVENT_FINAL:
            del e["payload"]["outputs"]
    return events


def _typed_run(tmp_path, task_cls, value, name="typed"):
    events, _ = _run(tmp_path, name, task_cls,
                     [*_TURNS[:1], {"reasoning": "done", "code": "SUBMIT(out=...)"}],
                     ["11", submit({"out": value})],
                     max_iterations=3, inputs={"document": "hello world"})
    return events


def test_a_typed_submit_in_an_older_trace_is_read_from_the_result(tmp_path):
    """`FINAL: {'out': Flat(title=...)}` is not a literal, so a 1.25.0 trace falls back to the
    `result` event, the value after `output_model` validation."""
    events = _typed_run(tmp_path, _Flat, {"title": "t", "tags": ["a"]})
    [new] = export_counterfactual_extracts(group_by_run(events))
    assert json.loads(new["assistant"]["content"]) == {"out": {"title": "t", "tags": ["a"]}}

    [old] = export_counterfactual_extracts(group_by_run(_without_final_outputs(events)))
    assert old["counterfactual"]["target_source"] == TARGET_RESULT
    assert old["assistant"] == new["assistant"] and old["messages"] == new["messages"]


def test_the_result_stands_in_only_when_it_is_the_submitted_value(tmp_path, caplog):
    # A `str` field that `output_model` parsed into a model: the result is a dict, the SUBMIT a str.
    class Strict(BaseModel):
        title: str

    class _Coerced(RLMTask):
        signature = "document: str -> out: str"
        output_field = "out"
        output_model = Strict

    class _Two(RLMTask):
        signature = "document: str -> out: Flat, note: str"
        output_field = "out"
        output_model = Flat

    coerced = _typed_run(tmp_path, _Coerced, '{"title": "t"}', "coerced")
    assert next(e for e in coerced if e["type"] == "result")["payload"]["output"] == {"title": "t"}
    # The repr is a literal here, so break it to reach the `result` fallback.
    for e in coerced:
        if e["type"] == "main_step" and e["payload"]["output"].startswith("FINAL: "):
            e["payload"]["output"] = "FINAL: {'out': Strict(title='t')}"

    # Two output fields: `result` holds only `out`.
    two, _ = _run(tmp_path, "two", _Two,
                  [*_TURNS[:1], {"reasoning": "done", "code": "SUBMIT(out=..., note='n')"}],
                  ["11", submit({"out": {"title": "t"}, "note": "n"})],
                  max_iterations=3, inputs={"document": "hello world"})

    # A result of the right shape, but the SUBMIT repr names another class.
    renamed = _typed_run(tmp_path, _Flat, {"title": "t"}, "renamed")
    for e in renamed:
        if e["type"] == "main_step" and e["payload"]["output"].startswith("FINAL: "):
            e["payload"]["output"] = e["payload"]["output"].replace("Flat(", "Other(")

    with caplog.at_level("WARNING", logger="rlm_harness.dataset"):
        for events in (coerced, two, renamed):
            assert export_counterfactual_extracts(group_by_run(_without_final_outputs(events))) == []
    assert caplog.text.count("skipped 1 record(s)") == 3


def test_runs_that_did_not_end_by_a_successful_submit_contribute_nothing(tmp_path, caplog):
    extracted, _ = _run(tmp_path, "extracted", _Plain, [*_TURNS, {"answer": "41", "score": 1}],
                        ["11", "2"], max_iterations=2)

    class Strict(BaseModel):
        title: str

    class _Strict(RLMTask):
        signature = "document: str -> out: str"
        output_field = "out"
        output_model = Strict

    failed, _ = _run(tmp_path, "failed", _Strict,
                     [*_TURNS[:1], {"reasoning": "bad", "code": "SUBMIT(out='nope')"}],
                     ["11", submit({"out": "nope"})], max_iterations=3,
                     inputs={"document": "hello world"})
    assert [e for e in extracted if e["type"] == EVENT_EXTRACT]
    # The failed run still has its SUBMIT trajectory; only the missing `result` excludes it.
    assert not [e for e in failed if e["type"] == "result"]
    assert next(e for e in failed if e["type"] == EVENT_FINAL)["payload"]["extract"] is False
    with caplog.at_level("WARNING", logger="rlm_harness.dataset"):
        assert export_counterfactual_extracts(group_by_run(extracted + failed)) == []
    assert "skipped" not in caplog.text


def test_the_other_exports_never_include_a_counterfactual(tmp_path):
    events, _ = _submit_run(tmp_path)
    runs = group_by_run(events)
    assert [r["kind"] for r in export_sft_turns(runs, render="adapter")] == ["action"] * 3
    assert all("assistant_source" not in r for r in export_sft_turns(runs))


def test_a_later_runs_result_never_vouches_for_a_failed_submit(tmp_path):
    """Two runs under one run_id: the first submitted and failed, the second succeeded without a
    capture, so its turns have no `rlm_setup`. Only the event right after a `final` is its result."""
    events, _ = _submit_run(tmp_path)
    failed = [e for e in events if e["type"] != "result"]
    later = [{"type": "main_step", "payload": {"turn": 0, "reasoning": "r", "code": "c",
                                               "output": "FINAL: {}"}},
             {"type": "final", "payload": {"final_reasoning": "r"}},
             {"type": "result", "payload": {"output": "x"}}]
    assert export_counterfactual_extracts({"r": failed + later}) == []
    # Two captured runs under one run_id each yield their own record.
    twice = [e for e in events if e["type"] != "run_end"] * 2
    assert len(export_counterfactual_extracts({"r": twice})) == 2


def test_an_earlier_cut_is_the_request_a_real_extract_over_that_history_sends(tmp_path):
    events, _ = _submit_run(tmp_path)
    cut_off, requests = _run(tmp_path, "one_turn", _Plain,
                             [_TURNS[0], {"answer": "41", "score": 1}], ["11"], max_iterations=1)
    real = next(e["payload"] for e in cut_off if e["type"] == EVENT_EXTRACT)
    [rec] = export_counterfactual_extracts(group_by_run(events), cut_back=(1,))
    assert rec["turn"] == 1 and rec["messages"] == requests[-1]
    assert messages_sha256(rec["messages"]) == real["messages_sha256"]


class Inner(BaseModel):
    x: int


class Nested(BaseModel):
    title: str
    inner: Inner
    items: list[Inner] = []


class _NestedTask(RLMTask):
    signature = "document: str -> out: Nested"
    output_field = "out"
    output_model = Nested


def test_a_nested_model_is_read_from_the_result_when_its_repr_matches(tmp_path):
    value = {"title": "t", "inner": {"x": 1}, "items": [{"x": 2}]}
    events = _typed_run(tmp_path, _NestedTask, value)
    old = _without_final_outputs(events)
    [rec] = export_counterfactual_extracts(group_by_run(old), custom_types={"Nested": Nested})
    assert rec["counterfactual"]["target_source"] == TARGET_RESULT
    assert json.loads(rec["assistant"]["content"]) == {"out": value}


def test_a_result_the_json_dump_or_output_model_changed_is_never_the_target(tmp_path, caplog):
    """Each `result` below differs from what was submitted, and the SUBMIT repr is what catches it."""
    import datetime
    import enum

    class Color(enum.Enum):
        RED = "red"

    class Stamped(BaseModel):
        when: datetime.datetime
        color: Color

    class _Stamped(RLMTask):
        signature = "document: str -> out: Stamped"
        output_field = "out"
        output_model = Stamped

    cases = [
        (_Stamped, {"when": "2024-01-02T03:04:05", "color": "red"}, {"Stamped": Stamped}),
    ]
    with caplog.at_level("WARNING", logger="rlm_harness.dataset"):
        for i, (task, value, types) in enumerate(cases):
            events = _typed_run(tmp_path, task, value, f"case{i}")
            assert next(e for e in events if e["type"] == "result")
            [rec] = export_counterfactual_extracts(group_by_run(events), custom_types=types)
            assert rec["counterfactual"]["target_source"] == TARGET_FINAL_OUTPUTS
            assert export_counterfactual_extracts(
                group_by_run(_without_final_outputs(events)), custom_types=types) == []
    assert caplog.text.count("skipped 1 record(s)") == 1

    # An `output_model` other than the annotation's class: a default field it adds.
    events = _without_final_outputs(_typed_run(tmp_path, _Flat, {"title": "t"}, "widened"))
    for e in events:
        if e["type"] == "result":
            e["payload"]["output"]["score"] = 0.5
    assert export_counterfactual_extracts(group_by_run(events)) == []

    # A model nested in a list, written by `json.dumps(default=str)` as its `str`, as `result` does
    # for a `list[Model]` output with no `output_model`.
    events = _without_final_outputs(_typed_run(
        tmp_path, _NestedTask, {"title": "t", "inner": {"x": 1}, "items": [{"x": 2}]}, "listed"))
    for e in events:
        if e["type"] == "result":
            e["payload"]["output"]["items"] = ["x=2"]
    assert export_counterfactual_extracts(group_by_run(events), custom_types={"Nested": Nested}) == []


def test_the_guides_end_to_end_snippet_runs(tmp_path, monkeypatch):
    """The snippet under "Using counterfactual extract records" in rlm_harness/README.md, run as
    written against one real trace in `traces/`."""
    import pathlib
    import re

    guide = (pathlib.Path(__file__).parent.parent / "rlm_harness" / "README.md").read_text("utf-8")
    section = guide.split("### Using counterfactual extract records", 1)[1]
    snippet = re.search(r"```python\n(.*?)```", section, re.DOTALL).group(1)

    (tmp_path / "traces").mkdir()
    _submit_run(tmp_path / "traces", "json")   # writes traces/submitted
    (tmp_path / "traces" / "submitted").rename(tmp_path / "traces" / "a.jsonl")
    # A run that really extracted, and one whose SUBMIT was rejected: the snippet keeps neither.
    _run(tmp_path / "traces", "b.jsonl", _Plain, [*_TURNS, {"answer": "41", "score": 1}],
         ["11", "2"], max_iterations=2)

    class Strict(BaseModel):
        title: str

    class _Strict(RLMTask):
        signature = "document: str -> out: str"
        output_field = "out"
        output_model = Strict

    _run(tmp_path / "traces", "c.jsonl", _Strict,
         [*_TURNS[:1], {"reasoning": "bad", "code": "SUBMIT(out='nope')"}],
         ["11", submit({"out": "nope"})], max_iterations=3, inputs={"document": "hello world"})
    monkeypatch.chdir(tmp_path)
    exec(compile(snippet, "README snippet", "exec"), {})  # noqa: S102 - the repo's own guide

    rows = [json.loads(line) for line in (tmp_path / "sft.jsonl").read_text("utf-8").splitlines()]
    # a: three action turns and its counterfactual. b ended by extract but succeeded: its two action
    # turns, not its real extract record. c failed: nothing.
    # The snippet keys runs by the path glob returns, which uses the platform's separator.
    assert [(r["run_id"].replace(os.sep, "/"), r["source"], r["kind"]) for r in rows] == [
        ("traces/a.jsonl:submitted", "real", "action")] * 3 + [
        ("traces/b.jsonl:b.jsonl", "real", "action")] * 2 + [
        ("traces/a.jsonl:submitted", "counterfactual", "extract")]
    assert rows[-1]["assistant_source"] == SOURCE_COUNTERFACTUAL and rows[-1]["verified"] is None
