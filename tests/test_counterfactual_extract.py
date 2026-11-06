"""`export_counterfactual_extracts` builds the extract call a SUBMIT run never made.

The reference is a real extract: a second run with the same action turns and a smaller
`max_iterations` hits the budget, so dspy sends the extract request itself. A counterfactual cut
before the first run's SUBMIT has the same history, so it must be that request byte for byte, and
hash to the `messages_sha256` the second run recorded.
"""

from __future__ import annotations

import asyncio
import json

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


def test_a_typed_output_needs_final_outputs(tmp_path, caplog):
    events, _ = _run(tmp_path, "flat", _Flat,
                     [*_TURNS[:1], {"reasoning": "done", "code": "SUBMIT(out=...)"}],
                     ["11", submit({"out": {"title": "t", "tags": ["a"]}})],
                     max_iterations=3, inputs={"document": "hello world"})
    [rec] = export_counterfactual_extracts(group_by_run(events))
    assert json.loads(rec["assistant"]["content"]) == {"out": {"title": "t", "tags": ["a"]}}

    for e in events:
        if e["type"] == EVENT_FINAL:
            del e["payload"]["outputs"]
    with caplog.at_level("WARNING", logger="rlm_harness.dataset"):
        assert export_counterfactual_extracts(group_by_run(events)) == []
    assert "skipped 1 record(s)" in caplog.text   # `Flat(title=...)` is not a literal


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
