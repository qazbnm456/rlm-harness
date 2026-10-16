"""`optimize.trainset_from_traces`: the questions people actually asked, already on disk.

The filter is the point. A corpus of 161 real runs carried 8 that failed before the planner ever
returned a prediction, and such a run has ZERO `main_step` events, because `RLMTask.arun` records
the main trajectory off that prediction. So it looks exactly like a run that finished instantly,
and a trainset built without that check teaches the optimizer that dying early is cheap.
"""

from __future__ import annotations

import json

import pytest

from rlm_harness import optimize

dspy = pytest.importorskip("dspy")


def _write(path, *, meta=None, turns=1, end=True, schema="rlm-harness/trace/v2"):
    events = [{"schema": schema, "run_id": path.stem, "step_id": 0, "ts": 1.0,
               "type": "run_start", "payload": {"meta": meta or {}, "rlm_harness": "1.20.0"}}]
    for i in range(turns):
        events.append({"schema": schema, "run_id": path.stem, "step_id": i + 1, "ts": 2.0 + i,
                       "type": "main_step", "payload": {"turn": i, "code": "pass"}})
    if end:
        events.append({"schema": schema, "run_id": path.stem, "step_id": 99, "ts": 99.0,
                       "type": "run_end", "payload": {"ok": turns > 0}})
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return path


def test_a_finished_run_becomes_an_example_with_its_inputs_declared(tmp_path):
    _write(tmp_path / "a.jsonl", meta={"question": "where is the lock?"})
    build = optimize.trainset_from_traces(
        tmp_path, inputs=["question"], label=lambda ev: {"answer": "in record()"})
    assert len(build) == 1
    ex = build.examples[0]
    assert ex.inputs().toDict() == {"question": "where is the lock?"}, (
        "the inputs must be DECLARED, or GEPA passes the gold answer to the program as an input"
    )
    assert ex.answer == "in record()"


def test_a_run_that_never_finished_is_dropped(tmp_path):
    """`run_end` absent: the process died, and whatever turns it managed are not a result."""
    _write(tmp_path / "ok.jsonl", meta={"q": "a"})
    _write(tmp_path / "died.jsonl", meta={"q": "b"}, end=False)
    build = optimize.trainset_from_traces(tmp_path, inputs=["q"], label=lambda ev: "gold")
    assert len(build) == 1
    assert build.dropped == {"no run_end": 1}


def test_a_failed_run_is_dropped_even_though_it_reached_run_end(tmp_path):
    """The shape that actually occurs, and the reason the turn check is not redundant.

    All 8 failures in a real 161-run corpus ended in `run_end` with `ok: false` and zero
    `main_step` events, because `arun` records the main trajectory off the final prediction and
    these failed before there was one. `run_end` is written from a `finally`, so checking only for
    it would have taken every one of them as a zero-cost success.
    """
    _write(tmp_path / "ok.jsonl", meta={"q": "a"})
    _write(tmp_path / "failed.jsonl", meta={"q": "b"}, turns=0)
    build = optimize.trainset_from_traces(tmp_path, inputs=["q"], label=lambda ev: "gold")
    assert len(build) == 1
    assert build.dropped == {"no main_step": 1}


def test_the_drop_counts_are_part_of_the_result_not_a_log(tmp_path):
    """A build that silently returns 153 of 161 is a corpus-validity question nobody asked."""
    _write(tmp_path / "a.jsonl", meta={"q": "a"})
    _write(tmp_path / "b.jsonl", meta={"q": "b"}, turns=0)
    _write(tmp_path / "c.jsonl", meta={}, turns=1)
    (tmp_path / "junk.jsonl").write_text("not json at all\n")
    build = optimize.trainset_from_traces(tmp_path, inputs=["q"], label=lambda ev: "gold")
    assert build.sources == 4
    assert len(build) == 1
    assert build.dropped["no main_step"] == 1
    assert build.dropped["inputs missing"] == 1
    assert sum(build.dropped.values()) == 3


def test_inputs_can_be_a_callable_because_consumers_disagree_on_where_the_input_lives(tmp_path):
    """Across five real consumers it was `meta.task`, a JSON blob at `meta.event`, and
    `meta.lead`. A fixed path would fit one of them."""
    _write(tmp_path / "a.jsonl",
           meta={"event": json.dumps({"_meta": {"repo": "owner/name", "number": 7}})})

    def lift(events):
        meta = next(e for e in events if e["type"] == "run_start")["payload"]["meta"]
        blob = json.loads(meta["event"])["_meta"]
        return {"change": f"{blob['repo']}#{blob['number']}"}

    build = optimize.trainset_from_traces(tmp_path, inputs=lift, label=lambda ev: "benign")
    assert build.examples[0].change == "owner/name#7"
    assert build.examples[0].inputs().toDict() == {"change": "owner/name#7"}


def test_a_label_that_declines_drops_only_its_own_run(tmp_path):
    """Returning None is how a consumer says "this run has no gold", which is most of them in a
    corpus where only some outputs were ever judged."""
    for n in "abc":
        _write(tmp_path / f"{n}.jsonl", meta={"q": n})
    build = optimize.trainset_from_traces(
        tmp_path, inputs=["q"],
        label=lambda ev: None if ev[0]["run_id"] == "b" else "gold")
    assert len(build) == 2
    assert build.dropped == {"label declined": 1}


def test_a_label_that_raises_drops_its_run_rather_than_the_build(tmp_path):
    """One unscoreable run in a corpus of hundreds is not a reason to produce nothing, and a
    traceback here would arrive after the caller had already paid to record them."""
    for n in "abc":
        _write(tmp_path / f"{n}.jsonl", meta={"q": n})

    def label(events):
        if events[0]["run_id"] == "b":
            raise KeyError("this consumer's manifest has no row for b")
        return "gold"

    build = optimize.trainset_from_traces(tmp_path, inputs=["q"], label=label)
    assert len(build) == 2
    assert build.dropped == {"label raised": 1}


def test_a_dict_label_becomes_several_gold_fields(tmp_path):
    _write(tmp_path / "a.jsonl", meta={"q": "a"})
    build = optimize.trainset_from_traces(
        tmp_path, inputs=["q"], label=lambda ev: {"answer": "x", "citation": "f.py:1"})
    ex = build.examples[0]
    assert (ex.answer, ex.citation) == ("x", "f.py:1")
    assert ex.inputs().toDict() == {"q": "a"}, "a gold field leaked into the inputs"


def test_require_can_be_loosened_and_the_caller_sees_what_that_admits(tmp_path):
    """Stated as a test because the default is the safe one and overriding it is a decision."""
    _write(tmp_path / "failed.jsonl", meta={"q": "b"}, turns=0)
    strict = optimize.trainset_from_traces(tmp_path, inputs=["q"], label=lambda ev: "g")
    loose = optimize.trainset_from_traces(tmp_path, inputs=["q"], label=lambda ev: "g",
                                          require=("run_end",))
    assert len(strict) == 0 and len(loose) == 1


def test_a_single_file_and_a_directory_are_both_accepted(tmp_path):
    f = _write(tmp_path / "a.jsonl", meta={"q": "a"})
    one = optimize.trainset_from_traces(str(f), inputs=["q"], label=lambda ev: "g")
    many = optimize.trainset_from_traces([str(tmp_path)], inputs=["q"], label=lambda ev: "g")
    assert len(one) == len(many) == 1


def test_a_v1_trace_still_builds(tmp_path):
    """Every consumer corpus measured so far is `trace/v1`, two of them under the pre-rename name,
    so a builder that only read v2 would read none of the data that exists."""
    _write(tmp_path / "old.jsonl", meta={"q": "a"}, schema="rlm-kit/trace/v1")
    build = optimize.trainset_from_traces(tmp_path, inputs=["q"], label=lambda ev: "g")
    assert len(build) == 1


def test_a_callable_returning_a_non_mapping_drops_rather_than_crashes(tmp_path):
    """`got.values()` sat outside the try, so this raised `AttributeError` mid-corpus, after every
    file had been read, in a function whose whole posture is drop-not-crash."""
    _write(tmp_path / "a.jsonl", meta={"q": "a"})
    build = optimize.trainset_from_traces(tmp_path, inputs=lambda ev: ["not", "a", "mapping"],
                                          label=lambda ev: "gold")
    assert len(build) == 0
    assert build.dropped == {"inputs not a mapping": 1}


def test_a_gold_field_colliding_with_an_input_drops_rather_than_crashes(tmp_path):
    """`dspy.Example(**got, **gold)` raised `TypeError: got multiple values for keyword argument`."""
    _write(tmp_path / "a.jsonl", meta={"q": "a"})
    build = optimize.trainset_from_traces(tmp_path, inputs=["q"],
                                          label=lambda ev: {"q": "clashes with the input"})
    assert len(build) == 0
    assert build.dropped == {"gold collides with an input": 1}
