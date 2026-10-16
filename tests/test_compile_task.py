"""`optimize.compile_task`: the kit owns the wiring, the consumer owns the metric.

Every test here runs with a FAKE optimizer and a fake program, because a real one is a model call
per example and this suite never spends money. What that leaves testable is exactly the kit's own
share: the paired holdout, what gets handed to GEPA, the fingerprinted save, and the refusals.

The measurements these choices came from are in `docs/PROMPT.md`: a paired before/after is the
only thing a single session can report, and a consumer corpus of 153 runs had its obvious quality
metric saturated at a perfect score on 151 of them.
"""

from __future__ import annotations

import json
import pathlib

import pytest

import rlm_harness

dspy = pytest.importorskip("dspy")

from rlm_harness import optimize
from rlm_harness.testing import ScriptedInterpreter


class _Cite(rlm_harness.RLMTask):
    signature = "q: str -> answer: str"
    output_field = "answer"
    instructions = "Answer the question."


class _Pred:
    def __init__(self, answer: str) -> None:
        self.answer = answer


class _FakeProgram:
    """Stands in for the built `dspy.RLM`: two predictors, and a callable that costs nothing."""

    def __init__(self, answer: str = "shipped") -> None:
        self.answer = answer
        self.calls: list[dict] = []
        self.generate_action = _Predictor("planner text")
        self.extract = _Predictor("extract text")

    def named_predictors(self):
        return [("generate_action", self.generate_action), ("extract", self.extract)]

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return _Pred(self.answer)


class _Predictor:
    def __init__(self, text: str) -> None:
        self.signature = _Sig(text)


class _Sig:
    def __init__(self, instructions: str) -> None:
        self.instructions = instructions

    def with_instructions(self, text: str) -> _Sig:
        return _Sig(text)


def _example(q: str, answer: str):
    return dspy.Example(q=q, answer=answer).with_inputs("q")


def _metric(example, prediction, *a, **k) -> float:
    return 1.0 if getattr(prediction, "answer", None) == example.answer else 0.0


@pytest.fixture
def task(tmp_path):
    rlm_harness.configure(rlm_harness.RLMConfig(
        main_model="openai/x", sub_model="openai/x", api_key="k", interpreter="mock",
        max_retries=1,
    ))
    return _Cite(interpreter=ScriptedInterpreter([]))


@pytest.fixture
def fake_gepa(monkeypatch):
    """A GEPA that returns an improved program, and records what it was handed."""
    seen: dict = {}

    class _GEPA:
        def __init__(self, **kwargs):
            seen["init"] = kwargs

        def compile(self, program, **kwargs):
            seen["compile"] = kwargs
            seen["program"] = program
            improved = _FakeProgram(answer="right")
            improved.generate_action = _Predictor("OPTIMIZED planner text")
            return improved

    monkeypatch.setattr(dspy, "GEPA", _GEPA)
    return seen


def test_a_missing_reflection_lm_is_refused_in_the_kits_words(task):
    """dspy raises a bare `AssertionError` for this, which says nothing about what to pass."""
    with pytest.raises(ValueError, match="needs reflection_lm"):
        optimize.compile_task(task, [_example("q", "a")], _metric)


def test_an_empty_trainset_is_refused(task):
    with pytest.raises(ValueError, match="non-empty trainset"):
        optimize.compile_task(task, [], _metric, reflection_lm=object())


def test_the_rlm_goes_to_gepa_unwrapped(task, fake_gepa, monkeypatch):
    """A wrapper is what a consumer reaches for, and it costs them the artifact.

    dspy prefixes each predictor with the attribute it hangs off, so a state saved from a module
    holding the RLM as `self.rlm` names `rlm.generate_action`, and `compiled_instructions` refuses
    it for naming no predictor this build has. Passing the RLM itself is what avoids that.
    """
    built = _FakeProgram()
    monkeypatch.setattr(type(task), "_build_rlm", lambda self: built)
    optimize.compile_task(task, [_example("q", "a")], _metric, reflection_lm=object())
    assert fake_gepa["program"] is built, "the program handed to GEPA was not the built RLM"
    assert [n for n, _ in fake_gepa["program"].named_predictors()] == [
        "generate_action", "extract"
    ], "the predictors GEPA mutates are not the RLM's own"


def test_the_holdout_is_measured_before_and_after_in_one_process(task, fake_gepa, monkeypatch):
    """The paired difference is the result. An absolute score from one session is not: the same
    artifact over the same examples scored 0.575 and 0.596 in a later session against a recorded
    0.733, so session drift is the size of the gain."""
    monkeypatch.setattr(type(task), "_build_rlm", lambda self: _FakeProgram(answer="wrong"))
    held = [_example("q1", "right"), _example("q2", "right")]
    result = optimize.compile_task(task, [_example("t", "right")], _metric,
                                   reflection_lm=object(), holdout=held)
    assert result.holdout_before == 0.0
    assert result.holdout_after == 1.0
    assert result.delta == 1.0
    assert result.per_example == {"before": [0.0, 0.0], "after": [1.0, 1.0]}
    assert result.rollouts == {"trainset": 1, "holdout_before": 2, "holdout_after": 2}


def test_without_a_holdout_the_delta_is_None_rather_than_a_number(task, fake_gepa, monkeypatch):
    """There is nothing to subtract, and a plausible-looking figure in that slot is worse than an
    absent one: it would be read as a result by everyone who did not write it."""
    monkeypatch.setattr(type(task), "_build_rlm", lambda self: _FakeProgram())
    result = optimize.compile_task(task, [_example("q", "a")], _metric, reflection_lm=object())
    assert result.delta is None
    assert result.holdout_before is None and result.holdout_after is None
    assert result.per_example is None
    assert result.program is not None, "a compile still happened"


def test_per_example_scores_survive_so_a_mean_cannot_hide_a_regression(task, fake_gepa,
                                                                      monkeypatch):
    """A compile that raises the mean while making one example worse is a different decision from
    one that raises all of them, and the mean cannot tell them apart."""
    monkeypatch.setattr(type(task), "_build_rlm", lambda self: _FakeProgram(answer="right"))
    held = [_example("q1", "right"), _example("q2", "other")]
    result = optimize.compile_task(task, [_example("t", "right")], _metric,
                                   reflection_lm=object(), holdout=held)
    assert result.per_example["before"] == [1.0, 0.0]
    assert result.per_example["after"] == [1.0, 0.0]


def test_the_artifact_is_saved_with_its_fingerprint(task, fake_gepa, monkeypatch, tmp_path):
    """Saved any other way, `compiled_instructions` refuses it for carrying no record of what it
    was compiled against, and the refusal is correct: nothing in the state answers that."""
    monkeypatch.setattr(type(task), "_build_rlm", lambda self: _FakeProgram())
    saved: dict = {}

    def _save(program, path, task=None):
        saved["program"], saved["path"], saved["task"] = program, path, task
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"rlm_harness": task.compiled_fingerprint(program)}, fh)

    monkeypatch.setattr(optimize, "save_program", _save)
    out = str(tmp_path / "artifact.json")
    result = optimize.compile_task(task, [_example("q", "a")], _metric,
                                   reflection_lm=object(), save_to=out)
    assert result.artifact == out
    assert saved["task"] is task, "save_program was called without the task, so no fingerprint"
    assert saved["program"] is result.program, "the saved program is not the compiled one"
    assert optimize.FINGERPRINT_KEY in json.loads(pathlib.Path(out).read_text())


def test_no_save_to_means_no_file_and_no_artifact_path(task, fake_gepa, monkeypatch):
    monkeypatch.setattr(type(task), "_build_rlm", lambda self: _FakeProgram())
    calls = []
    monkeypatch.setattr(optimize, "save_program", lambda *a, **k: calls.append(a))
    result = optimize.compile_task(task, [_example("q", "a")], _metric, reflection_lm=object())
    assert result.artifact is None
    assert not calls


def test_the_consumers_output_audit_runs_on_both_sides(task, fake_gepa, monkeypatch):
    """The kit cannot audit outputs it does not understand. On a real corpus two distinct ways of
    padding a page both held the quality metric at its ceiling, and one was visible only by
    reading the prose, so this callable is the consumer's and the kit only carries its answer."""
    monkeypatch.setattr(type(task), "_build_rlm", lambda self: _FakeProgram(answer="shipped"))
    seen = []

    def audit(examples, predictions):
        seen.append([p.answer for p in predictions])
        return {"answers": [p.answer for p in predictions]}

    held = [_example("q1", "right")]
    result = optimize.compile_task(task, [_example("t", "right")], _metric,
                                   reflection_lm=object(), holdout=held, audit_outputs=audit)
    assert seen == [["shipped"], ["right"]], "the audit did not see both sides"
    assert result.output_audit == {"before": {"answers": ["shipped"]},
                                   "after": {"answers": ["right"]}}


def test_no_audit_callable_leaves_the_field_absent_rather_than_empty(task, fake_gepa, monkeypatch):
    monkeypatch.setattr(type(task), "_build_rlm", lambda self: _FakeProgram())
    result = optimize.compile_task(task, [_example("q", "a")], _metric,
                                   reflection_lm=object(), holdout=[_example("h", "a")])
    assert result.output_audit is None, "an empty dict would read as an audit that found nothing"


def test_reflection_lm_and_auto_reach_gepa_and_the_trainset_reaches_compile(task, fake_gepa,
                                                                           monkeypatch):
    monkeypatch.setattr(type(task), "_build_rlm", lambda self: _FakeProgram())
    lm = object()
    train = [_example("a", "1"), _example("b", "2")]
    optimize.compile_task(task, train, _metric, reflection_lm=lm, auto="medium",
                          max_metric_calls=7)
    assert fake_gepa["init"]["reflection_lm"] is lm
    assert fake_gepa["init"]["auto"] == "medium"
    assert fake_gepa["init"]["max_metric_calls"] == 7, "extra gepa kwargs were dropped"
    assert fake_gepa["init"]["metric"] is _metric
    assert fake_gepa["compile"]["trainset"] == train
