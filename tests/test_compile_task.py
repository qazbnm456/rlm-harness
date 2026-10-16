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
    assert result.rollouts == {"trainset_examples": 1, "holdout_examples": 2,
                               "holdout_rollouts": 4}, (
        "the counts must say what they count: `trainset` read as the compile's cost, which is "
        "GEPA's own budget and about 790 metric calls at auto='light', not the trainset's size"
    )
    assert result.failed_rollouts == {"before": 0, "after": 0}


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


def test_a_failing_holdout_rollout_scores_rather_than_discarding_the_compile(task, fake_gepa,
                                                                             monkeypatch,
                                                                             tmp_path):
    """These rollouts are not behind `arun`'s retry, so one provider 500 on the last held-out
    example used to escape before the artifact was saved, taking an `auto="light"` compile of
    about 790 metric calls with it. dspy tolerates the same failure on the same rollouts."""
    class _Flaky(_FakeProgram):
        def __call__(self, **kw):
            if kw.get("q") == "boom":
                raise RuntimeError("litellm.InternalServerError: provider returned 500")
            return super().__call__(**kw)

    monkeypatch.setattr(type(task), "_build_rlm", lambda self: _Flaky(answer="right"))
    monkeypatch.setattr(optimize, "save_program", lambda *a, **k: None)
    out = str(tmp_path / "a.json")
    result = optimize.compile_task(
        task, [_example("t", "right")], _metric, reflection_lm=object(),
        holdout=[_example("ok", "right"), _example("boom", "right")], save_to=out)
    assert result.artifact == out, "the artifact was lost to a transient rollout error"
    assert result.failed_rollouts == {"before": 1, "after": 0}
    assert result.per_example["before"] == [1.0, 0.0], "the failure scored, rather than aborting"


def test_the_artifact_is_saved_before_the_after_evaluation_can_lose_it(monkeypatch, task,
                                                                      tmp_path):
    """Everything downstream of a complete artifact should be unable to discard it.

    The fake optimizer has to return a COUNTED program here. The first version let the default
    fake return a plain one, so the after-evaluation's rollout was invisible and the assertion
    that something ran after the save could not hold: the test's own instrument hid the half it
    was checking.
    """
    order = []

    class _Counted(_FakeProgram):
        def __call__(self, **kw):
            order.append("rollout")
            return super().__call__(**kw)

    class _GEPA:
        def __init__(self, **kw): pass
        def compile(self, program, **kw): return _Counted(answer="right")

    monkeypatch.setattr(dspy, "GEPA", _GEPA)
    monkeypatch.setattr(optimize, "save_program", lambda *a, **k: order.append("save"))
    monkeypatch.setattr(type(task), "_build_rlm", lambda self: _Counted())
    optimize.compile_task(task, [_example("t", "a")], _metric, reflection_lm=object(),
                          holdout=[_example("h", "a")], save_to=str(tmp_path / "a.json"))
    assert order == ["rollout", "save", "rollout"], (
        f"the artifact must be saved between the two evaluations, not after both: {order}"
    )


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


def test_auto_alone_reaches_gepa_with_the_metric_and_the_trainset(task, fake_gepa, monkeypatch):
    monkeypatch.setattr(type(task), "_build_rlm", lambda self: _FakeProgram())
    lm = object()
    train = [_example("a", "1"), _example("b", "2")]
    optimize.compile_task(task, train, _metric, reflection_lm=lm, auto="medium")
    assert fake_gepa["init"]["reflection_lm"] is lm
    assert fake_gepa["init"]["auto"] == "medium"
    assert fake_gepa["init"]["metric"] is _metric
    assert fake_gepa["compile"]["trainset"] == train


def test_a_budget_kwarg_displaces_auto_because_dspy_refuses_both(task, fake_gepa, monkeypatch):
    """dspy asserts EXACTLY one of `max_metric_calls`, `max_full_evals` and `auto` is set, so
    passing a budget beside the default `auto="light"` raised `AssertionError` out of the
    constructor. The first version of this test pinned that very combination, and the fake GEPA
    accepted it because a fake accepts anything: the gate passed on an input the real class
    rejects, which is the failure this release's own commit messages keep naming.
    """
    monkeypatch.setattr(type(task), "_build_rlm", lambda self: _FakeProgram())
    optimize.compile_task(task, [_example("a", "1")], _metric, reflection_lm=object(),
                          max_metric_calls=7)
    assert fake_gepa["init"]["max_metric_calls"] == 7, "the budget kwarg was dropped"
    assert "auto" not in fake_gepa["init"], (
        "auto was passed beside a budget kwarg, which real dspy.GEPA refuses with an "
        "AssertionError from its constructor"
    )


def test_auto_None_leaves_the_budget_to_the_caller(task, fake_gepa, monkeypatch):
    monkeypatch.setattr(type(task), "_build_rlm", lambda self: _FakeProgram())
    optimize.compile_task(task, [_example("a", "1")], _metric, reflection_lm=object(),
                          auto=None, max_full_evals=3)
    assert "auto" not in fake_gepa["init"]
    assert fake_gepa["init"]["max_full_evals"] == 3


def test_the_optimizer_is_built_before_anything_is_evaluated(task, monkeypatch):
    """Its constructor validates the metric's arity and the budget combination for free. Built
    after the before-evaluation, as it first was, those same exceptions arrive at the price of one
    FULL rollout per held-out example."""
    order = []

    class _GEPA:
        def __init__(self, **kw):
            order.append("gepa")
        def compile(self, program, **kw):
            return _FakeProgram()

    monkeypatch.setattr(dspy, "GEPA", _GEPA)

    class _Watched(_FakeProgram):
        def __call__(self, **kw):
            order.append("rollout")
            return super().__call__(**kw)

    monkeypatch.setattr(type(task), "_build_rlm", lambda self: _Watched())
    optimize.compile_task(task, [_example("a", "1")], _metric, reflection_lm=object(),
                          holdout=[_example("h", "1")])
    assert order[0] == "gepa", f"a rollout was paid for before the optimizer validated: {order}"


def test_exact_field_metric_says_so_rather_than_scoring_a_flat_zero(tmp_path):
    """The two new features' defaults did not compose: `trainset_from_traces` stores a non-dict
    `label` return at `.label`, and `exact_field_metric("answer")` over that trainset found
    nothing on every example and returned 0.0 for all of them. A metric that is identically zero
    gives GEPA no gradient, which is the failure this module warns about, and the whole budget
    would have been spent discovering it."""
    metric = optimize.exact_field_metric("answer")
    good = dspy.Example(q="x", answer="right").with_inputs("q")
    assert metric(good, _Pred("right")) == 1.0
    assert metric(good, _Pred("wrong")) == 0.0

    mislabelled = dspy.Example(q="x", label="right").with_inputs("q")
    with pytest.raises(AttributeError, match="no 'answer' on the example"):
        metric(mislabelled, _Pred("right"))


def test_a_metric_that_cannot_score_this_trainset_raises_at_zero_rollouts(task, fake_gepa,
                                                                          monkeypatch):
    """`dspy.GEPA.__init__` validates the metric's ARITY for free; this validates that it can
    score THIS trainset. A metric looking for a field the examples do not carry scores every
    rollout zero, gives the optimizer no gradient, and GEPA's own executor swallows the exception
    on the trainset side, so the whole budget goes before anyone learns why."""
    rolled = []

    class _Watched(_FakeProgram):
        def __call__(self, **kw):
            rolled.append(kw)
            return super().__call__(**kw)

    monkeypatch.setattr(type(task), "_build_rlm", lambda self: _Watched())
    mislabelled = dspy.Example(q="x", label="right").with_inputs("q")
    with pytest.raises(AttributeError, match="no 'answer' on the example"):
        optimize.compile_task(task, [mislabelled], optimize.exact_field_metric("answer"),
                              reflection_lm=object(), holdout=[mislabelled])
    assert rolled == [], f"rollouts were paid before the metric was probed: {rolled}"


def test_a_metric_raising_is_not_scored_as_a_failed_rollout(task, fake_gepa, monkeypatch):
    """The guard around a rollout used to cover the metric too, so the raise added in this same
    release to stop a mis-named gold burning a whole budget was caught, logged and scored 0.0.
    A rollout that failed is a run that scored badly; a metric that cannot score is a
    configuration error, and only one of those is worth continuing through."""
    monkeypatch.setattr(type(task), "_build_rlm", lambda self: _FakeProgram())

    def _angry(example, prediction, *a, **k):
        raise AttributeError("this metric cannot score this trainset")

    with pytest.raises(AttributeError, match="cannot score this trainset"):
        optimize.compile_task(task, [_example("q", "a")], _angry, reflection_lm=object(),
                              holdout=[_example("h", "a")])


def test_an_audit_that_raises_does_not_discard_the_measurement(task, fake_gepa, monkeypatch,
                                                               tmp_path):
    """A failing rollout puts `None` in the predictions, and the kit's own documented idiom
    (`[p.answer for p in predictions]`) raises on it. On the after side that runs downstream of
    the save, so it left the artifact on disk and took the paired delta with it."""
    class _Flaky(_FakeProgram):
        def __call__(self, **kw):
            if kw.get("q") == "boom":
                raise RuntimeError("provider 500")
            return super().__call__(**kw)

    monkeypatch.setattr(type(task), "_build_rlm", lambda self: _Flaky(answer="right"))
    monkeypatch.setattr(optimize, "save_program", lambda *a, **k: None)
    result = optimize.compile_task(
        task, [_example("t", "right")], _metric, reflection_lm=object(),
        holdout=[_example("boom", "right")],
        audit_outputs=lambda ex, preds: {"answers": [p.answer for p in preds]},
        save_to=str(tmp_path / "a.json"))
    assert result.delta is not None, "the paired measurement was discarded by the audit"
    assert "error" in result.output_audit["before"], "the audit's failure was not recorded"
    # One, not two: the compiled program the fake optimizer returns does not raise, so only the
    # BEFORE arm lost a rollout. Which is the point of counting per side rather than in total.
    assert result.failed_rollouts == {"before": 1, "after": 0}


def test_an_explicit_auto_beside_a_budget_kwarg_is_a_conflict(task, fake_gepa, monkeypatch):
    """Displacing a default nobody chose is the kit's own pattern; silently dropping one of two
    things the caller asked for is not. dspy names all three in its own assertion and it now
    costs nothing, because the optimizer is built before any rollout."""
    monkeypatch.setattr(type(task), "_build_rlm", lambda self: _FakeProgram())
    optimize.compile_task(task, [_example("a", "1")], _metric, reflection_lm=object(),
                          auto="heavy", max_metric_calls=7)
    assert fake_gepa["init"]["auto"] == "heavy", "an explicit auto was dropped"
    assert fake_gepa["init"]["max_metric_calls"] == 7
    # dspy asserts on the pair; the kit passes both so its message, which names all three, is
    # what the caller sees, at zero rollouts.


def test_an_unknown_auto_is_refused_before_the_holdout_is_spent(task, fake_gepa, monkeypatch):
    """dspy accepts any string in its constructor and raises `KeyError` from `compile()`, which
    is after the before-evaluation."""
    monkeypatch.setattr(type(task), "_build_rlm", lambda self: _FakeProgram())
    with pytest.raises(ValueError, match="auto='turbo' is not one of"):
        optimize.compile_task(task, [_example("a", "1")], _metric, reflection_lm=object(),
                              auto="turbo", holdout=[_example("h", "1")])


def test_a_None_valued_kwarg_that_is_not_a_budget_key_still_reaches_gepa(task, fake_gepa,
                                                                        monkeypatch):
    """`seed=None` and `max_merge_invocations=None` are accepted by dspy and have non-None
    defaults, so dropping them gave a caller who explicitly asked for randomness determinism."""
    monkeypatch.setattr(type(task), "_build_rlm", lambda self: _FakeProgram())
    optimize.compile_task(task, [_example("a", "1")], _metric, reflection_lm=object(), seed=None)
    assert "seed" in fake_gepa["init"] and fake_gepa["init"]["seed"] is None
