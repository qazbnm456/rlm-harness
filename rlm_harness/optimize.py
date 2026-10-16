"""GEPA optimization harness: compile a task's instructions against your own metric.

The point of building on dspy is that a task can be *compiled*, its instructions optimised
against a metric, rather than hand-tuned forever. This module owns the wiring and leaves you the
one thing it cannot have an opinion about.

- :func:`compile_task` runs ``dspy.GEPA`` over a task's own program, measures a holdout before and
  after IN ONE PROCESS, and saves the artifact with the fingerprint
  :attr:`~rlm_harness.task.RLMTask.compiled_instructions` needs to accept it back.
- :func:`audit_instruction` reads a compiled instruction for what the score cannot see: trainset
  answers copied into it, and capabilities dropped on the way. ``compile_task`` produces it.
- :func:`trainset_from_traces` builds a trainset out of runs this kit already recorded, with the
  gold value coming from a callable of yours.
- :func:`save_program` / :func:`load_program` wrap dspy persistence; ``save_program(..., task=)``
  is what writes that fingerprint.
- ``exact_field_metric`` and ``schema_valid_metric`` are metric templates, usable where an exact
  label exists.

**The metric is yours, and it is the part that decides whether the money bought anything.** Two
measured warnings, both in ``docs/PROMPT.md``: a plausible quality signal on a real consumer
corpus scored a perfect 1.000 on 151 of 153 examples, which leaves an optimizer nothing to climb;
and the cost axis that did have a gradient was satisfiable by doing less work rather than better
work. Measure your metric's spread before spending a compile on it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)

#: Distinguishes "the caller did not mention `auto`" from "the caller asked for `auto`". The
#: default yields to an explicit budget kwarg; an explicit `auto` beside one is a conflict.
_UNSET = object()

#: dspy accepts any string for `auto` in its constructor and raises `KeyError` from `compile()`,
#: which is after the holdout has been spent.
_AUTO_SETTINGS = ("light", "medium", "heavy")


# A metric scores a rollout. NOTE: this alias is documentation, not a gate -- no type checker runs
# in CI, so nothing makes a wrong annotation here fail. What does fail is `dspy.GEPA.__init__`,
# which `compile_task` now constructs before spending anything precisely so that it validates the
# real arity for free. The shape is dspy's, not a simplification of it: `dspy.GEPA.__init__`
# does `inspect.signature(metric).bind(None, None, None, None, None)` and raises on failure, while
# `Evaluate` calls it with two arguments and `bootstrap_trace_data` with three. So the only working
# signature is `(gold, pred, trace=None, pred_name=None, pred_trace=None)`, or `*args`. A type that
# said `(example, prediction)` endorsed, through `py.typed`, the one shape guaranteed to fail.
Metric = Callable[..., float]


class _ProbePrediction:
    """A prediction stub that records whether it was read. Every attribute answers ``None``.

    The recording is the whole point. The pre-flight asks whether a metric can score a trainset
    AT ALL, and the exception type cannot answer that: ``None.strip()`` and ``None.get(...)`` both
    raise ``AttributeError``, exactly as a missing field on the example does, so refusing on the
    type alone rejected two of five ordinary metric shapes at zero rollouts. A second stub
    mirroring the example's fields does not close it either, because a metric may read a
    prediction field the example never had.

    What separates them is WHEN: a metric that raises without ever looking at the prediction
    failed on the example, and no prediction could have saved it. One that read the prediction
    first is reacting to the stub, which says nothing about the trainset.
    """

    def __init__(self) -> None:
        self.touched = False

    def __getattr__(self, name: str) -> None:
        # Only called for attributes not found normally, so `touched` itself never trips it.
        self.touched = True


def _metric_cannot_score(metric: Metric, example: Any) -> bool:
    """Would this metric raise on this trainset no matter what the model returned?

    True only when the metric raised an ``AttributeError`` BEFORE reading the prediction. Returns
    a bool rather than raising, so the caller owns the message; any other exception is the metric
    reacting to a stub and says nothing either way.
    """
    probe = _ProbePrediction()
    try:
        metric(example, probe)
    except AttributeError:
        return not probe.touched
    except Exception:
        return False
    return False


def exact_field_metric(field: str) -> Metric:
    """Metric: 1.0 when ``prediction.field == example.field``, else 0.0.

    Works with dspy.Example / dspy.Prediction (attribute access) and plain objects, so it is
    testable without dspy.

    **A missing field on the EXAMPLE raises rather than scoring zero**, because the two are not
    the same thing and only one of them is worth paying for. The mistake it catches composes two
    of this module's own defaults: :func:`trainset_from_traces` stores a non-dict ``label``
    return at ``example.label``, so ``exact_field_metric("answer")`` over that trainset finds
    nothing on every example and returns a flat zero. A metric that is identically zero gives
    GEPA no gradient, which is the failure this module's docstring spends a paragraph warning
    about, and it would have cost the whole budget to discover. Return a dict from ``label`` to
    name the field, or name it here.
    """

    def metric(example: Any, prediction: Any, *args: Any, **kwargs: Any) -> float:
        if not hasattr(example, field):
            raise AttributeError(
                f"exact_field_metric({field!r}) found no {field!r} on the example, so it would "
                f"score every rollout 0.0 and give the optimizer no gradient. If the trainset "
                f"came from `trainset_from_traces`, a non-dict `label` return is stored at "
                f"`.label`: return {{{field!r}: gold}} instead, or point this metric at `label`."
            )
        expected = getattr(example, field, None)
        actual = getattr(prediction, field, None)
        return 1.0 if expected is not None and expected == actual else 0.0

    return metric


def schema_valid_metric(model: type[BaseModel], field: str) -> Metric:
    """Metric: 1.0 when ``prediction.field`` validates against ``model``."""

    def metric(example: Any, prediction: Any, *args: Any, **kwargs: Any) -> float:
        value = getattr(prediction, field, None)
        if value is None:
            return 0.0
        try:
            if isinstance(value, BaseModel):
                model.model_validate(value.model_dump())
            elif isinstance(value, dict):
                model.model_validate(value)
            else:
                model.model_validate_json(value)
            return 1.0
        except Exception:
            return 0.0

    return metric


@dataclass
class CompileResult:
    """What a compile produced, and whether it is worth keeping.

    ``holdout_before`` and ``holdout_after`` are measured in ONE process, which is what makes
    their difference a result: re-running a known artifact over the same held-out examples in a
    later session scored 0.575 and 0.596 against a recorded 0.733, so session drift of that order
    is the same size as the gain a compile earns. An absolute score from one session is not a
    figure anyone can reproduce; the paired difference is.

    ``delta`` is ``None`` rather than a number when no holdout was given, because there is nothing
    to subtract and a plausible-looking figure in that slot would be the worst of the options.

    **A non-zero ``failed_rollouts`` makes ``delta`` suspect rather than wrong.** A rollout that
    raised scored ``failure_score`` instead of aborting the compile, so a provider error on one
    arm only moves the difference. The count is per side, because which arm took it decides which
    way the difference moved.
    """

    program: Any
    score: float | None = None
    holdout_before: float | None = None
    holdout_after: float | None = None
    delta: float | None = None
    per_example: dict[str, list[float]] | None = None
    artifact: str | None = None
    instruction_audit: InstructionAudit | None = None
    output_audit: dict[str, Any] | None = None
    #: Counts of what THIS function ran. NOT the compile's cost: GEPA's own budget is
    #: `auto_budget(num_preds, num_candidates, valset_size)`, which is 790 metric calls for a
    #: two-predictor program at `auto="light"` over ten examples, and it is reported by dspy
    rollouts: dict[str, int] | None = None
    #: Held-out rollouts that raised and were scored `failure_score` instead of aborting, PER
    #: SIDE. One aggregate could not say which arm took the hit, and a failure caps that arm's
    #: mean: the same argument `per_example` exists for, one level up.
    failed_rollouts: dict[str, int] = field(default_factory=lambda: {"before": 0, "after": 0})


@dataclass
class TrainsetBuild:
    """The examples, and what was dropped getting them.

    The drops are not diagnostics, they are the result's other half. A corpus of 161 real runs
    carried 8 that failed before the planner ever returned a prediction, and such a run has ZERO
    ``main_step`` events: :meth:`RLMTask.arun` records the main trajectory off that prediction, so
    with none there is nothing to record. It is therefore indistinguishable from a run that
    finished instantly, and a trainset built without that check teaches the optimizer that dying
    early is cheap. A run that failed AFTER a prediction, output validation exhausting the retry
    budget being the common way, does record its turns and does reach the trainset.
    """

    examples: list[Any]
    dropped: dict[str, int]
    sources: int

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self):
        return iter(self.examples)


def trainset_from_traces(
    paths: Any,
    *,
    inputs: Sequence[str] | Callable[[list[dict]], dict],
    label: Callable[[list[dict]], Any],
    require: Sequence[str] = ("run_end", "main_step"),
) -> TrainsetBuild:
    """Build a trainset from runs this kit already recorded.

    The questions people actually asked beat ten someone wrote by hand, and they are already on
    disk. What is NOT on disk is whether each answer was any good, so ``label`` is yours: it takes
    a run's events and returns the gold value, or ``None`` to drop that run. This kit emits
    trajectories and never reward (``docs/INVARIANTS.md``), and that boundary is why the signature
    looks like this.

    ``inputs`` names the fields to lift from ``run_start``'s ``meta``, or is a callable taking the
    events and returning the input dict, because consumers do not agree on where the input lives:
    across five real ones it was under ``meta.task``, inside a JSON blob at ``meta.event``, and at
    ``meta.lead``.

    ``require`` is the filter that makes the result a trainset rather than a log. Defaults to a run
    whose file reached its terminal event and that took at least one turn. Note what it does NOT
    check: ``run_end`` is recorded from a ``finally`` and carries ``ok: false`` on a failure, so
    its presence means the FILE is complete, never that the run succeeded. In the corpus above all
    8 failures carry ``run_end``, and the turn check is what caught them. Screen on ``ok`` inside
    ``label`` if success is what you want.

    Every file is read with :func:`rlm_harness.load_events` rather than a hand-rolled parser,
    which is not a style note: ``str.splitlines()`` breaks on eleven separators and a model's
    reasoning text contains them, so a hand-rolled reader reported ten corrupt lines in an intact
    corpus.
    """
    import dspy

    from .trace import load_events

    files = _trace_files(paths)
    examples: list[Any] = []
    dropped: dict[str, int] = {}

    def _drop(why: str) -> None:
        dropped[why] = dropped.get(why, 0) + 1

    for path in files:
        try:
            events = load_events(str(path))
        except Exception:
            _drop("unreadable")
            continue
        if not events:
            _drop("empty")
            continue
        present = {e.get("type") for e in events}
        missing = [r for r in require if r not in present]
        if missing:
            _drop(f"no {missing[0]}")
            continue
        try:
            got = inputs(events) if callable(inputs) else _lift(events, inputs)
        except Exception:
            _drop("inputs raised")
            continue
        if not isinstance(got, dict) or not all(isinstance(k, str) for k in got):
            # Outside the `try` above, so a callable returning a list used to raise
            # `AttributeError: 'list' object has no attribute 'values'` mid-corpus, after every
            # file had been read, in a function whose whole posture is drop-not-crash.
            _drop("inputs not a string-keyed mapping")
            continue
        if not got or any(v is None for v in got.values()):
            _drop("inputs missing")
            continue
        try:
            gold = label(events)
        except Exception:
            # A label that raises drops its run rather than the build: one unscoreable run in a
            # corpus of hundreds is not a reason to produce nothing.
            _drop("label raised")
            continue
        if gold is None:
            _drop("label declined")
            continue
        fields = gold if isinstance(gold, dict) else {"label": gold}
        if set(fields) & set(got):
            # `dspy.Example(**got, **fields)` raised `TypeError: got multiple values for keyword
            # argument` here, which is the same crash-instead-of-drop as above.
            _drop("gold collides with an input")
            continue
        examples.append(dspy.Example(**got, **fields).with_inputs(*got))
    return TrainsetBuild(examples=examples, dropped=dropped, sources=len(files))


def _trace_files(paths: Any) -> list[Any]:
    """One path or many, as a string or a `Path`.

    `isinstance(paths, str)` alone was the first version, and it turned a `Path` into
    `list(Path)`, which raises `TypeError: 'PosixPath' object is not iterable` out of the middle
    of a build. A smoke test that happened to pass a string did not see it.
    """
    import os
    import pathlib

    given = [paths] if isinstance(paths, (str, os.PathLike)) else list(paths)
    out: list[Any] = []
    for one in given:
        p = pathlib.Path(one).expanduser()
        out.extend(sorted(p.rglob("*.jsonl")) if p.is_dir() else [p])
    return out


def _lift(events: list[dict], names: Sequence[str]) -> dict:
    """Pull named fields out of `run_start`'s meta, which is the caller's own namespace."""
    start = next((e for e in events if e.get("type") == "run_start"), {})
    meta = (start.get("payload") or {}).get("meta") or {}
    return {n: meta.get(n) for n in names}


@dataclass
class InstructionAudit:
    """What a compiled instruction gained and lost, for the things a score cannot see.

    Every field here answers something a score cannot. Five come from a measured compile: the run
    that raised a held-out set by 0.221 also wrote four of ten trainset answers into the
    instruction and dropped one capability silently, and neither is visible in the number. The
    last three come from the other direction, naming what was NOT measured, so that an absent
    check cannot read as a clean result.
    """

    length_ratio: float
    trainset_answers_in: list[int]
    holdout_answers_in: list[int]
    capabilities_lost: list[str]
    capabilities_kept: list[str]
    #: Which predictors were read. An audit of one predictor cannot speak for the other, and GEPA
    #: moves them one at a time.
    predictors: list[str] = field(default_factory=list)
    #: False when no holdout was supplied, which makes `memorised` unanswerable rather than clean.
    controlled: bool = False
    #: False when no capability map was supplied, which makes `capabilities_lost` empty by
    #: construction rather than by measurement.
    capabilities_checked: bool = False

    @property
    def memorised(self) -> bool | None:
        """Did it write trainset answers in that it did NOT write for the holdout?

        ``None`` when the question cannot be answered: with no holdout there was no control, and
        ``bool(hits) and not []`` cannot tell "the control ran and stayed clean" from "there was
        no control". Both return the same number when everything they measure is working, which
        is this repo's own test for an absent measurement reported as a real one.

        The holdout is the control because a trainset hit alone cannot tell memorisation from a
        matcher firing on ordinary language. ``False`` when both sides hit: the claim is then
        unsupported, not disproved.
        """
        if not self.controlled:
            return None
        return bool(self.trainset_answers_in) and not self.holdout_answers_in


def audit_instruction(
    baseline: Any,
    compiled: Any,
    *,
    trainset: Sequence[Any] = (),
    holdout: Sequence[Any] = (),
    answer_of: Callable[[Any], str] | None = None,
    capabilities: dict[str, Sequence[str]] | None = None,
) -> InstructionAudit:
    """Read a compiled instruction for what the score is blind to.

    ``baseline`` and ``compiled`` are each one instruction or a ``{predictor: text}`` mapping.
    Pass the mapping: GEPA moves one component at a time, so an audit that read the planner alone
    reported a pristine result for exactly the artifact it exists to catch, with the gold answer
    verbatim in ``extract``.

    **Answers written in.** An optimizer rewards itself for getting trainset examples right, and
    one way to get them right is to carry them. A measured run put four of ten trainset answers
    into the instruction and none of the six held-out ones, so the gain was real and the artifact
    also shipped a crib sheet. In a periodic loop this is the failure that compounds, because
    every round starts from the previous instruction and nothing removes what the last one wrote.

    ``holdout`` is the control and not a nicety. Matching on the trainset alone cannot tell
    memorisation from a matcher that fires on ordinary language; running the SAME match over
    answers the optimizer never saw is what separates them. :attr:`InstructionAudit.memorised`
    is the reading, not the raw count.

    Matching is verbatim, on normalised whitespace and case, and that makes this a LOWER BOUND: a
    paraphrased answer is undetectable here. The same limit sank a guardrail check that reported
    five of seven capabilities deleted when one had gone, so it is stated rather than hidden.

    **Capabilities lost.** Pass ``capabilities`` as a name mapped to several wordings, because the
    literal-phrase version of this check is the one that was wrong. The real loss in that run was
    the sub-LM: ``llm_query`` appears zero times after compilation, so the model is no longer told
    it has one, and nothing in the score moved because that task never needed it.

    **Length as a ratio**, never a character count, because a count measures something a release
    can edit and goes stale with the next docstring.
    """
    # Both sides may be a single string or a {predictor: text} mapping. A mapping is the shape
    # that matters: GEPA's default `component_selector="round_robin"` moves one component at a
    # time, so a winning candidate that improved only `extract` is ordinary (the load path says so
    # itself), and an audit that read the planner alone reported a pristine result for exactly the
    # artifact it exists to catch, with the gold answer verbatim in `extract`.
    base_map = baseline if isinstance(baseline, dict) else {"": baseline}
    comp_map = compiled if isinstance(compiled, dict) else {"": compiled}
    names = [n for n in comp_map if n in base_map]
    base_n = _norm(" \n ".join(base_map[n] for n in names))
    comp_n = _norm(" \n ".join(comp_map[n] for n in names))
    get = answer_of

    def _hits(examples: Sequence[Any]) -> list[int]:
        found = []
        for i, ex in enumerate(examples):
            for candidate in _candidates(ex, get):
                answer = _norm(candidate)
                if len(answer) < _MIN_ANSWER_CHARS:
                    continue
                if answer in comp_n and answer not in base_n:
                    found.append(i)
                    break
        return found

    lost, kept = [], []
    for name, wordings in (capabilities or {}).items():
        in_base = any(_norm(w) in base_n for w in wordings)
        in_comp = any(_norm(w) in comp_n for w in wordings)
        if in_base and not in_comp:
            lost.append(name)
        elif in_base:
            kept.append(name)

    base_chars = sum(len(base_map[n]) for n in names)
    comp_chars = sum(len(comp_map[n]) for n in names)
    return InstructionAudit(
        length_ratio=(comp_chars / base_chars) if base_chars else float("nan"),
        trainset_answers_in=_hits(trainset),
        holdout_answers_in=_hits(holdout),
        capabilities_lost=sorted(lost),
        capabilities_kept=sorted(kept),
        predictors=[n for n in names if n],
        # Whether the control COULD fire, not whether a holdout was supplied: a holdout whose
        # examples yield no candidate above the floor cannot clear the matcher, and a control
        # that could not fire reading as a clean one is the same defect as `memorised` concluding
        # without one.
        controlled=any(
            any(len(_norm(c)) >= _MIN_ANSWER_CHARS for c in _candidates(ex, get))
            for ex in holdout
        ),
        capabilities_checked=bool(capabilities),
    )


#: Below this an "answer" is too short to tell a quotation from a coincidence, and the holdout
#: control would fire on both sides. Picked to exclude single words and short labels.
_MIN_ANSWER_CHARS = 24


def _norm(text: Any) -> str:
    """Case-folded, whitespace-collapsed. dspy flattens an instruction with `.replace("\n", "  ")`
    before the model sees it, so a newline in the saved answer must not defeat the match."""
    return " ".join(str(text or "").split()).casefold()


def _candidates(example: Any, answer_of: Callable[[Any], str] | None) -> list[str]:
    """Every string that could be this example's gold, most specific first.

    A LIST, because an example whose input and output fields are not declared has no single
    answer, and this check must be wrong in the direction that reports a hit to look at rather
    than the one that stays silent. The rule is the one ``verify_quote``'s false negative was
    kept for.

    - ``answer_of`` given: trust it, one candidate.
    - a `dspy.Example`: the inputs are declared, so each non-input field is a candidate, plus
      their join when there is more than one. A dict ``label`` from
      :func:`trainset_from_traces` is that shape, and the join alone missed it.
    - a plain mapping: each value on its own, with no join. Nothing there declares which key is
      the question, so a joined string would carry it, and an instruction does not quote the
      question.

    A `dspy.Example` whose every field is an input yields nothing and is skipped: that is a blind
    spot rather than a false clean, because there is no answer to find.
    """
    if answer_of is not None:
        return [answer_of(example)]
    if hasattr(example, "inputs") and hasattr(example, "toDict"):
        declared = set(example.inputs().toDict())
        rest = [str(v) for k, v in example.toDict().items() if k not in declared]
        # Each field on its own AND the join. A multi-gold example is what
        # `trainset_from_traces` produces from a dict `label`, which is the shape
        # `exact_field_metric`'s own error message tells callers to use, and the join alone missed
        # it for the reason the mapping branch below does not join: a combined string appears in
        # no instruction verbatim.
        return [*rest, " ".join(rest)] if len(rest) > 1 else rest
    if isinstance(example, dict):
        return [str(v) for v in example.values()]
    return [str(example)]


def compile_task(
    task: Any,
    trainset: Sequence[Any],
    metric: Metric,
    *,
    reflection_lm: Any = None,
    holdout: Sequence[Any] | None = None,
    auto: Any = _UNSET,
    save_to: str | None = None,
    audit_outputs: Callable[[Sequence[Any], Sequence[Any | None]], dict] | None = None,
    capabilities: dict[str, Sequence[str]] | None = None,
    answer_of: Callable[[Any], str] | None = None,
    **gepa_kwargs: Any,
) -> CompileResult:
    """Compile an ``RLMTask``'s program with ``dspy.GEPA``, and measure whether it helped.

    The kit owns the wiring; you own the metric. What it does for you:

    - **Builds the program with :meth:`RLMTask._build_rlm`**, so the sandbox guard, the per-pass
      interpreter factory, tool timing and description normalisation all apply to every optimizer
      pass, and passes it to GEPA DIRECTLY rather than wrapping it. A wrapper is tempting (it is
      where a consumer hangs its own tracing) and it costs you the artifact: dspy prefixes each
      predictor with the attribute it hangs off, so a state saved from a wrapper names
      ``rlm.generate_action`` and :attr:`RLMTask.compiled_instructions` refuses it for naming no
      predictor this build has.
    - **Evaluates ``holdout`` before and after, in this process.** See :class:`CompileResult` for
      why only a paired difference is a result.
    - **Saves with the fingerprint**, via :func:`save_program` with ``task=``, so the artifact can
      be loaded back by :attr:`RLMTask.compiled_instructions` instead of being refused.

    What it refuses to do for you: pick a metric. One metric call is one FULL rollout, so the
    metric decides what the money buys, and a plausible one can be measured saturated. On a real
    consumer corpus of 153 runs the obvious quality signal scored a perfect 1.000 on 151 of them,
    which gives an optimizer nothing to climb, and the cost axis that did have a gradient was
    satisfiable by doing less work rather than better work. ``docs/PROMPT.md`` has both.

    **The metric is probed against the first trainset example before anything is spent**, with a
    stub prediction, so a metric that cannot score this trainset at all is refused at zero
    rollouts rather than after the budget. A metric that merely reads its prediction is not
    refused; see :func:`_metric_cannot_score` for how the two are told apart.

    ``reflection_lm`` has no default and is checked here rather than left to dspy, which raises
    ``AssertionError: GEPA requires a reflection language model, or custom instruction proposer to
    be provided``. It must be a ``dspy.LM`` that can return JSON, which 3.4.1 and later require
    because the default proposer goes through ``dspy.Predict`` and ``JSONAdapter``;
    ``docs/PROMPT.md`` carries the command that settles which dspy builds one by default.

    ``audit_outputs(examples, predictions)`` is yours, run over the holdout before and after.
    **A prediction may be ``None``**, for a rollout that failed and was scored rather than
    aborting, so the callable has to tolerate it: ``[p.answer for p in predictions]`` raises on
    one. The kit guards the call so a raise cannot discard the measurement, and reports the
    failure in the result instead. The
    kit cannot audit your outputs because it does not know what they mean: on the corpus above,
    two distinct ways of padding a page both held the quality metric at its ceiling, one of them
    visible only by reading the prose. Whatever the callable returns is carried in the result.
    """
    import dspy

    if reflection_lm is None:
        raise ValueError(
            "compile_task needs reflection_lm=dspy.LM(...): the LM that reads a trajectory and "
            "proposes a new instruction. dspy raises a bare AssertionError without it. On dspy "
            "3.4.1 and later it must be able to return JSON, because the default proposer goes "
            "through dspy.Predict and JSONAdapter."
        )
    if not trainset:
        raise ValueError("compile_task needs a non-empty trainset: GEPA has nothing to run on.")

    train = list(trainset)
    held = list(holdout) if holdout else []

    # The metric, before anything is spent. `dspy.GEPA.__init__` validates its ARITY for free and
    # this validates that it can score THIS trainset: a metric looking for a field the examples do
    # not carry has every rollout scored `failure_score` by GEPA's own executor, which swallows the
    # exception on the trainset side, so the optimizer gets no gradient and the whole budget goes
    # before anyone learns why. It needs no real prediction, only a stub -- and the stub records whether
    # the metric read it, for the reason in `_metric_cannot_score`.
    if _metric_cannot_score(metric, train[0]):
        metric(train[0], _ProbePrediction())   # raise it where the caller can read it
    program = task._build_rlm()
    # Captured BEFORE the compile. dspy 3.4.0's GEPA hands back a deepcopy and leaves the
    # student's own text intact, so this is not the only moment it could be read; it is the moment
    # it is CERTAINLY the text GEPA was seeded from, before a save, a retry or a future dspy that
    # does mutate can touch it, and that is what an audit's baseline has to be.
    baseline_text = _instruction_map(program)

    # `auto=None` is the route to `max_metric_calls` / `max_full_evals`: dspy asserts that EXACTLY
    # one of the three is set, so passing `auto` beside a budget kwarg is refused. Built HERE,
    # before anything is evaluated, because this constructor validates the metric's arity and the
    # budget combination for free, with no LM call. It used to be built after the before-evaluation,
    # which bought the same exception at the price of one full rollout per held-out example.
    # Scoped to the two budget keys. Dropping every `None` discarded `seed=None` and
    # `max_merge_invocations=None`, which dspy accepts and whose non-None defaults then applied:
    # a caller who explicitly asked for randomness got determinism back.
    budget_keys = {"max_metric_calls", "max_full_evals"}
    gepa_kwargs = {k: v for k, v in gepa_kwargs.items()
                   if not (k in budget_keys and v is None)}
    given_budget = budget_keys & set(gepa_kwargs)
    if auto is _UNSET:
        # A default nobody chose yields to an explicit budget, the way an explicit `tools=` beats
        # a class-body declaration. An EXPLICIT `auto` beside a budget is two things the caller
        # asked for and the kit silently dropped one; dspy's own assertion names all three and
        # now costs nothing, because the optimizer is built before any rollout.
        if not given_budget:
            gepa_kwargs["auto"] = "light"
    elif auto is not None:
        gepa_kwargs["auto"] = auto
    if auto not in (_UNSET, None) and auto not in _AUTO_SETTINGS:
        raise ValueError(
            f"auto={auto!r} is not one of {sorted(_AUTO_SETTINGS)}. dspy accepts it in the "
            "constructor and raises KeyError from compile(), which is after the holdout has been "
            "evaluated."
        )
    optimizer = dspy.GEPA(metric=metric, reflection_lm=reflection_lm, **gepa_kwargs)

    before = _evaluate(program, held, metric) if held else None
    audit_before = _run_audit(audit_outputs, held, before)
    compiled = optimizer.compile(program, trainset=train)

    # Saved BEFORE the after-evaluation: the artifact is what the compile produced and it is
    # complete here, so nothing downstream of it should be able to lose it.
    artifact = None
    if save_to:
        save_program(compiled, save_to, task=task)
        artifact = save_to

    after = _evaluate(compiled, held, metric) if held else None
    audit_after = _run_audit(audit_outputs, held, after)

    audit = None
    compiled_text = _instruction_map(compiled)
    if baseline_text and compiled_text:
        audit = audit_instruction(baseline_text, compiled_text, trainset=train, holdout=held,
                                  answer_of=answer_of, capabilities=capabilities)

    return CompileResult(
        program=compiled,
        # Kept for the pre-1.20.0 field name; `holdout_after` is the one to read.
        score=after[0] if after else None,
        holdout_before=before[0] if before else None,
        holdout_after=after[0] if after else None,
        delta=(after[0] - before[0]) if (before and after) else None,
        per_example=({"before": before[2], "after": after[2]} if (before and after) else None),
        artifact=artifact,
        instruction_audit=audit,
        output_audit=({"before": audit_before, "after": audit_after}
                      if (audit_outputs and held) else None),
        failed_rollouts={"before": before[3] if before else 0,
                         "after": after[3] if after else 0},
        rollouts={"trainset_examples": len(train), "holdout_examples": len(held),
                  "holdout_rollouts": 2 * len(held)},
    )


def _instruction_map(program: Any) -> dict[str, str]:
    """Every predictor's instruction text, keyed by name. Empty when there is no such seam.

    EVERY predictor, not the planner: GEPA moves one component at a time, so reading the planner
    alone left an `extract`-only compile looking pristine.

    Empty rather than raising: a consumer whose program shape the kit does not recognise should
    still get a compile and a score, with the audit absent and visibly so, instead of a crash
    after the rollouts have been paid for.
    """
    out: dict[str, str] = {}
    try:
        named = dict(program.named_predictors())
    except Exception:
        return out
    for name, predictor in named.items():
        text = getattr(getattr(predictor, "signature", None), "instructions", None)
        if isinstance(text, str):
            out[name] = text
    return out


def _run_audit(audit: Callable[..., dict] | None, examples: Sequence[Any],
               evaluated: tuple | None) -> dict | None:
    """Run the consumer's output audit, and never let it discard the measurement.

    Guarded for the same reason `save_program` moved above the after-evaluation: on the after
    side this runs downstream of the save, so a raise left the artifact on disk and took
    `holdout_after`, `delta` and the instruction audit with it, and the only way back was
    re-running the whole compile. A failing rollout also puts `None` in the predictions, which the
    obvious ``[p.answer for p in predictions]`` raises on.
    """
    if audit is None or evaluated is None:
        return None
    try:
        return audit(examples, evaluated[1])
    except Exception as exc:
        logger.warning("audit_outputs raised; recording the failure instead", exc_info=True)
        # A reserved key, so a reader can tell the kit's failure record from a consumer's own
        # field named `error`.
        return {"rlm_harness_error": f"{type(exc).__name__}: {exc}"}


def _evaluate(program: Any, examples: Sequence[Any], metric: Metric,
              failure_score: float = 0.0) -> tuple[float, list[Any], list[float], int]:
    """Score a program over examples: the mean, the predictions, every score, and the failures.

    The per-example scores come back because a mean over six examples hides which way each one
    moved, and a compile that raises the mean while making two examples worse is a different
    decision from one that raises all six.

    **A failing rollout scores ``failure_score`` instead of propagating.** These rollouts are not
    behind :meth:`RLMTask.arun`'s retry, so without this one provider 500 on the last held-out
    example discarded an entire compile: the exception escaped before the artifact was saved, and
    an ``auto="light"`` run on a two-predictor program over ten examples is 790 metric calls.
    dspy is tolerant for the same reason on the same rollouts, via
    ``Evaluate(failure_score=…, max_errors=…)``. A run that failed is a run that scored badly,
    which is information, not an abort.
    """
    preds, scores, failed = [], [], 0
    for ex in examples:
        inputs = ex.inputs().toDict() if hasattr(ex, "inputs") else dict(ex)
        try:
            pred = program(**inputs)
        except Exception:
            logger.warning("holdout rollout failed; scoring it %s", failure_score, exc_info=True)
            preds.append(None)
            scores.append(failure_score)
            failed += 1
            continue
        # OUTSIDE the guard, deliberately. Covering the metric too made a metric that cannot score
        # this trainset at all read as a transient rollout failure: `exact_field_metric`'s raise,
        # added in this same release so a mis-named gold field would not burn a whole budget at a
        # flat zero, was caught, logged and scored 0.0. A rollout that failed is a run that scored
        # badly; a metric that cannot score is a configuration error, and only one of those is
        # worth continuing through. `ex.inputs()` is outside it for the same reason: an example
        # never given `with_inputs()` is a wiring mistake, not a bad rollout.
        preds.append(pred)
        scores.append(float(metric(ex, pred)))
    return (sum(scores) / len(scores) if scores else 0.0), preds, scores, failed


FINGERPRINT_KEY = "rlm_harness"
"""Top-level key a fingerprinted state carries, beside dspy's own per-predictor blocks.

dspy's loader looks each predictor up by name and ignores keys it does not know, so the extra
block costs ``load_program`` nothing.
"""


def save_program(program: Any, path: str, task: Any | None = None) -> None:
    """Persist a compiled dspy program to ``path`` (JSON state).

    Pass ``task`` (the :class:`~rlm_harness.task.RLMTask` that was compiled) to record WHAT the
    program was compiled against: the signature, each tool's name and model-facing docstring, the
    output model's JSON schema, and what the sandbox tells the model its REPL can do, which are
    the four `compiled_instructions` compares, plus the task's own instructions and the budget
    caps, which it records and reports rather than refusing on. The record describes the TASK you
    pass, not the program you save.
    :meth:`~rlm_harness.task.RLMTask.compiled_fingerprint` is the whole block, including the
    fields recorded for a reader rather than compared.
    :attr:`~rlm_harness.task.RLMTask.compiled_instructions` requires that block and refuses a
    state without it, because nothing recoverable from the state itself answers the question it
    has to ask.

    ``program`` is the RLM itself, not a module wrapping it: dspy prefixes each predictor with the
    attribute it hangs off, so a wrapper's state names predictors the task's own build does not
    have, and the sandbox description can only be read off the object that carries the interpreter
    factory.

    The reason it cannot is worth knowing before deciding to omit ``task``. An optimizer rewrites
    an instruction WHOLESALE, and the string it rewrites contains dspy's rendered tool
    documentation, so the artifact holds the optimizer's own paraphrase of the tools as they were
    and no trace of which tools those were. Loading it into a build whose tools have moved hands
    the model a description of behaviour the code no longer has, and no amount of reading the
    artifact detects it.
    """
    if task is not None and not str(path).endswith(".json"):
        # Checked BEFORE the save: raising after it leaves a fingerprint-less file on disk under
        # the name the caller is about to reuse, and `compiled_instructions` would refuse that.
        raise ValueError(
            f"save_program(..., task=...) writes its fingerprint into the JSON state, so {path!r} "
            "must end with .json. Drop `task=` to save without one, and know that "
            "`compiled_instructions` will then refuse the file."
        )
    program.save(path)
    if task is None:
        return

    import json

    with open(path, encoding="utf-8") as fh:
        state = json.load(fh)
    # `program=`, so the record includes what the model was told its REPL can do. Without it
    # the key is absent on the saved side and the key-wise check skips it, which is a gate
    # that passes every test written against an injected key and protects no real save.
    state[FINGERPRINT_KEY] = task.compiled_fingerprint(program=program)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)


def load_program(program: Any, path: str) -> Any:
    """Load saved state into ``program`` and return it.

    The whole state, as dspy restores it: instructions, demos, and any per-predictor LM the file
    carries. :attr:`~rlm_harness.task.RLMTask.compiled_instructions` is the narrower door that
    takes the instruction text only.
    """
    program.load(path)
    return program
