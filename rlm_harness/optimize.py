"""GEPA optimization harness: compile a task's instructions against your own metric.

The point of building on dspy is that a task can be *compiled*, its instructions optimised
against a metric, rather than hand-tuned forever. This module owns the wiring and leaves you the
one thing it cannot have an opinion about.

- :func:`compile_task` runs ``dspy.GEPA`` over a task's own program, measures a holdout before and
  after IN ONE PROCESS, and saves the artifact with the fingerprint
  :attr:`~rlm_harness.task.RLMTask.compiled_instructions` needs to accept it back.
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

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

# A metric scores (example, prediction) -> float in [0, 1].
Metric = Callable[[Any, Any], float]


def exact_field_metric(field: str) -> Metric:
    """Metric: 1.0 when ``prediction.field == example.field``, else 0.0.

    Works with dspy.Example / dspy.Prediction (attribute access) and plain
    objects, so it is testable without dspy.
    """

    def metric(example: Any, prediction: Any, *args: Any, **kwargs: Any) -> float:
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
    rollouts: dict[str, int] | None = None


@dataclass
class InstructionAudit:
    """What a compiled instruction gained and lost, for the things a score cannot see.

    Every field here exists because a measured compile did it: the run that raised a held-out set
    by 0.221 also wrote four of ten trainset answers into the instruction, and dropped one
    capability silently. Neither is visible in the number.
    """

    length_ratio: float
    trainset_answers_in: list[int]
    holdout_answers_in: list[int]
    capabilities_lost: list[str]
    capabilities_kept: list[str]

    @property
    def memorised(self) -> bool:
        """Did it write trainset answers in that it did NOT write for the holdout?

        The holdout count is the CONTROL, and it is why this is a property rather than a bare
        trainset count: if holdout answers appear too, the matcher is finding common language
        rather than answers, and the trainset figure means nothing. A compile that memorises shows
        trainset hits with none on the holdout.
        """
        return bool(self.trainset_answers_in) and not self.holdout_answers_in


def audit_instruction(
    baseline: str,
    compiled: str,
    *,
    trainset: Sequence[Any] = (),
    holdout: Sequence[Any] = (),
    answer_of: Callable[[Any], str] | None = None,
    capabilities: dict[str, Sequence[str]] | None = None,
) -> InstructionAudit:
    """Read a compiled instruction for what the score is blind to.

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
    base_n = _norm(baseline)
    comp_n = _norm(compiled)
    get = answer_of or _answer_of

    def _hits(examples: Sequence[Any]) -> list[int]:
        found = []
        for i, ex in enumerate(examples):
            answer = _norm(get(ex))
            if len(answer) < _MIN_ANSWER_CHARS:
                continue
            if answer in comp_n and answer not in base_n:
                found.append(i)
        return found

    lost, kept = [], []
    for name, wordings in (capabilities or {}).items():
        in_base = any(_norm(w) in base_n for w in wordings)
        in_comp = any(_norm(w) in comp_n for w in wordings)
        if in_base and not in_comp:
            lost.append(name)
        elif in_base:
            kept.append(name)

    return InstructionAudit(
        length_ratio=(len(compiled) / len(baseline)) if baseline else float("nan"),
        trainset_answers_in=_hits(trainset),
        holdout_answers_in=_hits(holdout),
        capabilities_lost=sorted(lost),
        capabilities_kept=sorted(kept),
    )


#: Below this an "answer" is too short to tell a quotation from a coincidence, and the holdout
#: control would fire on both sides. Picked to exclude single words and short labels.
_MIN_ANSWER_CHARS = 24


def _norm(text: Any) -> str:
    """Case-folded, whitespace-collapsed. dspy flattens an instruction with `.replace("\n", "  ")`
    before the model sees it, so a newline in the saved answer must not defeat the match."""
    return " ".join(str(text or "").split()).casefold()


def _answer_of(example: Any) -> str:
    """The gold value of an example, without the caller having to name the field.

    A `dspy.Example`'s inputs are declared, so whatever is NOT an input is the label. Falls back
    to the whole example's text, which over-matches rather than under-matches: this check should
    be wrong in the direction that reports a hit to look at, never the one that stays silent.
    """
    if hasattr(example, "inputs") and hasattr(example, "toDict"):
        inputs = set(example.inputs().toDict())
        rest = {k: v for k, v in example.toDict().items() if k not in inputs}
        if rest:
            return " ".join(str(v) for v in rest.values())
    return str(example)


def compile_task(
    task: Any,
    trainset: Sequence[Any],
    metric: Metric,
    *,
    reflection_lm: Any = None,
    holdout: Sequence[Any] | None = None,
    auto: str = "light",
    save_to: str | None = None,
    audit_outputs: Callable[[Sequence[Any], Sequence[Any]], dict] | None = None,
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

    ``reflection_lm`` has no default and is checked here rather than left to dspy, which raises
    ``AssertionError: GEPA requires a reflection language model, or custom instruction proposer to
    be provided``. It must be a ``dspy.LM`` that can return JSON, which 3.4.1 requires because its
    default proposer goes through ``dspy.Predict`` and ``JSONAdapter``.

    ``audit_outputs(examples, predictions)`` is yours, run over the holdout before and after. The
    kit cannot audit your outputs because it does not know what they mean: on the corpus above,
    two distinct ways of padding a page both held the quality metric at its ceiling, one of them
    visible only by reading the prose. Whatever the callable returns is carried in the result.
    """
    import dspy

    if reflection_lm is None:
        raise ValueError(
            "compile_task needs reflection_lm=dspy.LM(...): the LM that reads a trajectory and "
            "proposes a new instruction. dspy raises a bare AssertionError without it. On dspy "
            "3.4.1 it must be able to return JSON, because the default proposer goes through "
            "dspy.Predict and JSONAdapter."
        )
    if not trainset:
        raise ValueError("compile_task needs a non-empty trainset: GEPA has nothing to run on.")

    train = list(trainset)
    held = list(holdout) if holdout else []
    program = task._build_rlm()
    # Captured BEFORE the compile, because GEPA mutates the predictor's signature in place on the
    # candidate it returns and there is no second chance to read what the build shipped.
    baseline_text = _planner_text(program)

    before = _evaluate(program, held, metric) if held else None
    audit_before = audit_outputs(held, before[1]) if (audit_outputs and before) else None

    optimizer = dspy.GEPA(metric=metric, auto=auto, reflection_lm=reflection_lm, **gepa_kwargs)
    compiled = optimizer.compile(program, trainset=train)

    after = _evaluate(compiled, held, metric) if held else None
    audit_after = audit_outputs(held, after[1]) if (audit_outputs and after) else None

    audit = None
    compiled_text = _planner_text(compiled)
    if baseline_text and compiled_text:
        audit = audit_instruction(baseline_text, compiled_text, trainset=train, holdout=held,
                                  answer_of=answer_of, capabilities=capabilities)

    artifact = None
    if save_to:
        save_program(compiled, save_to, task=task)
        artifact = save_to

    return CompileResult(
        program=compiled,
        score=after[0] if after else None,
        holdout_before=before[0] if before else None,
        holdout_after=after[0] if after else None,
        delta=(after[0] - before[0]) if (before and after) else None,
        per_example=({"before": before[2], "after": after[2]} if (before and after) else None),
        artifact=artifact,
        instruction_audit=audit,
        output_audit=({"before": audit_before, "after": audit_after} if audit_outputs else None),
        rollouts={"trainset": len(train), "holdout_before": len(held),
                  "holdout_after": len(held)},
    )


def _planner_text(program: Any) -> str | None:
    """The instruction an optimizer rewrites, or None when this program has no such seam.

    Returns None rather than raising: a consumer with a program shape the kit does not recognise
    should still get a compile and a score, with the audit absent and visibly so, instead of a
    crash after the rollouts have been paid for.
    """
    for name, predictor in (dict(program.named_predictors()) or {}).items():
        if name.endswith("generate_action"):
            text = getattr(getattr(predictor, "signature", None), "instructions", None)
            return text if isinstance(text, str) else None
    return None


def _evaluate(program: Any, examples: Sequence[Any],
              metric: Metric) -> tuple[float, list[Any], list[float]]:
    """Score a program over examples, returning the mean, the predictions and every score.

    The per-example scores come back because a mean over six examples hides which way each one
    moved, and a compile that raises the mean while making two examples worse is a different
    decision from one that raises all six.
    """
    preds, scores = [], []
    for ex in examples:
        inputs = ex.inputs().toDict() if hasattr(ex, "inputs") else dict(ex)
        pred = program(**inputs)
        preds.append(pred)
        scores.append(float(metric(ex, pred)))
    return (sum(scores) / len(scores) if scores else 0.0), preds, scores


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
