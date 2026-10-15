"""GEPA optimization harness: PHASE 1 SKELETON.

The whole point of choosing DSPy for RLM is that tasks can be *compiled*
(prompt + few-shot demos optimised against a metric) rather than hand-tuned.
This module wires that interface and ships ready-to-use metric templates.

What is implemented now (Phase 1):
- Metric templates (``exact_field_metric``, ``schema_valid_metric``): pure,
  tested, usable today.
- ``save_program`` / ``load_program`` thin wrappers over dspy persistence.

What is deferred to Phase 2 (needs a labelled trainset you provide):
- ``compile_task`` raises ``NotImplementedError`` with the exact call it will
  make. The reference body is in the docstring so wiring it up is mechanical.
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
    program: Any
    score: float | None = None


def compile_task(
    task: Any,
    trainset: Sequence[Any],
    metric: Metric,
    *,
    auto: str = "light",
    **gepa_kwargs: Any,
) -> CompileResult:
    """Compile an ``RLMTask``'s underlying program with dspy.GEPA. (Phase 2.)

    Reference implementation to be enabled once a trainset exists::

        import dspy
        program = task._build_rlm()
        optimizer = dspy.GEPA(metric=metric, auto=auto, **gepa_kwargs)
        compiled = optimizer.compile(program, trainset=list(trainset))
        return CompileResult(program=compiled)

    **``gepa_kwargs`` MUST carry ``reflection_lm``**, which has no default::

        compile_task(task, trainset, metric, reflection_lm=dspy.LM("openai/gpt-4.1-mini"))

    Without it dspy 3.4.0 and 3.4.1 both raise ``AssertionError: GEPA requires a reflection
    language model, or custom instruction proposer to be provided``, which is a failure the body
    above cannot prevent and nothing here used to mention. It must be a
    ``dspy.LM`` that can return JSON, which 3.4.1 requires because its default proposer went
    through ``dspy.Predict`` + ``JSONAdapter``. Reaching GEPA at all has prerequisites this
    function cannot supply, and ``docs/PROMPT.md`` carries them: one metric call is one FULL
    rollout, ``auto="light"`` on a two-predictor program resolves to 790 of them at a ten-example
    valset, and an optimized program re-enters a run through
    ``RLMTask(compiled_instructions=…)``, which takes the TEXT and still builds the program
    itself. A state compiled against different tool documentation is refused there rather than
    silently reinstating the descriptions of whatever version compiled it.

    THE BLOCKER THIS USED TO CARRY IS GONE as of 1.14.0. It said `optimizer.compile` calls the
    program without a positional interpreter, so dspy would build one from the metadata CARRIER
    `_build_rlm` passed, which raised when invoked. dspy 3.4.0 deleted the positional seam, so the
    kit now passes a REAL factory (`_dspy_compat.interpreter_kwargs`) and every optimizer pass gets
    its own sandbox, built by the kit with its guard, and shut down by dspy afterwards. Nothing
    here needs a special interpreter any more.

    It is intentionally not active yet: compiling without a representative,
    labelled trainset produces a confidently-wrong program, which is worse than
    none. Supply the trainset + metric, then swap this stub for the body above.
    """
    raise NotImplementedError(
        "compile_task is a Phase-2 stub. Provide a labelled trainset and a "
        "metric, then enable the reference body documented in this function."
    )


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
