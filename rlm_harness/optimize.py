"""GEPA optimization harness — PHASE 1 SKELETON.

The whole point of choosing DSPy for RLM is that tasks can be *compiled*
(prompt + few-shot demos optimised against a metric) rather than hand-tuned.
This module wires that interface and ships ready-to-use metric templates.

What is implemented now (Phase 1):
- Metric templates (``exact_field_metric``, ``schema_valid_metric``) — pure,
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

    ONE THING TO FIX WHEN ENABLING IT: `optimizer.compile` calls the program WITHOUT a positional
    interpreter, so dspy builds one from `interpreter_factory` — and since 1.5.0 `_build_rlm` may
    pass a metadata CARRIER there that deliberately raises when invoked (see
    `_dspy_compat.interpreter_instructions_kwargs`). This is the one code path that breaks that
    function's "never invoked" premise, and it must supply a real interpreter of its own.

    It is intentionally not active yet: compiling without a representative,
    labelled trainset produces a confidently-wrong program, which is worse than
    none. Supply the trainset + metric, then swap this stub for the body above.
    """
    raise NotImplementedError(
        "compile_task is a Phase-2 stub. Provide a labelled trainset and a "
        "metric, then enable the reference body documented in this function."
    )


def save_program(program: Any, path: str) -> None:
    """Persist a compiled dspy program to ``path`` (JSON state)."""
    program.save(path)


def load_program(program: Any, path: str) -> Any:
    """Load saved state into ``program`` and return it."""
    program.load(path)
    return program
