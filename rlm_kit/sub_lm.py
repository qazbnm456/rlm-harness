"""Phase A — ``intercept_sub_lm``: the one hook to intercept the RLM's sub-LM.

``dspy.RLM`` exposes no hook to intercept a sub-LLM response before it returns to
the main model — and its built-in ``llm_query`` / ``llm_query_batched`` tools just
call ``self.sub_lm(prompt)``. So the ONLY interception point is the sub_lm object
itself. ``intercept_sub_lm`` wraps a ``dspy.LM``: ``RLM`` only sees "a sub_lm", but
inside we emit a ``sub_call`` trace event for every escalation and (optionally) run
a deterministic pipeline — call the base model, validate the format, post-process.
Tracing is the always-on job; validators/postprocessors are opt-in.

Design decisions baked in (per the approved plan):

- The sub-LM intercept does **deterministic transforms only** (validate + post-process).
  Agentic actions (calling an external tool) are *not* forced here; they are
  exposed to the main LM as RLM tools (see ``model_as_tool`` and
  ``rlm_kit.skills``), so the decision stays in the LM's hands and lands in the
  trajectory.
- Multi-model routing is done today via ``model_as_tool`` (LM-decided), not the
  unmerged official ``sub_lms`` API. When ``sub_lms`` ships, swap it in without
  touching task code.

``dspy`` is imported lazily so this module stays importable (and the pipeline
logic stays unit-testable) without a full dspy install.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

from .trace import current_recorder, record_tool_call, recorder_scope


def bind_recorder_to_sub_lm(sub_lm: Any, recorder: Any) -> Any:
    """Wrap ``sub_lm`` so every call re-establishes ``recorder`` as active in the CALLING thread.

    ``dspy.RLM.llm_query_batched`` runs sub-LM calls in a ``ThreadPoolExecutor`` whose workers do NOT
    inherit the recorder ``ContextVar`` — so without this, a batched escalation runs with
    ``current_recorder() is None`` and records no ``sub_call`` (the lifeline metric under-counts; a
    single ``llm_query``, same thread, is fine). The binding is PER RUN (one wrapper per run holds that
    run's recorder), so concurrent runs sharing the base sub-LM never cross-contaminate. dspy stores
    and merely CALLS ``sub_lm`` (no isinstance check), so a duck-typed proxy is a valid drop-in. A
    no-op passthrough when ``recorder`` is ``None``."""
    if recorder is None:
        return sub_lm

    class _RecorderBoundSubLM:
        def __call__(self, *args: Any, **kwargs: Any):
            with recorder_scope(recorder):
                return sub_lm(*args, **kwargs)

        def __getattr__(self, name: str) -> Any:   # delegate dspy's bookkeeping (model/kwargs/…)
            return getattr(sub_lm, name)

    return _RecorderBoundSubLM()

logger = logging.getLogger(__name__)

# A validator returns None/"" when the text is acceptable, or an error string.
Validator = Callable[[str], str | None]
# A post-processor maps the validated text to its final form.
PostProcessor = Callable[[str], str]


class SubLMValidationError(RuntimeError):
    """Raised when the intercepted sub-LM exhausts its retry budget on an invalid response."""


def _import_base_lm():
    import dspy

    return dspy.LM


def intercept_sub_lm(
    base_lm: Any,
    *,
    validators: Sequence[Validator] = (),
    postprocessors: Sequence[PostProcessor] = (),
    max_retries: int = 2,
    name: str = "sub_lm",
) -> Any:
    """Wrap ``base_lm`` so the RLM's sub-LM escalations are intercepted and traced.

    This is THE hook for the sub-LM: ``dspy.RLM`` calls ``self.sub_lm(prompt)`` from
    its built-in ``llm_query`` / ``llm_query_batched`` tools, and the returned object
    sits in that ``sub_lm`` slot. On every call it records a ``sub_call`` trace event
    (the escalation's input + the sub-LM's raw/processed output) — that is the
    always-on job, and the only thing most consumers need. Passing ``validators`` /
    ``postprocessors`` additionally runs a deterministic validate → post-process
    pipeline (retrying on validation failure up to ``max_retries``); omit them and it
    is a pure tracing wrapper.

    ``base_lm`` is any ``dspy.LM`` (your local model, a cheaper API model, ...).
    The returned object is a drop-in ``sub_lm`` for ``RLMTask``/``dspy.RLM``.

    Constructed via a factory (not a module-level class) so importing this module
    never triggers a dspy import; the subclass is created on first call.
    """
    base_lm_cls = _import_base_lm()

    class _InterceptedSubLM(base_lm_cls):  # type: ignore[misc, valid-type]
        """A dspy.LM that delegates generation to ``base_lm`` then runs a pipeline."""

        def __init__(self) -> None:
            # Mirror the base model's identity so dspy bookkeeping stays sane,
            # without re-running base_lm's network/setup.
            self.model = getattr(base_lm, "model", name)
            self.kwargs = dict(getattr(base_lm, "kwargs", {}) or {})
            self._base = base_lm
            self._validators = list(validators)
            self._postprocessors = list(postprocessors)
            self._max_retries = max(1, max_retries)
            self._name = name

        # dspy.LM is callable as lm(prompt=..., messages=...) -> list[str].
        def __call__(self, *args: Any, **kwargs: Any):
            recorder = current_recorder()
            last_error: str | None = None
            # Capture the call input (the escalation prompt) for the trace / RL data.
            prompt = kwargs.get("prompt")
            if prompt is None:
                prompt = kwargs.get("messages")
            if prompt is None and args:
                prompt = args[0]
            input_repr = None if prompt is None else str(prompt)[:4000]

            for attempt in range(1, self._max_retries + 1):
                outputs = self._base(*args, **kwargs)
                texts = list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]
                processed, error = self._run_pipeline(texts[0] if texts else "")

                if recorder is not None:
                    recorder.record(
                        "sub_call",
                        {
                            # A sub_call is always one sub-LM escalation, reached via
                            # the RLM's built-in llm_query/llm_query_batched (which is
                            # the only thing that calls sub_lm). `kind` labels that role
                            # explicitly; `name` is this wrapper's label. We can't record
                            # WHICH built-in triggered it — dspy calls sub_lm identically
                            # for both — so don't infer llm_query vs _batched from here.
                            "kind": "sub_lm",
                            "name": self._name,
                            "model": self.model,
                            "attempt": attempt,
                            "input": input_repr,
                            "raw": texts[0] if texts else "",
                            "processed": processed,
                            "error": error,
                        },
                    )

                if error is None:
                    # Replace only the first completion with the processed text;
                    # preserve any additional completions untouched.
                    if texts:
                        texts[0] = processed
                    return texts
                last_error = error
                logger.warning(
                    "sub-LM %s validation failed (attempt %d/%d): %s",
                    self._name, attempt, self._max_retries, error,
                )

            raise SubLMValidationError(
                f"sub-LM {self._name!r} could not produce a valid response "
                f"after {self._max_retries} attempts: {last_error}"
            )

        def _run_pipeline(self, text: str) -> tuple[str, str | None]:
            for validate in self._validators:
                err = validate(text)
                if err:
                    return text, err
            processed = text
            for post in self._postprocessors:
                processed = post(processed)
            return processed, None

    return _InterceptedSubLM()


def model_as_tool(name: str, lm: Any, *, description: str = "") -> Callable[[str], str]:
    """Expose an extra model as an RLM tool for LM-decided multi-model routing.

    The main LM can call this from the REPL when it explicitly wants a different
    model than the default ``sub_lm``. Each call is recorded as a ``tool_call``.
    """

    def query_model(prompt: str) -> str:
        outputs = lm(prompt=prompt)
        text = outputs[0] if isinstance(outputs, (list, tuple)) and outputs else str(outputs)
        record_tool_call(f"model:{name}", args={"prompt": prompt}, result=text)
        return text

    query_model.__name__ = f"query_{name}"
    query_model.__qualname__ = query_model.__name__
    query_model.__doc__ = description or (
        f"Send a prompt to the '{name}' model and return its text response."
    )
    return query_model
