"""Phase A — ``intercept_sub_lm``: the one hook to intercept the RLM's sub-LM.

``dspy.RLM`` exposes no hook to intercept a sub-LLM response before it returns to
the main model — and its built-in ``llm_query`` / ``llm_query_batched`` tools just
call ``self.sub_lm(prompt)``. So the ONLY interception point is the sub_lm object
itself. ``intercept_sub_lm`` wraps a ``dspy.LM``: ``RLM`` only sees "a sub_lm", but
inside we emit a ``sub_call`` trace event for every escalation and (optionally) run
a deterministic pipeline — call the base model, validate the format, post-process.
**Since 1.7.0 a consumer no longer has to reach for this to get the event**: ``RLMTask`` wraps a
plain ``sub_lm`` for recording at the seam that binds the recorder, because an escalation that is
invisible is indistinguishable from one that never happened. Call it yourself for the
validators/postprocessors, or to opt out by declaring ``records_sub_call = True`` on a wrapper of
your own.

Design decisions baked in (per the approved plan):

- The sub-LM intercept does **deterministic transforms only** (validate + post-process).
  Agentic actions (calling an external tool) are *not* forced here; they are
  exposed to the main LM as RLM tools (see ``model_as_tool`` and
  ``rlm_harness.skills``), so the decision stays in the LM's hands and lands in the
  trajectory.
- Multi-model routing is done today via ``model_as_tool`` (LM-decided), not the
  unmerged official ``sub_lms`` API. When ``sub_lms`` ships, swap it in without
  touching task code.

``dspy`` is imported lazily so this module stays importable (and the pipeline
logic stays unit-testable) without a full dspy install.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from typing import Any

from . import _dspy_compat
from ._toolname import sanitize_tool_name
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
    (the escalation's input + the sub-LM's raw/processed output). **Since 1.7.0 most consumers do
    not need to call this at all**: ``RLMTask`` wraps a plain ``sub_lm`` for that recording
    automatically. Reach for it when you want the deterministic validate → post-process pipeline —
    pass ``validators`` / ``postprocessors`` and it retries on validation failure up to
    ``max_retries``. Omit them and it is a pure tracing wrapper, which is exactly what the
    automatic path installs.

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

        #: Marks this object as ALREADY emitting its own ``sub_call`` events, so
        #: ``_ensure_sub_call_recording`` leaves it alone. A duck-typed protocol, not an
        #: isinstance check: a consumer with its OWN recording wrapper sets this to ``True`` and
        #: the kit stays out of the way, the same convention ``execution_instructions`` uses on an
        #: interpreter. Probed with ``is True``, never truthiness — see that function.
        records_sub_call = True

        def __getattr__(self, attr: str) -> Any:
            # Delegate anything this wrapper does not define to the wrapped LM. `__init__` copies
            # only `model` and `kwargs`, so without this the wrapper is missing everything else a
            # real dspy.LM carries (`history`, `cache`, `num_retries`, `callbacks`, …). That was
            # tolerable while wrapping was opt-in; since 1.7.0 the kit substitutes this object for
            # the caller's LM automatically, so "observationally identical to the bare one" has to
            # hold for attribute access too, not just for the return value. `records_sub_call` and
            # the real attributes are found normally and never reach here.
            #
            # Read `_base` out of `__dict__`, NOT as `self._base`: `copy`/`deepcopy`/`pickle`
            # rebuild an instance WITHOUT calling `__init__`, so `_base` is absent and
            # `self._base` would re-enter this method forever. `dspy.BaseLM.copy()` does exactly
            # that, and it is the documented way to get a rollout-id variant of an LM, so the
            # recursion would fire on a supported path.
            base = self.__dict__.get("_base")
            if base is None:
                raise AttributeError(attr)
            return getattr(base, attr)

        # dspy's `RLM._query_lm` accepts a typed `LMResponse` OR the legacy `list[str | dict]`.
        # This returns whichever the BASE LM returned, never a shape of its own choosing.
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
                # SHAPE-PRESERVING. This used to be `[outputs]` for anything non-list, which turned
                # a typed `LMResponse` into `[LMResponse]` and made dspy raise "Sub-LM response must
                # contain text, got LMResponse" — invisible on the default path, fatal under
                # `dspy.context(experimental=True)`, and on course to become the DEFAULT after dspy
                # 3.4. Both the read and the rebuild are resolved in `_dspy_compat`, never here.
                raw = _dspy_compat.sub_lm_response_text(outputs)
                processed, error = self._run_pipeline(raw if raw is not None else "")

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
                            "raw": raw,          # None = the shape was not recognised
                            "processed": processed,
                            "error": error,
                        },
                    )

                if error is None:
                    # A no-op pipeline returns the base LM's object UNTOUCHED — identity, not a
                    # reconstruction. That is what makes automatic wrapping safe: a sub-LM the kit
                    # wrapped on the caller's behalf must be indistinguishable from the bare one.
                    #
                    # `raw is None` means the shim did not RECOGNISE the shape. Return it untouched
                    # too, so dspy raises its own clear error. Rebuilding it as `[""]` would turn a
                    # loud TypeError into a silent EMPTY completion that reaches the planner and
                    # then the RL data as a real escalation answer — and since 1.7.0 wraps every
                    # sub-LM automatically, that would be inflicted on callers who never opted in.
                    if raw is None or processed == raw:
                        return outputs
                    return _dspy_compat.sub_lm_response_with_text(outputs, processed)
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
        # The one model-backed tool the KIT itself records (`make_model_tool` and
        # `make_harness_tool` are side-effect-free bases whose consumer owns the recording), so
        # it is also the one whose duration the kit can supply without the consumer doing it.
        t0 = time.perf_counter()
        outputs = lm(prompt=prompt)
        # Read through the shim, never by indexing: dspy's LMs return a typed `LMResponse` on the
        # experimental path and the legacy list otherwise, and `outputs[0]` on the former yielded
        # `str(LMResponse)` — the whole repr, handed to the model AND written to the trace as the
        # tool's result. Same defect the sub-LM path carried; fixed in the same place, once.
        text = _dspy_compat.sub_lm_response_text(outputs)
        if text is None:
            text = str(outputs)
        record_tool_call(f"model:{name}", args={"prompt": prompt},
                         duration_s=time.perf_counter() - t0, result=text)
        return text

    # SANITISED, because `name` is a model id and a real one is not an identifier:
    # `openai/gpt-4o-mini` gave `query_openai/gpt-4o-mini`, which dspy refuses outright
    # — the tool could never be registered at all. The RAW id stays on
    # the `record_tool_call(f"model:{name}", …)` above: that is the trace identity and it
    # must not move, so a reader still sees which model was actually consulted.
    query_model.__name__ = sanitize_tool_name(f"query_{name}")
    query_model.__qualname__ = query_model.__name__
    query_model.__doc__ = description or (
        f"Send a prompt to the '{name}' model and return its text response."
    )
    return query_model


def _ensure_sub_call_recording(sub_lm: Any) -> Any:
    """Return a sub-LM that emits ``sub_call`` events, wrapping only if it does not already.

    ``CLAUDE.md`` states as an invariant that a sub-LM escalation "is recorded as a ``sub_call``".
    Before 1.7.0 that held only when the CONSUMER remembered to call :func:`intercept_sub_lm`
    itself — a plain ``dspy.LM`` is invoked by dspy directly and records nothing. Surveyed across
    nine consumers, four never wrapped; two of those four had corpora, 141 traces, in which
    ``sub_call`` was identically zero and therefore indistinguishable from "measured, and the model
    never escalated". That ambiguity reached a design decision in this repo before it was caught.

    Wrapping with no ``validators``/``postprocessors`` is a PURE tracing wrapper: a no-op pipeline
    returns the base LM's response object untouched, so the wrapped sub-LM is observationally
    identical to the bare one apart from the event it emits.

    **The probe is ``is True``, deliberately, not truthiness.** ``getattr`` on a ``unittest.mock``
    double manufactures a truthy attribute for any name, so a truthiness test would silently decide
    a mock "already records" and skip it — recreating the exact absent-event failure this exists to
    fix, one layer up. The ``except Exception`` covers a lazy proxy whose ``__getattr__`` raises
    something other than ``AttributeError``; a probe must never be able to fail a run.
    """
    if sub_lm is None:
        return sub_lm
    try:
        if getattr(sub_lm, "records_sub_call", False) is True:
            return sub_lm
        return intercept_sub_lm(sub_lm)
    except Exception:
        # The WHOLE body is guarded, not just the probe. `intercept_sub_lm` reads `.model` and
        # `.kwargs` off the base LM, which is enough to raise on a bare `unittest.mock.Mock` (its
        # `.kwargs` is a Mock, and `dict()` of it raises) or on a lazy proxy that survived the
        # probe. Auto-wrapping is an observability convenience the caller never asked for; it must
        # never be the reason a run fails to start.
        logger.warning(
            "could not auto-wrap sub_lm for sub_call tracing; escalations will go unrecorded",
            exc_info=True,
        )
        return sub_lm
