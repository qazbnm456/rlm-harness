"""The ``RLMTask`` base class — the one abstraction this scaffold exists for.

A task is declared by subclassing ``RLMTask`` and filling four fields:

    class Summarize(RLMTask):
        signature = "document: str -> article: Article"
        output_field = "article"
        output_model = Article                 # a pydantic BaseModel
        instructions = "Summarize the document into a title and a paragraph."
        tools = [make_schema_validator(Article)]

Everything else — building ``dspy.RLM``, choosing the sandbox, budget caps,
retrying on validation failure, observability — is inherited. A consumer's
near-identical RLM call sites collapse to a few lines each (see
``examples/harness_run.py``).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any, Callable, ClassVar, Optional, Sequence, Type

import dspy
from pydantic import BaseModel

from ._retry import run_with_retry
from .config import RLMConfig
from .runtime import get_config, get_sub_lm
from .sandbox import build_interpreter
from .sub_lm import bind_recorder_to_sub_lm
from .trace import current_recorder

logger = logging.getLogger(__name__)

try:
    from dspy.utils.callback import BaseCallback
except Exception:  # noqa: BLE001 — dspy internals moved; live main-step timing degrades to post-hoc ts
    BaseCallback = object  # type: ignore


class _MainStepTimer(BaseCallback):  # type: ignore[misc, valid-type]
    """dspy parse callback that timestamps each ROOT planner turn LIVE, so the ``TraceRecorder`` can
    backfill main_step ts. dspy.RLM only exposes its trajectory post-hoc (on the final ``Prediction``),
    so without this the recorder stamps every main_step at finalize time.

    A ROOT-planner turn is the only adapter parse carrying BOTH ``reasoning`` and ``code`` (a lifeline
    parse lacks ``code``; the extract-fallback parse carries the output fields, not these) — the same
    filter a streaming consumer's callback uses. Holds a DIRECT recorder reference (not the
    contextvar) so it works regardless of which thread dspy parses on; the recorder's note_main_step
    is itself thread-safe.
    """

    def __init__(self, recorder: Any) -> None:
        self._recorder = recorder

    def on_adapter_parse_end(self, call_id, outputs, exception=None):  # noqa: ANN001
        if isinstance(outputs, dict) and "reasoning" in outputs and "code" in outputs:
            self._recorder.note_main_step(outputs.get("reasoning"))


@contextlib.contextmanager
def _live_main_timing(recorder: Any):
    """Install :class:`_MainStepTimer` into dspy's callback list for the duration, MERGING with any
    callbacks the consumer already set (dspy gathers ``settings.callbacks + instance.callbacks``, so
    appending coexists with e.g. a consumer's SSE callback). A no-op when there is no recorder, or when
    dspy's callback context can't be entered — the trace then keeps post-hoc main_step ts, no worse
    than before.
    """
    if recorder is None or not hasattr(recorder, "note_main_step"):
        yield
        return
    try:
        existing = list(dspy.settings.get("callbacks") or [])
        cm = dspy.context(callbacks=existing + [_MainStepTimer(recorder)])
        cm.__enter__()
    except Exception:  # noqa: BLE001 — instrumentation must never break the run
        logger.debug("live main-step timing unavailable; main_step ts stays post-hoc", exc_info=True)
        yield
        return
    try:
        yield
    finally:
        try:
            cm.__exit__(None, None, None)
        except Exception:  # noqa: BLE001
            logger.debug("main-step timing context exit failed", exc_info=True)


class RLMTask:
    """Base class for a single RLM-backed task. Subclass and set the class vars."""

    #: DSPy signature string, e.g. "context: str -> answer: AnswerModel".
    signature: ClassVar[str] = ""
    #: Name of the output field in the signature (the part after "->").
    output_field: ClassVar[str] = ""
    #: Optional pydantic model the output is validated/coerced into.
    output_model: ClassVar[Optional[Type[BaseModel]]] = None
    #: Natural-language instructions attached to the signature.
    instructions: ClassVar[str] = ""
    #: Tools (plain callables) the RLM may invoke inside the REPL.
    tools: ClassVar[Sequence[Callable[..., Any]]] = ()

    def __init__(
        self,
        *,
        config: Optional[RLMConfig] = None,
        sub_lm: Optional["dspy.LM"] = None,
        max_retries: Optional[int] = None,
    ) -> None:
        if not self.signature:
            raise ValueError(f"{type(self).__name__} must define `signature`")
        if not self.output_field:
            raise ValueError(f"{type(self).__name__} must define `output_field`")

        self._config = config or get_config()
        self._sub_lm = sub_lm or get_sub_lm()
        self._max_retries = (
            max_retries if max_retries is not None else self._config.max_retries
        )

    def _build_rlm(self) -> "dspy.RLM":
        # Resolve a custom output type (e.g. "-> finding: Finding") explicitly via
        # dspy's custom_types. Otherwise dspy.Signature resolves the type *name* by
        # walking the call stack's globals/locals — which works only while a caller
        # frame happens to hold the name, and raises "Unknown name" for
        # dynamically-built types or runner-driven call paths. (See CHANGELOG.md.)
        sig_kwargs: dict[str, Any] = {}
        instructions = self.instructions or None
        if self.output_model is not None:
            sig_kwargs["custom_types"] = {self.output_model.__name__: self.output_model}
            # dspy silently drops custom_types when instructions is None (it
            # re-parses the signature via Signature(sig, "") without them). Pass an
            # empty string instead of None so the explicit binding survives even for
            # a task that declared no instructions.
            if instructions is None:
                instructions = ""
        signature = dspy.Signature(
            self.signature, instructions=instructions, **sig_kwargs
        )
        interpreter = build_interpreter(
            self._config.interpreter,
            allow_insecure=self._config.allow_insecure_sandbox,
            container=self._config.container,
        )
        # We now construct the deno/pyodide interpreter ourselves (to inject the
        # JSON-literal aliases), so its teardown is ours: dspy.RLM only shuts down
        # an interpreter it built itself. Stash it for _teardown_interpreter().
        self._built_interpreter = interpreter

        kwargs: dict[str, Any] = {
            "sub_lm": self._sub_lm,
            "tools": list(self.tools),
        }
        if interpreter is not None:
            kwargs["interpreter"] = interpreter

        # Budget controls are passed best-effort: dspy's exact kwarg names have
        # shifted across releases, so tolerate their absence rather than crash.
        # (All-or-nothing: one unknown kwarg drops the whole dict to dspy defaults.)
        budget = {
            "max_iterations": self._config.max_iterations,
            "max_llm_calls": self._config.max_llm_calls,
            "max_output_chars": self._config.max_output_chars,
        }
        try:
            return dspy.RLM(signature, **kwargs, **budget)
        except TypeError:
            logger.debug("dspy.RLM rejected budget kwargs; building without them.")
            return dspy.RLM(signature, **kwargs)

    async def arun(self, **inputs: Any) -> Any:
        """Run the task asynchronously, returning the validated output.

        If a :class:`rlm_kit.trace.TraceRecorder` is active in the current
        context, the main LM trajectory and the final result are recorded after
        the run (sub-LM and tool events are recorded live during it).
        """
        rlm = self._build_rlm()
        # Bind the active recorder to the sub_lm so dspy's llm_query_batched — which fans the sub-LM
        # across a ThreadPoolExecutor whose workers DON'T inherit the recorder ContextVar — still records
        # each escalation as a sub_call (else the lifeline metric under-counts). Per-run (this rlm is
        # fresh), so concurrent runs sharing the base sub-LM don't cross-contaminate.
        _rec = current_recorder()
        if _rec is not None and getattr(rlm, "sub_lm", None) is not None:
            rlm.sub_lm = bind_recorder_to_sub_lm(rlm.sub_lm, _rec)
        captured: dict[str, Any] = {}

        async def runner() -> Any:
            # Capture each turn's LIVE timestamp as dspy parses it, so the post-hoc
            # record_main_trajectory can backfill real per-turn ts. begin_main_capture resets the
            # buffer per attempt (a retry re-runs the RLM; only the final attempt is recorded).
            recorder = current_recorder()
            if recorder is not None and hasattr(recorder, "begin_main_capture"):
                recorder.begin_main_capture()
            with _live_main_timing(recorder):
                prediction = await rlm.aforward(**inputs)
            captured["prediction"] = prediction
            return prediction

        try:
            try:
                result = await run_with_retry(
                    runner,
                    output_field=self.output_field,
                    output_model=self.output_model,
                    max_retries=self._max_retries,
                    logger=logger,
                )
            except Exception:
                # The run FAILED (e.g. the result never coerced into output_model after the retry
                # budget). Still record the LAST attempt's trajectory so the failed run is
                # navigable/debuggable — recording only on success left a failed run with ZERO
                # main_steps, blind on the planner side (exactly when you most need to see what it
                # did). We do NOT record a result (there is none); run_end already carries the error,
                # and every reader keys success off the RESULT event, so the run stays correctly
                # "failed" and the SFT keep-filter (complete+valid) still excludes it. Then re-raise.
                recorder = current_recorder()
                if recorder is not None and "prediction" in captured:
                    recorder.record_main_trajectory(captured["prediction"])
                raise

            recorder = current_recorder()
            if recorder is not None:
                if "prediction" in captured:
                    recorder.record_main_trajectory(captured["prediction"])
                recorder.record_result(result)
            return result
        finally:
            self._teardown_interpreter()

    def _teardown_interpreter(self) -> None:
        """Shut down the sandbox interpreter built for this run, if any.

        dspy.RLM tears down only an interpreter it constructed itself; because we
        now supply the deno/pyodide one (to inject the JSON-literal aliases), its
        lifecycle is ours. Best-effort: a mock interpreter's ``shutdown`` is a
        no-op, and a teardown failure must never mask the run's result/exception.
        """
        interp = getattr(self, "_built_interpreter", None)
        if interp is None:
            return
        self._built_interpreter = None
        shutdown = getattr(interp, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:  # noqa: BLE001 — teardown must not mask the result
                logger.debug("interpreter shutdown raised; ignoring", exc_info=True)

    def run(self, **inputs: Any) -> Any:
        """Synchronous convenience wrapper around :meth:`arun` for scripts.

        Do not call this from inside a running event loop; use ``arun`` there.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.arun(**inputs))
        raise RuntimeError(
            "RLMTask.run() cannot be called from a running event loop; "
            "await RLMTask.arun() instead."
        )
