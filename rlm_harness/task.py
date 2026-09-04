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
import threading
from collections.abc import Callable, Sequence
from typing import Any, ClassVar

import dspy
from pydantic import BaseModel

from . import _dspy_compat
from ._retry import run_with_retry
from .config import RLMConfig
from .runtime import get_config, get_sub_lm
from .sandbox import SandboxCancelled, build_interpreter
from .sub_lm import _ensure_sub_call_recording, bind_recorder_to_sub_lm
from .trace import _ensure_tool_timing, current_recorder

logger = logging.getLogger(__name__)

try:
    from dspy.utils.callback import BaseCallback
except Exception:
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

    **Stages the OUTERMOST parse only, and that is load-bearing.** ``Adapter.__init_subclass__``
    re-wraps ``format`` and ``parse`` with ``with_callbacks`` for EVERY subclass — unconditionally,
    whether or not that subclass redefines them — so each subclass level adds a callback fire that
    a ``super().parse(...)`` call then traverses. Measured, one root turn:

        stock ``JSONAdapter``                     1 fire
        ``runtime._LenientJSONAdapter`` (DEFAULT) 2   (it calls ``super().parse``)
        a consumer subclass overriding NOTHING    3
        ...that also calls ``super().parse``      4

    So this is not a fixed double to divide by two: the depth is a property of the caller's adapter
    hierarchy, which is why the fix counts nesting rather than assuming a count. A consumer
    subclassing the kit's adapter — a plausible thing to do — was affected before this and needed
    no change to be covered by it. Two stamps per turn made ``record_main_trajectory``'s
    earliest-unused match order-unsafe: when a model repeats a ``reasoning`` string across turns
    (a retry loop emitting ``"Retrying tool call - …"`` does exactly this), a later turn consumed
    an earlier turn's spare stamp and inherited a time from several turns back. Measured on 85 real
    traces: every trace with a ts inversion had a duplicated reasoning, none of the 58 with unique
    reasoning inverted, and 2.1% of per-turn deltas came out NEGATIVE — one of them -338.7s, which
    a consumer rendered. Deduplicating HERE keeps the staged list 1:1 with turns, which is what
    makes the match an identity map; fixing it in the matcher instead cannot repair the case where
    the duplicate turns are ADJACENT (it merely stops the delta going negative while the stamp
    stays wrong, trading a loud failure for a silent one).

    The nesting depth is per-THREAD (dspy may parse on a worker thread) and read BEFORE the
    decrement, so ``n == 1`` is the outermost frame. ``n <= 1`` rather than ``n == 1`` is the
    deliberate degrade path: if a future dspy stops firing ``on_adapter_parse_start`` the counter
    never rises, every fire is treated as outermost, and behaviour falls back to exactly what it
    was before this change — not to staging NOTHING, which would silently return every main_step ts
    to the flush-time fallback with no test going red.
    """

    def __init__(self, recorder: Any) -> None:
        self._recorder = recorder
        self._depth = threading.local()

    def on_adapter_parse_start(self, call_id, instance=None, inputs=None):
        self._depth.n = getattr(self._depth, "n", 0) + 1

    def on_adapter_parse_end(self, call_id, outputs, exception=None):
        depth = getattr(self._depth, "n", 0)
        self._depth.n = max(0, depth - 1)
        if depth > 1:
            return  # a nested parse (the kit's own adapter calling super()) — the outer one stages
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
    except Exception:
        logger.debug("live main-step timing unavailable; main_step ts stays post-hoc", exc_info=True)
        yield
        return
    try:
        yield
    finally:
        try:
            cm.__exit__(None, None, None)
        except Exception:
            logger.debug("main-step timing context exit failed", exc_info=True)


def _applied_budgets(sub_lm: Any, config: Any, caps_dropped: bool) -> dict[str, Any]:
    """The generation caps ACTUALLY APPLIED, per role, for `run_end.payload.budgets`.

    Read off the LMs rather than from `RLMConfig`: `runtime` builds an LM from config only for a
    role still `None`, so an injected `main_lm`/`sub_lm` is used VERBATIM and the configured cap can
    be one the call never used -- exactly the consumer whose run died of a truncation. The MAIN LM
    is `dspy.settings.lm` (the task holds no reference to it); the sub-LM is the task's own.

    Named keys only -- `_dspy_compat.applied_lm_budget` never serialises `lm.kwargs`, which carries
    `api_key` for every LM the kit builds, and a trace is a shipped artifact.
    """
    out: dict[str, Any] = {}
    with contextlib.suppress(Exception):
        main = _dspy_compat.applied_lm_budget(dspy.settings.lm)
        if main is not None:
            out["main"] = main
    with contextlib.suppress(Exception):
        sub = _dspy_compat.applied_lm_budget(sub_lm)
        if sub is not None:
            out["sub"] = sub
    # The ITERATION caps, which are a DIFFERENT exhaustion from the token cap above -- and
    # `max_output_chars` is a THIRD, independent truncation mechanism (dspy head+tail-caps each
    # REPL output), which a reader diagnosing "truncation" has to be able to rule out.
    with contextlib.suppress(Exception):
        out["iterations"] = {
            "max_iterations": config.max_iterations,
            "max_llm_calls": config.max_llm_calls,
            "max_output_chars": config.max_output_chars,
            # TRUE when `_build_rlm`'s `except TypeError` fired: dspy rejected the budget kwargs and
            # every cap above reverted to dspy's own default. Without this the three numbers read as
            # applied when they were not.
            "dropped": bool(caps_dropped),
        }
    return out


class RLMTask:
    """Base class for a single RLM-backed task. Subclass and set the class vars."""

    #: DSPy signature string, e.g. "context: str -> answer: AnswerModel".
    signature: ClassVar[str] = ""
    #: Name of the output field in the signature (the part after "->").
    output_field: ClassVar[str] = ""
    #: Optional pydantic model the output is validated/coerced into.
    output_model: ClassVar[type[BaseModel] | None] = None
    #: Natural-language instructions attached to the signature.
    instructions: ClassVar[str] = ""
    #: Tools (plain callables) the RLM may invoke inside the REPL.
    #: NOT a ``ClassVar``: since 1.1.0 an instance may carry its own tool list (declare it in the
    #: class body, set ``self.tools`` in a subclass ``__init__``, or pass ``tools=`` — see below).
    #: The annotation said ``ClassVar`` while ``examples/harness_run.py`` was already assigning
    #: per instance, so it was describing a rule the kit itself did not follow; the package ships
    #: ``py.typed``, so that lie reached consumers' type checkers.
    tools: Sequence[Callable[..., Any]] = ()

    #: Class-level default so `resolved_tools` still answers for a subclass that forgets to call
    #: `super().__init__()` — it would otherwise raise AttributeError from a read-only property.
    _tools_override: Sequence[Callable[..., Any]] | None = None

    def __init__(
        self,
        *,
        config: RLMConfig | None = None,
        sub_lm: dspy.LM | None = None,
        max_retries: int | None = None,
        interpreter: Any | None = None,
        cancel_event: threading.Event | None = None,
        tools: Sequence[Callable[..., Any]] | None = None,
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
        # An explicit interpreter OBJECT overrides `config.interpreter` (the string that
        # `sandbox.build_interpreter` maps to a sandbox). This is a TEST/advanced seam — mainly
        # `rlm_harness.testing.ScriptedInterpreter`, to drive the forward path offline — and it bypasses
        # `build_interpreter` (and its insecure-interpreter guard) exactly like an injected `sub_lm`
        # bypasses the real model: the caller supplies and owns the double. The default (None) keeps the
        # string path and the guard.
        self._interpreter = interpreter
        # An "advanced seam" kwarg exactly like `interpreter=` above, placed on __init__ rather than
        # `arun()` for the same reason: every real consumer already constructs a fresh RLMTask
        # instance per run, so per-instance placement loses nothing versus per-call. Threaded into
        # `build_interpreter(...)` in `_build_rlm()`; has NO effect when `interpreter=` (above)
        # bypasses `build_interpreter` entirely — a caller supplying their own interpreter object
        # owns its cancellation behavior too, exactly like `ScriptedInterpreter` owns its own.
        self._cancel_event = cancel_event
        # Set per build by `_build_rlm`: the interpreter that `arun` passes to forward() as
        # its first positional arg. Initialised here so the attribute always exists, even for
        # a caller that inspects a task it never ran.
        self._forward_interpreter: Any | None = None

        # `tools=` is stashed and resolved in `_build_rlm`, NOT written to `self.tools` here.
        # That is what makes it ORDER-INDEPENDENT. Assigning `self.tools` in `__init__` would
        # make the winner depend on where the subclass calls `super().__init__()`:
        #     self.tools = [...]  ; super().__init__(**kw)   -> the kwarg wins
        #     super().__init__(**kw) ; self.tools = [...]    -> the kwarg is SILENTLY LOST
        # and the second is the more idiomatic ordering. Resolving at build time means the
        # explicit kwarg always wins, whichever way the subclass is written.
        #
        # This is NOT an injection seam like `interpreter=` / `sub_lm=` above — those bypass a
        # guard (the sandbox builder, the real model) and the caller owns the double. This
        # bypasses nothing; it is a per-instance override of a declaration field. Keep the two
        # ideas apart: conflating them dilutes a distinction the sandbox guard depends on.
        self._tools_override: Sequence[Callable[..., Any]] | None = (
            None if tools is None else list(tools)
        )

    @property
    def resolved_tools(self) -> Sequence[Callable[..., Any]]:
        """The tools this task will actually hand the model — the ONE derivation.

        An explicit ``tools=`` kwarg REPLACES the class-body / ``self.tools`` declaration; it
        never merges, because merging would make the effective list depend on inheritance depth.
        ``tools=[]`` is therefore a deliberate "no tools", distinct from the ``None`` default,
        which leaves the declaration path completely untouched.

        Exposed rather than inlined into :meth:`_build_rlm` because the answer to "what will
        this task give the model?" is otherwise unanswerable from outside: ``self.tools`` stops
        being the whole truth once ``tools=`` is in play, and anything that needs the real list
        (``rlm_harness.testing.assert_task_repl_safe``, a consumer's own introspection, a
        debugger) would have to re-derive the rule — the two-derivations-of-one-value hazard.
        """
        if self._tools_override is not None:
            return self._tools_override
        return self.tools

    def _build_rlm(self) -> dspy.RLM:
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
        interpreter = self._interpreter if self._interpreter is not None else build_interpreter(
            self._config.interpreter,
            allow_insecure=self._config.allow_insecure_sandbox,
            container=self._config.container,
            turn_timeout_s=self._config.sandbox_turn_timeout_s,
            cancel_event=self._cancel_event,
        )
        # We now construct the deno/pyodide interpreter ourselves (to inject the
        # JSON-literal aliases), so its teardown is ours: dspy.RLM only shuts down
        # an interpreter it built itself. Stash it for _teardown_interpreter().
        self._built_interpreter = interpreter

        # Tools go in TIMED. `_ensure_tool_timing` publishes a start time that
        # `record_tool_call` reads when the tool did not measure itself; it never records an event
        # of its own, so it cannot double-count. Applied HERE rather than in each factory for the
        # reason 1.7.0 made `sub_call` automatic: a field every author must remember is missing for
        # someone -- and a consumer whose tools are pure delegation to our factories has no seam of
        # its own to add one. This covers the consumer's own tools too. See `_ensure_tool_timing`.
        kwargs: dict[str, Any] = {
            "sub_lm": self._sub_lm,
            "tools": [_ensure_tool_timing(t) for t in self.resolved_tools],
        }

        # The caller's interpreter goes to forward()/aforward() as the first POSITIONAL
        # argument (dspy >= 3.3.0), not to the constructor — so stash it for `arun`.
        # OWNERSHIP stays ours: dspy shuts down only an interpreter it created itself, which
        # is what keeps `_teardown_interpreter` correct. So never SUPPLY the interpreter via
        # `interpreter_factory=`: dspy DOES shut down whatever that factory returns, which
        # would double-shutdown our sandbox.
        self._forward_interpreter = interpreter

        # ...which is why the kwargs below need reading carefully: from dspy 3.3.1 they MAY
        # contain an `interpreter_factory`, and it is NOT a way of supplying an interpreter.
        # dspy sources the prompt's "Execution environment:" text from that object's
        # `execution_instructions` attribute, so this passes a metadata CARRIER dspy only ever
        # reads — never calls, and it raises if it ever is. Without it every run is described to
        # the model as Pyodide, including a `container` run that can genuinely spawn
        # subprocesses. See `_dspy_compat.interpreter_instructions_kwargs`.
        kwargs.update(_dspy_compat.interpreter_instructions_kwargs(interpreter))

        # Budget caps are mapped onto the names the installed dspy accepts (3.3.x renamed
        # `max_iterations` to `max_iters`). The `except TypeError` below is now only a
        # backstop for an unknown future signature — and it is a LOSSY one, so the
        # mapping has to be right: it drops every cap to dspy's defaults, silently.
        # RESET per build, not just set on failure: without this the flag survives into a later
        # run of the SAME task instance and claims caps were dropped when they were not.
        self._budget_caps_dropped = False
        budget = _dspy_compat.rlm_budget_kwargs(
            max_iterations=self._config.max_iterations,
            max_llm_calls=self._config.max_llm_calls,
            max_output_chars=self._config.max_output_chars,
        )
        try:
            return dspy.RLM(signature, **kwargs, **budget)
        except TypeError:
            # Recorded, not just logged: this path drops all three iteration caps to dspy's own
            # defaults, so a trace carrying the CONFIGURED values here would be a fiction. A log
            # line rotates; the trace is what outlives the run.
            self._budget_caps_dropped = True
            logger.warning(
                "dspy.RLM rejected the budget kwargs %s — building WITHOUT budget caps "
                "(max_iterations/max_llm_calls/max_output_chars fall back to dspy's own "
                "defaults). This usually means dspy renamed them again; update "
                "rlm_harness._dspy_compat._BUDGET_ALIASES.",
                sorted(budget),
            )
            return dspy.RLM(signature, **kwargs)

    async def arun(self, **inputs: Any) -> Any:
        """Run the task asynchronously, returning the validated output.

        If a :class:`rlm_harness.trace.TraceRecorder` is active in the current
        context, the main LM trajectory and the final result are recorded after
        the run (sub-LM and tool events are recorded live during it).
        """
        rlm = self._build_rlm()
        forward_args = _dspy_compat.forward_interpreter_args(self._forward_interpreter)
        # Two wrappers, and the ORDER is load-bearing.
        #
        # INNER (`_ensure_sub_call_recording`, 1.7.0): make the escalation record itself even when
        # the caller never asked. Without it a plain `dspy.LM` sub_lm emits NO sub_call at all, and
        # a corpus of zeros is indistinguishable from "the model never escalated" — an ambiguity
        # that reached a design decision in this repo before it was caught. A caller who wrapped
        # its own sub-LM (for a validate/post-process pipeline) declares `records_sub_call` and is
        # left untouched, so the customisation tier is unaffected.
        #
        # OUTER (`bind_recorder_to_sub_lm`): re-establish the recorder in the CALLING thread, since
        # dspy's llm_query_batched fans the sub-LM across workers. It must stay outermost: its
        # __call__ enters `recorder_scope` and only then calls inward, while the interceptor reads
        # `current_recorder()` at call time. Reversed, the interceptor sees None and SILENTLY skips
        # the record — no error, no event, which is the failure mode hardest to notice.
        #
        # Per-run (this rlm is fresh), so concurrent runs sharing the base sub-LM don't
        # cross-contaminate. Both are skipped entirely when there is no recorder.
        _rec = current_recorder()
        if _rec is not None and getattr(rlm, "sub_lm", None) is not None:
            rlm.sub_lm = bind_recorder_to_sub_lm(_ensure_sub_call_recording(rlm.sub_lm), _rec)
        captured: dict[str, Any] = {}
        # Opened around the whole retry loop, and REUSING a tracker the caller already installed:
        # dspy creates one only when none is set, so installing unconditionally would shadow a
        # consumer's own `with dspy.track_usage(): await task.arun(...)` and hand them zero entries
        # for everything inside -- this kit writing a structural zero into someone else's
        # measurement. Closed in the same `finally` as the interpreter teardown.
        _usage_stack = contextlib.ExitStack()
        tracker = _usage_stack.enter_context(_dspy_compat.usage_tracking())
        # One entry per ATTEMPT. `run_with_retry` re-runs the whole trajectory, and the attempt
        # whose turns reach the trace is NOT always the last: `captured["prediction"]` is
        # last-writer-wins, so a run whose FINAL attempt raises keeps an EARLIER attempt's turns.
        # Scoping usage to "the attempt with the turns" would discard the fatal call's usage --
        # the one number this records at all.
        attempt_usage: list[dict[str, Any]] = []

        async def runner() -> Any:
            # Capture each turn's LIVE timestamp as dspy parses it, so the post-hoc
            # record_main_trajectory can backfill real per-turn ts. begin_main_capture resets the
            # buffer per attempt (a retry re-runs the RLM; only the final attempt is recorded).
            recorder = current_recorder()
            if recorder is not None and hasattr(recorder, "begin_main_capture"):
                recorder.begin_main_capture()
            index = len(attempt_usage)
            baseline = _dspy_compat.usage_baseline(tracker) if tracker is not None else {}
            try:
                with _live_main_timing(recorder):
                    # On dspy 3.3.x a caller-owned interpreter is the first POSITIONAL
                    # argument here rather than a constructor kwarg; empty tuple when the task
                    # has no caller-owned interpreter.
                    prediction = await rlm.aforward(*forward_args, **inputs)
                captured["prediction"] = prediction
                captured["attempt"] = index
                return prediction
            finally:
                # In a `finally`, because the RAISING attempt is the entire point -- read after
                # the call returns and its slice is lost with the exception.
                if tracker is not None:
                    # Appended UNCONDITIONALLY: `index` is `len(attempt_usage)`, so skipping an
                    # attempt that recorded no calls would hand the NEXT attempt the same index.
                    # An attempt with no calls is also a fact worth keeping -- it failed before
                    # reaching the LM.
                    attempt_usage.append(
                        {"attempt": index, "calls": _dspy_compat.usage_since(tracker, baseline)}
                    )

        def _stage_budgets_and_usage() -> None:
            """Fold the applied caps and the per-attempt usage into `run_end`.

            Called on the success AND failure paths: the failure path is the one this exists for.
            Which attempt's turns reached the trace is resolvable only HERE, after the loop --
            `captured["prediction"]` is last-writer-wins, so an attempt that produced a prediction
            may still have had its turns overwritten by a later one. A flag stamped per attempt
            would mark BOTH, telling a reader the usage and the turns agree when they do not.
            """
            rec = current_recorder()
            if rec is None:
                return
            with contextlib.suppress(Exception):
                budgets = _applied_budgets(
                    self._sub_lm, self._config, getattr(self, "_budget_caps_dropped", False)
                )
                if budgets and hasattr(rec, "note_budgets"):
                    rec.note_budgets(budgets)
                if attempt_usage and hasattr(rec, "note_usage"):
                    recorded = captured.get("attempt")
                    for entry in attempt_usage:
                        entry["turns_recorded"] = entry["attempt"] == recorded
                    rec.note_usage(attempt_usage)

        try:
            try:
                result = await run_with_retry(
                    runner,
                    output_field=self.output_field,
                    output_model=self.output_model,
                    max_retries=self._max_retries,
                    logger=logger,
                    # A SandboxCancelled means a caller explicitly asked for this run to
                    # STOP — retrying would transparently respawn the sandbox and restart
                    # the whole trajectory from scratch, silently absorbing the cancel.
                    non_retryable=(SandboxCancelled,),
                    # An LM error dspy itself calls non-retryable (auth/billing/config/
                    # invalid-request/unsupported-model) burns the whole retry budget
                    # re-running the same doomed trajectory for nothing. Fails fast instead,
                    # except ContextWindowExceededError — see is_fast_fail_lm_error's docstring
                    # for why that one keeps retrying.
                    is_fast_fail=_dspy_compat.is_fast_fail_lm_error,
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
                _stage_budgets_and_usage()
                raise

            recorder = current_recorder()
            if recorder is not None:
                if "prediction" in captured:
                    recorder.record_main_trajectory(captured["prediction"])
                recorder.record_result(result)
            _stage_budgets_and_usage()
            return result
        finally:
            self._teardown_interpreter()
            _usage_stack.close()

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
            except Exception:
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
