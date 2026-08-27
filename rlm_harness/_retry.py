"""The validation + retry engine shared by every RLM task.

This replaces the hand-rolled ``while execute_count < MAX_RETRY`` loops that were
copy-pasted across the original CVE app. It is deliberately free of any ``dspy``
import: it operates on a ``runner`` coroutine that returns a prediction-like
object (anything with attribute access), so it can be unit-tested with plain
objects and exercises the real logic we ship.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel

_DEFAULT_LOG = logging.getLogger(__name__)

_ERR_LOG_CAP = 600


def short_error(exc: BaseException, limit: int = _ERR_LOG_CAP) -> str:
    """Render a caught exception for a single log line, keeping the head AND tail.

    A failed attempt is logged with the exception's message, but some exceptions carry
    an enormous message: dspy's ``AdapterParseError`` embeds the ENTIRE raw LM completion,
    which for a degenerate/repetitive model is thousands of lines that then flood the
    terminal. The useful diagnostics live at the ends (the exception type + adapter at the
    head, the expected/actual field summary at the tail), so keep both and elide the middle.
    The full completion is not lost for debugging: it is on the wire in the model call.

    PUBLIC and SemVer-frozen since 1.5.0, re-exported from ``rlm_harness.__all__``. It was
    promoted because two independent consumers had reached into ``_retry`` for it — the kit's
    own signal that an internal seam should become a named hook rather than stay private.

    What is frozen is the BEHAVIOUR, not the string. Callers may rely on: a rendering already
    within ``limit`` is returned unchanged and begins ``f"{type(exc).__name__}: "``; a longer one
    keeps BOTH ends and states how much was dropped in between; the result is never longer than
    ``limit`` plus the elision marker; and it never raises for an ``Exception`` whose ``__str__``
    raises one (a ``BaseException`` such as ``KeyboardInterrupt`` from ``__str__`` still
    propagates, deliberately — that is not an error to render, it is an interrupt to honour).
    NOT frozen: the exact elision marker, how the budget is split between head and tail, the
    value of the default ``limit`` (only that a positive default exists), and — for a ``limit``
    too small to hold even the type name — anything beyond the length bound. Parse the output at
    your own risk; it is for humans and logs."""
    try:
        text = f"{type(exc).__name__}: {exc}"
    except Exception:
        # Every call site is an `except` block, so raising here would REPLACE a diagnosable
        # failure with an undiagnosable one. An exception whose own __str__ raises is rare but
        # real (a pydantic/adapter error holding a half-built object); the type name alone is
        # still the single most useful thing in the log line.
        text = f"{type(exc).__name__}: <unprintable: its __str__ raised>"
    if len(text) <= limit:
        return text
    head = (limit * 2) // 3
    # `max(1, ...)`: at limit<=1 the split leaves tail==0, and `text[-0:]` slices the WHOLE
    # string — turning the one call whose entire job is bounding output into an amplifier.
    tail = max(1, limit - head)
    return f"{text[:head]}... [{len(text) - limit} chars elided] ...{text[-tail:]}"


#: Pre-1.5.0 spelling, kept because consumers were importing it from here before it was public.
#: Not documented, not in ``__all__``, and not covered by the SemVer promise — use
#: ``rlm_harness.short_error``. Costs nothing to keep, and dropping it would break the very
#: callers whose need justified promoting the function.
_short_error = short_error


class RLMTaskError(RuntimeError):
    """Raised when a task fails to produce a valid result within the retry budget."""


def coerce_output(value: Any, model: type[BaseModel] | None) -> Any:
    """Coerce a raw RLM output field into a validated pydantic model.

    Accepts a model instance (returned as-is), a ``dict`` (validated), or a JSON
    string (parsed and validated). If ``model`` is ``None`` the value passes
    through untouched. Raises ``pydantic.ValidationError`` (or ``ValueError`` for
    unexpected types) on failure so the caller can retry.
    """
    if model is None:
        return value
    if isinstance(value, model):
        return value
    if isinstance(value, BaseModel):
        # A different model came back; revalidate via its dumped data.
        return model.model_validate(value.model_dump())
    if isinstance(value, dict):
        return model.model_validate(value)
    if isinstance(value, (str, bytes, bytearray)):
        return model.model_validate_json(value)
    raise ValueError(
        f"Cannot coerce output of type {type(value).__name__} into {model.__name__}"
    )


async def run_with_retry(
    runner: Callable[[], Awaitable[Any]],
    *,
    output_field: str,
    output_model: type[BaseModel] | None = None,
    max_retries: int = 3,
    logger: logging.Logger | None = None,
    non_retryable: tuple[type[BaseException], ...] = (),
    is_fast_fail: Callable[[BaseException], bool] | None = None,
) -> Any:
    """Run ``runner`` until it yields a valid output or the budget is exhausted.

    On each attempt: await ``runner``, pull ``output_field`` off the result, and
    (if ``output_model`` is set) validate/coerce it. Any exception — a model
    error, a missing field, a validation failure — consumes one attempt. After
    ``max_retries`` attempts the last error is wrapped in :class:`RLMTaskError`.

    ``non_retryable`` is a closed allowlist of exception TYPES a caller has
    already decided are not worth retrying — e.g. an explicit user-driven
    cancellation. A match propagates the ORIGINAL exception object verbatim,
    consuming NO attempt and never wrapped in :class:`RLMTaskError`: retrying an
    exception the caller raised on purpose to STOP the run would silently defeat
    the reason it exists (a cancelled sandbox turn respawning the whole
    trajectory from scratch), and wrapping it would make the caller's own
    ``except SandboxCancelled:`` (or whatever type they passed) unable to see it.
    The default ``()`` matches nothing, so every existing caller is unaffected.

    ``is_fast_fail`` is the same "don't retry, propagate verbatim, consume no
    attempt" behavior for a caught exception a static type tuple cannot express —
    e.g. "this dspy LM error is in a category dspy itself calls non-retryable,
    except for the one subtype that is worth retrying here for a reason dspy has
    no way to know about" (see ``_dspy_compat.is_fast_fail_lm_error``). Checked
    only for exceptions that fall through ``non_retryable`` first, since a type
    match there is cheaper and the two are not expected to overlap. This module
    stays dspy-free by construction: the predicate, like ``runner`` itself, is
    supplied by a dspy-aware caller. The default ``None`` never fires, so every
    existing caller is unaffected.
    """
    log = logger or _DEFAULT_LOG
    if max_retries < 1:
        raise ValueError("max_retries must be >= 1")

    last_error: BaseException | None = None
    for attempt in range(1, max_retries + 1):
        try:
            prediction = await runner()
            if not hasattr(prediction, output_field):
                raise AttributeError(
                    f"RLM prediction has no field {output_field!r}"
                )
            raw = getattr(prediction, output_field)
            return coerce_output(raw, output_model)
        except non_retryable:
            raise
        except Exception as exc:
            if is_fast_fail is not None and is_fast_fail(exc):
                raise
            last_error = exc
            log.warning(
                "RLM attempt %d/%d failed: %s", attempt, max_retries, short_error(exc)
            )

    raise RLMTaskError(
        f"Failed to produce a valid '{output_field}' after {max_retries} attempts"
    ) from last_error
