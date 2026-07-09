"""The validation + retry engine shared by every RLM task.

This replaces the hand-rolled ``while execute_count < MAX_RETRY`` loops that were
copy-pasted across the original CVE app. It is deliberately free of any ``dspy``
import: it operates on a ``runner`` coroutine that returns a prediction-like
object (anything with attribute access), so it can be unit-tested with plain
objects and exercises the real logic we ship.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Optional, Type

from pydantic import BaseModel

_DEFAULT_LOG = logging.getLogger(__name__)

_ERR_LOG_CAP = 600


def _short_error(exc: BaseException, limit: int = _ERR_LOG_CAP) -> str:
    """Render a caught exception for a single log line, keeping the head AND tail.

    A failed attempt is logged with the exception's message, but some exceptions carry
    an enormous message: dspy's ``AdapterParseError`` embeds the ENTIRE raw LM completion,
    which for a degenerate/repetitive model is thousands of lines that then flood the
    terminal. The useful diagnostics live at the ends (the exception type + adapter at the
    head, the expected/actual field summary at the tail), so keep both and elide the middle.
    The full completion is not lost for debugging: it is on the wire in the model call."""
    text = f"{type(exc).__name__}: {exc}"
    if len(text) <= limit:
        return text
    head = (limit * 2) // 3
    tail = limit - head
    return f"{text[:head]}... [{len(text) - limit} chars elided] ...{text[-tail:]}"


class RLMTaskError(RuntimeError):
    """Raised when a task fails to produce a valid result within the retry budget."""


def coerce_output(value: Any, model: Optional[Type[BaseModel]]) -> Any:
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
    output_model: Optional[Type[BaseModel]] = None,
    max_retries: int = 3,
    logger: Optional[logging.Logger] = None,
) -> Any:
    """Run ``runner`` until it yields a valid output or the budget is exhausted.

    On each attempt: await ``runner``, pull ``output_field`` off the result, and
    (if ``output_model`` is set) validate/coerce it. Any exception — a model
    error, a missing field, a validation failure — consumes one attempt. After
    ``max_retries`` attempts the last error is wrapped in :class:`RLMTaskError`.
    """
    log = logger or _DEFAULT_LOG
    if max_retries < 1:
        raise ValueError("max_retries must be >= 1")

    last_error: Optional[BaseException] = None
    for attempt in range(1, max_retries + 1):
        try:
            prediction = await runner()
            if not hasattr(prediction, output_field):
                raise AttributeError(
                    f"RLM prediction has no field {output_field!r}"
                )
            raw = getattr(prediction, output_field)
            return coerce_output(raw, output_model)
        except Exception as exc:  # noqa: BLE001 — we intentionally retry on anything
            last_error = exc
            log.warning(
                "RLM attempt %d/%d failed: %s", attempt, max_retries, _short_error(exc)
            )

    raise RLMTaskError(
        f"Failed to produce a valid '{output_field}' after {max_retries} attempts"
    ) from last_error
