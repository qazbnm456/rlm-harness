"""Provider-agnostic ``make_model_tool`` — the generic "model-backed tool + validate"
core (mirrors ``fetch.py`` / ``search.py``).

A model-as-tool — a SECONDARY model the RLM root calls as a tool to PRODUCE something
(YAML, code, SQL, …) which is then deterministically validated — is a recurring shape.
The reusable mechanics are: call the model, retry only *transient* endpoint errors,
capture the answer + any thinking-mode reasoning, then run a validator on the output.

rlm-kit owns ONLY that generic core. The consuming project supplies the ``chat_fn`` (its
endpoint/model/prompt), a ``validate`` callable (its domain validator), and — around the
returned ``ModelToolResult`` — its own tool name, result-message wording, and tracing
(exactly as the fetch / web_search consumers wrap their bases). The factory returns a
``call(spec) -> ModelToolResult``; it does NOT format strings or record traces.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# A chat function maps a spec to the model's output. It may return:
#   - a plain string (the answer), or
#   - a ``(content, reasoning)`` tuple, or
#   - any object exposing ``.content`` / ``.reasoning`` attributes.
ChatFn = Callable[[str], Any]

# A validator maps the model's raw text to a result object that exposes ``.ok`` (bool) and
# ``.errors`` (list[str]). Whatever it returns is passed through verbatim as ``.validated``
# so the caller can read its domain-specific fields (e.g. a parsed id, cleaned output).
Validate = Callable[[str], Any]


@dataclass
class ModelToolResult:
    """Structured outcome of one model-tool call — the caller formats the user-facing reply."""

    ok: bool                              # the validator's verdict (False on endpoint error)
    raw: str                              # the model's raw output ("" on endpoint error / circuit break)
    reasoning: str | None = None       # thinking-mode reasoning, if the chat_fn surfaced it
    errors: list[str] = field(default_factory=list)  # validator errors (or the endpoint error)
    validated: Any = None                 # the full object the validator returned
    endpoint_error: str | None = None  # set (ok=False) iff the model call failed after retries
    circuit_broken: bool = False          # True (ok=False, no model call) iff the breaker short-circuited


def _split(out: Any) -> tuple[str, str | None]:
    """Normalise a chat_fn return into ``(content, reasoning)``."""
    if isinstance(out, str):
        return out, None
    if isinstance(out, tuple):
        return (out[0] if out else ""), (out[1] if len(out) > 1 else None)
    return getattr(out, "content", "" if out is None else str(out)), getattr(out, "reasoning", None)


def make_model_tool(
    chat_fn: ChatFn,
    validate: Validate,
    *,
    transient_retries: int = 1,
    max_consecutive_invalid: int | None = None,
) -> Callable[[str], ModelToolResult]:
    """Build the generic call: chat (retrying transient errors) → validate → ModelToolResult.

    ``transient_retries`` retries ONLY exceptions from ``chat_fn`` (endpoint flakiness); a
    validator that returns ``ok=False`` is NOT retried (that is the caller's repair loop, e.g.
    re-spec and call again). On exhausted retries the result has ``endpoint_error`` set and
    ``ok=False``.

    ``max_consecutive_invalid`` (default ``None`` = off) is a run-scoped CIRCUIT BREAKER: once the
    validator has returned ``ok=False`` that many times in a ROW, the next call SHORT-CIRCUITS —
    it does NOT invoke the model and returns ``circuit_broken=True`` (``ok=False``, empty ``raw``).
    A productive repair loop recovers within a couple of declines, so a long unbroken decline run
    means the model cannot satisfy specs of this shape; short-circuiting caps wasted model calls and
    lets the caller redirect the root LM (escalate / finalize) instead of letting it thrash. The
    counter RESETS on any validator-``ok``; an endpoint error does NOT count (it is infra, not a
    content decline). This factory only FLAGS the break — the caller owns the user-facing message,
    same split as the rest. The factory is sync and side-effect-free (no tracing, no message
    templating) — wrap the result in your project's tool with its own name/messages/tracing.

    The breaker state lives in this closure, so build ONE tool per run (as the consumers do) and it
    resets naturally for the next run.
    """
    retries = max(0, transient_retries)
    consecutive_invalid = 0

    def call(spec: str) -> ModelToolResult:
        nonlocal consecutive_invalid
        if max_consecutive_invalid is not None and consecutive_invalid >= max_consecutive_invalid:
            return ModelToolResult(
                ok=False, raw="", reasoning=None,
                errors=[f"circuit breaker: {consecutive_invalid} consecutive invalid outputs"],
                validated=None, circuit_broken=True,
            )
        raw, reasoning = "", None
        for attempt in range(retries + 1):
            try:
                raw, reasoning = _split(chat_fn(spec))
                break
            except Exception as exc:
                if attempt >= retries:
                    # endpoint error: infra flakiness, NOT a content decline → does not trip the breaker
                    return ModelToolResult(
                        ok=False, raw="", reasoning=None,
                        errors=[str(exc)], validated=None, endpoint_error=str(exc),
                    )
        validated = validate(raw)
        ok = bool(getattr(validated, "ok", False))
        consecutive_invalid = 0 if ok else consecutive_invalid + 1
        return ModelToolResult(
            ok=ok,
            raw=raw,
            reasoning=reasoning,
            errors=list(getattr(validated, "errors", []) or []),
            validated=validated,
        )

    return call
