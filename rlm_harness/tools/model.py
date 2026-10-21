"""Provider-agnostic ``make_model_tool``: the generic "model-backed tool + validate"
core (mirrors ``fetch.py`` / ``search.py``).

A model-as-tool: a SECONDARY model the RLM root calls as a tool to PRODUCE something
(YAML, code, SQL, …) which is then deterministically validated. Is a recurring shape.
The reusable mechanics are: call the model, retry only *transient* endpoint errors,
capture the answer + any thinking-mode reasoning, then run a validator on the output.

rlm-harness owns ONLY that generic core. The consuming project supplies the ``chat_fn`` (its
endpoint/model/prompt), a ``validate`` callable (its domain validator), and: around the
returned ``ModelToolResult``: its own tool name, result-message wording, and tracing
(exactly as the fetch / web_search consumers wrap their bases). The factory returns a
``call(spec) -> ModelToolResult``; it does NOT format strings or record traces.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# What produced this result: `ok=False` has THREE distinct causes and they are not
# interchangeable: see `ModelToolResult.cause`. RE-EXPORTED from `rlm_harness.trace`, which owns them,
# because a live result and a RECORDED payload must answer this question with the same four words;
# two vocabularies for one distinction is how it gets collapsed again at the trace boundary
# (`trace.payload_cause` is the read side).
from ..trace import CAUSE_CIRCUIT_BROKEN, CAUSE_ENDPOINT, CAUSE_INVALID, CAUSE_OK

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
    """Structured outcome of one model-tool call: the caller formats the user-facing reply.

    **`ok=False` has three causes, and collapsing them is a real bug, not a nuance.** The
    validator rejected the output; the endpoint failed after retries; or the breaker
    short-circuited without calling the model at all. In the last two the validator NEVER RAN.

    That distinction has been got wrong downstream more than once, in more than one consumer, in
    ways that reach both training data and user-facing text: a label named `*_rejects` whose
    docstring says "the host-side validator rejected" incremented on a 502; a reviewer-facing
    string reading "failed its format check" shown for an endpoint timeout; a per-run metric that
    counted every `ok is False` beside a separate circuit-break count, so the two overlapped.
    The information was always here (`circuit_broken`, `endpoint_error`), but it had no NAME, so
    every consumer had to re-derive it and several silently did not. `cause` and `validator_ran`
    are that name. Read one of them before writing any string or label that attributes a failure.
    """

    ok: bool                              # the validator's verdict (False on endpoint error)
    raw: str                              # the model's raw output ("" on endpoint error / circuit break)
    reasoning: str | None = None       # thinking-mode reasoning, if the chat_fn surfaced it
    errors: list[str] = field(default_factory=list)  # validator errors (or the endpoint error)
    validated: Any = None                 # the full object the validator returned
    endpoint_error: str | None = None  # set (ok=False) iff the model call failed after retries
    circuit_broken: bool = False          # True (ok=False, no model call) iff the breaker short-circuited

    @property
    def cause(self) -> str:
        """Which of the four outcomes this is: `ok` / `invalid` / `endpoint` / `circuit_broken`.

        Ordered so the checks cannot disagree with each other: a short-circuit sets neither
        `endpoint_error` nor a validator verdict, and an endpoint failure never reaches the
        validator, so `invalid` is the ONLY reading left once both are excluded.
        """
        if self.circuit_broken:
            return CAUSE_CIRCUIT_BROKEN
        if self.endpoint_error is not None:
            return CAUSE_ENDPOINT
        return CAUSE_OK if self.ok else CAUSE_INVALID

    @property
    def validator_ran(self) -> bool:
        """Whether the domain validator was actually invoked.

        The direct question behind the mislabels above: only when this is True may a caller say
        the output "failed validation" / "failed its format check" / "was rejected". When it is
        False the model produced nothing to validate, and attributing that to the model's output
        blames it for infrastructure.
        """
        return self.cause in (CAUSE_OK, CAUSE_INVALID)


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
    is_fast_fail: Callable[[BaseException], bool] | None = None,
) -> Callable[[str], ModelToolResult]:
    """Build the generic call: chat (retrying transient errors) → validate → ModelToolResult.

    ``transient_retries`` retries ONLY exceptions from ``chat_fn`` (endpoint flakiness); a
    validator that returns ``ok=False`` is NOT retried (that is the caller's repair loop, e.g.
    re-spec and call again). On exhausted retries the result has ``endpoint_error`` set and
    ``ok=False``.

    ``is_fast_fail(exc)`` says an exception will REPEAT on the same input, so retrying it pays
    twice for one doomed call: an HTTP 400/401/404 from a wrong model name or key, a reply the
    endpoint cut at a fixed budget, a context-length refusal. A match returns at once, consuming
    no further attempt, and it also counts toward ``max_consecutive_invalid`` below, because a
    call the HOST has classified as repeating is not the infra flakiness that exemption was
    written for. Without this predicate nothing changes: every exception is retried as before, so
    the widening is reached only by opting in.

    The name and the shape are `_retry.run_with_retry`'s ``is_fast_fail``, deliberately, so the
    kit has one concept for "this will repeat" rather than two. A PREDICATE rather than an
    exception type to raise or a type tuple to pass, for the reason that invariant gives: the
    judgement cannot be written as a static ``except (A, B, C):``, and ``tools/`` must stay
    dspy-free, so the caller supplies it. A consumer running on a dspy LM can pass
    ``rlm_harness._dspy_compat.is_fast_fail_lm_error``, which mirrors dspy's own classification;
    one with its own client passes its own.

    Three of the four outcomes carry ``ok=False`` and they are NOT interchangeable: read
    ``result.cause`` (or ``result.validator_ran``) before attributing a failure to the model's
    output. See ``ModelToolResult``.

    ``max_consecutive_invalid`` (default ``None`` = off) is a run-scoped CIRCUIT BREAKER: once the
    validator has returned ``ok=False``, or ``is_fast_fail`` has matched, that many times in a
    ROW, the next call SHORT-CIRCUITS.
    it does NOT invoke the model and returns ``circuit_broken=True`` (``ok=False``, empty ``raw``).
    A productive repair loop recovers within a couple of declines, so a long unbroken decline run
    means the model cannot satisfy specs of this shape; short-circuiting caps wasted model calls and
    lets the caller redirect the root LM (escalate / finalize) instead of letting it thrash. The
    counter RESETS on any validator-``ok``; an UNCLASSIFIED endpoint error still does NOT count (it
    is infra, not a content decline). This factory only FLAGS the break: the caller owns the user-facing message,
    same split as the rest. The factory is sync and side-effect-free (no tracing, no message
    templating): wrap the result in your project's tool with its own name/messages/tracing.

    The breaker state lives in this closure, so build ONE tool per run (as the consumers do) and it
    resets naturally for the next run.
    """
    retries = max(0, transient_retries)
    # One counter for both kinds the breaker now stops, because the breaker's job is capping calls
    # that cannot produce a usable result, and a host-classified repeating failure is one of those.
    consecutive_unproductive = 0

    def call_model(spec: str) -> ModelToolResult:
        """Send `spec` to the configured model and validate what comes back. Returns a result
        whose `ok` says whether the output passed validation, with `validated` holding the parsed
        artifact when it did and `errors` saying why when it did not. Never raises: a failed
        endpoint sets `endpoint_error`, and too many invalid outputs in a row set
        `circuit_broken` and stop calling the model."""
        nonlocal consecutive_unproductive
        if max_consecutive_invalid is not None and consecutive_unproductive >= max_consecutive_invalid:
            broke = (
                f"circuit breaker: {consecutive_unproductive} consecutive calls that could not "
                "produce a usable result"
            )
            return ModelToolResult(
                ok=False, raw="", reasoning=None,
                errors=[broke], validated=None, circuit_broken=True,
            )
        raw, reasoning = "", None
        for attempt in range(retries + 1):
            try:
                raw, reasoning = _split(chat_fn(spec))
                break
            except Exception as exc:
                # The predicate is the caller's code and this function promises never to raise, so
                # a predicate that itself blows up is read as "no opinion" rather than allowed to
                # escape: the retry then behaves exactly as it did before this argument existed.
                fast = False
                if is_fast_fail is not None:
                    try:
                        fast = bool(is_fast_fail(exc))
                    except Exception:
                        fast = False
                if fast or attempt >= retries:
                    if fast:
                        consecutive_unproductive += 1
                    # Either way this is an endpoint error, never a content decline, so `cause`
                    # stays `endpoint` and `validator_ran` stays False: the validator never saw a
                    # thing. What the two cases differ in is the breaker, counted above. An
                    # exhausted retry is infra flakiness and still does not count; a fast-fail is
                    # the caller saying this call repeats, which is what the breaker caps.
                    #
                    # `str(exc) or type(exc).__name__`, because `str(exc)` is the EMPTY STRING for
                    # `httpx.ConnectTimeout` / `ReadTimeout` / `ConnectError`, `TimeoutError`,
                    # `OSError` and `http.client.RemoteDisconnected`: six of the most ordinary
                    # transport failures there are, and precisely the ones where a reader most needs
                    # to be told WHAT happened. Recording `''` gave every consumer an empty message
                    # to render ("endpoint failed: ") and made the field's own truthiness a lie
                    # about whether an endpoint error occurred, which is how `payload_cause` came to
                    # misread six failure modes as content declines. The class name is always
                    # available and always says something.
                    detail = str(exc) or type(exc).__name__
                    return ModelToolResult(
                        ok=False, raw="", reasoning=None,
                        errors=[detail], validated=None, endpoint_error=detail,
                    )
        validated = validate(raw)
        ok = bool(getattr(validated, "ok", False))
        consecutive_unproductive = 0 if ok else consecutive_unproductive + 1
        return ModelToolResult(
            ok=ok,
            raw=raw,
            reasoning=reasoning,
            errors=list(getattr(validated, "errors", []) or []),
            validated=validated,
        )

    return call_model
