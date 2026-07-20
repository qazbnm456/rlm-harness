"""Provider-agnostic ``make_harness_tool`` — delegate a sub-task to ANOTHER rlm-kit harness, wrapped
as a tool (the promoted "wrap a downstream harness as a tool" shape; mirrors ``model.py``).

A *harness* is a full RLM in its own right: it takes a long-text input, runs its own Root LM in a REPL
loop over that text with its OWN tools (MCP / skills / fetch), and SUBMITs a validated artifact. When a
task wants to hand a hard sub-problem to a more specialized harness, that delegation is a recurring
shape — and mechanically it is IDENTICAL to a model-as-tool call (a ``Callable[[str], Any]`` in, a
domain validator on the artifact out, degrade on failure). So this module REUSES ``make_model_tool``'s
retry → validate → circuit-break core (kept in one place, per the kit's hardening rule) and adds only
the one thing a harness has that a model does not: a **child-rollout link** (the child ran its own
trajectory; the parent records a pointer to it, never the child's turns).

THE LONG-TEXT-ENVIRONMENT CONTRACT (the reason this exists). The native advantage of the RLM framework
is that a signature input field holds near-unbounded text that dspy injects as the Root LM's REPL
ENVIRONMENT — a variable it reads/slices and loops over with its own tools. ``make_harness_tool`` makes
that the DEFAULT, enforced by SHAPE: a :data:`HarnessInvoke` takes ONE long-text argument and nothing
else, so the only thing a caller can hand across the boundary is its pre-assembled, full context — which
:func:`harness_from_endpoint` binds to the downstream harness's long-text input field (the field dspy
injects as the child's REPL environment). The point is not "call a sub-model"; it is "hand a big context
to a harness that runs a full RLM loop over it."

BASE/WRAP split (like the rest of ``tools/``). rlm-kit owns ONLY the generic core + the long-text-env
adapter. The consuming project supplies the ``call_endpoint`` (HOW to reach ITS harness — a subprocess
command, an in-process entry, an HTTP URL; the kit ships NONE and names NONE, exactly as
``make_command_tool`` demands an injected ``Runner`` and ships no executor), a ``validate`` callable, and
— around the returned :class:`HarnessToolResult` — its own tool name, messages, and tracing. The kit
never imports or names any specific downstream harness; the harness's identity lives only in the
consumer's runtime config. dspy-free (it only reuses ``make_model_tool``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from .model import ModelToolResult, Validate, make_model_tool

# A harness invocation maps ONE long text — the child Root LM's REPL environment — to the child's
# outcome. It may RAISE on a transient transport failure (spawn/connect flakiness); that is retried,
# exactly like a ``ChatFn``. A validator that returns ``ok=False`` is the caller's repair loop, not a
# transient error.
HarnessInvoke = Callable[[str], Any]


@dataclass
class HarnessInvocation:
    """What an adapter's ``invoke_fn`` returns for one delegation.

    ``content`` / ``reasoning`` are read by ``make_model_tool``'s normaliser (so a harness invocation
    slots into the reused core unchanged); the ``child_*`` fields are the delegation-boundary LINK —
    the child harness ran its OWN trajectory, and the caller records a pointer to it, never the child's
    turns."""

    content: str                          # the child harness's final artifact TEXT (what the validator sees)
    reasoning: Optional[str] = None       # the child's thinking, if the transport surfaced it
    child_run_id: Optional[str] = None    # the child rollout's own run_id
    child_trace: Optional[str] = None     # path / URI to the child's OWN trace (never inlined here)
    child_meta: Optional[dict] = None     # generic, e.g. {"elapsed_s": …, "child_steps": …}


@dataclass
class HarnessToolResult(ModelToolResult):
    """:class:`ModelToolResult` (ok / raw / reasoning / errors / validated / endpoint_error /
    circuit_broken) PLUS the child-rollout link, so the caller can record a parent→child pointer
    without re-parsing (or inlining) the child's trace. On endpoint error / circuit break no child
    ran, so the ``child_*`` fields are ``None``."""

    child_run_id: Optional[str] = None
    child_trace: Optional[str] = None
    child_meta: Optional[dict] = None


def make_harness_tool(
    invoke_fn: HarnessInvoke,
    validate: Validate,
    *,
    transient_retries: int = 1,
    max_consecutive_invalid: Optional[int] = None,
) -> Callable[[str], HarnessToolResult]:
    """Build the generic delegation call: invoke the child harness on one long text (retrying transient
    transport errors) → validate its artifact → circuit-break → :class:`HarnessToolResult`.

    Semantics are ``make_model_tool``'s verbatim (this composes it): ``transient_retries`` retries only
    exceptions from ``invoke_fn`` (a dead/slow child is infra flakiness); a validator ``ok=False`` is
    NOT retried (that is the caller's re-spec / escalate / finalize loop); ``max_consecutive_invalid``
    (default off) is a run-scoped circuit breaker that short-circuits after that many consecutive
    invalid artifacts WITHOUT invoking the child. So a hung, crashing, or looping child degrades to
    ``endpoint_error`` / ``circuit_broken`` (``ok=False``) — never an exception — and the parent run
    completes. Sync and side-effect-free (no tracing, no messages): wrap the result in your project's
    tool with its own name / messages / tracing, and record the parent→child link from the returned
    ``child_*`` fields. Breaker state lives in the closure, so build ONE per run (as the consumers do)
    and it resets for the next run."""
    # The child_* link travels out-of-band from make_model_tool (which only knows content/reasoning):
    # the inner chat stashes the last invocation's link in a single-slot holder, and `call` reads it
    # after the reused core returns. RLM tools are sync and one-call-at-a-time, so the slot is race-free.
    held: dict = {}

    def _chat(source: str) -> HarnessInvocation:
        out = invoke_fn(source)  # may raise on transient transport error → make_model_tool retries
        held.clear()
        held.update(
            child_run_id=getattr(out, "child_run_id", None),
            child_trace=getattr(out, "child_trace", None),
            child_meta=getattr(out, "child_meta", None),
        )
        return out  # make_model_tool reads .content / .reasoning off it

    base = make_model_tool(
        _chat, validate,
        transient_retries=transient_retries,
        max_consecutive_invalid=max_consecutive_invalid,
    )

    def call(source: str) -> HarnessToolResult:
        held.clear()  # cleared → child_* stays None on endpoint-error / circuit-break (no child ran)
        r = base(source)
        return HarnessToolResult(
            ok=r.ok, raw=r.raw, reasoning=r.reasoning, errors=r.errors, validated=r.validated,
            endpoint_error=r.endpoint_error, circuit_broken=r.circuit_broken,
            child_run_id=held.get("child_run_id"),
            child_trace=held.get("child_trace"),
            child_meta=held.get("child_meta"),
        )

    return call


def harness_from_endpoint(
    call_endpoint: Callable[[str], Any],
    *,
    read_output: Callable[[Any], HarnessInvocation],
) -> HarnessInvoke:
    """Bake the long-text-environment contract over ANY transport, returning an ``invoke_fn`` for
    :func:`make_harness_tool`.

    ``call_endpoint(long_text)`` MUST run the downstream harness with ``long_text`` bound to its
    long-text INPUT field — the field dspy injects as the child Root LM's REPL environment variable — so
    the child fully exploits its own RLM loop (REPL + its own MCP / skills / fetch) over the whole
    context. ``read_output`` maps the transport's raw reply into a :class:`HarnessInvocation`
    (extracting the artifact text + the child's run_id / trace pointer). The kit picks NO transport and
    names NO harness: ``call_endpoint`` is OPAQUE and consumer-supplied — a subprocess spawn, an
    in-process entry, an HTTP POST — exactly as ``make_command_tool`` takes an injected ``Runner`` and
    ships no executor. A transport failure should RAISE (so :func:`make_harness_tool` retries/degrades
    it); do not swallow it into an empty artifact."""
    def invoke(long_text: str) -> HarnessInvocation:
        return read_output(call_endpoint(long_text))

    return invoke
