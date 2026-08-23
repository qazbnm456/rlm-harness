"""Example: IN-PROCESS harness delegation — no subprocess, no HTTP.

``make_harness_tool`` + ``harness_from_endpoint`` (``tools/harness.py``) are transport-agnostic by
design: the kit ships no ``call_endpoint`` and never will (see their docstrings — "the kit ships
NONE and names NONE"). The two usual transports are a subprocess speaking ``serve_harness``'s wire
(``examples/harness_serve.py``) or an HTTP call to a hosted harness. This example shows a THIRD,
lighter-weight option a consumer can build themselves with two small kit primitives:
``rlm_harness.tools.run_isolated`` (bridge the sync tool call into the child's ``async arun()``)
and ``rlm_harness.tools.pointer_to_invocation`` (the canonical ``HarnessPointer`` ->
``HarnessInvocation`` mapping, reused unchanged from the subprocess/HTTP case).

**When to prefer this over subprocess/HTTP:** the child harness is TRUSTED, lives in the same
process/deployment, and low latency matters more than OS-level isolation — a build-vs-test-style
child delegated dozens of times per parent run, say. **When to keep subprocess/HTTP instead:** you
need real process/OS isolation (an untrusted or crash-prone child), a different runtime/language on
the child side, or the child genuinely runs on a remote machine. An in-process call shares the
parent's Python process, memory, and model credentials with the child — nothing here adds a NEW
code-execution surface (the child still enforces its own ``RLMConfig``/sandbox guard on its own REPL
code, same as always), but it does mean a wedged or resource-hungry child shares the parent's process.

Illustrative — needs real model creds and a sandbox, so it is NOT imported by the test suite.
(``tests/test_harness_tool.py::test_in_process_transport_wiring`` exercises the SAME composition
offline, with a stub child instead of a real ``dspy.RLM``, so the wiring itself stays covered by CI.)
"""

from __future__ import annotations

import asyncio
import types

from pydantic import BaseModel, Field

from rlm_harness import RLMConfig, RLMTask, TraceRecorder, configure
from rlm_harness.serving import HarnessPointer
from rlm_harness.tools import (
    harness_from_endpoint,
    make_harness_tool,
    pointer_to_invocation,
    run_isolated,
)


class SubFinding(BaseModel):
    summary: str = Field(..., description="one-paragraph summary of the sub-problem")


class SubTask(RLMTask):
    """The downstream harness being delegated to — an ordinary ``RLMTask``, nothing special about
    it. Any ``RLMTask`` subclass works here; there is nothing "in-process-transport-aware" about
    the child side at all."""

    signature = "context: str -> finding: SubFinding"
    output_field = "finding"
    output_model = SubFinding
    instructions = "Read the context and summarize the key finding in one paragraph."


def make_in_process_call_endpoint(run_id_prefix: str = "child"):
    """Build a ``call_endpoint`` for ``harness_from_endpoint`` that runs ``SubTask`` IN-PROCESS.

    Returns a plain ``HarnessPointer``, built directly from ``SubTask``'s own known output shape —
    NOT via ``serving._default_to_pointer`` (that helper is a private internal reserved for
    ``serve_harness``'s zero-config CLI path; reaching into it here would violate the kit's own
    "never reach into a `_private` name" rule, and it buys nothing once the child's shape is
    already known, as it is here).
    """
    counter = {"n": 0}

    def call_endpoint(long_text: str) -> HarnessPointer:
        counter["n"] += 1
        child_run_id = f"{run_id_prefix}-{counter['n']}"
        trace_path = f"traces/{child_run_id}.jsonl"

        async def _run() -> HarnessPointer:
            # The child's OWN TraceRecorder is entered HERE — inside the coroutine `run_isolated`
            # runs on its dedicated thread — never around the `run_isolated()` call below. A fresh
            # `threading.Thread` starts with an empty `contextvars.Context` (see `run_isolated`'s
            # own docstring), so a recorder entered outside would be invisible to
            # `current_recorder()` in here and the child's own tool_calls/sub_calls would go
            # silently unrecorded. This also matches "the child owns its own separate rollout,
            # exported independently" from the delegation docs.
            with TraceRecorder(trace_path, run_id=child_run_id):
                finding = await SubTask().arun(context=long_text)
            return HarnessPointer(
                artifact=finding.summary, run_id=child_run_id, trace_path=trace_path
            )

        # `call_endpoint` is a plain sync function (the RLM tool contract); `run_isolated` bridges
        # into the child's async `arun()` regardless of whether THIS thread already has a running
        # loop — which it will, whenever the parent task itself is mid-`arun()`.
        return run_isolated(_run)

    return call_endpoint


def _non_empty(raw: str):
    return types.SimpleNamespace(ok=bool(raw.strip()), errors=[] if raw.strip() else ["empty artifact"])


def build_delegation_tool():
    """Compose the in-process transport into a ready-to-use ``tools=`` entry for a parent task."""
    invoke = harness_from_endpoint(
        make_in_process_call_endpoint(), read_output=pointer_to_invocation
    )
    return make_harness_tool(invoke, validate=_non_empty)


async def main() -> None:
    configure(RLMConfig.from_env())
    delegate_to_sub_task = build_delegation_tool()

    # A parent task would normally pass `tools=[delegate_to_sub_task]` and let the ROOT LM decide,
    # mid-REPL, whether to delegate — landing the decision in the trajectory as a `tool_call`,
    # exactly like the subprocess/HTTP transports. Called directly here for a minimal, runnable demo.
    result = delegate_to_sub_task("... a long pre-assembled context for the child to read ...")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
