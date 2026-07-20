"""make_harness_tool — the generic 'delegate to a downstream harness as a tool' primitive. All offline:
the transport (invoke_fn / call_endpoint) is injected, so no live child, no dspy, no Deno. Verifies it
reuses make_model_tool's retry/validate/circuit-break semantics AND carries the child-rollout link."""
import types

from rlm_kit.tools import (
    HarnessInvocation,
    HarnessToolResult,
    harness_from_endpoint,
    make_harness_tool,
)


def _V(ok, errors=None):
    """A validator result: exposes .ok / .errors, like a domain validator's return."""
    return types.SimpleNamespace(ok=ok, errors=errors or [])


def test_delegation_happy_path_carries_artifact_and_child_link():
    calls = []

    def invoke(long_text):
        calls.append(long_text)
        return HarnessInvocation(content="ARTIFACT", reasoning="thought", child_run_id="child-1",
                                 child_trace="children/child-1.jsonl", child_meta={"elapsed_s": 3})

    tool = make_harness_tool(invoke, lambda raw: _V(ok=True))
    r = tool("a very long pre-assembled context …")
    assert calls == ["a very long pre-assembled context …"]      # whole context reached the child
    assert isinstance(r, HarnessToolResult)
    assert r.ok and r.raw == "ARTIFACT" and r.reasoning == "thought"
    assert r.child_run_id == "child-1"
    assert r.child_trace == "children/child-1.jsonl"
    assert r.child_meta == {"elapsed_s": 3}


def test_transient_transport_error_is_retried():
    seq = iter([RuntimeError("spawn failed"), None])

    def invoke(long_text):
        exc = next(seq)
        if exc:
            raise exc
        return HarnessInvocation(content="OK", child_run_id="c2")

    r = make_harness_tool(invoke, lambda raw: _V(ok=True), transient_retries=1)("ctx")
    assert r.ok and r.raw == "OK" and r.child_run_id == "c2"


def test_dead_child_degrades_to_endpoint_error_not_exception():
    def invoke(long_text):
        raise RuntimeError("child crashed")

    r = make_harness_tool(invoke, lambda raw: _V(ok=True), transient_retries=1)("ctx")
    assert not r.ok and r.endpoint_error and "child crashed" in r.endpoint_error
    assert r.raw == ""
    assert r.child_run_id is None and r.child_trace is None and r.child_meta is None  # no child ran


def test_invalid_artifact_keeps_child_link_and_is_not_retried():
    n = {"count": 0}

    def invoke(long_text):
        n["count"] += 1
        return HarnessInvocation(content="draft", child_run_id="c9")

    r = make_harness_tool(invoke, lambda raw: _V(ok=False, errors=["nope"]))("ctx")
    assert not r.ok and r.errors == ["nope"]
    assert r.raw == "draft" and r.child_run_id == "c9"   # a child DID run — link preserved
    assert n["count"] == 1                               # validator-false is the repair loop, NOT retried


def test_circuit_breaker_short_circuits_without_invoking_the_child():
    n = {"count": 0}

    def invoke(long_text):
        n["count"] += 1
        return HarnessInvocation(content="junk", child_run_id="c")

    tool = make_harness_tool(invoke, lambda raw: _V(ok=False, errors=["bad"]), max_consecutive_invalid=2)
    tool("ctx")                          # two consecutive invalids arm the breaker
    tool("ctx")
    r = tool("ctx")                      # third short-circuits
    assert r.circuit_broken and not r.ok
    assert n["count"] == 2               # the child was NOT invoked on the broken call
    assert r.child_run_id is None        # no child ran → no link


def test_breaker_resets_after_a_valid_artifact():
    outcomes = iter([_V(ok=False), _V(ok=False), _V(ok=True), _V(ok=False)])

    def invoke(long_text):
        return HarnessInvocation(content="x", child_run_id="c")

    tool = make_harness_tool(invoke, lambda raw: next(outcomes), max_consecutive_invalid=3)
    assert not tool("ctx").ok           # 1 invalid
    assert not tool("ctx").ok           # 2 invalid
    assert tool("ctx").ok               # valid → resets the counter
    r = tool("ctx")                     # 1 invalid again — well under the threshold, so it still runs
    assert not r.ok and not r.circuit_broken


def test_harness_from_endpoint_binds_the_long_text_and_maps_the_reply():
    seen = {}

    def call_endpoint(long_text):        # OPAQUE transport — a stand-in for spawn/import/POST
        seen["text"] = long_text
        return {"yaml": "ARTIFACT", "run_id": "cx", "trace": "children/cx.jsonl"}

    def read_output(reply):
        return HarnessInvocation(content=reply["yaml"], child_run_id=reply["run_id"],
                                 child_trace=reply["trace"])

    invoke = harness_from_endpoint(call_endpoint, read_output=read_output)
    r = make_harness_tool(invoke, lambda raw: _V(ok=True))("LONG CONTEXT")
    assert seen["text"] == "LONG CONTEXT"                       # the whole context reached the transport
    assert r.ok and r.raw == "ARTIFACT"
    assert r.child_run_id == "cx" and r.child_trace == "children/cx.jsonl"


def test_harness_from_endpoint_propagates_a_transport_raise():
    def call_endpoint(long_text):
        raise ConnectionError("unreachable")

    invoke = harness_from_endpoint(call_endpoint, read_output=lambda x: x)
    r = make_harness_tool(invoke, lambda raw: _V(ok=True), transient_retries=0)("ctx")
    assert not r.ok and r.endpoint_error and "unreachable" in r.endpoint_error


def test_non_invocation_return_degrades_child_link_to_none():
    # a transport that returns a bare string (no child_* attributes) still yields a valid artifact —
    # the child link just defaults to None via getattr, no crash.
    r = make_harness_tool(lambda t: "BARE ARTIFACT", lambda raw: _V(ok=True))("ctx")
    assert r.ok and r.raw == "BARE ARTIFACT"
    assert r.child_run_id is None and r.child_trace is None and r.child_meta is None


def test_success_with_no_child_link_is_all_none():
    r = make_harness_tool(lambda t: HarnessInvocation(content="A"), lambda raw: _V(ok=True))("ctx")
    assert r.ok and r.raw == "A"
    assert r.child_run_id is None and r.child_trace is None and r.child_meta is None


def test_endpoint_error_after_a_prior_success_does_not_leak_a_stale_link():
    n = {"count": 0}

    def invoke(long_text):
        n["count"] += 1
        if n["count"] == 1:
            return HarnessInvocation(content="A", child_run_id="c-first")
        raise RuntimeError("child crashed")

    tool = make_harness_tool(invoke, lambda raw: _V(ok=True), transient_retries=0)
    first = tool("ctx")
    assert first.ok and first.child_run_id == "c-first"
    second = tool("ctx")                       # endpoint error AFTER a prior success
    assert not second.ok and second.endpoint_error
    assert second.child_run_id is None         # the start-of-call clear prevents a stale-link leak
