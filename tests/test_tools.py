import inspect

import pytest
from pydantic import BaseModel

from rlm_kit.optimize import exact_field_metric, schema_valid_metric
import rlm_kit.tools.fetch as fetch_mod
from rlm_kit.tools.fetch import (
    _ip_blocked,
    is_safe_url,
    make_fetch_tool,
    parse_cidrs,
    resolved_host_is_safe,
)
from rlm_kit.tools.model import ModelToolResult, make_model_tool
from rlm_kit.tools.search import make_web_search_tool, normalise_search_results
from rlm_kit.tools.validation import make_json_schema_validator, make_schema_validator
from rlm_kit.trace import EVENT_TOOL_CALL, TraceRecorder, load_events


# ---- make_model_tool (generic model-call + retry + validate core) --------

class _V:  # a minimal validator result (duck-typed: .ok / .errors / domain fields)
    def __init__(self, ok, errors=(), parsed=None):
        self.ok, self.errors, self.parsed = ok, list(errors), parsed


def test_model_tool_validates_and_passes_result_through():
    call = make_model_tool(lambda spec: "OUT:" + spec,
                           lambda raw: _V(ok=True, parsed=raw))
    r = call("x")
    assert isinstance(r, ModelToolResult)
    assert r.ok is True and r.raw == "OUT:x" and r.errors == []
    assert r.validated.parsed == "OUT:x"        # the validator object is passed through verbatim
    assert r.endpoint_error is None


def test_model_tool_surfaces_validator_failure_without_retry():
    calls = {"n": 0}
    def chat(spec):
        calls["n"] += 1
        return "bad"
    call = make_model_tool(chat, lambda raw: _V(ok=False, errors=["nope"]), transient_retries=2)
    r = call("x")
    assert r.ok is False and r.errors == ["nope"]
    assert calls["n"] == 1                        # a validator FAIL is not retried (caller's repair loop)


def test_model_tool_retries_transient_then_succeeds():
    calls = {"n": 0}
    def flaky(spec):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("boom")
        return "ok-now"
    call = make_model_tool(flaky, lambda raw: _V(ok=True), transient_retries=1)
    r = call("x")
    assert r.ok is True and r.raw == "ok-now" and calls["n"] == 2


def test_model_tool_endpoint_error_after_exhausted_retries():
    def always_fail(spec):
        raise TimeoutError("down")
    call = make_model_tool(always_fail, lambda raw: _V(ok=True), transient_retries=1)
    r = call("x")
    assert r.ok is False and r.raw == "" and r.endpoint_error == "down"
    assert "down" in r.errors[0]


def test_model_tool_splits_reasoning_from_tuple_and_object():
    # (content, reasoning) tuple
    r1 = make_model_tool(lambda s: ("ans", "thought"), lambda raw: _V(ok=True))("x")
    assert r1.raw == "ans" and r1.reasoning == "thought"
    # an object exposing .content / .reasoning
    class _O:
        content, reasoning = "obj-ans", "obj-thought"
    r2 = make_model_tool(lambda s: _O(), lambda raw: _V(ok=True))("x")
    assert r2.raw == "obj-ans" and r2.reasoning == "obj-thought"


def test_model_tool_circuit_breaker_trips_after_consecutive_declines():
    # After max_consecutive_invalid declines in a row, the next call SHORT-CIRCUITS: no model call,
    # circuit_broken=True. Caps wasted calls when the model can't satisfy specs of this shape.
    calls = {"n": 0}
    def chat(spec):
        calls["n"] += 1
        return "bad"
    call = make_model_tool(chat, lambda raw: _V(ok=False, errors=["nope"]),
                           max_consecutive_invalid=3)
    for _ in range(3):
        assert call("x").ok is False           # 3 real declines (model called)
    assert calls["n"] == 3
    r = call("x")                              # 4th call trips the breaker
    assert r.circuit_broken is True and r.ok is False and r.raw == ""
    assert calls["n"] == 3                     # the model was NOT called on the broken call
    assert call("x").circuit_broken is True    # stays broken (no model call) until reset
    assert calls["n"] == 3


def test_model_tool_circuit_breaker_resets_on_ok():
    # A validator-ok resets the streak, so interleaved declines never trip — only an UNBROKEN run does.
    seq = iter([False, False, True, False, False])  # max consecutive declines = 2
    call = make_model_tool(lambda s: "x", lambda raw: _V(ok=next(seq)),
                           max_consecutive_invalid=3)
    results = [call("x") for _ in range(5)]
    assert not any(r.circuit_broken for r in results)   # streak never reached 3


def test_model_tool_endpoint_error_does_not_trip_breaker():
    # An endpoint error is infra flakiness, not a content decline — it must not advance the breaker.
    state = {"fail": True}
    def chat(spec):
        if state["fail"]:
            raise TimeoutError("down")
        return "ok"
    call = make_model_tool(chat, lambda raw: _V(ok=True),
                           transient_retries=0, max_consecutive_invalid=2)
    for _ in range(5):
        r = call("x")
        assert r.endpoint_error == "down" and r.circuit_broken is False  # never trips on infra errors


def test_model_tool_circuit_breaker_off_by_default():
    call = make_model_tool(lambda s: "x", lambda raw: _V(ok=False), )  # no max_consecutive_invalid
    for _ in range(10):
        assert call("x").circuit_broken is False   # default None = breaker disabled


class Finding(BaseModel):
    title: str
    severity: str


# ---- validation tool -----------------------------------------------------

def test_schema_validator_accepts_valid_json():
    v = make_schema_validator(Finding)
    assert v.__name__ == "validate_finding"
    assert "successful" in v('{"title": "t", "severity": "high"}').lower()


def test_schema_validator_reports_failure():
    v = make_schema_validator(Finding)
    assert "failed" in v('{"title": "t"}').lower()


# ---- JSON-schema validator (make_json_schema_validator) -------------------

_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "severity": {"enum": ["info", "low", "medium", "high", "critical"]},
    },
    "required": ["id", "severity"],
}


def test_json_schema_validator_passes_a_valid_object():
    v = make_json_schema_validator(_JSON_SCHEMA)
    assert v({"id": "x", "severity": "high"}) == []          # [] == valid


def test_json_schema_validator_reports_each_violation_with_a_path():
    v = make_json_schema_validator(_JSON_SCHEMA)
    errs = v({"severity": "spicy"})                          # missing id + bad enum
    assert any("id" in e for e in errs)                      # required-field violation
    assert any("severity" in e and "spicy" in e for e in errs)  # located on the bad field


def test_json_schema_validator_loads_schema_from_a_path(tmp_path):
    import json as _json
    p = tmp_path / "schema.json"
    p.write_text(_json.dumps(_JSON_SCHEMA))
    v = make_json_schema_validator(str(p))
    assert v({"id": "x", "severity": "high"}) == []
    assert v({"id": 1, "severity": "high"})                  # id wrong type → non-empty


def test_json_schema_validator_truncates_a_flood_of_errors():
    schema = {"type": "object", "additionalProperties": {"type": "string"}}
    v = make_json_schema_validator(schema, max_errors=3)
    errs = v({f"k{i}": i for i in range(50)})                # 50 int values → 50 violations
    assert len(errs) == 4 and "truncated" in errs[-1]        # 3 + the truncation marker


# ---- SSRF guard ----------------------------------------------------------

@pytest.mark.parametrize(
    "url",
    [
        "https://nvd.nist.gov/vuln/detail/CVE-2024-0001",
        "http://example.com/advisory",
    ],
)
def test_safe_urls_allowed(url):
    assert is_safe_url(url) is True


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/admin",
        "http://127.0.0.1:8000/",
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata
        "http://10.0.0.5/internal",
        "http://192.168.1.1/",
        "http://[::1]/",
        "https://service.internal/secret",
        "https://printer.local/",
        "file:///etc/passwd",
        "ftp://example.com/x",
        "",
        "not a url",
    ],
)
def test_unsafe_urls_blocked(url):
    assert is_safe_url(url) is False


# ---- resolved-IP re-check (DNS-rebinding defence) + allow-list -----------

def test_parse_cidrs_skips_bad_entries():
    nets = parse_cidrs(("198.18.0.0/16", "garbage", "10.0.0.0/8"))
    assert len(nets) == 2                                    # the bad entry is dropped, not fatal
    assert parse_cidrs(()) == () and parse_cidrs(None) == ()


def test_ip_blocked_default_allow_list_and_fail_closed():
    assert _ip_blocked("8.8.8.8") is False                  # public → allowed
    assert _ip_blocked("198.18.2.128") is True              # reserved (benchmarking) → blocked
    assert _ip_blocked("not-an-ip") is True                 # unparseable → fail closed
    nets = parse_cidrs(("198.18.0.0/16",))
    assert _ip_blocked("198.18.2.128", nets) is False       # allow-listed proxy range → external
    assert _ip_blocked("127.0.0.1", nets) is True           # loopback still blocked
    assert _ip_blocked("169.254.169.254", nets) is True     # cloud metadata still blocked


def test_resolved_host_is_safe_honors_allow_list(monkeypatch):
    # A fake-IP proxy resolves every host into 198.18.x.x; refused by default, allowed when listed.
    monkeypatch.setattr(fetch_mod.socket, "getaddrinfo",
                        lambda host, port, *a, **k: [(2, 1, 6, "", ("198.18.2.128", port))])
    assert resolved_host_is_safe("plugins.svn.example", 443) is False
    assert resolved_host_is_safe("plugins.svn.example", 443,
                                 allow_nets=parse_cidrs(("198.18.0.0/16",))) is True
    # a host that resolves to a REAL public address is safe without any allow-list
    monkeypatch.setattr(fetch_mod.socket, "getaddrinfo",
                        lambda host, port, *a, **k: [(2, 1, 6, "", ("93.184.216.34", port))])
    assert resolved_host_is_safe("example.com", 443) is True


def test_resolved_host_is_safe_fails_closed_on_resolution_error(monkeypatch):
    def boom(*a, **k):
        raise OSError("no DNS")

    monkeypatch.setattr(fetch_mod.socket, "getaddrinfo", boom)
    assert resolved_host_is_safe("x", 443, allow_nets=parse_cidrs(("198.18.0.0/16",))) is False
    assert resolved_host_is_safe("", 443) is False          # empty host → refused


def test_fetch_tool_blocks_before_calling_fetcher():
    called = {"n": 0}

    def fetcher(url):
        called["n"] += 1
        return "content"

    tool = make_fetch_tool(fetcher)
    out = tool("http://169.254.169.254/")
    assert "Refused" in out
    assert called["n"] == 0


def test_fetch_tool_allows_safe_url():
    tool = make_fetch_tool(lambda url: f"fetched {url}")
    assert tool("https://example.com/x") == "fetched https://example.com/x"


def test_fetch_tool_is_sync_not_coroutine():
    # dspy.RLM invokes tools synchronously; the tool must NOT be a coroutine function
    # (an async tool would serialise to a coroutine repr in the sandbox and never run).
    tool = make_fetch_tool(lambda url: "body")
    assert not inspect.iscoroutinefunction(tool)
    assert isinstance(tool("https://example.com/x"), str)


def test_fetch_tool_records_size_and_status_not_body(tmp_path):
    # The fetched body lands in a REPL variable; the trace records only ok + size.
    tool = make_fetch_tool(lambda url: "x" * 1234)
    path = str(tmp_path / "t.jsonl")
    with TraceRecorder(path, run_id="r1"):
        tool("https://example.com/big")
    tc = [e for e in load_events(path) if e["type"] == EVENT_TOOL_CALL][0]
    assert tc["payload"]["ok"] is True
    assert tc["payload"]["result_len"] == 1234
    assert "result" not in tc["payload"]          # the body is NOT recorded


def test_fetch_tool_refusal_records_not_ok(tmp_path):
    tool = make_fetch_tool(lambda url: "never called")
    path = str(tmp_path / "t.jsonl")
    with TraceRecorder(path, run_id="r1"):
        out = tool("http://169.254.169.254/")
    assert "Refused" in out
    tc = [e for e in load_events(path) if e["type"] == EVENT_TOOL_CALL][0]
    assert tc["payload"]["ok"] is False


def test_fetch_tool_catches_fetcher_error_as_text(tmp_path):
    def boom(url):
        raise ValueError("connreset")

    tool = make_fetch_tool(boom)
    path = str(tmp_path / "t.jsonl")
    with TraceRecorder(path, run_id="r1"):
        out = tool("https://example.com/x")        # returns a string, does not raise
    assert "Fetch error" in out and "ValueError" in out
    tc = [e for e in load_events(path) if e["type"] == EVENT_TOOL_CALL][0]
    assert tc["payload"]["ok"] is False
    assert "error: ValueError" in tc["payload"]["note"]


# ---- web_search building blocks ------------------------------------------

def test_normalise_search_results_caps_filters_uniform():
    raw = [
        {"title": "A", "url": "https://example.com/a", "snippet": "s1"},
        {"title": "no url"},                                   # dropped: no url
        {"title": "meta", "url": "http://169.254.169.254/x"},  # dropped: internal
        "not a dict",                                          # dropped: not a dict
        {"url": "https://example.com/b"},                      # kept (empty title/snippet)
        {"url": "https://example.com/c"},
        {"url": "https://example.com/d"},                      # capped out (max 3)
    ]
    out = normalise_search_results(raw, max_results=3)
    assert len(out) == 3
    assert out[0] == {"title": "A", "url": "https://example.com/a", "snippet": "s1"}
    assert all(set(r) == {"title", "url", "snippet"} for r in out)
    assert all("169.254" not in r["url"] for r in out)


def test_normalise_keeps_internal_when_guard_disabled():
    raw = [{"url": "http://169.254.169.254/x"}]
    assert normalise_search_results(raw) == []                 # default drops the SSRF target
    assert len(normalise_search_results(raw, drop_unsafe_urls=False)) == 1


def test_web_search_tool_trims_query_and_normalises():
    def searcher(q):
        assert q == "cve-2026-1"                               # trimmed before the provider
        return [{"title": "t", "url": "https://x.example/p", "snippet": "sn"}]
    tool = make_web_search_tool(searcher, max_results=5)
    assert tool("  cve-2026-1 ") == [
        {"title": "t", "url": "https://x.example/p", "snippet": "sn"}]
    assert tool("   ") == "Refused: empty search query."       # empty query → no provider call


def test_web_search_tool_is_sync_not_coroutine():
    # dspy.RLM invokes tools synchronously; the tool must NOT be a coroutine function.
    tool = make_web_search_tool(lambda q: [{"url": "https://x.example/a"}])
    assert not inspect.iscoroutinefunction(tool)
    assert tool("q")[0]["url"] == "https://x.example/a"


def test_web_search_tool_records_empty_query_as_not_ok(tmp_path):
    tool = make_web_search_tool(lambda q: [{"url": "https://x.example/a"}])
    path = str(tmp_path / "t.jsonl")
    with TraceRecorder(path, run_id="r1"):
        assert tool("   ") == "Refused: empty search query."     # reactable string, not []
    tc = [e for e in load_events(path) if e["type"] == EVENT_TOOL_CALL][0]
    assert tc["payload"]["ok"] is False                           # degenerate input, not a success
    assert tc["payload"]["note"] == "empty query"


def test_web_search_tool_catches_searcher_error_as_text(tmp_path):
    def boom(q):
        raise RuntimeError("provider down")

    tool = make_web_search_tool(boom)
    path = str(tmp_path / "t.jsonl")
    with TraceRecorder(path, run_id="r1"):
        out = tool("cve-2026-1")                       # returns a string, does not raise
    assert isinstance(out, str) and "Search error" in out and "RuntimeError" in out
    tc = [e for e in load_events(path) if e["type"] == EVENT_TOOL_CALL][0]
    assert tc["payload"]["ok"] is False
    assert "error: RuntimeError" in tc["payload"]["note"]


# ---- optimize metric templates ------------------------------------------

def _ns(**kw):
    import types

    return types.SimpleNamespace(**kw)


def test_exact_field_metric():
    metric = exact_field_metric("label")
    assert metric(_ns(label="rce"), _ns(label="rce")) == 1.0
    assert metric(_ns(label="rce"), _ns(label="xss")) == 0.0
    assert metric(_ns(label=None), _ns(label=None)) == 0.0


def test_schema_valid_metric():
    metric = schema_valid_metric(Finding, "finding")
    assert metric(None, _ns(finding={"title": "t", "severity": "x"})) == 1.0
    assert metric(None, _ns(finding={"title": "t"})) == 0.0
    assert metric(None, _ns(finding=None)) == 0.0
