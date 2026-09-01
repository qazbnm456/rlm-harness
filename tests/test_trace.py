import json
import threading
import types

import pytest

from rlm_harness.trace import (
    CAUSE_CIRCUIT_BROKEN,
    CAUSE_ENDPOINT,
    CAUSE_INVALID,
    CAUSE_OK,
    EVENT_FINAL,
    EVENT_MAIN_STEP,
    EVENT_RESULT,
    EVENT_RUN_END,
    EVENT_RUN_START,
    EVENT_TOOL_CALL,
    TraceRecorder,
    current_recorder,
    group_by_run,
    load_events,
    payload_cause,
    record_tool_call,
)


def _counter():
    n = {"v": 0.0}

    def clock():
        n["v"] += 1.0
        return n["v"]

    return clock


def test_recorder_writes_jsonl_with_monotonic_steps(tmp_path):
    path = str(tmp_path / "trace.jsonl")
    with TraceRecorder(path, run_id="r1", clock=_counter()) as rec:
        assert current_recorder() is rec
        rec.record("sub_call", {"x": 1})
        rec.record("tool_call", {"y": 2})
    assert current_recorder() is None  # reset on exit

    events = load_events(path)
    types_seen = [e["type"] for e in events]
    assert types_seen == [EVENT_RUN_START, "sub_call", "tool_call", EVENT_RUN_END]
    assert [e["step_id"] for e in events] == [0, 1, 2, 3]
    assert all(e["run_id"] == "r1" for e in events)
    assert all(e["schema"] == "rlm-harness/trace/v1" for e in events)


def test_on_event_observer_fires_live_for_every_event(tmp_path):
    # The live observer gets each event AS it is recorded (run_start, the calls, run_end) — what
    # a streaming UI uses to stream sandbox-invoked tool_calls that dspy's on_tool never sees.
    seen = []
    path = str(tmp_path / "trace.jsonl")
    with TraceRecorder(path, run_id="r1", clock=_counter(), on_event=seen.append) as rec:
        rec.record("tool_call", {"tool": "fetch_url"})
        rec.record("sub_call", {"x": 1})
    types = [e["type"] for e in seen]
    assert types == [EVENT_RUN_START, "tool_call", "sub_call", EVENT_RUN_END]   # live + in order
    assert seen == load_events(path)                                            # same events as the file


def test_on_event_observer_error_never_breaks_the_trace(tmp_path):
    path = str(tmp_path / "trace.jsonl")
    def boom(_):
        raise RuntimeError("observer blew up")
    with TraceRecorder(path, run_id="r1", clock=_counter(), on_event=boom) as rec:
        rec.record("tool_call", {"tool": "x"})                                  # must not raise
    assert [e["type"] for e in load_events(path)] == [EVENT_RUN_START, "tool_call", EVENT_RUN_END]


def test_run_end_records_error(tmp_path):
    path = str(tmp_path / "trace.jsonl")
    try:
        with TraceRecorder(path, run_id="r1", clock=_counter()):
            raise ValueError("boom")
    except ValueError:
        pass
    end = [e for e in load_events(path) if e["type"] == EVENT_RUN_END][0]
    assert end["payload"]["ok"] is False
    assert "boom" in end["payload"]["error"]


def test_record_main_trajectory_from_fake_prediction(tmp_path):
    path = str(tmp_path / "trace.jsonl")
    pred = types.SimpleNamespace(
        trajectory=[
            {"reasoning": "think", "code": "print(1)", "output": "1"},
            {"reasoning": "more", "code": "print(2)", "output": "2"},
        ],
        final_reasoning="done",
    )
    with TraceRecorder(path, run_id="r1", clock=_counter()) as rec:
        rec.record_main_trajectory(pred)
        rec.record_result({"answer": 42})

    events = load_events(path)
    main = [e for e in events if e["type"] == EVENT_MAIN_STEP]
    assert len(main) == 2
    assert main[0]["payload"]["turn"] == 0
    assert main[1]["payload"]["code"] == "print(2)"
    final = [e for e in events if e["type"] == EVENT_FINAL][0]
    assert final["payload"]["final_reasoning"] == "done"
    result = [e for e in events if e["type"] == EVENT_RESULT][0]
    assert result["payload"]["output"] == {"answer": 42}


def test_main_step_ts_backfilled_from_live_capture(tmp_path):
    # The live per-turn stamps (captured while the run was in flight) override the post-hoc clock,
    # matched to the trajectory by reasoning — so a main_step's ts is WHEN it happened, not finalize.
    path = str(tmp_path / "trace.jsonl")
    pred = types.SimpleNamespace(
        trajectory=[
            {"reasoning": "think", "code": "c0", "output": "o0"},
            {"reasoning": "more", "code": "c1", "output": "o1"},
        ],
        final_reasoning="done",
    )
    with TraceRecorder(path, run_id="r1", clock=_counter()) as rec:
        rec.begin_main_capture()
        rec.note_main_step("think", ts=100.5)   # live, as turn 0 was parsed
        rec.note_main_step("more", ts=120.0)    # live, as turn 1 was parsed
        rec.record_main_trajectory(pred)
    main = [e for e in load_events(path) if e["type"] == EVENT_MAIN_STEP]
    assert [e["payload"]["turn"] for e in main] == [0, 1]
    assert [e["ts"] for e in main] == [100.5, 120.0]    # live ts, NOT the _counter() fallback


def test_main_step_ts_falls_back_to_clock_without_capture(tmp_path):
    # No live capture (replay, or no callback wired) → ts is clock() exactly as before.
    path = str(tmp_path / "trace.jsonl")
    pred = types.SimpleNamespace(
        trajectory=[{"reasoning": "a", "code": "c", "output": "o"}], final_reasoning=None)
    with TraceRecorder(path, run_id="r1", clock=_counter()) as rec:  # run_start consumes ts=1.0
        rec.record_main_trajectory(pred)
    main = [e for e in load_events(path) if e["type"] == EVENT_MAIN_STEP][0]
    assert main["ts"] == 2.0    # clock-driven fallback, unchanged behavior


def test_main_step_double_parse_resolves_to_first_stamp(tmp_path):
    # Two stamps staged for one turn → consume the EARLIEST. This USED to be the production
    # shape: the kit's own `_LenientJSONAdapter.parse` calls `super().parse(...)` and dspy wraps
    # `parse` per defining class, so each turn fired the callback twice. `_MainStepTimer` dedupes
    # that at source since 1.6.1 — NOT dspy's behaviour, ours. The recorder-level rule is kept and
    # pinned anyway: any caller staging two stamps for one turn still gets the true (first) time.
    path = str(tmp_path / "trace.jsonl")
    pred = types.SimpleNamespace(
        trajectory=[{"reasoning": "r", "code": "c", "output": "o"}], final_reasoning=None)
    with TraceRecorder(path, run_id="r1", clock=_counter()) as rec:
        rec.begin_main_capture()
        rec.note_main_step("r", ts=10.0)
        rec.note_main_step("r", ts=10.2)   # the duplicate fire
        rec.record_main_trajectory(pred)
    main = [e for e in load_events(path) if e["type"] == EVENT_MAIN_STEP][0]
    assert main["ts"] == 10.0


def test_begin_main_capture_resets_between_attempts(tmp_path):
    # A retry re-runs the RLM; only the final attempt is recorded, so a stamp from a prior attempt
    # must not leak into the recorded trajectory.
    path = str(tmp_path / "trace.jsonl")
    pred = types.SimpleNamespace(
        trajectory=[{"reasoning": "final-turn", "code": "c", "output": "o"}], final_reasoning=None)
    with TraceRecorder(path, run_id="r1", clock=_counter()) as rec:
        rec.begin_main_capture()
        rec.note_main_step("stale-turn", ts=5.0)   # attempt 1
        rec.begin_main_capture()                    # attempt 2 starts → buffer cleared
        rec.note_main_step("final-turn", ts=200.0)
        rec.record_main_trajectory(pred)
    main = [e for e in load_events(path) if e["type"] == EVENT_MAIN_STEP][0]
    assert main["ts"] == 200.0   # the stale attempt-1 stamp did not match / leak


def test_record_main_trajectory_tolerates_missing_trajectory(tmp_path):
    path = str(tmp_path / "trace.jsonl")
    pred = types.SimpleNamespace()  # no trajectory, no final_reasoning
    with TraceRecorder(path, run_id="r1", clock=_counter()) as rec:
        rec.record_main_trajectory(pred)  # must not raise
    events = load_events(path)
    assert any(e["type"] == EVENT_FINAL for e in events)
    assert not any(e["type"] == EVENT_MAIN_STEP for e in events)


def test_load_events_filters_by_run(tmp_path):
    path = str(tmp_path / "trace.jsonl")
    with TraceRecorder(path, run_id="a", clock=_counter()) as rec:
        rec.record("sub_call", {})
    with TraceRecorder(path, run_id="b", clock=_counter()) as rec:
        rec.record("sub_call", {})
    assert {e["run_id"] for e in load_events(path, run_id="a")} == {"a"}
    grouped = group_by_run(load_events(path))
    assert set(grouped) == {"a", "b"}


def test_result_serialises_pydantic(tmp_path):
    from pydantic import BaseModel

    class M(BaseModel):
        a: int

    path = str(tmp_path / "trace.jsonl")
    with TraceRecorder(path, run_id="r1", clock=_counter()) as rec:
        rec.record_result(M(a=5))
    result = [e for e in load_events(path) if e["type"] == EVENT_RESULT][0]
    assert result["payload"]["output"] == {"a": 5}


def test_record_tool_call_emits_canonical_payload(tmp_path):
    path = str(tmp_path / "trace.jsonl")
    with TraceRecorder(path, run_id="r1", clock=_counter()):
        event = record_tool_call(
            "fetch_url", args={"url": "https://x"}, ok=True, result="body", note="ok"
        )
    # the helper returns the recorded event
    assert event["type"] == EVENT_TOOL_CALL
    tc = [e for e in load_events(path) if e["type"] == EVENT_TOOL_CALL][0]
    # the shape the replay/dataset readers consume: tool + args + merged extras
    assert tc["payload"] == {
        "tool": "fetch_url",
        "args": {"url": "https://x"},
        "ok": True,
        "result": "body",
        "note": "ok",
    }


def test_record_tool_call_omits_args_when_absent(tmp_path):
    path = str(tmp_path / "trace.jsonl")
    with TraceRecorder(path, run_id="r1", clock=_counter()):
        record_tool_call("validate", ok=False, errors=["bad"])
    tc = [e for e in load_events(path) if e["type"] == EVENT_TOOL_CALL][0]
    assert "args" not in tc["payload"]
    assert tc["payload"] == {"tool": "validate", "ok": False, "errors": ["bad"]}


def test_record_tool_call_noops_without_recorder():
    # No active recorder → no-op, returns None (a tool can call it unconditionally).
    assert current_recorder() is None
    assert record_tool_call("fetch_url", args={"url": "https://x"}, ok=True) is None


def test_record_is_thread_safe_under_concurrency(tmp_path):
    # llm_query_batched fans the wrapped sub_lm across threads → concurrent
    # sub_call records. The recorder must not race step_ids or interleave lines.
    path = str(tmp_path / "trace.jsonl")
    n = 200
    with TraceRecorder(path, run_id="r1", clock=_counter()) as rec:
        threads = [
            threading.Thread(target=lambda i=i: rec.record("sub_call", {"i": i}))
            for i in range(n)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    events = load_events(path)
    sub = [e for e in events if e["type"] == "sub_call"]
    assert len(sub) == n  # no line lost or corrupted
    step_ids = [e["step_id"] for e in events]
    assert len(set(step_ids)) == len(step_ids)  # every step_id unique (no race)


def test_jsonl_is_valid_json_per_line(tmp_path):
    path = str(tmp_path / "trace.jsonl")
    with TraceRecorder(path, run_id="r1", clock=_counter()) as rec:
        rec.record("sub_call", {"unicode": "漏洞"})
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            json.loads(line)  # raises if malformed


def test_recorder_scope_reestablishes_recorder_in_a_worker_thread(tmp_path):
    # A ThreadPoolExecutor worker does NOT inherit the recorder ContextVar (unlike an asyncio task), so
    # current_recorder() is None there — which is why dspy's llm_query_batched lost batched sub_calls.
    # recorder_scope re-establishes it so a record() from the worker lands in the trace.
    from concurrent.futures import ThreadPoolExecutor

    from rlm_harness.trace import current_recorder, recorder_scope

    path = str(tmp_path / "trace.jsonl")
    saw_none = {}
    with TraceRecorder(path, run_id="r1", clock=_counter()) as rec:
        def work():
            saw_none["before"] = current_recorder() is None   # the bug: empty in a fresh worker thread
            with recorder_scope(rec):
                assert current_recorder() is rec
                rec.record("sub_call", {"i": 1})
        with ThreadPoolExecutor(max_workers=1) as ex:
            ex.submit(work).result()
    assert saw_none["before"] is True   # confirms the ContextVar did NOT propagate
    sub = [e for e in load_events(path) if e["type"] == "sub_call"]
    assert len(sub) == 1                # …and recorder_scope fixed it


# ---- payload_cause: the READ side of the same distinction --------------------------------
#
# A consumer reading a trace has only the payload, and `ok` alone cannot tell a validator
# rejection from an endpoint failure or a circuit break. Reading it as one thing has shipped in
# four separate consumers — into training labels, a scored rubric criterion, and delivered report
# text. The shapes below are taken from real recorded traces, not invented.


def test_an_endpoint_payload_has_NO_ok_key_and_that_is_the_whole_trap():
    """The exact shape a real consumer records on the endpoint path: `error=` only. `ok` is ABSENT,
    so `payload.get("ok")` is None — falsy — and every `not payload.get("ok")` counter downstream
    silently absorbs infrastructure failures as content declines. In one measured corpus that was
    113 of 116 'declines' in a run whose validator ran zero times."""
    payload = {"tool": "generate_nuclei_template", "error": "harness exited 1"}

    assert "ok" not in payload
    assert not payload.get("ok"), "this is why the naive counter is wrong"
    assert payload_cause(payload) == CAUSE_ENDPOINT


def test_the_endpoint_string_is_read_under_either_conventional_key():
    """Consumers have used both `error=` and `endpoint_error=`. Reading only one would leave the
    other silently reading as a validator rejection."""
    assert payload_cause({"error": "conn reset"}) == CAUSE_ENDPOINT
    assert payload_cause({"endpoint_error": "conn reset"}) == CAUSE_ENDPOINT
    assert payload_cause({"ok": False, "endpoint_error": "conn reset"}) == CAUSE_ENDPOINT


def test_an_endpoint_error_that_STRINGIFIED_TO_NOTHING_is_still_an_endpoint_failure():
    """The empty string is the COMMON case, not a corner one, and this function shipped getting it
    wrong for a while.

    `endpoint_error` is filled with `str(exc)`, and that is `''` for `httpx.ConnectTimeout` /
    `ReadTimeout` / `ConnectError`, `TimeoutError`, `OSError` and `RemoteDisconnected` — six of the
    most ordinary transport failures there are. The original `payload.get("endpoint_error") or ...`
    sent every one of them down the `CAUSE_INVALID` branch: a dropped connection recorded as a
    content decline, which is the exact misclassification this function exists to prevent.

    The write side (`ModelToolResult.cause`) has always used `is not None`. A read-side "mirror"
    that disagrees with the thing it mirrors is worse than no mirror, because both look right in
    isolation. Reported by a downstream consumer that pinned the divergence rather than adopting it.
    """
    import http.client

    import httpx

    for exc in (
        httpx.ConnectTimeout(""), httpx.ReadTimeout(""), httpx.ConnectError(""),
        TimeoutError(), OSError(), http.client.RemoteDisconnected(),
    ):
        assert str(exc) == "", f"{type(exc).__name__} is assumed to stringify empty"
        assert payload_cause({"ok": False, "endpoint_error": str(exc)}) == CAUSE_ENDPOINT
        assert payload_cause({"ok": False, "error": str(exc)}) == CAUSE_ENDPOINT

    # And the reading it must NOT break: an absent key, and an explicit null on the success path.
    assert payload_cause({"ok": True}) == CAUSE_OK
    assert payload_cause({"ok": True, "endpoint_error": None}) == CAUSE_OK
    assert payload_cause({"ok": False, "endpoint_error": None, "errors": ["schema"]}) == CAUSE_INVALID


def test_payload_cause_agrees_with_ModelToolResult_cause_on_every_shape():
    """The two must not be able to drift: one is documented as the other's read-side mirror, and the
    divergence above existed precisely because nothing compared them."""
    from rlm_harness.tools import ModelToolResult

    for kwargs in (
        {"ok": True, "raw": "x"},
        {"ok": False, "errors": ["schema"]},
        {"ok": False, "endpoint_error": "502"},
        {"ok": False, "endpoint_error": ""},          # the case that used to disagree
        {"ok": False, "circuit_broken": True},
        {"ok": False, "circuit_broken": True, "endpoint_error": "502"},
    ):
        live = ModelToolResult(raw=kwargs.pop("raw", ""), **kwargs)
        recorded = {
            "ok": live.ok,
            "endpoint_error": live.endpoint_error,
            "circuit_broken": live.circuit_broken,
        }
        assert payload_cause(recorded) == live.cause, kwargs


def test_the_four_causes_are_distinguishable_from_recorded_payloads():
    causes = [
        payload_cause({"ok": True, "raw": "yaml"}),
        payload_cause({"ok": False, "errors": ["schema"]}),
        payload_cause({"error": "502"}),
        payload_cause({"ok": False, "circuit_broken": True, "errors": ["breaker"]}),
    ]

    assert causes == [CAUSE_OK, CAUSE_INVALID, CAUSE_ENDPOINT, CAUSE_CIRCUIT_BROKEN]
    assert len(set(causes)) == 4, "without this the assertion above could pass with one hardcoded"


def test_a_circuit_break_outranks_an_endpoint_string():
    """Order matters and must not be able to disagree with itself: a short-circuit called neither
    the model nor the validator, so it is the strongest statement available about the call."""
    assert payload_cause({"ok": False, "circuit_broken": True,
                          "error": "stale"}) == CAUSE_CIRCUIT_BROKEN


def test_a_non_model_tool_reads_as_ok_or_invalid_which_is_what_ok_already_said():
    """`payload_cause` is safe to apply to any tool_call: with no breaker and no endpoint string it
    degrades to exactly the `ok` boolean, so a caller need not branch on tool identity first."""
    assert payload_cause({"tool": "read_skill", "result_len": 900}) == CAUSE_INVALID
    assert payload_cause({"tool": "read_skill", "ok": True}) == CAUSE_OK


def test_the_live_result_and_the_recorded_payload_agree():
    """One vocabulary, checked rather than asserted in prose — the constants really are the same
    objects, and the two derivations really do return the same word for the same outcome."""
    from rlm_harness.tools.model import CAUSE_ENDPOINT as LIVE_ENDPOINT
    from rlm_harness.tools.model import ModelToolResult

    assert LIVE_ENDPOINT is CAUSE_ENDPOINT
    for result, payload in (
        (ModelToolResult(ok=True, raw="y"), {"ok": True}),
        (ModelToolResult(ok=False, raw="y"), {"ok": False}),
        (ModelToolResult(ok=False, raw="", endpoint_error="502"), {"error": "502"}),
        (ModelToolResult(ok=False, raw="", circuit_broken=True), {"circuit_broken": True}),
    ):
        assert result.cause == payload_cause(payload), (result, payload)


def test_export_actions_carries_the_cause_and_the_endpoint_string(tmp_path):
    """The record that reaches a TRAINER. The endpoint string rode nowhere at all before this: it
    is recorded under `error`, and `_action_record` carried only ok/output/errors — so a downstream
    reader could not reconstruct the split even by hand."""
    from rlm_harness.dataset import export_actions

    path = tmp_path / "t.jsonl"
    with TraceRecorder(str(path), run_id="r"):
        record_tool_call("gen", args={"spec": "s"}, ok=False, errors=["schema"], raw="bad")
        record_tool_call("gen", args={"spec": "s"}, error="harness exited 1")
        record_tool_call("gen", args={"spec": "s"}, ok=False, circuit_broken=True, errors=["brk"])

    actions = export_actions({"r": load_events(str(path), "r")})
    outcomes = [a["outcome"] for a in actions if a["kind"] == "tool"]

    assert [o["cause"] for o in outcomes] == [CAUSE_INVALID, CAUSE_ENDPOINT, CAUSE_CIRCUIT_BROKEN]
    assert outcomes[1]["error"] == "harness exited 1"
    assert outcomes[0]["error"] is None and outcomes[2]["error"] is None


def test_an_explicitly_recorded_cause_wins_over_the_derivation(tmp_path):
    """The write side is allowed to be authoritative — it is the code that knows. Only the fallback
    is a derivation, so a tool whose outcome the three keys cannot express can still say so."""
    from rlm_harness.dataset import export_actions

    path = tmp_path / "t.jsonl"
    with TraceRecorder(str(path), run_id="r"):
        record_tool_call("gen", ok=False, cause=CAUSE_ENDPOINT, errors=["looks like a reject"])

    (outcome,) = [a["outcome"] for a in export_actions({"r": load_events(str(path), "r")})
                  if a["kind"] == "tool"]

    assert outcome["cause"] == CAUSE_ENDPOINT


# ---- 1.6.0: the three fields that make a trace measurable ------------------------------
#
# All three are ADDITIVE within trace/v1 — no new event type, nothing removed or re-typed. They
# exist because an analysis of ~400 real traces could not answer three basic questions: which kit
# wrote this, how long did that tool take, and was a turn's wall-clock the model generating or the
# sandbox executing.

def test_run_start_names_the_kit_that_wrote_the_trace(tmp_path):
    """`schema` is the FORMAT version and says nothing about the producer, so a corpus spanning
    several releases is un-attributable — you cannot tell a behaviour change from a version
    change. Asserted against a REAL recorder: nothing else pins the actual run_start payload."""
    import rlm_harness

    p = tmp_path / "t.jsonl"
    with TraceRecorder(str(p), run_id="r"):
        pass
    start = [json.loads(x) for x in p.read_text().splitlines()][0]
    assert start["type"] == "run_start"
    assert start["payload"]["rlm_harness"] == rlm_harness.__version__
    assert start["schema"] == "rlm-harness/trace/v1"        # the FORMAT version, unchanged


def test_the_version_sits_beside_meta_never_inside_it(tmp_path):
    """`meta` is the caller's namespace (`rubric_to_meta` writes there); the kit must not squat
    in it, and both readers of it must keep resolving."""
    from rlm_harness.dataset import _run_meta

    p = tmp_path / "t.jsonl"
    with TraceRecorder(str(p), run_id="r", meta={"source": "S"}):
        pass
    events = [json.loads(x) for x in p.read_text().splitlines()]
    assert events[0]["payload"]["meta"] == {"source": "S"}   # untouched, no kit key inside
    assert _run_meta(events) == {"source": "S"}


def test_tool_call_duration_is_conditional_and_rounded(tmp_path):
    """Written only when given, like `args` — an unconditional null would land on every tool_call
    of every trace forever."""
    p = tmp_path / "t.jsonl"
    with TraceRecorder(str(p), run_id="r"):
        record_tool_call("a", args={})
        record_tool_call("b", args={}, duration_s=1.23456789)
    calls = [json.loads(x) for x in p.read_text().splitlines() if '"tool_call"' in x]
    assert "duration_s" not in calls[0]["payload"]
    assert calls[1]["payload"]["duration_s"] == 1.234568


def test_exec_duration_lands_on_the_turn_that_produced_it(tmp_path):
    """Matched on the code that ran, never on position."""
    p = tmp_path / "t.jsonl"
    with TraceRecorder(str(p), run_id="r") as rec:
        rec.note_exec_duration(0.5, "c0")
        rec.note_exec_duration(2.5, "c1")
        rec.record_main_trajectory(types.SimpleNamespace(
            trajectory=[{"reasoning": "r0", "code": "c0"}, {"reasoning": "r1", "code": "c1"}],
            final_reasoning="done",
        ))
    steps = [json.loads(x) for x in p.read_text().splitlines() if '"main_step"' in x]
    assert [s["payload"]["exec_duration_s"] for s in steps] == [0.5, 2.5]


def test_a_turn_dspy_never_executed_does_not_steal_the_next_turns_duration(tmp_path):
    """THE regression, and it is reachable by the model any time it tags a fence ```json.

    dspy raises SyntaxError out of `_strip_code_fences` for an explicitly non-Python fence and
    records that turn from the UNSTRIPPED text without calling execute() at all. A positional zip
    then credits the skipped turn with the next turn's time and shifts every later one — a
    confidently WRONG attribution with nothing to signal it, which is worse than none."""
    p = tmp_path / "t.jsonl"
    with TraceRecorder(str(p), run_id="r") as rec:
        # only turns 1 and 2 reached the sandbox
        rec.note_exec_duration(5.0, "heavy()")
        rec.note_exec_duration(0.1, "cheap()")
        rec.record_main_trajectory(types.SimpleNamespace(
            trajectory=[
                {"reasoning": "r0", "code": "```json\n{}\n```"},   # never executed
                {"reasoning": "r1", "code": "heavy()"},
                {"reasoning": "r2", "code": "cheap()"},
            ],
            final_reasoning="done",
        ))
    steps = [json.loads(x) for x in p.read_text().splitlines() if '"main_step"' in x]
    assert "exec_duration_s" not in steps[0]["payload"]      # never ran → absent, not 5.0
    assert steps[1]["payload"]["exec_duration_s"] == 5.0     # its own time, not shifted
    assert steps[2]["payload"]["exec_duration_s"] == 0.1     # not lost off the end


def test_two_turns_running_identical_code_each_get_their_own_duration(tmp_path):
    """Earliest-unused, the same rule the `reasoning` timestamp match already uses."""
    p = tmp_path / "t.jsonl"
    with TraceRecorder(str(p), run_id="r") as rec:
        rec.note_exec_duration(1.0, "same()")
        rec.note_exec_duration(2.0, "same()")
        rec.record_main_trajectory(types.SimpleNamespace(
            trajectory=[{"reasoning": "a", "code": "same()"}, {"reasoning": "b", "code": "same()"}],
            final_reasoning="d",
        ))
    steps = [json.loads(x) for x in p.read_text().splitlines() if '"main_step"' in x]
    assert [s["payload"]["exec_duration_s"] for s in steps] == [1.0, 2.0]


def test_a_turn_with_no_recorded_execution_has_the_key_ABSENT(tmp_path):
    """An interpreter the kit does not wrap (a caller-injected one, `ScriptedInterpreter`) stages
    nothing. Absent is the honest answer; a 0.0 would read as "measured and found instant"."""
    p = tmp_path / "t.jsonl"
    with TraceRecorder(str(p), run_id="r") as rec:
        rec.note_exec_duration(0.5, "c0")                 # only turn 0 ran in a wrapped sandbox
        rec.record_main_trajectory(types.SimpleNamespace(
            trajectory=[{"reasoning": "r0", "code": "c0"}, {"reasoning": "r1", "code": "c1"}],
            final_reasoning="d",
        ))
    steps = [json.loads(x) for x in p.read_text().splitlines() if '"main_step"' in x]
    assert steps[0]["payload"]["exec_duration_s"] == 0.5
    assert "exec_duration_s" not in steps[1]["payload"]


def test_exec_durations_reset_per_attempt(tmp_path):
    """`run_with_retry` re-runs the RLM and only the FINAL attempt is recorded, so the buffer
    clears with the ts buffer — otherwise attempt 1's durations would shift attempt 2's onto the
    wrong turns."""
    p = tmp_path / "t.jsonl"
    with TraceRecorder(str(p), run_id="r") as rec:
        rec.note_exec_duration(99.0, "c0")                # attempt 1, discarded
        rec.begin_main_capture()
        rec.note_exec_duration(1.0, "c0")                 # attempt 2, the one recorded
        rec.record_main_trajectory(types.SimpleNamespace(
            trajectory=[{"reasoning": "r0", "code": "c0"}], final_reasoning="d",
        ))
    steps = [json.loads(x) for x in p.read_text().splitlines() if '"main_step"' in x]
    assert steps[0]["payload"]["exec_duration_s"] == 1.0


def test_a_shipped_outbound_tool_records_its_own_duration(tmp_path):
    """End-to-end, not just the plumbing: the tools whose cost is a WAIT on something outside this
    process must actually carry `duration_s`, or `compute_tool_waste` is blind to exactly the
    calls it exists to account for."""
    from rlm_harness.metrics import compute_tool_waste
    from rlm_harness.tools import make_fetch_tool

    def _slow_fetcher(url):
        import time as _t

        _t.sleep(0.02)
        return "body"

    p = tmp_path / "t.jsonl"
    with TraceRecorder(str(p), run_id="r"):
        make_fetch_tool(_slow_fetcher)("https://example.com/x")
    events = [json.loads(x) for x in p.read_text().splitlines()]
    call = next(e for e in events if e["type"] == "tool_call")
    assert call["payload"]["duration_s"] >= 0.02

    # ...and it reaches the metric as measured time, not as "unknown"
    waste = compute_tool_waste(events)["fetch_url"]
    assert waste.measured_calls == 1 and waste.total_seconds >= 0.02
    assert waste.wasted_seconds == 0.0          # it succeeded


def test_a_refusal_called_outside_a_task_records_no_duration(tmp_path):
    """What is pinned here is the seam's BOUNDARY, not the old "a refusal is never timed" rule.

    That rule is gone as of 1.8.3: through a task, `_ensure_tool_timing` wraps every tool and the
    refusal branch carries a true ~0 rather than an absent field — because `None` in `ToolWaste`
    means "nobody measured", so spending it on "measured, and it was instant" makes the two
    indistinguishable. Called DIRECTLY, as here, there is no wrapper and nothing to read, which is
    unchanged and is why this file's other exact-payload assertions still hold.
    `tests/test_tool_durations.py` pins the task-seam side."""
    from rlm_harness.tools import make_fetch_tool

    p = tmp_path / "t.jsonl"
    with TraceRecorder(str(p), run_id="r"):
        make_fetch_tool(lambda u: "body")("http://127.0.0.1/admin")
    call = next(json.loads(x) for x in p.read_text().splitlines() if '"tool_call"' in x)
    assert call["payload"]["ok"] is False
    assert "duration_s" not in call["payload"]


# ---- monotonic matching: a later turn must never claim an earlier turn's staged entry ---------
#
# The bug these pin was found in SHIPPED 1.6.0 and had already reached a consumer's rendered UI
# (a -338.7s turn duration). Root cause was two stamps staged per turn (`task.py:_MainStepTimer`,
# fixed there); the cursor here is the second line of defence. See `record_main_trajectory`.


def _traj(*reasonings, code_for=lambda r: f"code-{r}"):
    return types.SimpleNamespace(
        trajectory=[{"reasoning": r, "code": code_for(r), "output": "o"} for r in reasonings],
        final_reasoning=None,
    )


def test_a_surplus_staged_stamp_cannot_drag_a_later_turn_backwards(tmp_path):
    """THE cursor regression. Stages TWO stamps per turn — what production did before
    `_MainStepTimer` learned to dedupe — with a reasoning repeated at a non-adjacent turn. Under
    the old scan-from-zero rule the last turn claimed the FIRST turn's spare stamp and the emitted
    ts went BACKWARDS, which is what rendered as a negative per-turn duration downstream."""
    path = str(tmp_path / "trace.jsonl")
    with TraceRecorder(path, run_id="r1", clock=_counter()) as rec:
        rec.begin_main_capture()
        for r, t in [("A", 1.0), ("X", 5.0), ("B", 7.0), ("X", 9.0)]:
            rec.note_main_step(r, ts=t)
            rec.note_main_step(r, ts=t + 0.1)   # the surplus fire
        rec.record_main_trajectory(_traj("A", "X", "B", "X"))
    ts = [e["ts"] for e in load_events(path) if e["type"] == EVENT_MAIN_STEP]
    assert ts == sorted(ts), f"a turn was handed an earlier turn's stamp: {ts}"
    assert ts == [1.0, 5.0, 7.0, 9.0]


def test_one_stamp_per_turn_is_what_makes_ADJACENT_duplicates_correct(tmp_path):
    """The cursor CANNOT fix an adjacent duplicate — with two stamps staged, turn 1 takes turn 0's
    spare and lands ~0.1s early: no longer negative, so the symptom hides while the value stays
    wrong. Correctness here comes only from staging ONE stamp per turn, which is why the real fix
    lives in `task.py:_MainStepTimer` and not in this matcher. Both halves are asserted so the
    trade-off cannot be silently re-litigated."""
    def _run(name, stamps):
        path = str(tmp_path / f"{name}.jsonl")
        with TraceRecorder(path, run_id="r1", clock=_counter()) as rec:
            rec.begin_main_capture()
            for r, t in stamps:
                rec.note_main_step(r, ts=t)
            rec.record_main_trajectory(_traj("X", "X"))
        return [e["ts"] for e in load_events(path) if e["type"] == EVENT_MAIN_STEP]

    assert _run("deduped", [("X", 5.0), ("X", 9.0)]) == [5.0, 9.0]
    # ...and the pre-fix staging, kept as documentation of what the cursor does NOT buy:
    doubled = _run("doubled", [("X", 5.0), ("X", 5.1), ("X", 9.0), ("X", 9.1)])
    assert doubled == [5.0, 5.1] and doubled == sorted(doubled)


def test_matched_turns_ts_is_non_decreasing_for_every_duplicate_shape(tmp_path):
    """The general invariant, asserted ONLY over turns whose stamp was matched: an unmatched turn
    falls back to clock() at flush time, which is later than every live stamp and would break
    monotonicity for reasons that are not this bug."""
    shapes = [
        ["X", "X", "X"],
        ["A", "A", "B", "A"],
        ["A", "B", "A", "B"],
        ["A", "B", "B", "A", "A"],
    ]
    for i, shape in enumerate(shapes):
        path = str(tmp_path / f"t{i}.jsonl")
        stamps = [10.0 * (n + 1) for n in range(len(shape))]
        with TraceRecorder(path, run_id="r", clock=_counter()) as rec:
            rec.begin_main_capture()
            for r, t in zip(shape, stamps):
                rec.note_main_step(r, ts=t)
            rec.record_main_trajectory(_traj(*shape))
        ts = [e["ts"] for e in load_events(path) if e["type"] == EVENT_MAIN_STEP]
        assert ts == stamps, f"{shape}: {ts}"


def test_a_setup_execution_cannot_be_claimed_by_a_later_turn(tmp_path):
    """dspy runs a setup `execute()` BEFORE the turn loop, so its duration is staged ahead of every
    turn. A turn whose code collides with it must not claim it — the cursor is what stops that."""
    path = str(tmp_path / "trace.jsonl")
    with TraceRecorder(path, run_id="r1", clock=_counter()) as rec:
        rec.begin_main_capture()
        rec.note_exec_duration(99.0, "setup")     # dspy's pre-loop execute
        rec.note_exec_duration(1.0, "turn-0")
        rec.note_exec_duration(2.0, "setup")      # a turn that happens to run the same source
        rec.record_main_trajectory(
            types.SimpleNamespace(
                trajectory=[
                    {"reasoning": "r0", "code": "turn-0", "output": "o"},
                    {"reasoning": "r1", "code": "setup", "output": "o"},
                ],
                final_reasoning=None,
            )
        )
    main = [e for e in load_events(path) if e["type"] == EVENT_MAIN_STEP]
    assert main[0]["payload"]["exec_duration_s"] == 1.0
    assert main[1]["payload"]["exec_duration_s"] == 2.0, "claimed the pre-loop setup duration"


# ---- opt-in metrics snapshot on run_end --------------------------------------------------------


def _run_end(path):
    return [e for e in load_events(path) if e["type"] == EVENT_RUN_END][0]["payload"]


def test_metrics_snapshot_is_OFF_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("RLM_TRACE_METRICS", raising=False)
    path = str(tmp_path / "t.jsonl")
    with TraceRecorder(path, run_id="r") as rec:
        rec.record(EVENT_MAIN_STEP, {"turn": 0, "code": "x=1"})
    assert "metrics" not in _run_end(path)


def test_an_explicit_record_metrics_False_beats_the_environment(tmp_path, monkeypatch):
    """The kwarg is documented as the way to bypass the environment, and only the bypass-to-ON
    direction was pinned. `if record_metrics is None` narrowed to `if not record_metrics` survived
    the whole suite — under which an env var silently overrides an explicit caller opt-out and
    writes prompts into a trace the caller asked to keep clean."""
    monkeypatch.setenv("RLM_TRACE_METRICS", "1")
    path = str(tmp_path / "t.jsonl")
    with TraceRecorder(path, run_id="r", record_metrics=False) as rec:
        rec.record(EVENT_MAIN_STEP, {"turn": 0, "code": "x=1"})
    assert "metrics" not in _run_end(path)


def test_metrics_snapshot_turns_on_by_env_read_at_construction(tmp_path, monkeypatch):
    """The env is resolved at `__init__`, so the value is fixed for the recorder's life and a
    mid-run change cannot half-apply."""
    monkeypatch.setenv("RLM_TRACE_METRICS", "1")
    path = str(tmp_path / "t.jsonl")
    with TraceRecorder(path, run_id="r") as rec:
        monkeypatch.delenv("RLM_TRACE_METRICS")      # too late to matter
        rec.record(EVENT_MAIN_STEP, {"turn": 0, "code": "x=1"})
    assert _run_end(path)["metrics"]["main_steps"] == 1


def test_metrics_snapshot_is_SKIPPED_not_zeroed_when_the_reread_finds_nothing(tmp_path):
    """`load_events` returns `[]` for a rotated file, `/dev/null`, or a `run_id` that does not
    match — and `compute_run_facts([])` would then emit `main_steps: 0` and the rest, which is
    indistinguishable from a measured zero AND is streamed live to every consumer's `on_event`.
    An absent event is not a measurement, one layer over.

    Driven with a `run_id` mismatch specifically: a path that cannot be READ raises instead, the
    suppression swallows it, and no `metrics` key appears with or without the guard — so that
    driver would leave the mutation green."""
    path = str(tmp_path / "t.jsonl")
    rec = TraceRecorder(path, run_id="written-as-this", record_metrics=True)
    with rec:
        rec.record(EVENT_MAIN_STEP, {"turn": 0, "code": "x=1"})
        rec.run_id = "now-looks-for-this"
    payload = _run_end(path)
    assert "metrics" not in payload, "an all-zero snapshot was emitted for a run it could not read"
    assert payload["ok"] is True, "run_end itself must still be written"


def test_metrics_snapshot_is_computed_from_the_FILE_filtered_by_run_id(tmp_path):
    """Consistent-by-construction with the bytes it sits beside — and the `run_id` filter matters
    because the handle is opened in append mode and one file may hold several runs."""
    path = str(tmp_path / "t.jsonl")
    with TraceRecorder(path, run_id="a", record_metrics=True) as rec:
        for i in range(3):
            rec.record(EVENT_MAIN_STEP, {"turn": i, "code": "x=1"})
    with TraceRecorder(path, run_id="b", record_metrics=True) as rec:
        rec.record(EVENT_MAIN_STEP, {"turn": 0, "code": "x=1"})
    got = {e["run_id"]: e["payload"]["metrics"]["main_steps"]
           for e in load_events(path) if e["type"] == EVENT_RUN_END}
    assert got == {"a": 3, "b": 1}


def test_run_end_survives_a_BaseException_escaping_the_snapshot(tmp_path, monkeypatch):
    """`suppress(Exception)` does not catch `BaseException`, so `run_end` is recorded from a
    `finally` — otherwise a Ctrl-C during the re-read loses it on precisely the killed-run path
    that is most worth analysing.

    The interrupt must be raised INSIDE the snapshot, not in the `with` body: a body raise arrives
    at `__exit__` as an ARGUMENT and never unwinds it, so the `finally` is never exercised and the
    test passes with the record moved back into the `try`. That vacuous form shipped first."""
    import rlm_harness.metrics as metrics_mod

    def interrupted(_events):
        raise KeyboardInterrupt("Ctrl-C during the re-read")

    monkeypatch.setattr(metrics_mod, "compute_run_facts", interrupted)
    path = str(tmp_path / "t.jsonl")
    # `pytest.raises`, not `suppress`: the interrupt must still PROPAGATE. Asserting only that
    # `run_end` exists holds for `suppress(BaseException)` too, and that mutant survived the whole
    # suite — a recorder inside a coroutine would then silently eat its own `CancelledError`.
    with pytest.raises(KeyboardInterrupt), \
            TraceRecorder(path, run_id="k", record_metrics=True) as rec:
        rec.record(EVENT_MAIN_STEP, {"turn": 0, "code": "x=1"})
    assert any(e["type"] == EVENT_RUN_END for e in load_events(path))


def test_a_raise_inside_the_snapshot_keeps_run_end_and_the_original_exception(tmp_path, monkeypatch):
    """Observability must never break the run or replace its diagnosis."""
    import rlm_harness.metrics as metrics_mod

    def boom(_events):
        raise RuntimeError("metrics bug")

    monkeypatch.setattr(metrics_mod, "compute_run_facts", boom)
    path = str(tmp_path / "t.jsonl")
    with pytest.raises(ValueError, match="the real failure"), \
            TraceRecorder(path, run_id="r", record_metrics=True) as rec:
        rec.record(EVENT_MAIN_STEP, {"turn": 0, "code": "x=1"})
        raise ValueError("the real failure")
    payload = _run_end(path)
    assert "metrics" not in payload and payload["ok"] is False


def test_the_reread_follows_the_path_as_it_was_at_ENTER_not_at_init(tmp_path, monkeypatch):
    """`open()` resolves a relative path at `__enter__`, so the re-read must resolve it there too.

    Two chdirs, and both matter. The one AFTER entering catches dropping the stash entirely. The one
    BETWEEN constructing and entering catches stashing at `__init__` instead — that mutant passes an
    enter-then-chdir test, because both placements stash before that chdir, and it survived the whole
    suite. Under it `open()` writes to the new cwd while `_abs_path` points at the old one, so the
    re-read raises, `suppress` eats it, and the snapshot silently disappears."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    rec = TraceRecorder("rel.jsonl", run_id="r", record_metrics=True)   # constructed HERE
    monkeypatch.chdir(tmp_path)                                        # ...entered THERE
    with rec:
        rec.record(EVENT_MAIN_STEP, {"turn": 0, "code": "x=1"})
        (tmp_path / "after").mkdir()
        monkeypatch.chdir(tmp_path / "after")                          # ...and exited somewhere else
    assert _run_end(str(tmp_path / "rel.jsonl"))["metrics"]["main_steps"] == 1


def test_a_truncated_file_with_this_runs_events_but_no_run_start_emits_nothing(tmp_path):
    """The guard is `run_start present`, not `events non-empty`, and the difference is reachable.

    A log-rotated file can drop its head while keeping this `run_id`'s later events. `not events`
    would then compute facts from a truncated stream and publish them as this run's — a measured-
    looking number over a fragment. Weakening the guard that way survived every other test, because
    the drivers the plan specified (`/dev/null`, a `run_id` mismatch) both yield an EMPTY list, and
    an empty list satisfies both forms."""
    import json

    path = tmp_path / "rotated.jsonl"
    # Hand-built: this run's turns survived the rotation, its run_start did not.
    with path.open("w") as fh:
        for turn in range(3):
            fh.write(json.dumps({
                "schema": "rlm-harness/trace/v1", "run_id": "r", "step_id": turn, "ts": 1.0,
                "type": EVENT_MAIN_STEP, "payload": {"turn": turn, "code": "x=1"},
            }) + "\n")

    rec = TraceRecorder(str(path), run_id="r", record_metrics=True)
    rec._abs_path = str(path)
    assert rec._snapshot_facts() is None, "facts were computed from a headless fragment"
