from rlm_harness.dataset import export_actions


def _run():
    return {
        "r1": [
            {"type": "run_start", "step_id": 0, "payload": {}},
            {"type": "main_step", "step_id": 1, "payload": {"reasoning": "plan", "code": "c"}},
            {"type": "tool_call", "step_id": 2, "payload": {
                "tool": "gen", "args": {"spec": "s"}, "reasoning": "qr",
                "raw": "id: x", "ok": True, "errors": []}},
            {"type": "sub_call", "step_id": 3, "payload": {
                "model": "gpt-5.5", "input": "ask", "processed": "answer"}},
            {"type": "main_step", "step_id": 4, "payload": {"reasoning": "assemble", "code": "c2"}},
        ]
    }


def test_export_actions_kinds_and_order():
    recs = export_actions(_run(), reward=lambda ev: 1.0)
    assert [r["kind"] for r in recs] == ["planner", "tool", "sub", "planner"]
    assert all(r["reward"] == 1.0 for r in recs)


def test_export_actions_tool_and_sub_payloads():
    recs = export_actions(_run())
    tool = recs[1]
    assert tool["tool"] == "gen" and tool["outcome"]["ok"] is True
    assert tool["action"]["input"] == {"spec": "s"}
    sub = recs[2]
    assert sub["action"]["input"] == "ask" and sub["outcome"]["output"] == "answer"


def test_export_actions_state_accumulates():
    recs = export_actions(_run())
    assert recs[0]["state"] == []
    assert len(recs[3]["state"]) == 3  # planner, tool, sub seen before the 2nd planner step


def test_export_actions_reward_none_when_unset():
    recs = export_actions(_run())
    assert all(r["reward"] is None for r in recs)


def test_export_actions_tool_output_fallback():
    # A tool_call payload may carry its output under raw / result / preview (record_tool_call pins
    # none). The exporter reads a fallback so no tool's output is dropped, with "raw" winning first
    # for back-compat with existing traces.
    runs = {
        "r": [
            {"type": "tool_call", "step_id": 1,
             "payload": {"tool": "kit_tool", "result": "R", "ok": True}},   # model_as_tool/list_skills
            {"type": "tool_call", "step_id": 2,
             "payload": {"tool": "mcp", "preview": "P", "ok": True}},       # read_skill / MCP
            {"type": "tool_call", "step_id": 3,
             "payload": {"tool": "search", "results": ["a", "b"], "ok": True}},  # web_search
            {"type": "tool_call", "step_id": 4,
             "payload": {"tool": "gen", "raw": "RAW", "result": "R2", "ok": True}},  # raw wins
            {"type": "tool_call", "step_id": 5,
             "payload": {"tool": "quiet", "ok": True}},                     # no output key -> None
        ]
    }
    outputs = [r["outcome"]["output"] for r in export_actions(runs)]
    assert outputs == ["R", "P", ["a", "b"], "RAW", None]


def test_an_endpoint_failure_that_stringified_to_nothing_still_carries_its_cause_and_error():
    """A trainer reading this record must not see `cause: endpoint` beside `error: null`.

    `endpoint_error` is `str(exc)`, which is `''` for the six most ordinary transport failures
    (`httpx.ConnectTimeout`/`ReadTimeout`/`ConnectError`, `TimeoutError`, `OSError`,
    `RemoteDisconnected`). Both fields used to lose it: `cause` because `payload_cause` tested the
    key for TRUTHINESS rather than presence, and `error` because `a or b` skips a present-but-empty
    `a` and falls through to an absent `b`. Presence, not truthiness, in both places.
    """
    runs = {
        "r1": [
            {"type": "tool_call", "step_id": 0, "payload": {
                "tool": "gen", "ok": False, "endpoint_error": "", "raw": ""}},
            {"type": "tool_call", "step_id": 1, "payload": {
                "tool": "gen", "ok": False, "error": "", "raw": ""}},
            {"type": "tool_call", "step_id": 2, "payload": {
                "tool": "gen", "ok": False, "errors": ["schema"], "raw": "bad"}},
        ]
    }
    outcomes = [r["outcome"] for r in export_actions(runs)]

    assert [o["cause"] for o in outcomes] == ["endpoint", "endpoint", "invalid"]
    assert outcomes[0]["error"] == "" and outcomes[1]["error"] == ""
    assert outcomes[2]["error"] is None, "a validator rejection carries no endpoint string"


# --- causal order, against a fixture shaped like a REAL trace -------------------------------
#
# Every fixture above is hand-written in causal order with no `ts` at all, which is why the
# defect below survived: the recorder never produces that shape. `record_main_trajectory` writes
# every `main_step` in one batch once `aforward()` has returned, so a real run's turns all carry
# HIGHER step_ids than its live tool/sub calls.

def _realistic_run():
    """Two turns, each making one tool call, written the way the recorder writes them.

    A turn's `ts` is stamped when its reasoning was PARSED, so it precedes the tool calls that
    turn's own code then makes. Causal order is therefore planner, tool, planner, tool.
    """
    return {
        "r": [
            {"type": "run_start", "step_id": 0, "ts": 100.0, "payload": {}},
            # live, mid-run
            {"type": "tool_call", "step_id": 1, "ts": 101.5,
             "payload": {"tool": "grep_repo", "ok": True, "result": "a"}},
            {"type": "sub_call", "step_id": 2, "ts": 103.5,
             "payload": {"model": "m", "input": "q", "processed": "a"}},
            # the whole trajectory, flushed at the end — note the step_ids
            {"type": "main_step", "step_id": 3, "ts": 101.0,
             "payload": {"turn": 0, "reasoning": "first", "code": "c0"}},
            {"type": "main_step", "step_id": 4, "ts": 103.0,
             "payload": {"turn": 1, "reasoning": "second", "code": "c1"}},
        ]
    }


def test_records_come_out_in_causal_order_not_write_order():
    """Sorting the three action types together by `step_id` put EVERY turn after EVERY live call,
    because the trajectory batch lands last. The interleave comes from `ts`."""
    recs = export_actions(_realistic_run())
    assert [r["kind"] for r in recs] == ["planner", "tool", "planner", "sub"]


def test_state_is_the_actions_that_actually_preceded():
    """The defect was systematic and in OPPOSITE directions per kind: a tool record's prior
    actions contained no turns at all, and a turn's contained every tool call. Under `step_id`
    this read [], ['tool'], ['tool','sub'], ['tool','sub','planner']."""
    recs = export_actions(_realistic_run())
    assert [[h["kind"] for h in r["state"]] for r in recs] == [
        [],
        ["planner"],
        ["planner", "tool"],
        ["planner", "tool", "planner"],
    ]


def test_turn_order_survives_a_backfilled_ts_that_went_backwards():
    """`payload["turn"]` is authoritative among turns, never `ts` — a turn whose live stamp could
    not be matched falls back to the flush time, which is LATER than every real stamp, so ordering
    turns by `ts` would move it to the end. The pre-sort on `payload["turn"]` is the whole
    guarantee; there is no clamp in the merge (the first version had one and it was provably dead).
    This fixture carries no live events on purpose, so it isolates turn order from the interleave."""
    runs = {"r": [
        {"type": "main_step", "step_id": 3, "ts": 101.0,
         "payload": {"turn": 0, "reasoning": "first", "code": "c0"}},
        # turn 1's stamp was unmatched -> flush time, far in the future
        {"type": "main_step", "step_id": 4, "ts": 999.0,
         "payload": {"turn": 1, "reasoning": "second", "code": "c1"}},
        {"type": "main_step", "step_id": 5, "ts": 102.0,
         "payload": {"turn": 2, "reasoning": "third", "code": "c2"}},
    ]}
    recs = export_actions(runs)
    assert [r["action"]["reasoning"] for r in recs] == ["first", "second", "third"]


def test_a_corpus_with_no_timestamps_keeps_write_order():
    """Every fixture predating this change carries no `ts`, and neither does any trace from before
    the field existed. There is no causal signal in such input, so `step_id` order is the only
    order available and must not become something else."""
    recs = export_actions(_run())
    assert [r["kind"] for r in recs] == ["planner", "tool", "sub", "planner"]


def test_a_run_with_no_interleave_signal_says_so(caplog):
    """When every turn carries a FLUSH time the merge reproduces the pre-1.11.2 order exactly —
    same records, same wrong `state` — so a re-export looks like it did something and did not.
    That state is undetectable from the output, which is why it is logged."""
    import logging

    flushed = {"r": [
        {"type": "tool_call", "step_id": 1, "ts": 101.0,
         "payload": {"tool": "grep_repo", "ok": True, "result": "a"}},
        {"type": "sub_call", "step_id": 2, "ts": 102.0,
         "payload": {"model": "m", "input": "q", "processed": "a"}},
        # both turns stamped at the flush, i.e. after everything live
        {"type": "main_step", "step_id": 3, "ts": 500.0,
         "payload": {"turn": 0, "reasoning": "first", "code": "c0"}},
        {"type": "main_step", "step_id": 4, "ts": 500.0,
         "payload": {"turn": 1, "reasoning": "second", "code": "c1"}},
    ]}
    with caplog.at_level(logging.DEBUG, logger="rlm_harness.dataset"):
        recs = export_actions(flushed)
    assert [r["kind"] for r in recs] == ["tool", "sub", "planner", "planner"], (
        "with no signal the order must be the honest one, not a guess"
    )
    assert len(caplog.records) == 1, "the degraded state must be reported"

    # ...and a run that DOES carry the signal must stay quiet, or the line is noise.
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="rlm_harness.dataset"):
        export_actions(_realistic_run())
    assert caplog.records == []


def test_detector_silent_when_only_one_family_is_present(caplog):
    """A run of only turns, or only live calls, has nothing to interleave — reporting "no signal"
    there would fire on every run that makes no tool calls, which is a whole task shape."""
    import logging

    turns_only = {"r": [{"type": "main_step", "step_id": 1, "ts": 500.0,
                         "payload": {"turn": 0, "reasoning": "a", "code": "c"}}]}
    live_only = {"r": [{"type": "tool_call", "step_id": 1, "ts": 101.0,
                        "payload": {"tool": "t", "ok": True}}]}
    for runs in (turns_only, live_only):
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="rlm_harness.dataset"):
            export_actions(runs)
        assert caplog.records == [], "nothing to interleave is not a missing signal"


def test_one_unmatched_stamp_is_enough_to_report(caplog):
    """A flush time is later than everything live, so a SINGLE bad stamp on turn 0 swallows the
    rest of the run — the detector must not wait for every turn to be bad."""
    import logging

    partial = {"r": [
        {"type": "tool_call", "step_id": 1, "ts": 101.0,
         "payload": {"tool": "t", "ok": True}},
        {"type": "main_step", "step_id": 2, "ts": 500.0,   # unmatched -> flush
         "payload": {"turn": 0, "reasoning": "first", "code": "c0"}},
        {"type": "main_step", "step_id": 3, "ts": 502.0,   # live, but after the damage
         "payload": {"turn": 1, "reasoning": "second", "code": "c1"}},
    ]}
    with caplog.at_level(logging.DEBUG, logger="rlm_harness.dataset"):
        recs = export_actions(partial)
    assert [r["kind"] for r in recs] == ["tool", "planner", "planner"]
    assert len(caplog.records) == 1 and "turn 0" in caplog.records[0].getMessage()


def test_a_turn_zero_stamped_after_the_first_live_event_is_reported(caplog):
    """The comparison is against the run's FIRST live event, not its last, and this is the case
    that separates them. Nothing live can happen before turn 0 is parsed, so a live event EARLIER
    than turn 0's stamp proves that stamp is not a live one — even when turn 0 still sits in the
    middle of the live events rather than after all of them."""
    import logging

    runs = {"r": [
        {"type": "tool_call", "step_id": 1, "ts": 101.0, "payload": {"tool": "a", "ok": True}},
        {"type": "tool_call", "step_id": 2, "ts": 102.0, "payload": {"tool": "b", "ok": True}},
        # between the two, so `turns[0] >= live[-1]` would read this as healthy
        {"type": "main_step", "step_id": 3, "ts": 101.5,
         "payload": {"turn": 0, "reasoning": "first", "code": "c0"}},
    ]}
    with caplog.at_level(logging.DEBUG, logger="rlm_harness.dataset"):
        export_actions(runs)
    assert len(caplog.records) == 1, "compared against the last live event instead of the first"
