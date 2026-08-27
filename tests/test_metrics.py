"""Trace utilization metrics — pure functions over plain trace/v1 event dicts. All offline,
dspy-free (constructs events by hand, same style as test_rubric.py — no real TraceRecorder needed).
"""

from rlm_harness.metrics import (
    RunUtilization,
    ToolWaste,
    compute_run_utilization,
    compute_tool_waste,
    compute_tool_waste_by_run,
    compute_utilization_by_run,
)
from rlm_harness.trace import EVENT_MAIN_STEP, EVENT_SUB_CALL, EVENT_TOOL_CALL


def _main_step():
    return {"type": EVENT_MAIN_STEP, "payload": {}}


def _tool_call(name):
    return {"type": EVENT_TOOL_CALL, "payload": {"tool": name}}


def _sub_call():
    return {"type": EVENT_SUB_CALL, "payload": {}}


def test_compute_run_utilization_counts_and_rates():
    events = [
        _main_step(),
        _tool_call("fetch_url"),
        _main_step(),
        _tool_call("fetch_url"),
        _tool_call("run_command"),
        _sub_call(),
        _main_step(),
    ]
    u = compute_run_utilization(events)
    assert u.main_steps == 3
    assert u.tool_calls_total == 3
    assert u.tool_calls_by_name == {"fetch_url": 2, "run_command": 1}
    assert u.sub_calls_total == 1
    assert u.tool_call_rate == 1.0          # 3 tool calls / 3 main steps
    assert abs(u.sub_call_rate - 1 / 3) < 1e-9


def test_zero_main_steps_gives_none_rates_not_zero_or_error():
    # A run that never took a root-LM turn has no denominator — 0.0 would misleadingly read as
    # "measured and found to be zero usage" rather than "undefined."
    events = []
    u = compute_run_utilization(events)
    assert u.main_steps == 0
    assert u.tool_calls_total == 0
    assert u.tool_call_rate is None
    assert u.sub_call_rate is None


def test_zero_tool_and_sub_calls_with_nonzero_main_steps_gives_real_zero_rates():
    # Distinct from the "no denominator" case above: this IS a real, meaningful measurement.
    events = [_main_step(), _main_step()]
    u = compute_run_utilization(events)
    assert u.main_steps == 2
    assert u.tool_call_rate == 0.0
    assert u.sub_call_rate == 0.0


def test_crashed_run_has_zero_main_steps_but_live_tool_and_sub_activity():
    # The REAL, reachable trigger for the None-rate branch: task.py's arun() only calls
    # record_main_trajectory `if "prediction" in captured` — a run that fails before
    # rlm.aforward() ever returns a Prediction has zero main_step events, even though tool_call/
    # sub_call events were recorded LIVE during the run. This is not the synthetic fully-empty
    # trace case above; it is a genuinely partial, crashed trajectory with real activity.
    events = [_tool_call("read_file"), _sub_call(), _tool_call("read_file")]
    u = compute_run_utilization(events)
    assert u.main_steps == 0
    assert u.tool_calls_total == 2
    assert u.sub_calls_total == 1
    assert u.tool_call_rate is None
    assert u.sub_call_rate is None


def test_compute_utilization_by_run_partitions_a_multi_run_event_list():
    events = [
        {"run_id": "r1", **_main_step()},
        {"run_id": "r1", **_tool_call("fetch_url")},
        {"run_id": "r2", **_main_step()},
        {"run_id": "r2", **_main_step()},
    ]
    by_run = compute_utilization_by_run(events)
    assert set(by_run) == {"r1", "r2"}
    assert isinstance(by_run["r1"], RunUtilization)
    assert by_run["r1"].main_steps == 1 and by_run["r1"].tool_calls_total == 1
    assert by_run["r2"].main_steps == 2 and by_run["r2"].tool_calls_total == 0


# ---- compute_tool_waste: which calls produced nothing usable, and what they cost ----------
#
# The motivating number: on the corpus that prompted 1.6.0, 57% of all tool wall-clock produced
# output the consumer's own validator rejected — invisible, because a trace carried no durations
# and `ok` alone cannot tell a rejection from an endpoint failure.

def _tc(name, *, duration_s=None, **payload):
    p = {"tool": name, **payload}
    if duration_s is not None:
        p["duration_s"] = duration_s
    return {"type": EVENT_TOOL_CALL, "payload": p}


def test_outcomes_are_split_by_cause_not_by_ok():
    """`ok` is frequently ABSENT on an endpoint-failure payload, so `payload.get("ok")` reads
    None and a naive counter absorbs infrastructure failures as content declines — the mistake
    `payload_cause` exists to prevent, and which has shipped four times."""
    w = compute_tool_waste([
        _tc("gen", ok=True),
        _tc("gen", ok=False),                                  # validator rejected
        _tc("gen", error="connection reset"),                  # endpoint: note NO `ok` key
        _tc("gen", ok=False, circuit_broken=True),             # nothing was called at all
    ])["gen"]
    assert (w.calls, w.ok, w.invalid, w.endpoint_errors, w.circuit_broken) == (4, 1, 1, 1, 1)


def test_invalid_rate_is_denominated_over_calls_that_reached_the_validator():
    """A circuit break ran no validator and an endpoint failure produced no output to judge;
    counting either in the denominator understates how often output was actually rejected."""
    w = compute_tool_waste([
        _tc("gen", ok=True), _tc("gen", ok=False),
        _tc("gen", error="boom"), _tc("gen", ok=False, circuit_broken=True),
    ])["gen"]
    assert w.invalid_rate == 0.5           # 1 of the 2 judged, NOT 1 of 4


def test_no_validator_ever_ran_gives_none_not_zero():
    w = compute_tool_waste([_tc("gen", error="boom")])["gen"]
    assert w.invalid_rate is None


def test_wasted_seconds_counts_invalid_and_endpoint_but_not_ok_or_broken():
    w = compute_tool_waste([
        _tc("gen", ok=True, duration_s=10.0),
        _tc("gen", ok=False, duration_s=30.0),                     # wasted
        _tc("gen", error="boom", duration_s=5.0),                  # wasted
        _tc("gen", ok=False, circuit_broken=True, duration_s=0.1),  # cheap by design, not waste
    ])["gen"]
    assert w.total_seconds == 45.1
    assert w.wasted_seconds == 35.0
    assert abs(w.wasted_share - 35.0 / 45.1) < 1e-9


def test_a_trace_without_durations_reports_none_not_zero_and_never_infers():
    """The degradation path for every trace written before 1.6.0. `None` means "not recorded";
    `0.0` would read as "measured and found to be free". Inferring it from the gaps between
    events is the error this whole release exists to stop — it charges a turn's model generation
    to that turn's first tool call."""
    w = compute_tool_waste([_tc("gen", ok=False), _tc("gen", ok=True)])["gen"]
    assert w.calls == 2 and w.invalid == 1        # counts still work
    assert w.total_seconds is None
    assert w.wasted_seconds is None
    assert w.wasted_share is None
    assert w.measured_calls == 0


def test_partially_measured_tool_reports_only_what_it_measured():
    w = compute_tool_waste([_tc("gen", ok=False, duration_s=7.0), _tc("gen", ok=False)])["gen"]
    assert w.calls == 2 and w.measured_calls == 1 and w.wasted_seconds == 7.0


def test_compute_tool_waste_by_run_partitions_by_run_id():
    evs = [
        {"run_id": "a", **_tc("gen", ok=True)},
        {"run_id": "b", **_tc("gen", ok=False)},
    ]
    by_run = compute_tool_waste_by_run(evs)
    assert by_run["a"]["gen"].ok == 1 and by_run["b"]["gen"].invalid == 1


def test_it_is_reward_free():
    """Same charter as the rest of this module: counts, seconds and rates — never a score."""
    fields = set(ToolWaste.__dataclass_fields__)
    assert not any("reward" in f or "score" in f for f in fields), fields
