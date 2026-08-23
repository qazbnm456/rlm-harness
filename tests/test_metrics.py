"""Trace utilization metrics — pure functions over plain trace/v1 event dicts. All offline,
dspy-free (constructs events by hand, same style as test_rubric.py — no real TraceRecorder needed).
"""

from rlm_harness.metrics import RunUtilization, compute_run_utilization, compute_utilization_by_run
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
