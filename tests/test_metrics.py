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
from rlm_harness.trace import EVENT_FINAL, EVENT_MAIN_STEP, EVENT_SUB_CALL, EVENT_TOOL_CALL


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


# ---- compute_run_facts: the generic half of a rubric's facts -----------------------------------


def _facts(events):
    from rlm_harness.metrics import compute_run_facts

    return compute_run_facts(events)


def _step(turn=0, code="x = 1"):
    """A richer main_step than the module-level `_main_step()`, which other tests share."""
    return {"type": EVENT_MAIN_STEP, "payload": {"turn": turn, "code": code}}


def _call(name="g", **payload):
    return {"type": EVENT_TOOL_CALL, "payload": {"tool": name, "ok": True, **payload}}


def test_run_facts_keys_are_exactly_the_public_constant_and_json_safe():
    """The dict is BUILT against `RUN_FACT_KEYS`, not merely documented to match it — and every
    value is a scalar. `compute_tool_waste` returns frozen `ToolWaste` dataclasses, which
    `record()`'s `json.dumps(..., default=str)` would silently stringify into the source of truth
    as `"ToolWaste(tool='g', calls=1, ...)"`. The round-trip is what catches that."""
    import json

    from rlm_harness.metrics import RUN_FACT_KEYS

    facts = _facts([_step(), _call(), _sub_call()])
    assert tuple(facts) == RUN_FACT_KEYS
    assert json.loads(json.dumps(facts)) == facts


def test_wasted_and_total_seconds_are_None_when_nothing_was_measured():
    """`ToolWaste` makes BOTH `*_seconds` `None`, never `0.0`, under the same condition. Summing
    with `or 0.0` would regress that discipline inside the dict that feeds a rubric — where an
    unmeasured cost would read as a measured zero."""
    facts = _facts([_step(), _call()])                        # no duration_s anywhere
    assert facts["tool_wasted_seconds"] is None
    assert facts["tool_total_seconds"] is None
    assert facts["tool_measured_calls"] == 0


def test_seconds_are_a_partial_sum_and_measured_calls_shows_the_partiality():
    """One measured call in fifty is otherwise indistinguishable from fifty in fifty — which is the
    whole reason `tool_measured_calls` is in the key set."""
    events = [_step(), _call(duration_s=1.5)] + [_call() for _ in range(4)]
    facts = _facts(events)
    assert facts["tool_total_seconds"] == 1.5
    assert facts["tool_measured_calls"] == 1 and facts["tool_calls"] == 5


def test_fence_refused_counts_a_valid_python_cell_whose_fence_is_in_a_string():
    """THE case that matters: 60 of 60 real refusals had the fence BURIED rather than leading (55 of
    them valid Python assigning a documentation page whose text contains a fenced example). A rule that only caught cells STARTING with a fence
    would find none of them."""
    buried = _step(0, 'md = """# Overview\n\n```bash\nrun\n```\n"""')
    ran = _step(1, "x = 1")
    fenced_python = _step(2, "```python\nx = 1\n```")
    facts = _facts([buried, ran, fenced_python])
    assert facts["fence_refused_turns"] == 1
    assert facts["main_steps"] == 3


def test_budget_exhausted_is_tri_state_and_matched_on_exact_equality():
    """`None` is not `False`: a run whose `aforward` raised records no `final` at all (31 of 503
    real runs), and shape drift can leave `final_reasoning` absent. And the success path writes the
    MODEL's own reasoning, so a substring test would let a model quoting the phrase flip the fact."""
    from rlm_harness import _dspy_compat

    marker = _dspy_compat.forced_final_marker()
    assert _facts([{"type": "final", "payload": {"final_reasoning": marker}}])["budget_exhausted"]
    assert _facts([{"type": "final", "payload": {"final_reasoning": "done"}}])["budget_exhausted"] is False
    assert _facts([_step()])["budget_exhausted"] is None
    assert _facts([{"type": "final", "payload": {}}])["budget_exhausted"] is None
    quoting = f"I will {marker} as my answer"
    assert _facts([{"type": "final", "payload": {"final_reasoning": quoting}}])["budget_exhausted"] is False


def test_run_facts_by_run_splits_a_multi_run_file():
    """The sibling both other computers already ship — without it a multi-run file conflates."""
    from rlm_harness.metrics import compute_run_facts_by_run

    events = [dict(_step(0), run_id="a"), dict(_step(1), run_id="a"),
              dict(_step(0), run_id="b")]
    by_run = compute_run_facts_by_run(events)
    assert by_run["a"]["main_steps"] == 2 and by_run["b"]["main_steps"] == 1


def test_every_run_fact_key_carries_the_value_it_names():
    """A pinned key set says nothing about whether each key is WIRED to the right source.

    **The test asserts pairwise distinctness of its own expectations before using them**, because
    hand-checking that property failed twice: the first fixture had `circuit_breaks == ok == 1`, so
    the assertion labelled "fed from ok?" passed under exactly that mis-wiring; the second fixed
    that pair and left `endpoint_errors == circuit_breaks` and `declines == main_steps`, so three
    more swaps survived the whole suite. A fixture for this job is only as good as its
    least-distinct pair, and checking that by eye is what kept failing.
    """
    events = (
        [_step(i) for i in range(6)]                                    # main_steps
        + [{"type": EVENT_SUB_CALL, "payload": {}} for _ in range(7)]   # sub_calls
        + [_call("a", ok=False) for _ in range(1)]                      # decline, unmeasured
        + [_call("a", ok=False, duration_s=1.0) for _ in range(2)]      # declines, measured
        + [_call("b", error="boom", duration_s=1.75) for _ in range(2)] # endpoint errors, measured
        + [_call("c", circuit_broken=True) for _ in range(4)]           # circuit breaks
        + [_call("d", ok=True, duration_s=2.5)]                         # ok, measured
    )
    expected = {
        "main_steps": 6, "sub_calls": 7, "tool_calls": 10,
        "tool_declines": 3, "tool_endpoint_errors": 2, "tool_circuit_breaks": 4,
        "tool_ok": 1, "tool_measured_calls": 5, "tool_total_seconds": 8.0,
    }
    # Every number this test asserts goes through the guard, not just the dict — `wasted_seconds`
    # and the two rates are asserted below and must not collide with anything either.
    vals = [*expected.values(), 5.5, 10 / 6, 7 / 6]
    assert len(set(vals)) == len(vals), f"fixture has a masking pair: {sorted(vals)}"

    f = _facts(events)
    for key, want in expected.items():
        assert f[key] == want, f"{key} is {f[key]}, expected {want} — wired to the wrong source?"
    assert f["tool_call_rate"] == 10 / 6 and f["sub_call_rate"] == 7 / 6, "rates swapped?"
    assert f["tool_wasted_seconds"] == 5.5     # the 2 declines + 2 endpoint errors that had a clock


def test_fence_refusal_counts_only_MAIN_STEP_events():
    """A `tool_call` whose payload happens to carry a fenced `code` string is not a turn."""
    not_a_turn = {"type": EVENT_TOOL_CALL, "payload": {"tool": "g", "code": "```bash\nx\n```"}}
    assert _facts([_step(0), not_a_turn])["fence_refused_turns"] == 0


def test_budget_exhausted_takes_the_LAST_final_when_a_run_id_carries_two():
    """Documented rule; nothing else pins which end of the list wins."""
    from rlm_harness import _dspy_compat

    marker = _dspy_compat.forced_final_marker()
    events = [{"type": EVENT_FINAL, "payload": {"final_reasoning": "done"}},
              {"type": EVENT_FINAL, "payload": {"final_reasoning": marker}}]
    assert _facts(events)["budget_exhausted"] is True
    assert _facts(list(reversed(events)))["budget_exhausted"] is False


def test_compute_run_facts_accepts_a_generator():
    """`compute_run_facts` materialises its argument first. Without that, the first consumer drains
    the stream and every later one sees an empty run — silently, with plausible zeros rather than an
    error, which is the failure mode this module keeps refusing to ship."""
    events = [_step(0), _step(1, "```bash\nx\n```"), _call(),
              {"type": EVENT_SUB_CALL, "payload": {}}]
    facts = _facts(e for e in events)
    # The assertions must reach past the FIRST consumer: `compute_run_utilization` runs first and
    # would drain the generator itself, so `main_steps`/`tool_calls` look right either way. What
    # goes wrong silently is everything after it.
    assert facts["main_steps"] == 2 and facts["tool_calls"] == 1
    assert facts["tool_ok"] == 1, "compute_tool_waste saw an exhausted stream"
    assert facts["fence_refused_turns"] == 1, "the fence scan saw an exhausted stream"


def test_a_MEASURED_zero_waste_stays_0_and_does_not_become_None():
    """The other direction of the same distinction, and the one nothing covered.

    `_sum_or_none` filters on `is not None`, not truthiness. `compute_tool_waste` sets
    `wasted_seconds = 0.0` — a MEASURED zero — for a tool that carried durations and had no invalid
    or endpoint outcome, which is the commonest healthy shape there is. A `if v` filter drops those
    `0.0`s and reports `None`, inverting exactly the distinction `_sum_or_none` exists to protect:
    "every tool ran clean and we timed them" would read as "nothing was measured".

    The suite's other seconds test covers `or 0.0` (unmeasured wrongly becoming zero); this covers
    the mirror image, and the mutant survived every other test in the suite."""
    facts = _facts([_step(0), _call("a", ok=True, duration_s=2.0),
                    _call("b", ok=True, duration_s=1.0)])
    assert facts["tool_wasted_seconds"] == 0.0, "a measured zero was reported as unmeasured"
    assert facts["tool_total_seconds"] == 3.0
    assert facts["tool_measured_calls"] == 2
