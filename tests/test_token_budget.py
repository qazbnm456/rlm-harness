"""Token budgets and per-attempt usage in the trace (1.10.0).

A truncated completion and a malformed one raise the SAME exception type, so a consumer looking at
`AdapterParseError` cannot tell which it had. dspy detects truncation (`finish_reason == "length"`)
and only `logger.warning`s it. What survives is the token COUNT, and `completion_tokens ==
max_tokens` carries the same fact plus one more: it shows a turn APPROACHING the cap, where a
boolean fires only after death.
"""

import contextlib
import json

import pytest

from rlm_harness import _dspy_compat as compat

dspy = pytest.importorskip("dspy")


# --- applied caps, read off the LM ---------------------------------------------------------

def test_the_cap_is_found_under_either_key_and_says_which():
    """dspy rewrites `max_tokens` to `max_completion_tokens` in `LM._get_initial_kwargs` for OpenAI
    reasoning models, so a reader of `max_tokens` alone gets `None` for exactly the thinking-model
    case this feature exists to explain."""
    plain = compat.applied_lm_budget(dspy.LM("openai/gpt-4o-mini", max_tokens=4096))
    assert plain == {"cap": 4096, "key": "max_tokens"}

    reasoning = compat.applied_lm_budget(dspy.LM("openai/o3", max_tokens=16384))
    assert reasoning == {"cap": 16384, "key": "max_completion_tokens"}, (
        "the reasoning-model cap was not found -- reading `max_tokens` alone returns None here"
    )


def test_no_cap_reads_absent_not_zero():
    """The key is PRESENT with value `None` when no cap was set, so this cannot separate
    'never set' from 'explicitly None'. Both are absent, which is honest either way."""
    assert compat.applied_lm_budget(dspy.LM("openai/gpt-4o-mini")) is None


def test_the_shim_returns_no_credential():
    """`lm.kwargs` carries `api_key` for every LM the kit builds. Named keys ONLY."""
    lm = dspy.LM("openai/gpt-4o-mini", max_tokens=4096, api_key="SENTINEL-CREDENTIAL")
    assert "SENTINEL-CREDENTIAL" in json.dumps(lm.kwargs), "precondition: the key is in kwargs"
    assert "SENTINEL-CREDENTIAL" not in json.dumps(compat.applied_lm_budget(lm))


# --- the usage tracker ---------------------------------------------------------------------

class _Tracker:
    """The shape `UsageTracker` presents to us: `usage_data`, model-name STRING keys, append-only."""

    def __init__(self, data=None):
        self.usage_data = data if data is not None else {}

    def add(self, model, completion_tokens):
        self.usage_data.setdefault(model, []).append(
            {"prompt_tokens": 10, "completion_tokens": completion_tokens}
        )


def test_a_reused_trackers_pre_existing_calls_are_not_counted_as_ours():
    """Reuse avoids shadowing a consumer's tracker, but their calls are in it. Read a SLICE."""
    tracker = _Tracker()
    tracker.add("openai/main", 11)                      # the consumer's, before our run
    baseline = compat.usage_baseline(tracker)
    tracker.add("openai/main", 22)                      # ours
    tracker.add("openai/tool", 5)                       # ours, on a model with NO baseline

    ours = compat.usage_since(tracker, baseline)
    assert [c["completion_tokens"] for c in ours["openai/main"]] == [22], "counted their call too"
    assert [c["completion_tokens"] for c in ours["openai/tool"]] == [5], (
        "a model first seen mid-scope must not need a baseline entry"
    )


def test_a_model_absent_from_the_tracker_reads_empty_without_inserting_a_key():
    """`usage_data` is a `defaultdict(list)` in dspy, so `data[missing]` returns `[]` silently AND
    inserts a bogus key. `.get` is what keeps a read from mutating what it reads."""
    from collections import defaultdict

    tracker = _Tracker(defaultdict(list))
    tracker.add("openai/main", 7)
    before = len(tracker.usage_data)
    assert compat.usage_since(tracker, {"openai/main": 1}) == {}
    assert len(tracker.usage_data) == before, "the read inserted a key"


def test_an_outer_tracker_is_REUSED_not_shadowed():
    """dspy creates a tracker only when none is installed, so installing unconditionally hands a
    consumer's own `with dspy.track_usage(): ...` zero entries for everything inside."""
    from dspy.utils.usage_tracker import track_usage

    with track_usage() as outer, compat.usage_tracking() as inner:
        assert inner is outer, "we shadowed the caller's tracker instead of reusing it"


def test_a_tracker_is_installed_when_the_caller_has_none():
    with compat.usage_tracking() as tracker:
        assert tracker is not None
        assert compat.current_usage_tracker() is tracker


# --- per-attempt usage on the RUN, including the run that dies ------------------------------

class _StubRLM:
    """Stands in for `dspy.RLM`. `sub_lm` is assignable because `arun` rebinds it."""

    def __init__(self, script, tracker_calls):
        # One entry per attempt. "bad" returns a prediction MISSING the output field, so the
        # attempt is captured (`captured["prediction"]` is assigned) and then fails validation --
        # which is what puts an EARLIER attempt's turns in the trace when a LATER one raises.
        self._script = list(script)          # "ok" | "bad" | "raise"
        self._calls = tracker_calls          # a list the stub appends usage rows to
        self.sub_lm = None
        self.attempts = 0

    async def aforward(self, *args, **kwargs):
        step = self._script[self.attempts]
        self.attempts += 1
        self._calls(self.attempts)
        if step == "raise":
            raise ValueError(f"attempt-{self.attempts}-died")
        if step == "bad":
            return dspy.Prediction(reasoning=f"ATTEMPT-{self.attempts}")
        return dspy.Prediction(answer=f"ATTEMPT-{self.attempts}")


def _run(tmp_path, script, max_retries, api_key=None):
    """Drive a real `arun` over `_StubRLM`, returning the parsed run_end payload."""
    import asyncio

    import rlm_harness.runtime as rt
    from rlm_harness.config import RLMConfig
    from rlm_harness.task import RLMTask
    from rlm_harness.trace import TraceRecorder

    # A cap on the MAIN LM that differs from the sub-LM's, so the per-role read is exercised.
    main = dspy.LM("openai/main", max_tokens=8192, **({"api_key": api_key} if api_key else {}))
    rt.configure(RLMConfig(main_model="x", sub_model="x", interpreter="mock", observe=False),
                 main_lm=main, sub_lm=main)

    def note_call(n):
        tracker = compat.current_usage_tracker()
        if tracker is not None:
            tracker.usage_data.setdefault("openai/stub", []).append(
                {"prompt_tokens": 1, "completion_tokens": 100 * n}
            )

    class T(RLMTask):
        signature = "context: str -> answer: str"
        output_field = "answer"

        def _build_rlm(self):
            return _StubRLM(script, note_call)

    task = T(max_retries=max_retries, sub_lm=dspy.LM("openai/sub", max_tokens=2048))
    path = tmp_path / "t.jsonl"
    with TraceRecorder(str(path), run_id="r"), contextlib.suppress(Exception):
        asyncio.run(task.arun(context="x"))
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return next(e["payload"] for e in events if e["type"] == "run_end")


def _trace_of(tmp_path, script, max_retries, api_key=None):
    """Like `_run`, but returns the trace PATH so a test can assert on the bytes written."""
    _run(tmp_path, script, max_retries, api_key=api_key)
    return tmp_path / "t.jsonl"


def test_a_run_whose_FINAL_attempt_raises_still_records_that_attempts_usage(tmp_path):
    """The case the whole feature exists for. `captured["prediction"]` is assigned only after a
    SUCCESSFUL `aforward`, so the turns in the trace come from attempt 1 while attempt 2 is the one
    that died -- and attempt 2's usage is the number a reader needs."""
    payload = _run(tmp_path, ["bad", "raise"], max_retries=2)
    usage = payload["usage"]
    assert [e["attempt"] for e in usage] == [0, 1]
    assert [e["turns_recorded"] for e in usage] == [True, False]
    assert usage[1]["calls"]["openai/stub"][0]["completion_tokens"] == 200, (
        "the fatal attempt's usage was discarded"
    )


def test_three_attempts_flag_exactly_the_one_whose_turns_survived(tmp_path):
    """THE case that separates the rule from the mutation. Attempts 1 and 2 both produce a
    prediction, so a flag stamped at attempt end marks BOTH -- but `captured["prediction"]` is
    last-writer-wins, so only attempt 2's turns are in the trace. Neither of the other two run
    shapes catches this: with one predicting attempt the naive stamp is accidentally right, and
    with none it marks none."""
    payload = _run(tmp_path, ["bad", "bad", "raise"], max_retries=3)
    flagged = [e["attempt"] for e in payload["usage"] if e["turns_recorded"]]
    assert flagged == [1], f"expected exactly attempt 1 (0-based) flagged, got {flagged}"


def test_a_run_that_never_produced_a_prediction_records_every_attempt(tmp_path):
    """`main_steps == 0` -- the shape of the incident that prompted this. No attempt is flagged,
    and no attempt's usage is dropped for lacking turns to align to."""
    payload = _run(tmp_path, ["raise", "raise"], max_retries=2)
    usage = payload["usage"]
    assert [e["attempt"] for e in usage] == [0, 1]
    assert not any(e["turns_recorded"] for e in usage)


def test_budgets_record_the_cap_that_was_APPLIED(tmp_path):
    """Read off the LM, not `RLMConfig`: an injected LM is used verbatim, so the configured cap can
    be one the call never used."""
    payload = _run(tmp_path, ["ok"], max_retries=1)
    assert payload["budgets"]["sub"] == {"cap": 2048, "key": "max_tokens"}


def test_budgets_and_usage_are_ABSENT_when_nothing_staged_them(tmp_path):
    """A legacy reader must be able to tell 'not recorded' from 'recorded, and empty' -- the same
    optionality `test_contract.py` pins for the 1.6.0 payload additions."""
    from rlm_harness.trace import TraceRecorder

    path = tmp_path / "bare.jsonl"
    with TraceRecorder(str(path), run_id="r"):
        pass
    end = [json.loads(x) for x in path.read_text().splitlines() if x.strip()][-1]
    assert "budgets" not in end["payload"] and "usage" not in end["payload"]


def test_no_credential_reaches_the_TRACE_FILE(tmp_path):
    """The shim-level test above cannot see a leak introduced at the task level, and a trace is a
    shipped artifact -- replay, the dataset exporters, every consumer's corpus. So assert on the
    bytes that actually get written."""
    path = _trace_of(tmp_path, ["ok"], max_retries=1, api_key="SENTINEL-CREDENTIAL")
    assert "SENTINEL-CREDENTIAL" not in path.read_text()


def test_the_iteration_caps_and_the_lossy_fallback_are_recorded(tmp_path):
    """`max_output_chars` is a THIRD truncation mechanism (dspy head+tail-caps each REPL output),
    independent of the token cap -- a reader diagnosing "truncation" has to be able to rule it out.
    And `dropped` says whether `_build_rlm`'s `except TypeError` fired, because that path reverts
    every cap to dspy's default and the configured numbers would otherwise read as applied."""
    payload = _run(tmp_path, ["ok"], max_retries=1)
    iters = payload["budgets"]["iterations"]
    assert set(iters) == {"max_iterations", "max_llm_calls", "max_output_chars", "dropped"}
    assert iters["dropped"] is False, "no fallback fired, so the caps above are real"


def _run_real_build(tmp_path, rlm_factory, config_kwargs, task=None, name='real'):
    """Drive `arun` through the REAL `_build_rlm`, so its `except TypeError` fallback can fire.

    `_run` above overrides `_build_rlm` wholesale, which means the branch that sets the dropped
    flag never executes there -- two mutations silencing that flag passed the whole suite.
    """
    import asyncio

    import rlm_harness.runtime as rt
    from rlm_harness.config import RLMConfig
    from rlm_harness.task import RLMTask
    from rlm_harness.trace import TraceRecorder

    lm = dspy.LM("openai/main", max_tokens=8192)
    rt.configure(RLMConfig(main_model="x", sub_model="x", interpreter="mock", observe=False,
                           **config_kwargs), main_lm=lm, sub_lm=lm)

    if task is None:
        class T(RLMTask):
            signature = "context: str -> answer: str"
            output_field = "answer"

        task = T(max_retries=1, interpreter=object())
    path = tmp_path / f"{name}.jsonl"
    with (
        pytest.MonkeyPatch.context() as mp,
        TraceRecorder(str(path), run_id="r"),
        contextlib.suppress(Exception),
    ):
        mp.setattr(dspy, "RLM", rlm_factory)
        asyncio.run(task.arun(context="x"))
    events = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    return next(e["payload"] for e in events if e["type"] == "run_end"), task


class _OkRLM:
    def __init__(self, *a, **kw):
        self.sub_lm = None

    async def aforward(self, *a, **kw):
        return dspy.Prediction(answer="ok")


def _rlm_rejecting_budget_kwargs(*args, **kwargs):
    """dspy having renamed the budget kwargs again -- exactly what `_build_rlm` guards against."""
    for name in ("max_iters", "max_iterations", "max_llm_calls", "max_output_chars"):
        if name in kwargs:
            raise TypeError(f"RLM.__init__() got an unexpected keyword argument '{name}'")
    return _OkRLM()


def test_the_iteration_caps_carry_the_CONFIGURED_values(tmp_path):
    """Asserting the key set alone let a mutant blank all three to `None` and pass -- which would
    ship `max_output_chars: null`, read as 'no head+tail cap', ruling out the very mechanism the
    field was added to let a reader rule out."""
    payload, _ = _run_real_build(
        tmp_path, _OkRLM,
        {"max_iterations": 7, "max_llm_calls": 13, "max_output_chars": 4321},
    )
    assert payload["budgets"]["iterations"] == {
        "max_iterations": 7, "max_llm_calls": 13, "max_output_chars": 4321, "dropped": False,
    }


def test_dropped_is_TRUE_when_the_real_TypeError_fallback_fires(tmp_path):
    """The only test that runs the real `except TypeError`. Asserting `dropped is False` elsewhere
    holds even when the whole mechanism is deleted, so it defended nothing: two mutations that
    silence the flag passed all 966 tests. Here dspy rejects the budget kwargs, every cap reverts
    to dspy's own default, and the trace has to disown the configured numbers rather than present
    them as applied."""
    payload, _ = _run_real_build(
        tmp_path, _rlm_rejecting_budget_kwargs,
        {"max_iterations": 7, "max_llm_calls": 13, "max_output_chars": 4321},
    )
    iters = payload["budgets"]["iterations"]
    assert iters["dropped"] is True, "the lossy fallback fired and the trace did not say so"
    assert iters["max_iterations"] == 7, "the configured numbers stay, disowned by the flag"


def test_the_dropped_flag_does_not_leak_into_a_later_build(tmp_path):
    """`self._budget_caps_dropped` is per-instance state. Set once and never reset, a later clean
    build would still claim the caps were dropped."""
    cfg = {"max_iterations": 7, "max_llm_calls": 13, "max_output_chars": 4321}
    first, task = _run_real_build(tmp_path, _rlm_rejecting_budget_kwargs, cfg, name="a")
    assert first["budgets"]["iterations"]["dropped"] is True

    # The SAME instance, rebuilding cleanly. Without the per-build reset the flag survives.
    second, _ = _run_real_build(tmp_path, _OkRLM, cfg, task=task, name="b")
    assert second["budgets"]["iterations"]["dropped"] is False, "the flag leaked into a later build"
