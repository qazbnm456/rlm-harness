import types

import pytest
from pydantic import BaseModel

import rlm_harness
from rlm_harness._retry import (
    RLMTaskError,
    _short_error,
    coerce_output,
    run_with_retry,
    short_error,
)


class Finding(BaseModel):
    title: str
    severity: str


def pred(**fields):
    """A stand-in for a dspy.Prediction: attribute access over fields."""
    return types.SimpleNamespace(**fields)


# ---- coerce_output -------------------------------------------------------

def test_coerce_passthrough_when_no_model():
    assert coerce_output("anything", None) == "anything"


def test_coerce_from_instance():
    f = Finding(title="t", severity="high")
    assert coerce_output(f, Finding) is f


def test_coerce_from_dict():
    out = coerce_output({"title": "t", "severity": "low"}, Finding)
    assert isinstance(out, Finding) and out.severity == "low"


def test_coerce_from_json_string():
    out = coerce_output('{"title": "t", "severity": "med"}', Finding)
    assert isinstance(out, Finding) and out.severity == "med"


def test_coerce_invalid_raises():
    with pytest.raises(Exception):
        coerce_output('{"title": "t"}', Finding)  # missing severity


# ---- run_with_retry ------------------------------------------------------

async def test_success_first_try():
    async def runner():
        return pred(finding={"title": "t", "severity": "high"})

    out = await run_with_retry(runner, output_field="finding", output_model=Finding)
    assert isinstance(out, Finding) and out.title == "t"


async def test_retries_then_succeeds():
    calls = {"n": 0}

    async def runner():
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("transient model error")
        return pred(finding={"title": "ok", "severity": "low"})

    out = await run_with_retry(
        runner, output_field="finding", output_model=Finding, max_retries=3
    )
    assert out.title == "ok"
    assert calls["n"] == 2


async def test_validation_failure_triggers_retry_then_exhausts():
    calls = {"n": 0}

    async def runner():
        calls["n"] += 1
        return pred(finding={"title": "t"})  # always invalid (no severity)

    with pytest.raises(RLMTaskError):
        await run_with_retry(
            runner, output_field="finding", output_model=Finding, max_retries=2
        )
    assert calls["n"] == 2  # consumed the full budget


async def test_missing_output_field_retries():
    async def runner():
        return pred(other="x")

    with pytest.raises(RLMTaskError):
        await run_with_retry(
            runner, output_field="finding", output_model=Finding, max_retries=1
        )


async def test_no_model_returns_raw_field():
    async def runner():
        return pred(answer="plain text")

    out = await run_with_retry(runner, output_field="answer")
    assert out == "plain text"


# ---- non_retryable: a caller-chosen exception propagates verbatim, no retry --------------


class _Cancelled(RuntimeError):
    """Stand-in for `sandbox.SandboxCancelled` — `_retry.py` stays dspy/sandbox-free."""


async def test_non_retryable_exception_is_never_retried_and_never_wrapped():
    calls = {"n": 0}
    original = _Cancelled("stop now")

    async def runner():
        calls["n"] += 1
        raise original

    with pytest.raises(_Cancelled) as ei:
        await run_with_retry(
            runner,
            output_field="finding",
            output_model=Finding,
            max_retries=3,
            non_retryable=(_Cancelled,),
        )
    assert ei.value is original  # the ORIGINAL object, never wrapped in RLMTaskError
    assert calls["n"] == 1  # no retry attempt happened


async def test_an_exception_outside_non_retryable_is_retried_as_before():
    calls = {"n": 0}

    async def runner():
        calls["n"] += 1
        raise RuntimeError("transient")

    with pytest.raises(RLMTaskError):
        await run_with_retry(
            runner,
            output_field="finding",
            output_model=Finding,
            max_retries=3,
            non_retryable=(_Cancelled,),
        )
    assert calls["n"] == 3  # today's behavior for anything NOT in the allowlist: unchanged


async def test_default_non_retryable_matches_nothing():
    calls = {"n": 0}

    async def runner():
        calls["n"] += 1
        raise _Cancelled("would-be-cancelled, but no allowlist was passed")

    with pytest.raises(RLMTaskError):
        await run_with_retry(
            runner, output_field="finding", output_model=Finding, max_retries=2
        )
    assert calls["n"] == 2  # the default () matches nothing — retried like any other exception


# ---- is_fast_fail: a dynamic predicate gets the same treatment as non_retryable -----------


async def test_is_fast_fail_match_propagates_verbatim_with_no_retry():
    calls = {"n": 0}
    original = RuntimeError("dspy says this one is hopeless")

    async def runner():
        calls["n"] += 1
        raise original

    with pytest.raises(RuntimeError) as ei:
        await run_with_retry(
            runner,
            output_field="finding",
            output_model=Finding,
            max_retries=3,
            is_fast_fail=lambda exc: True,
        )
    assert ei.value is original  # the ORIGINAL object, never wrapped in RLMTaskError
    assert calls["n"] == 1  # no retry attempt happened


async def test_is_fast_fail_non_match_is_retried_as_before():
    calls = {"n": 0}

    async def runner():
        calls["n"] += 1
        raise RuntimeError("transient")

    with pytest.raises(RLMTaskError):
        await run_with_retry(
            runner,
            output_field="finding",
            output_model=Finding,
            max_retries=3,
            is_fast_fail=lambda exc: False,
        )
    assert calls["n"] == 3  # the predicate said no every time — unchanged retry behavior


async def test_default_is_fast_fail_is_none_and_never_fires():
    calls = {"n": 0}

    async def runner():
        calls["n"] += 1
        raise RuntimeError("would-be-fast-failed, but no predicate was passed")

    with pytest.raises(RLMTaskError):
        await run_with_retry(
            runner, output_field="finding", output_model=Finding, max_retries=2
        )
    assert calls["n"] == 2  # the default None never fires — retried like any other exception


async def test_non_retryable_type_match_wins_over_is_fast_fail():
    """`non_retryable` is checked first (a cheaper, static type match); a predicate that would
    also match must not change that this still propagates via the type-based `except` clause,
    not the predicate branch — asserted by NEVER calling the predicate at all."""
    calls = {"n": 0}
    predicate_calls = {"n": 0}
    original = _Cancelled("stop now")

    async def runner():
        calls["n"] += 1
        raise original

    def is_fast_fail(exc):
        predicate_calls["n"] += 1
        return True

    with pytest.raises(_Cancelled) as ei:
        await run_with_retry(
            runner,
            output_field="finding",
            output_model=Finding,
            max_retries=3,
            non_retryable=(_Cancelled,),
            is_fast_fail=is_fast_fail,
        )
    assert ei.value is original
    assert calls["n"] == 1
    assert predicate_calls["n"] == 0  # non_retryable's `except` clause caught it first


# ---- short_error: bound the logged exception ------------------------------

def test_short_error_leaves_a_small_message_intact():
    assert short_error(ValueError("nope")) == "ValueError: nope"


def test_short_error_caps_a_huge_exception_and_keeps_head_and_tail():
    # dspy's AdapterParseError embeds the ENTIRE raw LM completion; a degenerate model makes it
    # thousands of lines. short_error keeps the head (type + start) and tail, elides the middle.
    huge = "HEAD-marker " + ("loop " * 5000) + "TAIL-marker"
    out = short_error(RuntimeError(huge))
    assert len(out) < 700                                # bounded, not the ~25k-char original
    assert out.startswith("RuntimeError: HEAD-marker")   # head kept
    assert out.endswith("TAIL-marker")                   # tail kept
    assert "chars elided" in out


def test_short_error_honours_an_explicit_limit():
    out = short_error(RuntimeError("x" * 500), limit=80)
    assert out.startswith("RuntimeError: xxx")
    assert "chars elided" in out
    assert len(out) < 500


def test_short_error_stays_bounded_at_every_limit():
    """The frozen contract promises a length bound, and it is the ONE property the function
    exists for. At limit<=1 the head/tail split left tail==0, and `text[-0:]` slices the WHOLE
    string — so the smallest budgets produced the LONGEST output."""
    huge = "x" * 5000
    for limit in (0, 1, 2, 5, 30, 600):
        out = short_error(RuntimeError(huge), limit=limit)
        assert len(out) < len(huge), f"limit={limit} amplified instead of bounding"
        assert len(out) <= limit + 60, f"limit={limit} exceeded the bound: {len(out)}"


def test_short_error_never_raises_on_a_hostile_exception():
    """It renders exceptions from anywhere, including a model-shaped payload whose __str__ is
    itself broken. Blowing up inside the error path would replace a diagnosable failure with an
    undiagnosable one."""
    class _Hostile(RuntimeError):
        def __str__(self):
            raise ValueError("no string for you")

    out = short_error(_Hostile())
    assert out.startswith("_Hostile: ")
    assert "unprintable" in out


def test_short_error_is_public_and_the_old_private_name_still_resolves():
    """Promoted in 1.5.0 because two independent consumers had reached into `_retry` for it.
    The private spelling must keep working or the promotion breaks the callers that motivated
    it — it is an alias, not a copy."""
    assert rlm_harness.short_error is short_error
    assert "short_error" in rlm_harness.__all__
    assert _short_error is short_error


async def test_retry_log_does_not_flood_on_huge_exception(caplog):
    # Regression: a failed attempt must not dump the full (possibly enormous) exception message —
    # that is what floods the terminal when the root model degenerates into a repetition loop.
    flood = "loop " * 5000
    async def runner():
        raise RuntimeError(f"Adapter failed. LM Response: {flood} end-of-error")

    with caplog.at_level("WARNING", logger="rlm_harness._retry"), pytest.raises(RLMTaskError):
        await run_with_retry(runner, output_field="finding", max_retries=1)

    msg = caplog.records[-1].getMessage()
    assert len(msg) < 800                    # bounded, not the ~25k-char flood
    assert "Adapter failed" in msg           # head kept
    assert "end-of-error" in msg             # tail kept
    assert "chars elided" in msg
