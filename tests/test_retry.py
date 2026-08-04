import types

import pytest
from pydantic import BaseModel

from rlm_kit._retry import RLMTaskError, _short_error, coerce_output, run_with_retry


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


# ---- _short_error: bound the logged exception -----------------------------

def test_short_error_leaves_a_small_message_intact():
    assert _short_error(ValueError("nope")) == "ValueError: nope"


def test_short_error_caps_a_huge_exception_and_keeps_head_and_tail():
    # dspy's AdapterParseError embeds the ENTIRE raw LM completion; a degenerate model makes it
    # thousands of lines. _short_error keeps the head (type + start) and tail, elides the middle.
    huge = "HEAD-marker " + ("loop " * 5000) + "TAIL-marker"
    out = _short_error(RuntimeError(huge))
    assert len(out) < 700                                # bounded, not the ~25k-char original
    assert out.startswith("RuntimeError: HEAD-marker")   # head kept
    assert out.endswith("TAIL-marker")                   # tail kept
    assert "chars elided" in out


async def test_retry_log_does_not_flood_on_huge_exception(caplog):
    # Regression: a failed attempt must not dump the full (possibly enormous) exception message —
    # that is what floods the terminal when the root model degenerates into a repetition loop.
    flood = "loop " * 5000
    async def runner():
        raise RuntimeError(f"Adapter failed. LM Response: {flood} end-of-error")

    with caplog.at_level("WARNING", logger="rlm_kit._retry"), pytest.raises(RLMTaskError):
        await run_with_retry(runner, output_field="finding", max_retries=1)

    msg = caplog.records[-1].getMessage()
    assert len(msg) < 800                    # bounded, not the ~25k-char flood
    assert "Adapter failed" in msg           # head kept
    assert "end-of-error" in msg             # tail kept
    assert "chars elided" in msg
