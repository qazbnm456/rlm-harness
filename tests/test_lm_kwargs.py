"""Per-role LM request parameters (1.12.0): `RLMConfig.main_lm_kwargs` / `sub_lm_kwargs`.

The kit ships the MECHANISM and no vocabulary, because the key that bounds a model's thinking is
server-specific. Measured on one vLLM deployment: `thinking_token_budget` works while
`max_thinking_tokens`, `thinking_budget`, `reasoning_budget` and
`chat_template_kwargs.thinking_budget` are silently ignored, and `reasoning_effort`, the one name
litellm maps across providers, moved reasoning the WRONG way on that model. So what the caller
writes is what reaches the server, and the kit's only vocabulary is on the READ path, for the
trace.

The load-bearing property of the whole feature: **a passthrough cannot report that the server
ignored a key.** Every test here that pins a loud failure (malformed JSON, a refused key, a budget
that can never act) exists because silence is the failure mode this knob is most exposed to.
"""

import contextlib
import json

import pytest

from rlm_harness import _dspy_compat as compat
from rlm_harness.config import RLMConfig

dspy = pytest.importorskip("dspy")

VLLM = {"extra_body": {"thinking_token_budget": 16384}}


def _cfg(**kw):
    return RLMConfig(main_model="main-model", sub_model="sub-model", **kw)


# --- parsing, and every way it is allowed to fail LOUDLY ------------------------------------

def test_both_roles_parse_from_their_own_env_var(monkeypatch):
    monkeypatch.setenv("RLM_MAIN_LM_KWARGS", json.dumps(VLLM))
    monkeypatch.setenv("RLM_SUB_LM_KWARGS", '{"reasoning_effort": "low"}')
    cfg = RLMConfig.from_env()
    assert cfg.main_lm_kwargs == VLLM
    assert cfg.sub_lm_kwargs == {"reasoning_effort": "low"}


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_unset_or_blank_is_None_not_an_empty_dict(monkeypatch, raw):
    """`None` and `{}` differ downstream: `configure` must send NOTHING, and requirement 1 is that
    the default is byte-identical to not having the field."""
    if raw is None:
        monkeypatch.delenv("RLM_MAIN_LM_KWARGS", raising=False)
    else:
        monkeypatch.setenv("RLM_MAIN_LM_KWARGS", raw)
    assert RLMConfig.from_env().main_lm_kwargs is None


def test_malformed_json_raises_naming_the_variable(monkeypatch):
    """A silent `{}` here is the one outcome that must never happen: the kit cannot detect a key the
    SERVER ignored, so it has to be loud about a key it could not even parse."""
    monkeypatch.setenv("RLM_SUB_LM_KWARGS", '{"extra_body": {"thinking_token_budget": 16384}')
    with pytest.raises(ValueError, match="RLM_SUB_LM_KWARGS"):
        RLMConfig.from_env()


@pytest.mark.parametrize("raw", ['[1, 2]', '"a string"', "42", "null"])
def test_valid_json_of_the_wrong_shape_raises_TypeError_naming_the_variable(monkeypatch, raw):
    """TypeError, matching `__post_init__`'s check for the same mistake made in code: one rule,
    enforced the same way whichever door the value came through."""
    monkeypatch.setenv("RLM_MAIN_LM_KWARGS", raw)
    with pytest.raises(TypeError, match="RLM_MAIN_LM_KWARGS"):
        RLMConfig.from_env()


@pytest.mark.parametrize("key", ["model", "api_key", "base_url", "custom_llm_provider", "timeout"])
def test_a_configure_owned_key_is_refused_at_parse_time(key):
    """Refused rather than silently dropped OR silently winning. Nothing records an override of
    these, so a silent one leaves a trace that reads exactly like a run that went somewhere else."""
    with pytest.raises(ValueError, match=key):
        _cfg(main_lm_kwargs={key: "x"})
    with pytest.raises(ValueError, match=key):
        _cfg(sub_lm_kwargs={key: "x"})


def test_the_refusal_names_every_offending_key_and_the_alternative():
    with pytest.raises(ValueError) as exc:
        _cfg(main_lm_kwargs={"base_url": "x", "timeout": 1, "extra_body": {}})
    msg = str(exc.value)
    assert "'base_url'" in msg and "'timeout'" in msg
    assert "RLM_BASE_URL" in msg and "RLM_REQUEST_TIMEOUT" in msg
    assert "extra_body" not in msg, "a legitimate key must not be reported as refused"


def test_a_refused_key_is_refused_through_env_too(monkeypatch):
    monkeypatch.setenv("RLM_MAIN_LM_KWARGS", '{"api_key": "sk-leak"}')
    with pytest.raises(ValueError, match="api_key"):
        RLMConfig.from_env()


def test_max_tokens_is_ALLOWED_and_that_is_the_per_role_cap():
    """The line is not "everything the kit sets", it is whether the TRACE can see the override.
    `max_tokens` is read back off the LM into `budgets`, so a per-role override is self-documenting,
    and is deliberately how a consumer gets a per-role generation cap with no second field."""
    cfg = _cfg(max_tokens=32768, sub_lm_kwargs={"max_tokens": 4096})
    assert cfg.sub_lm_kwargs == {"max_tokens": 4096}


def test_a_non_dict_field_is_rejected_in_code_as_well():
    with pytest.raises(TypeError, match="main_lm_kwargs"):
        _cfg(main_lm_kwargs=[("extra_body", {})])


# --- the READ path: what the trace is allowed to claim --------------------------------------

class _LM:
    def __init__(self, kwargs):
        self.kwargs = kwargs


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"thinking_token_budget": 64}, {"value": 64, "key": "thinking_token_budget"}),
        ({"extra_body": {"thinking_token_budget": 16384}},
         {"value": 16384, "key": "extra_body.thinking_token_budget"}),
        ({"max_thinking_tokens": 8}, {"value": 8, "key": "max_thinking_tokens"}),
        ({"thinking": {"type": "enabled", "budget_tokens": 4096}},
         {"value": 4096, "key": "thinking.budget_tokens"}),
        ({"extra_body": {"thinking": {"budget_tokens": 32}}},
         {"value": 32, "key": "extra_body.thinking.budget_tokens"}),
    ],
)
def test_the_recognised_shapes_report_the_path_they_were_found_at(kwargs, expected):
    """`key` is the dotted PATH, because a top-level litellm parameter and a raw `extra_body` key
    are different mechanisms and a trace reader must not have to guess which one ran."""
    assert compat.applied_thinking_budget(_LM(kwargs)) == expected


@pytest.mark.parametrize("kwargs", [
    {},
    {"max_tokens": 8192},
    {"extra_body": {"enable_thinking": True}},          # bool is an int in Python; not a budget
    {"extra_body": {"thinking_token_budget": "16384"}},  # a string is not a budget
    {"thinking_token_budget": None},
    {"chat_template_kwargs": {"thinking_budget": 4096}},  # measured: silently ignored by vLLM
])
def test_an_unrecognised_shape_reads_as_absent(kwargs):
    """Absent means NOT RECOGNISED, never "no budget was set": the kit owns no wire vocabulary, so
    an unknown key still REACHES the server and is simply not annotated."""
    assert compat.applied_thinking_budget(_LM(kwargs)) is None


def test_the_read_path_returns_no_credential():
    """Named keys only. A passthrough is caller-supplied and may carry anything; `lm.kwargs` also
    carries `api_key` for every LM the kit builds, and a trace is a shipped artifact."""
    found = compat.applied_thinking_budget(
        _LM({"api_key": "sk-secret", "extra_body": {"thinking_token_budget": 16384, "tok": "s3cr3t"}})
    )
    assert found == {"value": 16384, "key": "extra_body.thinking_token_budget"}
    assert "sk-secret" not in json.dumps(found) and "s3cr3t" not in json.dumps(found)


def test_a_non_dict_kwargs_attribute_does_not_raise():
    assert compat.applied_thinking_budget(_LM(None)) is None
    assert compat.applied_thinking_budget(object()) is None


# --- the readers are PUBLIC (1.13.0) -------------------------------------------------------

def test_both_readers_are_importable_from_the_top_level():
    """A consumer records the budget in its own `run_start` meta, which is the only copy that
    survives a run killed before `run_end` is written. Without these it would have to import a
    private name or keep a second key list, and a second list is the drift the read-path design
    exists to avoid."""
    import rlm_harness
    assert rlm_harness.applied_lm_budget(_LM({"max_tokens": 32768})) == {"cap": 32768,
                                                                        "key": "max_tokens"}
    assert rlm_harness.applied_thinking_budget(_LM(VLLM)) == {
        "value": 16384, "key": "extra_body.thinking_token_budget"}


def test_importing_the_package_still_does_not_import_dspy():
    """The readers touch only `lm.kwargs`, so exporting them must not drag dspy into
    `import rlm_harness`. Checked in a FRESH interpreter: this process has already imported dspy."""
    import subprocess
    import sys
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys, rlm_harness; print('dspy' in sys.modules)"],
        capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False", out.stdout


def test_the_readers_agree_with_what_the_trace_records(tmp_path):
    """The point of making them public is that a consumer's OWN record cannot drift from the
    kit's. Same LMs, same numbers, whichever side reads them."""
    cfg = _cfg(interpreter="mock", max_tokens=32768, main_lm_kwargs=VLLM)
    budgets = _run_end(tmp_path, cfg)["budgets"]
    import rlm_harness
    assert rlm_harness.applied_thinking_budget(dspy.settings.lm) == budgets["thinking"]["main"]
    assert rlm_harness.applied_lm_budget(dspy.settings.lm) == budgets["main"]


# --- configure(): the merge is per-ROLE, and the default sends nothing ----------------------

def _configure(cfg, **kw):
    import rlm_harness.runtime as rt
    rt.configure(cfg, **kw)
    from rlm_harness import get_sub_lm
    return dspy.settings.lm, get_sub_lm()


def test_a_main_passthrough_does_not_reach_the_sub_LM():
    main, sub = _configure(_cfg(main_lm_kwargs=VLLM))
    assert compat.applied_thinking_budget(main) == {"value": 16384,
                                                   "key": "extra_body.thinking_token_budget"}
    assert compat.applied_thinking_budget(sub) is None, "the sub role must be untouched"


def test_a_sub_passthrough_does_not_reach_the_main_LM():
    main, sub = _configure(_cfg(sub_lm_kwargs={"extra_body": {"thinking_token_budget": 512}}))
    assert compat.applied_thinking_budget(sub) == {"value": 512,
                                                  "key": "extra_body.thinking_token_budget"}
    assert compat.applied_thinking_budget(main) is None


def test_unset_puts_NO_new_key_on_the_wire():
    """Requirement 1: the default is byte-identical to 1.11.2. Not "an empty extra_body", no key
    at all, because an empty object is still a key some servers parse."""
    main, sub = _configure(_cfg())
    for lm in (main, sub):
        assert "extra_body" not in lm.kwargs
        assert compat.applied_thinking_budget(lm) is None


def test_a_role_kwarg_wins_over_the_shared_one_and_the_trace_shows_it():
    """Per-role `max_tokens` with no second config field, and `applied_lm_budget` reads the
    override off the LM, so the trace reports the cap the call actually carried."""
    main, sub = _configure(_cfg(max_tokens=32768, sub_lm_kwargs={"max_tokens": 4096}))
    assert compat.applied_lm_budget(main)["cap"] == 32768
    assert compat.applied_lm_budget(sub)["cap"] == 4096


# --- the two warnings ----------------------------------------------------------------------

def test_a_passthrough_on_an_INJECTED_role_warns_that_it_did_nothing(caplog):
    """The same class of silence as the `request_timeout_s` warning: the kwargs reach `dspy.LM` and
    nothing else, so on a role this function did not build they do NOTHING."""
    with caplog.at_level("WARNING", logger="rlm_harness.runtime"):
        _configure(_cfg(main_lm_kwargs=VLLM), main_lm=dspy.LM("openai/injected"))
    assert "main_lm_kwargs" in caplog.text and "supplied explicitly" in caplog.text


def test_no_warning_when_the_role_was_built_here(caplog):
    with caplog.at_level("WARNING", logger="rlm_harness.runtime"):
        _configure(_cfg(main_lm_kwargs=VLLM))
    assert "is ignored" not in caplog.text


def test_a_thinking_budget_at_or_above_the_cap_warns(caplog):
    """Requirement 5. A warning, never an error: the combination is INERT, generation stops at the
    smaller number either way, so refusing it would be the kit overreaching."""
    with caplog.at_level("WARNING", logger="rlm_harness.runtime"):
        _configure(_cfg(max_tokens=8192, main_lm_kwargs={"extra_body": {"thinking_token_budget": 8192}}))
    assert "can never act" in caplog.text


def test_a_thinking_budget_below_the_cap_is_silent(caplog):
    """The shape that works: 16384 against a 32768 cap, measured to leave every capped call usable."""
    with caplog.at_level("WARNING", logger="rlm_harness.runtime"):
        _configure(_cfg(max_tokens=32768, main_lm_kwargs=VLLM))
    assert "can never act" not in caplog.text


def test_an_injected_LM_carrying_both_is_warned_about_too(caplog):
    """Read off the BUILT LM, not off `RLMConfig`, so this also covers an LM the caller built."""
    lm = dspy.LM("openai/x", max_tokens=4096, extra_body={"thinking_token_budget": 9999})
    with caplog.at_level("WARNING", logger="rlm_harness.runtime"):
        _configure(_cfg(), main_lm=lm)
    assert "can never act" in caplog.text


# --- the trace ----------------------------------------------------------------------------

class _StubRLM:
    """Stands in for `dspy.RLM`; `sub_lm` is assignable because `arun` rebinds it."""

    def __init__(self):
        self.sub_lm = None

    async def aforward(self, *args, **kwargs):
        return dspy.Prediction(answer="done")


def _run_end(tmp_path, cfg, **configure_kw):
    import asyncio

    import rlm_harness.runtime as rt
    from rlm_harness.task import RLMTask
    from rlm_harness.trace import TraceRecorder

    rt.configure(cfg, **configure_kw)

    class T(RLMTask):
        signature = "context: str -> answer: str"
        output_field = "answer"

        def _build_rlm(self):
            return _StubRLM()

    path = tmp_path / "t.jsonl"
    task = T(sub_lm=dspy.LM("openai/sub", **(cfg.sub_lm_kwargs or {})))
    with TraceRecorder(str(path), run_id="r"), contextlib.suppress(Exception):
        asyncio.run(task.arun(context="x"))
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return next(e["payload"] for e in events if e["type"] == "run_end")


def test_the_thinking_budget_reaches_the_trace_per_role(tmp_path):
    """Requirement 4. Without it a corpus cannot tell the populations apart, which is the whole
    reason the runaway was diagnosable in the first place."""
    cfg = _cfg(interpreter="mock", max_tokens=32768, main_lm_kwargs=VLLM,
               sub_lm_kwargs={"extra_body": {"thinking_token_budget": 2048}})
    budgets = _run_end(tmp_path, cfg)["budgets"]
    assert budgets["thinking"] == {
        "main": {"value": 16384, "key": "extra_body.thinking_token_budget"},
        "sub": {"value": 2048, "key": "extra_body.thinking_token_budget"},
    }


def test_thinking_is_ABSENT_from_budgets_when_no_role_carries_one(tmp_path):
    """Absent-or-populated, never empty: the same optionality `test_contract.py` pins for every
    other payload addition, so a reader can tell "not recorded" from "recorded, and none"."""
    budgets = _run_end(tmp_path, _cfg(interpreter="mock"))["budgets"]
    assert "thinking" not in budgets


def test_the_established_main_and_sub_shapes_are_UNCHANGED(tmp_path):
    """Why `thinking` is its own key. `budgets.main`'s PRESENCE means "this role carried a token
    cap" and the guide says so, so a `main` holding only a thinking budget would change what an
    existing reader's `budgets["main"]["cap"]` may assume. This pins the shape that must not move."""
    cfg = _cfg(interpreter="mock", max_tokens=None, main_lm_kwargs=VLLM)
    budgets = _run_end(tmp_path, cfg)["budgets"]
    assert "main" not in budgets, "no token cap was set, so the role key must stay absent"
    assert budgets["thinking"]["main"]["value"] == 16384
