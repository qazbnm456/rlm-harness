"""What the LMs ``configure()`` builds send, measured on a local OpenAI-compatible server.

Each test runs the kit's real path (``configure`` building the LM, an ``RLMTask`` on the ``mock``
interpreter, dspy's engines) against ``conftest.FakeOpenAIServer`` and asserts on the requests the
server received: the path, the model id, the token-cap field, the response format, and that every
server option either arrives or is refused at ``configure()``, never dropped on the way.
"""

from __future__ import annotations

import asyncio

import pytest

dspy = pytest.importorskip("dspy")

import rlm_harness
from rlm_harness import RLMConfig, RLMTask, TraceRecorder, _dspy_compat, load_events
from rlm_harness.trace import EVENT_RLM_SETUP

SERVED = "qwen/qwen3-next"
PATH = "/v1/chat/completions"
#: The field each dialect's native preset carries the token cap in, as measured on dspy 3.5.0.
CAP_FIELD = {
    "openai": "max_completion_tokens",
    "vllm": "max_tokens",
    "ollama": "max_tokens",
    "openrouter": "max_tokens",
    "groq": "max_tokens",
    "together_ai": "max_completion_tokens",
}
CAP_FIELDS = frozenset(CAP_FIELD.values())
#: The native provider id ``rlm_setup.lm.dialect`` records for each dialect.
LM15_DIALECT = {"openai": "openai-chat", "vllm": "vllm"}


class _Task(RLMTask):
    signature = "q: str -> answer: str"
    output_field = "answer"


def _configure(server, **overrides):
    cfg = {"main_model": SERVED, "sub_model": SERVED, "base_url": server.base_url, "api_key": "k",
           "interpreter": "mock", "max_iterations": 2}
    cfg.update(overrides)
    rlm_harness.configure(RLMConfig(**cfg))


def _run(tmp_path):
    path = str(tmp_path / "t.jsonl")
    with TraceRecorder(path, run_id="r"):
        out = asyncio.run(_Task().arun(q="hi"))
    return out, load_events(path)


def _setup(events):
    return next(e for e in events if e["type"] == EVENT_RLM_SETUP)["payload"]


def _schema_fields(body):
    return list(body["response_format"]["json_schema"]["schema"]["properties"])


# ---- the planner request, per dialect and adapter ---------------------------------------------


@pytest.mark.parametrize("adapter", ["json", "chat", "default"])
@pytest.mark.parametrize("dialect", ["openai", "vllm"])
def test_the_planner_request_on_each_dialect_and_adapter(fake_openai, tmp_path, dialect, adapter):
    _configure(fake_openai, endpoint_dialect=dialect, adapter=adapter)
    out, events = _run(tmp_path)

    assert out == "42"
    requests = fake_openai.requests
    assert len(requests) == 3, "two planner turns, then the extract"
    for req in requests:
        body = req["body"]
        assert req["path"] == PATH
        assert body["model"] == SERVED
        assert body[CAP_FIELD[dialect]] == 8192
        assert not (CAP_FIELDS - {CAP_FIELD[dialect]}) & set(body)
        assert "reasoning_effort" not in body and "reasoning" not in body
    planner, extract = [r["body"] for r in requests[:2]], requests[2]["body"]
    if adapter == "json":
        for body in planner:
            assert body["response_format"]["type"] == "json_schema"
            assert _schema_fields(body) == ["plan", "code"]
        assert _schema_fields(extract) == ["answer"]
    else:
        marker = "[[ ## plan ## ]]" if adapter == "chat" else "<plan>"
        for body in (*planner, extract):
            assert "response_format" not in body
        for body in planner:
            assert marker in body["messages"][0]["content"]

    setup = _setup(events)
    assert setup["native_reasoning"] is False
    lm = setup["lm"]["main"]
    assert (lm["engine"], lm["api_family"], lm["dialect"]) == ("lm15", "chat", LM15_DIALECT[dialect])
    assert lm["model"] == _dspy_compat.endpoint_model(dialect, SERVED)


@pytest.mark.parametrize("dialect", sorted(CAP_FIELD))
def test_each_dialect_sends_the_cap_in_its_own_field(fake_openai, dialect):
    _configure(fake_openai, endpoint_dialect=dialect, max_tokens=321)
    dspy.settings.lm("hi")
    (req,) = fake_openai.requests
    assert req["path"] == PATH
    assert req["body"]["model"] == SERVED
    assert req["body"][CAP_FIELD[dialect]] == 321
    assert not (CAP_FIELDS - {CAP_FIELD[dialect]}) & set(req["body"])


def test_a_served_name_that_starts_with_openai_is_written_with_the_prefix_twice(fake_openai):
    _configure(fake_openai, main_model="openai/openai/gpt-oss-120b", sub_model="openai/openai/gpt-oss-120b")
    dspy.settings.lm("hi")
    assert fake_openai.requests[0]["body"]["model"] == "openai/gpt-oss-120b"


# ---- the explicit plan, and the native mode that drops it ---------------------------------------


@pytest.mark.parametrize("native", [False, True])
def test_native_reasoning_decides_whether_a_reasoning_model_keeps_its_plan(fake_openai, tmp_path,
                                                                           native):
    """`o3-mini` is a model dspy's catalog marks as reasoning-capable. By default the kit keeps the
    planner's `plan` field and sends no reasoning effort of its own; `native_reasoning=True` hands
    the field to dspy, which drops it from the schema and asks for the model's reasoning instead."""
    fake_openai.reasoning_content = "thinking it through"
    _configure(fake_openai, main_model="o3-mini", sub_model="o3-mini", max_tokens=16000,
               native_reasoning=native)
    out, events = _run(tmp_path)

    assert out == "42"
    planner = [r["body"] for r in fake_openai.requests[:2]]
    for body in planner:
        if native:
            assert _schema_fields(body) == ["code"]
            assert "reasoning_effort" in body
        else:
            assert _schema_fields(body) == ["plan", "code"]
            assert "reasoning_effort" not in body
    assert _setup(events)["native_reasoning"] is native


# ---- server options: carried, or refused at configure() -----------------------------------------


@pytest.mark.parametrize("dialect", ["openai", "vllm"])
@pytest.mark.parametrize(("option", "wire"), [
    ({"temperature": 0.3, "top_p": 0.9, "seed": 7, "stop": ["\n\n"]},
     {"temperature": 0.3, "top_p": 0.9, "seed": 7, "stop": ["\n\n"]}),
    ({"reasoning_effort": "low"}, {"reasoning_effort": "low"}),
    ({"reasoning_effort": "none"}, {"reasoning_effort": "none"}),
])
def test_an_option_the_native_route_carries_arrives(fake_openai, dialect, option, wire):
    _configure(fake_openai, endpoint_dialect=dialect, main_lm_kwargs=option)
    dspy.settings.lm("hi")
    body = fake_openai.requests[0]["body"]
    assert {k: body.get(k) for k in wire} == wire


@pytest.mark.parametrize("dialect", ["openai", "vllm"])
@pytest.mark.parametrize(("option", "named"), [
    ({"top_k": 40}, "top_k"),
    ({"extra_body": {"thinking_token_budget": 64}}, "extra_body"),
    ({"thinking_token_budget": 64}, "thinking_token_budget"),
    ({"chat_template_kwargs": {"enable_thinking": False}}, "chat_template_kwargs"),
    ({"enable_thinking": False}, "enable_thinking"),
    ({"min_p": 0.05}, "min_p"),
    ({"repetition_penalty": 1.1}, "repetition_penalty"),
])
def test_an_option_the_native_route_would_not_send_is_refused_at_configure(fake_openai, dialect,
                                                                            option, named):
    """Never dropped silently: `top_k` is one dspy's native engine would leave out of the request with
    only a note, and the rest it would refuse at call time. All of them stop `configure()`, naming the
    option and the route that carries it, before any request is sent."""
    with pytest.raises(ValueError, match=named) as got:
        _configure(fake_openai, endpoint_dialect=dialect, sub_lm_kwargs=option)
    assert 'engine": "litellm"' in str(got.value) and "sub_lm_kwargs" in str(got.value)
    assert fake_openai.requests == []


# ---- replies a reasoning model sends --------------------------------------------------------------


@pytest.mark.parametrize("dialect", ["openai", "vllm"])
def test_a_reasoning_channel_reply_is_promoted(fake_openai, tmp_path, dialect):
    """A reasoning model that puts the whole turn in `reasoning_content` and sends no content still
    completes the run: the adapter parses the promoted reasoning text."""
    fake_openai.reasoning_only = True
    _configure(fake_openai, endpoint_dialect=dialect)
    out, _ = _run(tmp_path)
    assert out == "42"


def test_a_brace_less_reply_parses_through_the_wrap(fake_openai, tmp_path):
    """Schema-guided decoding sometimes drops the outer braces; the adapter wraps once and parses."""
    fake_openai.script = [{"role": "assistant", "content": '"plan": "p", "code": "print(1)"'}]
    _configure(fake_openai)
    out, events = _run(tmp_path)
    assert out == "42"
    assert len(fake_openai.requests) == 3
    steps = [e["payload"] for e in events if e["type"] == "main_step"]
    assert [s.get("parse_wrapped") for s in steps] == [True, None]


# ---- the litellm engine ---------------------------------------------------------------------------


def test_a_model_string_that_needs_litellm_raises_the_install_hint_without_it(monkeypatch):
    monkeypatch.setattr(_dspy_compat, "litellm_installed", lambda: False)
    with pytest.raises(ImportError, match=r"rlm-harness\[litellm\]"):
        rlm_harness.configure(RLMConfig(main_model="vllm/qwen", sub_model="vllm/qwen",
                                        interpreter="mock"))


def test_engine_litellm_in_a_roles_kwargs_raises_the_install_hint_without_it(fake_openai,
                                                                              monkeypatch):
    monkeypatch.setattr(_dspy_compat, "litellm_installed", lambda: False)
    with pytest.raises(ImportError, match=r"rlm-harness\[litellm\]"):
        _configure(fake_openai, main_lm_kwargs={"engine": "litellm"})


#: litellm sends the cap as `max_completion_tokens` on both prefixes (measured with litellm 1.109).
LITELLM_CAP = "max_completion_tokens"
LITELLM_KWARGS = {"engine": "litellm", "top_k": 40,
                  "extra_body": {"chat_template_kwargs": {"enable_thinking": False},
                                 "thinking_token_budget": 64}}


@pytest.mark.parametrize("dialect", ["openai", "vllm"])
def test_the_litellm_engine_carries_what_the_native_one_refuses(fake_openai, tmp_path, dialect):
    pytest.importorskip("litellm")
    _configure(fake_openai, endpoint_dialect=dialect, main_lm_kwargs=LITELLM_KWARGS)
    out, events = _run(tmp_path)

    assert out == "42"
    planner = fake_openai.requests[:2]
    for req in fake_openai.requests:
        body = req["body"]
        assert req["path"] == PATH
        assert body["model"] == SERVED
        assert body[LITELLM_CAP] == 8192
        assert body["top_k"] == 40
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        assert body["thinking_token_budget"] == 64
        assert "reasoning_effort" not in body
    for req in planner:
        assert req["body"]["response_format"]["type"] == "json_schema"
        assert _schema_fields(req["body"]) == ["plan", "code"]

    lm = _setup(events)["lm"]["main"]
    assert lm["engine"] == "litellm"
    assert lm["kwargs"]["top_k"] == 40
    assert lm["kwargs"]["extra_body"]["thinking_token_budget"] == 64


def test_the_litellm_engine_promotes_a_reasoning_channel_reply(fake_openai, tmp_path):
    pytest.importorskip("litellm")
    fake_openai.reasoning_only = True
    _configure(fake_openai, main_lm_kwargs={"engine": "litellm"})
    out, _ = _run(tmp_path)
    assert out == "42"
