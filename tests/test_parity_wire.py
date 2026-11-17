"""The exported SFT records are the requests the server received, on every route the kit builds.

Each run goes through ``configure()``'s real LM, dspy's engines and a local OpenAI-compatible server
(``conftest.FakeOpenAIServer``), so the reference for every record is the request body the server
recorded, never the kit's own capture. The trajectory has four planner turns (one truncated output,
one error turn) and then the extract fallback; a second run with the same first four turns ends by
SUBMIT, and its counterfactual extract must equal the real one.
"""

from __future__ import annotations

import asyncio

import pytest

dspy = pytest.importorskip("dspy")

import rlm_harness
from rlm_harness import (
    RLMConfig,
    RLMTask,
    TraceRecorder,
    _dspy_compat,
    export_counterfactual_extracts,
    group_by_run,
    load_events,
)
from rlm_harness.dataset import export_sft_turns
from rlm_harness.prompt_record import SOURCE_COMPLETION, messages_sha256
from rlm_harness.testing import ScriptedInterpreter, submit
from rlm_harness.trace import EVENT_EXTRACT, EVENT_MAIN_STEP, EVENT_RLM_SETUP

SERVED = "qwen/qwen3-next"
CHARS = 50
PLANNER_TURNS = [
    {"plan": "look at q", "code": "```python\nprint('A' * 200)\n```"},
    {"plan": "count", "code": "x = len(q)\nprint(x)"},
    {"plan": "oops", "code": "raise ValueError('bad')"},
    {"plan": "again", "code": "print(2)"},
]


def _error(tools, variables):
    raise _dspy_compat.recoverable_interpreter_error()("ValueError: bad")


STEPS = ["A" * 200, "11", _error, "2"]


class _Task(RLMTask):
    signature = "q: str -> answer: str"
    output_field = "answer"


ROUTES = [
    pytest.param("openai", "json", None, id="openai-json"),
    pytest.param("openai", "chat", None, id="openai-chat"),
    pytest.param("openai", "default", None, id="openai-xml"),
    pytest.param("vllm", "json", None, id="vllm-json"),
    pytest.param("vllm", "chat", None, id="vllm-chat"),
    pytest.param("openai", "json", "litellm", id="litellm-json"),
    pytest.param("openai", "json", "native", id="openai-json-native-reasoning"),
]


def _run(server, tmp_path, name, *, dialect, adapter, engine, max_iterations, steps, replies,
         native=False):
    if engine == "litellm":
        pytest.importorskip("litellm")
    # `o3-mini` is in dspy's catalog as reasoning-capable, which is what native mode acts on.
    model, cap = ("o3-mini", 16000) if native else (SERVED, 8192)
    rlm_harness.configure(RLMConfig(
        main_model=model, sub_model=model, base_url=server.base_url, api_key="k", max_tokens=cap,
        endpoint_dialect=dialect, adapter=adapter, max_iterations=max_iterations,
        max_output_chars=CHARS, main_lm_kwargs={"engine": engine} if engine else None,
        native_reasoning=native))
    server.script = list(replies)
    first = len(server.requests)
    path = str(tmp_path / f"{name}.jsonl")
    with TraceRecorder(path, run_id=name):
        out = asyncio.run(_Task(interpreter=ScriptedInterpreter(steps)).arun(q="hello world"))
    return out, load_events(path), [r["body"] for r in server.requests[first:]]


@pytest.mark.parametrize(("dialect", "adapter", "engine"), ROUTES)
def test_every_exported_record_is_the_request_the_server_received(fake_openai, tmp_path, dialect,
                                                                    adapter, engine):
    fake_openai.reasoning_content = "thinking it through"
    native = engine == "native"
    route = {"dialect": dialect, "adapter": adapter, "engine": None if native else engine,
             "native": native}
    out, events, sent = _run(fake_openai, tmp_path, "extract", max_iterations=4, steps=STEPS,
                             replies=[*PLANNER_TURNS, {"answer": "final"}], **route)
    assert out == "final"
    assert len(sent) == 5, "four planner turns, then the extract"

    steps = [e["payload"] for e in events if e["type"] == EVENT_MAIN_STEP]
    assert [s["history_len"] for s in steps] == [0, 1, 2, 3]
    assert [s["iteration"] for s in steps] == ["1/4", "2/4", "3/4", "4/4"]
    assert steps[2]["output"].startswith("[Error]")
    assert steps[0]["output"] == "A" * 200, "the trajectory keeps the whole output"
    assert "A" * 200 not in str(sent[1]["messages"]), "later turns see it truncated"
    for step, body in zip(steps, sent, strict=False):
        assert step["messages_sha256"] == messages_sha256(body["messages"])
        assert step["reasoning_content"] == "thinking it through"
        assert isinstance(step["completion"], str) and step["completion"]
    # Each turn is stamped when it was parsed, during the run: before the trajectory is flushed,
    # which is when `rlm_setup` is written.
    flushed = next(e["ts"] for e in events if e["type"] == EVENT_RLM_SETUP)
    stamps = [e["ts"] for e in events if e["type"] == EVENT_MAIN_STEP]
    assert stamps == sorted(stamps)
    if not native:
        # In native mode the reasoning arrives in the reply, not in the parse the timer reads, so
        # the turns keep their flush-time stamps.
        assert stamps[-1] < flushed
    extract = next(e["payload"] for e in events if e["type"] == EVENT_EXTRACT)
    assert extract["history_len"] == 4

    records = export_sft_turns(group_by_run(events), render="adapter")
    assert [r["kind"] for r in records] == ["action"] * 4 + ["extract"]
    for record, body in zip(records, sent, strict=True):
        assert record["messages"] == body["messages"], f"{record['kind']} turn {record['turn']}"
        assert record["verified"] is True
        assert record["assistant_source"] == SOURCE_COMPLETION
        assert record["reasoning_content"] == "thinking it through"
    # Each action request is a multi-turn chat: three messages per earlier turn.
    assert [len(r["messages"]) for r in records[:4]] == [2, 5, 8, 11]
    # A turn's own recorded output cap wins over the run's: the render truncates each earlier turn
    # with the cap that turn was shown under.
    for e in events:
        if e["type"] == EVENT_RLM_SETUP:
            e["payload"]["max_output_chars"] = 10_000
        elif e["type"] == EVENT_MAIN_STEP:
            e["payload"]["max_output_chars"] = CHARS
    again = export_sft_turns(group_by_run(events), render="adapter")
    assert [r["verified"] for r in again[:4]] == [True] * 4

    # The same first four turns, then SUBMIT: its counterfactual extract is the real one above.
    _, submitted, _ = _run(fake_openai, tmp_path, "submit", max_iterations=5,
                           steps=[*STEPS, submit({"answer": "final"})],
                           replies=[*PLANNER_TURNS, {"plan": "done", "code": "SUBMIT('final')"}],
                           **route)
    (counterfactual,) = export_counterfactual_extracts(group_by_run(submitted))
    assert counterfactual["messages"] == sent[4]["messages"]
    assert counterfactual["turn"] == 4


# ---- traces recorded by rlm-harness 1.26 on dspy 3.4 ----------------------------------------------


def _legacy_trace() -> list[dict]:
    """A trajectory as rlm-harness 1.26 recorded it on dspy 3.4: the planner was shown its earlier
    turns as a `repl_history` input, one single-turn request per call."""
    field = {"kind": "builtin", "type": "str", "extra": {}}
    action = {"instructions": "Answer.", "fields": [
        {"name": "variables_info", "io": "input", **field},
        {"name": "repl_history", "io": "input", "kind": "repl_history", "type": "REPLHistory",
         "extra": {}},
        {"name": "iteration", "io": "input", **field},
        {"name": "reasoning", "io": "output", **field},
        {"name": "code", "io": "output", **field}]}
    extract = {"instructions": "Extract.", "fields": [
        {"name": "variables_info", "io": "input", **field},
        {"name": "repl_history", "io": "input", "kind": "repl_history", "type": "REPLHistory",
         "extra": {}},
        {"name": "answer", "io": "output", **field}]}
    setup = {"adapter": "json", "adapter_class": "rlm_harness.runtime._LenientJSONAdapter",
             "dspy": "3.4.1", "max_iterations": 2, "max_output_chars": 100,
             "variables_info": ["q: str"], "input_fields": ["q"], "output_fields": ["answer"],
             "predictors": {"action": action, "extract": extract}, "attempt": 0}

    def ev(i, kind, payload):
        return {"schema": "rlm-harness/trace/v2", "run_id": "old", "step_id": i, "ts": float(i),
                "type": kind, "payload": payload}

    step = {"reasoning": "look", "code": "print(1)", "output": "1", "history_len": 0,
            "iteration": "1/1", "messages_sha256": "0" * 64, "completion": "{}", "attempt": 0}
    return [ev(0, "run_start", {"meta": {}}), ev(1, "rlm_setup", setup),
            ev(2, "main_step", {"turn": 0, **step}),
            ev(3, "final", {"final_reasoning": "look", "extract": False,
                            "outputs": {"answer": "1"}}),
            ev(4, "result", {"output": "1"})]


def test_a_trace_from_rlm_harness_1_26_is_skipped_by_the_render_exports_and_named(caplog):
    runs = group_by_run(_legacy_trace())
    with caplog.at_level("WARNING", logger="rlm_harness.dataset"):
        assert export_sft_turns(runs, render="adapter") == []
        assert export_counterfactual_extracts(runs) == []
    assert caplog.text.count("Export them with rlm-harness 1.26.x") == 2, caplog.text
    # The format-agnostic default reads it as it always did.
    (record,) = export_sft_turns(runs)
    assert record["output"] == {"reasoning": "look", "code": "print(1)"}


# ---- what the kit keeps out of dspy.RLM -------------------------------------------------------------


def test_a_history_processor_never_reaches_the_rlm_through_the_kit(fake_openai):
    rlm_harness.configure(RLMConfig(main_model=SERVED, sub_model=SERVED,
                                    base_url=fake_openai.base_url, api_key="k"))
    with pytest.raises(ValueError, match="history_processor"):
        asyncio.run(_Task(interpreter=ScriptedInterpreter([])).arun(
            q="x", history_processor=lambda history: history))
    assert fake_openai.requests == []


def test_dspy_takes_history_processor_per_call():
    """The refusal above is for an option dspy really takes per call, so it has to be kept out."""
    assert set(_dspy_compat.RLM_RUNTIME_OPTIONS_REFUSED) <= set(dspy.RLM._RUNTIME_OPTION_NAMES)


# ---- the fields and keys a consumer reads ----------------------------------------------------------


def test_the_fields_a_consumer_reads_are_all_present_on_dspy_3_5(fake_openai, tmp_path):
    """One traced run over the fake server, with a tool call, a sub-LM escalation and a SUBMIT, then
    a failed run: every field and export key on the consumer's stability checklist is present."""
    from rlm_harness import record_tool_call
    from rlm_harness.testing import call

    def lookup(key: str) -> str:
        """Look a key up."""
        record_tool_call("lookup", args={"key": key}, ok=False, error="down",
                         endpoint_error="down", circuit_broken=False, reason="unreachable")
        return "v"

    class _Tooled(_Task):
        tools = [lookup]

    fake_openai.reasoning_content = "thinking"
    rlm_harness.configure(RLMConfig(main_model=SERVED, sub_model=SERVED, base_url=fake_openai.base_url,
                                    api_key="k", max_iterations=4))
    fake_openai.script = [{"plan": "look", "code": "lookup('a')"},
                          {"plan": "ask", "code": "llm_query('hi')"},
                          {"role": "assistant", "content": "sub answer"},
                          {"plan": "done", "code": "SUBMIT('42')"}]
    path = str(tmp_path / "t.jsonl")
    with TraceRecorder(path, run_id="ok", meta={"task": "t"}):
        asyncio.run(_Tooled(interpreter=ScriptedInterpreter(
            [call("lookup", key="a"), call("llm_query", prompt="hi"), submit({"answer": "42"})]),
            run_tag="arm").arun(q="hi"))
    fake_openai.script = [{"role": "assistant", "content": "not json"}] * 4
    with pytest.raises(Exception), TraceRecorder(path, run_id="bad"):
        asyncio.run(_Task(interpreter=ScriptedInterpreter([])).arun(q="hi"))

    events = load_events(path)
    by_type: dict[str, list[dict]] = {}
    for e in events:
        by_type.setdefault((e["run_id"], e["type"]), []).append(e["payload"])

    def has(payload, *keys):
        missing = [k for k in keys if k not in payload]
        assert not missing, (missing, sorted(payload))

    for step in by_type[("ok", "main_step")]:
        has(step, "turn", "completion", "reasoning_content", "code", "output", "iteration")
    (sub,) = by_type[("ok", "sub_call")]
    has(sub, "kind", "input", "raw", "cause")
    tool = by_type[("ok", "tool_call")][0]
    has(tool, "tool", "ok", "error", "endpoint_error", "circuit_broken", "reason")
    for run in ("ok", "bad"):
        has(by_type[(run, "run_end")][0], "ok", "error")
    has(by_type[("bad", "run_end")][0], "error_chain")
    has(by_type[("ok", "result")][0], "output")
    has(by_type[("ok", "final")][0], "extract")
    setup = by_type[("ok", "rlm_setup")][0]
    has(setup, "adapter", "lm", "max_iterations", "inputs_sha256")
    assert by_type[("ok", "run_start")][0]["meta"] == {"task": "t"}

    from rlm_harness import payload_cause
    from rlm_harness.trace import CAUSE_ENDPOINT

    assert payload_cause(tool) == CAUSE_ENDPOINT
    runs = group_by_run(e for e in events if e["run_id"] == "ok")
    (record, *_) = export_sft_turns(runs, render="adapter")
    has(record, "messages", "assistant", "reasoning_content", "verified", "kind", "turn", "run_id",
        "adapter", "assistant_source", "run_tag", "attempt")
    has(record["assistant"], "content")
