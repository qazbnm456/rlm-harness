"""AntigravityLM tests: the adapter that runs a completion through the `agy` CLI.

CI has no `agy` and no Antigravity sign-in, so every call here runs a FAKE `agy`: a Python script
started as ``[sys.executable, script]`` through the adapter's real subprocess path (argv, stdin,
working directory, deadline, process group). It records what it was handed and answers with the
JSON shapes measured from `agy` 1.4.12, which the module docstring lists.

What the fake CANNOT catch is `agy` drifting: a renamed flag, a moved field, an agent frontmatter
key that stops meaning what it meant. Those were measured against a signed-in `agy`, and the
module docstring says how to measure them again.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time

import pytest

pytest.importorskip("dspy")

import dspy

import rlm_harness
from rlm_harness import antigravity_lm

_FAKE_AGY = r'''
import json, os, pathlib, sys, time

cfg = json.loads(pathlib.Path(os.environ["FAKE_AGY_CONFIG"]).read_text(encoding="utf-8"))
argv = sys.argv[1:]

def opt(flag):
    return argv[argv.index(flag) + 1] if flag in argv else None

agent = opt("--agent")
agent_file = pathlib.Path.cwd() / ".agents" / "agents" / f"{agent}.md"
schema = opt("--json-schema")
record = {
    "argv": argv,
    "stdin": sys.stdin.buffer.read().decode("utf-8"),
    "cwd": os.getcwd(),
    "cwd_listing": sorted(p.name for p in pathlib.Path.cwd().iterdir()),
    "pid": os.getpid(),
    "started": time.time(),
    "agent_text": agent_file.read_text(encoding="utf-8") if agent and agent_file.is_file() else None,
    "schema_text": pathlib.Path(schema).read_text(encoding="utf-8") if schema else None,
}
if cfg.get("pidfile"):
    pathlib.Path(cfg["pidfile"]).write_text(str(os.getpid()))
if cfg.get("in_flight"):
    pathlib.Path(cfg["in_flight"]).write_text("1")
if cfg.get("gate"):
    end = time.monotonic() + 10
    while not pathlib.Path(cfg["gate"]).exists() and time.monotonic() < end:
        time.sleep(0.01)
time.sleep(cfg.get("sleep", 0))
record["ended"] = time.time()
with open(cfg["log"], "a", encoding="utf-8") as f:
    f.write(json.dumps(record) + "\n")
sys.stderr.write(cfg.get("stderr", ""))
for line in cfg.get("stdout", []):
    sys.stdout.write(line + "\n")
if cfg.get("result") is not None:
    sys.stdout.write(json.dumps({"event": "result", "result": cfg["result"]}) + "\n")
sys.stdout.flush()
sys.exit(cfg.get("exit", 0))
'''

#: A real `agy` 1.4.12 result, from a one-word answer on the generated agent.
_USAGE = {"input_tokens": 382, "output_tokens": 52, "thinking_tokens": 50,
          "cache_read_tokens": 0, "total_tokens": 434}


def _ok(response="ok\n", **extra):
    return {"conversation_id": "conv-1", "status": "SUCCESS", "response": response,
            "duration_seconds": 0.1, "num_turns": 1, "usage": dict(_USAGE), **extra}


class _FakeAgy:
    def __init__(self, tmp_path, monkeypatch):
        self.script = tmp_path / "fake_agy.py"
        self.script.write_text(_FAKE_AGY, encoding="utf-8")
        self.config = tmp_path / "fake_agy.json"
        self.log = tmp_path / "fake_agy.log"
        self.tmp = tmp_path
        self.argv0 = [sys.executable, str(self.script)]
        monkeypatch.setenv("FAKE_AGY_CONFIG", str(self.config))
        for name in antigravity_lm._BILLING_ENV:
            monkeypatch.delenv(name, raising=False)
        self.answer(_ok())

    def answer(self, result=None, **cfg):
        self.config.write_text(json.dumps({"log": str(self.log), "result": result, **cfg}), encoding="utf-8")

    def calls(self):
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text(encoding="utf-8").splitlines()]

    def lm(self, model="gemini-test", **kwargs):
        return rlm_harness.AntigravityLM(model, executable=self.argv0, **kwargs)


@pytest.fixture
def agy(tmp_path, monkeypatch):
    return _FakeAgy(tmp_path, monkeypatch)


def _complete(lm, prompt="hi", *, response_format=None, system=None):
    """Drive the LM's own engine with a real `lm15.Request` (see the ClaudeAgentLM tests)."""
    T = dspy.lm15
    request = T.Request(
        model=lm.model,
        messages=[T.Message(role="user", parts=[T.TextPart(text=prompt)])],
        system=system,
        config=T.Config(response_format=response_format),
    )
    return lm.engine.complete(request)


def _agent_body(agent_text):
    """The Markdown body after the frontmatter, i.e. what `agy` makes the system prompt."""
    assert agent_text.startswith("---\n")
    _, frontmatter, body = agent_text.split("---\n", 2)
    return frontmatter, body


def _stdin_event(call):
    lines = call["stdin"].splitlines()
    assert len(lines) == 1, "a print-mode turn is ONE stream-json event; more would be more turns"
    return json.loads(lines[0])


# --- the export and the LM's shape -------------------------------------------------------------


def test_the_lazy_export_and_the_prefix():
    assert {"AntigravityLM", "AGY_PREFIX"} <= set(rlm_harness.__all__)
    assert rlm_harness.AntigravityLM.__name__ == "AntigravityLM"
    assert rlm_harness.AGY_PREFIX == "agy/"


def test_a_call_goes_through_THIS_adapter_and_not_litellm(agy):
    """The same pin as `ClaudeAgentLM`'s: a managed LM with no engine would be routed to
    litellm, so the engines are asserted, and a `copy_lm` variant must keep them."""
    from rlm_harness._dspy_compat import copy_lm

    lm = agy.lm()
    assert isinstance(lm.engine, antigravity_lm._AgyEngine), "dspy would route this call to litellm"
    assert isinstance(lm._async_engine_spec, antigravity_lm._AsyncAgyEngine), "the async seat is unrouted"
    variant = copy_lm(lm, rollout_id=1)
    assert type(variant) is type(lm)
    assert variant(prompt="hi") == ["ok"], "the copy lost its engine"


def test_the_lm_does_not_default_to_a_live_cache(agy):
    from rlm_harness._dspy_compat import lm_cache_is_live

    assert lm_cache_is_live(agy.lm()) is False
    assert lm_cache_is_live(agy.lm(cache=True)) is True


def test_the_trace_label_is_prefixed_and_agy_gets_the_bare_id(agy):
    lm = agy.lm("gemini-3.8-flash-low")
    assert lm.model == "agy/gemini-3.8-flash-low"
    _complete(lm)
    argv = agy.calls()[-1]["argv"]
    assert argv[argv.index("--model") + 1] == "gemini-3.8-flash-low"


@pytest.mark.parametrize("model", ["", "   "])
def test_a_model_id_is_required(agy, model):
    with pytest.raises(ValueError, match="agy models"):
        agy.lm(model)


# --- the pure-completion guarantee -------------------------------------------------------------


def test_a_completion_is_a_generated_agent_with_no_tools_and_no_ambient_components(agy):
    """The pure-completion guarantee, asserted on what `agy` is handed.

    `excludeDefaultComponents: true` is what keeps the user's global and workspace rules, skills
    and `agy`'s own default prompt out, and `tools: []` is what leaves the model nothing to call:
    both measured against a signed-in `agy` with a control (module docstring). Dropping either
    line is the regression this pins. `--dangerously-skip-permissions` must never appear: it
    exists to auto-approve tools, and this call is supposed to have none to approve.
    """
    _complete(agy.lm())
    call = agy.calls()[-1]
    frontmatter, _ = _agent_body(call["agent_text"])
    assert "excludeDefaultComponents: true\n" in frontmatter, "agy's default prompt and rules are back"
    assert "tools: []\n" in frontmatter, "the model was handed a toolset"
    assert "--dangerously-skip-permissions" not in call["argv"]
    assert "--disable-slash-commands" in call["argv"], "a prompt starting with / would run a command"
    argv = call["argv"]
    assert argv[argv.index("--input-format") + 1] == "stream-json"
    assert argv[argv.index("--output-format") + 1] == "stream-json"


def test_a_call_runs_in_a_private_workspace_holding_nothing_a_model_could_read(agy, monkeypatch, tmp_path):
    """`agy`'s workspace is its working directory, so the caller's cwd must never be it: a
    repository there would be the workspace the agent, its rules and its file mentions see."""
    monkeypatch.chdir(tmp_path)
    _complete(agy.lm())
    call = agy.calls()[-1]
    assert os.path.realpath(call["cwd"]) != os.path.realpath(str(tmp_path))
    assert call["cwd_listing"] == [".agents", ".rlm-schemas"]


def test_the_system_prompt_is_the_agent_body_verbatim(agy):
    """The planner's instructions ARE the agent body, byte for byte: measured, `agy` makes the
    body the system prompt. The adapter adds no instruction of its own, and a body full of dspy's
    markers (and a line that looks like a frontmatter fence) survives untouched."""
    system = "PLANNER RULES\n---\n[[ ## code ## ]]\n{\"a\": 1}\n"
    lm = agy.lm()
    _complete(lm, system=system)
    _, body = _agent_body(agy.calls()[-1]["agent_text"])
    assert body == system, "the task instructions did not reach the model unchanged"
    # The sub-LM seat's measured shape: no system prompt, so an EMPTY body, not a kit-written one.
    _complete(lm)
    _, body = _agent_body(agy.calls()[-1]["agent_text"])
    assert body == ""


def test_agents_are_content_addressed(agy):
    lm = agy.lm()
    for system in ("A", "B", "A"):
        _complete(lm, system=system)
    names = [call["argv"][call["argv"].index("--agent") + 1] for call in agy.calls()]
    assert names[0] == names[2] != names[1]


def test_the_prompt_travels_on_stdin_and_never_in_argv(agy):
    """Linux refuses ONE argument over 131,072 bytes (`E2BIG`), and a planner prompt can be
    longer. The control is the length assertion: without it this passes for a prompt argv
    would also have carried."""
    prompt = "QUOKKA-HEAD " + "x" * 200_000 + " QUOKKA-TAIL"
    assert len(prompt.encode()) > 131_072
    _complete(agy.lm(), prompt)
    call = agy.calls()[-1]
    assert _stdin_event(call) == {"event": "user", "message": {"role": "user", "content": prompt}}
    # Either end, so a TRUNCATED copy in argv (which would dodge E2BIG and still be wrong) is caught.
    assert not any("QUOKKA" in arg for arg in call["argv"])


def test_text_dspy_would_send_reaches_agy_unchanged(agy):
    """Non-ASCII text survives the stdin encoding, and a leading slash is left to
    `--disable-slash-commands` rather than rewritten here."""
    prompt = "/usage @dataclass 中文 ✓"
    _complete(agy.lm(), prompt)
    assert _stdin_event(agy.calls()[-1])["message"]["content"] == prompt


def test_a_non_text_part_is_refused_rather_than_stringified(agy):
    T = dspy.lm15
    request = T.Request(
        model="agy/x",
        messages=[T.Message(role="user", parts=[T.BinaryPart(data="AA==", media_type="image/png")])],
        config=T.Config(),
    )
    with pytest.raises(TypeError, match="cannot send a .* part"):
        agy.lm().engine.complete(request)
    assert agy.calls() == [], "agy was started for a request it could not carry"


# --- the answer --------------------------------------------------------------------------------


def test_a_plain_answer_is_the_response_without_agys_trailing_newline(agy):
    agy.answer(_ok("pong\n"))
    response = _complete(agy.lm())
    assert response.message.parts[0].text == "pong"
    assert response.id == "conv-1", "the call can no longer be found among agy's conversations"


def test_a_structured_call_gets_the_schema_and_the_finish_tool(agy):
    """`--json-schema` is enforced through a tool named `finish`: with `tools: []` agy ignored
    the schema SILENTLY (SUCCESS, four turns of prose, no structured_output). So a schema call
    must carry both, and the schema is the one dspy canonicalised, byte for byte."""
    schema = {"type": "object", "properties": {"headline": {"type": "string"}},
              "required": ["headline"], "additionalProperties": False}
    canonical = {"type": "json_schema", "name": "DSPyProgramOutputs", "strict": True, "schema": schema}
    agy.answer(_ok("prose\n", structured_output={"headline": "ok"}))
    _complete(agy.lm(), response_format=canonical)
    call = agy.calls()[-1]
    assert json.loads(call["schema_text"]) == schema
    frontmatter, _ = _agent_body(call["agent_text"])
    assert "tools: [finish]\n" in frontmatter, "agy would ignore the schema without its finish tool"
    assert "excludeDefaultComponents: true\n" in frontmatter


def test_the_structured_answer_is_structured_output_and_never_the_joined_response(agy):
    """`response` JOINS every attempt of a call, rejected JSON included (measured: three
    documents after two failed validations), so it is never what reaches dspy."""
    joined = '{"items": ["Red (3)"]}\n{"items": [{"letters": 3}]}\n{"items": [{"count": 3}]}\n'
    agy.answer(_ok(joined, structured_output={"items": [{"count": 3}]}))
    schema = {"type": "object", "properties": {"items": {"type": "array"}}}
    response = _complete(agy.lm(), response_format={"type": "json_schema", "schema": schema})
    assert response.message.parts[0].text == '{"items": [{"count": 3}]}'


def test_a_structured_call_with_no_structured_answer_is_refused(agy):
    agy.answer(_ok('{"headline": "ok"}\n'))
    schema = {"type": "object", "properties": {"headline": {"type": "string"}}}
    with pytest.raises(RuntimeError, match="without a structured_output"):
        _complete(agy.lm(), response_format={"type": "json_schema", "schema": schema})


@pytest.mark.parametrize("response_format", [
    pytest.param({"type": "json_object"}, id="json-object-fallback"),
    pytest.param({"type": "json_schema", "schema": {}}, id="empty-schema"),
    pytest.param({"type": "json_schema", "schema": {"type": "array"}}, id="non-object-root"),
])
def test_a_response_format_agy_cannot_enforce_asks_for_none(agy, response_format):
    """A non-object root makes `agy` exit 1 at startup, so it is dropped, with the other two
    shapes that carry no schema. The control: the plain path's own agent and no flag."""
    _complete(agy.lm(), response_format=response_format)
    call = agy.calls()[-1]
    assert "--json-schema" not in call["argv"]
    assert "tools: []\n" in _agent_body(call["agent_text"])[0]


def test_a_class_response_format_is_read_too():
    """Reached only by a caller assembling a `Request` by hand: dspy canonicalises the class to
    the dict before an engine sees it."""
    import pydantic

    class Out(pydantic.BaseModel):
        headline: str

    assert antigravity_lm._schema_of(Out) == Out.model_json_schema()


# --- failure -----------------------------------------------------------------------------------


@pytest.mark.parametrize(("cfg", "match"), [
    pytest.param(
        {"exit": 1, "result": {"conversation_id": "", "status": "ERROR", "response": "",
                               "error": "invalid model selection: model x is not recognized",
                               "finished_reason": "error", "usage": {}}},
        "is not recognized", id="bad-model-exit-1"),
    pytest.param({"exit": 0, "result": _ok("partial\n", finished_reason="print_timeout")},
                 "print_timeout", id="cut-off-turn"),
    pytest.param({"exit": 3, "result": _ok(), "stderr": 'AGY_ERROR: {"status":"RESOURCE_EXHAUSTED"}\n'},
                 "RESOURCE_EXHAUSTED", id="non-zero-exit-with-a-success-result"),
    pytest.param({"exit": 2, "result": None, "stderr": "flag needs an argument: -p\n"},
                 "flag needs an argument", id="no-result-event"),
    pytest.param({"exit": 0, "result": _ok("\n")}, "empty response", id="empty-answer"),
])
def test_a_failed_call_raises_with_what_agy_said(agy, cfg, match):
    agy.answer(**cfg)
    with pytest.raises(RuntimeError, match=match):
        _complete(agy.lm())


def test_noise_on_stdout_does_not_hide_the_result(agy):
    agy.answer(_ok("ok\n"), stdout=['{"event":"init","init":{}}', "not json at all"])
    assert _complete(agy.lm()).message.parts[0].text == "ok"


def test_dspys_retry_budget_is_live_but_never_multiplies_an_agy_call(agy):
    """`ClaudeAgentLM`'s pin, for this adapter: dspy's `num_retries` budget is live on a managed
    LM, and only the error CLASSIFICATION keeps it from re-running a failed `agy` call. The
    first two assertions are the control."""
    agy.answer(exit=1, result={"status": "ERROR", "response": "", "error": "boom", "usage": {}})
    lm = agy.lm()
    assert lm.num_retries > 0, "dspy's retry budget is zero, so this test cannot see the hazard"
    assert hasattr(lm, "_engine_spec"), "not managed, so dspy's budget would be zero anyway"
    with pytest.raises(Exception) as caught:
        lm(prompt="hi")
    assert dspy.is_retryable_lm_error(caught.value) is False, (
        "dspy now calls this adapter's failure retryable: cap `num_retries` on the LM"
    )
    assert len(agy.calls()) == 1, f"agy was started {len(agy.calls())} times for one failed call"


# --- deadline, concurrency, async --------------------------------------------------------------


def test_the_call_deadline_fires_and_kills_agy(agy):
    """A deadline that raised but left `agy` running would keep spending the account's quota
    on an answer nobody reads, and hold one of the two concurrency slots' worth of processes."""
    pidfile = agy.tmp / "pid"
    agy.answer(_ok(), sleep=30, pidfile=str(pidfile))
    lm = agy.lm(timeout_s=1.0)
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="timed out after 1.0s"):
        _complete(lm)
    assert time.monotonic() - started < 4, "the deadline waited for agy instead of killing it"
    assert pidfile.exists(), "agy never started, so this measured no deadline"
    if os.name != "nt":
        with pytest.raises(ProcessLookupError):
            os.kill(int(pidfile.read_text()), 0)


def test_the_call_deadline_fires_on_the_async_seat_too(agy):
    """Through `acall`, the planner route, so the message is observed through dspy's error
    boundary, and it must stay non-retryable there too."""
    agy.answer(_ok(), sleep=30)
    with pytest.raises(Exception, match="timed out after 0.5s") as caught:
        asyncio.run(agy.lm(timeout_s=0.5).acall(prompt="hi"))
    assert dspy.is_retryable_lm_error(caught.value) is False


def test_the_async_engine_is_wired_and_not_just_constructed(agy):
    agy.answer(_ok("async ok\n"))
    assert asyncio.run(agy.lm().acall(prompt="hi")) == ["async ok"]


def test_the_async_seat_does_not_block_the_loop_it_was_awaited_on(agy):
    """The planner's `aforward` runs on a loop `repl.execute` blocks. The fake `agy` will not
    answer until the CALLER's loop runs another task while the call is in flight, so an engine
    body that blocked the loop reaches its own deadline instead."""
    in_flight = agy.tmp / "in_flight"
    gate = agy.tmp / "gate"
    agy.answer(_ok("async ok\n"), in_flight=str(in_flight), gate=str(gate))

    async def main():
        lm = agy.lm(timeout_s=8.0)

        async def unblock():
            while not in_flight.exists():
                await asyncio.sleep(0.01)
            gate.write_text("open")

        answer, _ = await asyncio.gather(lm.acall(prompt="hi"), unblock())
        return answer

    assert asyncio.run(main()) == ["async ok"], "the planner seat blocked its own event loop"


def test_no_more_than_two_agy_processes_run_at_once(agy):
    """The politeness cap, measured on the processes themselves. The control is the lower bound:
    without overlap at all, the cap would be untested rather than respected."""
    from concurrent.futures import ThreadPoolExecutor

    agy.answer(_ok(), sleep=0.4)
    lm = agy.lm()
    with ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(lambda _: _complete(lm), range(5)))
    spans = [(c["started"], c["ended"]) for c in agy.calls()]
    peak = max(sum(1 for s, e in spans if s <= t < e) for t, _ in spans)
    assert peak <= 2, f"{peak} agy processes ran at once"
    assert peak == 2, "nothing ran concurrently, so this measured no cap"


# --- usage -------------------------------------------------------------------------------------


def test_usage_reaches_dspys_tracker_as_agy_reported_it(agy):
    """`output_tokens` already includes the thinking tokens (measured: 52 of which 50), so the
    completion count is `output_tokens` alone and thinking rides beside it, never added."""
    with dspy.track_usage() as tracker:
        agy.lm("m")(prompt="hi")
    assert tracker.get_total_tokens() == {"agy/m": {
        "prompt_tokens": 382, "completion_tokens": 52, "total_tokens": 434,
        "prompt_tokens_details": None, "completion_tokens_details": None,
    }}
    response = _complete(agy.lm("m"))
    assert response.provider_data == {"thinking_tokens": 50, "cache_read_tokens": 0}


def test_an_unreported_usage_stays_absent(agy):
    """Absent is not zero. The real-zero arm is the control: it must still reach the tracker."""
    agy.answer({k: v for k, v in _ok().items() if k != "usage"})
    with dspy.track_usage() as tracker:
        agy.lm("m")(prompt="hi")
    assert tracker.get_total_tokens() == {}, "an absent usage was recorded as a zero-token call"
    agy.answer(_ok(usage={"input_tokens": 0, "output_tokens": 0}))
    with dspy.track_usage() as tracker:
        agy.lm("m")(prompt="hi")
    assert tracker.get_total_tokens()["agy/m"]["total_tokens"] == 0


# --- construction ------------------------------------------------------------------------------


@pytest.mark.parametrize("name", antigravity_lm._BILLING_ENV)
def test_construction_refuses_a_variable_that_moves_the_bill(agy, monkeypatch, name):
    monkeypatch.setenv(name, "set")
    with pytest.raises(RuntimeError, match=name):
        agy.lm()
    assert agy.lm(allow_api_key=True).model == "agy/gemini-test"


def test_n_above_one_is_one_agy_call_per_candidate_and_warns(agy, caplog):
    with caplog.at_level("WARNING", logger="rlm_harness.antigravity_lm"):
        lm = agy.lm(n=3)
    assert "3 calls per LM call" in "\n".join(r.getMessage() for r in caplog.records)
    assert len(lm(prompt="hi")) == 3
    assert len(agy.calls()) == 3, "n=3 no longer costs three agy calls, so the warning is wrong"
    # The control: a plain call is one process.
    assert len(agy.lm()(prompt="hi")) == 1
    assert len(agy.calls()) == 4


@pytest.mark.parametrize("kwargs", [{}, {"n": 1}], ids=["absent", "explicit-one"])
def test_no_warning_when_n_cannot_cost_anything(agy, caplog, kwargs):
    with caplog.at_level("WARNING", logger="rlm_harness.antigravity_lm"):
        agy.lm(**kwargs)
    assert not [r for r in caplog.records if "calls per LM call" in r.getMessage()]


def _no_agy_anywhere(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path / "empty-path"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))


def test_a_missing_agy_says_how_to_get_one(agy, monkeypatch, tmp_path):
    _no_agy_anywhere(monkeypatch, tmp_path)
    with pytest.raises(FileNotFoundError, match="install.sh"):
        rlm_harness.AntigravityLM("m")
    with pytest.raises(FileNotFoundError, match="not an executable file"):
        rlm_harness.AntigravityLM("m", executable=tmp_path / "nope")
    with pytest.raises(ValueError, match="empty sequence"):
        rlm_harness.AntigravityLM("m", executable=[])


@pytest.mark.skipif(os.name == "nt", reason="the install script's ~/.local/bin is the POSIX layout")
def test_agy_is_found_where_the_install_script_puts_it(agy, monkeypatch, tmp_path):
    """The official install script's TARGET_DIR is `~/.local/bin`, which is off PATH until
    `agy install` adds it: the state this adapter was first run in."""
    _no_agy_anywhere(monkeypatch, tmp_path)
    binary = tmp_path / "home" / ".local" / "bin" / "agy"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)
    assert rlm_harness.AntigravityLM("m")._call._argv0 == [str(binary)]
    on_path = tmp_path / "empty-path" / "agy"
    on_path.parent.mkdir()
    on_path.write_text("#!/bin/sh\n", encoding="utf-8")
    on_path.chmod(0o755)
    assert rlm_harness.AntigravityLM("m")._call._argv0 == [str(on_path)], "PATH must win"
