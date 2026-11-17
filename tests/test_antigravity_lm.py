"""AntigravityLM tests: the adapter that runs a completion through the `agy` CLI.

CI has no `agy` and no Antigravity sign-in, so every call here runs a FAKE `agy`: a Python script
started as ``[sys.executable, script]`` through the adapter's real subprocess path (argv, stdin,
environment, working directory, deadline, kill). It is put in place of `_resolve_executable`'s
answer rather than through `executable=`, which takes one path. It records what it was handed and
answers with the JSON shapes measured from `agy` 1.4.12, which the module docstring lists.

What the fake CANNOT catch is `agy` drifting: a renamed flag, a moved field, an agent frontmatter
key that stops meaning what it meant. Those were measured against a signed-in `agy`, and the
module docstring says how to measure them again.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
import time

import pytest

pytest.importorskip("dspy")

import dspy

import rlm_harness
from rlm_harness import community
from rlm_harness.community import antigravity_lm

_FAKE_AGY = r'''
import json, os, pathlib, sys, time

cfg = json.loads(pathlib.Path(os.environ["FAKE_AGY_CONFIG"]).read_text(encoding="utf-8"))
argv = sys.argv[1:]

def opt(flag):
    return argv[argv.index(flag) + 1] if flag in argv else None

agent = opt("--agent")
agent_file = pathlib.Path.cwd() / ".agents" / "agents" / f"{agent}.md"
schema = opt("--json-schema")
time.sleep(cfg.get("read_delay", 0))
record = {
    "argv": argv,
    "stdin": sys.stdin.buffer.read().decode("utf-8"),
    "env_keys": sorted(os.environ),
    "cwd": os.getcwd(),
    "cwd_listing": sorted(p.name for p in pathlib.Path.cwd().iterdir()),
    "pid": os.getpid(),
    "started": time.time(),
    "agent_text": agent_file.read_text(encoding="utf-8") if agent and agent_file.is_file() else None,
    "schema_text": pathlib.Path(schema).read_text(encoding="utf-8") if schema else None,
}
if cfg.get("child_pidfile"):
    # A process of `agy`'s own, like a server it starts: a kill must reach it too.
    import subprocess
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    pathlib.Path(cfg["child_pidfile"]).write_text(str(child.pid))
if cfg.get("pidfile"):
    pathlib.Path(cfg["pidfile"]).write_text(str(os.getpid()))
if cfg.get("in_flight"):
    pathlib.Path(cfg["in_flight"]).write_text("1")
if cfg.get("in_flight_dir"):
    (pathlib.Path(cfg["in_flight_dir"]) / str(os.getpid())).write_text("1")
if cfg.get("gate"):
    end = time.monotonic() + cfg.get("gate_timeout", 10)
    while not pathlib.Path(cfg["gate"]).exists() and time.monotonic() < end:
        time.sleep(0.01)
time.sleep(cfg.get("sleep", 0))
record["ended"] = time.time()
# One file per call, numbered by an exclusive create, never an append to one shared file. Two
# fakes released by the same gate appended at once and the Windows job kept 4 of 5 records:
# append there is a seek to the end and then a write, two steps (CPython gh-86772, open).
n = 0
while True:
    try:
        with open(pathlib.Path(cfg["log"]) / f"{n:06d}.json", "x", encoding="utf-8") as f:
            f.write(json.dumps(record))
        break
    except FileExistsError:
        n += 1
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
        self.log = tmp_path / "fake_agy_calls"
        self.log.mkdir()
        self.tmp = tmp_path
        self.monkeypatch = monkeypatch
        argv0 = [sys.executable, str(self.script)]
        monkeypatch.setattr(antigravity_lm, "_resolve_executable", lambda executable: list(argv0))
        monkeypatch.setenv("FAKE_AGY_CONFIG", str(self.config))
        for name in antigravity_lm._BILLING_ENV:
            monkeypatch.delenv(name, raising=False)
        self.answer(_ok())

    def answer(self, result=None, **cfg):
        self.config.write_text(json.dumps({"log": str(self.log), "result": result, **cfg}), encoding="utf-8")

    def calls(self):
        """Every call's record, in the order the fakes wrote them."""
        return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(self.log.glob("*.json"))]

    def lm(self, model="gemini-test", **kwargs):
        return community.AntigravityLM(model, **kwargs)


@pytest.fixture
def agy(tmp_path, monkeypatch):
    return _FakeAgy(tmp_path, monkeypatch)


def _wait_for(predicate, *, timeout, what):
    end = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < end, f"gave up waiting for {what}"
        time.sleep(0.02)


def _count_spawns(monkeypatch):
    """Every `agy` process started from here on, counted at `Popen` rather than in the fake's log:
    a process killed while Python is still starting never writes to the log, so the log cannot
    tell "never started" from "started and killed at once"."""
    spawned = []
    real = subprocess.Popen

    class Counting(real):
        def __init__(self, args, *rest, **kwargs):
            if "--agent" in list(args):
                spawned.append(args)
            super().__init__(args, *rest, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", Counting)
    return spawned


def _alive_on_windows(pid):
    """Probe a pid without touching it: on Windows ``os.kill(pid, 0)`` would TERMINATE it."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    try:
        code = wintypes.DWORD()
        return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259  # STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def _alive(pid):
    if os.name == "nt":
        return _alive_on_windows(pid)
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    if not os.path.isdir("/proc/self"):
        return True  # no procfs (macOS) to tell a zombie apart; launchd reaps orphans there
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
            # A zombie has exited: an orphan is reaped by whatever runs as pid 1, which in a
            # container may never do it.
            return f.read().rsplit(")", 1)[-1].split()[0] != "Z"
    except (FileNotFoundError, ProcessLookupError):
        # Collected while this looked: `open` raises the first once the process is gone, and
        # `read` raises the second when it was collected after the file was opened.
        return False


def _assert_gone(pid):
    """``pid`` has exited, within a few seconds, on every platform."""
    end = time.monotonic() + 5
    while _alive(pid):
        assert time.monotonic() < end, f"process {pid} is still running"
        time.sleep(0.05)


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


def _message_texts(request):
    """The user text of an ``lm15.Request``, to tell concurrent calls apart."""
    return " ".join(getattr(part, "text", "") for m in request.messages for part in m.parts)


def _stdin_event(call):
    lines = call["stdin"].splitlines()
    assert len(lines) == 1, "a print-mode turn is ONE stream-json event; more would be more turns"
    return json.loads(lines[0])


# --- the export and the LM's shape -------------------------------------------------------------


def test_it_is_a_community_route_exported_from_community_only():
    """Outside the kit's SemVer promise (docs/CONVENTIONS.md): reachable as
    `rlm_harness.community.AntigravityLM` and never from `rlm_harness`, so the route can be
    withdrawn in a minor release. Its prefix is read from its `ROUTES` entry, not repeated."""
    assert community.AntigravityLM is antigravity_lm.AntigravityLM
    assert "AntigravityLM" not in rlm_harness.__all__
    assert not hasattr(rlm_harness, "AntigravityLM")
    route = community.route_for("agy/x")
    assert route.attr == "AntigravityLM" and route.vendor_permission == community.NOT_EXPLICIT
    assert antigravity_lm.AGY_PREFIX == route.prefix == "agy/"


def test_building_one_warns_once_per_process_that_google_has_not_allowed_this(agy, monkeypatch, caplog):
    """`NOT_EXPLICIT`, so the constructor must reach `community._note_constructed`, which warns
    the first time and then stays quiet."""
    monkeypatch.setattr(community, "_noted", set())
    with caplog.at_level("WARNING", logger="rlm_harness.community"):
        agy.lm()
        agy.lm()
    found = [r.getMessage() for r in caplog.records if r.name == "rlm_harness.community"]
    assert len(found) == 1, found
    assert "'agy/'" in found[0] and "community/README.md" in found[0]


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


def test_a_long_prompt_reaches_an_agy_that_is_slow_to_read_it(agy):
    """An awaited call waits for `agy` in slices. Fed through a pipe, a prompt larger than the
    pipe was cut off there on Python 3.11 and 3.12, where a retried `communicate()` does not go
    on writing its input, and `agy` waited for the rest until the deadline. So the prompt is a
    file. The fake reads its stdin late, so the first slice ends with the prompt still unread;
    the control is the prompt's size, well past a pipe's 64 KiB."""
    prompt = "QUOKKA-HEAD " + "x" * 1_000_000 + " QUOKKA-TAIL"
    agy.answer(_ok("late ok\n"), read_delay=0.5)
    assert asyncio.run(agy.lm(timeout_s=30).acall(prompt=prompt)) == ["late ok"]
    assert _stdin_event(agy.calls()[-1])["message"]["content"] == prompt


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


_HEADLINE = {"type": "json_schema",
             "schema": {"type": "object", "properties": {"headline": {"type": "string"}}}}


@pytest.mark.parametrize(("answer", "response_format"), [
    pytest.param(_ok("\n"), None, id="empty-answer"),
    pytest.param(_ok('{"headline": "ok"}\n'), _HEADLINE, id="schema-call-without-structured-output"),
])
def test_bad_model_output_comes_back_as_an_empty_completion_with_a_warning(agy, caplog, answer, response_format):
    """Returned, not raised: raised, dspy wraps it as a non-retryable `LMUnexpectedError` and the
    task ends on one bad answer. Empty, so a schema call's joined `response` (which parses, here)
    is still never what comes back; the warning names the conversation so the cause is findable."""
    agy.answer(answer)
    with caplog.at_level("WARNING", logger="rlm_harness.community.antigravity_lm"):
        response = _complete(agy.lm(), response_format=response_format)
    assert response.message.parts[0].text == ""
    assert "conv-1" in caplog.text


def test_bad_output_is_retried_by_the_task_and_a_failed_call_ends_it(agy):
    """The decision the class docstring states, checked against the predicate `RLMTask` hands
    `run_with_retry`. Bad output reaches the adapter and fails to parse, which the task retries; a
    failed `agy` call is an LM error dspy will not retry, which ends the task at once. Through
    `llm_query`, which calls the LM without an adapter, an empty answer is just ``""``.

    The adapter is the kit's own "json"-mode one, which sends the schema whatever the LM claims;
    dspy's stock `JSONAdapter` falls back to `json_object` for this LM and never reaches the case."""
    from dspy.utils.exceptions import AdapterParseError

    from rlm_harness._dspy_compat import is_fast_fail_lm_error
    from rlm_harness.runtime import _LenientJSONAdapter

    lm = agy.lm()
    agy.answer(_ok('{"answer": "x"}\n'))  # a schema call answered without a structured_output
    with dspy.context(lm=lm, adapter=_LenientJSONAdapter()), pytest.raises(AdapterParseError) as bad:
        dspy.Predict("question -> answer")(question="hi")
    assert "--json-schema" in agy.calls()[-1]["argv"], "the plain path ran, so this tested nothing"
    assert is_fast_fail_lm_error(bad.value) is False, "the task would end on one bad answer"

    agy.answer(_ok("\n"))
    assert lm(prompt="hi") == [""]

    agy.answer(exit=1, result={"status": "ERROR", "response": "", "error": "boom", "usage": {}})
    with dspy.context(lm=lm, adapter=_LenientJSONAdapter()), pytest.raises(Exception) as failed:
        dspy.Predict("question -> answer")(question="hi")
    assert is_fast_fail_lm_error(failed.value) is True, "the task would retry a call agy gave up on"


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
    _assert_gone(int(pidfile.read_text()))


def test_the_deadline_kills_what_agy_started_too(agy):
    """`agy` and everything it started, which is what the docstrings say: killing `agy` alone
    would orphan a process of its own. The fake starts one and the test watches that pid. On
    Windows this is the only test of the `taskkill /T` path, so it must not be skipped there."""
    child_pidfile = agy.tmp / "child"
    agy.answer(_ok(), sleep=30, child_pidfile=str(child_pidfile))
    with pytest.raises(TimeoutError):
        _complete(agy.lm(timeout_s=1.5))
    assert child_pidfile.exists(), "the fake never started its own process, so this measured nothing"
    _assert_gone(int(child_pidfile.read_text()))


def test_a_deadline_that_passed_in_the_queue_never_starts_agy(agy, monkeypatch):
    """Starting `agy` with no time left would still send the request and save a conversation
    before the kill. Driven on the call object, because reaching this through the queue would
    need the deadline to fall between acquiring a slot and starting the process."""
    lm = agy.lm()
    spawned = _count_spawns(monkeypatch)
    T = dspy.lm15
    request = T.Request(model=lm.model, messages=[T.Message(role="user", parts=[T.TextPart(text="hi")])],
                        config=T.Config())
    with pytest.raises(TimeoutError):
        lm._call._run(request, time.monotonic() - 1, None)
    assert spawned == [], "agy was started after the deadline had passed"


class _Interrupt(BaseException):
    """Stands in for KeyboardInterrupt, which pytest would read as the user stopping the run."""


def test_an_interrupt_while_waiting_on_agy_kills_it(agy, monkeypatch):
    """Ctrl-C on the thread running a sync call. On POSIX `agy` runs in its own session, so the
    terminal's SIGINT never reaches it, and before this the interrupt left it running to its
    deadline. The interrupt is raised once `agy` is known to be running."""
    pidfile = agy.tmp / "pid"
    agy.answer(_ok(), sleep=30, pidfile=str(pidfile))
    real = subprocess.Popen.communicate
    waits = []

    def interrupted(self, input=None, timeout=None):
        waits.append(timeout)
        if len(waits) > 1:
            return real(self, input, timeout)  # the kill and the collection that follow
        _wait_for(pidfile.exists, timeout=20, what="the fake agy to start")
        raise _Interrupt

    monkeypatch.setattr(subprocess.Popen, "communicate", interrupted)
    with pytest.raises(_Interrupt):
        _complete(agy.lm(timeout_s=120))
    _assert_gone(int(pidfile.read_text()))


def test_a_call_that_already_collected_its_agy_does_not_signal_that_pid(agy, monkeypatch):
    """Once `agy` is collected its pid is free for another process, and on POSIX the kill goes to
    the process GROUP of that number. So an exception that arrives after the collection (Ctrl-C
    between `agy` exiting and its output being returned) must not kill. The control is the test
    above, where the same exception before the collection does."""
    real = subprocess.Popen.communicate
    waits, killed = [], []

    def collected_then_interrupted(self, input=None, timeout=None):
        waits.append(timeout)
        if len(waits) > 1:
            return real(self, input, timeout)
        real(self, input, None)
        assert self.returncode is not None, "the fake agy was not collected, so this tested nothing"
        raise _Interrupt

    monkeypatch.setattr(subprocess.Popen, "communicate", collected_then_interrupted)
    monkeypatch.setattr(antigravity_lm, "_kill", killed.append)
    with pytest.raises(_Interrupt):
        _complete(agy.lm())
    assert killed == [], "a kill was sent to the pid of an agy already collected"


def test_collecting_a_killed_agy_never_raises_over_the_error_already_on_its_way(monkeypatch):
    """`_reap` runs while another exception is on its way out. Its own `communicate` can fail in
    a state that exception left behind, and that failure must not replace it: the fallback is
    `wait` and closing the pipes."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    proc.kill()

    def broken(self, input=None, timeout=None):
        raise ValueError("flush of closed file")

    monkeypatch.setattr(subprocess.Popen, "communicate", broken)
    antigravity_lm._reap(proc)
    assert proc.returncode is not None, "the killed process was not collected"
    assert proc.stdout.closed and proc.stderr.closed


def test_cancelling_an_awaited_call_kills_agy(agy):
    """`asyncio.to_thread` cannot stop its thread, so before this a cancelled `acall` left `agy`
    running until its own deadline, and `asyncio.run` waited for that thread on the way out. The
    elapsed bound is the half that holds on every platform (the fake sleeps 30 s), the pid the
    POSIX half."""
    pidfile = agy.tmp / "pid"
    agy.answer(_ok(), sleep=30, pidfile=str(pidfile))

    async def main():
        task = asyncio.ensure_future(agy.lm(timeout_s=120).acall(prompt="hi"))
        end = time.monotonic() + 20
        while not pidfile.exists():
            assert time.monotonic() < end, "the fake agy never started"
            await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    started = time.monotonic()
    asyncio.run(main())
    assert time.monotonic() - started < 20, "the cancelled call's agy ran on and asyncio.run waited"
    _assert_gone(int(pidfile.read_text()))


def test_cancelling_an_awaited_call_leaves_the_kill_to_the_calls_own_thread(agy, monkeypatch):
    """The awaiting side only sets a flag, for two reasons. A kill can be slow (`taskkill` on
    Windows, up to 10 s), and on the event loop's thread it would hold every other task for that
    long. And a second thread can signal `agy` just after the call's own thread collected it,
    when the pid is no longer `agy`'s. The stand-in kill waits until the loop has got past the
    cancelled call, which a kill running on the loop's thread never sees happen."""
    pidfile = agy.tmp / "pid"
    agy.answer(_ok(), sleep=30, pidfile=str(pidfile))
    lm = agy.lm(timeout_s=120)
    real_kill, real_complete = antigravity_lm._kill, lm._call.complete
    loop_moved_on = threading.Event()
    seen, kills = {}, []

    def slow_kill(proc):
        kills.append((threading.get_ident(), loop_moved_on.wait(20)))
        real_kill(proc)

    def tracked(request, cancelled=None):
        seen["call_thread"] = threading.get_ident()
        return real_complete(request, cancelled)

    monkeypatch.setattr(antigravity_lm, "_kill", slow_kill)
    monkeypatch.setattr(lm._call, "complete", tracked)

    async def main():
        seen["loop_thread"] = threading.get_ident()
        task = asyncio.ensure_future(lm.acall(prompt="hi"))
        end = time.monotonic() + 20
        while not pidfile.exists():
            assert time.monotonic() < end, "the fake agy never started"
            await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        loop_moved_on.set()

    asyncio.run(main())
    assert kills, "nothing was killed, so this measured nothing"
    assert all(moved_on for _, moved_on in kills), "the event loop was held while agy was being killed"
    assert {thread for thread, _ in kills} == {seen["call_thread"]} != {seen["loop_thread"]}
    _assert_gone(int(pidfile.read_text()))


def test_a_call_cancelled_before_it_starts_never_starts_agy(agy, monkeypatch):
    """A cancel can land in the same poll that a slot frees in, so the thread holds a slot for a
    call nobody waits for. Driven on the call object with the cancel already made, because the
    race itself cannot be scheduled from here."""
    lm = agy.lm()
    spawned = _count_spawns(monkeypatch)
    T = dspy.lm15
    request = T.Request(model=lm.model, messages=[T.Message(role="user", parts=[T.TextPart(text="hi")])],
                        config=T.Config())
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(antigravity_lm._Cancelled):
        lm._call._run(request, time.monotonic() + 60, cancelled)
    assert spawned == [], "agy was started for a call already cancelled"


def test_a_call_whose_coroutine_is_closed_kills_agy(agy):
    """A coroutine closed while it waits gets `GeneratorExit`, never `CancelledError`, so a
    handler for cancellation alone misses it. The coroutine is driven by hand into that state.
    (A loop merely closed with the call pending does NOT get here in time: the interpreter joins
    the worker thread on exit before it collects the coroutine. The class docstring says so.)"""
    pidfile = agy.tmp / "pid"
    agy.answer(_ok(), sleep=30, pidfile=str(pidfile))
    lm = agy.lm(timeout_s=120)
    T = dspy.lm15
    request = T.Request(model=lm.model, messages=[T.Message(role="user", parts=[T.TextPart(text="hi")])],
                        config=T.Config())

    async def main():
        coro = lm._call.acomplete(request)
        coro.send(None)  # runs it up to the await on its worker thread
        end = time.monotonic() + 20
        while not pidfile.exists():
            assert time.monotonic() < end, "the fake agy never started"
            await asyncio.sleep(0.02)
        coro.close()

    started = time.monotonic()
    asyncio.run(main())
    assert time.monotonic() - started < 20, "the closed call's agy ran on and asyncio.run waited"
    _assert_gone(int(pidfile.read_text()))


def test_a_call_cancelled_while_queued_gives_up_its_place_and_never_starts_agy(agy, monkeypatch):
    """Two calls hold both slots at the fake's gate and a third queues behind them. Cancelled
    there, its worker thread must leave the queue at once, WHILE both slots are still held: a
    thread still waiting would sit there until a slot freed (up to the deadline), and
    `asyncio.run` would wait for it on the way out. And it must never start `agy`, which is
    counted at `Popen`, after the slots free too."""
    in_flight, gate = agy.tmp / "in_flight", agy.tmp / "gate"
    in_flight.mkdir()
    agy.answer(_ok(), in_flight_dir=str(in_flight), gate=str(gate), gate_timeout=60)
    lm = agy.lm(timeout_s=120)
    spawned = _count_spawns(monkeypatch)
    queued_thread_done = threading.Event()
    real_complete = lm._call.complete

    def tracked(request, cancelled=None):
        try:
            return real_complete(request, cancelled)
        finally:
            if "queued" in _message_texts(request):
                queued_thread_done.set()

    monkeypatch.setattr(lm._call, "complete", tracked)

    async def main():
        holders = [asyncio.ensure_future(lm.acall(prompt="hold")) for _ in range(2)]
        try:
            end = time.monotonic() + 30
            while len(list(in_flight.iterdir())) < 2:
                assert time.monotonic() < end, "the two holding calls never started"
                await asyncio.sleep(0.02)
            queued = asyncio.ensure_future(lm.acall(prompt="queued"))
            await asyncio.sleep(0.3)
            queued.cancel()
            with pytest.raises(asyncio.CancelledError):
                await queued
            left = await asyncio.to_thread(queued_thread_done.wait, 5)
            assert left, "the cancelled call's thread is still queued behind two held slots"
        finally:
            gate.write_text("open")
            await asyncio.gather(*holders)

    asyncio.run(main())
    assert len(spawned) == 2, f"the cancelled call started agy anyway ({len(spawned)} processes)"
    prompts = [_stdin_event(call)["message"]["content"] for call in agy.calls()]
    assert prompts == ["hold", "hold"]


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
    """The politeness cap, measured on the processes themselves and held open by the fake's gate
    rather than by a sleep. A 0.4 s sleep was shorter than a Windows process start, so on Windows
    two calls never overlapped and the test saw a peak of 1. Waiting for two to be in flight is
    the control (a cap of 1, or none, never gets there); a third process STARTED while they hold
    the gate is the failure, counted at `Popen` so a slow process start cannot hide it."""
    from concurrent.futures import ThreadPoolExecutor

    in_flight, gate = agy.tmp / "in_flight", agy.tmp / "gate"
    in_flight.mkdir()
    agy.answer(_ok(), in_flight_dir=str(in_flight), gate=str(gate), gate_timeout=60)
    lm = agy.lm()
    spawned = _count_spawns(monkeypatch=agy.monkeypatch)
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(_complete, lm) for _ in range(5)]
        try:
            _wait_for(lambda: len(list(in_flight.iterdir())) >= 2, timeout=30,
                      what="two agy processes to be in flight")
            time.sleep(0.5)  # room for a third Popen, were the cap broken
            started = len(spawned)
        finally:
            gate.write_text("open")
        for future in futures:
            future.result(timeout=60)
    assert started == 2, f"{started} agy processes were started while two held the gate"
    assert len(agy.calls()) == 5


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
def test_every_call_starts_agy_without_a_variable_that_moves_the_bill(agy, monkeypatch, name):
    """Read when each call starts, not when the LM was built. The LM is built BEFORE the variable
    is set, the `load_dotenv()`-after-`configure()` order a constructor-only check let through.
    The control is `allow_api_key=True`, which must pass it on, and the fake's own variable,
    which must survive the filtering."""
    lm = agy.lm()
    monkeypatch.setenv(name, "set-after-construction")
    passing = agy.lm(allow_api_key=True)
    _complete(lm)
    assert name not in agy.calls()[-1]["env_keys"], f"agy was started with {name} set"
    assert "FAKE_AGY_CONFIG" in agy.calls()[-1]["env_keys"]
    _complete(passing)
    assert name in agy.calls()[-1]["env_keys"], "allow_api_key=True no longer passes it through"


@pytest.mark.parametrize("timeout_s", [None, float("inf"), float("nan"), 0, -1.0, "600"],
                         ids=["none", "infinite", "nan", "zero", "negative", "string"])
def test_a_deadline_that_is_not_a_finite_number_above_zero_is_refused_when_built(agy, timeout_s):
    """`None` used to raise `TypeError` at the first call, and an infinite deadline
    `OverflowError` only once both slots were busy. The control is every other test's deadline."""
    with pytest.raises(ValueError, match="timeout_s"):
        agy.lm(timeout_s=timeout_s)


def test_n_above_one_is_one_agy_call_per_candidate_and_warns(agy, caplog):
    with caplog.at_level("WARNING", logger="rlm_harness.community.antigravity_lm"):
        lm = agy.lm(n=3)
    assert "3 calls per LM call" in "\n".join(r.getMessage() for r in caplog.records)
    assert len(lm(prompt="hi")) == 3
    assert len(agy.calls()) == 3, "n=3 no longer costs three agy calls, so the warning is wrong"
    # The control: a plain call is one process.
    assert len(agy.lm()(prompt="hi")) == 1
    assert len(agy.calls()) == 4


@pytest.mark.parametrize("kwargs", [{}, {"n": 1}], ids=["absent", "explicit-one"])
def test_no_warning_when_n_cannot_cost_anything(agy, caplog, kwargs):
    with caplog.at_level("WARNING", logger="rlm_harness.community.antigravity_lm"):
        agy.lm(**kwargs)
    assert not [r for r in caplog.records if "calls per LM call" in r.getMessage()]


def _no_agy_anywhere(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path / "empty-path"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))


def test_a_missing_agy_says_how_to_get_one(monkeypatch, tmp_path):
    _no_agy_anywhere(monkeypatch, tmp_path)
    with pytest.raises(FileNotFoundError, match="install.sh"):
        community.AntigravityLM("m")
    with pytest.raises(FileNotFoundError, match="not an executable file"):
        community.AntigravityLM("m", executable=tmp_path / "nope")


@pytest.mark.skipif(os.name == "nt", reason="the install script's ~/.local/bin is the POSIX layout")
def test_agy_is_found_where_the_install_script_puts_it(monkeypatch, tmp_path):
    """The official install script's TARGET_DIR is `~/.local/bin`, which is off PATH until
    `agy install` adds it: the state this adapter was first run in."""
    _no_agy_anywhere(monkeypatch, tmp_path)
    binary = tmp_path / "home" / ".local" / "bin" / "agy"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)
    assert community.AntigravityLM("m")._call._argv0 == [str(binary)]
    on_path = tmp_path / "empty-path" / "agy"
    on_path.parent.mkdir()
    on_path.write_text("#!/bin/sh\n", encoding="utf-8")
    on_path.chmod(0o755)
    assert community.AntigravityLM("m")._call._argv0 == [str(on_path)], "PATH must win"
