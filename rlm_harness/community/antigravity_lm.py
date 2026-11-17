"""Run rlm-harness on a Google Antigravity sign-in through the `agy` CLI.

`AntigravityLM` is a `dspy.LM` whose completions run through `agy`, the Antigravity CLI, on the
account you signed it in with. It is a community route (`rlm_harness/community/README.md` says
what Google has and has not said about this use, and when that was read): exported from
`rlm_harness.community` only, outside the kit's SemVer promise, and it logs one warning per
process when built, because Google has not explicitly allowed this use. It goes in through the
same public seam as `ClaudeAgentLM`, `configure(main_lm=..., sub_lm=...)`, or through
`configure()`'s routing of an `"agy/<model id>"` model string by its `community.ROUTES` entry,
and it is built the same way: an ENGINE (`complete(Request) -> Response`, passed as
`dspy.LM(engine=...)`), never a `forward()` subclass, because dspy 3.4 deprecated that interface
and schedules its removal for 3.5.

Each LM call is one `agy` process in print mode, with no tools and none of the user's settings:

    agy --input-format stream-json --output-format stream-json --disable-slash-commands \\
        --agent <generated> --model <id> [--json-schema <file>]

**What every choice below rests on was MEASURED against `agy` 1.4.12 on 2026-11-03**, signed in
with a personal Google account, and each is an executable-but-not-from-here claim in the
`docs/VERIFY.md` sense: it needs a signed-in `agy`, which CI does not have. The probes are
described with each claim so the next version can be checked the same way.

- **A pure completion is a generated custom agent, not a flag.** `agy` has no option that removes
  its tools or its system prompt. A custom agent in `<cwd>/.agents/agents/<name>.md` with
  `excludeDefaultComponents: true` and `tools: []` does all three things this adapter needs, each
  measured against a control: the model can read no file (asked to read a file in its cwd it
  answered that it had no tool, where the default agent read it), the user's global rules
  (`~/.gemini/GEMINI.md`) and the workspace's (`AGENTS.md`) stay out (the default agent obeyed
  both; this one obeyed neither), and the agent's Markdown body BECOMES the system prompt (a body
  of "answer BANANA" was obeyed over the user's message). A trivial prompt cost 382 input tokens
  this way against 6,900 on the default agent.
- **Structured output needs the `finish` tool, and only that.** `--json-schema` is enforced
  through a tool named `finish`. With `tools: []` the flag is SILENTLY ignored: the run reports
  `SUCCESS` after four turns of prose with no `structured_output` at all. With `tools: [finish]`
  it validates against the schema (a pydantic-style one with `$defs`, `$ref` and
  `additionalProperties: false` included), retries in the same call when validation fails, and
  returns the validated object as `structured_output`. A schema call therefore gets
  `tools: [finish]` and a plain one gets `tools: []`.
- **The answer to a structured call is `structured_output`, never `response`.** `response` JOINS
  the text of every attempt in the call, the prose and each rejected JSON included, so on a call
  where validation retried it is several documents concatenated.
- **The prompt goes on stdin, never in argv.** Linux refuses a single argument longer than
  131,072 bytes (`E2BIG`, measured at 131,073), and an RLM planner prompt can exceed that. A
  144,610-byte prompt sent as a stream-json user event arrived whole: the model answered with a
  word stated only on its last line. That was measured with stdin as a pipe. Stdin is now a
  `tempfile.TemporaryFile` holding the same event (`_AgyCall._run` says why), and on `agy` 1.5.9
  on 2026-11-15 a 147,449-byte prompt sent that way was answered the same. So every call writes
  its prompt to disk while it runs: the file is created in `tempfile.gettempdir()` (`TMPDIR`,
  `TEMP` or `TMP` when set) with the same rules as `mkstemp()`, and is deleted when the call ends.
  In Python's words, under Unix its directory entry "is either not created at all or is removed
  immediately after the file is created"; on other platforms, Windows included, it has a visible
  name in that directory until then.
- **Failure is `status != "SUCCESS"`, a `finished_reason`, or a non-zero exit.** A bad model id
  exits 1 with `status: "ERROR"` and an `error` string naming the models the account can use.
  `finished_reason` is set only when a turn ended early (`print_timeout`, `interrupted`,
  `error`), per `agy changelog`, which also documents exit code 3 and an `AGY_ERROR: {...}` stderr
  line for a model or quota failure. That last path was not reproduced here; the adapter needs
  nothing from it beyond the exit code and keeps stderr's tail in its error message.
- **Two environment variables are documented to move a call off the sign-in.** Google's install
  page (https://antigravity.google/docs/install/) documents `GEMINI_API_KEY`, which bills an API
  key once `settings.json` sets `modelProvider: "gemini"` and "on its own has no effect", and
  `AGY_ADC_AUTH=true`, which signs in with Application Default Credentials and bills a Google
  Cloud project. The binary also contains `GOOGLE_API_KEY`, `GOOGLE_GENAI_USE_VERTEXAI`,
  `GOOGLE_GENAI_USE_ENTERPRISE`, `GOOGLE_GEMINI_BASE_URL` and `GOOGLE_VERTEX_BASE_URL`. Each was
  set ALONE to a bogus value on a signed-in `agy` 1.5.0 on 2026-11-11 (a bogus `GEMINI_API_KEY`
  alone too), and every one of those calls still answered normally; the same page says
  `GOOGLE_API_KEY` "has no effect". That rules out each one acting by itself on this path. It
  does not rule out a combination, such as a Vertex switch together with a project and ADC, which
  was not tried; `GEMINI_API_KEY` is itself a variable that only acts in combination.

Settle any of these again with a signed-in `agy` from an empty directory:
`agy --output-format json --agent <name> --model <id> -p "..."`, with the agent file above
(prefix a variable, e.g. `GOOGLE_GENAI_USE_VERTEXAI=true agy ...`, for the environment probe).

**Things this adapter does NOT control**, stated so nobody assumes otherwise:

- **Every call is saved as an Antigravity conversation**, under `~/.gemini/antigravity-cli`
  (`conversations/<id>.db` and `brain/<id>/`). Measured: about 400 KB per call, prompt included,
  so a run of fifty calls leaves roughly 20 MB of transcripts holding whatever the task read.
  `agy` has no option to skip that, and this adapter does not delete another program's private
  state. The conversation id is the response id, so a call can be found again.
- **`agy` retries on its own.** Its changelog says it waits out a server-requested delay up to
  30s and stops at once on a daily or billing cap. So this adapter adds no retry rung of its own,
  where `ClaudeAgentLM` adds one.
- **Sampling controls do not exist here.** `agy` takes none, so `temperature`, `top_p` and
  `max_tokens` are tolerated and ignored. `n` is not ignored: see the class docstring.
- **AI credits past the plan's quota.** `agy`'s `useG1Credits` setting "uses your AI credits for
  model requests after your plan's quota runs out" (https://antigravity.google/docs/cli/using/).
  It lives in the user's `settings.json`, which this adapter neither reads nor overrides.

Google's own words on print mode and on third-party access are quoted, with links and the date
they were read, in this route's row of `rlm_harness/community/README.md`. The use this adapter is
built for is print mode on your own machine, signed in with your own account, for your own use.

The `agy` executable is found on `PATH`, then at `~/.local/bin/agy`, which is where the official
install script puts it (`TARGET_DIR="$HOME/.local/bin"`; settle with
`curl -fsSL https://antigravity.google/cli/install.sh | grep TARGET_DIR=`) and which is not on
`PATH` until `agy install` adds it. Pass `executable=` to choose another.
"""

from __future__ import annotations

import asyncio
import atexit
import hashlib
import json
import logging
import math
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import dspy

from . import _note_constructed, _route

logger = logging.getLogger(__name__)

#: The sentinel `AntigravityLM` stamps its own `dspy.LM.model` with, read from this route's entry
#: in `community.ROUTES`, the table `runtime.configure()` routes an `"agy/<id>"`
#: `main_model`/`sub_model` string by. A consumer composing its own model-string convention reads
#: it from here instead of hardcoding the literal a second time.
AGY_PREFIX = _route("AntigravityLM").prefix

#: Environment variables that move `agy` off the signed-in account onto another bill, removed
#: from every call's environment unless `allow_api_key=True`. `GEMINI_API_KEY` bills the key when
#: `settings.json` selects `modelProvider: "gemini"`; `AGY_ADC_AUTH` signs in with Application
#: Default Credentials, which bills a Google Cloud project. The other auth-looking names in the
#: binary were measured to have no effect on this path (module docstring).
_BILLING_ENV = ("GEMINI_API_KEY", "AGY_ADC_AUTH")

#: How often the thread running an awaited call checks whether its awaiting task was cancelled,
#: both while the call is queued for a slot and while `agy` runs.
_CANCEL_POLL_S = 0.05

#: Politeness cap, shared by every instance in the process (the main and sub seat alike).
#: dspy.RLM's `llm_query_batched` would otherwise fan eight threads of CLI spawns at one personal
#: account, the same reason `ClaudeAgentLM` caps at 2.
_MAX_CONCURRENT = 2
_SEMAPHORE = threading.BoundedSemaphore(_MAX_CONCURRENT)

#: How much of stderr an error message carries. Enough for an `AGY_ERROR` line and an `error:`
#: line; a whole stderr would put an unbounded provider payload into an exception.
_STDERR_TAIL = 2000


def _resolve_executable(executable: str | os.PathLike[str] | None) -> list[str]:
    """The argv prefix that starts `agy`, or a `FileNotFoundError` saying how to get one.

    A list rather than a path only so the tests can put a fake `agy` here, behind the public
    `executable=`, which takes one path.
    """
    if executable is not None:
        found = shutil.which(os.fspath(executable))
        if found is None:
            raise FileNotFoundError(f"executable={os.fspath(executable)!r} is not an executable file")
        return [found]
    found = shutil.which("agy")
    if found is not None:
        return [found]
    # The official install script's TARGET_DIR. Off PATH until `agy install` adds it, which is how
    # this adapter was first run: the binary was installed and `which agy` found nothing.
    fallback = Path.home() / ".local" / "bin" / "agy"
    if fallback.is_file() and os.access(fallback, os.X_OK):
        return [str(fallback)]
    raise FileNotFoundError(
        "AntigravityLM needs the `agy` CLI and found it neither on PATH nor at ~/.local/bin/agy. "
        "Install it (curl -fsSL https://antigravity.google/cli/install.sh | bash), sign in once by "
        "running `agy`, or pass executable= with its path."
    )


class _Workspace:
    """The private directory every call runs in: the generated agents, the schemas, nothing else.

    One per process, created on first use and removed at exit. It is the call's working
    directory, so it is also `agy`'s workspace, and it holds no file a model could be shown.
    Agent and schema files are CONTENT-ADDRESSED: two calls that need the same system prompt
    share one file, and concurrent writers of the same name write the same bytes.
    """

    def __init__(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="rlm-agy-"))
        self._agents = self.root / ".agents" / "agents"
        self._agents.mkdir(parents=True)
        self._schemas = self.root / ".rlm-schemas"
        self._schemas.mkdir()
        self._lock = threading.Lock()
        atexit.register(shutil.rmtree, self.root, True)

    def _write_once(self, path: Path, text: str) -> None:
        with self._lock:
            if path.exists():
                return
            staging = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
            staging.write_text(text, encoding="utf-8")
            os.replace(staging, path)

    def agent(self, system: str | None, *, structured: bool) -> str:
        """The name of an agent whose system prompt is exactly ``system`` (empty for ``None``).

        The body is the caller's text verbatim: this adapter adds no instruction of its own to
        what the model reads. `tools` is `[finish]` for a schema call because `--json-schema` is
        enforced through that tool, and `[]` otherwise (see the module docstring).

        The frontmatter's `description` is the only text here the kit writes. It does not appear
        to reach the model: an agent whose description said to answer only in French, with an
        empty body, was answered in English (one call on `agy` 1.4.12, 2026-11-03, so an
        observation rather than a guarantee). Keep it a label either way.
        """
        tools = "[finish]" if structured else "[]"
        body = system or ""
        digest = hashlib.sha256(f"{tools}\0{body}".encode()).hexdigest()[:24]
        name = f"rlm-{digest}"
        self._write_once(
            self._agents / f"{name}.md",
            "---\n"
            f"name: {name}\n"
            "description: Plain completion.\n"
            "excludeDefaultComponents: true\n"
            f"tools: {tools}\n"
            "---\n"
            f"{body}",
        )
        return name

    def schema(self, schema: dict[str, Any]) -> Path:
        """A file holding ``schema``, for `--json-schema`. A file rather than an argument, so a
        schema is never subject to argv's length limit or to a shell's reading of it."""
        text = json.dumps(schema, sort_keys=True)
        path = self._schemas / f"{hashlib.sha256(text.encode()).hexdigest()[:24]}.json"
        self._write_once(path, text)
        return path


_WORKSPACE: _Workspace | None = None
_WORKSPACE_LOCK = threading.Lock()


def _workspace() -> _Workspace:
    # Module-level for the same reason as `claude_agent_lm._bridge`: `copy_lm` makes a SHALLOW
    # copy, so state held on an instance would be shared by reference with no owner.
    global _WORKSPACE
    with _WORKSPACE_LOCK:
        if _WORKSPACE is None or not _WORKSPACE.root.is_dir():
            _WORKSPACE = _Workspace()
        return _WORKSPACE


def _user_prompt(messages: list[dict[str, Any]]) -> str:
    """Flatten dspy's message list into the ONE user message a print-mode turn takes.

    dspy.RLM sends one user message per call on both seats, so the single-message case is the
    live one; the flatten is the general case for a consumer's own `Predict` with demos or
    history, and matches `ClaudeAgentLM`'s.
    """
    if len(messages) == 1:
        return str(messages[0].get("content", ""))
    parts = [f"{m.get('role', 'user').capitalize()}: {m.get('content', '')}" for m in messages]
    parts.append("Assistant:")
    return "\n\n".join(parts)


def _message_text(message: Any) -> str:
    """The text of one ``lm15.Message``, refusing a part a text prompt cannot carry."""
    texts: list[str] = []
    for part in message.parts:
        text = getattr(part, "text", None)
        if text is None:
            raise TypeError(
                f"AntigravityLM cannot send a {type(part).__name__} part: each call sends one text "
                f"prompt. Use a plain dspy.LM for multimodal input."
            )
        texts.append(str(text))
    return "".join(texts)


def _request_to_prompts(request: Any) -> tuple[str | None, str]:
    """``(system prompt, user prompt)`` from an ``lm15.Request``. lm15 carries the system prompt
    in its own field (the planner seat arrives with one, the sub-LM seat with ``None``)."""
    messages = [{"role": m.role, "content": _message_text(m)} for m in (request.messages or ())]
    return (request.system or None, _user_prompt(messages))


def _schema_of(response_format: Any) -> dict[str, Any] | None:
    """The JSON Schema to enforce, or ``None`` for a plain completion.

    dspy canonicalises the kit's pydantic `response_format` into ``{"type": "json_schema",
    "schema": {...}}`` before the engine sees it, so the dict is the live case; a class is read
    too, for a caller assembling a `Request` by hand. A schema whose root is not an object is
    dropped rather than passed: `agy` refuses one at startup with exit code 1, per its changelog.
    """
    if response_format is None:
        return None
    if isinstance(response_format, dict):
        schema = response_format.get("schema")
    else:
        to_schema = getattr(response_format, "model_json_schema", None)
        schema = to_schema() if callable(to_schema) else None
    if isinstance(schema, dict) and schema.get("type") == "object":
        return schema
    return None


def _token_count(usage: dict[str, Any], key: str) -> int:
    value = usage.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _start_options() -> dict[str, Any]:
    # POSIX: its own session, so `_kill` can end `agy` and everything it started with one
    # `killpg`. The cost is that the terminal's Ctrl-C no longer reaches it (see the class
    # docstring). Windows needs no group for that: `taskkill /T` walks the tree instead.
    # Measured on `agy` 1.5.0 (Linux, 2026-11-11): a print-mode call ran as ONE process, with
    # nothing else in its session, so this guards a version that starts something rather than
    # one observed to.
    if os.name == "nt":
        return {}
    return {"start_new_session": True}


def _kill(proc: subprocess.Popen) -> None:
    """End `agy` and every process it started, not only `agy`."""
    if os.name == "nt":
        # By absolute path: a bare name is searched for in the current directory first.
        taskkill = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "taskkill.exe")
        try:
            ended = subprocess.run(
                [taskkill, "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True, timeout=10, check=False,
            ).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            ended = False
        if not ended:
            try:
                proc.kill()
            except OSError:
                pass
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _reap(proc: subprocess.Popen) -> None:
    """Collect a killed `agy`, so its pipes close and its pid is released.

    Best effort, because it runs while another exception is on its way out and must never replace
    it, whatever state that exception left the process in. `communicate` can itself raise from
    some of those states, so `wait` and closing the pipes is the fallback.
    """
    try:
        proc.communicate(timeout=5)
        return
    except (subprocess.TimeoutExpired, ValueError, OSError):
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    for pipe in (proc.stdin, proc.stdout, proc.stderr):
        if pipe is not None:
            try:
                pipe.close()
            except OSError:
                pass


class _Cancelled(Exception):
    """Ends the worker thread of a call whose awaiting task was cancelled. Nobody reads it."""


def _tail(text: str) -> str:
    text = text.strip()
    return text if len(text) <= _STDERR_TAIL else "..." + text[-_STDERR_TAIL:]


class _AgyCall:
    """One `agy` call, as both engines need it: ``Request`` in, ``lm15.Response`` out."""

    def __init__(
        self, *, model: str, model_id: str, argv0: list[str], timeout_s: float, allow_api_key: bool
    ) -> None:
        self._model = model          # the trace label, "agy/<id>"
        self._model_id = model_id    # what `--model` receives
        self._argv0 = argv0
        # End-to-end deadline per call, INCLUDING time queued behind the concurrency cap.
        self._timeout_s = timeout_s
        self._allow_api_key = allow_api_key

    def _timed_out(self) -> TimeoutError:
        return TimeoutError(f"agy call timed out after {self._timeout_s}s")

    def _env(self) -> dict[str, str] | None:
        """`agy`'s environment, read when the call starts rather than when the LM was built, so
        a key set in between (a `load_dotenv()` after `configure()`) cannot move the bill."""
        if self._allow_api_key:
            return None
        return {k: v for k, v in os.environ.items() if k.upper() not in _BILLING_ENV}

    def complete(self, request: Any, cancelled: threading.Event | None = None) -> Any:
        """``Engine.complete``: the sub-LM seat, which dspy drives from worker threads.
        ``cancelled`` is passed only by :meth:`acomplete`, which sets it when its await ends
        without a result. This thread is the only one that reads it, and the only one that kills
        or collects the call's `agy`."""
        deadline = time.monotonic() + self._timeout_s
        if not self._acquire(deadline, cancelled):
            raise self._timed_out()
        try:
            return self._run(request, deadline, cancelled)
        finally:
            _SEMAPHORE.release()

    def _acquire(self, deadline: float, cancelled: threading.Event | None) -> bool:
        if cancelled is None:
            return _SEMAPHORE.acquire(timeout=self._timeout_s)
        # Sliced, so a call cancelled while it waits for a slot gives up its place in the queue.
        while not cancelled.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            if _SEMAPHORE.acquire(timeout=min(remaining, _CANCEL_POLL_S)):
                return True
        raise _Cancelled()

    async def acomplete(self, request: Any) -> Any:
        """``AsyncEngine.complete``'s body: the planner seat. The call runs on a worker thread
        and is awaited, so the loop it was awaited on keeps turning while `agy` runs. Leaving the
        await any way but its result kills that `agy`: a cancellation, and also `GeneratorExit`
        from a coroutine closed while it waits. A coroutine left pending on a loop that is merely
        closed is NOT closed in time: the interpreter joins this worker thread on exit before it
        collects that coroutine (measured, see the class docstring)."""
        cancelled = threading.Event()
        try:
            return await asyncio.to_thread(self.complete, request, cancelled)
        except BaseException:
            # A flag and nothing more: the worker thread reads it within `_CANCEL_POLL_S` and does
            # the killing. A kill from here would run on the event loop's thread, which on
            # Windows waits for `taskkill`, and it could signal a pid the worker had just
            # collected. After an ordinary failure the call is over and nothing reads the flag.
            cancelled.set()
            raise

    def _run(self, request: Any, deadline: float, cancelled: threading.Event | None) -> Any:
        system, prompt = _request_to_prompts(request)
        schema = _schema_of(getattr(request.config, "response_format", None))
        workspace = _workspace()
        argv = [
            *self._argv0,
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--disable-slash-commands",
            "--agent", workspace.agent(system, structured=schema is not None),
            "--model", self._model_id,
        ]
        if schema is not None:
            argv += ["--json-schema", str(workspace.schema(schema))]
        event = json.dumps({"event": "user", "message": {"role": "user", "content": prompt}})
        if deadline - time.monotonic() <= 0:
            # Queued until the deadline: starting `agy` now would still send the request and save
            # a conversation before the kill.
            raise self._timed_out()
        if cancelled is not None and cancelled.is_set():
            # Cancelled while queued, including in the same poll a slot freed in: the slot is no
            # reason to start `agy`.
            raise _Cancelled()
        # The prompt is `agy`'s stdin as a FILE, not a pipe this thread feeds. An awaited call
        # waits in slices, and a retried `communicate()` goes on writing the input its first call
        # did not finish only from Python 3.13 on. Fed through a pipe, a prompt larger than the
        # pipe was cut off there and `agy` waited for the rest until the deadline: with a pipe
        # here, `test_a_long_prompt_reaches_an_agy_that_is_slow_to_read_it` fails on 3.11.17 and
        # 3.12.15 and passes on 3.13.17.
        #
        # The `try` sits inside the `with`, directly after `Popen`, as in `subprocess.run`: closing
        # the file first would put a step between starting `agy` and the code that kills it, and
        # an exception there (Ctrl-C, a failed close) would leave `agy` running. The file stays
        # open until `agy` is done, which costs nothing, since `agy` holds its own copy of it.
        with tempfile.TemporaryFile() as prompt_file:
            prompt_file.write((event + "\n").encode("utf-8"))
            prompt_file.seek(0)
            proc = subprocess.Popen(
                argv,
                stdin=prompt_file,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=workspace.root,
                env=self._env(),
                **_start_options(),
            )
            try:
                out, err = self._wait(proc, deadline, cancelled)
            except BaseException:
                # The deadline, a cancelled awaiting task, Ctrl-C on this thread, anything else:
                # on POSIX `agy` is in its own session, so nothing but this would stop it. No
                # other thread collects `proc`, so one still uncollected here still owns its pid.
                if proc.returncode is None:
                    _kill(proc)
                _reap(proc)
                raise
        return self._response(
            proc.returncode,
            out.decode("utf-8", errors="replace"),
            err.decode("utf-8", errors="replace"),
            structured=schema is not None,
        )

    def _wait(
        self, proc: subprocess.Popen, deadline: float, cancelled: threading.Event | None
    ) -> tuple[bytes, bytes]:
        """`agy`'s stdout and stderr once it exits. An awaited call waits in slices, so the
        thread that owns ``proc`` sees a cancellation within `_CANCEL_POLL_S`; the caller kills
        on whatever this raises."""
        while True:
            if cancelled is not None and cancelled.is_set():
                raise _Cancelled()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise self._timed_out()
            try:
                return proc.communicate(
                    timeout=remaining if cancelled is None else min(remaining, _CANCEL_POLL_S)
                )
            except subprocess.TimeoutExpired:
                pass

    def _response(self, returncode: int, stdout: str, stderr: str, *, structured: bool) -> Any:
        result: dict[str, Any] | None = None
        for line in stdout.splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict) and event.get("event") == "result" and isinstance(event.get("result"), dict):
                result = event["result"]
        if result is None:
            raise RuntimeError(f"agy exited {returncode} without a result event: {_tail(stderr)!r}")
        status = result.get("status")
        reason = result.get("finished_reason")
        if returncode != 0 or status != "SUCCESS" or reason:
            detail = result.get("error") or _tail(stderr)
            raise RuntimeError(
                f"agy call failed (exit {returncode}, status {status!r}, finished_reason "
                f"{reason!r}): {detail!r}"
            )
        conversation = str(result.get("conversation_id") or "") or uuid.uuid4().hex
        # Bad model OUTPUT is returned, not raised: an empty completion, as a litellm-backed LM
        # returns for an empty message, so an adapter rejects it as `AdapterParseError` and the
        # task retries it. Raised, dspy would wrap it as a non-retryable `LMUnexpectedError` and
        # the task would end on one bad answer. The warning keeps the cause next to the effect.
        if structured:
            answer = result.get("structured_output")
            if isinstance(answer, dict):
                text = json.dumps(answer)
            else:
                # Never fall back to `response`: on a schema call it joins every attempt,
                # rejected ones included, so it is not an answer dspy could parse safely.
                logger.warning(
                    "agy answered a schema call without a structured_output (conversation %s); "
                    "returning an empty completion for the adapter to reject.",
                    conversation,
                )
                text = ""
        else:
            text = result.get("response")
            text = text if isinstance(text, str) else ""
            # `agy` ends every response with one newline of its own.
            text = text.removesuffix("\n")
            if not text:
                logger.warning(
                    "agy returned an empty response (conversation %s); returning an empty "
                    "completion, which an adapter rejects and llm_query passes on as ''.",
                    conversation,
                )

        lm15 = dspy.lm15
        usage = result.get("usage")
        if isinstance(usage, dict) and usage:
            prompt_tokens = _token_count(usage, "input_tokens")
            completion_tokens = _token_count(usage, "output_tokens")
            reported_usage = lm15.Usage(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )
            # `output_tokens` already INCLUDES the thinking tokens (measured: 52 output, of which
            # 50 thinking, for a one-word answer), so thinking is kept beside it, never added.
            # Whether `input_tokens` includes `cache_read_tokens` is unmeasured: every probe
            # reported a cache read of zero. Both ride along verbatim rather than being guessed into
            # the totals.
            extras = {k: usage[k] for k in ("thinking_tokens", "cache_read_tokens") if k in usage}
        else:
            # Absent is not zero: an all-None Usage produces no tracker entry, as in ClaudeAgentLM.
            reported_usage = lm15.Usage()
            extras = {}
        return lm15.Response(
            id=conversation,
            model=self._model,
            message=lm15.Message(role="assistant", parts=[lm15.TextPart(text=text)]),
            finish_reason="stop",
            usage=reported_usage,
            provider_data=extras or None,
        )


class _AgyEngine:
    """``dspy.clients.engines.Engine``: the sync surface over :class:`_AgyCall`."""

    def __init__(self, call: _AgyCall) -> None:
        self._call = call

    def complete(self, request: Any) -> Any:
        return self._call.complete(request)


class _AsyncAgyEngine:
    """``dspy.clients.engines.AsyncEngine``: the async surface over :class:`_AgyCall`."""

    def __init__(self, call: _AgyCall) -> None:
        self._call = call

    async def complete(self, request: Any) -> Any:
        return await self._call.acomplete(request)


class AntigravityLM(dspy.LM):
    """A dspy LM whose completions run through the `agy` CLI on an Antigravity sign-in.

    Satisfies both rlm-harness seats: the planner arrives on the async engine, the sub-LM seat on
    the sync one from `llm_query[_batched]` worker threads, and both run the same `agy` call
    under one process-wide concurrency cap of 2. Works under `intercept_sub_lm` unchanged.

    `model` is an id `agy models` lists for your account, such as one of its Gemini variants; the
    list depends on the plan. The trace label becomes `agy/<model>`.

    **Google has not explicitly allowed this use**, so the route is `NOT_EXPLICIT` in
    `community.ROUTES` and the first instance built in a process logs a warning pointing at
    `rlm_harness/community/README.md`, which quotes what Google has said.

    **`n` costs calls.** dspy runs one engine call per candidate, so `n=k` is k `agy` processes
    against one account. The constructor warns above 1, as `ClaudeAgentLM`'s does.

    **`max_tokens` is carried and not applied.** `agy` has no output cap, so like `ClaudeAgentLM`
    this LM tolerates and ignores `max_tokens=`. `applied_lm_budget` still reports one passed here,
    and `completion_tokens` can exceed it; the `agy/` prefix on the model key is how a trace reader
    tells.

    **A connection kwarg is refused** (`api_key`, `base_url`, `timeout`, ...): dspy raises
    `ValueError` for one on an LM with a custom engine, because this LM talks to a local CLI, not
    an endpoint. `timeout_s=` is the call deadline: end to end, INCLUDING time queued behind the
    concurrency cap. It must be a finite number above 0, or the constructor raises `ValueError`.
    On expiry `agy` and every process it started are killed (one `killpg` on POSIX,
    `taskkill /T` on Windows) and `TimeoutError` is raised.

    **What stops a running call.** Its deadline. For an awaited `acall`, leaving the await any way
    but its result: a cancelled task, `asyncio.wait_for`, a coroutine closed while it waits; a call
    cancelled while still queued for a slot never starts `agy`. NOT a loop that is closed with the
    call still pending, as a driver using `loop.run_until_complete` leaves it after Ctrl-C:
    nothing cancels the call, and the interpreter waits for its `agy` on exit (measured: 8.5 s
    for a fake `agy` that slept 8 s, Ctrl-C at 1 s). `asyncio.run` cancels it, so drive a run with
    that, as `RLMTask.run` does. For a sync call, any exception raised in the
    thread running it, a `KeyboardInterrupt` included. What does NOT stop one is Ctrl-C reaching
    `agy` itself: on POSIX `agy` runs in its own session so one signal can kill its whole group,
    and that keeps the terminal's SIGINT from it. So under `asyncio.run`, which is how
    `RLMTask.run` drives a run, the FIRST Ctrl-C cancels the planner's awaited call and kills its
    `agy`, but a sub-LM call does not see it: `llm_query` runs on the loop's thread inside the
    sandbox step, where `asyncio.run` only schedules a cancel, and `llm_query_batched` runs on
    worker threads. Such a call runs until it answers or meets its deadline. A second Ctrl-C
    raises `KeyboardInterrupt` on the main thread and kills a call running there.

    **The billing guard applies to every call.** `GEMINI_API_KEY` and `AGY_ADC_AUTH` can move `agy`
    onto an API key or a Google Cloud project, and nothing in a call's output says which account
    paid. Each call starts `agy` with both removed from the environment as it is at that moment,
    so one set after construction is removed too. `allow_api_key=True` passes them through.

    **Which failures end the task.** A deadline, a status other than `SUCCESS`, a turn cut short,
    a non-zero exit and a missing result event raise `TimeoutError` / `RuntimeError`. dspy wraps
    those as `LMUnexpectedError`, which it does not retry. On the PLANNER seat the kit's task retry
    treats that as fatal (`_dspy_compat.is_fast_fail_lm_error`), the same as `ClaudeAgentLM`:
    `agy` already did its own rate-limit retries, so another attempt resends a call it gave up on.
    On the SUB-LM seat dspy.RLM catches it instead and shows it to the model: `llm_query` raises
    it inside the sandbox code, and `llm_query_batched` returns it as an `"[ERROR] ..."` string.

    Bad model OUTPUT is NOT raised. An empty answer, or a schema call answered without a
    `structured_output`, comes back as an empty completion with a warning naming the
    conversation, which is what a litellm-backed LM returns for an empty message. On the planner
    seat an adapter rejects it as `AdapterParseError`, so the task retries it within
    `max_retries`; on the sub-LM seat the code receives `""`, from `llm_query_batched` too, where a
    raised error used to become `"[ERROR] ..."`. With `cache=True` (off by default) dspy caches
    that empty completion like any other, so a retry sending the identical request replays it
    rather than asking `agy` again, as it would for a litellm LM. Tests pin both halves.
    """

    def __init__(
        self,
        model: str,
        *,
        timeout_s: float = 600.0,
        executable: str | os.PathLike[str] | None = None,
        allow_api_key: bool = False,
        **kwargs: Any,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("AntigravityLM needs a model id: run `agy models` to list yours")
        model = model.strip()
        finite = isinstance(timeout_s, (int, float)) and math.isfinite(timeout_s)
        if not finite or timeout_s <= 0:
            # Here rather than at the first call: `None` would raise `TypeError` there, and an
            # infinite one `OverflowError` only once both slots were busy.
            raise ValueError(
                "AntigravityLM timeout_s must be a finite number of seconds above 0, "
                f"got {timeout_s!r}"
            )
        argv0 = _resolve_executable(executable)
        _note_constructed("AntigravityLM")
        n = kwargs.get("n")
        if isinstance(n, int) and not isinstance(n, bool) and n > 1:
            logger.warning(
                "AntigravityLM(n=%d): each candidate is a separate agy call, so this costs %d calls "
                "per LM call. Pass n=1, or omit it, unless you mean to sample %d times.",
                n, n, n,
            )
        full_model = f"{AGY_PREFIX}{model}"
        call = _AgyCall(
            model=full_model, model_id=model, argv0=argv0, timeout_s=timeout_s,
            allow_api_key=allow_api_key,
        )
        # dspy.LM defaults cache=True and an engine-backed LM is managed, so dspy WOULD consult a
        # cache. Off unless asked, for the reason `ClaudeAgentLM` gives: an LM handed in through
        # `configure(main_lm=...)` keeps its own flag, so the kit's default cannot reach it.
        kwargs.setdefault("cache", False)
        super().__init__(
            model=full_model,
            engine=_AgyEngine(call),
            async_engine=_AsyncAgyEngine(call),
            **kwargs,
        )
        self._call = call
