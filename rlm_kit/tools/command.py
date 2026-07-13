"""Reusable ``run_command`` tooling — execute a local command through a
consumer-supplied, ISOLATED runner (mirrors ``fetch.py`` / ``search.py``).

An agent built on the RLM often needs to run a local command (a build, a test, a
git op) the way a coding agent does. The reusable half is the same as every other
tool here: enforce the sync contract, turn a failure into text the RLM can react to,
and record ONE ``tool_call`` in the canonical shape. rlm-kit owns only that half — it
ships NO executor and picks NO isolation mechanism.

SECURITY — the runner's isolation IS the boundary. A ``run_command`` tool executes
model-CHOSEN commands. Like the ``fetch`` / ``web_search`` providers and MCP servers,
it runs HOST-SIDE — *outside* the pyodide/deno sandbox that isolates the RLM's own
REPL code. A naive ``subprocess.run`` runner is therefore arbitrary code execution
steered by the model (and by any untrusted content the model has read) — the SAME
class of danger as the refused ``local`` interpreter (see ``sandbox.py``). So the
``runner`` is a REQUIRED injection and the kit never ships one: for anything
processing untrusted input it MUST execute inside a disposable, network-restricted
container / VM / OS-sandbox (``examples/command_runner.py`` shows one). A command
allowlist is NOT a substitute — a shell allowlist is routinely bypassed
(``make`` / ``npm run`` script edits, ``find -exec``, ``git -c``, ``$(...)``
substitution, env-var injection), which is why this module ships no allowlist
primitive: the ``guard`` hook is a SHAPE-only pre-flight, never a security claim.

STATE — one-shot by default; sessions live in the runner. ``run_command`` returns ONE
command's result and holds no shell state of its own. Whether cwd / env / filesystem
writes / background processes PERSIST across calls is the RUNNER's contract, not this
wrapper's: a fresh-container-per-call runner (``examples/command_runner.py``) is a
stateless INSPECT surface; an edit-build-test loop needs a STATEFUL runner — a closure
over a long-lived sandbox (``docker create`` + ``docker exec``, an E2B / Modal / Daytona
handle, or a SWE-ReX ``BashSession``) — which fits THIS SAME seam with no API change. The
RLM's REPL is itself persistent (the model can hold outputs in variables across turns),
so a stateless runner goes further here than in a bare shell agent. Interactive tools and
tmux-style sessions are out of scope for a one-shot result; wrap a session backend in the
runner if you need them (and add an additive ``session_id`` to the payload at that point).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional, Union

from ..trace import record_tool_call

# How much stderr to keep in the trace. The full streams ride back to the model in the
# returned dict (a REPL value it reads); the JSONL keeps only a preview + lengths,
# mirroring how ``fetch_url`` records size not body.
_STDERR_PREVIEW = 500

# A command is either an argv list (preferred — no shell parsing) or a shell string.
Command = Union[list, str]


@dataclass
class CommandResult:
    """Structured outcome of one command execution — the RUNNER's return contract.

    ``make_command_tool`` converts it to a ``{"exit_code", "stdout", "stderr"}`` dict for
    the model, because dspy's interpreter JSON-bridges a ``list``/``dict`` tool result into
    a real REPL value (``run_command(...)["stdout"]``) but sends any other type through
    ``str()`` — a dataclass would reach the model only as its ``repr`` string, unsliceable.
    """

    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration_ms: Optional[float] = None  # the runner's real spawn→exit window, if it timed the call


# A runner executes a command in ISOLATION and returns its CommandResult. SYNC —
# dspy.RLM invokes tools synchronously (its sandbox bridge never awaits), so the runner
# and the tool are sync; wrap an async container/sandbox client into a sync call yourself.
# The kit ships NO runner: you MUST supply an isolated one (see the module docstring). A
# STATEFUL runner (a closure over a long-lived sandbox — docker exec / E2B / Modal /
# Daytona / SWE-ReX) fits this signature unchanged; session semantics are its concern.
Runner = Callable[[Command], "CommandResult"]

# A guard is an optional SHAPE-only pre-flight: return None to allow, or a short reason
# string to refuse. NOT a security boundary (see the module docstring) — use it for argv
# normalisation / size caps, never as an allowlist you trust.
Guard = Callable[[Command], Optional[str]]


def make_command_tool(
    runner: Runner, *, guard: Optional[Guard] = None
) -> Callable[[Command], Union[dict, str]]:
    """Wrap an ISOLATED, caller-supplied (SYNC) ``runner`` into a sync ``run_command``
    tool for ``RLMTask(tools=…)``.

    SYNC because dspy.RLM's interpreter invokes tools synchronously (no await); an
    ``async def`` tool there returns an un-awaited coroutine the model never sees the
    result of, so ``runner`` must be sync too.

    The wrapper runs the optional ``guard`` first (a refusal short-circuits BEFORE the
    runner), turns a runner exception into a short string (rather than raising) so the RLM
    reacts to it as text, and records exactly ONE ``tool_call`` per call carrying only the
    OUTCOME — ``ok`` (exit code 0), ``exit_code``, ``stdout_len``, a capped
    ``stderr_preview`` and ``duration_ms`` — NOT the full stdout. On success the model
    receives a ``{"exit_code", "stdout", "stderr"}`` dict (dspy JSON-bridges a dict into a
    real REPL value it reads/slices); the trace keeps only lengths + a preview, so the
    JSONL source-of-truth stays lean, mirroring how ``fetch_url`` records size not body.

    The runner's ISOLATION is the security boundary; this factory adds none. See the
    module docstring — never pass an un-sandboxed host executor for untrusted input.
    """

    def run_command(command: Command) -> Union[dict, str]:
        """Run a local command via an isolated runner. Returns a
        ``{"exit_code", "stdout", "stderr"}`` dict — e.g. ``run_command("ls")["stdout"]`` —
        or a short error/refusal string. Execution isolation is the caller's runner; this
        tool does not sandbox."""
        if guard is not None:
            reason = guard(command)
            if reason is not None:                     # any string (even "") refuses; None allows
                record_tool_call(
                    "run_command", args={"command": command}, ok=False,
                    note=f"refused: {reason}",
                )
                return f"Refused: {reason}"
        started = time.monotonic()
        try:
            result = runner(command)
        except Exception as exc:  # noqa: BLE001 — surface as text so the RLM can react
            record_tool_call(
                "run_command", args={"command": command}, ok=False,
                note=f"error: {type(exc).__name__}",
            )
            return f"Command error for {command!r}: {type(exc).__name__}: {str(exc)[:160]}"
        # The runner owns duration (it alone sees the real spawn→exit window, container
        # startup included); fall back to the wrapper's wall-clock only when it left it None.
        elapsed_ms = (time.monotonic() - started) * 1000.0
        duration_ms = result.duration_ms if result.duration_ms is not None else elapsed_ms
        record_tool_call(
            "run_command", args={"command": command},
            ok=(result.exit_code == 0), exit_code=result.exit_code,
            stdout_len=len(result.stdout), stderr_preview=result.stderr[:_STDERR_PREVIEW],
            duration_ms=duration_ms,
        )
        # Return a dict (not the dataclass): dspy JSON-bridges list/dict into a real REPL
        # value; any other type reaches the model only as str(repr), unsliceable.
        return {"exit_code": result.exit_code, "stdout": result.stdout, "stderr": result.stderr}

    return run_command
