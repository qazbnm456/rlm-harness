"""Reusable ``run_command`` tooling — execute a local command through a
consumer-supplied, ISOLATED runner (mirrors ``fetch.py`` / ``search.py``).

An agent built on the RLM often needs to run a local command (a build, a test, a
git op) the way a coding agent does. The reusable half is the same as every other
tool here: enforce the sync contract, turn a failure into text the RLM can react to,
and record ONE ``tool_call`` in the canonical shape. rlm-harness owns only that half — it
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

import shlex
import time
from collections.abc import Callable
from dataclasses import dataclass

from ..trace import record_tool_call

# How much stderr to keep in the trace. The full streams ride back to the model in the
# returned dict (a REPL value it reads); the JSONL keeps only a preview + lengths,
# mirroring how ``fetch_url`` records size not body.
_STDERR_PREVIEW = 500

# A command is either an argv list (preferred — no shell parsing) or a shell string.
Command = list | str


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
    duration_ms: float | None = None  # the runner's real spawn→exit window, if it timed the call


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
Guard = Callable[[Command], str | None]


def make_command_tool(
    runner: Runner, *, guard: Guard | None = None
) -> Callable[[Command], dict | str]:
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

    def run_command(command: Command) -> dict | str:
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
        except Exception as exc:
            record_tool_call(
                "run_command", args={"command": command}, ok=False,
                duration_s=time.monotonic() - started,
                note=f"error: {type(exc).__name__}",
            )
            return f"Command error for {command!r}: {type(exc).__name__}: {str(exc)[:160]}"
        # The runner owns duration (it alone sees the real spawn→exit window, container
        # startup included); fall back to the wrapper's wall-clock only when it left it None.
        elapsed_ms = (time.monotonic() - started) * 1000.0
        duration_ms = result.duration_ms if result.duration_ms is not None else elapsed_ms
        # TWO durations, deliberately, and they are not the same quantity. `duration_ms` is the
        # RUNNER's own figure — it alone sees the real spawn→exit window — and stays the precise
        # answer for this tool. `duration_s` is the wrapper's wall-clock, which is what
        # `metrics.compute_tool_waste` compares ACROSS tools; a metric mixing a runner-reported
        # spawn window with other tools' wall-clock would be comparing two different things.
        record_tool_call(
            "run_command", args={"command": command},
            ok=(result.exit_code == 0), exit_code=result.exit_code,
            stdout_len=len(result.stdout), stderr_preview=result.stderr[:_STDERR_PREVIEW],
            duration_ms=duration_ms, duration_s=elapsed_ms / 1000.0,
        )
        # Return a dict (not the dataclass): dspy JSON-bridges list/dict into a real REPL
        # value; any other type reaches the model only as str(repr), unsliceable.
        return {"exit_code": result.exit_code, "stdout": result.stdout, "stderr": result.stderr}

    return run_command


# `git log`'s own broad-history options this guard refuses. Two shapes: an EXACT boolean flag, and
# a PREFIXABLE one that optionally takes `=<pattern>` (matched on the token up to its first `=`).
# Scoped to `git log` only (not a general git lockdown) — mirrors an eval/training-run convention
# for refusing a model access to other branches/tags/reflogs it should not be looking at, not a
# general git restriction (e.g. `git branch -a` is untouched).
_GIT_LOG_DENYLIST_EXACT = frozenset(
    {"--all", "--reflog", "--walk-reflogs", "-g", "--alternate-refs"}
)
_GIT_LOG_DENYLIST_PREFIXABLE = frozenset({"--branches", "--remotes", "--tags", "--glob"})

# Global git options that legally precede the subcommand and take a following value token
# (`git -C <path> log ...`, `git -c <k=v> log ...`) — consumed as a pair, never mistaken for the
# subcommand itself.
_GIT_GLOBAL_OPTS_WITH_VALUE = frozenset({"-C", "-c"})


def refuse_broad_git_history(command: Command) -> str | None:
    """An OPTIONAL ``guard`` for :func:`make_command_tool` that refuses a ``git log`` invocation
    carrying a broad-history option (``--all``, ``--branches``, ``--remotes``, ``--tags``,
    ``--glob``, ``--reflog``, ``--walk-reflogs``, ``-g``, ``--alternate-refs``).

    An eval/training-run convention: stop a model from reading other branches/tags/reflogs it
    should not have access to (task-specific hints, other agents' solutions in a shared repo).
    Like every ``guard`` (see the module docstring), this is SHAPE-only — a pattern match on
    tokens, not a security boundary. It refuses only ``git log`` itself; it does not restrict any
    other git subcommand (``git branch -a`` is untouched — a deliberate, narrow scope, not a
    general git lockdown).

    **Detection**: an argv list is used as-is; a shell string is tokenized with ``shlex.split``.
    ``tokens[0]`` must be ``git`` (or end with ``/git``); tokens after it are walked, skipping
    recognized GLOBAL git options that legally precede the subcommand (``-C <path>`` / ``-c <k=v>``
    consume their value token; ``--git-dir=…`` / ``--work-tree=…`` / ``--namespace=…`` and any
    other bare ``-``-prefixed token, e.g. ``--no-pager``, are skipped on their own) until it hits
    either the literal token ``"log"`` (the rest are ``log``'s own arguments, scanned against the
    denylist) or a non-flag, non-``log`` token (some OTHER git subcommand — not our concern,
    returns ``None``). Matching is token-based throughout, never a substring/``in`` search over
    the raw text — so ``git commit -m "please git log --all this"`` is correctly left alone (its
    second token is ``commit``, not ``log``).

    **Known, deliberate non-goal — this is not a shell parser.** A compound shell string chaining
    multiple commands (``"echo hi && git log --all"``, ``"a; git log --all"``, ``"a | git log
    --all"``) is NOT decomposed into sub-commands: ``shlex.split`` has no concept of ``&&``/``;``/
    ``|`` as separators, so such a string's ``tokens[0]`` is whatever precedes the chain and the
    whole thing passes through unrefused. This is exactly why a ``guard`` is documented as
    shape-only, never a security boundary — prefer argv-list commands (this module's own
    preference) where there is no shell-parsing ambiguity at all.
    """
    tokens = command if isinstance(command, list) else shlex.split(command)
    if not tokens:
        return None
    prog = tokens[0]
    if prog != "git" and not prog.endswith("/git"):
        return None

    i = 1
    while i < len(tokens):
        token = tokens[i]
        if token == "log":
            i += 1
            break
        if token in _GIT_GLOBAL_OPTS_WITH_VALUE:
            i += 2  # consume the option AND its value token
            continue
        if token.startswith("-"):
            i += 1  # a bare global flag (--no-pager, --git-dir=…, …) — skip it alone
            continue
        return None  # a non-flag, non-"log" token: some other git subcommand
    else:
        return None  # walked off the end without ever finding "log"

    for arg in tokens[i:]:
        base = arg.split("=", 1)[0]
        if base in _GIT_LOG_DENYLIST_EXACT or base in _GIT_LOG_DENYLIST_PREFIXABLE:
            return f"broad git-history option {arg!r} on `git log`"
    return None
