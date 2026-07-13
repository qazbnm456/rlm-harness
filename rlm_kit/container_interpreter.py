"""``ContainerInterpreter`` — the environment interpreter (``interpreter="container"``).

Runs the RLM's REPL INSIDE a real isolated container instead of the default Deno/Pyodide
(WASM) sandbox, so the model's own Python can ``subprocess.run(...)`` natively and hold
persistent filesystem/process state. It is the container analog of dspy's
``PythonInterpreter``: same ``CodeInterpreter`` protocol, same JSON-RPC message shapes, same
``FinalOutput`` encoding for ``SUBMIT`` — but a host↔container broker (over the ``docker run
-i`` stdio pipe) replaces the Deno bridge. See ``_sandbox_agent.py`` for the in-container half.

This module is dspy-bearing (it needs ``FinalOutput``); ``sandbox.build_interpreter`` imports it
LAZILY in the ``"container"`` branch, so ``import rlm_kit`` stays dspy-free (and docker-free — the
``docker`` CLI is an external binary checked at ``start()``, not a Python dependency).

Security: the runner's isolation IS the boundary, and it is a STRONGER one than Deno for the
subprocess case — ``--network=none`` makes the stdio broker the ONLY channel in/out, and the LM
credentials never enter the container (tool/``llm_query`` callbacks run HOST-side; only results
cross the pipe). This is the OPPOSITE of the refused ``local`` interpreter, not a relaxation of it.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

from .config import ContainerConfig

# The in-container agent source, delivered via ``python -u -c`` (no bind mount, no host dir
# exposed). ~7 KB, well under argv limits. Read once at import.
_AGENT_SRC = Path(__file__).with_name("_sandbox_agent.py").read_text(encoding="utf-8")

_MAX_SKIP_LINES = 100
_STARTUP_MIN_BUDGET = 180.0  # generous first-run budget so an image pull isn't clipped by the timeout


def _jsonrpc(method: str, params: Optional[dict], id: Any = None) -> str:
    msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        msg["params"] = params
    if id is not None:
        msg["id"] = id
    return json.dumps(msg)


class _Sandbox:
    """A spawned agent process + its duplex line pipe — the transport seam. Docker is the
    default (``_spawn_docker``); a bare-subprocess variant (``_spawn_subprocess``, for tests) or a
    future E2B/Modal transport swaps in behind the same surface. The broker logic in
    ``ContainerInterpreter`` is transport-agnostic."""

    def __init__(self, proc: "subprocess.Popen", kill_fn: Callable[["subprocess.Popen"], None]):
        self._proc = proc
        self._kill_fn = kill_fn

    def send(self, line: str) -> None:
        self._proc.stdin.write(line + "\n")
        self._proc.stdin.flush()

    def recv(self) -> str:
        return self._proc.stdout.readline()

    def poll(self) -> Optional[int]:
        return self._proc.poll()

    def stderr_tail(self, n: int = 2000) -> str:
        # Safe only once the process has exited (read() blocks until EOF).
        try:
            return ((self._proc.stderr.read() if self._proc.stderr else "") or "")[-n:]
        except Exception:
            return ""

    def wait(self, timeout: float) -> None:
        self._proc.wait(timeout=timeout)

    def kill(self) -> None:
        try:
            self._kill_fn(self._proc)
        finally:
            for stream in (self._proc.stdin, self._proc.stdout, self._proc.stderr):
                try:
                    if stream is not None:
                        stream.close()
                except Exception:
                    pass


def _docker_argv(agent_src: str, config: ContainerConfig, name: str) -> list:
    """Build the ``docker run`` argv (a pure function, unit-tested without a daemon).

    Safety caps ride unconditionally (``--network=none`` + ``--memory`` + ``--pids-limit`` +
    ``--cap-drop=ALL``). ``--cpus`` is emitted only when set (uncapped by default). ``--read-only``
    (opt-in) requires a writable tmpfs ``/tmp`` — the agent's ``tempfile`` capture needs it — and
    pins ``TMPDIR=/tmp`` so a custom image's ``TMPDIR`` can't defeat it. ``workdir`` is mounted
    READ-ONLY so model code can inspect it but never mutate host files."""
    argv = [
        "docker", "run", "-i", "--rm", "--name", name,
        "--network=" + config.network,
        "--memory=" + config.memory,
        "--pids-limit=" + str(config.pids_limit),
    ]
    if config.cpus:
        argv.append("--cpus=" + str(config.cpus))
    if config.cap_drop:
        argv.append("--cap-drop=ALL")
    if config.read_only:
        argv += ["--read-only", "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m", "-e", "TMPDIR=/tmp"]
    if config.workdir:
        argv += ["-v", f"{config.workdir}:/workspace:ro", "-w", "/workspace"]
    argv += [config.image, "python", "-u", "-c", agent_src]
    return argv


def _spawn_docker(agent_src: str, config: ContainerConfig) -> _Sandbox:
    """Spawn the agent in a throwaway, isolated container (safety caps built into the argv)."""
    from dspy.primitives.code_interpreter import CodeInterpreterError

    if shutil.which("docker") is None:
        raise CodeInterpreterError(
            "docker binary not found; interpreter='container' requires Docker "
            "(install Docker, or use the default 'pyodide' interpreter)"
        )
    if config.workdir:
        # Reject a non-absolute workdir: docker reads a bare relative name as an (empty) NAMED
        # VOLUME, not a bind mount — a silent-wrong footgun. `from_env` normalizes to absolute, but
        # a programmatic `ContainerConfig(workdir="reldir")` reaches here unnormalized.
        if not os.path.isabs(config.workdir):
            raise CodeInterpreterError(
                f"container workdir must be an absolute path (got {config.workdir!r}); a relative "
                "name would be mounted by docker as an empty named volume"
            )
        if not os.path.isdir(config.workdir):
            raise CodeInterpreterError(
                f"container workdir is not an existing directory: {config.workdir!r}"
            )
    name = "rlm-kit-env-" + uuid.uuid4().hex[:10]
    argv = _docker_argv(agent_src, config, name)
    proc = subprocess.Popen(
        argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8",
    )

    def _kill(p: "subprocess.Popen") -> None:
        # --rm reaps the container when the client exits, but force-remove by name in case the
        # client was killed before the container stopped (e.g. a watchdog timeout).
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        try:
            p.kill()
        except Exception:
            pass

    return _Sandbox(proc, _kill)


def _spawn_subprocess(agent_src: str, config: Optional[ContainerConfig] = None) -> _Sandbox:
    """Test/CI transport: run the (stdlib-only) agent as a bare child process — NO isolation.
    Exercises the full broker without Docker so CI stays green. Never a production runner."""
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", agent_src],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8",
    )

    def _kill(p: "subprocess.Popen") -> None:
        try:
            p.kill()
        except Exception:
            pass

    return _Sandbox(proc, _kill)


class ContainerInterpreter:
    """dspy ``CodeInterpreter`` backed by a persistent isolated container."""

    def __init__(
        self,
        config: Optional[ContainerConfig] = None,
        *,
        tools: Optional[dict[str, Callable[..., Any]]] = None,
        spawn: Callable[[str, ContainerConfig], _Sandbox] = _spawn_docker,
    ):
        self._config = config or ContainerConfig()
        self.tools: dict[str, Callable[..., Any]] = dict(tools) if tools else {}  # RLM mutates in place
        self.output_fields: Optional[list[dict]] = None                            # RLM sets per forward()
        self._tools_registered = False                                             # RLM resets per forward()
        self._spawn = spawn
        self._sandbox: Optional[_Sandbox] = None
        self._request_id = 0

    # ---- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        if self._sandbox is not None and self._sandbox.poll() is None:
            return
        self._tools_registered = False
        self._sandbox = self._spawn(_AGENT_SRC, self._config)
        self._health_check()

    def shutdown(self) -> None:
        sb = self._sandbox
        self._sandbox = None
        if sb is None:
            return
        try:
            sb.send(_jsonrpc("shutdown", None))
            sb.wait(timeout=10)
        except Exception:
            pass
        sb.kill()

    def __enter__(self) -> "ContainerInterpreter":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.shutdown()

    def _teardown_dead(self) -> None:
        """Drop a killed/dead sandbox so the next execute() respawns FRESH (state is lost)."""
        sb = self._sandbox
        self._sandbox = None
        self._tools_registered = False
        if sb is not None:
            try:
                sb.kill()
            except Exception:
                pass

    # ---- watchdog-guarded receive --------------------------------------------

    def _recv_guarded(self, remaining: float, context: str):
        """Read one frame, killing the sandbox if it stays blocked longer than ``remaining``
        seconds. Returns ``(msg | None, elapsed, timed_out)``. ``remaining`` is the caller's
        REMAINING sandbox-compute budget — the watchdog clocks ONLY time blocked here, never the
        host tool dispatch between frames, so a slow ``llm_query`` or long build never trips it."""
        from dspy.primitives.code_interpreter import CodeInterpreterError

        if remaining <= 0:
            self._sandbox.kill()
            return None, 0.0, True
        fired = {"v": False}

        def _fire() -> None:
            fired["v"] = True
            try:
                self._sandbox.kill()  # unblocks the readline below with EOF
            except Exception:
                pass

        timer = threading.Timer(remaining, _fire)
        timer.start()
        t0 = time.monotonic()
        try:
            line = self._sandbox.recv()
        except (ValueError, OSError):
            # the watchdog closed the pipe under a blocked readline — treat as no data
            line = ""
        finally:
            timer.cancel()
        elapsed = time.monotonic() - t0
        # A real timeout always leaves ``line`` empty (the kill EOFs the readline). If the timer
        # RACED a valid frame — fired after recv() returned data but before cancel() — honour the
        # frame rather than discarding a completed result (possibly a final SUBMIT); _fire already
        # killed the sandbox, so the next execute() respawns fresh.
        if fired["v"] and not line:
            return None, elapsed, True
        if not line:
            code = self._sandbox.poll()
            if code is None:
                # stdout hit EOF while the process is still ALIVE — untrusted model code can close
                # the RPC fd and keep running. Kill it so the stderr drain below cannot block
                # forever on a live process (a model-triggerable host hang that would otherwise
                # bypass the watchdog and teardown entirely).
                self._sandbox.kill()
                code = self._sandbox.poll()
            stderr = self._sandbox.stderr_tail()
            raise CodeInterpreterError(f"container exited (code {code}) {context}: {stderr[-2000:]}")
        line = line.strip()
        if not line.startswith("{"):
            return None, elapsed, False
        try:
            return json.loads(line), elapsed, False
        except json.JSONDecodeError:
            return None, elapsed, False

    def _health_check(self) -> None:
        from dspy.primitives.code_interpreter import CodeInterpreterError

        try:
            self._request_id += 1
            hid = self._request_id
            self._sandbox.send(_jsonrpc("execute", {"code": "print(1+1)"}, hid))
            budget = max(self._config.timeout_s, _STARTUP_MIN_BUDGET)  # first run may pull the image
            used = 0.0
            while used <= budget:
                msg, elapsed, timed_out = self._recv_guarded(budget - used, "during health check")
                used += elapsed
                if timed_out:
                    raise CodeInterpreterError("container failed to start within the startup budget")
                if msg is None:
                    continue
                if "result" in msg and msg.get("id") == hid:
                    out = (msg["result"].get("output") or "").strip()
                    if out != "2":
                        raise CodeInterpreterError(f"unexpected health-check response: {msg}")
                    return
                if "error" in msg:
                    raise CodeInterpreterError(f"health check error: {msg['error'].get('message')}")
            raise CodeInterpreterError("health check produced no result within the startup budget")
        except BaseException:
            # ANY startup failure (timeout, bad response, dead container) must tear the sandbox
            # down, else a later execute()->start() early-returns on the still-live-but-broken
            # container without re-checking and sends it real work.
            self._teardown_dead()
            raise

    # ---- tool registration & brokering ---------------------------------------

    def _extract_parameters(self, fn: Callable) -> list[dict]:
        import inspect

        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            return []
        params = []
        for name, param in sig.parameters.items():
            p = {"name": name}
            ann = param.annotation
            if ann is not inspect.Parameter.empty and ann in (str, int, float, bool, list, dict):
                p["type"] = ann.__name__
            params.append(p)
        return params

    def _register_tools(self) -> None:
        if self._tools_registered:
            return
        params: dict = {}
        if self.tools:
            params["tools"] = [
                {"name": name, "parameters": self._extract_parameters(fn)}
                for name, fn in self.tools.items()
            ]
        if self.output_fields:
            params["outputs"] = self.output_fields
        if params:
            self._send_request("register", params, "registering tools/outputs")
        self._tools_registered = True

    def _send_request(self, method: str, params: dict, context: str) -> dict:
        """A request with no in-flight tool callbacks (register) — bounded by the budget. Any
        failure tears the sandbox down so a later start() respawns rather than reusing a broken one."""
        from dspy.primitives.code_interpreter import CodeInterpreterError

        try:
            self._request_id += 1
            rid = self._request_id
            self._sandbox.send(_jsonrpc(method, params, rid))
            budget = max(self._config.timeout_s, _STARTUP_MIN_BUDGET)
            used = 0.0
            while used <= budget:
                msg, elapsed, timed_out = self._recv_guarded(budget - used, context)
                used += elapsed
                if timed_out:
                    raise CodeInterpreterError(f"timed out {context}")
                if msg is None:
                    continue
                if msg.get("id") != rid:
                    raise CodeInterpreterError(f"response id mismatch {context}")
                if "error" in msg:
                    raise CodeInterpreterError(f"error {context}: {msg['error'].get('message', 'unknown')}")
                return msg
            raise CodeInterpreterError(f"too many non-JSON lines {context}")
        except BaseException:
            self._teardown_dead()
            raise

    def _handle_tool_call(self, request: dict) -> None:
        """Broker: run the sandbox-requested tool HERE on the host, reply over the pipe.

        Container analog of dspy's ``PythonInterpreter._handle_tool_call``. Credentials the tool
        needs stay in THIS process; only the result crosses the pipe. Not counted against the
        execution timeout (the watchdog is disarmed while we are here)."""
        from dspy.primitives.code_interpreter import CodeInterpreterError

        rid = request["id"]
        params = request.get("params", {})
        name = params.get("name")
        kwargs = params.get("kwargs", {})
        try:
            if name not in self.tools:
                raise KeyError(f"Unknown tool: {name}")
            result = self.tools[name](**kwargs)
            is_json = isinstance(result, (list, dict))
            reply = {"jsonrpc": "2.0", "id": rid, "result": {
                "value": json.dumps(result) if is_json else (str(result) if result is not None else ""),
                "type": "json" if is_json else "string",
            }}
        except Exception as exc:  # noqa: BLE001 — the TOOL failed; report it back to the sandbox
            reply = {"jsonrpc": "2.0", "id": rid,
                     "error": {"code": -32007, "message": str(exc), "data": {"type": type(exc).__name__}}}
        # Sending the reply is separate: a send failure means the sandbox died (e.g. a watchdog kill
        # raced this callback), which must surface as a CodeInterpreterError dspy catches — not a
        # bare BrokenPipeError it does not.
        try:
            self._sandbox.send(json.dumps(reply))
        except (OSError, ValueError) as exc:
            raise CodeInterpreterError(f"sandbox died before a tool reply could be delivered: {exc}")

    # ---- variable injection --------------------------------------------------

    def _serialize_value(self, value: Any) -> str:
        from dspy.primitives.code_interpreter import CodeInterpreterError

        if value is None or isinstance(value, (str, bool, int, float)):
            return repr(value)
        if isinstance(value, (list, tuple)):
            return "[" + ", ".join(self._serialize_value(v) for v in value) + "]"
        if isinstance(value, dict):
            return "{" + ", ".join(
                f"{self._serialize_value(k)}: {self._serialize_value(v)}" for k, v in value.items()
            ) + "}"
        if isinstance(value, set):
            return "[" + ", ".join(sorted(self._serialize_value(v) for v in value)) + "]"
        raise CodeInterpreterError(f"Unsupported variable type: {type(value).__name__}")

    def _inject_variables(self, code: str, variables: dict[str, Any]) -> str:
        import keyword

        from dspy.primitives.code_interpreter import CodeInterpreterError

        assignments = []
        for k, v in variables.items():
            if not k.isidentifier() or keyword.iskeyword(k):
                raise CodeInterpreterError(f"Invalid variable name: {k!r}")
            assignments.append(f"{k} = {self._serialize_value(v)}")
        return ("\n".join(assignments) + "\n" + code) if assignments else code

    # ---- the CodeInterpreter entry point -------------------------------------

    def execute(self, code: str, variables: Optional[dict[str, Any]] = None) -> Any:
        from dspy.primitives.code_interpreter import CodeInterpreterError, FinalOutput

        code = self._inject_variables(code, variables or {})
        self.start()
        self._register_tools()

        self._request_id += 1
        exec_id = self._request_id
        self._sandbox.send(_jsonrpc("execute", {"code": code}, exec_id))

        armed_total = 0.0
        skipped = 0
        while skipped <= _MAX_SKIP_LINES:
            msg, elapsed, timed_out = self._recv_guarded(
                self._config.timeout_s - armed_total, "during execution"
            )
            armed_total += elapsed
            if timed_out:
                self._teardown_dead()
                raise CodeInterpreterError(
                    f"execution timed out after {self._config.timeout_s:g}s of sandbox compute; "
                    "the container was killed and will restart with FRESH state on the next call"
                )
            if msg is None:
                skipped += 1
                continue
            if msg.get("method") == "tool_call":  # sandbox -> host callback (host time, uncounted)
                self._handle_tool_call(msg)
                continue
            if "result" in msg:
                if msg.get("id") != exec_id:
                    raise CodeInterpreterError(f"response id mismatch: expected {exec_id}")
                result = msg["result"]
                if "final" in result:
                    return FinalOutput(result["final"])
                return result.get("output", None)
            if "error" in msg:
                err = msg["error"]
                if err.get("code") == -32000:
                    raise SyntaxError(f"Invalid Python syntax: {err.get('message')}")
                etype = err.get("data", {}).get("type", "Error")
                raise CodeInterpreterError(f"{etype}: {err.get('message')}")
            raise CodeInterpreterError(f"unexpected frame from sandbox: {msg}")

        raise CodeInterpreterError("too many non-JSON lines during execution")

    def __call__(self, code: str, variables: Optional[dict[str, Any]] = None) -> Any:
        return self.execute(code, variables)
