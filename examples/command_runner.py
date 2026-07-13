"""Example: a container-backed ISOLATED runner for ``make_command_tool``.

rlm-kit ships NO command executor on purpose — the runner's isolation IS the
security boundary (see ``rlm_kit/tools/command.py``). This example shows the
reference pattern: run each model-chosen command inside a disposable, network-off
Docker container with the workspace mounted READ-ONLY, so a command the model
chooses (or is steered into by untrusted content it read) cannot touch the host,
reach the network, or persist.

This is a STATELESS **inspect** runner — a fresh container per call with a read-only
mount, so nothing persists between commands. Good for read-only investigation (grep,
find, cat, read-only `git`). An edit-build-test loop needs a STATEFUL runner instead:
keep one sandbox alive and `docker exec` into it per call (or hold an E2B / Modal /
Daytona sandbox handle, or a SWE-ReX ``BashSession``). That fits the SAME
``make_command_tool(runner)`` seam with no API change — only the runner's lifecycle
differs. Isolation stays the runner's job either way.

Illustrative — needs Docker plus real model creds and a sandbox, so it is NOT
imported by the test suite. Run:

    docker pull python:3.11-slim
    export RLM_MAIN_MODEL=...  RLM_API_KEY=...
    python examples/command_runner.py

Harden further for genuinely untrusted code: a VM / microVM (Firecracker, or a managed
E2B/Modal sandbox) instead of a container, for kernel-level isolation. A command
allowlist is NOT a substitute for this isolation.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import time
from typing import Union

from rlm_kit import RLMConfig, RLMTask, TraceRecorder, configure
from rlm_kit.tools import CommandResult, make_command_tool


def make_docker_runner(
    *, image: str = "python:3.11-slim", workdir: str, timeout: float = 30.0
):
    """Build a SYNC runner that executes a command in a throwaway, network-off
    container with ``workdir`` mounted read-only at ``/work``. Returns a
    ``CommandResult`` — matching ``make_command_tool``'s ``runner`` contract."""

    def run(command: Union[list, str]) -> CommandResult:
        # A shell string runs via `sh -c`; an argv list is passed through literally. `timeout`
        # runs INSIDE the container so a runaway command is killed daemon-side — a subprocess
        # timeout would only SIGKILL the `docker run` client and leave the container running.
        payload = ["sh", "-c", command] if isinstance(command, str) else list(command)
        inner = ["timeout", "--signal=KILL", str(timeout), *payload]
        argv = [
            "docker", "run", "--rm",
            "--network=none",             # no egress — cannot exfiltrate or phone home
            "--user", "65534:65534",      # nobody — no root inside the container
            "--read-only",                # immutable rootfs
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",  # scratch that never touches the host
            "--memory=512m", "--cpus=1", "--pids-limit=256",  # cap the blast radius of a fork/OOM bomb
            "-v", f"{workdir}:/work:ro",  # workspace visible but NOT writable
            "-w", "/work",
            image, *inner,
        ]
        started = time.monotonic()
        # Outer timeout is a backstop above the in-container one (container startup + kill grace).
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout + 15)
        return CommandResult(
            exit_code=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            duration_ms=(time.monotonic() - started) * 1000.0,
        )

    return run


class Inspect(RLMTask):
    signature = "request: str -> answer: str"
    output_field = "answer"
    instructions = (
        "You can run shell commands with run_command(cmd) — it executes inside an "
        "isolated, network-off container with the workspace mounted read-only. "
        "run_command returns a dict: out = run_command('ls'); read out['stdout'], "
        "out['stderr'], out['exit_code']. Investigate what you need, then answer."
    )

    def __init__(self, workdir: str, **kw):
        # The runner is the security boundary; make_command_tool only wraps it (sync
        # contract + tracing). The kit ships no runner — we supply the isolated one.
        self.tools = [make_command_tool(make_docker_runner(workdir=workdir))]
        super().__init__(**kw)


async def main() -> None:
    configure(RLMConfig.from_env())
    task = Inspect(workdir=os.getcwd())
    with TraceRecorder("./traces/command-run.jsonl", run_id="command-001"):
        answer = await task.arun(request="How many Python files are in this project?")
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
