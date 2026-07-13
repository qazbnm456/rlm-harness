"""Tests for the container (environment) interpreter.

The broker is exercised in CI WITHOUT Docker by running the stdlib-only in-container agent
(`_sandbox_agent.py`) as a bare subprocess — the real bidirectional protocol, just no
isolation. A separate Docker-gated test proves real container isolation. The whole file is
skipped if dspy is absent (the interpreter returns dspy's `FinalOutput` / raises
`CodeInterpreterError`).
"""

import shutil

import pytest

pytest.importorskip("dspy.primitives.code_interpreter")

from dspy.primitives.code_interpreter import CodeInterpreterError, FinalOutput  # noqa: E402

from rlm_kit.config import ContainerConfig  # noqa: E402
from rlm_kit.container_interpreter import ContainerInterpreter, _spawn_subprocess  # noqa: E402


def _interp(**cfg_kw) -> ContainerInterpreter:
    """A ContainerInterpreter driven by the bare-subprocess transport (no Docker, no isolation)."""
    return ContainerInterpreter(ContainerConfig(**cfg_kw), spawn=_spawn_subprocess)


# ---- broker end-to-end via the stdlib-only agent as a bare subprocess (CI-safe) ----

def test_execute_captures_stdout_and_persists_state():
    interp = _interp()
    try:
        assert "hi" in interp.execute("print('hi')")
        interp.execute("x = 41")
        assert "42" in interp.execute("x += 1\nprint(x)")     # state persists across execute()
    finally:
        interp.shutdown()


def test_native_subprocess_runs_in_the_repl():
    # The capability WASM forbids: the REPL can spawn a real child process.
    interp = _interp()
    try:
        out = interp.execute(
            "import subprocess\n"
            "print(subprocess.run(['echo', 'SUBPROC_OK'], capture_output=True, text=True).stdout)"
        )
        assert "SUBPROC_OK" in out
    finally:
        interp.shutdown()


def test_tool_callback_brokers_to_the_host():
    calls = []

    def add(a: int, b: int):
        calls.append((a, b))
        return a + b

    interp = _interp()
    interp.tools["add"] = add                                 # RLM mutates .tools in place; we do too
    try:
        out = interp.execute("print(add(2, 3))")
        assert "5" in out and calls == [(2, 3)]               # ran on the HOST, result brokered back
    finally:
        interp.shutdown()


def test_submit_returns_final_output():
    interp = _interp()
    interp.output_fields = [{"name": "answer", "type": "str"}]
    try:
        r = interp.execute("SUBMIT(answer='done')")
        assert isinstance(r, FinalOutput) and r.output == {"answer": "done"}
    finally:
        interp.shutdown()


def test_submit_survives_user_except_exception():
    # The BaseException fix: a model's `except Exception` around SUBMIT must NOT swallow the signal.
    interp = _interp()
    interp.output_fields = [{"name": "answer", "type": "str"}]
    try:
        r = interp.execute(
            "try:\n"
            "    SUBMIT(answer='ok')\n"
            "except Exception:\n"
            "    print('SWALLOWED')\n"
        )
        assert isinstance(r, FinalOutput) and r.output == {"answer": "ok"}
    finally:
        interp.shutdown()


def test_syntax_error_maps_to_SyntaxError():
    interp = _interp()
    try:
        with pytest.raises(SyntaxError):
            interp.execute("def (:")
    finally:
        interp.shutdown()


def test_runtime_error_maps_to_CodeInterpreterError():
    interp = _interp()
    try:
        with pytest.raises(CodeInterpreterError):
            interp.execute("1 / 0")
    finally:
        interp.shutdown()


def test_timeout_kills_then_respawns_fresh():
    interp = _interp(timeout_s=0.5)
    try:
        with pytest.raises(CodeInterpreterError) as ei:
            interp.execute("while True:\n    pass")
        assert "timed out" in str(ei.value)
        # after a timeout kill the next call respawns with FRESH state and works again
        assert "alive" in interp.execute("print('alive')")
    finally:
        interp.shutdown()


def test_shutdown_is_idempotent():
    interp = _interp()
    interp.execute("pass")
    interp.shutdown()
    interp.shutdown()                                         # a second shutdown must not raise


def test_stdout_eof_on_a_live_process_does_not_hang():
    # Regression (HIGH): untrusted model code can close the RPC fd (stdout EOF) while staying
    # alive. The interpreter must KILL the live sandbox before draining its stderr — otherwise the
    # blocking read-to-EOF hangs the host forever, bypassing the watchdog and teardown.
    interp = _interp()

    class _FakeSandbox:
        def __init__(self):
            self.alive = True
            self.killed = False

        def recv(self):
            return ""                                         # immediate stdout EOF

        def poll(self):
            return None if self.alive else -9                 # alive until killed

        def kill(self):
            self.killed = True
            self.alive = False

        def stderr_tail(self, n=2000):
            assert not self.alive, "stderr_tail must not run on a live process (it would block)"
            return "boom"

    fake = _FakeSandbox()
    interp._sandbox = fake
    with pytest.raises(CodeInterpreterError):
        interp._recv_guarded(5.0, "test")
    assert fake.killed                                        # the live sandbox was killed before drain


# ---- config + wiring (no spawn, CI-safe) ----

def test_container_config_from_env(monkeypatch):
    monkeypatch.setenv("RLM_CONTAINER_IMAGE", "custom:tag")
    monkeypatch.setenv("RLM_CONTAINER_TIMEOUT", "45")
    monkeypatch.setenv("RLM_CONTAINER_PIDS_LIMIT", "128")
    cfg = ContainerConfig.from_env()
    assert cfg.image == "custom:tag" and cfg.timeout_s == 45.0 and cfg.pids_limit == 128

    from rlm_kit import RLMConfig

    rc = RLMConfig.from_env()
    assert rc.interpreter == "pyodide"                        # default interpreter is UNCHANGED
    assert rc.container.image == "custom:tag"                 # container options are wired in


def test_build_interpreter_selects_container_lazily():
    from rlm_kit.sandbox import build_interpreter

    interp = build_interpreter("container", container=ContainerConfig(memory="256m"))
    assert isinstance(interp, ContainerInterpreter)           # construction spawns nothing
    assert interp._config.memory == "256m"


def test_local_refusal_is_untouched():
    from rlm_kit.sandbox import SandboxSecurityError, build_interpreter

    with pytest.raises(SandboxSecurityError):                 # container must not have loosened this
        build_interpreter("local")


# ---- docker argv construction (pure function, no daemon) ----

def test_docker_argv_defaults_are_capped_but_uncpu_unmounted():
    from rlm_kit.container_interpreter import _docker_argv

    argv = _docker_argv("AGENT", ContainerConfig(), "n1")
    assert "--network=none" in argv
    assert any(a.startswith("--memory=") for a in argv)
    assert any(a.startswith("--pids-limit=") for a in argv)
    assert "--cap-drop=ALL" in argv                           # dropped by default
    assert not any(a.startswith("--cpus") for a in argv)      # uncapped by default (no build throttle)
    assert "--read-only" not in argv and "-v" not in argv     # writable rootfs, no mount by default
    assert argv[-4:] == ["python", "-u", "-c", "AGENT"]       # the agent command always comes last
    assert argv[argv.index("AGENT") - 4] == ContainerConfig().image  # image precedes python -u -c


def test_docker_argv_cpus_and_cap_drop_toggle():
    from rlm_kit.container_interpreter import _docker_argv

    argv = _docker_argv("A", ContainerConfig(cpus="2", cap_drop=False), "n")
    assert "--cpus=2" in argv and "--cap-drop=ALL" not in argv


def test_docker_argv_read_only_adds_tmpfs_and_tmpdir():
    from rlm_kit.container_interpreter import _docker_argv

    argv = _docker_argv("A", ContainerConfig(read_only=True), "n")
    assert "--read-only" in argv
    assert "--tmpfs" in argv and any(a.startswith("/tmp:") for a in argv)  # agent's tempfile needs it
    assert "TMPDIR=/tmp" in argv                              # a custom image's TMPDIR can't defeat it


def test_docker_argv_workdir_mounts_read_only():
    from rlm_kit.container_interpreter import _docker_argv

    argv = _docker_argv("A", ContainerConfig(workdir="/repo"), "n")
    assert "-v" in argv and "/repo:/workspace:ro" in argv and "-w" in argv and "/workspace" in argv


def test_container_config_phase2_env(monkeypatch):
    monkeypatch.setenv("RLM_CONTAINER_CPUS", "1.5")
    monkeypatch.setenv("RLM_CONTAINER_CAP_DROP", "false")
    monkeypatch.setenv("RLM_CONTAINER_READ_ONLY", "true")
    monkeypatch.setenv("RLM_CONTAINER_WORKDIR", "somedir")     # relative → normalised to absolute
    cfg = ContainerConfig.from_env()
    assert cfg.cpus == "1.5" and cfg.cap_drop is False and cfg.read_only is True
    import os
    assert os.path.isabs(cfg.workdir)                          # a bare name is not left for docker to
    assert cfg.workdir.endswith("somedir")                    # misread as an (empty) named volume


def test_spawn_docker_rejects_missing_workdir(monkeypatch):
    # workdir validation fires before spawning, so this needs no real daemon (docker is mocked present).
    import rlm_kit.container_interpreter as ci

    monkeypatch.setattr(ci.shutil, "which", lambda _cmd: "/usr/bin/docker")
    with pytest.raises(CodeInterpreterError):
        ci._spawn_docker("AGENT", ContainerConfig(workdir="/no/such/dir/xyz-123"))


def test_spawn_docker_rejects_relative_workdir(monkeypatch, tmp_path):
    # A relative workdir that EXISTS under cwd passes isdir, but docker would mount it as an empty
    # NAMED VOLUME (silent-wrong). The programmatic path (bypassing from_env's abspath) must reject it.
    import rlm_kit.container_interpreter as ci

    monkeypatch.setattr(ci.shutil, "which", lambda _cmd: "/usr/bin/docker")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "reldir").mkdir()
    with pytest.raises(CodeInterpreterError):
        ci._spawn_docker("AGENT", ContainerConfig(workdir="reldir"))


# ---- real Docker isolation (gated: skips where the docker CLI is absent) ----

@pytest.mark.skipif(shutil.which("docker") is None, reason="docker CLI not available")
def test_container_isolation_real_docker():
    import os

    os.environ["RLM_TEST_SECRET_XYZ"] = "leaked-if-visible"
    interp = ContainerInterpreter(ContainerConfig(timeout_s=60.0))   # real docker spawn
    try:
        assert "Linux" in interp.execute("import platform; print(platform.system())")
        # a host-only env var / credential never enters the container
        out = interp.execute("import os; print(os.environ.get('RLM_TEST_SECRET_XYZ', 'ABSENT'))")
        assert "ABSENT" in out and "leaked" not in out
    finally:
        interp.shutdown()
        os.environ.pop("RLM_TEST_SECRET_XYZ", None)
