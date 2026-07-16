"""MCP client tests — the async->sync bridge is the load-bearing mechanic, so these run a REAL
stdio MCP server (a tiny FastMCP echo) as a subprocess and drive it through ``mcp_tools``. Skips
when the optional ``mcp`` extra (or dspy) is absent, like the other dspy-bearing tests."""

import contextlib
import sys

import pytest

pytest.importorskip("mcp")
pytest.importorskip("dspy")

from rlm_kit.mcp import (  # noqa: E402
    _CLOSE_GRACE,
    McpCatalog,
    McpConnection,
    _args_from_schema,
    mcp_tools,
    result_text,
)
from rlm_kit.trace import TraceRecorder, load_events  # noqa: E402


@pytest.fixture(autouse=True)
def _force_direct_connection(monkeypatch):
    # httpx's trust_env picks up an OS/system proxy (e.g. macOS's system-config fallback) even with NO
    # proxy env vars set, which would route loopback MCP-over-HTTP through a proxy and make the tarpit /
    # connection-refused tests measure the PROXY, not httpx. Force a direct connection everywhere.
    monkeypatch.setenv("NO_PROXY", "*")
    monkeypatch.setenv("no_proxy", "*")

# A minimal stdio MCP server: one `echo` tool. Written to a temp file and spawned per test.
_ECHO_SERVER = '''
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("echo-test")


@mcp.tool()
def echo(text: str) -> str:
    """Echo the text back, prefixed."""
    return f"echo: {text}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
'''


def _server(tmp_path):
    p = tmp_path / "echo_server.py"
    p.write_text(_ECHO_SERVER)
    return {"command": sys.executable, "args": [str(p)]}


# ---- pure helpers --------------------------------------------------------

def test_args_from_schema_extracts_properties():
    assert _args_from_schema({"type": "object", "properties": {"text": {"type": "string"}}}) == {
        "text": {"type": "string"}
    }
    assert _args_from_schema(None) == {}
    assert _args_from_schema({"no": "properties"}) == {}


def test_result_text_joins_text_blocks_and_flags_errors():
    import types

    block = types.SimpleNamespace(text="hello")
    ok = types.SimpleNamespace(content=[block], structuredContent=None, isError=False)
    assert result_text(ok) == "hello"
    err = types.SimpleNamespace(content=[block], structuredContent=None, isError=True)
    assert result_text(err).startswith("[tool reported an error]")
    structured = types.SimpleNamespace(content=[], structuredContent={"n": 1}, isError=False)
    assert '"n": 1' in result_text(structured)


# ---- the bridge: real stdio server, sync call, teardown ------------------

def test_mcp_tools_discovers_and_calls_sync(tmp_path):
    with mcp_tools(_server(tmp_path), timeout=30) as tools:
        assert [t.name for t in tools] == ["echo"]
        echo = tools[0]
        assert echo.args["text"]["type"] == "string"            # inputSchema mapped for the RLM
        # THE load-bearing assertion: a SYNC call drives the async session across the bridge.
        assert echo(text="hi") == "echo: hi"
        assert echo(text="again") == "echo: again"              # session stays alive across calls


def test_mcp_tool_call_is_traced(tmp_path):
    p = tmp_path / "t.jsonl"
    with TraceRecorder(str(p), run_id="r"):
        with mcp_tools(_server(tmp_path), timeout=30) as tools:
            tools[0](text="hi")
    calls = [e for e in load_events(str(p), "r")
             if e["type"] == "tool_call" and e["payload"]["tool"] == "echo"]
    assert calls and calls[0]["payload"]["ok"] is True
    assert "echo: hi" in calls[0]["payload"]["preview"]


def test_mcp_tools_teardown_leaves_no_live_thread(tmp_path):
    import threading

    before = {t.name for t in threading.enumerate()}
    with mcp_tools(_server(tmp_path), timeout=30) as tools:
        tools[0](text="x")
    # the background MCP thread is joined on exit — no leaked "rlm-kit-mcp" thread.
    after = {t.name for t in threading.enumerate()}
    assert "rlm-kit-mcp" not in (after - before)


def test_bad_server_spec_raises_and_cleans_up():
    import threading

    before = {t.name for t in threading.enumerate()}
    with pytest.raises(ValueError):
        with mcp_tools({"nonsense": 1}, timeout=5):
            pass
    # start() failed, but mcp_tools' finally still closed the bridge — no leaked background thread.
    assert "rlm-kit-mcp" not in ({t.name for t in threading.enumerate()} - before)


# ---- HTTP transport: real streamable-HTTP server, sync call --------------

_HTTP_SERVER = '''
import sys
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("echo-http", host="127.0.0.1", port=int(sys.argv[1]))


@mcp.tool()
def echo(text: str) -> str:
    return f"echo: {text}"


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
'''


def _free_port() -> int:
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_port(port: int, timeout: float = 20.0) -> None:
    import socket
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError(f"HTTP MCP server did not start on :{port}")


@contextlib.contextmanager
def _running_http_server(tmp_path):
    """Spin up the streamable-HTTP echo server as a subprocess; yield its /mcp URL; tear it down."""
    import subprocess

    port = _free_port()
    server = tmp_path / "http_server.py"
    server.write_text(_HTTP_SERVER)
    proc = subprocess.Popen(
        [sys.executable, str(server), str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        _wait_port(port)
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            proc.kill()


def test_mcp_tools_streamable_http(tmp_path):
    import subprocess

    port = _free_port()
    server = tmp_path / "http_server.py"
    server.write_text(_HTTP_SERVER)
    proc = subprocess.Popen(
        [sys.executable, str(server), str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        _wait_port(port)
        # exercises the streamable-HTTP transport (3-tuple streams) end-to-end, not just stdio.
        with mcp_tools({"url": f"http://127.0.0.1:{port}/mcp"}, timeout=20) as tools:
            assert [t.name for t in tools] == ["echo"]
            assert tools[0](text="hi") == "echo: hi"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            proc.kill()


# ---- per-call timeout + cancel (a hung tool must not wedge the session) ---

_SLOW_SERVER = '''
import asyncio
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("slow-test")


@mcp.tool()
async def slow() -> str:
    await asyncio.sleep(30)
    return "done"


@mcp.tool()
def echo(text: str) -> str:
    return f"echo: {text}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
'''


def test_hung_tool_times_out_and_session_survives(tmp_path):
    import time

    server = tmp_path / "slow_server.py"
    server.write_text(_SLOW_SERVER)
    # a short per-call timeout: `slow` sleeps 30s, so it must trip the timeout+cancel quickly.
    with mcp_tools({"command": sys.executable, "args": [str(server)]}, timeout=5) as tools:
        by_name = {t.name: t for t in tools}
        t0 = time.monotonic()
        out = by_name["slow"]()
        assert time.monotonic() - t0 < 20              # tripped at ~5s, NOT the 30s sleep
        assert "timed out" in out.lower()              # surfaced as a reactable string
        # the cancel kept the session usable — a fast call still works afterwards.
        assert by_name["echo"](text="ok") == "echo: ok"


# ---- McpCatalog: multi-server progressive transport ----------------------

def _named(tmp_path, name):
    return {"name": name, "description": f"{name} server", **_server(tmp_path)}


def test_mcp_catalog_progressive_surface(tmp_path):
    cat = McpCatalog([_named(tmp_path, "echo")], timeout=30)
    try:
        assert cat.servers() == [("echo", "echo server")]
        assert cat.has_server("echo") and not cat.has_server("nope")
        cat.load("echo")                                  # no-op under eager
        assert cat.tool_names("echo") == ["echo"]
        tool = cat.tools("echo")[0]                       # RAW mcp Tool object, not a dspy.Tool
        assert tool.name == "echo" and hasattr(tool, "inputSchema")
        assert cat.call("echo", "echo", {"text": "hi"}) == "echo: hi"   # flattened result TEXT
    finally:
        cat.close()


def test_mcp_catalog_lazy_is_per_transport(tmp_path):
    # connect="lazy" is PER-TRANSPORT: a stdio server still connects EAGERLY in __init__ (a local
    # subprocess spawn stays pre-run), while a URL (streamable-HTTP) server DEFERS to first load().
    specs = [
        _named(tmp_path, "echo"),                                        # stdio → eager even under lazy
        {"name": "remote", "description": "d", "url": "http://127.0.0.1:1/mcp"},  # url → deferred
    ]
    cat = McpCatalog(specs, connect="lazy", timeout=5)
    try:
        assert cat.tool_names("echo") == ["echo"]   # stdio connected in __init__ despite lazy
        assert cat.tools("remote") == []            # url deferred — never connected, no attempt made
    finally:
        cat.close()


def test_mcp_catalog_lazy_http_connects_on_load(tmp_path):
    with _running_http_server(tmp_path) as url:
        cat = McpCatalog([{"name": "echo", "url": url}], connect="lazy", timeout=20)
        try:
            assert cat.tools("echo") == []                              # deferred until load()
            cat.load("echo")
            assert cat.tool_names("echo") == ["echo"]                   # connected on demand
            assert cat.call("echo", "echo", {"text": "hi"}) == "echo: hi"
        finally:
            cat.close()


def test_mcp_catalog_records_nothing(tmp_path):
    p = tmp_path / "t.jsonl"
    with TraceRecorder(str(p), run_id="r"):
        cat = McpCatalog([_named(tmp_path, "echo")], timeout=30)
        try:
            cat.call("echo", "echo", {"text": "hi"})
        finally:
            cat.close()
    # a pure transport — the CONSUMER's meta-tool owns the tool_call event, not the catalog.
    assert [e for e in load_events(str(p), "r") if e["type"] == "tool_call"] == []


def test_mcp_catalog_bad_spec_raises():
    with pytest.raises(ValueError, match="'name'"):
        McpCatalog([{"description": "no name", "url": "http://x"}])


def test_mcp_catalog_rejects_unknown_connect(tmp_path):
    with pytest.raises(ValueError, match="eager"):
        McpCatalog([_named(tmp_path, "echo")], connect="bogus")   # rejected before any connect


def test_mcp_connection_close_before_start_is_safe():
    # McpConnection is public: closing one that was never started must not raise (join would).
    McpConnection({"url": "http://127.0.0.1:1/mcp"}).close()


def test_mcp_catalog_partial_eager_failure_cleans_up(tmp_path):
    import threading

    before = {t.name for t in threading.enumerate()}
    specs = [_named(tmp_path, "echo"), {"name": "bad", "command": "/nonexistent-cmd-xyz"}]
    with pytest.raises(Exception):  # noqa: B017 — the 'bad' server fails eager connect
        McpCatalog(specs, timeout=5)
    # the good server connected first, then 'bad' failed — the partial connect was torn down, no leak.
    assert "rlm-kit-mcp" not in ({t.name for t in threading.enumerate()} - before)


# ---- lazy-connect safety: a wedged connect is BOUNDED and REAPED, not a hang+leak ----------

def test_mcp_catalog_lazy_refused_connect_raises_fast(tmp_path):
    import time

    # A closed port: a lazy load() must RAISE quickly (well under timeout), not hang. NO_PROXY=* (the
    # autouse fixture) keeps this a real connection-refused, not a proxy 502.
    cat = McpCatalog([{"name": "dead", "url": "http://127.0.0.1:1/mcp"}], connect="lazy", timeout=10)
    try:
        t0 = time.monotonic()
        with pytest.raises(Exception):  # noqa: B017 — connection refused surfaces out of load()
            cat.load("dead")
        assert time.monotonic() - t0 < 10   # refused fast — did not burn the whole timeout
    finally:
        cat.close()


def test_mcp_catalog_lazy_wedged_http_connect_is_bounded_and_reaped():
    import socket
    import threading
    import time

    # A TARPIT: accepts the TCP connection but never answers the MCP handshake. A lazy load() against
    # it must (a) stay BOUNDED (raise, not wedge the caller forever) and (b) leave no leaked thread —
    # close()'s phase-2 cancel unwinds the httpx stream and reaps the background thread.
    tarpit = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tarpit.bind(("127.0.0.1", 0))
    tarpit.listen(8)
    port = tarpit.getsockname()[1]
    held: list = []
    stop = threading.Event()

    def _accept():
        tarpit.settimeout(0.25)
        while not stop.is_set():
            try:
                held.append(tarpit.accept()[0])   # hold the connection open, never respond
            except OSError:
                pass

    accepter = threading.Thread(target=_accept, daemon=True)
    accepter.start()
    try:
        before = {t.name for t in threading.enumerate()}
        cat = McpCatalog([{"name": "tarpit", "url": f"http://127.0.0.1:{port}/mcp"}],
                         connect="lazy", timeout=2.0)
        outcome: dict = {}

        def _load():
            try:
                cat.load("tarpit")
            except Exception as exc:  # noqa: BLE001
                outcome["error"] = type(exc).__name__

        worker = threading.Thread(target=_load, daemon=True)
        t0 = time.monotonic()
        worker.start()
        worker.join(2.0 + 2 * _CLOSE_GRACE + 10)     # generous watchdog: a REGRESSION wedges, not fails
        assert not worker.is_alive(), "load() on a tarpit wedged — not bounded"
        assert "error" in outcome                    # it RAISED (timeout / connect error), didn't hang
        assert time.monotonic() - t0 < 2.0 + 2 * 2.0 + 8   # grace = min(_CLOSE_GRACE, timeout=2) = 2
        cat.close()
        deadline = time.monotonic() + _CLOSE_GRACE + 3    # poll for the reap (grace-edge timing)
        while ("rlm-kit-mcp" in ({t.name for t in threading.enumerate()} - before)
               and time.monotonic() < deadline):
            time.sleep(0.1)
        assert "rlm-kit-mcp" not in ({t.name for t in threading.enumerate()} - before)
    finally:
        stop.set()
        tarpit.close()
        for conn in held:
            with contextlib.suppress(OSError):
                conn.close()


def test_mcp_connection_wedged_stdio_child_is_reaped(tmp_path):
    import os
    import time

    # A stdio "server" that NEVER speaks MCP: record its pid, then sleep. start() times out on the
    # handshake; close()'s phase-2 cancel must unwind stdio_client's __aexit__ and TERMINATE the child
    # (this also covers today's eager-path child leak). timeout=3 gives the stdio unwind headroom.
    pidfile = tmp_path / "child.pid"
    silent = tmp_path / "silent_server.py"
    silent.write_text(
        "import os, time\n"
        f"open({str(pidfile)!r}, 'w').write(str(os.getpid()))\n"
        "time.sleep(3600)\n"
    )
    conn = McpConnection({"command": sys.executable, "args": [str(silent)]}, timeout=3.0)
    with pytest.raises(Exception):  # noqa: B017 — start() times out (the child never initializes)
        conn.start()
    conn.close()

    deadline = time.monotonic() + _CLOSE_GRACE + 8
    pid = None
    while pid is None and time.monotonic() < deadline:
        try:
            pid = int(pidfile.read_text())
        except (OSError, ValueError):
            time.sleep(0.05)
    assert pid is not None, "the stdio child never spawned"
    reaped = False
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)          # 0 == existence probe; raises when the child is gone
            time.sleep(0.1)
        except (ProcessLookupError, PermissionError):
            reaped = True
            break
    assert reaped, f"stdio child {pid} was not reaped by close()"


def test_mcp_connection_healthy_close_is_fast_and_uncancelled(tmp_path):
    import time

    # A HEALTHY connection is awaiting _stop, so close()'s phase 1 exits the thread in ms — phase 2
    # (cancel) must NOT fire (it would add a whole grace window and cancel the serve task).
    conn = McpConnection(_server(tmp_path), timeout=30)
    conn.start()
    assert [t.name for t in conn.tools] == ["echo"]
    t0 = time.monotonic()
    conn.close()
    assert time.monotonic() - t0 < 2.0                                   # phase-1 join returned fast
    assert not (conn._serve_task is not None and conn._serve_task.cancelled())  # phase 2 never fired
