"""MCP client tests — the async->sync bridge is the load-bearing mechanic, so these run a REAL
stdio MCP server (a tiny FastMCP echo) as a subprocess and drive it through ``mcp_tools``. Skips
when the optional ``mcp`` extra (or dspy) is absent, like the other dspy-bearing tests."""

import sys

import pytest

pytest.importorskip("mcp")
pytest.importorskip("dspy")

from rlm_kit.mcp import McpCatalog, McpConnection, _args_from_schema, mcp_tools, result_text  # noqa: E402
from rlm_kit.trace import TraceRecorder, load_events  # noqa: E402

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


def test_mcp_catalog_lazy_defers_connect(tmp_path):
    cat = McpCatalog([_named(tmp_path, "echo")], connect="lazy", timeout=30)
    try:
        assert cat.tools("echo") == []                    # not connected yet under lazy
        cat.load("echo")
        assert cat.tool_names("echo") == ["echo"]         # connected on demand
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
