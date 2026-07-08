"""MCP client — expose an EXTERNAL MCP server's tools to an ``RLMTask`` as SYNC tools.

rlm-kit is an MCP **client only**: it never runs a server and bundles none. You point
``mcp_tools(...)`` at someone else's server — a local stdio command, or a remote
streamable-HTTP URL — and get that server's tools back as sync ``dspy.Tool``s ready for
``RLMTask(tools=…)``.

Why a bridge: the MCP Python SDK is **async** (``ClientSession.call_tool`` is a coroutine), but
``dspy.RLM`` invokes tools **synchronously** from its sandbox bridge
(``PythonInterpreter._handle_tool_call``: ``self.tools[name](**kwargs)`` — no ``await``). So this
module runs the ``ClientSession`` in a dedicated background thread + event loop, kept alive for the
whole ``with`` block, and each tool is a sync wrapper that bridges one call across the thread
boundary via ``run_coroutine_threadsafe(...).result(timeout)``. (dspy's own
``dspy.Tool.from_mcp_tool`` produces an *async* tool for ``dspy.ReAct.acall`` — unusable on the
RLM's sync path, which is why this bridge exists.)

SECURITY: MCP tools execute HOST-SIDE — *outside* the sandbox. A stdio server is a subprocess this
process spawns; an HTTP server is a remote you trust. Treat an MCP server as a trusted dependency,
and its tool OUTPUT as untrusted LM context (a prompt-injection surface, like fetched web content).

Optional: needs ``pip install "rlm-kit[mcp]"``.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import json
import threading
from typing import Any, Iterator, Optional, Union

from .trace import record_tool_call

# Head of a tool result recorded for inspection (a replay UI shows it) — like read_skill / fetch,
# the trace keeps only a preview, not the full (possibly bulk) output that goes to the RLM's REPL.
_PREVIEW = 700

# A server spec: a bare URL string, {"url": ...} (streamable-HTTP), or
# {"command": ..., "args": [...], "env": {...}} (stdio subprocess).
ServerSpec = Union[str, dict]


def _require_mcp() -> None:
    try:
        import mcp  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "MCP support requires the optional dependency: pip install 'rlm-kit[mcp]'"
        ) from exc


def _result_text(result: Any) -> str:
    """Flatten a ``CallToolResult`` to text: join the ``TextContent`` blocks; fall back to
    ``structuredContent`` (as JSON) when there is no text; prefix an error marker if ``isError``."""
    parts = [
        block.text
        for block in (getattr(result, "content", None) or [])
        if getattr(block, "text", None) is not None
    ]
    out = "\n".join(parts).strip()
    if not out:
        structured = getattr(result, "structuredContent", None)
        if structured is not None:
            try:
                out = json.dumps(structured, ensure_ascii=False, default=str)
            except Exception:  # noqa: BLE001
                out = str(structured)
    if getattr(result, "isError", False):
        out = f"[tool reported an error] {out}".strip()
    return out


def _args_from_schema(input_schema: Any) -> dict:
    """Map an MCP tool's ``inputSchema`` (a JSON Schema object) to ``dspy.Tool``'s ``args``
    (a dict of {arg: schema-fragment}) — i.e. its ``properties``, or ``{}`` if absent."""
    if isinstance(input_schema, dict):
        props = input_schema.get("properties")
        if isinstance(props, dict):
            return props
    return {}


class _MCPBridge:
    """An MCP ``ClientSession`` driven from a background thread + event loop, kept alive until
    :meth:`close`. The session API is async; RLM tools are sync, so sync callers bridge a coroutine
    across the thread boundary via ``run_coroutine_threadsafe(...).result(timeout)``."""

    def __init__(self, server: ServerSpec, *, timeout: float) -> None:
        self._server = server
        self._timeout = timeout
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, name="rlm-kit-mcp", daemon=True)
        self._ready = threading.Event()
        self._error: Optional[BaseException] = None
        self._stop: Optional[asyncio.Event] = None
        self._session: Any = None
        self.tools: list = []

    # -- background thread: owns the loop + the LIVE session ----------------
    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except BaseException as exc:  # noqa: BLE001 — surfaced to start()
            self._error = exc
            self._ready.set()
        finally:
            with contextlib.suppress(Exception):
                self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            self._loop.close()

    async def _serve(self) -> None:
        from mcp import ClientSession

        self._stop = asyncio.Event()
        async with self._transport() as streams:
            read, write = streams[0], streams[1]
            async with ClientSession(read, write) as session:
                await session.initialize()
                listed = await session.list_tools()
                self._session = session
                self.tools = list(listed.tools)
                self._ready.set()
                await self._stop.wait()  # keep session + transport alive until close()

    def _transport(self):
        srv = self._server
        if isinstance(srv, str) or (isinstance(srv, dict) and "url" in srv):
            import mcp.client.streamable_http as _sh

            # The SDK renamed streamablehttp_client → streamable_http_client (the old name is now
            # deprecated). Prefer the new name, fall back to the old so the mcp>=1.0 floor keeps
            # working; both accept a bare url and yield the same (read, write, get_session_id)
            # transport, so the call site is unchanged.
            streamable_client = getattr(_sh, "streamable_http_client", None) or _sh.streamablehttp_client
            return streamable_client(srv if isinstance(srv, str) else srv["url"])
        if not (isinstance(srv, dict) and srv.get("command")):
            raise ValueError(
                "MCP server spec must be a URL string, {'url': ...}, or "
                "{'command': ..., 'args': [...]}"
            )
        from mcp import StdioServerParameters
        from mcp.client.stdio import stdio_client

        return stdio_client(
            StdioServerParameters(
                command=srv["command"], args=list(srv.get("args", [])), env=srv.get("env")
            )
        )

    # -- sync API for the main thread --------------------------------------
    def start(self) -> None:
        self._thread.start()
        if not self._ready.wait(self._timeout):
            raise TimeoutError(f"MCP server did not become ready within {self._timeout}s")
        if self._error is not None:
            raise self._error

    def call(self, name: str, arguments: dict) -> Any:
        if self._session is None:
            raise RuntimeError("MCP session is not connected")
        fut = asyncio.run_coroutine_threadsafe(
            self._session.call_tool(name, arguments or {}), self._loop
        )
        try:
            return fut.result(self._timeout)
        except concurrent.futures.TimeoutError:
            # Don't leave a hung call_tool coroutine running in the loop — the session is serial,
            # so it would wedge every later call. Request its cancellation and surface the timeout.
            fut.cancel()
            raise TimeoutError(f"MCP tool {name!r} timed out after {self._timeout}s") from None

    def close(self) -> None:
        if self._stop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(self._timeout)


def _make_tool(dspy_mod: Any, bridge: _MCPBridge, mcp_tool: Any, prefix: str):
    name = f"{prefix}{mcp_tool.name}"
    desc = mcp_tool.description or f"MCP tool {name}"

    def call(**kwargs: Any) -> str:
        try:
            result = bridge.call(mcp_tool.name, kwargs)
        except Exception as exc:  # noqa: BLE001 — surface as text so the RLM can react
            record_tool_call(name, args=kwargs, ok=False, note=f"error: {type(exc).__name__}")
            return f"MCP tool {name!r} error: {type(exc).__name__}: {str(exc)[:200]}"
        text = _result_text(result)
        ok = not getattr(result, "isError", False)
        record_tool_call(
            name, args=kwargs, ok=ok, preview=text[:_PREVIEW],
            note="ok" if ok else "tool reported an error",
        )
        return text

    call.__name__ = name
    call.__doc__ = desc
    return dspy_mod.Tool(call, name=name, desc=desc, args=_args_from_schema(mcp_tool.inputSchema))


@contextlib.contextmanager
def mcp_tools(server: ServerSpec, *, timeout: float = 30.0, prefix: str = "") -> Iterator[list]:
    """Connect to an EXTERNAL MCP server and yield its tools as sync ``dspy.Tool``s for
    ``RLMTask(tools=…)``. rlm-kit is a CLIENT only — point this at someone else's server.

    ``server``: a stdio spec ``{"command": "npx", "args": ["-y", "some-mcp"], "env": {...}}``, or a
    streamable-HTTP spec ``{"url": "https://host/mcp"}`` (or a bare URL string). ``prefix`` is an
    optional tool-name prefix to disambiguate tools when wiring several servers.

    The connection is LIVE for the ``with`` block and torn down on exit (a stdio subprocess is
    terminated). Tool calls are SYNC (bridged from the SDK's async API). Each call records a
    ``tool_call`` trace event. Needs ``rlm-kit[mcp]``.

        with mcp_tools({"url": "https://mcp.example.com/mcp"}) as tools:
            result = MyTask(tools=tools).run(...)

    SECURITY: tools run HOST-SIDE (outside the sandbox); treat the server as a trusted dependency
    and its output as untrusted LM context."""
    _require_mcp()
    import dspy

    bridge = _MCPBridge(server, timeout=timeout)
    try:
        # start() inside the try so a start failure (timeout / a server that errors on init) still
        # runs close() — otherwise the background thread + any spawned stdio subprocess would leak.
        bridge.start()
        yield [_make_tool(dspy, bridge, t, prefix) for t in bridge.tools]
    finally:
        bridge.close()
