"""MCP client — expose an EXTERNAL MCP server's tools to an ``RLMTask`` as SYNC tools.

rlm-harness is an MCP **client only**: it never runs a server and bundles none. You point
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

Two public surfaces:

- ``mcp_tools(server)`` — the SINGLE-server convenience: one server's tools as ``dspy.Tool``s for
  ``RLMTask(tools=…)``, materialized up front, each call self-recording a ``tool_call``.
- ``McpCatalog(specs)`` + ``McpConnection`` — a MULTI-server, queryable transport for a consumer
  building a PROGRESSIVE tool surface (list servers → load one on demand → read its tools → call).
  It returns RAW MCP objects (not ``dspy.Tool``s) and records NOTHING — the consumer's own tool
  wrapper owns any ``tool_call`` — so it stays dspy-free and the consumer maps tools to its shape.
  ``result_text`` flattens a ``CallToolResult`` to text.

Optional: needs ``pip install "rlm-harness[mcp]"``.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import json
import threading
from collections.abc import Iterator
from typing import Any

from ._toolname import signature_from_json_schema, unique_tool_names
from .trace import record_tool_call

# Head of a tool result recorded for inspection (a replay UI shows it) — like read_skill / fetch,
# the trace keeps only a preview, not the full (possibly bulk) output that goes to the RLM's REPL.
_PREVIEW = 700

# Grace window for each phase of McpConnection.close (graceful stop, then cancel). Capped by the
# connection's own timeout so a small timeout doesn't inflate teardown.
_CLOSE_GRACE = 5.0

# A server spec: a bare URL string, {"url": ...} (streamable-HTTP), or
# {"command": ..., "args": [...], "env": {...}} (stdio subprocess).
ServerSpec = str | dict


def _require_mcp() -> None:
    try:
        import mcp  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "MCP support requires the optional dependency: pip install 'rlm-harness[mcp]'"
        ) from exc


def result_text(result: Any) -> str:
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
            except Exception:
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


class McpConnection:
    """A live connection to ONE external MCP server: a ``ClientSession`` driven from a dedicated
    background thread + event loop, kept alive until :meth:`close`. The SDK's session API is async;
    callers here are sync, so each sync call bridges one coroutine across the thread boundary via
    ``run_coroutine_threadsafe(...).result(timeout)``. PUBLIC — a consumer building its own tool
    surface can drive a connection directly; :class:`McpCatalog` manages many of these.

    ``server`` is a bare URL string, ``{"url": ...}`` (streamable-HTTP), or
    ``{"command": ..., "args": [...], "env": {...}}`` (stdio subprocess). After :meth:`start`,
    ``tools`` holds the server's listed MCP ``Tool`` objects; :meth:`call` returns the raw
    ``CallToolResult`` (flatten it with :func:`result_text`). Needs ``rlm-harness[mcp]``."""

    def __init__(self, server: ServerSpec, *, timeout: float = 30.0) -> None:
        self._server = server
        self._timeout = timeout
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, name="rlm-harness-mcp", daemon=True)
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._stop: asyncio.Event | None = None
        self._serve_task: Any = None  # the _serve() Task; close() cancels it to unwind a WEDGED connect
        self._session: Any = None
        self.tools: list = []

    # -- background thread: owns the loop + the LIVE session ----------------
    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._serve_task = self._loop.create_task(self._serve())  # legal before run_until_complete
            self._loop.run_until_complete(self._serve_task)
        except asyncio.CancelledError:
            pass  # close() cancelled the serve task — a clean shutdown, not an _error
        except BaseException as exc:
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
        # Phase 1 — graceful: ask _serve to return (it unwinds the session + transport cleanly). A
        # HEALTHY connection is awaiting `self._stop.wait()`, so this exits the thread in milliseconds.
        if self._stop is not None and self._loop.is_running():
            with contextlib.suppress(RuntimeError):  # loop may close between the check and the call
                self._loop.call_soon_threadsafe(self._stop.set)
        if self._thread.ident is None:
            return  # never started — join would raise (a public close() may precede start())
        grace = min(_CLOSE_GRACE, self._timeout)
        self._thread.join(grace)
        # Phase 2 — cancel: a WEDGED connect (e.g. a tarpit server) never reached `await
        # self._stop.wait()`, so setting _stop was a no-op and the thread is still alive. Cancel the
        # serve task to unwind through the session/transport __aexit__ (close the httpx stream /
        # terminate the stdio child) and reap the thread — instead of leaking it plus the child/socket.
        if self._thread.is_alive() and self._serve_task is not None and self._loop.is_running():
            with contextlib.suppress(RuntimeError):
                self._loop.call_soon_threadsafe(self._serve_task.cancel)
            self._thread.join(grace)


def _defers(spec: dict) -> bool:
    """Whether ``connect="lazy"`` defers this server's connect to its first ``load()``. Mirrors
    :meth:`McpConnection._transport`'s precedence (a ``url`` wins over a ``command``): only URL
    (streamable-HTTP) servers defer — a stdio server's local subprocess spawn stays eager (pre-run)."""
    return "url" in spec


class McpCatalog:
    """A queryable, long-lived transport over SEVERAL external MCP servers — for a consumer building a
    PROGRESSIVE tool surface (list servers → load one on demand → read its tools → call one). Each
    server runs behind its own :class:`McpConnection` (a background-thread session). The catalog
    records NOTHING (the consumer's own tool wrapper owns any ``tool_call``) and returns RAW MCP
    ``Tool`` objects (name / description / ``inputSchema``), not ``dspy.Tool``s — so it stays
    dspy-free and the consumer maps tools to its own shape.

    ``specs`` is a list of dicts, each ``{"name", "description", ...connection...}`` where the
    connection is ``"url"`` (streamable-HTTP) or ``"command"``/``"args"``/``"env"`` (stdio) — the
    same fields :class:`McpConnection` accepts. ``connect="eager"`` (default) connects every server
    host-side up front and tears down a partial connect on failure. ``connect="lazy"`` defers each
    **URL (streamable-HTTP)** server's connect to its first :meth:`load` — safe mid-run: the handshake
    runs on the connection's OWN background thread + loop (the caller's wait is ``timeout``-bounded,
    and a wedged connect is cancelled and reaped by :meth:`close`); **stdio** servers still connect
    eagerly in ``__init__`` (deferring a local subprocess spawn buys nothing, and keeps the spawn out
    of the loop). ``connect="lazy"`` is opt-in/experimental. Needs ``rlm-harness[mcp]``."""

    def __init__(self, specs: list[dict], *, connect: str = "eager", timeout: float = 60.0) -> None:
        _require_mcp()
        if connect not in ("eager", "lazy"):
            raise ValueError(f"connect must be 'eager' or 'lazy', got {connect!r}")
        self._specs: dict[str, dict] = {}
        for s in specs:
            if not isinstance(s, dict) or not s.get("name"):
                raise ValueError("each MCP catalog spec must be a dict with a 'name'")
            self._specs[str(s["name"])] = s
        self._timeout = timeout
        self._conns: dict[str, McpConnection] = {}
        # eager: connect every server up front. lazy: connect only the servers that DON'T defer
        # (stdio — a local spawn stays pre-run) up front, and leave the URL servers for their first
        # load(). A spec with neither url nor command classifies as non-deferring and fails fast in
        # _transport, same as under eager.
        try:
            for name, spec in self._specs.items():
                if connect == "eager" or not _defers(spec):
                    self._connect(name)
        except Exception:
            # A server's connect failed — the servers already connected are live threads +
            # subprocesses with no object left for the caller to close(). Tear them down before
            # propagating, so a partial connect never leaks.
            self.close()
            raise

    def _connect(self, server: str) -> McpConnection:
        if server in self._conns:
            return self._conns[server]
        if server not in self._specs:
            raise KeyError(server)
        conn = McpConnection(self._specs[server], timeout=self._timeout)
        try:
            conn.start()
        except Exception:
            with contextlib.suppress(Exception):
                conn.close()  # a failed start still spawned a thread/subprocess — don't leak it
            raise
        self._conns[server] = conn
        return conn

    def servers(self) -> list[tuple[str, str]]:
        """``[(name, description)]`` for every DECLARED server (connected or not)."""
        return [(name, str(spec.get("description", ""))) for name, spec in self._specs.items()]

    def has_server(self, server: str) -> bool:
        return server in self._specs

    def load(self, server: str) -> None:
        """Connect ``server`` (no-op if already connected; the on-demand path under ``connect='lazy'``)."""
        self._connect(server)

    def tools(self, server: str) -> list:
        """The raw MCP ``Tool`` objects of a CONNECTED server (``[]`` if not yet loaded)."""
        conn = self._conns.get(server)
        return list(conn.tools) if conn is not None else []

    def tool_names(self, server: str) -> list[str]:
        return [t.name for t in self.tools(server)]

    def call(self, server: str, tool: str, args: dict | None = None) -> str:
        """Call ``tool`` on a CONNECTED ``server`` and return the flattened result TEXT."""
        conn = self._conns.get(server)
        if conn is None:
            raise RuntimeError(f"MCP server {server!r} is not connected (load it first)")
        return result_text(conn.call(tool, args or {}))

    def close(self) -> None:
        for conn in self._conns.values():
            with contextlib.suppress(Exception):
                conn.close()
        self._conns.clear()


def _repl_alias(name: str, repl: str) -> dict:
    """The optional ``repl_name`` payload field — emitted ONLY when sanitising changed the name.

    Additive within ``rlm-harness/trace/v1`` (a new OPTIONAL payload field is allowed; the
    envelope, the event types and the established fields are untouched). Conditional so the
    common case stays byte-identical to pre-1.0.2 payloads — nine consumers hold golden
    fixtures, and a field that appears on every MCP event would churn all of them for nothing.

    It has to exist at all because the mapping is UNRECOVERABLE offline: the sanitised name
    depends on the server's whole tool list at run time, which never enters the trace. Without
    it, a reader correlating the planner's code (``main_step.payload.code`` shows the model
    typing ``get_weather(...)``) against the tool events (which record ``get-weather``) has no
    way to join them. Read it as ``payload.get("repl_name") or payload["tool"]``, which
    degrades correctly for older traces and for every non-MCP tool.
    """
    return {"repl_name": repl} if repl != name else {}


def _make_tool(dspy_mod: Any, bridge: McpConnection, mcp_tool: Any, prefix: str,
               repl_name: str | None = None):
    # THREE identities, and conflating any two of them is a bug:
    #   `mcp_tool.name` — the WIRE name. What `bridge.call(...)` sends back to the server.
    #                     Never derived, never sanitised.
    #   `name`          — prefix + wire name. The TRACE identity (`record_tool_call`), so a
    #                     reader sees what the operator configured. Also never sanitised.
    #   `repl_name`     — what the MODEL types in the sandbox. MUST be a Python identifier.
    # Hyphens and dots are the MCP naming norm (`get-weather`, `db.query`) and dspy refuses
    # both, aborting the WHOLE registration — one bad name takes every other tool with it.
    # `repl_name` is computed by the CALLER across the server's full tool list, because
    # uniqueness cannot be decided one tool at a time (`get-weather` and `get.weather` both
    # clean to `get_weather`).
    name = f"{prefix}{mcp_tool.name}"
    repl = repl_name or name
    desc = mcp_tool.description or f"MCP tool {name}"
    schema = mcp_tool.inputSchema if isinstance(mcp_tool.inputSchema, dict) else None
    props = _args_from_schema(schema)

    def call(**kwargs: Any) -> str:
        # The REPL sandbox proxy forwards EVERY declared param — incl. a defaulted optional the model
        # omitted, sent as None. Drop None so an unset optional isn't posted as JSON null into a strict
        # (additionalProperties:false / typed) server schema.
        args = {k: v for k, v in kwargs.items() if v is not None}
        try:
            result = bridge.call(mcp_tool.name, args)     # WIRE name — unprefixed, unsanitised
        except Exception as exc:
            record_tool_call(name, args=args, ok=False, note=f"error: {type(exc).__name__}",
                             **_repl_alias(name, repl))
            # Model-facing text uses the name the model can actually call.
            return f"MCP tool {repl!r} error: {type(exc).__name__}: {str(exc)[:200]}"
        text = result_text(result)
        ok = not getattr(result, "isError", False)
        record_tool_call(
            name, args=args, ok=ok, preview=text[:_PREVIEW],
            note="ok" if ok else "tool reported an error",
            **_repl_alias(name, repl),
        )
        return text

    call.__name__ = repl   # cosmetic; dspy reads the explicit `name=` at the return below
    call.__doc__ = desc
    # dspy.RLM injects `tool.func` (this `call`) into its PythonInterpreter, which builds the sandbox
    # tool proxy from ``inspect.signature(func)`` — NOT from ``dspy.Tool.args``. A bare ``**kwargs``
    # wrapper therefore registers a single param literally named "kwargs", so the model calls e.g.
    # ``get_vulnerability(kwargs=...)`` and the server rejects the unexpected property. The rule
    # itself lives in `signature_from_json_schema` (public since 1.1.0) so a consumer building its
    # own tools from `McpCatalog` names uses the SAME derivation rather than re-deriving it.
    #
    # UNCONDITIONAL — no `if schema is not None` guard. A schema-less tool used to keep the broken
    # ``**kwargs`` proxy while the comment here claimed otherwise; the SDK makes `inputSchema` a
    # required dict so that branch is not reachable through `mcp_tools`, but a caller constructing
    # `_make_tool` directly could hit it, and a zero-parameter signature is right for a genuinely
    # no-argument tool either way.
    try:
        call.__signature__ = signature_from_json_schema(schema)
    except (ValueError, TypeError):
        # KNOWINGLY DEGRADED, and the only honest outcome: a property named `from` or `db.query`
        # cannot be an `inspect.Parameter`. Sanitising it is NOT an option — the proxy forwards the
        # parameter name to the server as a JSON key, so a renamed property sends wrong wire
        # arguments. Keeping `**kwargs` leaves a tool the model can still call (dspy accepts it;
        # only the arg NAMES are lost), whereas stamping a bad name would emit `def tool(from):`
        # into the Deno stub and abort registration for EVERY tool on the task.
        # `assert_repl_safe` rejects this shape by design — it is a real degradation, recorded
        # here rather than hidden.
        call.__repl_degraded__ = (  # type: ignore[attr-defined]
            f"tool {name!r}: a property name in this server's schema cannot be a Python "
            f"parameter, so the wrapper keeps its **kwargs shape and the model cannot pass "
            f"arguments by name"
        )
    # `name=` is what dspy VALIDATES and what it registers in the sandbox — NOT
    # `call.__name__` (`dspy.Tool` only falls back to the function's name when `name=` is
    # omitted). Sanitising `__name__` alone is a placebo: the raw name still reaches dspy
    # and still raises. This line is the fix.
    return dspy_mod.Tool(call, name=repl, desc=desc, args=props)


@contextlib.contextmanager
def mcp_tools(server: ServerSpec, *, timeout: float = 30.0, prefix: str = "") -> Iterator[list]:
    """Connect to an EXTERNAL MCP server and yield its tools as sync ``dspy.Tool``s for
    ``RLMTask(tools=…)``. rlm-harness is a CLIENT only — point this at someone else's server.

    ``server``: a stdio spec ``{"command": "npx", "args": ["-y", "some-mcp"], "env": {...}}``, or a
    streamable-HTTP spec ``{"url": "https://host/mcp"}`` (or a bare URL string). ``prefix`` is an
    optional tool-name prefix to disambiguate tools when wiring several servers.

    The connection is LIVE for the ``with`` block and torn down on exit (a stdio subprocess is
    terminated). Tool calls are SYNC (bridged from the SDK's async API). Each call records a
    ``tool_call`` trace event. Needs ``rlm-harness[mcp]``.

        with mcp_tools({"url": "https://mcp.example.com/mcp"}) as tools:
            result = MyTask(tools=tools).run(...)

    SECURITY: tools run HOST-SIDE (outside the sandbox); treat the server as a trusted dependency
    and its output as untrusted LM context."""
    _require_mcp()
    import dspy

    bridge = McpConnection(server, timeout=timeout)
    try:
        # start() inside the try so a start failure (timeout / a server that errors on init) still
        # runs close() — otherwise the background thread + any spawned stdio subprocess would leak.
        bridge.start()
        # Resolve every REPL name in ONE pass over the server's full tool list: uniqueness
        # is a property of the SET, so per-tool sanitising could map two server tools onto
        # the same identifier and trade an invalid-name failure for a duplicate-name one.
        repl_names = unique_tool_names(f"{prefix}{t.name}" for t in bridge.tools)
        yield [
            _make_tool(dspy, bridge, t, prefix, repl_names[f"{prefix}{t.name}"])
            for t in bridge.tools
        ]
    finally:
        bridge.close()
