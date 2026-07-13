"""In-container agent for the ``container`` interpreter — runs INSIDE the sandbox.

STDLIB ONLY. Imports nothing from ``rlm_kit`` or ``dspy``: it is delivered to a bare
``python:*-slim`` container via ``python -u -c <this source>`` (see
``container_interpreter.py``) and must run with no third-party packages.

It speaks line-delimited JSON-RPC 2.0 over the container's original stdin/stdout (the
``docker run -i`` pipe), the same message shapes dspy's Deno ``runner.js`` uses, so the
host ``ContainerInterpreter`` is a drop-in analog of dspy's ``PythonInterpreter``:

  host -> agent:
    {"method":"register","params":{"tools":[{"name","parameters":[{"name","type?"}]}],
                                   "outputs":[{"name","type?"}]},"id":N}
    {"method":"execute","params":{"code": "..."},"id":N}
    {"method":"shutdown"}                                          (notification)

  agent -> host (while an execute is in flight — a tool/llm_query callback):
    {"method":"tool_call","params":{"name":T,"kwargs":{...}},"id":"cb-K"}
      -> host replies {"result":{"value":V,"type":"string"|"json"},"id":"cb-K"}
                 or   {"error":{"code":C,"message":M,"data":{"type":E}},"id":"cb-K"}

  agent -> host (terminating an execute):
    {"result":{"output":"captured stdout+stderr"},"id":N}          normal
    {"result":{"final":{...},"output":"..."},"id":N}               SUBMIT() called
    {"error":{"code":-32000,...},"id":N}                           SyntaxError (host re-raises)
    {"error":{"code":-32007,...},"id":N}                           runtime error

fd hygiene: fds 0/1 are dup'd for the RPC channel at startup, then fd 0 is pointed at
/dev/null. During execute, fds 1/2 are redirected into a temp file, so Python ``print()``
AND a native ``subprocess`` child both inherit the capture file and cannot corrupt the
RPC channel — this is what makes native ``subprocess.run(...)`` in the REPL safe here.
"""
import json
import os
import sys
import tempfile
import threading
import traceback

# ---- claim the RPC channel before any user code runs -------------------------
RPC_IN = os.fdopen(os.dup(0), "r", encoding="utf-8")
RPC_OUT = os.fdopen(os.dup(1), "w", encoding="utf-8")
_devnull = os.open(os.devnull, os.O_RDONLY)
os.dup2(_devnull, 0)          # user code reading stdin gets EOF, not RPC frames
os.close(_devnull)

_RPC_LOCK = threading.Lock()  # serializes tool_call frames if user code spawns threads
NAMESPACE = {"__name__": "__rlm_sandbox__"}
TOOL_PARAMS = {}              # tool name -> ordered param names (positional mapping)
_callback_id = 0


class _SubmitCalled(BaseException):
    """SUBMIT() control signal. A ``BaseException`` (NOT ``Exception``) so that user
    code doing ``try: SUBMIT(...) except Exception`` cannot swallow the terminate signal."""

    def __init__(self, output):
        self.output = output


def _send(obj):
    RPC_OUT.write(json.dumps(obj) + "\n")
    RPC_OUT.flush()


def _tool_call(name, args, kwargs):
    """Broker one tool invocation to the host and return its result."""
    global _callback_id
    params = TOOL_PARAMS.get(name, [])
    ka = dict(kwargs)
    for i, a in enumerate(args):
        if i >= len(params):
            raise TypeError("%s() got too many positional arguments" % name)
        pname = params[i]
        if pname in ka:
            raise TypeError("%s() got multiple values for %r" % (name, pname))
        ka[pname] = a
    with _RPC_LOCK:
        _callback_id += 1
        cid = "cb-%d" % _callback_id
        _send({"jsonrpc": "2.0", "method": "tool_call",
               "params": {"name": name, "kwargs": ka}, "id": cid})
        line = RPC_IN.readline()
    if not line:
        raise RuntimeError("host closed RPC channel during tool call")
    resp = json.loads(line)
    if resp.get("id") != cid:
        raise RuntimeError("tool_call response id mismatch: %r" % (resp.get("id"),))
    if "error" in resp:
        raise RuntimeError("[tool error] %s" % resp["error"].get("message", "unknown"))
    r = resp["result"]
    if r.get("type") == "json":
        return json.loads(r["value"])
    return r["value"]


def _make_proxy(name):
    def proxy(*args, **kwargs):
        return _tool_call(name, args, kwargs)
    proxy.__name__ = name
    return proxy


def _register(params):
    for t in params.get("tools", []):
        TOOL_PARAMS[t["name"]] = [p["name"] for p in t.get("parameters", [])]
        NAMESPACE[t["name"]] = _make_proxy(t["name"])
    outputs = params.get("outputs")
    if outputs:
        names = [o["name"] for o in outputs]
        sig = ", ".join(
            "%s: %s" % (o["name"], o["type"]) if o.get("type") else o["name"]
            for o in outputs
        )
        body = ", ".join("'%s': %s" % (n, n) for n in names)
        src = "def SUBMIT(%s):\n    raise _SubmitCalled({%s})\n" % (sig, body)
        exec(src, {"_SubmitCalled": _SubmitCalled}, NAMESPACE)
    elif "SUBMIT" not in NAMESPACE:
        def SUBMIT(**kwargs):
            raise _SubmitCalled(kwargs)
        NAMESPACE["SUBMIT"] = SUBMIT


def _run_code(compiled):
    """exec() in the persistent namespace with fds 1/2 captured to a temp file."""
    tmp = tempfile.TemporaryFile()
    sys.stdout.flush()
    sys.stderr.flush()
    saved1, saved2 = os.dup(1), os.dup(2)
    os.dup2(tmp.fileno(), 1)
    os.dup2(tmp.fileno(), 2)
    final = None
    try:
        try:
            exec(compiled, NAMESPACE)
        except _SubmitCalled as s:
            final = s.output
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved1, 1)
        os.dup2(saved2, 2)
        os.close(saved1)
        os.close(saved2)
        tmp.seek(0)
        captured = tmp.read().decode("utf-8", "replace")
        tmp.close()
    return final, captured


def main():
    while True:
        line = RPC_IN.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            sys.stderr.write("agent: skipping malformed frame\n")
            continue
        method, mid = msg.get("method"), msg.get("id")

        if method == "shutdown":
            break

        if method == "register":
            try:
                _register(msg.get("params", {}))
                _send({"jsonrpc": "2.0", "result": {"ok": True}, "id": mid})
            except Exception as e:
                _send({"jsonrpc": "2.0", "id": mid,
                       "error": {"code": -32007, "message": str(e),
                                 "data": {"type": type(e).__name__}}})
            continue

        if method == "execute":
            code = msg.get("params", {}).get("code", "")
            try:
                compiled = compile(code, "<sandbox>", "exec")
            except SyntaxError as e:
                _send({"jsonrpc": "2.0", "id": mid,
                       "error": {"code": -32000, "message": str(e),
                                 "data": {"type": "SyntaxError"}}})
                continue
            try:
                final, captured = _run_code(compiled)
            except Exception as e:
                _send({"jsonrpc": "2.0", "id": mid,
                       "error": {"code": -32007,
                                 "message": "%s: %s" % (type(e).__name__, e),
                                 "data": {"type": type(e).__name__,
                                          "traceback": traceback.format_exc()[-2000:]}}})
                continue
            result = {"output": captured}
            if final is not None:
                result["final"] = final
            _send({"jsonrpc": "2.0", "result": result, "id": mid})
            continue

        _send({"jsonrpc": "2.0", "id": mid,
               "error": {"code": -32601, "message": "unknown method %r" % method,
                         "data": {"type": "CodeInterpreterError"}}})


if __name__ == "__main__":
    main()
