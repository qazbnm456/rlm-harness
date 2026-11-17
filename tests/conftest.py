"""Guards that have to run before any test writes its first file."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


#: `tempfile` reads `$TMPDIR`, `$TEMP` and `$TMP`, then the platform's own temp locations, and its
#: candidate list ends with the CURRENT DIRECTORY. On a machine where nothing earlier in that list
#: can be written, every `TemporaryDirectory()` in this suite lands in the repository instead.
#: pytest's own `tmp_path` reads the same `gettempdir()`, so using the fixture everywhere would not
#: help. It happened once and left no trace while it ran: three `tmp*/m.db` stores reached a commit,
#: from the `TemporaryDirectory()` blocks in `tests/test_tool_descriptions.py` that hand a
#: `MemoryStore` a path under them, and hatchling put them in an sdist, because nothing in the suite
#: had any reason to look. One session-start check covers every temp consumer, including pytest's;
#: a `.gitignore` entry would only have hidden the next one.
#:
#: Scoped to a checkout, because what this prevents is a stray file reaching a commit. An unpacked
#: sdist carries this file too, and a build system that points `$TMPDIR` inside the source it
#: unpacked has nothing to commit a stray into, so it gets its test run.
def pytest_sessionstart(session: pytest.Session) -> None:
    repo = Path(__file__).resolve().parent.parent
    if not (repo / ".git").exists():
        return
    temp_dir = Path(tempfile.gettempdir()).resolve()
    if temp_dir.is_relative_to(repo):
        raise pytest.UsageError(
            f"tempfile.gettempdir() resolves to {temp_dir}, inside the repository at {repo}. "
            "Every temporary file this suite writes would be created in the working tree. "
            "Point $TMPDIR at a writable directory outside the repository and run again."
        )


class FakeOpenAIServer:
    """A local OpenAI-compatible Chat Completions server, for tests of what the kit puts on the wire.

    Records every request as ``{"path", "body"}`` in ``requests``. Answers each with the next entry of
    ``script`` (a message dict, or a callable taking the request body and returning one), and once the
    script is spent with :meth:`answer`, which fills the fields the request asks for: the planner's
    ``plan`` and ``code``, or the task's ``answer``, in the format the adapter expects. Setting
    ``reasoning_only`` puts that answer in ``reasoning_content`` and sends no content, as a reasoning
    model behind vLLM sometimes does.
    """

    PLANNER = {"plan": "look at q", "code": "print(1)"}
    EXTRACT = {"answer": "42"}

    def __init__(self) -> None:
        import http.server
        import json
        import threading

        self.requests: list[dict] = []
        self.script: list = []
        self.reasoning_only = False
        self.reasoning_content: str | None = None
        server = self

        class _Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                server.requests.append({"path": self.path, "body": body})
                message = server._next(body)
                out = {"id": "chatcmpl-1", "object": "chat.completion", "created": 0,
                       "model": body.get("model"),
                       "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                       "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}}
                data = json.dumps(out).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args) -> None:
                pass

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.05},
                                        daemon=True)
        self._thread.start()
        self.base_url = f"http://127.0.0.1:{self._server.server_port}/v1"

    def _next(self, body: dict) -> dict:
        if self.script:
            step = self.script.pop(0)
            return step(body) if callable(step) else step
        return self.answer(body)

    @staticmethod
    def asked_fields(body: dict) -> list[str]:
        """The output fields a request asks for: from its JSON schema, else from its system prompt."""
        schema = ((body.get("response_format") or {}).get("json_schema") or {}).get("schema") or {}
        if schema.get("properties"):
            return list(schema["properties"])
        system = body["messages"][0]["content"] if body["messages"][0]["role"] == "system" else ""
        planner = "[[ ## code ## ]]" in system or "<code>" in system
        return list(FakeOpenAIServer.PLANNER if planner else FakeOpenAIServer.EXTRACT)

    def answer(self, body: dict) -> dict:
        import json

        values = {**self.PLANNER, **self.EXTRACT}
        fields = {name: values.get(name, "x") for name in self.asked_fields(body)}
        system = body["messages"][0]["content"] if body["messages"][0]["role"] == "system" else ""
        if body.get("response_format"):
            text = json.dumps(fields)
        elif "[[ ## " in system:
            text = "".join(f"[[ ## {k} ## ]]\n{v}\n\n" for k, v in fields.items()) + "[[ ## completed ## ]]"
        else:
            text = "\n".join(f"<{k}>\n{v}\n</{k}>" for k, v in fields.items())
        if self.reasoning_only:
            return {"role": "assistant", "content": None, "reasoning_content": text}
        message = {"role": "assistant", "content": text}
        if self.reasoning_content is not None:
            message["reasoning_content"] = self.reasoning_content
        return message

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def fake_openai():
    server = FakeOpenAIServer()
    try:
        yield server
    finally:
        server.close()
