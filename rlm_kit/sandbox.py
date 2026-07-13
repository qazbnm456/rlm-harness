"""Sandbox / code-interpreter selection — the security boundary of the scaffold.

RLM works by letting the model write and execute Python in a REPL. When that
REPL is fed half-trusted scraped content (a common case the moment a task pulls
from the web or any untrusted source), the interpreter choice *is* the attack surface.

Policy:

- ``pyodide`` / ``deno`` → dspy.RLM's default sandboxed (WASM / subprocess)
  interpreter, but constructed *here* as a thin subclass that pre-binds the JSON
  literals ``true``/``false``/``null`` in the REPL namespace (see
  ``_build_sandboxed_interpreter``). Same isolation as dspy's own default. Safe
  for untrusted content.
- ``mock`` → a no-op interpreter for tests.
- ``container`` → the environment interpreter (``container_interpreter.py``,
  opt-in): the REPL runs *inside* an isolated Docker container so model code can
  spawn subprocesses natively. A STRONGER boundary than the WASM sandbox
  (``--network=none``, LM creds host-side, caps dropped) and the OPPOSITE of
  ``local`` — handled *before* the insecure-interpreter check below, never routed
  through it. Needs the ``docker`` CLI (imported lazily; this module stays dspy-free).
- ``local`` → executes model-written code directly on the host. This is
  effectively arbitrary code execution and is refused unless the caller has
  *explicitly* opted in. The opt-in cannot be reached by accident.

``dspy`` is imported lazily inside the branches that need it so this module —
and the security guard in particular — stays importable and testable without a
full dspy install.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Interpreters that run model-written code on the host with no isolation.
INSECURE_INTERPRETERS = frozenset({"local"})


class SandboxSecurityError(RuntimeError):
    """Raised when an insecure interpreter is requested without explicit opt-in."""


def build_interpreter(
    kind: str, *, allow_insecure: bool = False, container: Any = None
) -> Optional[Any]:
    """Return a dspy ``CodeInterpreter`` for ``kind``, or ``None`` for the default.

    ``None`` means "let dspy.RLM construct its own default sandboxed interpreter",
    which is the secure path. A non-None return is produced for ``mock``, ``container``
    (needs the ``container`` options + the ``docker`` CLI), and (when explicitly allowed)
    ``local``.
    """
    normalized = (kind or "pyodide").lower()

    if normalized in ("pyodide", "deno"):
        # dspy.RLM's default PythonInterpreter is the sandboxed WASM/subprocess
        # engine. We construct it ourselves (instead of returning None and letting
        # dspy build its own) only so we can pre-bind the JSON-literal aliases — the
        # isolation is identical. dspy.RLM merges its execution tools (SUBMIT,
        # llm_query, …) onto our instance, and ``RLMTask`` owns its teardown.
        return _build_sandboxed_interpreter()

    if normalized == "mock":
        return _build_mock_interpreter()

    if normalized == "container":
        # The environment interpreter: the REPL runs inside an isolated container so model
        # code can spawn subprocesses natively. NOT routed through INSECURE_INTERPRETERS — it
        # is a STRONGER boundary than the WASM sandbox, the opposite of `local`. Lazily imported
        # (it is dspy-bearing) so this module and ``import rlm_kit`` stay dspy-free.
        return _build_container_interpreter(container)

    if normalized in INSECURE_INTERPRETERS:
        if not allow_insecure:
            raise SandboxSecurityError(
                "Interpreter 'local' executes model-written code directly on the "
                "host with no isolation. For a system that processes untrusted "
                "content this is remote code execution. Opt in explicitly via "
                "RLMConfig(allow_insecure_sandbox=True) or "
                "RLM_ALLOW_INSECURE_SANDBOX=1 if you understand the risk."
            )
        logger.warning(
            "INSECURE SANDBOX ACTIVE: 'local' interpreter runs model-written code "
            "on the host. Never enable this while processing untrusted input."
        )
        return _build_local_interpreter()

    # config.RLMConfig validates this earlier, but guard here too for direct callers.
    raise ValueError(f"Unknown interpreter kind: {kind!r}")


# JSON literals a model trained on JSON habitually emits inside the Python REPL —
# e.g. ``SUBMIT({"valid": true})`` — which raise ``NameError: name 'true' is not
# defined`` and make the model thrash on the identical call (a single run lost
# 14/25 REPL turns to exactly this). Pre-binding the three to their Python values
# makes the REPL tolerant of that one most-common JSON-in-Python slip.
_JSON_LITERAL_ALIASES = {"true": True, "false": False, "null": None}

# Built once, lazily — the class can only be defined after dspy is importable, and
# this module deliberately stays dspy-free at import time (see module docstring).
_sandboxed_interpreter_cls: Optional[type] = None


def _build_sandboxed_interpreter() -> Any:
    """dspy's default deno/pyodide sandbox, wrapped to pre-bind ``true``/``false``/
    ``null`` in the REPL namespace.

    Construction spawns no subprocess (dspy's ``PythonInterpreter`` starts Deno
    lazily on first ``execute``), so this is cheap and an interpreter that is never
    run never starts Deno. The aliases are injected as REPL *variables*, which dspy
    serialises to ``true = True`` / ``false = False`` / ``null = None`` atop every
    executed cell; a real user variable of the same name still shadows them.
    """
    global _sandboxed_interpreter_cls
    if _sandboxed_interpreter_cls is None:
        from dspy.primitives.python_interpreter import PythonInterpreter

        class _JsonLiteralInterpreter(PythonInterpreter):
            _JSON_ALIASES = _JSON_LITERAL_ALIASES

            def execute(self, code: str, variables: Optional[dict] = None) -> Any:
                # Caller variables win on a name clash (never expected for these).
                return super().execute(
                    code, {**self._JSON_ALIASES, **(variables or {})}
                )

        _sandboxed_interpreter_cls = _JsonLiteralInterpreter

    return _sandboxed_interpreter_cls()


def _build_container_interpreter(container: Any) -> Any:
    """Construct the container-backed environment interpreter. Lazily imported because
    ``container_interpreter`` is dspy-bearing; ``sandbox.py`` itself stays dspy-free."""
    from .config import ContainerConfig
    from .container_interpreter import ContainerInterpreter

    return ContainerInterpreter(container or ContainerConfig())


def _build_mock_interpreter() -> Any:
    """A do-nothing interpreter usable in tests without a real sandbox.

    Implements the minimal ``execute``/``shutdown`` surface; dspy is not required.
    """

    class _MockInterpreter:  # minimal CodeInterpreter surface
        def execute(self, code: str, variables: Optional[dict] = None) -> str:
            return ""

        def shutdown(self) -> None:
            return None

    return _MockInterpreter()


def _build_local_interpreter() -> Any:  # pragma: no cover - never run in tests/CI
    """Construct dspy's local Python interpreter. Insecure by definition."""
    from dspy.primitives.python_interpreter import PythonInterpreter

    return PythonInterpreter()
