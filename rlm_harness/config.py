"""Single source of truth for RLM runtime configuration.

Everything the scaffold needs to stand up a Recursive Language Model — model
names, credentials, the sandbox interpreter, budget caps, retry policy — lives
here and is driven by environment variables. No other module reads ``os.environ``.

This module intentionally has **no** ``dspy`` import so it stays trivially
importable and unit-testable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# Interpreters the scaffold knows how to build. "pyodide"/"deno" are the
# sandboxed WASM/subprocess interpreters DSPy ships by default and are safe for
# untrusted content. "mock" is for tests. "container" runs the REPL inside an
# isolated Docker container so model code can spawn subprocesses natively (a
# STRONGER boundary than the WASM sandbox for that case; see container_interpreter.py).
# "local" runs model-written code on the host and is gated behind an explicit
# opt-in (see sandbox.py).
KNOWN_INTERPRETERS = frozenset({"pyodide", "deno", "mock", "container", "local"})

# How the RLM coaxes structured output fields out of the model.
#   "json"    — DEFAULT. Schema-guided structured output: a brace-tolerant JSONAdapter
#               (runtime._LenientJSONAdapter) forces the ``json_schema`` response_format and
#               absorbs guided output. Works on ANY endpoint that supports structured output
#               — OpenAI-proper AND vLLM/NIM (which reject schema-less json_object but
#               accept json_schema). On a constraint-decoding server the decoder enforces
#               the schema, so even a weak / imperfectly-formatting model emits valid output.
#   "chat"    — dspy.ChatAdapter with the JSONAdapter fallback DISABLED: text field-markers
#               only, never sends ``response_format``. For an endpoint that supports NO
#               structured output at all. The model must follow the markers reliably — a
#               weak model that drops a field has NO recovery (dspy's own ChatAdapter would
#               fall back to bare json_object, which the kit turns off because vLLM rejects
#               it; so we don't get that recovery either). Not as portable as it looks.
#   "default" — impose nothing; leave dspy's stock adapter (ChatAdapter WITH the json
#               fallback) in place. Recovers via json_object on OpenAI-proper endpoints,
#               but that fallback is rejected by vLLM/NIM.
KNOWN_ADAPTERS = frozenset({"chat", "json", "default"})

# Default per-call generation cap. Generous on purpose so a reasoning model's
# chain-of-thought + answer both fit, rather than relying on a server's small default
# cap (which truncates reasoning before the answer → empty content). See max_tokens.
#
# A FLOOR, not a ceiling: a consumer whose turns are long (a reasoning root, or one that
# assembles a large structured result in a single turn) will need more, and the symptom
# is NOT the empty-content one this default exists to prevent — see ``max_tokens``.
_DEFAULT_MAX_TOKENS = 8192

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUTHY


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _env_optional_float(name: str) -> float | None:
    """Like ``_env_float``, but for a knob whose "unset" state is genuinely ``None``
    rather than a fallback numeric default (unset/blank -> ``None``; malformed -> lets
    ``float(raw)`` raise, exactly as ``_env_float``/``_env_int`` already do for a
    malformed value — no new failure mode). Neither existing "optional env var" shape
    in this module transfers cleanly: ``max_tokens`` is hand-rolled with a non-``None``
    default, and ``ContainerConfig.cpus`` is the only other ``Optional``-typed
    env-sourced field, but it is a string, not a number."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    return float(raw)


@dataclass(frozen=True)
class ContainerConfig:
    """Options for the ``container`` interpreter (see ``container_interpreter.py``).

    Safety-by-default: ``--network=none`` (no egress) + ``--memory`` + ``--pids-limit`` +
    ``--cap-drop=ALL`` cap model-written code, and ``timeout_s`` bounds sandbox compute per
    ``execute`` cell. ``cpus`` is unset (uncapped) by default so a CPU cap can't throttle a
    build into the wall-clock timeout. ``read_only`` (opt-in) makes the rootfs read-only for an
    inspect-only task (paired with a tmpfs ``/tmp`` the agent needs); ``workdir`` mounts a host
    dir READ-ONLY at ``/workspace``. The default is capable: a disposable, no-egress container
    the model can freely write inside (nothing persists to the host).
    """

    image: str = "python:3.11-slim"
    network: str = "none"          # no egress: the stdio broker is the only channel in/out
    memory: str = "512m"           # docker --memory (blast-radius cap on a memory balloon)
    pids_limit: int = 256          # docker --pids-limit (cap on a fork bomb)
    timeout_s: float = 120.0       # per-execute SANDBOX-COMPUTE budget (host tool time is not counted)
    cpus: str | None = None     # docker --cpus (unset = uncapped; capping can throttle a build)
    cap_drop: bool = True          # docker --cap-drop=ALL (drop Linux caps; model code rarely needs any)
    read_only: bool = False        # docker --read-only rootfs (+ a tmpfs /tmp); opt-in inspect mode
    workdir: str | None = None  # host dir mounted READ-ONLY at /workspace (absolute; must exist)

    @classmethod
    def from_env(cls) -> ContainerConfig:
        raw_workdir = os.getenv("RLM_CONTAINER_WORKDIR")
        return cls(
            image=os.getenv("RLM_CONTAINER_IMAGE", "python:3.11-slim"),
            network=os.getenv("RLM_CONTAINER_NETWORK", "none"),
            memory=os.getenv("RLM_CONTAINER_MEMORY", "512m"),
            pids_limit=_env_int("RLM_CONTAINER_PIDS_LIMIT", 256),
            timeout_s=_env_float("RLM_CONTAINER_TIMEOUT", 120.0),
            cpus=(os.getenv("RLM_CONTAINER_CPUS") or "").strip() or None,
            cap_drop=_env_bool("RLM_CONTAINER_CAP_DROP", True),
            read_only=_env_bool("RLM_CONTAINER_READ_ONLY", False),
            # Normalise to an absolute path so a bare relative name isn't silently read by docker as
            # an (empty) named volume; a missing dir is caught at start() with a clear error.
            workdir=os.path.abspath(os.path.expanduser(raw_workdir)) if raw_workdir else None,
        )


@dataclass(frozen=True)
class RLMConfig:
    """Immutable runtime configuration for an RLM task.

    Build one with :meth:`from_env` (the common path) or construct directly in
    tests. Pass it to :func:`rlm_harness.runtime.configure`.
    """

    main_model: str
    sub_model: str
    api_key: str | None = None
    base_url: str | None = None

    # Sandbox / interpreter selection. Defaults to the secure WASM sandbox.
    interpreter: str = "pyodide"
    allow_insecure_sandbox: bool = False

    # Options for the ``container`` interpreter; ignored by every other interpreter.
    container: ContainerConfig = field(default_factory=ContainerConfig)

    # Structured-output adapter (see KNOWN_ADAPTERS). Defaults to "json" — schema-guided
    # structured output works on any endpoint that supports it (OpenAI-proper AND vLLM/NIM,
    # which accept json_schema) and is robust even when the model formats imperfectly, since
    # the decoder enforces the schema. Switch to "chat" only for an endpoint with no
    # structured-output support at all (then the model must follow the text field-markers).
    adapter: str = "json"

    # Per-call generation cap for the main/sub LM (passed to ``dspy.LM(max_tokens=...)``).
    # Defaults to a generous value rather than ``None`` on purpose: with ``None`` the kit sends
    # no max_tokens and the SERVER applies its own default cap (e.g. 1000 on some vLLM/NIM
    # setups). A reasoning model emits its chain-of-thought (``reasoning_content``) BEFORE the
    # answer (``content``), so a turn whose reasoning exceeds that small cap is truncated
    # mid-thought and ``content`` comes back EMPTY → "empty or null response". Sending a generous
    # cap leaves room for reasoning + answer on any endpoint. Set ``None`` to defer to the server.
    #
    # Raising it is NOT free on every endpoint: an OpenAI-compatible server commonly validates
    # ``prompt_tokens + max_tokens`` against the context window, so a bigger cap removes usable
    # PROMPT budget — and an RLM planner's prompt grows every turn, which is exactly where that
    # bites. Weigh it against the failure below rather than raising it reflexively.
    #
    # **The default is a floor, and overrunning it presents as a DIFFERENT failure.** 8192 must
    # hold one turn's chain-of-thought AND its structured answer. A long turn that overruns it is
    # cut off mid-JSON, so the adapter cannot parse the reply and the run surfaces as
    # ``RLMTaskError: Failed to produce a valid '<field>'`` caused by ``AdapterParseError`` — a
    # truncation, not a model that cannot follow the schema, and repeatedly misdiagnosed as the
    # latter because the quoted response looks well-formed right up to where it stops. Reading the
    # END of that quoted text tells them apart: a truncated one has no closing brace. A reasoning
    # root and a turn that assembles a large structured result compound the risk; more than one
    # consumer has settled on 16384. Raising it costs nothing on turns that do not need the room,
    # since this bounds generation rather than reserving it.
    max_tokens: int | None = _DEFAULT_MAX_TOKENS

    # Budget controls — passed best-effort to dspy.RLM.
    max_iterations: int = 10
    max_llm_calls: int = 30

    # Head+tail cap (in CHARACTERS — unrelated to ``max_tokens``) that dspy.RLM applies to each
    # REPL output before it enters the planner prompt; the planner never sees the omitted middle.
    # Default matches dspy's own. Raise it when the planner must read large printed results whole,
    # but prefer slicing/summarising in REPL code — retained chars cost prompt tokens every turn.
    max_output_chars: int = 10_000

    # A per-`execute()` SANDBOX-COMPUTE safety-net timeout for the pyodide/deno interpreter,
    # mirroring ContainerConfig.timeout_s's own precedent for the container interpreter — but
    # `None` (disabled) by default, deliberately NOT matching that precedent's `120.0`. Two
    # independent reasons: (1) this kit has real, already-shipped downstream consumers whose
    # existing long-running-but-legitimate turns must not start failing the moment this exists;
    # (2) unlike ContainerConfig.timeout_s, this budget has no hook to exclude host-side
    # tool/sub-LM dispatch time (dspy's PythonInterpreter.execute() is opaque here), so it is
    # measurably MORE likely to misfire on a legitimate multi-tool-call turn than the container
    # analogy implies — a "generous" always-on default would be the WRONG default, not merely an
    # unnecessary one. See `sandbox.py`'s `_build_sandboxed_interpreter` for the mechanism.
    sandbox_turn_timeout_s: float | None = None

    # Wall-clock cap on ONE model HTTP request ATTEMPT (passed to ``dspy.LM(timeout=...)``, which
    # hands it to litellm). ``None`` (the default) sends nothing — which is NOT the same as no cap:
    # litellm then applies its own ``COMPLETION_HTTP_FALLBACK_SECONDS`` of 600.0
    # (``litellm_core_utils/completion_timeout.py``; verified by execution, not read off the docs).
    # So the real default is 600s per attempt, and this field REPLACES that number rather than
    # introducing a bound where there was none.
    #
    # **It does not bound a run to its own value, because an attempt is not a request.** dspy
    # passes ``num_retries=3`` and litellm's first call hands the OpenAI SDK ``max_retries=2``, so
    # a dead endpoint is retried — the run-level wait is a MULTIPLE of this, plus backoff. Size a
    # caller-side budget on the multiple, not on this number.
    #
    # Why it exists at all: ``sandbox_turn_timeout_s`` bounds the sandbox side of a turn and
    # nothing here bounded the model side, so a consumer could not choose the number. Observed on
    # a real deployment against a self-hosted OpenAI-compatible endpoint: one request never came
    # back, the socket stayed ESTABLISHED with both queues empty, and the worker slept in
    # ``epoll_wait`` for 38 minutes at 0.3% CPU while that same endpoint answered unrelated
    # requests in half a second. Note 600s x 4 attempts is about 40 minutes, so that observation
    # is consistent with the litellm default being retried rather than with nothing being
    # watching — an honest reading of it, since no attempt counter was captured at the time.
    #
    # Left at ``None`` so the default stays exactly what it was before this field existed. A
    # legitimately long turn does exist (a reasoning model assembling a large structured answer),
    # and a consumer whose turns exceed 600s must set this UP, not merely leave it alone.
    request_timeout_s: float | None = None

    # Retry policy in _retry.py: how many times to run the WHOLE task (a full RLM trajectory) until
    # its output coerces into output_model. Default 1 = no retry, because a retry re-runs the entire
    # RLM from scratch — silently MULTIPLYING the max_iterations budget (3 retries ⇒ up to 3×
    # max_iterations turns) and re-doing every fetch/search/tool call. That budget multiplication
    # breaks the contract a consumer (and its UI) builds on, and a re-run rarely fixes a PERSISTENT
    # coercion failure (same model + schema → same bad output). Raise this only when transient infra
    # flakiness genuinely warrants a whole-run retry, knowing the budget cost.
    max_retries: int = 1

    # Observability (Langfuse + OpenInference) is opt-in.
    observe: bool = False

    def __post_init__(self) -> None:
        if self.interpreter not in KNOWN_INTERPRETERS:
            raise ValueError(
                f"Unknown interpreter {self.interpreter!r}; "
                f"expected one of {sorted(KNOWN_INTERPRETERS)}"
            )
        if self.adapter not in KNOWN_ADAPTERS:
            raise ValueError(
                f"Unknown adapter {self.adapter!r}; "
                f"expected one of {sorted(KNOWN_ADAPTERS)}"
            )
        if self.max_retries < 1:
            raise ValueError("max_retries must be >= 1")

    @classmethod
    def from_env(cls) -> RLMConfig:
        """Build configuration from environment variables.

        Recognised variables (all optional except where a sane default is shown):

        - ``RLM_MAIN_MODEL`` / ``AI_MODEL_NAME`` (default ``openai/gpt-4o``) — the
          REPL/root model. An INSTRUCT or a REASONING model both work: ``_LenientJSONAdapter``
          promotes ``reasoning_content`` to the answer when a reasoning root leaves ``content``
          empty (some emit the whole structured turn into the thinking channel). Caveats for a
          reasoning root: its native chain-of-thought is still DISCARDED (dspy reads only the
          structured turn), so it spends tokens the trace won't keep, and a too-small ``max_tokens``
          can truncate it mid-thought (→ empty content) — keep the cap generous (see ``max_tokens``).
          The second var is a fallback so this scaffold drops into projects that already use
          ``AI_MODEL_NAME`` without re-keying env.
        - ``RLM_SUB_MODEL`` / ``SUB_AI_MODEL_NAME`` (default: same as main) —
          model for recursive subcalls.
        - ``RLM_API_KEY`` / ``AI_API_KEY`` — API key (the second is a fallback so
          this scaffold can drop into projects that already use ``AI_API_KEY``).
        - ``RLM_BASE_URL`` / ``AI_BASE_URL`` — optional custom OpenAI-compatible endpoint.
          When set, ``configure`` pins ``custom_llm_provider="openai"`` so the model names
          above can be the PLAIN id the endpoint serves (e.g. ``qwen/qwen3-next``) — no
          ``openai/`` (or other litellm provider) prefix needed; a prefixed name still works.
          With no base_url, write the model's own provider prefix (``openai/gpt-4o``,
          ``anthropic/claude-...``) as litellm expects.
        - ``RLM_INTERPRETER`` (default ``pyodide``).
        - ``RLM_ADAPTER`` (default ``json``) — ``chat`` | ``json`` | ``default``;
          see ``KNOWN_ADAPTERS``. ``json`` (schema-guided) works on any endpoint that
          supports structured output; ``chat`` is for endpoints that support none.
        - ``RLM_MAX_TOKENS`` (default ``8192``) — per-call generation cap for the LM;
          generous by default so a reasoning model's chain-of-thought + answer both fit
          instead of hitting a server's small default cap (which truncates → empty content).
        - ``RLM_ALLOW_INSECURE_SANDBOX`` (default ``false``).
        - ``RLM_MAX_ITERATIONS`` (default ``10``).
        - ``RLM_MAX_LLM_CALLS`` (default ``30``).
        - ``RLM_MAX_OUTPUT_CHARS`` (default ``10000``) — head+tail character cap on REPL
          output fed back to the planner (distinct from ``RLM_MAX_TOKENS``).
        - ``RLM_REQUEST_TIMEOUT`` (default: unset, which is NOT no cap — litellm then applies
          its own 600s) — wall-clock seconds for ONE
          model HTTP request. Its sibling on the model side of a turn; see
          ``RLMConfig.request_timeout_s`` for the hang it exists to bound and why it has no
          default.
        - ``RLM_SANDBOX_TURN_TIMEOUT`` (default: unset, i.e. disabled) — a per-``execute()``
          sandbox-compute safety-net timeout in seconds for the pyodide/deno interpreter. See
          ``RLMConfig.sandbox_turn_timeout_s`` for why this defaults to disabled rather than a
          generous always-on value.
        - ``RLM_MAX_RETRIES`` (default ``1``).
        - ``RLM_OBSERVE`` (default ``false``).
        """
        main_model = (
            os.getenv("RLM_MAIN_MODEL")
            or os.getenv("AI_MODEL_NAME")
            or "openai/gpt-4o"
        )
        sub_model = (
            os.getenv("RLM_SUB_MODEL")
            or os.getenv("SUB_AI_MODEL_NAME")
            or main_model
        )
        _mt = os.getenv("RLM_MAX_TOKENS")
        return cls(
            main_model=main_model,
            sub_model=sub_model,
            api_key=os.getenv("RLM_API_KEY") or os.getenv("AI_API_KEY"),
            base_url=os.getenv("RLM_BASE_URL") or os.getenv("AI_BASE_URL"),
            interpreter=os.getenv("RLM_INTERPRETER", "pyodide"),
            container=ContainerConfig.from_env(),
            adapter=os.getenv("RLM_ADAPTER", "json"),
            max_tokens=int(_mt) if _mt and _mt.strip() else _DEFAULT_MAX_TOKENS,
            allow_insecure_sandbox=_env_bool("RLM_ALLOW_INSECURE_SANDBOX", False),
            max_iterations=_env_int("RLM_MAX_ITERATIONS", 10),
            max_llm_calls=_env_int("RLM_MAX_LLM_CALLS", 30),
            max_output_chars=_env_int("RLM_MAX_OUTPUT_CHARS", 10_000),
            sandbox_turn_timeout_s=_env_optional_float("RLM_SANDBOX_TURN_TIMEOUT"),
            request_timeout_s=_env_optional_float("RLM_REQUEST_TIMEOUT"),
            max_retries=_env_int("RLM_MAX_RETRIES", 1),
            observe=_env_bool("RLM_OBSERVE", False),
        )
