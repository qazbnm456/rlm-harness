"""Single source of truth for RLM runtime configuration.

Everything the scaffold needs to stand up a Recursive Language Model — model
names, credentials, the sandbox interpreter, budget caps, retry policy — lives
here and is driven by environment variables. No other module reads ``os.environ``.

This module intentionally has **no** ``dspy`` import so it stays trivially
importable and unit-testable.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

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

# Keys ``configure()`` owns on the ``dspy.LM`` it builds, refused in a per-role passthrough.
#
# The line is NOT "everything the kit sets" -- it is **whether the TRACE can see the override**.
# ``max_tokens`` (and dspy's ``max_completion_tokens`` rewrite of it) is read back OFF THE LM into
# ``run_end.payload.budgets`` by ``_dspy_compat.applied_lm_budget``, so a per-role override is
# self-documenting in the trace: a reader sees the cap the call actually carried, which is the
# whole reason that shim reads the LM instead of ``RLMConfig``. It is therefore ALLOWED, and is
# deliberately how a consumer gets a per-role generation cap without a second config field.
#
# The keys below are recorded NOWHERE. Silently changing WHERE a run went (``base_url``,
# ``api_key``, ``custom_llm_provider``) or what bounded it (``timeout``) would leave a trace that
# reads exactly like a run that went somewhere else -- so this refuses at parse time rather than
# letting either side win silently. A consumer that genuinely needs a per-role ENDPOINT is doing
# a connection-identity change rather than passing a request parameter: build that role's LM
# yourself and inject it with ``configure(main_lm=…)`` / ``configure(sub_lm=…)``.
_LM_KWARGS_REFUSED = {
    "model": "dspy.LM takes it positionally; set main_model / sub_model (RLM_MAIN_MODEL / RLM_SUB_MODEL)",
    "api_key": "set api_key (RLM_API_KEY); a per-role endpoint needs configure(main_lm=…)/configure(sub_lm=…)",
    "base_url": "set base_url (RLM_BASE_URL); a per-role endpoint needs configure(main_lm=…)/configure(sub_lm=…)",
    "custom_llm_provider": "configure() derives it from base_url",
    "timeout": "set request_timeout_s (RLM_REQUEST_TIMEOUT)",
}


def _env_lm_kwargs(name: str) -> dict[str, Any] | None:
    """Parse a per-role ``dspy.LM`` kwargs passthrough from a JSON env var (unset/blank -> ``None``).

    Malformed input raises HERE, at config-parse time, naming the variable -- the same posture
    ``_env_int``/``_env_float`` already take by letting ``int()``/``float()`` raise. The
    alternative is a silent ``{}``, which is the failure this whole knob exists to avoid: a
    passthrough cannot report that the SERVER ignored a key, so the kit must at least be loud
    about the keys it could not even parse.
    """
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{name} is not valid JSON ({exc}). It must be a JSON OBJECT, e.g. "
            f'{name}=\'{{"extra_body": {{"thinking_token_budget": 16384}}}}\' -- mind the quoting '
            f"layer you are setting it through; see the guide's "
            f'"Per-role LM request parameters" section.'
        ) from exc
    if not isinstance(parsed, dict):
        # TypeError rather than ValueError, matching ``__post_init__``'s check for the same
        # mistake made in code: valid JSON of the wrong SHAPE is one rule, enforced the same way
        # whichever door the value came through. Both messages name the variable, because "which
        # one did I get wrong" is the only question a reader has here.
        raise TypeError(
            f"{name} must be a JSON OBJECT mapping dspy.LM kwarg names to values, "
            f"got {type(parsed).__name__}."
        )
    return parsed


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

    # Per-ROLE passthrough of extra ``dspy.LM`` kwargs, merged over what ``configure()`` builds for
    # that role only (``None`` = send nothing, byte-identical to not having this field).
    #
    # It exists because the answer to "bound this model's THINKING" is server-specific and the kit
    # must not pretend otherwise. Measured on one vLLM deployment: ``thinking_token_budget`` works,
    # while ``max_thinking_tokens``, ``thinking_budget``, ``reasoning_budget`` and
    # ``chat_template_kwargs.thinking_budget`` are all silently IGNORED -- and ``reasoning_effort``,
    # the one name litellm maps across providers, moved reasoning the WRONG way on that model
    # (``low`` 2443-2562 and ``medium`` 2778-2837 tokens against 2237 at the default) while breaking
    # the output-format instruction. A kit-owned name for any of this would be a promise that it
    # means the same thing everywhere, which that data falsifies inside ONE model family. So the kit
    # ships the MECHANISM and no vocabulary: what you write is what reaches the server.
    #
    # Two LEVELS, and the difference is where the silent no-ops above come from. A TOP-LEVEL key is
    # a litellm parameter and goes through litellm's own per-provider handling (``reasoning_effort``
    # is one of these). A key inside ``extra_body`` is passed RAW into the request body with no
    # mapping at all -- which is what a server-specific name like ``thinking_token_budget`` needs:
    #
    #     RLMConfig(main_lm_kwargs={"extra_body": {"thinking_token_budget": 16384}})
    #
    # **A passthrough cannot tell you the server ignored your key.** Nothing here can: an ignored
    # key and an honoured one look identical on the wire. The check is after the fact, in the trace
    # -- ``run_end.payload.budgets.thinking`` records what the LM carried and
    # ``run_end.payload.usage`` carries the provider's own ``reasoning_tokens``, so a run that was
    # supposedly capped at 16384 and reports 32768 of reasoning tells you the key did nothing.
    #
    # Keys in ``_LM_KWARGS_REFUSED`` raise in ``__post_init__`` rather than being silently dropped
    # or silently winning; see that table for the line between refused and allowed.
    main_lm_kwargs: dict[str, Any] | None = None
    sub_lm_kwargs: dict[str, Any] | None = None

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
        # Refuse a kit-owned key HERE, so a directly-constructed config is held to the same rule as
        # one built from env (``from_env`` goes through this constructor).
        for field_name, kwargs in (
            ("main_lm_kwargs", self.main_lm_kwargs),
            ("sub_lm_kwargs", self.sub_lm_kwargs),
        ):
            if kwargs is None:
                continue
            if not isinstance(kwargs, dict):
                raise TypeError(
                    f"{field_name} must be a dict of dspy.LM kwargs or None, "
                    f"got {type(kwargs).__name__}"
                )
            refused = sorted(set(kwargs) & set(_LM_KWARGS_REFUSED))
            if refused:
                detail = "; ".join(f"{k!r}: {_LM_KWARGS_REFUSED[k]}" for k in refused)
                raise ValueError(
                    f"{field_name} may not set {refused} -- configure() owns these and nothing "
                    f"records an override, so a silent one would leave a trace that reads like a "
                    f"run that went somewhere else. {detail}"
                )

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
        - ``RLM_MAIN_LM_KWARGS`` / ``RLM_SUB_LM_KWARGS`` (default: unset) — a JSON OBJECT of
          extra ``dspy.LM`` kwargs for that ROLE only, merged over what ``configure()`` builds.
          The kit ships the mechanism and no vocabulary, because the key that bounds a model's
          thinking is server-specific: ``{"extra_body": {"thinking_token_budget": 16384}}`` on
          vLLM. A top-level key is a litellm parameter (mapped per provider); a key under
          ``extra_body`` goes RAW into the request body. Malformed JSON, a non-object, or a
          ``configure()``-owned key (see ``_LM_KWARGS_REFUSED``) raises here rather than being
          silently dropped. See ``RLMConfig.main_lm_kwargs``.
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
            main_lm_kwargs=_env_lm_kwargs("RLM_MAIN_LM_KWARGS"),
            sub_lm_kwargs=_env_lm_kwargs("RLM_SUB_LM_KWARGS"),
            max_retries=_env_int("RLM_MAX_RETRIES", 1),
            observe=_env_bool("RLM_OBSERVE", False),
        )
