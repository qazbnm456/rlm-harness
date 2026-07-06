"""Single source of truth for RLM runtime configuration.

Everything the scaffold needs to stand up a Recursive Language Model — model
names, credentials, the sandbox interpreter, budget caps, retry policy — lives
here and is driven by environment variables. No other module reads ``os.environ``.

This module intentionally has **no** ``dspy`` import so it stays trivially
importable and unit-testable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

# Interpreters the scaffold knows how to build. "pyodide"/"deno" are the
# sandboxed WASM/subprocess interpreters DSPy ships by default and are safe for
# untrusted content. "mock" is for tests. "local" runs model-written code on the
# host and is gated behind an explicit opt-in (see sandbox.py).
KNOWN_INTERPRETERS = frozenset({"pyodide", "deno", "mock", "local"})

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


@dataclass(frozen=True)
class RLMConfig:
    """Immutable runtime configuration for an RLM task.

    Build one with :meth:`from_env` (the common path) or construct directly in
    tests. Pass it to :func:`rlm_kit.runtime.configure`.
    """

    main_model: str
    sub_model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    # Sandbox / interpreter selection. Defaults to the secure WASM sandbox.
    interpreter: str = "pyodide"
    allow_insecure_sandbox: bool = False

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
    max_tokens: Optional[int] = _DEFAULT_MAX_TOKENS

    # Budget controls — passed best-effort to dspy.RLM.
    max_iterations: int = 10
    max_llm_calls: int = 30

    # Head+tail cap (in CHARACTERS — unrelated to ``max_tokens``) that dspy.RLM applies to each
    # REPL output before it enters the planner prompt; the planner never sees the omitted middle.
    # Default matches dspy's own. Raise it when the planner must read large printed results whole,
    # but prefer slicing/summarising in REPL code — retained chars cost prompt tokens every turn.
    max_output_chars: int = 10_000

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
    def from_env(cls) -> "RLMConfig":
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
        - ``RLM_MAX_RETRIES`` (default ``3``).
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
            adapter=os.getenv("RLM_ADAPTER", "json"),
            max_tokens=int(_mt) if _mt and _mt.strip() else _DEFAULT_MAX_TOKENS,
            allow_insecure_sandbox=_env_bool("RLM_ALLOW_INSECURE_SANDBOX", False),
            max_iterations=_env_int("RLM_MAX_ITERATIONS", 10),
            max_llm_calls=_env_int("RLM_MAX_LLM_CALLS", 30),
            max_output_chars=_env_int("RLM_MAX_OUTPUT_CHARS", 10_000),
            max_retries=_env_int("RLM_MAX_RETRIES", 1),
            observe=_env_bool("RLM_OBSERVE", False),
        )
