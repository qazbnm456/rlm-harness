"""One-time runtime initialization: wire dspy + (optional) observability.

Replaces the original app's scattered ``agent.py`` global setup. Call
:func:`configure` once at process start; tasks then read the shared config and
sub-LM via :func:`get_config` / :func:`get_sub_lm`.
"""

from __future__ import annotations

import logging
from typing import Optional

import dspy

from .config import RLMConfig

logger = logging.getLogger(__name__)


class _Runtime:
    configured: bool = False
    config: Optional[RLMConfig] = None
    main_lm: Optional["dspy.LM"] = None
    sub_lm: Optional["dspy.LM"] = None


_STATE = _Runtime()


class _LenientJSONAdapter(dspy.JSONAdapter):
    """``JSONAdapter`` for schema-guided servers, with stock dspy's ``json_object`` fallback
    REMOVED and a brace-tolerant parse added.

    Two deviations from stock ``JSONAdapter``, both because a schema-guided server
    (vLLM / NVIDIA NIM) rejects bare ``response_format={"type":"json_object"}`` outright
    (*"'json_object' requires a JSON schema"*):

    1. **No `json_object` fallback** (``__call__`` / ``acall``). Stock ``JSONAdapter``, when its
       ``json_schema`` attempt raises for ANY reason — including a transient upstream 502 — falls
       back to bare ``json_object`` and re-calls. On vLLM/NIM that fallback is dead-on-arrival
       (guaranteed 400), it masks the real error, and it wastes the retry on a format the server
       always rejects. We instead always send ``json_schema`` and let a failure propagate, so the
       task-level retry re-tries the format the server actually accepts. (We drive
       ``ChatAdapter``'s call path directly — for a ``JSONAdapter`` instance it raises rather than
       falling back, see dspy ``chat_adapter`` ``isinstance(self, JSONAdapter)``.)
    2. **Brace-tolerant `parse`.** Schema-guided decoding intermittently drops the outer
       ``{`` ``}`` and returns just ``"a": 1, "b": 2``; stock parse then can't recover a dict.
       We retry once with the body wrapped in braces; a genuinely non-JSON response still raises.

    (A future, NIM-specific alternative is to drive ``nvext.guided_json`` directly instead of
    relying on ``response_format`` — deliberately deferred; this keeps the path provider-agnostic.)"""

    def _schema_kwargs(self, signature, lm_kwargs):
        from dspy.adapters.json_adapter import _get_structured_outputs_response_format

        # Always the schema-bearing json_schema form — never bare json_object.
        return {
            **lm_kwargs,
            "response_format": _get_structured_outputs_response_format(
                signature, self.use_native_function_calling
            ),
        }

    def __call__(self, lm, lm_kwargs, signature, demos, inputs):
        return dspy.ChatAdapter.__call__(
            self, lm, self._schema_kwargs(signature, lm_kwargs), signature, demos, inputs
        )

    async def acall(self, lm, lm_kwargs, signature, demos, inputs):
        return await dspy.ChatAdapter.acall(
            self, lm, self._schema_kwargs(signature, lm_kwargs), signature, demos, inputs
        )

    def parse(self, signature, completion):
        from dspy.utils.exceptions import AdapterParseError

        try:
            return super().parse(signature, completion)
        except AdapterParseError:
            body = (completion or "").strip()
            # The wrap targets a brace-LESS object body ('"a": 1, "b": 2'). A completion that
            # already starts with "{" failed for some other reason (incomplete / missing a
            # required field); wrapping it would just double the brace ("{{...}") and make it
            # worse — let the original error stand.
            if body.startswith("{"):
                raise
            return super().parse(signature, "{" + body + "}")

    def _call_postprocess(self, processed_signature, original_signature, outputs, lm, lm_kwargs):
        """Promote ``reasoning_content`` to the answer when a reasoning root left ``content`` empty.

        A REASONING model (qwen3 / deepseek / glm / gpt-oss) served over an OpenAI-compatible API
        sometimes emits the WHOLE structured turn into the ``reasoning_content`` channel and returns
        ``content`` (the dict's ``text``) null. dspy's base ``_call_postprocess`` then sees an empty
        ``text`` and raises *"The LM returned an empty or null response"* — killing the turn before
        it can be parsed, even though the answer is sitting in ``reasoning_content``. We promote it so
        the normal text path runs. This is what lets a reasoning model be the RLM ROOT at all.

        GUARDED on ``not text``: a well-behaved model (answer in ``content``, chain-of-thought in
        ``reasoning_content``) is untouched, so its native thinking stays discarded as before — we
        only reach for ``reasoning_content`` when there is otherwise nothing to parse."""
        patched = [
            ({**o, "text": o["reasoning_content"]}
             if isinstance(o, dict) and not o.get("text") and o.get("reasoning_content") else o)
            for o in outputs
        ]
        return super()._call_postprocess(
            processed_signature, original_signature, patched, lm, lm_kwargs
        )


def configure(config: Optional[RLMConfig] = None) -> RLMConfig:
    """Initialise dspy (and observability if enabled) from ``config``.

    Idempotent-friendly: calling it again reconfigures cleanly. Returns the
    effective config so callers can log/inspect it.
    """
    cfg = config or RLMConfig.from_env()

    # litellm (dspy.LM's backend) defaults to an aiohttp transport whose pooled ClientSession is bound to
    # the current asyncio loop. A driver that runs each task in its own short-lived `asyncio.run` loop
    # (e.g. a server handling per-request runs) then leaves a dangling session when that loop closes, and
    # aiohttp logs a noisy "Unclosed connector". Force litellm onto httpx — no aiohttp session is created,
    # so nothing dangles. Set before the LMs are built (→ before the first LM call); best-effort.
    try:
        import litellm
        litellm.disable_aiohttp_transport = True
    except Exception:  # noqa: BLE001 — litellm not importable (unexpected under dspy); nothing to quiet
        pass

    lm_kwargs = dict(api_key=cfg.api_key, base_url=cfg.base_url)
    if cfg.max_tokens is not None:
        lm_kwargs["max_tokens"] = cfg.max_tokens
    if cfg.base_url:
        # A base_url means a custom OpenAI-compatible endpoint. dspy.LM's backend is litellm,
        # which routes by parsing a provider out of the model string ("provider/model"); a bare
        # id like "qwen/qwen3-next" then reads "qwen" as the provider and fails ("LLM Provider
        # NOT provided"). Pinning custom_llm_provider="openai" routes via the OpenAI wire
        # protocol to base_url and sends the model name verbatim — so the user writes the plain
        # id their endpoint serves (matching the generator's bare-name convention), with no ugly
        # "openai/" prefix. A still-prefixed "openai/..." name keeps working (litellm strips it).
        lm_kwargs["custom_llm_provider"] = "openai"
    # Both LMs are plain dspy.LM. In "json" mode it's _LenientJSONAdapter (not the LM) that
    # forces the json_schema response_format, so the LM needs no special capability flag.
    main_lm = dspy.LM(cfg.main_model, **lm_kwargs)
    sub_lm = dspy.LM(cfg.sub_model, **lm_kwargs)
    # Pass the adapter explicitly (None == dspy's stock default) so a re-configure
    # is clean. The "chat" default never emits response_format — see _build_adapter.
    #
    # `dspy.configure` is OWNER-LOCKED: dspy records the first thread + async task that calls it and
    # raises if a LATER call comes from a different thread/task. That breaks a long-lived driver that
    # runs each task in its own worker thread (e.g. a server handling per-request runs): run #2's
    # configure is on a new thread and crashes. But the global LM config is already set by the first
    # call and is READABLE from every thread, so on a non-owner thread we simply reuse it. Swallow ONLY
    # that ownership error; re-raise anything else.
    try:
        dspy.configure(lm=main_lm, adapter=_build_adapter(cfg.adapter))
    except RuntimeError as exc:
        msg = str(exc)
        if "thread that initially configured it" not in msg and "same async task" not in msg:
            raise
        logger.debug("dspy already configured by another thread/task; reusing the global config")

    if cfg.observe:
        _try_instrument()

    _STATE.configured = True
    _STATE.config = cfg
    _STATE.main_lm = main_lm
    _STATE.sub_lm = sub_lm
    logger.info(
        "rlm-kit configured: main=%s sub=%s interpreter=%s adapter=%s observe=%s",
        cfg.main_model,
        cfg.sub_model,
        cfg.interpreter,
        cfg.adapter,
        cfg.observe,
    )
    return cfg


def _build_adapter(name: str) -> Optional["dspy.Adapter"]:
    """Map the configured adapter name to a dspy adapter instance (or None).

    ``"chat"`` builds a ``ChatAdapter`` with ``use_json_adapter_fallback=False`` — the
    fallback would, on a parse error, silently retry through ``JSONAdapter`` and emit
    ``response_format={"type":"json_object"}``, which endpoints like vLLM reject
    (they require a schema). Turning it off keeps the task portable: ChatAdapter alone
    never sends ``response_format``. ``"json"`` drives schema-guided structured output (the
    ``_LenientJSONAdapter`` below forces the json_schema form);
    ``"default"`` (→ ``None``) leaves dspy's own default in place.
    """
    if name == "json":
        # Brace-tolerant JSONAdapter that forces the json_schema form schema-guided servers
        # (vLLM / NIM) accept — never the bare json_object they reject.
        return _LenientJSONAdapter()
    if name == "chat":
        return dspy.ChatAdapter(use_json_adapter_fallback=False)
    return None


def _try_instrument() -> None:
    """Best-effort OpenInference/DSPy instrumentation + Langfuse client; never fatal."""
    try:
        from openinference.instrumentation.dspy import DSPyInstrumentor

        DSPyInstrumentor().instrument()
        logger.info("DSPy instrumentation enabled.")
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Observability requested but instrumentation failed (%s); "
            "install the 'observe' extra. Continuing without it.",
            exc,
        )

    # Bootstrap the Langfuse client so consumers that observe via @observe /
    # spans don't have to call get_client() themselves. Optional and non-fatal.
    try:
        from langfuse import get_client

        get_client()
        logger.info("Langfuse client initialized.")
    except Exception as exc:  # noqa: BLE001
        logger.debug("Langfuse client not initialized (%s); continuing.", exc)


def _require_configured() -> None:
    if not _STATE.configured:
        raise RuntimeError(
            "rlm-kit is not configured. Call rlm_kit.configure(RLMConfig.from_env()) "
            "once before running a task."
        )


def get_config() -> RLMConfig:
    _require_configured()
    assert _STATE.config is not None
    return _STATE.config


def get_sub_lm() -> "dspy.LM":
    _require_configured()
    assert _STATE.sub_lm is not None
    return _STATE.sub_lm
