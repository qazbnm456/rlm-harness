"""One-time runtime initialization: wire dspy + (optional) observability.

Replaces the original app's scattered ``agent.py`` global setup. Call
:func:`configure` once at process start; tasks then read the shared config and
sub-LM via :func:`get_config` / :func:`get_sub_lm`.
"""

from __future__ import annotations

import contextlib
import contextvars
import importlib
import logging
from typing import Any

import dspy

from . import _dspy_compat
from .community import Route, route_for
from .config import RLMConfig

logger = logging.getLogger(__name__)


class _Runtime:
    configured: bool = False
    config: RLMConfig | None = None
    main_lm: dspy.LM | None = None
    sub_lm: dspy.LM | None = None


_STATE = _Runtime()

#: Where ``_LenientJSONAdapter.parse`` reports that its brace-wrap retry is what parsed a reply. The
#: per-call capture in ``task.py`` sets a fresh list for each predictor call and reads it when the
#: call ends, so the trace can mark that step ``parse_wrapped``. A list rather than a flag, because a
#: context copied into another thread still shares the list object. ``None`` outside a captured
#: call, where nothing is recorded.
_PARSE_WRAPPED: contextvars.ContextVar[list[bool] | None] = contextvars.ContextVar(
    "rlm_harness_parse_wrapped", default=None
)


#: The texts of this call's replies that came from the reasoning channel because the reply had no
#: text. ``_LenientJSONAdapter._call_postprocess`` sets it around dspy's parse of one call's replies
#: and ``parse``, which is handed the text and nothing else, reads it to say so when such a text does
#: not parse. A context variable rather than adapter state, because one adapter instance serves
#: concurrent calls (``configure`` builds it once and puts it in ``dspy.settings``).
_PROMOTED_REASONING: contextvars.ContextVar[frozenset[str]] = contextvars.ContextVar(
    "rlm_harness_promoted_reasoning", default=frozenset()
)


def _promote_reasoning(response: Any) -> tuple[Any, str | None]:
    """``response`` with its thinking added as its text when it carried no text and no tool call.

    A reasoning model served over an OpenAI-compatible API sometimes puts the whole structured turn in
    its reasoning channel and returns no content, and dspy then stops the turn with "The LM returned an
    empty or null response" although the answer is there. Returns the rebuilt response and the
    promoted text, or ``response`` unchanged and ``None``. The thinking part stays, so a reader of the
    response still sees where the text came from.
    """
    import dataclasses

    from dspy.lm15 import TextPart, ThinkingPart

    message = getattr(response, "message", None)
    parts = tuple(getattr(message, "parts", None) or ())
    if not parts or getattr(response, "tool_calls", None):
        return response, None
    if any(isinstance(part, TextPart) and part.text for part in parts):
        return response, None
    thinking = "".join(part.text for part in parts if isinstance(part, ThinkingPart) and part.text)
    if not thinking:
        return response, None
    rebuilt = dataclasses.replace(message, parts=(*parts, TextPart(thinking)))
    return dataclasses.replace(response, message=rebuilt), thinking


class _LenientJSONAdapter(dspy.JSONAdapter):
    """``JSONAdapter`` for schema-guided servers: always a ``json_schema`` response format, a
    brace-tolerant parse, and a reply from the reasoning channel promoted to its text.

    1. **Always ``json_schema``** (``_prepare_response_format``). Stock ``JSONAdapter`` sends bare
       ``json_object``, or nothing, when dspy's catalog does not list structured-output support for
       the model, which is the usual case for a self-hosted model id. A schema-guided server (vLLM /
       NVIDIA NIM) rejects ``json_object`` outright (*"'json_object' requires a JSON schema"*), and a
       server given no format has nothing to guide its decoder. So the kit always sends the schema of
       the signature the prompt is rendered from: for the planner, ``plan`` and ``code``. dspy's
       ``ChatAdapter`` never falls back from a ``JSONAdapter``, so a failed call propagates and the
       task-level retry re-sends the format the server accepts.
    2. **Brace-tolerant ``parse``.** Schema-guided decoding intermittently drops the outer
       ``{`` ``}`` and returns just ``"a": 1, "b": 2``; stock parse then can't recover a dict.
       We retry once with the body wrapped in braces, unless it is empty, whitespace-only, or
       already starts with a brace; a genuinely non-JSON response still raises. A reply the retry
       parsed is reported through ``_PARSE_WRAPPED``, and the trace marks its call
       ``parse_wrapped: true``.
    3. **Reasoning-channel promotion** (``_call_postprocess``), so a reasoning model can be the RLM
       root; see :func:`_promote_reasoning`.
    """

    def _prepare_response_format(self, lm, lm_kwargs, signature):
        lm_kwargs["response_format"] = _dspy_compat.structured_output_format(
            signature, self.use_native_function_calling
        )

    def parse(self, signature, completion):
        from dspy.utils.exceptions import AdapterParseError

        try:
            return super().parse(signature, completion)
        except AdapterParseError as original:
            body = (completion or "").strip()
            # The wrap targets a brace-LESS object body ('"a": 1, "b": 2'). A completion that
            # already starts with "{" failed for some other reason (incomplete / missing a
            # required field); wrapping it would just double the brace ("{{...}") and make it
            # worse: let the original error stand.
            #
            # An EMPTY or whitespace-only completion is excluded for a different reason: wrapping
            # nothing yields "{}", and dspy's `apply_output_field_defaults` supplies a value for an
            # output field left unset, from a default, a default factory, or an annotation that
            # admits None, in that order. So the wrap would ANSWER a reply that said nothing.
            # Wrapping an empty body is never a recovery, only an invention.
            if not body or body.startswith("{"):
                raise
            try:
                parsed = super().parse(signature, "{" + body + "}")
            except AdapterParseError:
                # The RETRY failed too, and its error quotes the text WE built, braces included.
                # A consumer debugging a rare failure read `{We need answer with JSON...` as a
                # model emitting broken JSON; the brace was this method's. So the error that leaves
                # here carries the reply as the model wrote it, and says what the kit tried.
                #
                # This text can reach the MODEL as well as a reader: with dspy's own handler for an
                # unparseable turn (stanfordnlp/dspy#10676) the turn's error goes into the REPL
                # history and the planner sees it. So it names what to do differently and carries no
                # number, per `docs/PROMPT.md`.
                if completion in _PROMOTED_REASONING.get():
                    note = (
                        "This reply carried no content. The text below is its reasoning channel, "
                        "promoted so it could be parsed, and it did not parse as the output fields "
                        "either. Put the answer in the reply itself, as a JSON object with the "
                        "fields named below."
                    )
                else:
                    note = (
                        "This reply did not parse as the output fields, and retrying it as a JSON "
                        "object did not either. The text below is the reply as written, with "
                        "nothing added. Reply with a JSON object holding the fields named below."
                    )
                raise AdapterParseError(
                    adapter_name=original.adapter_name,
                    signature=original.signature,
                    lm_response=original.lm_response,
                    message=note,
                ) from None
            sink = _PARSE_WRAPPED.get()
            if sink is not None:
                sink.append(True)
            return parsed

    def _call_postprocess(self, processed_signature, original_signature, responses):
        """Promote a reasoning-channel-only reply to its text, then parse as dspy does.

        Guarded on a reply with no text and no tool call: a well-behaved model (answer in the text,
        chain-of-thought in the reasoning channel) is untouched. A promoted text is listed in
        ``_PROMOTED_REASONING`` for the duration of dspy's parse, so ``parse`` can say so when that
        free-form thinking does not parse: a different event from a malformed answer.
        """
        promoted: set[str] = set()
        rebuilt = []
        for response in responses:
            response, text = _promote_reasoning(response)
            rebuilt.append(response)
            if text is not None:
                promoted.add(text)
        token = _PROMOTED_REASONING.set(frozenset(promoted))
        try:
            return super()._call_postprocess(processed_signature, original_signature, rebuilt)
        finally:
            _PROMOTED_REASONING.reset(token)


def _maybe_subscription_lm(model: str) -> tuple[dspy.LM, Route] | None:
    """The LM for ``model`` and the ``community.ROUTES`` entry that built it, when ``model`` starts
    with a route's prefix, else ``None`` (leaving the caller to build a plain ``dspy.LM`` from
    ``config`` exactly as before). The route's adapter module is imported only when its prefix
    matches, so a call that never names a community prefix imports none of them.

    Raises ``ValueError`` (not ``SystemExit``) for a bare prefix with no model id: `configure()` is
    called from arbitrary contexts: a server's request handler, a notebook cell, a library
    embedding rlm-harness, and `SystemExit` propagating out of a shared library call is a far more
    dangerous default there than in a purpose-built CLI, which can catch `ValueError` and convert
    it to its own fail-fast behavior at its own top level if it wants that.
    """
    route = route_for(model)
    if route is None:
        return None
    name = model[len(route.prefix) :].strip()
    if not name:
        raise ValueError(
            f"{model!r} names no model: expected {route.prefix}<id>, e.g. {route.prefix}{route.example}."
        )
    adapter = getattr(importlib.import_module(route.module), route.attr)
    return adapter(name), route


def _warn_if_thinking_budget_cannot_act(role: str, lm: Any) -> None:
    """Warn when a THINKING ceiling is >= the generation cap, so it can never bind first.

    Read off the BUILT LM rather than off `RLMConfig`, so this also covers an injected LM that
    carries both itself, and so it sees dspy's own `max_tokens` -> `max_completion_tokens`
    rewrite for OpenAI reasoning models.

    A warning, never an error: the combination is INERT rather than harmful, output generation
    stops at the smaller number either way, and the kit has no way to know a server's semantics
    well enough to refuse. Best-effort on both halves: an unrecognised budget key reads as absent
    (see `_dspy_compat.applied_thinking_budget`) and simply produces no warning.
    """
    with contextlib.suppress(Exception):
        thinking = _dspy_compat.applied_thinking_budget(lm)
        if thinking is None:
            return
        cap = _dspy_compat.applied_lm_budget(lm)
        if cap is None or thinking["value"] < cap["cap"]:
            return
        logger.warning(
            "%s LM: thinking budget %s (%s) is >= its generation cap %s (%s), so the output cap "
            "binds first and the thinking budget can never act. Lower the thinking budget, or "
            "raise the generation cap above it.",
            role, thinking["value"], thinking["key"], cap["cap"], cap["key"],
        )


def _model_string(cfg: RLMConfig, model: str) -> str:
    """The ``dspy.LM`` model string for a configured model: rewritten for a ``base_url``, else as given."""
    return _dspy_compat.endpoint_model(cfg.endpoint_dialect, model) if cfg.base_url else model


def _check_lm_route(role: str, lm: Any) -> None:
    """Refuse an LM ``configure()`` built whose calls would not carry what its config asks for.

    Two ways a role's settings can fail to reach the model: the model string or an option needs
    dspy's litellm engine and litellm is not installed, or dspy's native engine refuses an option or
    leaves it out of the request (it records an ``Adaptation`` and sends the rest). Both raise here,
    before any call, naming what carries the setting.
    """
    route = _dspy_compat.lm_route(lm)
    if route.engine == "litellm" and not _dspy_compat.litellm_installed():
        raise ImportError(
            f"The {role} LM {lm.model!r} needs dspy's litellm engine, and litellm is not installed. "
            f"Install it with: pip install 'rlm-harness[litellm]'"
        )
    if route.refused:
        problem = f"dspy's native engine refuses its request ({route.refused})"
    elif route.dropped:
        problem = (f"dspy's native engine would leave {list(route.dropped)} out of its request, "
                   f"since this route has no field for them")
    else:
        return
    # Measured on dspy 3.5.0: no server preset carries an option the plain OpenAI wire refuses or
    # drops (`top_k`, `extra_body`, `chat_template_kwargs`, a thinking-token budget, `min_p`,
    # `repetition_penalty`), so the litellm engine is the one route that sends them as written.
    raise ValueError(
        f"The {role} LM {lm.model!r}: {problem}. To send it as written, install "
        f"rlm-harness[litellm] and add \"engine\": \"litellm\" to {role}_lm_kwargs."
    )


def configure(
    config: RLMConfig | None = None,
    *,
    main_lm: dspy.LM | None = None,
    sub_lm: dspy.LM | None = None,
) -> RLMConfig:
    """Initialise dspy (and observability if enabled) from ``config``.

    Idempotent-friendly: calling it again reconfigures cleanly. Returns the
    effective config so callers can log/inspect it.

    Pass ``main_lm`` / ``sub_lm`` to use a PRE-BUILT LM verbatim instead of constructing
    one from ``config``: a ``dspy.utils.DummyLM`` in tests, or a cached / custom client in
    production. Whichever you inject is stored and (for the sub-LM) handed back by
    ``get_sub_lm``; an injected ``main_lm`` also becomes the dspy global. This is the public
    seam for supplying a test double, so nothing needs to reach into the private runtime state.

    **Community auto-routing.** For a role whose caller-supplied override is ``None``, if
    ``cfg.main_model``/``cfg.sub_model`` starts with a prefix in ``community.ROUTES`` (such as
    ``"claude-agent-sdk/"``, the sentinel ``ClaudeAgentLM`` stamps its own model string with),
    that role is built by that route's class instead of a plain ``dspy.LM``, with ONLY the
    stripped model id: never ``lm_kwargs``
    (``api_key``/``base_url``/``model_type``/``engine``/``max_tokens``), which are meaningless (or,
    for ``base_url``, actively misleading) for this adapter. **``max_tokens`` is in that list, so
    ``RLM_MAX_TOKENS`` is inert for an auto-routed subscription role**: the SDK exposes no output
    cap, so nothing would honour it. The upside is that no cap is recorded either, where passing
    ``max_tokens=`` to a hand-built ``ClaudeAgentLM`` stages one the call never applies. An
    explicit ``main_lm=``/``sub_lm=`` kwarg always wins outright regardless of the model string.
    the prefix is only consulted for a role left ``None``. This can raise, in addition to this
    function's own errors: ``ValueError`` for a bare prefix with no model id (e.g.
    ``"claude-agent-sdk/"``); ``RuntimeError`` if ``ANTHROPIC_API_KEY`` is set (``ClaudeAgentLM``
    refuses to start while it's set: the CLI silently prefers it over subscription OAuth);
    ``ImportError`` with an install hint if the optional ``subscription`` extra isn't installed.
    None of these three can collide with this function's own unrelated ``except RuntimeError``
    ownership-error swallow below. The LM-construction step happens entirely BEFORE that
    ``try:`` block starts, so control flow cannot reach it from here regardless of message text.

    ``cfg.request_timeout_s`` is deliberately in that "never forwarded" set too, and it is the
    one whose absence is worth stating: it becomes ``dspy.LM(timeout=…)``, a bound on ONE HTTP
    request that dspy retries around. ``ClaudeAgentLM`` has its own deadline
    instead: end-to-end per call, INCLUDING time queued behind the SDK's concurrency semaphore,
    so the two are not the same number and mapping one onto the other would make queued
    sub-LM calls time out from waiting alone. An auto-routed role therefore stays on
    ``ClaudeAgentLM``'s default; ``configure(main_lm=ClaudeAgentLM(model, timeout_s=…))`` is how
    you choose it. Setting ``request_timeout_s`` with an auto-routed role logs a warning saying
    exactly this, because a silently-ignored timeout reads as a bounded route that is not.
    """
    cfg = config or RLMConfig.from_env()
    # Captured BEFORE auto-routing, because a per-role passthrough is a no-op on any LM this
    # function does not build -- explicitly injected and subscription-auto-routed alike.
    injected = {"main": main_lm is not None, "sub": sub_lm is not None}
    #: Each auto-routed role ("main_lm"/"sub_lm") and the class its route built.
    auto_routed: dict[str, str] = {}
    if main_lm is None:
        routed = _maybe_subscription_lm(cfg.main_model)
        if routed is not None:
            main_lm, auto_routed["main_lm"] = routed[0], routed[1].attr
    if sub_lm is None:
        routed = _maybe_subscription_lm(cfg.sub_model)
        if routed is not None:
            sub_lm, auto_routed["sub_lm"] = routed[0], routed[1].attr

    # litellm defaults to an aiohttp transport whose pooled ClientSession is bound to the current
    # asyncio loop. A driver that runs each task in its own short-lived `asyncio.run` loop (e.g. a
    # server handling per-request runs) then leaves a dangling session when that loop closes, and
    # aiohttp logs a noisy "Unclosed connector". Force litellm onto httpx when it is installed, before
    # the first LM call: an LM on dspy's litellm engine then creates no aiohttp session.
    if _dspy_compat.litellm_installed():
        with contextlib.suppress(Exception):
            import litellm

            litellm.disable_aiohttp_transport = True

    # `cache` goes in unconditionally, unlike `timeout` below, because sending it explicitly IS the
    # change: dspy.LM's own default is True and the RLM workload wants False. It sits in the shared
    # dict rather than being forced after the merge, so `main_lm_kwargs={"cache": true}` still wins
    # for a consumer who wants dspy's behaviour on one role. See `RLMConfig.lm_cache` for why the
    # per-LM flag rather than `dspy.configure_cache`, and for which injected LMs it cannot reach.
    lm_kwargs: dict[str, Any] = {"cache": cfg.lm_cache}
    if cfg.api_key is not None:
        lm_kwargs["api_key"] = cfg.api_key
    if cfg.max_tokens is not None:
        lm_kwargs["max_tokens"] = cfg.max_tokens
    if cfg.base_url:
        # A base_url is an OpenAI-compatible server. The model string becomes the dialect's prefix
        # plus the id the server serves, on dspy's native engine and the Chat Completions API; see
        # `_dspy_compat.endpoint_model` and `endpoint_lm_kwargs`.
        lm_kwargs["base_url"] = cfg.base_url
        lm_kwargs.update(_dspy_compat.endpoint_lm_kwargs())
    elif cfg.endpoint_dialect != "openai":
        logger.warning(
            "endpoint_dialect=%r is ignored: it selects the wire a base_url endpoint speaks, and no "
            "base_url is set. Without one the model string's own prefix selects the provider.",
            cfg.endpoint_dialect,
        )
    if cfg.request_timeout_s is not None:
        # A client setting every dspy engine takes: the native engine bounds each wait for the
        # reply with it. Set ONLY when the consumer asked for one: sending `timeout=None`
        # explicitly is not the same as sending nothing, and the default must stay dspy's.
        lm_kwargs["timeout"] = cfg.request_timeout_s
        if auto_routed:
            # It reaches `dspy.LM` and nothing else, so on a subscription-routed role it is a
            # NO-OP, and a silent one is how a consumer ends up believing a route is bounded
            # when it is not. Deliberately NOT forwarded as the adapter's `timeout_s=`: every
            # community route takes one, and it is a different quantity (an END-TO-END per-call
            # deadline that INCLUDES time queued behind the adapter's concurrency cap, not a
            # per-HTTP-request bound that dspy retries around), so one number
            # cannot mean both. Under `llm_query_batched`'s thread fan-out a value that is
            # generous per request would make queued sub-LM calls time out from waiting alone.
            routed = ", ".join(f"{role} to {attr}" for role, attr in auto_routed.items())
            logger.warning(
                "request_timeout_s=%s bounds the model requests of an LM configure() builds, and does not "
                "reach the auto-routed %s (%s). Each of those is bounded by its adapter's own "
                "end-to-end per-call deadline, a DIFFERENT quantity, which is why this knob does "
                "not drive it. To choose that number, build the LM yourself and pass it in, "
                "e.g. configure(main_lm=<Adapter>(model, timeout_s=...)).",
                cfg.request_timeout_s,
                "role" if len(auto_routed) == 1 else "roles",
                routed,
            )
    # Both LMs are plain dspy.LM. In "json" mode it's _LenientJSONAdapter (not the LM) that
    # sets the json_schema response_format, so the LM needs no special capability flag.
    # An injected main_lm/sub_lm (explicit, or resolved above via subscription auto-routing) is
    # used verbatim: we build a plain dspy.LM from config ONLY for a role that is still None here.
    #
    # Per-ROLE passthrough, merged OVER the shared kwargs for that role only. The merge is
    # SHALLOW, and that is correct only because the kit sets no nested kwarg of its own here
    # today: `extra_body` is never in `lm_kwargs`, so `{**lm_kwargs, **role}` cannot drop
    # anything the kit needed. **IF A FUTURE CHANGE MAKES THE KIT SET `extra_body` (or any other
    # nested kwarg) ITSELF, this silently clobbers the caller's value**: at that moment this has
    # to become a deliberate deep merge with a stated conflict rule, not something discovered
    # from a request that quietly lost a key. `config._LM_KWARGS_REFUSED` keeps the FLAT keys the
    # kit owns out of here. The kit-set keys a role may override (`max_tokens`, `engine`,
    # `model_type`, `cache`) are each read back off the LM into the trace, which is why they are
    # allowed.
    main_kwargs = {**lm_kwargs, **(cfg.main_lm_kwargs or {})}
    sub_kwargs = {**lm_kwargs, **(cfg.sub_lm_kwargs or {})}
    for role, role_kwargs in (("main", cfg.main_lm_kwargs), ("sub", cfg.sub_lm_kwargs)):
        if role_kwargs and (injected[role] or f"{role}_lm" in auto_routed):
            # Same class of silence as the request_timeout_s warning above: the kwargs reach
            # `dspy.LM` and nothing else, so on a role whose LM this function did not build they
            # do NOTHING, and a consumer reading its own config would believe otherwise.
            logger.warning(
                "%s_lm_kwargs=%s is ignored: the %s LM was %s, so configure() did not build it. "
                "Pass these kwargs to an LM you build and inject, e.g. dspy.LM(model, **kwargs).",
                role, sorted(role_kwargs), role,
                f"auto-routed to {auto_routed[f'{role}_lm']}"
                if f"{role}_lm" in auto_routed else "supplied explicitly",
            )
    built: dict[str, Any] = {}
    if main_lm is None:
        main_lm = built["main"] = dspy.LM(_model_string(cfg, cfg.main_model), **main_kwargs)
    if sub_lm is None:
        sub_lm = built["sub"] = dspy.LM(_model_string(cfg, cfg.sub_model), **sub_kwargs)
    for role, lm in built.items():
        _check_lm_route(role, lm)
    for role, lm in (("main", main_lm), ("sub", sub_lm)):
        _warn_if_thinking_budget_cannot_act(role, lm)
    # Pass the adapter explicitly so a re-configure is clean: `_build_adapter` builds every
    # adapter the kit offers, dspy's default XML one included, with the kit's settings.
    #
    # `dspy.configure` is OWNER-LOCKED: dspy records the first thread + async task that calls it and
    # raises if a LATER call comes from a different thread/task. That breaks a long-lived driver that
    # runs each task in its own worker thread (e.g. a server handling per-request runs): run #2's
    # configure is on a new thread and crashes. But the global LM config is already set by the first
    # call and is READABLE from every thread, so on a non-owner thread we simply reuse it. Swallow ONLY
    # that ownership error; re-raise anything else.
    try:
        # `disable_history` rides the SAME call, so it shares that call's owner-lock fate: when
        # another thread configured dspy first, the except below reuses the global config and this
        # setting is not applied either. That is narrow rather than benign: a thread that reached
        # here through the kit's own `configure()` already applied it, so the gap is a NON-kit caller
        # configuring dspy first. Stated because a global setting that silently fails to apply is
        # worse than one that was never offered.
        dspy.configure(
            lm=main_lm,
            adapter=_build_adapter(cfg.adapter, native_reasoning=cfg.native_reasoning),
            disable_history=cfg.disable_history,
        )
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
        "rlm-harness configured: main=%s sub=%s interpreter=%s adapter=%s observe=%s",
        cfg.main_model,
        cfg.sub_model,
        cfg.interpreter,
        cfg.adapter,
        cfg.observe,
    )
    return cfg


def _build_adapter(name: str, *, native_reasoning: bool = False) -> dspy.Adapter:
    """The adapter instance for the configured adapter name.

    ``"json"`` is ``_LenientJSONAdapter``, which always sends the ``json_schema`` response format.
    ``"chat"`` is a ``ChatAdapter`` and ``"default"`` dspy's default ``XMLAdapter``, each with
    ``use_json_adapter_fallback=False``: the fallback would, on a parse error, retry through
    ``JSONAdapter`` with ``response_format={"type":"json_object"}``, which endpoints like vLLM reject
    (they require a schema). Every adapter gets the kit's ``native_response_types``, so the planner
    keeps its explicit ``plan`` field unless ``native_reasoning`` is on.
    """
    if name == "json":
        adapter: dspy.Adapter = _LenientJSONAdapter()
    elif name == "chat":
        adapter = dspy.ChatAdapter(use_json_adapter_fallback=False)
    else:
        adapter = _dspy_compat.default_adapter()
        adapter.use_json_adapter_fallback = False
    adapter.native_response_types = _dspy_compat.adapter_native_response_types(native_reasoning)
    return adapter


def _try_instrument() -> None:
    """Best-effort OpenInference/DSPy instrumentation + Langfuse client; never fatal."""
    try:
        from openinference.instrumentation.dspy import DSPyInstrumentor

        DSPyInstrumentor().instrument()
        logger.info("DSPy instrumentation enabled.")
    except Exception as exc:
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
    except Exception as exc:
        logger.debug("Langfuse client not initialized (%s); continuing.", exc)


def _require_configured() -> None:
    if not _STATE.configured:
        raise RuntimeError(
            "rlm-harness is not configured. Call rlm_harness.configure(RLMConfig.from_env()) "
            "once before running a task."
        )


def get_config() -> RLMConfig:
    """The effective ``RLMConfig`` that ``configure`` stored. PUBLIC accessor: re-exported as
    ``rlm_harness.get_config``. Lets a consumer read back the active config (budgets, model names,
    interpreter) without reaching into private runtime state. Requires ``configure`` to have run."""
    _require_configured()
    assert _STATE.config is not None
    return _STATE.config


def get_sub_lm() -> dspy.LM:
    """The configured base sub-LM (the recursion seat). PUBLIC accessor: re-exported as
    ``rlm_harness.get_sub_lm``. Returns the BARE LM: escalations through it are traced anyway,
    because ``RLMTask`` wraps whatever it is given for ``sub_call`` recording (1.7.0). A consumer
    that additionally wants a validated / post-processed sub-LM wraps THIS with
    ``intercept_sub_lm`` and passes the result as ``RLMTask(sub_lm=...)``; using the
    configured instance (rather than reconstructing ``dspy.LM(cfg.sub_model, ...)``) keeps a
    single source of truth so it can't drift from what ``configure`` built. Requires
    ``configure`` to have run."""
    _require_configured()
    assert _STATE.sub_lm is not None
    return _STATE.sub_lm
