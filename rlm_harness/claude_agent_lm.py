"""Run rlm-harness on a Claude Pro/Max SUBSCRIPTION via the official Claude Agent SDK.

`ClaudeAgentLM` is a `dspy.BaseLM` adapter over `claude-agent-sdk`, injected through the
kit's existing public seam `configure(main_lm=..., sub_lm=...)` — the kit itself is
unchanged. Each LM call is one stateless `query()` through the Claude Code CLI on YOUR OWN
subscription login: the officially sanctioned path for individual subscribers, as opposed
to the blocked OAuth-token-against-the-API routes. Every call is a pure completion — no
agent loop, no tools, no filesystem access, no settings leakage (`tools=[]`,
`setting_sources=[]`) — so rlm-harness's sandbox stays the only place code runs. `max_turns` is
1 for a plain completion, 8 when structured `output_format` needs the SDK's extra validation
round.

Optional: needs `pip install "rlm-harness[subscription]"` (the `claude-agent-sdk` client) plus a
logged-in Claude Code CLI. `import rlm_harness` stays dspy/SDK-free — this module is imported only
on first `rlm_harness.ClaudeAgentLM` access (PEP 562), and the SDK only when an instance is built.

Setup:
  1. Install the Claude Code CLI and log in with your Pro/Max account (`claude` → `/login`),
     or mint a long-lived token: `claude setup-token` → export `CLAUDE_CODE_OAUTH_TOKEN`.
  2. `unset ANTHROPIC_API_KEY` — the CLI silently prefers it over subscription OAuth, so a
     leftover key bills API credit; the constructor refuses to start while it is set.
  3. `pip install "rlm-harness[subscription]"` (or `uv sync --extra subscription`) into this venv,
     `brew install deno` for the default pyodide sandbox. See `examples/claude_agent_lm.py`
     for a runnable end-to-end demo.

Politeness policy (this adapter is for ORDINARY, INDIVIDUAL use of your own account):
  - Concurrency is capped at 2 (dspy.RLM's `llm_query_batched` would otherwise fan 8 threads
    of CLI spawns at one personal subscription).
  - One retry ladder, smallest at each rung: the CLI's own retries are pinned to 2 (default
    10), the adapter retries once after 30s on a rate-limit-shaped error, and the kit's
    `max_retries` default of 1 means no whole-trajectory re-runs. When the usage window is
    exhausted the run fails cleanly instead of grinding it.
  - Do NOT point this at batch RL-rollout generation or eval sweeps — that is not "ordinary,
    individual usage"; use the API for scale. Expect ~2-5s CLI-spawn overhead per call.

Trade-offs vs. a plain `dspy.LM`: no temperature/top_p/n controls (the SDK exposes none), no
dspy-side caching, and no prompt caching across planner turns. Structured output IS supported:
the kit's default `json` adapter puts a pydantic class in `response_format`, which this adapter
translates to the SDK's native schema-validated `output_format`.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import re
import threading
from typing import TYPE_CHECKING, Any

import dspy
import litellm

if TYPE_CHECKING:  # annotations only — never imported at runtime (the SDK is an optional extra)
    from claude_agent_sdk import ClaudeAgentOptions, ResultMessage

logger = logging.getLogger(__name__)

_BACKOFF_S = 30.0
# Phrase-level, not bare substrings: "rate"/"limit" alone would false-match ordinary error text
# ("failed to generate", "delimiter") and turn a non-retryable error into a 30s sleep + retry.
_RATE_LIMIT_RE = re.compile(r"rate.?limit|usage limit|overloaded|429|529")

#: The SDK's own name for the per-iteration breakdown, and the key this kit files it under. They
#: differ on purpose: `iterations` collides head-on with `run_end.payload.budgets.iterations`,
#: which is the RLM TURN CAP -- an unrelated thing entirely. Two fields of that name in one
#: `run_end` payload is a misreading waiting to happen, so the key is ours; the LIST is verbatim.
_SDK_ROUNDS_KEY = "iterations"
_ROUNDS_KEY = "api_rounds"


def _api_rounds(usage: dict | None) -> dict[str, list] | None:
    """The provider's per-SAMPLING-ITERATION breakdown, wrapped one level, or ``None`` when absent.

    **What it answers, and what it does NOT.** Anthropic's own type docs define an entry as "one
    sampling iteration" -- for ``message`` entries, "such as the turns of a server-side tool use
    loop" -- and name one use for it outright: "Calculate the context window size from the last
    ``message`` entry".

    That is the question this exists for, and the honest form of the claim is CONDITIONAL. When a
    call made ONE API request running ONE SAMPLING ITERATION -- the ordinary case here -- the
    top-level fields and that single entry are the same numbers, so ``prompt_tokens`` already IS
    the context size. "One ``message`` entry" is NOT the same condition and does not suffice: a
    server-side fallback puts two entries in one request, since "a declined hop produces the
    existing ``message`` entry" while the serving hop produces a ``fallback_message`` -- one
    ``message``, no compaction, and totals that cover both hops -- inferred rather than stated
    upstream, which carves only ``compaction`` out of the top-level fields. If a declined hop's
    tokens turned out to be excluded too, this condition would merely be conservative: it would
    decline an equality that happens to hold, never assert one that does not.

    What the totals cannot say is WHICH call they were. This field is what makes the answer
    unconditional, not what makes it possible -- so a missing ``api_rounds`` means "no breakdown
    was reported", never "there is no context number here". A reader who takes the second reading
    discards an exact number wherever the condition happened to hold, and neither of us can tell
    from the totals where that was.

    It is **not** a decomposition of this call's token fields, and reading it as one is the trap.
    Two independent reasons, both from upstream rather than from measurement here:

    - The CLI accumulates a call's token fields across every API request it makes, but keeps
      ``iterations`` from the LAST request wholesale rather than concatenating. So on a call that
      needed more than one request -- a structured-output retry, exactly the interesting case --
      the totals span all of them and these rounds cover only the last.
    - A ``compaction`` entry's tokens "are not included in the top-level ``usage`` fields" at all,
      by Anthropic's own statement.

    **What makes the condition usual FOR THIS ADAPTER** is structural rather than statistical, and
    worth stating as such rather than as a frequency nobody counted: ``_acomplete`` sends
    ``tools=[]``, which removes the server-side tool-use loop Anthropic names as A driver of
    multiple iterations -- "such as" is their word, so it is an example rather than the mechanism
    -- and a plain completion runs ``max_turns=1``, one API request. Neither
    closes a fallback or advisor entry, and a call with ``output_format`` may take several
    requests -- so this is a reason to expect the coincidence, never a guarantee of it.

    Four entry types exist (``message``, ``fallback_message``, ``compaction``,
    ``advisor_message``), and the type matters: a ``compaction`` entry reports the cost of the
    summarisation, NOT the size of the context it closed, so Anthropic says not to derive a
    context size from one "even when it is the last entry".

    **Why it is nested.** ``{"rounds": [...]}`` rather than the bare list, and that is load-bearing
    rather than tidy. dspy's ``UsageTracker`` merges a model's usage entries with ``(current or 0)
    + (v or 0)``; for a bare LIST that is concatenation, an empty list is FALSY so ``[] or 0`` is
    ``0``, and the mixed cases then raise ``TypeError: int + list`` -- out of
    ``dspy.Module.__call__`` itself whenever ``dspy.configure(track_usage=True)`` is set, with no
    ``get_total_tokens()`` anywhere in the caller's code. The merge RECURSES into a dict value
    before it reaches that arithmetic, so one wrapper makes the crash structurally impossible and
    additionally restores call ORDER, which the flat form scrambles. ``tests/test_dspy_compat.py``
    pins that dspy behaviour so a change there goes red here rather than in a consumer.

    **Why non-empty.** ``all(...)`` over an empty list is ``True``, so a guard that only checked
    "list of dicts" would CARRY ``[]``. That is not a corner case: the CLI seeds its usage
    accumulator from a zero literal whose ``iterations`` is ``[]`` and only replaces it when a
    response actually carries the field, so ``[]`` is the RUNTIME outcome for every call whose
    response does not. Requiring content collapses absent / ``null`` / ``[]`` / malformed into ONE
    outcome with one meaning: no key.

    The guard is on the CONTAINER only, never on an entry's keys: an entry carries ``type``, an
    optional ``model`` and the four token fields today, and enumerating would freeze a set that
    demonstrably moves.
    """
    if not usage:
        return None
    value = usage.get(_SDK_ROUNDS_KEY)
    if isinstance(value, list) and value and all(isinstance(entry, dict) for entry in value):
        return {"rounds": value}
    if value is not None:
        # Separates "the guard dropped it" from "the SDK never reported it". Without this, an
        # absent `api_rounds` across a whole corpus is indistinguishable between four causes --
        # one of which is the key having been RENAMED upstream, the failure this project has now
        # paid for repeatedly. A JSON `null` is silent here, which is correct: that IS absence.
        logger.debug("claude-agent-sdk reported %r in an unusable shape: %r", _SDK_ROUNDS_KEY, value)
    return None


#: The three fields Anthropic splits a prompt across. `input_tokens` is only the part that was
#: neither written to nor read from the cache, so it is NOT the prompt size on its own.
_PROMPT_TOKEN_KEYS = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")


def _token_count(usage: dict | None, key: str) -> int:
    """One field of an SDK usage dict as an int, or 0 when it is missing or not a number.

    ``bool`` is excluded deliberately -- it is an ``int`` in Python, so an SDK returning ``True``
    would otherwise be summed as 1. Every field of ``result.usage`` goes through here; before this
    release BOTH reads were a bare ``.get(..., 0)``, so a ``None`` in either one reached the
    ``prompt_tokens + completion_tokens`` below and raised ``TypeError``, failing the whole LM call
    over a missing count. (``result.total_cost_usd`` is read raw further down -- it is not part of
    the usage dict, and it is stashed on ``_hidden_params``, off this response's usage path
    entirely -- dspy is what reads it back. litellm reads that key only in its proxy, which is
    not in play here.)
    """
    value = (usage or {}).get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _prompt_tokens_from_sdk_usage(usage: dict | None) -> int:
    """The prompt SIZE from an Agent SDK ``result.usage``, summed across all three input fields.

    ``input_tokens`` alone is the UNCACHED REMAINDER, not the prompt. The SDK caches the system
    prompt and tool definitions by default, so on an RLM turn nearly the whole prompt lands in
    the cache fields instead. Measured on a live subscription call with a ~2k-token prompt:

        first sight   input_tokens=2  cache_creation=2047  cache_read=0     -> 2049
        repeat        input_tokens=2  cache_creation=0     cache_read=2047  -> 2049

    Reading ``input_tokens`` alone recorded **2**, which is 0.1% of the real figure, so every
    consumer reading this adapter's prompt size -- out of ``run_end.payload.usage`` or out of
    ``lm.history`` -- was three orders of magnitude low. It does NOT touch 1.10.0's truncation
    ratio, which is ``completion_tokens / cap`` and never divides by this number. (That ratio is
    usually absent for a subscription run anyway: ``applied_lm_budget`` reads ``lm.kwargs``, and
    both the default constructor and the auto-routed path build this LM with none, so it returns
    ``None``. Pass ``max_tokens=`` explicitly and a cap IS recorded -- one the SDK never applied,
    see the class docstring.) A provider
    reporting none of the cache fields is unaffected: absent keys contribute nothing and the
    result equals ``input_tokens``.

    **This is a SIZE, not a COST basis.** The three fields bill at different rates (a cache READ
    is roughly a tenth of an uncached input token; a cache WRITE somewhat more than one), so
    pricing this sum at the uncached rate overstates a warm turn by close to an order of
    magnitude. It is the right number for "how big was the prompt" and the wrong one for "what
    did it cost" -- the cost the SDK reports for itself lands in ``response_cost`` below, and
    neither is derived from the other.

    And it is per CALL, not per turn: with an ``output_format`` set the adapter runs ``max_turns=8``
    (the SDK's structured-output round), so ``result.usage`` aggregates whatever that loop spent.
    **Since 1.11.0 a call may also carry ``api_rounds``** -- see ``_api_rounds``. It is NOT a
    breakdown of this number (the two cover different spans, and a ``compaction`` entry's tokens
    are excluded from these fields entirely). On a single-request, single-ITERATION call the two
    agree and this number already gives the context size; what it cannot do is tell you that the
    call WAS one, which is the gap ``api_rounds`` closes.
    """
    return sum(_token_count(usage, key) for key in _PROMPT_TOKEN_KEYS)


#: The sentinel `ClaudeAgentLM` stamps its own `dspy.LM.model` with (see `__init__` below) — the
#: kit's own convention, not a consumer-invented one. `runtime.configure()` reads this SAME
#: constant back to auto-route a `claude-agent-sdk/<id>` `main_model`/`sub_model` string onto a
#: `ClaudeAgentLM` for a role the caller didn't already override with an explicit `main_lm=`/
#: `sub_lm=`. Exported (lazily, alongside `ClaudeAgentLM`) so a consumer composing its OWN
#: model-string convention on top gets the single source of truth instead of hardcoding the
#: literal a second time.
SUBSCRIPTION_PREFIX = "claude-agent-sdk/"


def _require_claude_agent_sdk():
    """Import the optional `claude-agent-sdk`, or raise a friendly install hint (the `mcp.py` guard).

    Deferred out of module top on purpose: keeping the import here means the module itself loads
    without the extra (so `rlm_harness.ClaudeAgentLM` is gettable, like `mcp_tools`), and the SDK is
    only required once an instance is actually built.
    """
    try:
        import claude_agent_sdk
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "ClaudeAgentLM requires the optional dependency: pip install 'rlm-harness[subscription]'"
        ) from exc
    return claude_agent_sdk


class _Bridge:
    """Process-wide background event loop the async SDK is driven from (the `mcp.py` pattern).

    dspy calls the LM synchronously (the sub-LM seat is `target_lm(prompt)` from worker
    threads) and the planner's `aforward` runs on a loop that `repl.execute` blocks — so SDK
    coroutines must run on a SEPARATE loop that sync callers reach via
    `run_coroutine_threadsafe(...).result(timeout)`.
    """

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self._run, name="claude-agent-lm", daemon=True).start()
        # Politeness cap. Created off-loop on purpose: asyncio sync primitives bind their loop
        # lazily on first acquire (py>=3.10), and every acquire happens on self.loop.
        self.semaphore = asyncio.Semaphore(2)

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()


_BRIDGE: _Bridge | None = None
_BRIDGE_LOCK = threading.Lock()


def _bridge() -> _Bridge:
    # Module-level, NOT instance state: `BaseLM.copy()` deepcopies the LM, and a thread/loop/
    # semaphore held on the instance would break it. All ClaudeAgentLM instances (main + sub
    # seat) share one loop and one politeness cap.
    global _BRIDGE
    with _BRIDGE_LOCK:
        if _BRIDGE is None:
            _BRIDGE = _Bridge()
        return _BRIDGE


def _split_messages(
    prompt: str | None, messages: list[dict[str, Any]] | None
) -> tuple[str | None, str]:
    """Map dspy's stateless message list onto the SDK's (system_prompt, prompt) pair.

    dspy.RLM sends system + ONE packed user message per call; the sub-LM seat sends a bare
    prompt. The multi-message flatten is a defensive general case for a consumer's own
    `Predict` with demos/history.
    """
    if messages is None:
        return None, prompt or ""
    system: list[str] = []
    rest: list[dict[str, Any]] = []
    for message in messages:
        (system if message.get("role") == "system" else rest).append(message)
    if len(rest) == 1:
        user_prompt = str(rest[0].get("content", ""))
    else:
        parts = [f"{m.get('role', 'user').capitalize()}: {m.get('content', '')}" for m in rest]
        parts.append("Assistant:")
        user_prompt = "\n\n".join(parts)
    return ("\n\n".join(str(m.get("content", "")) for m in system) or None, user_prompt)


def _translate_response_format(response_format: Any) -> dict[str, Any] | None:
    """Translate dspy's `response_format` into the SDK's native `output_format`.

    The kit's default `json` adapter (`_LenientJSONAdapter`) injects a pydantic model CLASS —
    exactly what the SDK's schema-validated structured output wants. A dict form (stock
    adapters' `{"type": "json_object"}` fallback) has no SDK equivalent and is dropped: the
    prompt already demands JSON and the parse side (`json_repair`) is tolerant.
    """
    if response_format is None:
        return None
    schema = getattr(response_format, "model_json_schema", None)
    if callable(schema):
        return {"type": "json_schema", "schema": schema()}
    return None


def _looks_rate_limited(text: str) -> bool:
    return _RATE_LIMIT_RE.search(text.lower()) is not None


class ClaudeAgentLM(dspy.BaseLM):
    """A dspy LM whose completions run through the Claude Agent SDK on a subscription login.

    Satisfies both rlm-harness seats: the planner calls `aforward(messages=...)` through the
    adapter, the sub-LM seat calls `forward(prompt)` synchronously from `llm_query[_batched]`
    worker threads — both funnel into one coroutine on the shared bridge loop. Works under
    `intercept_sub_lm` unchanged. Unknown lm_kwargs (temperature, max_tokens, ...) are
    tolerated and ignored: the SDK exposes no sampling controls.

    **Ignored, but since 1.10.0 no longer invisible.** `_dspy_compat.applied_lm_budget` reads
    `lm.kwargs`, so a `max_tokens=` passed here is staged into the trace's `budgets.main` — or
    `budgets.sub`, for this LM in the sub-LM seat — as a cap the call never applied, and
    `completion_tokens` can then exceed it. That is NOT a case 1.10.0 anticipated: it reads the
    cap off the LM rather than off `RLMConfig` because dspy's own `_check_truncation` reads that
    same dict, which is to say on the assumption that an LM's kwargs ARE what it applied. This
    adapter is where the assumption does not hold, so read a cap recorded for it as configuration
    rather than as a measurement — and see `applied_lm_budget`, whose "actually APPLIED" is the
    claim this counterexample qualifies. Build without the kwarg and no cap is recorded for the
    role.

    Token usage is recorded per CALL, and with `output_format` set a call can span more than one
    API request. `_api_rounds` carries the provider's per-sampling-iteration breakdown into the
    trace beside it — so a reader can get the context size unconditionally, rather than only on
    the calls where the totals happen to coincide with it.

    `model` is an alias (`"opus"` / `"sonnet"` / `"haiku"`) or a full Claude model id; the
    trace label becomes `claude-agent-sdk/<model>`.
    """

    def __init__(
        self,
        model: str = "sonnet",
        *,
        timeout_s: float = 600.0,
        allow_api_key: bool = False,
        cwd: str | None = None,
        **kwargs: Any,
    ) -> None:
        # Fail fast at construction (with the install hint) rather than deep inside a call.
        _require_claude_agent_sdk()
        if os.environ.get("ANTHROPIC_API_KEY") and not allow_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is set — the Claude Code CLI silently prefers it over your "
                "subscription OAuth, so this run would bill API credit. Unset it (or pass "
                "allow_api_key=True if that is genuinely what you want)."
            )
        super().__init__(model=f"{SUBSCRIPTION_PREFIX}{model}", **kwargs)
        self._alias = model
        # End-to-end deadline per call, INCLUDING time queued behind the semaphore.
        self._timeout_s = timeout_s
        self._cwd = cwd

    def forward(self, prompt=None, messages=None, **kwargs):
        future = asyncio.run_coroutine_threadsafe(
            self._acomplete(prompt, messages, kwargs), _bridge().loop
        )
        try:
            return future.result(self._timeout_s)
        except concurrent.futures.TimeoutError:
            # Mirror mcp.py: don't leave the coroutine running on the shared loop.
            future.cancel()
            raise TimeoutError(f"claude-agent-sdk call timed out after {self._timeout_s}s") from None

    async def aforward(self, prompt=None, messages=None, **kwargs):
        future = asyncio.run_coroutine_threadsafe(
            self._acomplete(prompt, messages, kwargs), _bridge().loop
        )
        # wait_for cancels the wrapped future on timeout, propagating to the bridge-loop task —
        # the async twin of forward's cancel-on-timeout.
        return await asyncio.wait_for(asyncio.wrap_future(future), self._timeout_s)

    # -- runs ON the bridge loop --------------------------------------------

    async def _acomplete(
        self, prompt: str | None, messages: list[dict[str, Any]] | None, kwargs: dict[str, Any]
    ) -> litellm.ModelResponse:
        sdk = _require_claude_agent_sdk()
        system_prompt, user_prompt = _split_messages(prompt, messages)
        output_format = _translate_response_format(kwargs.get("response_format"))
        options = sdk.ClaudeAgentOptions(
            model=self._alias,
            system_prompt=system_prompt,
            # A PURE completion: `tools=[]` empties the toolset (`allowed_tools` would merely
            # auto-approve, not restrict) and `setting_sources=[]` so the user's CLAUDE.md /
            # settings / MCP servers never leak into RLM planner calls. max_turns caps the agent
            # loop: 1 for a plain completion (the sub-LM seat), a generous 8 when output_format is
            # set — the SDK's structured-output step spends turns BEYOND the model's own answer (a
            # reformat/validation round), and a complex RLM planner call exhausted a tight cap of 2
            # in a live run. tools=[] keeps the headroom from ballooning: with no tools each turn is
            # just the model, so a clean structured output still returns in 1-2 turns; the cap only
            # absorbs the tail.
            tools=[],
            max_turns=8 if output_format else 1,
            output_format=output_format,
            setting_sources=[],
            env={"CLAUDE_CODE_MAX_RETRIES": "2"},
            cwd=self._cwd,
        )
        async with _bridge().semaphore:
            try:
                result, text = await self._query_once(user_prompt, options)
            except Exception as exc:
                if not _looks_rate_limited(str(exc)):
                    raise
                await asyncio.sleep(_BACKOFF_S)
                result, text = await self._query_once(user_prompt, options)
        usage = result.usage or {}
        # An unreported usage stays ABSENT rather than becoming three zeroes. litellm
        # materialises a `Usage(0, 0, 0)` when handed `usage={}`, but leaves the attribute OFF
        # entirely when the kwarg is omitted. Both dspy reads then report absence rather than a
        # count: the legacy one (`dict(getattr(response, "usage", {}) or {})`) yields `{}` and
        # `UsageTracker.add_usage` skips an empty entry, and the typed one under
        # `dspy.context(experimental=True)` (`usage_from_response`) yields `None`. So omitting is
        # what keeps "the SDK reported nothing" distinguishable from "this call used zero tokens"
        # -- the guide's own published promise ("Usage may be absent, and absent is not zero"),
        # which this adapter was quietly breaking on every call that reported none.
        #
        # The guarantee is at the RESPONSE level, not the field level: a usage dict that is
        # present but carries no number this kit recognises (`{"input_tokens": None}`, or a dict
        # of nothing but metadata) still reports zeroes. Not because litellm rejects a `None` --
        # it accepts one and coerces it to 0 -- but because the only two states the
        # `ModelResponse` CONSTRUCTOR can express are "no attribute" and "a `Usage` object", and
        # a `Usage` holds numbers.
        # There is no "present but unreadable" to hand downstream. Absent means absent; zero
        # still means "zero as far as we could tell".
        reported: dict[str, Any] = {}
        if usage:
            prompt_tokens = _prompt_tokens_from_sdk_usage(usage)
            completion_tokens = _token_count(usage, "output_tokens")
            reported["usage"] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
            # The rounds ride ALONGSIDE these totals, never instead of them, and neither is
            # derived from the other -- they do not even cover the same span. See `_api_rounds`.
            rounds = _api_rounds(usage)
            if rounds is not None:
                reported["usage"][_ROUNDS_KEY] = rounds
        response = litellm.ModelResponse(
            model=self.model,
            choices=[{"message": {"role": "assistant", "content": text}}],
            **reported,
        )
        if result.total_cost_usd is not None:
            # Cosmetic: the SDK's OWN cost figure for the call, NOT derived from the tokens
            # above. On a subscription nothing bills it and no rlm-harness budget reads it; it
            # just keeps `lm.history` honest.
            #
            # DO NOT reconcile this against the token counts above -- they cover different
            # things. On the live call this was measured from (n=1, so read it as "this happens"
            # rather than "this always happens"), `result.model_usage` showed the Agent SDK
            # routing a second model alongside the requested one -- a haiku, ~1.7k input tokens --
            # and `total_cost_usd` was the SUM across BOTH, 0.014149 spanning sonnet + haiku,
            # while `result.usage` reported the requested model alone. `usage` is an untyped
            # passthrough from the CLI, so that last half is not verifiable from the SDK's types;
            # what is certain is that the two fields answered different questions on that call,
            # and anyone dividing one by the other concludes the numbers are broken.
            #
            # The side model's tokens are deliberately NOT folded into the per-call usage above.
            # A trace describes what the POLICY did, and the SDK routing a helper model is
            # infrastructure the planner never chose -- counting it would attribute work to the
            # policy that the policy did not do, and a trainer reads this as the trajectory. If
            # it ever needs to be data, give it its OWN optional field rather than mixing scopes
            # into a per-call count.
            response._hidden_params["response_cost"] = result.total_cost_usd
        return response

    async def _query_once(
        self, user_prompt: str, options: ClaudeAgentOptions
    ) -> tuple[ResultMessage, str]:
        sdk = _require_claude_agent_sdk()
        result: ResultMessage | None = None
        async for message in sdk.query(prompt=user_prompt, options=options):
            if isinstance(message, sdk.ResultMessage):
                result = message
        if result is None:
            raise RuntimeError("claude-agent-sdk produced no ResultMessage")
        if result.is_error or result.subtype != "success":
            raise RuntimeError(f"claude-agent-sdk error ({result.subtype}): {result.result!r}")
        if result.structured_output is not None:
            text = json.dumps(result.structured_output)
        else:
            text = result.result or ""
        if not text:
            # Never hand dspy empty text — it would become a bare "empty or null response"
            # AdapterParseError with less context than this.
            raise RuntimeError("claude-agent-sdk returned an empty result")
        return result, text
