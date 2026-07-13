"""Run rlm-kit on a Claude Pro/Max SUBSCRIPTION via the official Claude Agent SDK.

`ClaudeAgentLM` is a `dspy.BaseLM` adapter over `claude-agent-sdk` (`pip install
"claude-agent-sdk>=0.1.60"`), injected through the kit's existing public seam
`configure(main_lm=..., sub_lm=...)` — the kit itself is unchanged. Each LM call is one
stateless `query()` through the Claude Code CLI on YOUR OWN subscription login: the
officially sanctioned path for individual subscribers, as opposed to the blocked
OAuth-token-against-the-API routes. Every call is a pure completion — no agent loop, no
tools, no filesystem access, no settings leakage (`tools=[]`, `max_turns=1`,
`setting_sources=[]`) — so rlm-kit's sandbox stays the only place code runs.

Setup:
  1. Install the Claude Code CLI and log in with your Pro/Max account (`claude` → `/login`),
     or mint a long-lived token: `claude setup-token` → export `CLAUDE_CODE_OAUTH_TOKEN`.
  2. `unset ANTHROPIC_API_KEY` — the CLI silently prefers it over subscription OAuth, so a
     leftover key bills API credit; the constructor refuses to start while it is set.
  3. `uv pip install "claude-agent-sdk>=0.1.60"` into this venv, `brew install deno` for the
     default pyodide sandbox, then `uv run --no-sync python -m examples.claude_agent_lm`.

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
import os
import re
import threading
from typing import Any, Optional

import dspy
import litellm
from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query
from pydantic import BaseModel, Field

from rlm_kit import RLMConfig, RLMTask, TraceRecorder, configure

_BACKOFF_S = 30.0
# Phrase-level, not bare substrings: "rate"/"limit" alone would false-match ordinary error text
# ("failed to generate", "delimiter") and turn a non-retryable error into a 30s sleep + retry.
_RATE_LIMIT_RE = re.compile(r"rate.?limit|usage limit|overloaded|429|529")


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


_BRIDGE: Optional[_Bridge] = None
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
    prompt: Optional[str], messages: Optional[list[dict[str, Any]]]
) -> tuple[Optional[str], str]:
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


def _translate_response_format(response_format: Any) -> Optional[dict[str, Any]]:
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

    Satisfies both rlm-kit seats: the planner calls `aforward(messages=...)` through the
    adapter, the sub-LM seat calls `forward(prompt)` synchronously from `llm_query[_batched]`
    worker threads — both funnel into one coroutine on the shared bridge loop. Works under
    `intercept_sub_lm` unchanged. Unknown lm_kwargs (temperature, max_tokens, ...) are
    tolerated and ignored: the SDK exposes no sampling controls.

    `model` is an alias (`"opus"` / `"sonnet"` / `"haiku"`) or a full Claude model id; the
    trace label becomes `claude-agent-sdk/<model>`.
    """

    def __init__(
        self,
        model: str = "sonnet",
        *,
        timeout_s: float = 600.0,
        allow_api_key: bool = False,
        cwd: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if os.environ.get("ANTHROPIC_API_KEY") and not allow_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is set — the Claude Code CLI silently prefers it over your "
                "subscription OAuth, so this run would bill API credit. Unset it (or pass "
                "allow_api_key=True if that is genuinely what you want)."
            )
        super().__init__(model=f"claude-agent-sdk/{model}", **kwargs)
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
        self, prompt: Optional[str], messages: Optional[list[dict[str, Any]]], kwargs: dict[str, Any]
    ) -> "litellm.ModelResponse":
        system_prompt, user_prompt = _split_messages(prompt, messages)
        output_format = _translate_response_format(kwargs.get("response_format"))
        options = ClaudeAgentOptions(
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
            except Exception as exc:  # noqa: BLE001 — retry once iff rate-limit-shaped
                if not _looks_rate_limited(str(exc)):
                    raise
                await asyncio.sleep(_BACKOFF_S)
                result, text = await self._query_once(user_prompt, options)
        usage = result.usage or {}
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)
        response = litellm.ModelResponse(
            model=self.model,
            choices=[{"message": {"role": "assistant", "content": text}}],
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        )
        if result.total_cost_usd is not None:
            # Cosmetic: what the tokens WOULD cost on the API. On a subscription nothing bills
            # it and no rlm-kit budget reads it; it just keeps `lm.history` honest.
            response._hidden_params["response_cost"] = result.total_cost_usd
        return response

    async def _query_once(
        self, user_prompt: str, options: ClaudeAgentOptions
    ) -> tuple[ResultMessage, str]:
        result: Optional[ResultMessage] = None
        async for message in query(prompt=user_prompt, options=options):
            if isinstance(message, ResultMessage):
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


# -- demo: one tiny task through a real dspy.RLM on the subscription ---------


class Summary(BaseModel):
    title: str = Field(..., description="a short title")
    gist: str = Field(..., description="one short sentence")


class Summarize(RLMTask):
    signature = "document: str -> summary: Summary"
    output_field = "summary"
    output_model = Summary
    instructions = (
        "Read the document and return a Summary JSON (a title and a one-sentence gist). "
        "Keep it short."
    )


async def main() -> None:
    # The config's model names are inert once LMs are injected (configure builds LMs from
    # config ONLY for seats not supplied) — they label the log line and the trace.
    cfg = configure(
        RLMConfig(main_model="claude-agent-sdk/sonnet", sub_model="claude-agent-sdk/haiku"),
        main_lm=ClaudeAgentLM("sonnet"),
        sub_lm=ClaudeAgentLM("haiku"),
    )
    print(f"main={cfg.main_model} sub={cfg.sub_model} interpreter={cfg.interpreter}")

    document = (
        "Recursive Language Models treat unbounded context as a variable in a sandboxed "
        "REPL and recursively call sub-models over it, instead of stuffing everything into "
        "one prompt."
    )
    with TraceRecorder("./traces/claude_agent_lm.jsonl", run_id="claude-agent-001"):
        result = await Summarize().arun(document=document)

    print("\n=== RESULT ===")
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
