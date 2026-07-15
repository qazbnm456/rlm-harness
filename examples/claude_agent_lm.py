"""Demo: run a tiny RLMTask through a real dspy.RLM on a Claude Pro/Max SUBSCRIPTION.

`ClaudeAgentLM` now ships in the kit — `from rlm_kit import ClaudeAgentLM`. The adapter's
setup, politeness policy, and trade-offs live in its module docstring
(`rlm_kit/claude_agent_lm.py`); this file is just the runnable demo. Prereqs:

  1. Log in to the Claude Code CLI with your Pro/Max account (`claude` → `/login`).
  2. `unset ANTHROPIC_API_KEY` (a leftover key would bill API credit; the adapter refuses).
  3. `uv sync --extra subscription` (or `pip install "rlm-kit[subscription]"`) + `brew install deno`.
  4. `uv run --no-sync python -m examples.claude_agent_lm`
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field

from rlm_kit import ClaudeAgentLM, RLMConfig, RLMTask, TraceRecorder, configure


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
