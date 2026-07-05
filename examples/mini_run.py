"""Minimal real end-to-end RLM run — verifies the forward() path with a live model.

Self-contained: configures from RLM_* env, runs one tiny task through a real
dspy.RLM (real Deno sandbox + real model), records the trajectory, and prints
both the validated result and a trajectory summary so we can confirm the live
shape matches what trace.py expects.

Run:  set RLM_API_KEY / RLM_MAIN_MODEL (and optionally RLM_BASE_URL/RLM_SUB_MODEL),
      then `uv run --no-sync python -m examples.mini_run`
"""

from __future__ import annotations

import asyncio
import json

from pydantic import BaseModel, Field

from rlm_kit import (
    RLMConfig,
    RLMTask,
    TraceRecorder,
    configure,
    export_rl,
    group_by_run,
    load_events,
)


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
    cfg = configure(RLMConfig.from_env())
    print(f"main={cfg.main_model} sub={cfg.sub_model} interpreter={cfg.interpreter}")

    # A single RLM attempt (max_retries defaults to 1 — no whole-RLM re-run). The in-run REPL
    # loop is bounded by max_iterations, not by max_retries.
    task = Summarize()

    trace_path = "./traces/mini_run.jsonl"
    document = (
        "Recursive Language Models treat unbounded context as a variable in a sandboxed "
        "REPL and recursively call sub-models over it, instead of stuffing everything into "
        "one prompt."
    )
    with TraceRecorder(trace_path, run_id="mini-001", meta={"task": "summarize"}):
        result = await task.arun(document=document)

    print("\n=== RESULT ===")
    print(result.model_dump_json(indent=2))

    events = load_events(trace_path, run_id="mini-001")
    kinds: dict[str, int] = {}
    for e in events:
        kinds[e["type"]] = kinds.get(e["type"], 0) + 1
    print("\n=== TRACE EVENT COUNTS ===")
    print(json.dumps(kinds, indent=2))

    rl = export_rl(group_by_run(events))
    print(f"\nexported {len(rl)} RL step records from the live trajectory")


if __name__ == "__main__":
    asyncio.run(main())
