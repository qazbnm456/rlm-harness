"""Demo: run a tiny RLMTask through a real dspy.RLM on a Google Antigravity sign-in (`agy`).

`AntigravityLM` is a community route: `from rlm_harness.community import AntigravityLM`, outside
the kit's SemVer promise. What it measured about `agy`, and what it does not control, is in its
module docstring (`rlm_harness/community/antigravity_lm.py`), and `rlm_harness/community/README.md`
quotes what Google has said about this use and when it was read. Google has not explicitly allowed
it, so building the LM logs one warning per process. This file is just the runnable demo.
Prereqs:

  1. Install the CLI (`curl -fsSL https://antigravity.google/cli/install.sh | bash`) and sign in
     once by running `agy`. The adapter finds it on PATH or at ~/.local/bin/agy.
  2. A Deno in dspy's runtime range `>=2.0.0,<3.0.0`: `brew install deno`, or
     `pip install "dspy[deno]"` for dspy's managed binary.
  3. Pick a model id from `agy models` (the list depends on the plan) and run:
     `AGY_MODEL=<id> uv run --no-sync python -m examples.antigravity_lm`

`GEMINI_API_KEY` and `AGY_ADC_AUTH` can move the bill off your sign-in; the adapter removes both
from every `agy` call's environment, so a set one does no harm here.

Each LM call is one `agy` process and is saved as an Antigravity conversation under
`~/.gemini/antigravity-cli`, so this demo leaves a few of them behind.
"""

from __future__ import annotations

import asyncio
import os

from pydantic import BaseModel, Field

from rlm_harness import RLMConfig, RLMTask, TraceRecorder, configure


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
    model = os.environ.get("AGY_MODEL")
    if not model:
        raise SystemExit("Set AGY_MODEL to an id `agy models` lists, e.g. AGY_MODEL=<id>.")
    # The `agy/` prefix is enough: configure() builds an AntigravityLM for each role that carries
    # it. Build one yourself (configure(main_lm=AntigravityLM(model, timeout_s=...))) to choose
    # its deadline or executable.
    cfg = configure(RLMConfig(main_model=f"agy/{model}", sub_model=f"agy/{model}"))
    print(f"main={cfg.main_model} sub={cfg.sub_model} interpreter={cfg.interpreter}")

    document = (
        "Recursive Language Models treat unbounded context as a variable in a sandboxed "
        "REPL and recursively call sub-models over it, instead of stuffing everything into "
        "one prompt."
    )
    with TraceRecorder("./traces/antigravity_lm.jsonl", run_id="antigravity-001"):
        result = await Summarize().arun(document=document)

    print("\n=== RESULT ===")
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
