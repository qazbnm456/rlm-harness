"""Example: a multi-megabyte input the model reads in slices, not one the prompt carries.

An input field normally reaches the sandbox as an inline Python literal, rebuilt every turn. For a
large text that is expensive in a way that does not show up until it hurts: a consumer measured
about 66x the payload in peak process-tree RSS for an 11.6 MB text, and on a 4 GB host that turned
every message over about 9 MB into a failure instead of an answer. `LargeText` stages the text as a
file the sandbox may read, which measured 0.71x the payload instead.

**The one thing to get right.** Model code must open `sandbox_path`, never `host_path`. dspy mounts
a granted file at `/sandbox/<basename>` and the host location does not resolve inside the sandbox,
so sandbox code handed the host path finds no file and reads nothing. Passing a `LargeText` as an
input value does this for you: the kit substitutes `sandbox_path` before dspy sees it, so the
variable the model receives is already the path that works.

**What constructing one does, and what nothing else can do.** It writes the text to a file whose
name the kit generates, and that file is the only path this run's sandbox is granted, read-only.
The grant exists because host code built the object. Nothing a model writes, no tool result and no
string that happens to look like a path can produce one, so none of them can open a grant or widen
it. `docs/INVARIANTS.md` opens with the sandbox being the security boundary, and this is the one
deliberate opening in it.

**Deleting the staged file is the caller's job**, through `close()` or the context manager below.
The kit does not delete it, because the same object can serve a second run.

Illustrative. Needs real model creds and a sandbox, so it is NOT imported by the test suite.
(`tests/test_large_text.py` covers the staging, the grant and its refusals offline.)

Run:

    export RLM_MAIN_MODEL=...  RLM_API_KEY=...
    python examples/large_text_run.py
"""

from __future__ import annotations

import asyncio

from rlm_harness import LargeText, RLMConfig, RLMTask, configure

#: Big enough that injecting it as a variable would be the expensive thing, small enough that the
#: example stays quick. The needle sits at the end, so answering it requires a seek rather than a
#: read of the whole file: that is the access pattern the staging exists to make affordable.
LINES = 60_000
NEEDLE = "the deploy was rolled back by the release engineer at 04:12 UTC"


def build_log() -> str:
    body = "".join(
        f"2026-10-21T03:{minute:02d}:{second:02d}Z INFO  worker-{minute % 7} heartbeat ok\n"
        for minute in range(60)
        for second in range(LINES // 60)
    )
    return body + f"2026-10-21T04:12:00Z NOTICE {NEEDLE}\n"


class Investigate(RLMTask):
    signature = "log: str, question: str -> answer: str"
    output_field = "answer"
    instructions = (
        "`log` is a path to a large file, not the text. Find what the question asks about by "
        "reading parts of the file: seek to the end for the latest entries, and read a slice "
        "rather than the whole file. Answer from what you read."
    )


async def main() -> None:
    cfg = configure(RLMConfig.from_env())
    print(f"main={cfg.main_model} sub={cfg.sub_model} interpreter={cfg.interpreter}")

    text = build_log()
    # A context manager, so the staged file is deleted even if the run raises.
    with LargeText(text) as log:
        print(f"staged {log.chars:,} characters at {log.sandbox_path}")
        print(f"(on this machine: {log.host_path}, which sandbox code cannot see)")

        answer = await Investigate().arun(
            log=log,
            question="What happened at 04:12?",
        )

    print("\n=== ANSWER ===")
    print(answer)
    print("\nthe needle was:", NEEDLE)


if __name__ == "__main__":
    asyncio.run(main())
