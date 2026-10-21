"""Example: a store a run reads from, writes to, and leaves something behind in.

A run's REPL forgets everything when it ends. `MemoryStore` (`memory.py`, `tools/memory.py`) is
the part that does not: one SQLite file, no extra and no service. This example walks the whole
arc on one live run, because the pieces only mean anything together:

- the host seeds the store, choosing each item's provenance from the code path the text arrived
  by (`observed` for text it took from a source, `said` for text a person wrote),
- the three tools go into the task, so the model can search and read memory and write back,
- `compute_memory_use` says what the run did with it, read, written, and read without ever
  naming again,
- `consolidate_run` asks a model once what the finished run should leave for a later one, and
  stores as `observed` only a fact whose quote the host finds in a source it trusts,
- `purge` takes something back out, which no tool and no model-facing path can reach.

**What the model cannot do is the point.** `memory_note(text, revises)` takes no argument that
names a provenance, so a model's words arrive as `inferred` whatever the prompt, the plan, or a
page the model read says. That label is what makes a note safe to keep: a note written after
reading a hostile page comes back in every later run whose search matches it, so a reader needs
to tell a model's own words from a source.

The store is written OUTSIDE the checkout by default, under `$TMPDIR`, because a database in a
working tree is a stray waiting to be committed (1.21.0 shipped a fix for three of them). Point
`RLM_MEMORY_DIR` at a directory you keep to watch memory carry across runs: run this twice and
the second run searches what the first one left.

Illustrative. Needs real model creds and a sandbox, so it is NOT imported by the test suite.
(`tests/test_memory*.py` and `tests/test_tool_descriptions.py` cover the store, the tools and the
model-facing passes offline, with a scripted LM instead of a real one.)

Run:

    export RLM_MAIN_MODEL=...  RLM_API_KEY=...
    python examples/memory_run.py
"""

from __future__ import annotations

import asyncio
import inspect
import os
import tempfile
import time
from pathlib import Path

from rlm_harness import (
    MemoryStore,
    RLMConfig,
    RLMTask,
    TraceRecorder,
    configure,
    load_events,
)
from rlm_harness.memory_tasks import consolidate_run
from rlm_harness.metrics import compute_memory_use
from rlm_harness.tools import (
    make_memory_get_tool,
    make_memory_note_tool,
    make_memory_search_tool,
)

# The text the host holds. The seeded item below is taken FROM it, which is what makes that item
# `observed`, and passing it to consolidate_run as evidence is what lets a fact quoting it be
# stored as `observed` too. Nothing the model produces is ever a source for itself.
HANDBOOK = (
    "Every piece of model-authored code runs inside the Deno sandbox, which is the security "
    "boundary: an allowlist of commands is not a substitute for it. The sandbox has no network "
    "access, and a turn that exceeds its wall-clock budget is restarted once before the run is "
    "abandoned."
)


class Answer(RLMTask):
    signature = "question: str -> answer: str"
    output_field = "answer"
    instructions = (
        "Call memory_search to look for what is already known before answering, and memory_get "
        "to read a promising item in full. Answer from what you find. When the answer rests on "
        "something worth keeping for a later run, call memory_note once with it."
    )


def _store_dir() -> Path:
    override = os.getenv("RLM_MEMORY_DIR")
    base = Path(override) if override else Path(tempfile.gettempdir()) / "rlm-harness-memory-example"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _show(label: str, items) -> None:
    print(f"\n=== {label} ===")
    for item in items:
        head = item.text if len(item.text) <= 90 else item.text[:90] + " …"
        print(f"  {item.id}  {item.provenance:<8}  origin={item.origin}")
        print(f"    {head}")


async def main() -> None:
    cfg = configure(RLMConfig.from_env())
    print(f"main={cfg.main_model} sub={cfg.sub_model} interpreter={cfg.interpreter}")

    where = _store_dir()
    print(f"store: {where / 'memory.db'}")

    # index_origins=True also indexes an item's origin, so a file name or an address is searchable.
    store = MemoryStore(where / "memory.db", index_origins=True)

    # The host writes the first two labels, and it is the host's job to pick them from the code
    # path rather than from anything the text says about itself. The kit cannot check these two.
    seeded = store.search(provenance=["observed", "said"], limit=5)
    if not seeded:
        store.add(
            HANDBOOK.split(". ", 1)[0] + ".",   # the handbook's own first sentence, not a summary
            provenance="observed",
            origin="handbook.md#sandbox-policy",
            tags=["sandbox"],
        )
        store.add(
            "Prefer answers that name the source they came from.",
            provenance="said",
            origin="user",
            tags=["style"],
        )
    _show("the store before the run", store.search(limit=10))

    # Reading and writing are separate factories, so a task that should only consult memory
    # leaves the note tool out. This one may write, so it gets all three.
    note_tool = make_memory_note_tool(store, tags=["example"])
    task = Answer(tools=[make_memory_search_tool(store), make_memory_get_tool(store), note_tool])

    # The guarantee does not depend on the model choosing to behave, so it can be read off the
    # tool rather than waited for: there is no argument here that names a provenance.
    print(f"\nthe write tool a model is given: {note_tool.__name__}{inspect.signature(note_tool)}")

    # A run id unique per invocation, so load_events below reads THIS run rather than every run
    # this example ever appended to the file.
    run_id = f"memory-{int(time.time())}"
    trace_path = str(where / "memory_run.jsonl")
    with TraceRecorder(trace_path, run_id=run_id, meta={"task": "answer"}):
        answer = await task.arun(question="What is the security boundary for model-authored code?")

    print("\n=== ANSWER ===")
    print(answer)

    events = load_events(trace_path, run_id=run_id)

    # What the run did with memory. An unreferenced read is an upper bound on waste, not a
    # measurement of it: a model can use what an item says without ever writing its id again.
    use = compute_memory_use(events)
    print("\n=== MEMORY USE ===")
    print(f"  read:         {sorted(use.read)}")
    print(f"  written:      {sorted(use.written)}")
    print(f"  unreferenced: {sorted(use.unreferenced)}")

    notes = store.search(provenance="inferred", limit=10)
    if notes:
        _show("what the model wrote, and it could only ever be inferred", notes)
    else:
        print("\n=== the model wrote no note this run, which is its own choice to make ===")

    # One call per run. `evidence` is how a run's sources reach `observed`: a fact is promoted
    # only when its quote is found in a source the host trusts, and what gets stored is then the
    # source's own span rather than the model's paraphrase.
    report = await consolidate_run(store, events, evidence={"handbook": HANDBOOK})
    print("\n=== CONSOLIDATED ===")
    print(f"  observed: {report.observed}")
    print(f"  inferred: {report.inferred}")
    for item_id, reason in report.downgraded:
        print(f"  downgraded {item_id}: {reason}")
    for text_head, reason in report.skipped:
        print(f"  skipped {text_head!r}: {reason}")
    if report.error:
        print(f"  error: {report.error}")

    _show("the store after the run", store.search(limit=20))

    # Erasure is the host's alone, and it is the one thing that leaves the file. An id is never
    # reused, so a purged id in an old trace names nothing rather than something else.
    added = [note.id for note in notes] + report.observed + report.inferred
    if added:
        gone = added[0]
        print("\n=== PURGED ===")
        print(f"  ids:    {store.purge([gone], reason='example cleanup')}")
        print(f"  get:    {store.get(gone)}")
        print(f"  record: {store.purged(gone)}")


if __name__ == "__main__":
    asyncio.run(main())
