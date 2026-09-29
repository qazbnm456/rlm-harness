# rlm-harness: agent guide

`rlm-harness` is a small, reusable scaffold over `dspy.RLM` (Recursive Language Models) for
building tasks of any kind. A task is a *declaration*: an `RLMTask` subclass with a `signature`,
`output_field`, optional `output_model`, `instructions`, and `tools`. Retry, validation, sandbox
selection, budget caps, and observability are inherited rather than written per task.

Python, managed with `uv`. `requires-python = ">=3.11"`; the default interpreter is 3.12.

## Verify

Run both before pushing. CI gates on both, and a green `pytest` does not imply a green lint:

```bash
uvx ruff check .
uv run --group dev --extra mcp --extra grep --extra gitignore python -m pytest -q
```

The extras are not optional. Without them the MCP, `regex`-timeout, and `.gitignore` tests skip
instead of running, which is how a suite stays green while the thing it exists to prove is untested.

That local run is one interpreter on one OS, and CI has ten jobs across four workflows. **Read
[`docs/VERIFY.md`](docs/VERIFY.md) when a change leans on stdlib, platform, or dspy behaviour, or
when you are about to release**: it covers the 3.11 floor, Windows, macOS, the published artifact,
packaging, the newest-dspy and newest-MCP jobs, and what each one is blind to. Add `--python 3.11`
yourself when a change touches `zipfile`/`tarfile`, `resource`, `multiprocessing`, or `asyncio`.

**Before claiming a change is done, actually run the two commands above and paste the output.**

A *live* `dspy.RLM` run additionally needs model credentials and a Deno sandbox; `examples/` show
it. Never in CI, it costs money.

## Before you change the kit

**Read [`docs/INVARIANTS.md`](docs/INVARIANTS.md) before editing anything under `rlm_harness/`.**
Every rule in it was broken once, in this repo or in a downstream consumer, and each one names the
failure it came from. Skipping it is how they get broken a second time. It is the authority on:

- the sandbox as the security boundary, and the two watchdog outcomes
- resolving every dspy API difference in `_dspy_compat.py`, never at a call site
- which modules must stay dspy-free, and why `import rlm_harness` must not import dspy
- what a tool may be: sync, explicit params, sanitized name
- the trace as a versioned, additive-only wire format, and trajectories-never-reward
- the public surface, and how a consumer extends instead of forking

**Two of those invariants fire when you are READING rather than editing**, so they have their own
moment: before quoting a rate, a count or a coverage figure out of a trace corpus, read
"An absent event is not a measurement" and "Before believing a count" in that file. The failure they
guard touches no source file, which is why it is easy to reach without ever opening one.

**Read [`docs/CONVENTIONS.md`](docs/CONVENTIONS.md) before bumping the version, before adding,
renaming or removing a public name, and before promoting a consumer's workaround into the kit.** It
covers SemVer on the public surface (deprecation aliases, never a hard rename) and the
consumer-driven hardening loop. One mechanical rule from it that no test enforces: `pyproject.toml`
`[project].version` and `rlm_harness/__init__.__version__` must stay in sync.

**Keep the public surface vendor-neutral.** Source, docs, and commit messages refer to downstream
consumers generically. A `.githooks/` check enforces this on commit; enable it in a fresh clone with
`git config core.hooksPath .githooks`.

## Before compacting

**Read [`docs/HANDOFF.md`](docs/HANDOFF.md) before auto-compacting, or when asked for a recap.** It
says what must survive and where it belongs: stable invariants into `docs/INVARIANTS.md`, resolved
changes into `CHANGELOG.md`, so a summary carries only the in-flight state those files miss.

## Where things are

- [`README.md`](README.md): the front page, the tool inventory, the extras.
- [`rlm_harness/README.md`](rlm_harness/README.md): "the guide". Full layout, usage, configuration,
  and the consumer walkthrough. The place to look before adding a feature.
- [`CHANGELOG.md`](CHANGELOG.md): what changed and why, per release.
- [`CONTRIBUTING.md`](CONTRIBUTING.md): the short version of this file.
