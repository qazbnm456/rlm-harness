# Contributing to rlm-kit

Thanks for helping improve `rlm-kit` — a small, reusable scaffold over `dspy.RLM`
(Recursive Language Models) for building tasks of any kind. This guide is the short
version; the deep design rules live in [`CLAUDE.md`](./CLAUDE.md) and the
extension contract in the guide's [**Building a consumer**](./rlm_kit/README.md#building-a-consumer).

## Development setup

```bash
uv sync --group dev
uv run --group dev python -m pytest    # the full suite — no live LLM, network, or Deno needed
uvx ruff check .                       # lint (CI enforces this)
```

The dspy-bearing tests use a `DummyLM` or skip when dspy is absent, so the suite
runs anywhere. A *live* `dspy.RLM` run additionally needs model credentials and a
Deno sandbox (`brew install deno`) — only `examples/` exercise that.

Before opening a PR: the suite is green, `ruff check` is clean, and any new
behavior has a test. CI runs the same on Python 3.11–3.13.

## The virtuous cycle — how this kit improves

`rlm-kit` is hardened by **dogfooding**: a real downstream consumer builds on the
scaffold, hits friction, and that friction becomes a fix *in the kit* so every
consumer benefits. When you find a rough edge, the question is "is this generic?"

- A **reusable mechanic** (a new tool primitive, a sandbox seam, a trace hook) is
  promoted into rlm-kit via the **base/wrap split**: the generic base + syntactic
  guard + factory live here; the provider + tracing live in the consumer. This is
  how `make_model_tool` / `make_fetch_tool` / `make_web_search_tool` are shaped.
- A **consumer-specific value** (a model name, a schema, a product term, a path)
  stays in the consumer, never here. Keep the public surface vendor-neutral —
  refer to consumers generically ("a consumer"), not by a specific project name.

So a good contribution either makes the generic half cleaner, or adds a new
primitive in the base/wrap shape — not a special case for one user.

## What not to break

These are load-bearing; see [`CLAUDE.md`](./CLAUDE.md) for the full list and the *why*.

- **The sandbox is the security boundary.** The default interpreter is sandboxed
  (`pyodide`/`deno`); the `local` interpreter stays refused unless explicitly opted in.
- **The trace is a frozen `rlm-kit/trace/v1` wire format.** Adding an optional payload
  field is fine; removing, renaming, or re-typing an event type / envelope key /
  established field is a `v2` break. `tests/test_contract.py` pins it — if it goes red,
  you're about to break a downstream reader, not the test.
- **Keep the dspy-free modules dspy-free.** `config.py`, `_retry.py`, `sandbox.py`,
  `tools/`, `trace.py`, `skills.py`, `replay.py`, `dataset.py` must not import dspy at
  module top, and `import rlm_kit` must not import dspy.
- **Tools passed to `RLMTask(tools=…)` must be sync.** dspy's interpreter calls them with
  a plain `()`; an `async def` tool returns an un-awaited coroutine and never runs.
- **rlm-kit produces trajectories, never reward.** The exporters carry a `reward=` hook the
  downstream trainer fills; scoring/training is a separate stage.

## Submitting changes

1. Fork and branch from `main`.
2. Make the change with a test; keep the suite green and `ruff check` clean.
3. Open a PR describing *what* and *why*. Reference any issue it closes.
4. A maintainer reviews against the invariants above.

By contributing, you agree your contributions are licensed under the project's
[MIT License](./LICENSE), and you are expected to follow the
[Code of Conduct](./CODE_OF_CONDUCT.md).

Security issues should **not** be filed as public issues — see [`SECURITY.md`](./SECURITY.md).
