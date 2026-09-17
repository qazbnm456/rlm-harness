# Contributing to rlm-harness

Thanks for helping improve `rlm-harness` — a small, reusable scaffold over `dspy.RLM`
(Recursive Language Models) for building tasks of any kind. This guide is the short
version; the deep design rules live in [`CLAUDE.md`](./CLAUDE.md) and the
extension contract in the guide's [**Building a consumer**](./rlm_harness/README.md#building-a-consumer).

## Development setup

```bash
uv sync --group dev

# the full suite, exactly as CI runs it — no live LLM, network, or Deno needed
uv run --group dev --extra mcp --extra grep --extra gitignore python -m pytest -q
uvx ruff check .                       # lint, a separate CI gate (not part of pytest)
```

**Pass the extras.** Each optional extra carries tests that SKIP when it is absent, so a bare
`uv run pytest` is green while leaving the MCP client, `make_grep_files_tool`'s real regex timeout,
and `list_candidate_paths`'s `.gitignore` parsing untested.

The dspy-bearing tests use a `DummyLM` or skip when dspy is absent, so the suite
runs anywhere. A *live* `dspy.RLM` run additionally needs model credentials and a
Deno sandbox (`brew install deno`; dspy requires Deno `>=2.0.0,<3.0.0`) — only `examples/`
exercise that.

**Enable the commit hooks in your clone:** `git config core.hooksPath .githooks`. They refuse a
commit that would publish a private downstream project's name, reading a denylist from
`~/.claude/private-names.txt` (or `$PRIVATE_NAMES_FILE`) — kept outside the repo on purpose, since
putting those names into a public checker would publish exactly what it exists to keep private. No
list means no check: you are told once, never blocked. CI cannot run it, so it only guards the
moment the mistake is made, which is local.

Before opening a PR: the suite is green, `ruff check` is clean, and any new
behavior has a test. CI runs the same on Python 3.11–3.13.

## The virtuous cycle — how this kit improves

`rlm-harness` is hardened by **dogfooding**: a real downstream consumer builds on the
scaffold, hits friction, and that friction becomes a fix *in the kit* so every
consumer benefits. When you find a rough edge, the question is "is this generic?"

- A **reusable mechanic** (a new tool primitive, a sandbox seam, a trace hook) is
  promoted into rlm-harness via the **base/wrap split**: the generic base + syntactic
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
- **The trace is a frozen `rlm-harness/trace/v1` wire format.** Adding an optional payload
  field is fine; removing, renaming, or re-typing an event type / envelope key /
  established field is a `v2` break. `tests/test_contract.py` pins it — if it goes red,
  you're about to break a downstream reader, not the test.
- **Keep the dspy-free modules dspy-free.** `config.py`, `_retry.py`, `sandbox.py`, `tools/`,
  `trace.py`, `skills.py`, `replay.py`, `dataset.py`, `serving.py`, `harness_serve.py`,
  `_dspy_compat.py`, `metrics.py`, `rubric.py`, `_toolname.py`, `atomic.py`, and `isolation.py`
  must not import dspy at module top, and `import rlm_harness` must not import dspy.
- **Tools passed to `RLMTask(tools=…)` must be sync, and must expose EXPLICIT params.** dspy's
  interpreter calls them with a plain `()`, so an `async def` tool returns an un-awaited coroutine
  and never runs; and dspy builds the in-sandbox proxy from the wrapped function's signature, so
  `*args`/`**kwargs` reaches the model as a parameter literally named `args`/`kwargs`. Assert it
  with `rlm_harness.testing.assert_repl_safe(tool)`, and register a new shipped factory in
  `tests/test_repl_safety.py`'s `_REPL_FACTORIES`.
- **rlm-harness produces trajectories, never reward.** The exporters carry a `reward=` hook the
  downstream trainer fills; scoring/training is a separate stage.
- **`__init__.__all__` is SemVer-frozen since 1.0.0.** Adding a public name is a minor release.
  Renaming or removing one ships the new name plus an alias that emits a `DeprecationWarning`, and
  the alias survives until the next major. The pre-1.0 hard rename — change it and fix the consumers
  in lockstep — is no longer available. `_`-prefixed names and module internals stay free to move.

## Submitting changes

1. Fork and branch from `main`.
2. Make the change with a test; keep the suite green and `ruff check` clean.
3. Open a PR describing *what* and *why*. Reference any issue it closes.
4. A maintainer reviews against the invariants above.

By contributing, you agree your contributions are licensed under the project's
[MIT License](./LICENSE), and you are expected to follow the
[Code of Conduct](./CODE_OF_CONDUCT.md).

Security issues should **not** be filed as public issues — see [`SECURITY.md`](./SECURITY.md).
