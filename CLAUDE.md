# rlm-kit — agent guide

`rlm-kit` is a small, reusable scaffold over `dspy.RLM` (Recursive Language Models)
for building tasks (of any kind). A task is a *declaration* — a `RLMTask` subclass with
a `signature`, `output_field`, optional `output_model`, `instructions`, and
`tools`; retry/validation, sandbox selection, budget caps, and observability are
inherited. See `README.md` for the overview; the full layout and usage live in
`rlm_kit/README.md` — "the guide".

One companion rule ships under `.claude/rules/`:

- `@.claude/rules/handoff.md` — what must survive context compaction, and how it routes into
  the tracked docs (invariants → this file, resolved changes → `CHANGELOG.md`). Read it before
  auto-compacting or when asked for a recap.

## Verify

- Run what CI gates on — BOTH jobs — before pushing:
  - `uvx ruff check .` — lint (ruff defaults, line-length 110). CI fails the build on any
    violation; it is NOT part of the pytest suite, so a green `pytest` is not enough on its own.
  - `uv run --group dev --extra mcp python -m pytest -q` — the full suite (CI runs it on
    3.11/3.12/3.13). `--extra mcp` so the MCP-client tests run instead of skipping. No live LLM,
    network, or Deno needed: the dspy-bearing tests use `DummyLM` or are skipped if dspy is absent.
- A *live* `dspy.RLM` run needs real model credentials **and** a Deno sandbox
  (`brew install deno`). Don't run it in CI; it costs money. `examples/` show it.
- Before claiming done, actually run BOTH commands and paste the output.

## Invariants — do not break

- **The sandbox is the security boundary.** Default interpreter is the sandboxed
  `pyodide`/`deno`. The `local` interpreter is host RCE and must stay **refused**
  unless `allow_insecure_sandbox=True` is explicitly set. Never weaken the guard
  in `sandbox.py`. The opt-in `container` interpreter (`container_interpreter.py`) runs the REPL
  inside an isolated Docker container so model code can spawn subprocesses — a *stronger* boundary
  than pyodide for that case (`--network=none` = no egress, LM creds stay host-side, caps dropped),
  the OPPOSITE of `local`; it is handled BEFORE the `INSECURE_INTERPRETERS` check and never routed
  through it. Keep it that way, and keep the default `pyodide`. The `RLMTask(interpreter=…)` kwarg is a
  TEST/advanced INJECTION seam (mainly `rlm_kit.testing.ScriptedInterpreter`, to drive the forward path
  offline): an explicit interpreter OBJECT overrides `config.interpreter` and bypasses `build_interpreter`
  — NOT a guard hole and NOT a weakening of the `local` refusal, but the exact analogue of an injected
  `sub_lm`/`main_lm` `DummyLM` bypassing the real model (the caller supplies and owns the double). The
  default (string → `build_interpreter`) keeps the guard; don't route the string path around it.
- **Keep the dspy-free modules dspy-free.** `config.py`, `_retry.py`, `sandbox.py`,
  `tools/`, `trace.py`, `skills.py`, `replay.py`, `dataset.py` must NOT import
  `dspy` at module top — that keeps their logic testable without dspy. Only
  `task.py`, `runtime.py`, `sub_lm.py` (lazily), `mcp.py`, `container_interpreter.py`,
  `testing.py`, and `claude_agent_lm.py` touch dspy — the last four live outside the dspy-free
  set and are lazily imported (by `__getattr__` / by `sandbox.build_interpreter`'s
  `"container"` branch / inside `testing.py`'s functions), so `sandbox.py`'s module top and
  `import rlm_kit` stay dspy-free. `claude_agent_lm.py` (optional `rlm-kit[subscription]`)
  additionally keeps its `claude-agent-sdk` import out of module top — deferred to instance
  construction — so `rlm_kit.ClaudeAgentLM` is gettable without the extra, like `mcp_tools`.
- **`import rlm_kit` must not import dspy.** `RLMTask` and `configure` are lazy
  re-exports in `__init__.py` (PEP 562). Don't make them eager.
- **Resolve custom output types via `output_model`.** `RLMTask._build_rlm` passes
  the output model through dspy's `custom_types=`. dspy otherwise resolves a type
  *name* by walking the call stack's globals/locals — which works only while a
  caller frame holds the name and raises `Unknown name` for dynamic types or
  runner-driven paths. Do NOT reintroduce reliance on that call-stack resolution.
- **Tools passed to `RLMTask(tools=…)` MUST be sync.** dspy's interpreter invokes a
  tool with a plain synchronous call (`PythonInterpreter._handle_tool_call`:
  `result = self.tools[name](**kwargs)`, then `str(result)`) — there is no `await` on
  either the `forward` or `aforward` path. An `async def` tool therefore returns an
  un-awaited coroutine: its body never runs and the model receives the literal
  `"<coroutine object …>"`. So `tools/` factories (`make_fetch_tool`,
  `make_web_search_tool`, …) and their `fetcher`/`searcher` inputs are sync. Don't make
  a tool `async`; wrap an async client into a sync call yourself.
- **A tool injected into the REPL MUST expose EXPLICIT params — never `*args`/`**kwargs`.** dspy.RLM
  builds the in-sandbox tool proxy from `inspect.signature(tool.func)` (NOT `dspy.Tool.args`), and this
  holds for BOTH backends — dspy's Deno `PythonInterpreter._extract_parameters` AND rlm-kit's
  `ContainerInterpreter._extract_parameters` read the wrapped func's signature. So a `**kwargs`/`*args`
  param is flattened into a single proxy param literally named `kwargs`/`args` (the model can only pass
  the value under that meaningless name — a strict MCP server rejects it, a plain tool mis-binds), and a
  required param placed AFTER a defaulted one makes the generated Deno `def` a SyntaxError that aborts the
  whole registration. When a wrapper must be `def call(**kwargs)` (e.g. `mcp._make_tool`, whose params
  come from a runtime JSON Schema), stamp `call.__signature__` from the schema — required-first,
  KEYWORD_ONLY — so the proxy exposes real names. Enforce it in a test with
  `rlm_kit.testing.assert_repl_safe(tool)` (see `tests/test_repl_safety.py`, which sweeps every shipped
  factory); a consumer exposing its own tools should assert the same.
- **MCP is CLIENT-ONLY, and its async SDK is bridged to sync (`mcp.py`, optional `rlm-kit[mcp]`).**
  `mcp_tools(server)` connects to an EXTERNAL MCP server (rlm-kit never IS a server, never bundles
  one — you point it at someone else's) and exposes that server's tools to `RLMTask`. The MCP SDK is
  async (`ClientSession.call_tool` is a coroutine) but RLM tools must be sync (above), so the session
  runs in a dedicated background thread + event loop kept alive for the `with` block, and each tool
  bridges one call via `run_coroutine_threadsafe(...).result(timeout)`. Do NOT reuse dspy's
  `dspy.Tool.from_mcp_tool` for this — it yields an ASYNC tool for `ReAct.acall`, unusable on the RLM
  sync path. MCP tools execute HOST-SIDE (outside the sandbox; a stdio server is a spawned
  subprocess) — treat the server as a TRUSTED dependency and its output as untrusted LM context (a
  prompt-injection surface, like `fetch_url`). `mcp.py` lives OUTSIDE `tools/` so it may import
  dspy + mcp; `mcp_tools` is a lazy `__getattr__` export so `import rlm_kit` stays dspy/mcp-free.
  Each call records a `tool_call` (trace/v1, no schema change). Keep it client-only + sync-bridged.
- **The sub-LM intercept does deterministic transforms only** (validate / post-process).
  Agentic actions (external tool calls) stay LM-decided via `tools=`, so the
  decision lands in the trajectory — keeping the run an RLM and the RL data honest.
  The split is *structural*, not stylistic: a **sub-LM** (`sub_lm=`, e.g.
  `intercept_sub_lm`) is framework-invoked and is the recursion seat — reached only
  through dspy.RLM's built-in `llm_query`/`llm_query_batched` (the sole callers of
  `sub_lm`) — its output may only be touched by deterministic code, and it is recorded
  as a `sub_call`. A **tool-LM** (`tools=`, e.g. `model_as_tool`) is a leaf the main LM
  *chooses* to call, recorded as a `tool_call`. Do NOT smuggle a model-judgement (asking
  another model to grade the output) into the sub-LM intercept — that is an agentic
  decision and must be a tool. `intercept_sub_lm` is THE sub-LM interception hook (the
  only point dspy exposes); don't try to recompose it from `make_model_tool`, which is
  tool-side. Full consumer-facing explanation: the guide (`rlm_kit/README.md`) "Sub-LM vs. tool".
- **The JSONL trace is the source of truth** for replay and RL datasets. Langfuse
  is an optional mirror only; never make `dataset.py` depend on Langfuse export.
  `TraceRecorder.record` is **lock-guarded** — dspy.RLM's `llm_query_batched` fans the
  wrapped sub_lm across threads, so concurrent `sub_call`s would otherwise race
  `step_id` or interleave JSONL lines; keep the lock (the Langfuse mirror stays
  outside it). All `tool_call` emission goes through `trace.record_tool_call` so the
  canonical payload shape lives in one place — don't hand-roll `record("tool_call", …)`.
- **Skills are KNOWLEDGE-only, progressive disclosure.** `load_skills_as_tools`
  (`skills.py`) gives the LM `list_skills` (name+description) → `read_skill` (full body),
  over `SKILL.md`/`<name>.md` files with `name`/`description` frontmatter — Agent-Skills
  convention. `read_skill` returns markdown TEXT only; it must NOT execute bundled scripts
  or expand linked files (don't add silent exec — the sandbox is the only place code runs).
  Third-party skills are usable but their text becomes LM context: treat untrusted skills as
  a prompt-injection surface. See the guide (`rlm_kit/README.md`) "Skills (progressive disclosure)".
- **The trace is a VERSIONED wire format — additive-only within v1.** `SCHEMA =
  "rlm-kit/trace/v1"` + the seven `EVENT_*` type strings + the `{schema, run_id, step_id, ts, type,
  payload}` envelope + the dataset-exporter record shapes are a CONTRACT that offline readers build
  on (replay, the `export_*` exporters, AND every consumer's report renderer / dataset / re-render).
  Within v1 you MAY add an optional payload field; you may NOT remove, rename, or re-type an existing
  event type, envelope key, or established payload field — that silently breaks every downstream
  reader without a test failure here. A breaking change bumps `SCHEMA` to `v2` with a migration.
  `tests/test_contract.py` pins all of this: if it goes red you are about to break a consumer, not
  the test.
- **rlm-kit produces TRAJECTORIES, never reward.** The kit runs the RLM, records the trace, and
  turns traces into datasets (`export_sft_turns` / `export_rl` / `export_actions`). It does NOT
  score them: every exporter carries a `reward=` HOOK the downstream trainer fills, and passes
  `reward=None` itself. Reward composition, credit assignment, and GRPO/SFT are a SEPARATE
  fine-tuning project — rlm-kit + its consumer are the ROLLOUT stage only. Emit raw labels/metrics;
  let the trainer score. (A prompt/policy convention that improves rollout QUALITY is in scope —
  better rollouts ≠ reward.)
- **The public surface is `__all__`; consumers EXTEND, they don't fork.** `__init__.__all__` + the
  trace schema + `RLMTask`'s declaration fields are the API a consumer builds on; a `_`-prefixed name
  or module internal (`trace._active`, `_retry`) is private and may change without notice. A consumer
  extends three ways and only these: subclass `RLMTask` (declaration), add a tool the **base/wrap**
  way (generic base + syntactic guard + factory HERE, provider + tracing in the consumer — as
  `make_model_tool` / `make_fetch_tool` / `make_web_search_tool` / `make_harness_tool` do), and read
  results through the trace + exporters. It must NEVER fork the harness or re-implement tracing. If a consumer needs an
  internal seam the kit doesn't expose, ADD a named, documented hook here (how `recorder_scope` in
  `trace.py` + `bind_recorder_to_sub_lm` in `sub_lm.py` were born — the cross-thread sub-LM recording
  fix; both are importable public functions, though not in the top-level `__all__`) — do not reach
  into a `_private` name. Full walkthrough: the guide (`rlm_kit/README.md`) "Building a consumer".
- **`make_harness_tool` delegates a sub-task to ANOTHER rlm-kit harness — long text IS the contract.**
  The promoted "wrap a downstream harness as a tool" shape (`tools/harness.py`), a THIN reuse of
  `make_model_tool`'s retry/validate/circuit-break core plus a child-rollout LINK. Its reason to exist is
  the RLM framework's native advantage: an input field holds near-unbounded text that dspy injects as the
  Root LM's REPL ENVIRONMENT. So a `HarnessInvoke` takes ONE long-text arg and nothing else (the contract
  enforced by SHAPE), and `harness_from_endpoint` binds that WHOLE context to the downstream harness's
  long-text input field — the child then runs a FULL RLM loop (REPL + its own MCP / skills / fetch) over
  it, not a one-shot completion. TRAJECTORY SEPARATION is load-bearing: the parent records ONE leaf
  tool_call + a `child_run_id` / `child_trace` link (additive within trace/v1), while the child owns its
  OWN trace/rollout, exported separately (both reward-free). The kit ships NO transport and NAMES no
  harness — the consumer injects `call_endpoint` (subprocess / in-process / HTTP) and the harness's
  identity lives only in the consumer's runtime config, exactly as `make_command_tool` takes an injected
  `Runner`. A dead / slow / looping child degrades (`endpoint_error` / `circuit_broken`), never sinking
  the parent run. Anticipatory by design: written for a FUTURE downstream harness, no consumer yet in the
  kit.
- **Keep the public surface vendor-neutral.** rlm-kit's package, source, docs, and commit messages
  refer to downstream consumers GENERICALLY ("a consumer", "a downstream UI") — never by a specific
  project name, and never reproducing a consumer's product domain. A consumer's own concrete values
  (model names, schemas, product terms, paths) live in the consumer, not here. This keeps the kit
  decoupled from any one user and the published artifact free of third-party specifics. The ONE
  exception is a single, clearly-delimited **"Built with rlm-kit"** adopters section in the README: it
  MAY list real, PUBLIC downstream projects by name + link + a one-line description. That is an adopters
  list, not design coupling — the kit's mechanics, examples, API docs, and commit messages still describe
  consumers generically, and a consumer's domain specifics still never appear anywhere else. Only list a
  consumer that is public and whose maintainer wants the association; never a private or internal one.

## Versioning

- Keep `pyproject.toml` `[project].version` and `rlm_kit/__init__.__version__` in
  sync. On a bump, fold the release's changes into `CHANGELOG.md`.

## Consumer-driven hardening

- This kit is driven by a real downstream consumer (a task that builds on the
  scaffold, pinning the kit as a git dep — overlaid editable for local co-dev). That dogfooding is the design loop: when the consumer
  forces a workaround, log the **reusable** gap and fix it in the kit — do not special-case
  the consumer. Generic mechanics get promoted here via the base/wrap split (a generic base +
  syntactic guard + factory in the kit; the provider + tracing in the consumer); consumer-specific
  values (model names, schemas, paths) stay in the consumer, not here.
