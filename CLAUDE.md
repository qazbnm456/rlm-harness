# rlm-harness — agent guide

`rlm-harness` is a small, reusable scaffold over `dspy.RLM` (Recursive Language Models)
for building tasks (of any kind). A task is a *declaration* — a `RLMTask` subclass with
a `signature`, `output_field`, optional `output_model`, `instructions`, and
`tools`; retry/validation, sandbox selection, budget caps, and observability are
inherited. See `README.md` for the overview; the full layout and usage live in
`rlm_harness/README.md` — "the guide".

One companion rule ships under `.claude/rules/`:

- `@.claude/rules/handoff.md` — what must survive context compaction, and how it routes into
  the tracked docs (invariants → this file, resolved changes → `CHANGELOG.md`). Read it before
  auto-compacting or when asked for a recap.

## Verify

- Run what CI gates on — BOTH jobs — before pushing:
  - `uvx ruff check .` — lint (ruff defaults, line-length 110). CI fails the build on any
    violation; it is NOT part of the pytest suite, so a green `pytest` is not enough on its own.
  - `uv run --group dev --extra mcp --extra grep --extra gitignore python -m pytest -q` — the
    full suite (CI runs it on 3.11/3.12/3.13). `--extra mcp` so the MCP-client tests run instead of
    skipping; `--extra grep` so `make_grep_files_tool`'s timeout tests exercise a REAL `regex`
    timeout instead of skipping (the whole point of that suite is verifying an actual timeout
    fires, not that the code merely imports); `--extra gitignore` so `list_candidate_paths`'s
    `.gitignore`-parsing tests exercise the real `pathspec` package instead of skipping. No live
    LLM, network, or Deno needed: the dspy-bearing tests use `DummyLM` or are skipped if dspy is
    absent.
- **`.github/workflows/ci.yml` also has a `packaging` job** — builds the wheel, installs it into a
  clean environment with NO lockfile, and runs a task from it. Every other job runs from the source
  tree via `uv run`, so a module missing from the wheel would ship silently. It is the PACKAGING
  axis only: an offline end-to-end run still returns the right answer while a renamed dspy kwarg
  silently drops the caller's budget cap, so it is blind to exactly the failures `_dspy_compat`
  exists for. Never let it stand in for the job below. It PINS dspy to `uv.lock`'s version on
  purpose: it runs on `pull_request`, and a floating dspy would let an upstream release redden
  a contributor's unrelated PR — which is the very reason the job below is kept off that trigger.
- **`.github/workflows/dspy-latest.yml` runs the same suite against the NEWEST published dspy —
  and since the 1.2.0 floor bump it is the ONLY dspy axis** (the floor and `uv.lock` are both on
  3.3.0, so `ci.yml` no longer covers a second version; the workflow says so in a `::notice::`
  and restores real two-version coverage by itself the day dspy 3.4 ships). Separate workflow, on a weekly cron + `workflow_dispatch` + push-to-main, never
  on a PR (an upstream break is not a contributor's problem). It exists because the two jobs above
  resolve dspy from `uv.lock`, so they test a version nobody installing from PyPI necessarily gets:
  dspy 3.3.0 renamed three things at once and the whole suite stayed green while the kit was
  completely unrunnable on a fresh install — one break loud, two silent (CHANGELOG 1.0.1). It
  resolves the newest version from PyPI at run time and ASSERTS it actually installed that; a
  hardcoded floor goes stale, and "fail if resolved == locked" false-alarms right after every lock
  bump. `--with "dspy==<exact>"` is what overrides the lock — a bare `--with dspy` (and
  `--isolated --with dspy`) resolve back to the LOCKED version and would make the job decorative.
  **When it goes red, the kit is broken for every fresh install — a release blocker, not a flake.**
  But do NOT read green as "a fresh install works": the overlay upgrades ONLY dspy (plus whatever
  transitive it forces), so everything else stays locked and a break from, say, the newest
  `pydantic` is invisible to it. Reproduce locally with
  `uv run --group dev --extra mcp --extra grep --with "dspy==<newest>" python -m pytest -q` — it leaves
  `uv.lock` untouched.
- A *live* `dspy.RLM` run needs real model credentials **and** a Deno sandbox
  (`brew install deno`). Don't run it in CI; it costs money. `examples/` show it.
- Before claiming done, actually run the two commands above and paste the output. (The
  newest-dspy workflow is NOT one of them — it needs network, and CI runs it for you.)

## Invariants — do not break

- **The sandbox is the security boundary.** Default interpreter is the sandboxed
  `pyodide`/`deno`. The `local` interpreter is host RCE and must stay **refused**
  unless `allow_insecure_sandbox=True` is explicitly set. Never weaken the guard
  in `sandbox.py`. The opt-in `container` interpreter (`container_interpreter.py`) runs the REPL
  inside an isolated Docker container so model code can spawn subprocesses — a *stronger* boundary
  than pyodide for that case (`--network=none` = no egress, LM creds stay host-side, caps dropped),
  the OPPOSITE of `local`; it is handled BEFORE the `INSECURE_INTERPRETERS` check and never routed
  through it. Keep it that way, and keep the default `pyodide`. The `RLMTask(interpreter=…)` kwarg is a
  TEST/advanced INJECTION seam (mainly `rlm_harness.testing.ScriptedInterpreter`, to drive the forward path
  offline): an explicit interpreter OBJECT overrides `config.interpreter` and bypasses `build_interpreter`
  — NOT a guard hole and NOT a weakening of the `local` refusal, but the exact analogue of an injected
  `sub_lm`/`main_lm` `DummyLM` bypassing the real model (the caller supplies and owns the double). The
  default (string → `build_interpreter`) keeps the guard; don't route the string path around it.
- **The `pyodide`/`deno` interpreter has a watchdog with TWO outcomes, and they must never be
  conflated.** `sandbox.py`'s `_build_sandboxed_interpreter`'s guarded `execute()` kills a wedged
  sandbox turn from a separate thread (mirroring `container_interpreter.py`'s own
  timer-armed-before-blocking-read idiom for the `container` kind) on either of two independent
  triggers: `RLMConfig.sandbox_turn_timeout_s` (a per-turn safety-net deadline, `None`/disabled by
  default — NOT matching `ContainerConfig.timeout_s`'s own `120.0` default, because this budget has
  no hook to exclude host-side tool/sub-LM dispatch time and would misfire more often as a result)
  or an externally-set `RLMTask(cancel_event=...)`. **A timeout raises dspy's RECOVERABLE
  interpreter error — `_dspy_compat.recoverable_interpreter_error()`, never a hardcoded class —
  (dspy's `_execute_code` catches it, the model retries next turn); a cancel raises
  `SandboxCancelled` (NOT recoverable — deliberately NOT a `CodeInterpreterError` subclass, so it
  propagates as a genuine run-ending failure).** Never make `SandboxCancelled` a
  `CodeInterpreterError` subclass, and never let a caller-driven cancel degrade into a
  retried/recoverable outcome. **Which class is "recoverable" is dspy-VERSION-DEPENDENT and must
  stay resolved, not written down:** dspy 3.3.0 inverted it — the base `CodeInterpreterError` had been
  the recoverable one, and 3.3.0 added `CodeExecutionError` for that role and made the base TERMINAL. Hardcoding the base class silently turns the safety-net timeout into a run-ending
  failure with no test going red (CHANGELOG 1.0.1). The same rule governs
  `container_interpreter.py`: its execute-path raises that are *meant* to hand the model another turn
  (execution timeout, an exception raised by the model's own code, a sandbox death mid tool-reply)
  use the resolved class; its setup/health/protocol failures keep the base class and are terminal by
  intent. `SandboxCancelled` needs no shim precisely because it stands outside dspy's hierarchy
  entirely — that is what makes it non-recoverable across every version, so keep it there.
  This distinction is only real end-to-end because
  `_retry.py`'s `run_with_retry` has a `non_retryable` allowlist and `RLMTask.arun()` passes
  `non_retryable=(SandboxCancelled,)` — without that wiring, `run_with_retry`'s own blanket
  `except Exception` would retry a cancel (transparently respawning the sandbox and restarting the
  whole trajectory from scratch) or wrap it in `RLMTaskError`, defeating the entire mechanism. Both
  knobs are `None`/unset by default and must stay a true no-op then: `execute()`'s FIRST check
  (`if self._turn_timeout_s is None and self._cancel_event is None:`) must keep calling
  `super().execute(...)` directly with no watcher thread ever created — this guard was accidentally
  dropped once during this feature's own design revision and only caught by a second adversarial
  review pass; keep it isolated and commented so it cannot be dropped silently again.
- **Every dspy API difference is resolved in `_dspy_compat.py` — one place, by introspection.**
  The kit declares only a FLOOR on dspy (`>=3.3.0` since 1.2.0) while consumers pin the KIT, so a
  consumer's fresh install picks up whatever dspy is current and the kit must survive dspy's
  renames without them noticing. The 1.2.0 floor bump deleted the 3.2.x BRANCHES but deliberately
  kept this module: its value was never "supports two versions", it is that every dspy fact lives
  at ONE introspected call site. Do NOT collapse a shim into its call site just because it now
  resolves a single answer — that is how the next rename gets to be silent again. Three
  landed at once in 3.3.0 (`RLM(interpreter=…)` → a positional arg of `forward()`/`aforward()`;
  `max_iterations` → `max_iters`; the recoverable/terminal interpreter-error inversion) and only the
  first failed loudly — see CHANGELOG 1.0.1. So: NEVER hardcode a dspy kwarg name, attribute, or
  error class at a call site, and never assert on a dspy internal (`rlm._interpreter`,
  `rlm.max_iterations`) in a test — add a shim here and a case to `tests/test_dspy_compat.py`, which
  asserts the shim's contract against the installed dspy so the next rename goes red HERE. The module
  is `_`-private and dspy-free at module top (every lookup imports dspy lazily, `lru_cache`d because
  the installed dspy cannot change mid-process). Two rules the shims encode and that must not drift:
  interpreter OWNERSHIP stays the kit's on every version (dspy shuts down only an interpreter it
  built itself, so `RLMTask._teardown_interpreter` stays correct) — which is exactly why 3.3.0's
  `interpreter_factory=` is the WRONG seam, dspy *does* shut down whatever that factory returns and
  would double-shutdown our sandbox; and a lossy fallback must be LOUD — `_build_rlm`'s `except
  TypeError` drops all three budget caps to dspy's defaults, so it `logger.warning`s rather than
  `logger.debug`s.
- **An LM error dspy itself calls non-retryable fails the task fast — except one carve-out.**
  `_dspy_compat.is_fast_fail_lm_error(exc)` mirrors dspy's own `is_retryable_lm_error`: an
  auth/billing/configuration failure, an invalid request, or an unsupported model/feature is not
  worth burning `run_with_retry`'s budget on, since every attempt re-sends the same doomed call.
  `_retry.py:run_with_retry` gained an `is_fast_fail` predicate hook alongside the existing
  `non_retryable` type allowlist — a predicate because "isinstance(exc, LMError) and NOT
  ContextWindowExceededError and NOT is_retryable_lm_error(exc)" cannot be expressed as a static
  `except (A, B, C):` tuple. A match propagates the ORIGINAL exception verbatim, consumes NO
  attempt, and is never wrapped in `RLMTaskError` — the same contract as `non_retryable`, checked
  second because a type match there is cheaper. `RLMTask.arun()` wires
  `is_fast_fail=_dspy_compat.is_fast_fail_lm_error` in; `_retry.py` itself stays dspy-free, the
  predicate is supplied by the dspy-aware caller. **The one carve-out:**
  `ContextWindowExceededError` is a non-retryable `LMInvalidRequestError` by dspy's classification,
  but that classification assumes a retry re-sends the identical request — `run_with_retry`
  instead re-runs the WHOLE trajectory, which can genuinely produce a shorter prompt that fits on
  a later attempt, so it is excluded and keeps retrying like any other exception. This was the
  contested, previously-unresolved half of the design noted in CHANGELOG 1.2.0; it is resolved
  now and must not be re-litigated by hardcoding `ContextWindowExceededError` back into the
  fast-fail set. Reached through the PUBLIC `dspy.is_retryable_lm_error`, never the private
  `dspy.utils.exceptions._RETRYABLE_LM_ERRORS` tuple it is built from — the same rule as every
  other shim in `_dspy_compat.py`. Degrades to `False` (never fast-fail, i.e. today's pre-1.2.1
  behavior) whenever `dspy.LMError` or `dspy.is_retryable_lm_error` is missing on the installed
  dspy, so a future rename fails safe rather than over-eagerly killing a retryable run.
- **Keep the dspy-free modules dspy-free.** `config.py`, `_retry.py`, `sandbox.py`,
  `tools/`, `trace.py`, `skills.py`, `replay.py`, `dataset.py`, `serving.py`, `harness_serve.py`,
  `_dspy_compat.py`
  must NOT import `dspy` at module top — that keeps their logic testable without dspy. Only
  `task.py`, `runtime.py`, `sub_lm.py` (lazily), `mcp.py`, `container_interpreter.py`,
  `testing.py`, and `claude_agent_lm.py` touch dspy — the last four live outside the dspy-free
  set and are lazily imported (by `__getattr__` / by `sandbox.build_interpreter`'s
  `"container"` branch / inside `testing.py`'s functions), so `sandbox.py`'s module top and
  `import rlm_harness` stay dspy-free. `claude_agent_lm.py` (optional `rlm-harness[subscription]`)
  additionally keeps its `claude-agent-sdk` import out of module top — deferred to instance
  construction — so `rlm_harness.ClaudeAgentLM` is gettable without the extra, like `mcp_tools`.
- **`import rlm_harness` must not import dspy.** `RLMTask` and `configure` are lazy
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
  holds for BOTH backends — dspy's Deno `PythonInterpreter._extract_parameters` AND rlm-harness's
  `ContainerInterpreter._extract_parameters` read the wrapped func's signature. So a `**kwargs`/`*args`
  param is flattened into a single proxy param literally named `kwargs`/`args` (the model can only pass
  the value under that meaningless name — a strict MCP server rejects it, a plain tool mis-binds), and a
  required param placed AFTER a defaulted one makes the generated Deno `def` a SyntaxError that aborts the
  whole registration. When a wrapper must be `def call(**kwargs)` (e.g. `mcp._make_tool`, whose params
  come from a runtime JSON Schema), stamp `call.__signature__` from the schema — required-first,
  KEYWORD_ONLY — so the proxy exposes real names. Enforce it in a test with
  `rlm_harness.testing.assert_repl_safe(tool)` (see `tests/test_repl_safety.py`, which sweeps every shipped
  factory); a consumer exposing its own tools should assert the same.
- **A tool's NAME is derived data too — sanitize it, and keep the raw name for the wire and the
  trace.** dspy validates the name at `RLM(...)` construction (identifier, not a keyword, unique
  across the task) and a failure aborts registration for EVERY tool, so one bad name silently takes
  the rest down. Any name built from data the kit does not control — an MCP server's tool name, a
  model id, a `pydantic` model's `__name__` — goes through `sanitize_tool_name` /
  `unique_tool_names`; uniqueness is a property of the SET, so resolve a whole tool list in one
  pass. All four such sites shipped broken; CHANGELOG 1.0.2. Both of those are PUBLIC since 1.1.0,
  along with the SHAPE half `signature_from_json_schema`, so a CONSUMER building its own tools
  (e.g. from `McpCatalog`'s raw names) uses the same derivation instead of writing a second one —
  and it needs BOTH halves: fixing only the name leaves a well-named `**kwargs` tool that
  `assert_repl_safe` still rejects. **Keep the three identities separate:**
  the WIRE name (what goes back to the server) and the TRACE name (`record_tool_call`) stay RAW —
  only the REPL-facing name is sanitized, with an optional `repl_name` payload field carrying the
  mapping when it differs. And `sanitize_tool_name` MUST stay a fixpoint on already-valid names,
  *including non-ASCII ones*: `str.isidentifier()` accepts them and so does dspy, so a
  `[^A-Za-z0-9_]` character class would rewrite names that work today (an all-CJK name collapses to
  `_`) — test character validity with `str.isidentifier()`, never with a character class. Note dspy
  reads `dspy.Tool(name=…)` when given, NOT `func.__name__`; sanitizing only `__name__` on a tool
  built with an explicit `name=` is a placebo (`assert_repl_safe` resolves it dspy's way and catches
  exactly that).
- **A fresh `threading.Thread` starts with an empty `contextvars.Context` — contextvars are NOT
  inherited into it.** `tools/_async.py`'s `run_isolated` (a bridging primitive for a consumer's own
  in-process harness-delegation transport — see `tools/harness.py`'s `pointer_to_invocation` and
  `examples/harness_local_run.py`) always runs its coroutine on a dedicated new thread, which means
  the SAME non-inheritance `trace.recorder_scope`'s docstring already documents for
  `dspy.RLM.llm_query_batched`'s `ThreadPoolExecutor` sub-LM workers applies here too — arguably more
  starkly (no partial context-copying at all). Concretely: a `TraceRecorder` entered AROUND a
  `run_isolated(...)` call is invisible to `current_recorder()` INSIDE the coroutine it runs, so any
  `record_tool_call`/`intercept_sub_lm` activity a delegated child triggers would go silently
  unrecorded. Any contextvar-scoped state needed inside the isolated call — notably a delegated
  child's OWN `TraceRecorder`, for its own separate rollout — must be established INSIDE the
  coroutine `run_isolated` runs, never around the call to `run_isolated` itself. Tested directly in
  `tests/test_async.py` (both the single-call isolation and a sequential-calls case, so a future
  "simplification" that reused a thread/pool across calls — leaving a recorder's `_active.set()`
  unmatched by a `reset()` on a persistent `Context` — would go red here).
- **MCP is CLIENT-ONLY, and its async SDK is bridged to sync (`mcp.py`, optional `rlm-harness[mcp]`).**
  `mcp_tools(server)` connects to an EXTERNAL MCP server (rlm-harness never IS a server, never bundles
  one — you point it at someone else's) and exposes that server's tools to `RLMTask`. The MCP SDK is
  async (`ClientSession.call_tool` is a coroutine) but RLM tools must be sync (above), so the session
  runs in a dedicated background thread + event loop kept alive for the `with` block, and each tool
  bridges one call via `run_coroutine_threadsafe(...).result(timeout)`. Do NOT reuse dspy's
  `dspy.Tool.from_mcp_tool` for this — it yields an ASYNC tool for `ReAct.acall`, unusable on the RLM
  sync path. MCP tools execute HOST-SIDE (outside the sandbox; a stdio server is a spawned
  subprocess) — treat the server as a TRUSTED dependency and its output as untrusted LM context (a
  prompt-injection surface, like `fetch_url`). `mcp.py` lives OUTSIDE `tools/` so it may import
  dspy + mcp; `mcp_tools` is a lazy `__getattr__` export so `import rlm_harness` stays dspy/mcp-free.
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
  tool-side. Full consumer-facing explanation: the guide (`rlm_harness/README.md`) "Sub-LM vs. tool".
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
  a prompt-injection surface. See the guide (`rlm_harness/README.md`) "Skills (progressive disclosure)".
- **The trace is a VERSIONED wire format — additive-only within v1.** `SCHEMA =
  "rlm-harness/trace/v1"` + the seven `EVENT_*` type strings + the `{schema, run_id, step_id, ts, type,
  payload}` envelope + the dataset-exporter record shapes are a CONTRACT that offline readers build
  on (replay, the `export_*` exporters, AND every consumer's report renderer / dataset / re-render).
  Within v1 you MAY add an optional payload field; you may NOT remove, rename, or re-type an existing
  event type, envelope key, or established payload field — that silently breaks every downstream
  reader without a test failure here. A breaking change bumps `SCHEMA` to `v2` with a migration.
  `tests/test_contract.py` pins all of this: if it goes red you are about to break a consumer, not
  the test. **The SCHEMA string itself is frozen from 1.0.0 too** — it read `rlm-kit/trace/v1` until
  the pre-publication rename to `rlm-harness` (CHANGELOG 1.0.0), which moved the vendor tag while
  leaving `v1` and the shape byte-identical. That was a ONE-TIME change, safe only because the format
  had zero external readers at the time and because every reader here and in all nine consumers keys
  off `type`, never off `schema` (audited before the change; traces predating it still read
  correctly). Do NOT treat it as a precedent: the string is now as frozen as everything else in this
  bullet, and a genuine break is still a `v2` with a migration.
- **rlm-harness produces TRAJECTORIES, never reward.** The kit runs the RLM, records the trace, and
  turns traces into datasets (`export_sft_turns` / `export_rl` / `export_actions`). It does NOT
  score them: every exporter carries a `reward=` HOOK the downstream trainer fills, and passes
  `reward=None` itself. Reward composition, credit assignment, and GRPO/SFT are a SEPARATE
  fine-tuning project — rlm-harness + its consumer are the ROLLOUT stage only. Emit raw labels/metrics;
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
  into a `_private` name. Full walkthrough: the guide (`rlm_harness/README.md`) "Building a consumer".
- **`make_harness_tool` delegates a sub-task to ANOTHER rlm-harness harness — long text IS the contract.**
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
  the parent run. (The CLIENT side; its SERVER-side mirror is `serve_harness` below. Both sides are the
  consumer guide's step 6, `rlm_harness/README.md`.)
- **`serve_harness` is the SERVER-side mirror of `make_harness_tool` — so connecting a harness needs no
  bespoke glue.** `make_harness_tool` is the CLIENT (the parent wraps a harness as a tool via an injected
  `call_endpoint`); `serving.py`'s `serve_harness(run, to_pointer, …)` + the `python -m
  rlm_harness.harness_serve <module:run>` entry are the SERVER — they turn any RLMTask harness into a process
  that speaks the contract. The kit owns ALL generic plumbing (read stdin → the child's RLM env, run_id,
  CWD isolation, the `HarnessPointer` wire, exit codes: 0=ran / 1=infra, keep the harness's identity +
  tracebacks OFF stdout). The consuming HARNESS supplies only the one thing the kit can't know — mapping
  ITS result object into a `HarnessPointer` (`to_pointer`) — in a ~5-line `serve` module in its OWN repo;
  the operator then points the client endpoint straight at `-m <harness>.serve`, no intermediate project.
  `HarnessPointer.to_json_line` flattens `meta` to top level so the wire is what the client's `read_output`
  parses. dspy-free. Anticipatory: written for a FUTURE downstream harness; the kit names none. **Worked
  example to copy: `examples/harness_serve.py`; both sides are step 6 ("Delegate to another harness — or
  be one") of the consumer guide (`rlm_harness/README.md`).**
- **Keep the public surface vendor-neutral.** rlm-harness's package, source, docs, and commit messages
  refer to downstream consumers GENERICALLY ("a consumer", "a downstream UI") — never by a specific
  project name, and never reproducing a consumer's product domain. A consumer's own concrete values
  (model names, schemas, product terms, paths) live in the consumer, not here. This keeps the kit
  decoupled from any one user and the published artifact free of third-party specifics. The ONE
  exception is a single, clearly-delimited **"Built with rlm-harness"** adopters section in the README: it
  MAY list real, PUBLIC downstream projects by name + link + a one-line description. That is an adopters
  list, not design coupling — the kit's mechanics, examples, API docs, and commit messages still describe
  consumers generically, and a consumer's domain specifics still never appear anywhere else. Only list a
  consumer that is public and whose maintainer wants the association; never a private or internal one.

## Versioning

- Keep `pyproject.toml` `[project].version` and `rlm_harness/__init__.__version__` in
  sync. On a bump, fold the release's changes into `CHANGELOG.md`.
- **Post-1.0, the public surface follows SemVer — and that RETIRES the lockstep hard rename.**
  Before 1.0.0 this kit renamed a public name with no alias and updated its consumers in the same
  breath (`make_middleware_lm` → `intercept_sub_lm` is the recorded case, and the CHANGELOG entry
  says so in as many words). That is over. The frozen surface is `__init__.__all__` + the
  `rlm-harness/trace/v1` wire format + `RLMTask`'s declaration fields, all pinned by
  `tests/test_contract.py`. **Adding** a public name is a MINOR bump. **Renaming or removing** one
  means: ship the new name, keep the old as an alias that emits a `DeprecationWarning`, note it in
  the CHANGELOG — and do not delete the alias before the next MAJOR. A `_`-prefixed name or module
  internal (`_retry`, `trace._active`) is outside the promise and may still move freely. The trace
  format keeps its own version and its own additive-only rule within v1; a break there is a `v2`
  migration, not a SemVer major on its own. If you find yourself wanting a hard rename because a
  consumer is the only caller, that is exactly the situation the rule exists for — there are nine
  of them now, and you cannot see all of their working trees.

## Consumer-driven hardening

- This kit is driven by a real downstream consumer (a task that builds on the
  scaffold, pinning the kit as a git dep — overlaid editable for local co-dev). That dogfooding is the design loop: when the consumer
  forces a workaround, log the **reusable** gap and fix it in the kit — do not special-case
  the consumer. Generic mechanics get promoted here via the base/wrap split (a generic base +
  syntactic guard + factory in the kit; the provider + tracing in the consumer); consumer-specific
  values (model names, schemas, paths) stay in the consumer, not here.
