# Changelog

All notable changes to `rlm-kit`. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/). Versions track
`rlm_kit/__init__.__version__` and `pyproject.toml` (kept in sync).

## [Unreleased]

The next release (0.2.0) — **not yet published to PyPI**; the version number is the
target, not a release. Harness-engineering layer, plus the first round of hardening
surfaced by dogfooding a real downstream consumer.

### Added

- **`rlm_kit.testing` — drive the RLM forward path OFFLINE (`ScriptedInterpreter` + `scripted_lm`), plus
  a `RLMTask(interpreter=…)` injection seam.** `dspy.RLM` runs the model's Python in a sandboxed
  interpreter, so the *forward* loop (planner turn → tool call → SUBMIT → validated result) normally needs
  a paid model + a Deno subprocess — which is why the kit's own tests and every consumer stop at
  `_build_rlm()` (construction) and never exercise the loop, exactly where wiring bugs hide (a prompt that
  names a tool `foo` while it registered as `foo_tool` is a `NameError` no construction test can see).
  `ScriptedInterpreter` is a `dspy` `CodeInterpreter` double that runs a fixed SCRIPT instead of executing
  model code: `dspy.RLM` injects the REAL tools onto its `.tools`, and each `execute()` runs the next
  STEP — dispatch a real tool (so its tracing runs) or SUBMIT a final result. Paired with `scripted_lm`
  (a `DummyLM` whose canned turns parse under the kit's JSON adapter) and injected via
  `RLMTask(interpreter=…)`, it drives the whole `planner → tools → result` chain with **no model, no
  Deno, no network**. `rlm_kit.testing` imports dspy LAZILY, so `import rlm_kit` / the dspy-free modules
  are untouched. It is a TEST seam: an injected interpreter OBJECT overrides `config.interpreter` and
  bypasses `sandbox.build_interpreter` (and its insecure-interpreter guard) exactly like an injected
  `DummyLM` bypasses the real model — the caller supplies the double explicitly. The default string path
  (`RLMConfig(interpreter=…)` → `build_interpreter`) is unchanged and keeps the guard. Surfaced by a
  downstream consumer whose test-strategy work found a real forward-only bug (a tool-name/prompt drift)
  that no construction test could catch — promoted here per the consumer-driven-hardening rule.
- **`interpreter="container"` — the environment interpreter: the RLM REPL runs inside an isolated
  Docker container.** The default `pyodide`/`deno` sandbox is WASM Python and cannot spawn a
  subprocess; the container interpreter runs the REPL inside a real container so the model's own
  Python can `subprocess.run(...)` natively and hold filesystem/process state across a run (one
  persistent container per run, torn down at run end) — the "environment" model of the original
  `dspy.RLM`, realized over a host↔container JSON-RPC broker (`container_interpreter.py` +
  the stdlib-only in-container `_sandbox_agent.py`, delivered via `python -c`). It is a **stronger**
  boundary than pyodide for the subprocess case, not a weaker one, and the OPPOSITE of the refused
  `local` interpreter: `--network=none` makes the stdio broker the only channel in/out (no egress),
  LM credentials never enter the container (`llm_query`/tool callbacks run host-side, only results
  cross the pipe), Linux caps are dropped, and memory/pids are capped — all from the first run. A
  per-cell watchdog bounds only *sandbox* compute (host tool time is not counted) and kills+respawns
  on a hang. Opt-in and configurable via `RLMConfig(interpreter="container", container=ContainerConfig(…))`
  / `RLM_INTERPRETER=container` + `RLM_CONTAINER_*` (image, timeout, memory, pids_limit, cpus,
  cap_drop, read_only, workdir, network); **the default stays `pyodide`**. Needs the `docker` CLI (a
  runtime check, not a Python dep — `import rlm_kit` stays dspy-free AND docker-free via a lazy import
  in `sandbox.build_interpreter`'s `"container"` branch). No trace-schema change: the broker runs
  host-side, so `tool_call`/`main_step` recording is unchanged. The `local` refusal is untouched.
- **`make_command_tool` — a traced, sync `run_command` tool over a consumer-supplied ISOLATED
  runner.** The reusable half of letting an agent run local commands the way a coding agent does:
  the kit enforces the sync contract, converts a runner failure to text the RLM reacts to, and
  records ONE `tool_call` carrying the outcome (`ok` / `exit_code` / `stdout_len` / `stderr_preview`
  / `duration_ms`) — additive payload on the existing `tool_call` event, no schema change. On success
  the model receives a `{"exit_code", "stdout", "stderr"}` dict (dspy JSON-bridges a dict into a real
  REPL value; the runner returns the typed `CommandResult`, the tool converts it — a dataclass would
  reach the model only as its unsliceable `repr`); the trace keeps only lengths + a preview, like
  `fetch_url` records size not body. The kit ships NO executor and
  picks NO isolation: `runner` is a REQUIRED injection, because a `run_command` tool executes
  model-CHOSEN commands HOST-SIDE (outside the sandbox) — a naive `subprocess.run` is the same class
  of host RCE as the refused `local` interpreter, so untrusted input demands a disposable,
  network-restricted container / VM / OS-sandbox. No allowlist primitive ships (a shell allowlist is
  routinely bypassed — `make`/`npm run` script edits, `find -exec`, `git -c`, `$(...)`, env
  injection); the optional `guard` is a shape-only pre-flight, never a security claim. The tool is
  one-shot and holds no shell state — session semantics (cwd/env/filesystem persistence) are the
  runner's contract, so a STATEFUL runner (a closure over a long-lived sandbox — `docker exec`, E2B /
  Modal / Daytona, a SWE-ReX `BashSession`) fits the same seam with no API change; a `session_id`
  payload field is the additive hook to add if a consumer ever needs model-managed sessions.
  `examples/command_runner.py` is a reference stateless *inspect* runner (fresh `--rm --network=none`
  container per call, read-only mount, in-container `timeout`, memory/pids caps).
  `make_command_tool` / `CommandResult` export from `rlm_kit.tools`. (Necessary shape, not premature:
  `dspy.RLM`'s pyodide/deno interpreter is WASM Python and cannot spawn a subprocess — verified — so
  shell execution can only come from a host-side tool.)
- **`make_json_schema_validator` — validate a parsed object against a JSON Schema (draft 2020-12).**
  The generic base for the "validate against an official, vendored, version-pinned upstream JSON
  schema" pattern: a consumer vendors the schema file + a refresh script (the provider-specific half)
  and layers its own bespoke checks on top; the kit owns only the validator wiring. Returns the
  violation messages (`"<path>: <reason>"`, truncated at `max_errors` so a huge invalid doc can't
  flood the trace) for a parsed dict — parsing stays the consumer's job, so it composes with any
  extract/parse step. `jsonschema` is an OPTIONAL dependency (`rlm-kit[jsonschema]`), imported lazily
  so `import rlm_kit` and the dspy-free `tools` package stay lean. Consumer-driven: a downstream
  consumer was hand-rolling structural gates that drift from the real upstream format; this is the
  reusable half of moving to authoritative schema validation.
- **`get_sub_lm` promoted to the public surface** (`rlm_kit.get_sub_lm`; lazy re-export, keeps
  `import rlm_kit` dspy-free). Returns the base sub-LM `configure` built — the instance a consumer
  wraps with `intercept_sub_lm` before passing as `RLMTask(sub_lm=...)`. Consumer-driven: TWO
  independent consumers were reaching into `rlm_kit.runtime.get_sub_lm` (a submodule internal) because
  the kit exposed no public way to get the configured sub-LM; per the "add a named hook, don't reach
  into a `_private` name" rule it is now that hook. Using it (vs reconstructing `dspy.LM(cfg.sub_model,
  …)`) keeps a single source of truth so the wrapped model can't drift from `configure`.
- **Public LM-injection seam + `get_config` accessor** (`configure(cfg, main_lm=…, sub_lm=…)`,
  `rlm_kit.get_config`). `configure` now accepts a pre-built `main_lm` / `sub_lm` and uses it
  verbatim instead of constructing one from config — a `dspy.utils.DummyLM` in tests, or a cached /
  custom client in production. `get_config` (lazy re-export) reads the active `RLMConfig` back.
  Consumer-driven: consumer test suites (and the kit's own) were poking private
  `rlm_kit.runtime._STATE` to inject a fake LM because there was no public path; this closes that
  reach-in. Backward-compatible (keyword-only, default `None`); no wire-format change.
- **The trace/v1 `EVENT_*` type constants are now exported** (`rlm_kit.EVENT_RUN_START`,
  `EVENT_MAIN_STEP`, `EVENT_SUB_CALL`, `EVENT_TOOL_CALL`, `EVENT_FINAL`, `EVENT_RESULT`,
  `EVENT_RUN_END`). A trace reader matches on these instead of hardcoding wire strings like `"result"`.
  Additive to `__all__`; the strings themselves are unchanged and still pinned by `test_contract.py`.
- **Reusable resolved-IP SSRF guard for the `direct`-fetch pattern** (`rlm_kit.tools.resolved_host_is_safe`
  + `parse_cidrs`). `is_safe_url` is only syntactic; the DNS-rebinding re-check (re-resolve each hop,
  refuse a private/reserved address) was left to each consumer's fetcher — and every consumer re-derived
  it. `resolved_host_is_safe(host, port, *, allow_nets=())` now ships that check ONCE, with an
  `allow_nets` carve-out (`parse_cidrs(["198.18.0.0/16"])`) for a host behind a fake-IP proxy / split-DNS
  VPN (Clash/Mihomo/Surge map every public host into the reserved `198.18.0.0/16`, which the strict
  re-check would refuse — starving the model of all fetched source). Empty `allow_nets` = unchanged
  strictness (`is_safe_url` still refuses localhost/metadata regardless). Consumer-driven: surfaced by a
  downstream `direct` fetcher refusing every host behind such a proxy.
- **`max_output_chars` is now configurable** (`RLMConfig.max_output_chars`, env
  `RLM_MAX_OUTPUT_CHARS`, default `10000` — dspy's own default, so behaviour is
  unchanged). dspy.RLM head+tail-truncates each REPL output to this many CHARACTERS
  before it enters the planner prompt — the planner never sees the omitted middle.
  Previously the knob was pinned at dspy's default; now it rides the same best-effort
  passthrough as `max_iterations` / `max_llm_calls`. (Distinct from `max_tokens`,
  which caps the model's own generation.)
- **MCP client — connect an external MCP server's tools to an RLM** (`rlm_kit.mcp.mcp_tools`,
  optional `rlm-kit[mcp]`). `with mcp_tools(server) as tools:` connects to someone else's
  [MCP](https://modelcontextprotocol.io) server (a local stdio command, or a remote streamable-HTTP
  URL), discovers its tools, and yields them as ready-to-use `dspy.Tool`s for `RLMTask(tools=…)`;
  the connection is live for the block and torn down on exit. rlm-kit is a CLIENT only (never a
  server, bundles none). The crux: the MCP SDK is async but dspy.RLM invokes tools synchronously, so
  the session runs in a dedicated background thread + event loop and each call bridges via
  `run_coroutine_threadsafe(...).result(timeout)` — dspy's own `Tool.from_mcp_tool` yields an ASYNC
  tool for `ReAct.acall`, unusable on the RLM sync path. A hung tool call trips the `timeout` and is
  cancelled (so it can't wedge the serial session); a start failure still tears the bridge down (no
  leaked thread/subprocess). Both stdio and streamable-HTTP transports are integration-tested
  against a real server. Each call records a `tool_call` (trace/v1, no schema change). MCP tools run HOST-SIDE (outside the sandbox; a stdio server is a spawned
  subprocess) — treat the server as a trusted dependency and its output as a prompt-injection
  surface. `mcp.py` lives outside the dspy-free `tools/` and `mcp_tools` is a lazy export, so
  `import rlm_kit` stays dspy/mcp-free.
- **The extension contract is now documented AND guarded** (`CLAUDE.md`, `README.md`,
  `tests/test_contract.py`) — so the next consumer builds on rlm-kit without reverse-engineering it.
  A README **"Building a consumer"** section states the 5-step recipe, the promotion rule
  (generic → kit via the base/wrap split; specific → consumer), and the rollout-vs-reward stage
  boundary; three matching CLAUDE.md hard invariants pin the contract — the trace is a VERSIONED
  `rlm-kit/trace/v1` wire format (additive-only within v1; removing/renaming/re-typing an event type,
  envelope key, or established payload field is a `v2` break), the kit produces TRAJECTORIES not
  reward (every exporter carries a `reward=` hook the downstream trainer fills), and the public
  surface is `__all__` (consumers EXTEND via subclass + base/wrap + read-the-trace; if a seam is
  missing, ADD a public hook — how `recorder_scope` / `bind_recorder_to_sub_lm` were born — never
  reach into a `_private` name). **`tests/test_contract.py`** freezes that v1 surface — SCHEMA, the
  seven `EVENT_*` strings, the recorded-event envelope, and the `export_actions` / `export_sft_turns`
  / `export_rl` record shapes + the public `__all__` — so a kit change that would silently break a
  downstream reader (a consumer's report + RL export, a consumer UI's replay) fails HERE in
  the kit's own suite, not opaquely in the consumer. (+7 tests → 148.)
- **README "Built with rlm-kit" adopters section** (`README.md`, `CLAUDE.md`). A single,
  clearly-delimited list of real, PUBLIC downstream projects built on the kit (currently
  `cve-reverser`), plus a neutral maintainer-contact line. It is an adopters list, NOT design
  coupling: the kit's mechanics, examples, API docs, and commit messages still describe consumers
  GENERICALLY, and a consumer's domain specifics still never appear elsewhere. A matching CLAUDE.md
  carve-out documents this as the ONE sanctioned exception to the vendor-neutral invariant — only a
  public consumer whose maintainer wants the association may be listed.
- **Fixed: batched lifeline escalations are now recorded** (`trace.recorder_scope` +
  `sub_lm.bind_recorder_to_sub_lm`, wired in `RLMTask.arun`; surfaced dogfooding a consumer's UI
  — a run that used `llm_query_batched` recorded ZERO `sub_call`s, so `lifeline_calls` under-counted).
  Root cause: `dspy.RLM.llm_query_batched` fans the sub-LM across a `ThreadPoolExecutor`, and a
  `ContextVar` is NOT inherited by executor worker threads (unlike an asyncio task) — so the sub-LM's
  `current_recorder()` was `None` there and the escalation went untraced (a single, same-thread
  `llm_query` was fine). `arun` now binds the active recorder to the sub-LM per run; the binding
  re-establishes the recorder ContextVar inside whatever thread the sub-LM is called from. Per-run, so
  concurrent runs sharing the base sub-LM don't cross-contaminate; dspy stores+calls `sub_lm` with no
  isinstance check, so the duck-typed proxy is a valid drop-in.

- **A FAILED run now records its trajectory too** (`RLMTask.arun`). `record_main_trajectory` ran only
  on success, so a run that exhausted the retry budget (e.g. the result never coerced into
  `output_model`) was written with ZERO `main_step`s — blind on the planner side, exactly when you most
  need to see what it did. Now the last attempt's trajectory is recorded before the error re-raises. No
  result event is recorded (there is none), so every reader still keys success off `RESULT` and the run
  stays correctly "failed" (the SFT keep-filter, complete+valid, still excludes it). Surfaced dogfooding
  a consumer UI's trajectory drawer — a failed run was unnavigable.

- **`read_skill` records a content `preview`** (a head of the skill, alongside the existing
  `result_len`), so a trace reader / replay UI can show WHAT was read, not just how long it was —
  matching how a model-tool call records its output. Inspection-only; the planner still gets the full text.

- **Live per-turn `main_step` timestamps** (`TraceRecorder.begin_main_capture` / `note_main_step`, a
  `record(ts=…)` override, and an auto-installed `_MainStepTimer` dspy parse callback in `RLMTask.arun`;
  surfaced by dogfooding a consumer UI's trajectory view). `dspy.RLM` only exposes its REPL
  trajectory on the FINAL `Prediction`, so `record_main_trajectory` stamped every `main_step` at finalize
  time (all identical) — a run was "blind mid-trajectory" for per-turn timing while tool_calls were
  already live-stamped. Now a per-turn parse callback (the only parse carrying both `reasoning` and `code`)
  captures each turn's LIVE time; `record_main_trajectory` matches it back by reasoning and backfills the
  event `ts`, keeping the full `{reasoning,code,output}` payload, `step_id`, and file order identical.
  Provably training-safe: the RL dataset and replay sort by `step_id` (never `ts`) and `elapsed_s` is
  `max(ts)-min(ts)` (main_steps are interior), so only a main_step's `ts` VALUE improves — now consistent
  with tool_calls ("when it happened"). Degrades to the old `clock()` stamp when no callback is wired (a
  replay) or the callback context can't be entered. The timer MERGES into dspy's callback list, so it
  coexists with a consumer's own callbacks (e.g. a consumer's SSE streamer).
- **`TraceRecorder` live observer** (`on_event=`, surfaced by dogfooding a consumer's UI). An
  optional callback fired (best-effort, OUTSIDE the lock) for every recorded event as it happens, so a
  consumer can stream the trajectory live. It is the correct live source for `tool_call` / `sub_call`:
  the RLM's tools run INSIDE the Deno/pyodide sandbox (the planner's REPL invokes them), which bypasses
  dspy's `on_tool` callback entirely — but the recorder sees each one. An observer exception is
  swallowed so it can never break the source-of-truth trace write.
- **Sub-LM-escalation convention** (README, surfaced by dogfooding a consumer): escalate to the
  sub-LM when a model-backed tool WALLS (repeated failures on the SAME gap) instead of circling it —
  circling a walled tool burns the iteration budget and can hit the cap unfinished; one focused sub-LM
  question often unblocks it (the sub-LM is the recovery seat). Convention in the consumer's task
  INSTRUCTIONS, not an API. A consumer can nudge its planner this way (after a few repeated tool declines on a gap), turning a
  run that would otherwise circle a stuck tool into one that escalates once and converges under the cap.
- **`make_model_tool` circuit breaker** (`max_consecutive_invalid=N`, default off; surfaced by dogfooding
  a consumer — a weak planner ignored the escalation PROMPT and hammered a model-tool dozens of
  times on an out-of-distribution input, crashing at the iteration cap). A run-scoped breaker: after N
  consecutive validator declines the next call SHORT-CIRCUITS (no model call, `ModelToolResult.circuit_broken=True`,
  empty `raw`), capping wasted calls and letting the consumer redirect the root LM (escalate / finalize)
  instead of letting it thrash. Resets on any validator-ok; an endpoint error does not count. The factory
  only FLAGS the break — the consumer owns the message + tracing (same base/wrap split). It's the
  deterministic backstop to the prompt-only sub-LM-escalation convention above.
- **Run-config-in-`run_start`-meta convention** (README corollary to the judgement-only recipe, surfaced
  by dogfooding a consumer): an OFFLINE, config-free consumer reads only what the trace records, so
  any per-run config it needs to interpret the run (the value a validator enforced, the budget a
  `hit_iteration_cap`-style metric compares against, the model roles) belongs in the `run_start` meta —
  honoring an env override end-to-end (live AND offline labels) instead of a reader guessing a hardcoded
  default; old traces lacking a key fall back gracefully. A consumer records its canonical author and
  `max_iterations` there so `rl_export` reads the real per-run values.
- **Judgement-only-SUBMIT recipe** (README, surfaced by dogfooding a consumer): the companion to
  grounded completeness, for the producer side of a model-backed tool. When a `make_model_tool` is the
  authoritative producer of an artifact, the root LM's `output_model` should carry JUDGEMENT + the
  producing tool-call's id — never the artifact bytes or a self-reported `valid` flag — and the result
  is ASSEMBLED on read (re-source the artifact verbatim from the tool-call event, derive validity from
  the validator) on the live path, re-render, AND the dataset exporters. Stops two failure modes: a
  root LM re-typing (and mangling) the tool's output, and the SFT SUBMIT turn teaching the policy to
  re-author the artifact / a self-reported `valid` lying to the keep-filter. Convention, not an API.
- **Grounded-completeness recipe** (README, surfaced by dogfooding a consumer): documents the
  agentic-RAG *sufficient-context* pattern as an RLM convention — hold a retrieved ground-truth in
  persistent REPL state, diff the generated artifact against it field-by-field, emit itemized gaps,
  and finalize only when the diff is clean. The fix for CONTENT-correctness defects a format validator
  can't see (a model self-assessing "complete" from memory ships half-right artifacts). Convention, not
  an API: it lives in the consumer's task INSTRUCTIONS and needs no new model (the main LM critiques
  cheaply against its own REPL state). A consumer uses it so the planner stops finalizing a generated artifact
  whose content only *looks* right.
- **JSON-literal REPL aliases** (`sandbox.py`, surfaced by dogfooding a consumer):
  the deno/pyodide sandbox is now constructed by the kit as a thin `PythonInterpreter`
  subclass that pre-binds `true`/`false`/`null` to `True`/`False`/`None` in the REPL
  namespace. A JSON-trained instruct model habitually writes JSON literals inside the
  Python REPL — e.g. `SUBMIT({"valid": true})` — which raised `NameError: name 'true'
  is not defined` and made the model **thrash on the identical call** (one consumer
  run burned 14/25 REPL turns on exactly this). Same isolation as dspy's own default
  interpreter; `RLMTask` now owns the interpreter's teardown (dspy only tears down one
  it built itself). A real user variable of the same name still shadows the alias.
- **Sub-LM interception hook** (`sub_lm.py`): `intercept_sub_lm` wraps a
  `dspy.LM` so the RLM's sub-LM escalations (via the built-in `llm_query` /
  `llm_query_batched`) are traced as `sub_call` events, with an optional deterministic
  validate → post-process pipeline; `model_as_tool` exposes extra models for LM-decided
  multi-model routing. *(Renamed from `make_middleware_lm` — see Changed.)*
- **Skills-as-tools** (`skills.py`): `load_skills_as_tools` surfaces a Skills
  directory to the RLM. Default `discovery="list"` gives the LM `list_skills` /
  `read_skill` (discover-then-read). `discovery="inject"` returns `read_skill`
  only, and the caller injects the catalog into the prompt itself via
  `render_skills_manifest(dir)` (or reads it structurally with `discover_skills`) —
  skipping the `list_skills` round-trip when the skill set is small and fixed.
- **Unified trajectory recording** (`trace.py`): `TraceRecorder` writes an
  append-only JSONL stream — main steps (`Prediction.trajectory`), every sub-LM
  call, every tool call — keyed by `run_id` + `step_id`. Optional Langfuse mirror.
- **Replay + datasets** (`replay.py`, `dataset.py`): reconstruct a run using
  recorded tool outputs; `export_sft_turns` / `export_rl` turn traces into training data.
- **`dataset.export_actions`** (surfaced by dogfooding a consumer): emits EVERY
  action — planner step, model-as-tool call, sub-LM escalation — as a first-class,
  `kind`-tagged RL record (so a trainer can split generator vs orchestrator data),
  with the pluggable run reward attached. `export_rl` stays planner-focused.
- **`dataset.export_sft_turns`** (surfaced by dogfooding a consumer): per-root-TURN
  SFT samples (`input = full history` SEEDED with the run's initial state from the
  `run_start` meta, `output = that turn`) — the RLM post-training recipe of arXiv 2512.24601
  (App. A: one sample per iteration, mask loss to `output`). The seed is the "first user
  input" a bare RLM trajectory lacks (the prompt lives in a REPL variable, not a chat turn).
- **`tools.make_model_tool`** (promoted from dogfooding a consumer): the generic
  "model-as-tool + validate" core — chat a secondary model, retry ONLY transient endpoint
  errors, capture thinking-mode reasoning, run a validator, return a `ModelToolResult`. Like
  the fetch / web_search bases, it picks no endpoint and templates no messages; the consuming
  project wraps it with its own `chat_fn` + validator + tool name/messages/tracing.
- **`sub_call` events now capture the escalation input** (`sub_lm.py`): the
  the intercepted sub-LM records the prompt the planner sent the sub-LM, not just its output —
  needed for RL data on escalations.
- **`trace.record_tool_call`** (surfaced by dogfooding a consumer): one helper that
  owns the `tool_call` emission — active-recorder lookup, `None`-guard, and the canonical
  `{tool, args, …}` payload shape the replay/dataset readers consume. Every tool wrapper
  (in a consumer: `fetch_url`, `web_search`, a model-tool generator,
  a validator) previously hand-rolled that boilerplate and re-derived the
  payload shape by hand — so the trace format, the replay/RL source of truth, was copied
  across each consumer instead of owned in one place. Now used internally by `model_as_tool`,
  the skills tools, and the `make_fetch_tool` / `make_web_search_tool` factories too; it
  no-ops without an active recorder, so a tool may call it unconditionally. `make_fetch_tool`
  records only the outcome (`ok` + `result_len`, or `note` on refusal/error) and NOT the
  fetched body — bulk content lands in a REPL variable, so recording it would only bloat the
  JSONL (mirrors `read_skill`); a fetcher error is caught and returned as text. `make_web_search_tool`
  is symmetric: both `ok=False` paths (empty query, searcher error) return a short reactable
  string rather than `[]` or a raised exception, so the planner gets actionable text in its
  REPL. *(trace.py, tools/)*
- **GEPA harness skeleton** (`optimize.py`): metric templates now; `compile_task`
  is a documented Phase-2 stub.
- **`ClaudeAgentLM` — run rlm-kit on a Claude Pro/Max SUBSCRIPTION (no API key), now shipped in
  the kit.** `from rlm_kit import ClaudeAgentLM` behind the opt-in extra
  `pip install "rlm-kit[subscription]"`: a `dspy.BaseLM` adapter over the official Claude Agent SDK
  (the sanctioned path for individual subscribers), injected through the existing
  `configure(main_lm=…, sub_lm=…)` seam — zero kit-core changes. Every call is a pure completion
  (`tools=[]`, `setting_sources=[]`, no agent loop), the async SDK is bridged to dspy's sync/async
  seats via a background event loop (the `mcp.py` pattern), concurrency is capped at 2 with a single
  rate-limit backoff (politeness: ordinary individual use, not batch rollouts), the kit's default
  `json` adapter's `response_format` is translated to the SDK's native schema-validated
  `output_format` (with `max_turns` headroom for its validation step), and a leftover
  `ANTHROPIC_API_KEY` fails fast so a subscription run can't silently bill API credit. Lazily
  exported (PEP 562) with the `claude-agent-sdk` import deferred to construction, so `import rlm_kit`
  stays dspy/SDK-free (the `mcp_tools` pattern). Previously lived only under `examples/` (not in the
  wheel), which forced every downstream consumer to vendor a byte-identical copy;
  `examples/claude_agent_lm.py` now shrinks to the runnable demo.
- **`run_label_bundle(runs, /, **label_fns)` — reward-free per-run LABEL surfaces** (`dataset.py`,
  public + contract-pinned). A companion MAPPER to the exporters: `{surface: {run_id: fn(events)}}`,
  where each keyword is a consumer-supplied fn turning one run's events into a dict of intrinsic labels
  (validity flags, objective metrics, a rubric's deterministic per-criterion facts) that ride BESIDE the
  trajectory records — so a downstream trainer reads ONE canonical bundle shape instead of each consumer
  re-deriving it. `runs` is positional-only so a label surface may itself be named `runs`; `reward` is a
  REFUSED surface name (rlm-kit produces trajectories, never reward — the trainer composes reward from
  these labels plus its own credit assignment). Consumer-driven: promoted from a downstream consumer's
  per-run labelling so every consumer shares one bundle shape.
- **Public multi-server MCP catalog: `McpConnection` + `McpCatalog` + `result_text`** (`mcp.py`,
  optional `rlm-kit[mcp]`). Alongside the single-server `mcp_tools(...)` convenience (one server's tools
  as self-tracing `dspy.Tool`s), the kit now exposes a MULTI-server, queryable transport for a consumer
  building a PROGRESSIVE tool surface: list servers → `load` one on demand → read its RAW MCP `Tool`s →
  `call`. `McpCatalog(specs)` manages one `McpConnection` per server — the now-public single-server bridge
  (a background-thread `ClientSession`, its async API sync-bridged), which `mcp_tools` is also refactored
  onto (behaviour unchanged). It connects eager by default (a subprocess spawn inside an async tool loop
  can hang asyncio) with `connect="lazy"` opt-in, and tears down a partial connect on failure. It returns
  RAW MCP objects (not `dspy.Tool`s) and records NOTHING — the consumer maps tools to its own shape and
  its own tool wrapper owns the `tool_call` — so it stays dspy-free. `result_text` flattens a
  `CallToolResult` to text. Consumer-driven: a downstream consumer had hand-copied the private
  single-server bridge to build a many-server catalog; this promotes the generalization so consumers drop
  the copy.

### Fixed

- **The co-dev editable overlay no longer shadows a consumer's namespace `tests/`** (dropped
  `tests/__init__.py`; guard test in `tests/test_packaging.py`). A consumer co-develops rlm-kit by
  overlaying an editable install (`uv pip install -e ../rlm-kit`), whose bare-path `.pth` puts the repo
  ROOT on the consumer's `sys.path`. Because rlm-kit shipped `tests/__init__.py` (a REGULAR package), a
  consumer's `import tests` bound to rlm-kit's `tests/` and SHADOWED the consumer's own namespace
  `tests/` — regardless of `sys.path` order (PEP 420: a regular package at any later entry beats an
  earlier namespace portion) — breaking its `from tests.conftest import ...` collection. rlm-kit's
  `tests/` is now a namespace dir (the `__init__.py` was empty; the suite is unchanged), so `rlm_kit`
  is the only regular package in the repo and the shadow is gone; a guard test keeps it that way. Wheel
  users were never affected (the wheel ships `rlm_kit` only). Note: rlm-kit and a consumer may share a
  test basename (e.g. `tests/test_config.py`) — harmless under pytest (namespace-dir tests import by
  file, not package path), but an explicit `import tests.test_config` in consumer code could resolve to
  rlm-kit's copy under the overlay; keep test basenames project-unique if that ever matters.
- **No more "Unclosed connector" warning from litellm** (`runtime.py`). litellm
  (dspy's LM backend) defaults to an aiohttp transport whose pooled `ClientSession`
  is bound to the per-run `asyncio.run` loop; when that loop closes, aiohttp logs a
  noisy "Unclosed connector" through the loop's exception handler. `RLMTask` now sets
  `litellm.disable_aiohttp_transport = True` before the first LM call, forcing litellm
  onto httpx so no aiohttp session is created and nothing dangles. Best-effort and
  idempotent — a litellm-free install just no-ops.
- **Retry logging no longer floods the terminal with a degenerate LM completion**
  (`_retry.py`). A failed attempt was logged with the caught exception's full string, and
  dspy's `AdapterParseError` embeds the ENTIRE raw LM completion in its message — so a root
  model that degenerates into a repetition loop (never emitting the expected output fields)
  dumped thousands of lines to stderr. `run_with_retry` now formats the logged exception
  through `_short_error`: the exception type + head + tail are kept (the adapter name and the
  expected/actual-fields summary survive), the middle is elided. Normal short errors still log
  in full; only a pathologically large message is capped. Consumer-driven: surfaced by a
  downstream studio whose general (non-fine-tuned) root model degenerated on a run.

### Changed / Hardened (surfaced by dogfooding a consumer)

- **`export_actions` reads a tool's output via a `raw → result → results → preview` fallback**
  (`dataset.py`, surfaced by dogfooding a consumer). A `tool_call` action's `outcome.output` read ONLY
  `payload["raw"]`, but `record_tool_call` pins no single output key and the kit's own tools disagree:
  `model_as_tool`/`list_skills` record under `result`, `read_skill` and the MCP tools under `preview`,
  `web_search` under `results`, while the `make_model_tool` consumer convention is `raw`. So an action
  record silently DROPPED the output of every tool that didn't happen to use `raw`. `export_actions` now
  reads the first present of `raw → result → results → preview` (`raw` still wins first, so existing
  traces export identically). Read-side and additive — no trace-schema change.
- **`RLMConfig.max_retries` now defaults to `1` (was `3`) — no whole-RLM retry by default** (`config.py`;
  breaking). `run_with_retry` re-runs the ENTIRE RLM on any output-coercion failure, so the old default
  of 3 silently MULTIPLIED `max_iterations` (up to 3× the turns) and re-did every fetch/search/tool
  call — breaking the budget contract a consumer + its UI rely on, while rarely fixing a PERSISTENT
  failure (same model + schema → same bad output; the dominant real failure is a TRANSIENT
  planner-endpoint hiccup, not a coercion bug). Now a run executes the RLM EXACTLY once and fails
  cleanly if it can't finalize (and, since record-on-failure, still records its trajectory). Raise
  `RLM_MAX_RETRIES` only when transient infra flakiness genuinely warrants a whole-run retry, knowing
  the budget cost.
- **`configure()` tolerates a non-owner thread/task** (`runtime.py`, surfaced by dogfooding
  a consumer's UI). `dspy.configure` is owner-locked — dspy records the first thread + async
  task to call it and raises *"can only be changed by the thread that initially configured it"* on a
  later call from a different one. A long-lived driver running each task in its own worker thread
  (a server handles per-request live runs via `asyncio.run` in a fresh thread) crashed on the 2nd run.
  The global LM config set by the first `configure` is READABLE from every thread, so on a non-owner
  thread the kit reuses it: swallow ONLY that ownership `RuntimeError` (thread or async-task variant)
  and continue; re-raise anything else.
- **Plain model ids with a custom endpoint** (`runtime.py`). When `base_url` is set,
  `configure` pins litellm's `custom_llm_provider="openai"`, so model names can be the bare id
  the endpoint serves (`qwen/qwen3-next`) instead of the misleading `openai/qwen/qwen3-next`.
  dspy.LM runs on litellm, which routes by parsing a provider out of the model string; a bare
  `qwen/...` makes it read `qwen` as the provider and fail (*"LLM Provider NOT provided"*), so
  the `openai/` prefix was a litellm routing tag — not a vendor claim — that read as if a Qwen
  model were an OpenAI one. The pin sends the id verbatim to `base_url` (matching the bare-name
  convention the raw-OpenAI-SDK generator already used); a still-prefixed `openai/...` name keeps
  working. With no base_url, write litellm's own prefix as before.
- **`RLMConfig.max_tokens` now defaults to `8192` instead of `None`** (`config.py`). With
  `None` the kit sent no `max_tokens`, so the SERVER applied its own default cap (1000 on the
  dogfooded NIM/vLLM). A **reasoning model** emits its chain-of-thought (`reasoning_content`)
  BEFORE the answer (`content`); a turn whose reasoning exceeds that small cap is truncated
  mid-thought (`finish_reason="length"`, `completion_tokens=1000`) and `content` comes back
  **empty** → dspy's "The LM returned an empty or null response", failing the run intermittently
  (only the verbose turns). This is **not** a vLLM/NIM or guided-decoding bug — it is any
  reasoning model behind any OpenAI-compatible server that caps `max_tokens` low by default.
  Shipping a generous default leaves room for reasoning + answer everywhere; set `RLM_MAX_TOKENS`
  (or `RLMConfig(max_tokens=None)`) to defer to the server. *(diagnosed by capturing per-call
  `finish_reason`/`reasoning_len`/`completion_tokens` on a telnet run; verified: 16 calls, 0 empty
  / 0 length-truncations at 16384 vs an empty at the 1000 cap.)*
- **CI/release workflows hardened for the public-repo + PyPI-publish threat model** (`.github/workflows/`).
  `ci.yml` now runs least-privilege (`permissions: contents: read` — it only checks out and tests/lints;
  specifying `permissions:` drops every unlisted scope to `none`, so a compromised action on an untrusted
  fork PR can't push, tag, or open issues) with `concurrency` cancel-in-progress. Both workflows now
  SHA-pin their third-party actions — `astral-sh/setup-uv`, and (highest blast radius, it uploads to PyPI)
  `pypa/gh-action-pypi-publish` — so a repointed tag can't inject code that runs with the token; GitHub-owned
  `checkout`/`*-artifact` stay on major tags. `release.yml`'s OIDC Trusted Publishing (no API token,
  `id-token: write` scoped to just the publish job) is untouched. *(Ported from the same hardening applied to
  a downstream consumer, itself borrowed from a public awesome-list repo's CI posture.)*
- **New `RLMConfig.adapter` (`RLM_ADAPTER`) selects the structured-output adapter; default
  `"json"` (schema-guided)** (`config.py`, `runtime.py`). Modes: `"json"` (default), `"chat"`,
  `"default"`.
  - **`"json"` drives schema-guided structured output end-to-end.** `_LenientJSONAdapter`
    makes a structured-output server constraint-decode the planner, so it emits valid output
    **even when the model formats imperfectly** — a `JSONAdapter` that **forces the `json_schema`
    response_format** (no `litellm.register_model` poke — it bypasses dspy's
    `supports_response_schema` gate directly), **removes stock dspy's `json_object` fallback**,
    AND tolerates a JSON object body emitted **without** the outer `{ }`. Stock `JSONAdapter`, when
    its `json_schema`
    attempt raises for ANY reason (incl. a transient upstream 502), falls back to bare
    `response_format={"type":"json_object"}` and re-calls — which vLLM/NIM reject with a 400
    (*"'json_object' requires a JSON schema"*) that masks the real error and burns the retry on a
    dead-on-arrival format. The lenient adapter instead always sends `json_schema` and lets a
    failure propagate (driving `ChatAdapter`'s call path, which for a `JSONAdapter` instance raises
    rather than falling back), so the task-level retry re-tries the format the server accepts; and
    it brace-wraps an unbraced object body before re-parsing (schema-guided backends intermittently
    drop the `{ }`). Works on **any** structured-output endpoint — OpenAI-proper AND vLLM/NVIDIA-NIM
    (which reject schema-less json_object but accept json_schema). New `RLMConfig.max_tokens`
    (`RLM_MAX_TOKENS`) caps per-call generation so verbose guided JSON isn't truncated mid-object.
  - **`"chat"`** → `dspy.ChatAdapter(use_json_adapter_fallback=False)`: never sends
    `response_format`; for an endpoint with NO structured-output support. The fallback is off
    because dspy's stock ChatAdapter recovers via bare `json_object`, which vLLM rejects — but
    that also means a weak model dropping a field has no recovery here, so `"chat"` is not as
    portable as it looks (it regressed an OpenAI-proper + mini-model run that `"json"`/`"default"`
    both handle).
  - **`"default"`** → leave dspy's stock adapter (ChatAdapter *with* the json_object fallback):
    recovers on OpenAI-proper endpoints, but the fallback is rejected by vLLM/NIM.
  *(Why this exists: dspy's stock ChatAdapter, on a parse error, retries through `JSONAdapter`
  and emits `response_format={"type":"json_object"}`; vLLM returns 400 "'json_object' requires a
  JSON schema" — a fronting proxy may mask it as "all channels failed". Surfaced + verified by
  dogfooding a consumer across a vLLM/NIM planner AND an OpenAI-proper gpt planner.)*
- **`make_fetch_tool` / `make_web_search_tool` are now SYNC** (were `async def`). dspy's
  interpreter invokes tools with a plain synchronous call and `str()`-serialises the result
  (`PythonInterpreter._handle_tool_call`, no `await` on `forward` *or* `aforward`), so an
  async tool returned an un-awaited coroutine — its body never ran and the model received
  `"<coroutine object …>"`. The async factories were thus unusable as RLM tools (a silent
  footgun, and why a consumer hand-rolled sync versions over the primitives). Now sync and
  directly usable in `RLMTask(tools=…)`; `fetcher`/`searcher` inputs are sync too. New
  CLAUDE.md invariant: tools passed to `RLMTask` must be sync. *(tools/fetch.py, tools/search.py)*

- **Renamed `make_middleware_lm` → `intercept_sub_lm`** (and `MiddlewareError` →
  `SubLMValidationError`). The old name hid the function's actual job: it is THE hook to
  intercept the RLM's sub-LM (dspy.RLM exposes no other — `llm_query` just calls
  `sub_lm(prompt)`), and its always-on job is `sub_call` tracing, with validate/
  post-process as opt-in. Pre-1.0 hard rename, no alias; the sole consumer
  is updated in lockstep. The module file was renamed `middleware.py` → `sub_lm.py` to match.
  *(sub_lm.py)*
- **`sub_call` payload labels its role explicitly.** Added `kind:"sub_lm"` and renamed the
  `middleware` field to `name` (the wrapper's label), so a reader/dataset sees "this is a
  sub-LM escalation" without decoding an implementation detail. `dataset.py`/`replay.py`
  read neither field, so the change is backward-compatible for the readers. *(sub_lm.py)*
- **`TraceRecorder.record` is now thread-safe.** `llm_query_batched` fans sub_lm calls
  across threads, so a wrapped sub_lm records `sub_call`s concurrently; the step-assignment
  + JSONL write now run under a lock so concurrent records can't race `step_id` or interleave
  lines (the JSONL is the replay/RL source of truth). The Langfuse mirror stays outside the
  lock. *(trace.py)*

- **`RLMTask._build_rlm` resolves custom output types deterministically.** dspy
  resolves a textual output type (e.g. `-> analysis_data: VulnerabilityReport`) by
  walking the call stack and searching each frame's globals/locals for the name.
  That happens to work when the consumer frame carrying the import is on the
  stack, but it is an implicit, call-path-dependent coupling: it raises
  `ValueError: Unknown name` for dynamically-built types or when the task is driven
  from a runner that never imported the type. `_build_rlm` now binds `output_model`
  explicitly via dspy's `custom_types=`, so resolution no longer depends on the
  call stack. (dspy silently drops `custom_types` when `instructions is None` — it
  re-parses the signature without them — so `_build_rlm` passes `""` rather than
  `None` when an `output_model` is set, keeping the binding for tasks that declared
  no instructions.) *(task.py)*
- **`RLMConfig.from_env()` falls back to `AI_MODEL_NAME` / `SUB_AI_MODEL_NAME`**
  so the kit drops into projects already keyed on those vars without re-keying
  env. `RLM_*` still wins when set. *(config.py)*
- **`configure(observe=True)` best-effort-bootstraps the Langfuse client**
  (previously only OpenInference was instrumented), so consumers don't have to
  call `get_client()` themselves. Non-fatal if Langfuse is absent. *(runtime.py)*
- **Reasoning models now work as the RLM ROOT** (surfaced by dogfooding a consumer —
  benchmarking a reasoning model as the planner). A reasoning model (qwen3 / deepseek / glm /
  gpt-oss) served over an OpenAI-compatible API sometimes emits the WHOLE structured turn
  into the `reasoning_content` channel and returns `content` (the dict's `text`) null;
  dspy's base `_call_postprocess` then raised *"The LM returned an empty or null response"*
  and the RLM died on its very first turn with zero REPL steps. `_LenientJSONAdapter._call_postprocess`
  now promotes `reasoning_content` to `text` when `text` is empty, then defers to the normal
  parse path. Guarded on `not text`, so a well-behaved model (answer in `content`, thinking in
  `reasoning_content`) is untouched and its native thinking stays discarded. This is distinct from
  the earlier `max_tokens`-truncation empty-content failure mode (that one truncates mid-thought;
  this one routes the whole answer to the wrong channel). *(runtime.py)*

### Docs

- **README split — the front page vs. the guide.** The top-level `README.md` now carries only what a
  first-time reader needs: the pitch (what/why + the declaration example), installation, a capability
  overview with a docs index, the adopters section, the security note, and develop/status. The deep
  documentation — layout, the harness-engineering layer, the tool surfaces, the rollout conventions,
  "Building a consumer", full configuration, and the offline forward-path harness — moved verbatim to
  **`rlm_kit/README.md` ("the guide")**, which GitHub renders when browsing the package folder and
  hatchling ships inside the wheel. Cross-references in `CLAUDE.md` / `CONTRIBUTING.md` now point at
  the guide; external deep links into the old top-README sections need re-pointing.

## [0.1.0]

- Initial scaffold: `RLMConfig` + `configure`, `RLMTask`, the retry/validation
  engine (`_retry.py`), the sandbox security guard (`sandbox.py`), tools (schema
  validator, SSRF-guarded fetch), examples, and tests.
