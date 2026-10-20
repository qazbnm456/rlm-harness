# Invariants: do not break

Read this before changing anything under `rlm_harness/`.

Every rule below was broken once, in production or in a downstream consumer, and each carries the
failure it came from. They are not style preferences: a rule with a `CHANGELOG` reference beside it
is there because that release had to fix it. When a rule seems to forbid the obvious approach, the
reason is in the rule.

- **The sandbox is the security boundary.** Default interpreter is the sandboxed
  `pyodide`/`deno`. The `local` interpreter is host RCE and must stay **refused**
  unless `allow_insecure_sandbox=True` is explicitly set. Never weaken the guard
  in `sandbox.py`. The opt-in `container` interpreter (`container_interpreter.py`) runs the REPL
  inside an isolated Docker container so model code can spawn subprocesses: a *stronger* boundary
  than pyodide for that case (`--network=none` = no egress, LM creds stay host-side, caps dropped),
  the OPPOSITE of `local`; it is handled BEFORE the `INSECURE_INTERPRETERS` check and never routed
  through it. Keep it that way, and keep the default `pyodide`. The `RLMTask(interpreter=…)` kwarg is a
  TEST/advanced INJECTION seam (mainly `rlm_harness.testing.ScriptedInterpreter`, to drive the forward path
  offline): an explicit interpreter OBJECT overrides `config.interpreter` and bypasses `build_interpreter`:
  NOT a guard hole and NOT a weakening of the `local` refusal, but the exact analogue of an injected
  `sub_lm`/`main_lm` `DummyLM` bypassing the real model (the caller supplies and owns the double). The
  default (string → `build_interpreter`) keeps the guard; don't route the string path around it.
- **The `pyodide`/`deno` interpreter has a watchdog with TWO outcomes, and they must never be
  conflated.** `sandbox.py`'s `_build_sandboxed_interpreter`'s guarded `execute()` kills a wedged
  sandbox turn from a separate thread (mirroring `container_interpreter.py`'s own
  timer-armed-before-blocking-read idiom for the `container` kind) on either of two independent
  triggers: `RLMConfig.sandbox_turn_timeout_s` (a per-turn safety-net deadline, `None`/disabled by
  default: NOT matching `ContainerConfig.timeout_s`'s own `120.0` default, because this budget has
  no hook to exclude host-side tool/sub-LM dispatch time and would misfire more often as a result)
  or an externally-set `RLMTask(cancel_event=...)`. **A timeout raises dspy's RECOVERABLE
  interpreter error: `_dspy_compat.recoverable_interpreter_error()`, never a hardcoded class:
  (dspy's `_execute_code` catches it, the model retries next turn); a cancel raises
  `SandboxCancelled` (NOT recoverable: deliberately NOT a `CodeInterpreterError` subclass, so it
  propagates as a genuine run-ending failure).** Never make `SandboxCancelled` a
  `CodeInterpreterError` subclass, and never let a caller-driven cancel degrade into a
  retried/recoverable outcome. **Which class is "recoverable" is dspy-VERSION-DEPENDENT and must
  stay resolved, not written down:** dspy 3.3.0 inverted it: the base `CodeInterpreterError` had been
  the recoverable one, and 3.3.0 added `CodeExecutionError` for that role and made the base TERMINAL. Hardcoding the base class silently turns the safety-net timeout into a run-ending
  failure with no test going red (CHANGELOG 1.0.1). The same rule governs
  `container_interpreter.py`: its execute-path raises that are *meant* to hand the model another turn
  (execution timeout, an exception raised by the model's own code, a sandbox death mid tool-reply)
  use the resolved class; its setup/health/protocol failures keep the base class and are terminal by
  intent. `SandboxCancelled` needs no shim precisely because it stands outside dspy's hierarchy
  entirely: that is what makes it non-recoverable across every version, so keep it there.
  This distinction is only real end-to-end because
  `_retry.py`'s `run_with_retry` has a `non_retryable` allowlist and `RLMTask.arun()` passes
  `non_retryable=(SandboxCancelled,)`, without that wiring, `run_with_retry`'s own blanket
  `except Exception` would retry a cancel (transparently respawning the sandbox and restarting the
  whole trajectory from scratch) or wrap it in `RLMTaskError`, defeating the entire mechanism. Both
  knobs are `None`/unset by default and must stay a true no-op then: `execute()`'s FIRST check
  (`if self._turn_timeout_s is None and self._cancel_event is None:`) must keep reaching
  `super().execute(...)` with no watcher thread ever created (since 1.6.0 both branches pass
  through a `perf_counter()` timing shell that stages the turn's duration for the trace: it
  creates no thread, and timing wraps the BRANCHES rather than the method precisely so this check
  stays first). This guard was accidentally
  dropped once during this feature's own design revision and only caught by a second adversarial
  review pass; keep it isolated and commented so it cannot be dropped silently again.
- **Every dspy API difference is resolved in `_dspy_compat.py`: one place, by introspection.**
  The kit declares a FLOOR **and, since 1.14.0, a CAP** on dspy (`>=3.4.1,<3.5.0`; 1.15.2
  tightened the cap to `<3.4.1` and 1.16.0 lifted it back once 3.4.1 was supported, see
  `pyproject.toml` for the two conditions that lifting had to meet; the floor moved from 3.4.0 to
  3.4.1 in 1.21.0, because 3.4.0's lazy-import proxy broke `import litellm`, and was `>=3.3.1`
  since 1.5.0 and `>=3.3.0` since 1.2.0). This sentence said `<3.4.1` and sent the reader to
  `pyproject.toml` "for why 3.4.1 is excluded" until 1.17.0, nine lines from the sibling sentence
  1.17.0 did sweep: 1.15.2 wrote it and 1.16.0 lifted the cap without coming back here, so the
  authority document stated two different caps at once. Consumers pin the KIT, so a consumer's fresh
  install used to pick up whatever dspy was current, and the kit had to survive dspy's renames
  without them noticing. **That premise is retired**: it was tested twice and broke a consumer both
  times (3.3.0 renamed three things, CHANGELOG 1.0.1; 3.4.0 deleted the interpreter seam), because
  a detector cannot protect an install that happens before anyone reads it. The cap is upstream's
  own boundary rather than a guess, since several 3.4.0 deprecations say "removal in 3.5". The
  shims still matter for exactly the same reason WITHIN the allowed range, and a cap does not make
  `dspy-latest.yml` decorative: `uv run --with "dspy==<newer>"` overrides the bound, measured, so
  that job still tests the newest dspy and still reddens the day the next minor ships, which is
  when the next floor bump is due. The 1.2.0 floor bump deleted the 3.2.x BRANCHES but deliberately
  kept this module: its value was never "supports two versions", it is that every dspy fact lives
  at ONE introspected call site. Do NOT collapse a shim into its call site just because it now
  resolves a single answer: that is how the next rename gets to be silent again. Three
  landed at once in 3.3.0 (`RLM(interpreter=…)` → a positional arg of `forward()`/`aforward()`;
  `max_iterations` → `max_iters`; the recoverable/terminal interpreter-error inversion) and only the
  first failed loudly: see CHANGELOG 1.0.1. So: NEVER hardcode a dspy kwarg name, attribute, or
  error class at a call site, and never assert on a dspy internal (`rlm._interpreter`,
  `rlm.max_iterations`) in a test: add a shim here and a case to `tests/test_dspy_compat.py`, which
  asserts the shim's contract against the installed dspy so the next rename goes red HERE. The module
  is `_`-private and dspy-free at module top (every lookup imports dspy lazily, `lru_cache`d because
  the installed dspy cannot change mid-process). Two rules the shims encode and that must not drift:
  **interpreter CREATION stays the kit's on every version, which is what keeps
  `sandbox.build_interpreter`'s guard on every pass; SHUTDOWN is dspy's for an interpreter it was
  handed, EXCEPT a caller-supplied double.** That sentence was rewritten in 1.14.0 and the rewrite
  is the lesson. It used to read "OWNERSHIP stays the kit's" and name `interpreter_factory=` as the
  WRONG seam for supplying an interpreter, because dspy shuts down whatever the factory returns.
  dspy 3.4.0 then DELETED the positional seam the kit relied on and left only the factory, so the
  guarantee had to be re-derived from a mechanism instead of restated: `_build_rlm` hands dspy a
  real factory (`_dspy_compat.interpreter_kwargs`), dspy calls it once per forward pass and shuts
  down its return value, and a CALLER-supplied interpreter goes out through `sandbox.caller_owned`,
  a view whose `shutdown()` is a no-op, so `RLMTask._teardown_interpreter` stays its single
  shutdown and `RLMTask(interpreter=…)` keeps meaning what it says. Two consequences worth keeping:
  a retry now gets a FRESH sandbox on the string path (it used to re-enter the same dirty REPL
  namespace while dspy rebuilt an empty history). **The handle for re-checking this on the next dspy
  is `dspy.predict.rlm.resolve_interpreter_factory`**: it prefers the module's own factory over
  `dspy.settings.interpreter_factory`, which is what makes a settings-level factory unable to
  substitute the runtime that executes model code while the kit always supplies one. Read that
  function, not this sentence, when the version moves. And `caller_owned` must declare the
  `CodeInterpreter` protocol members STATICALLY, because from CPython 3.12 `isinstance` against a
  `@runtime_checkable` Protocol resolves through `inspect.getattr_static` and never consults
  `__getattr__`: a purely dynamic proxy passes on the 3.11 floor and fails on 3.12/3.13, so the
  test for it sweeps `getattr_static` rather than asserting `isinstance`.
  **And the carrier that preceded it is the cautionary half:** 1.5.0 passed an
  `interpreter_factory` as a metadata CARRIER for `execution_instructions` (without it EVERY run,
  including a `container` run that can genuinely spawn subprocesses, is described to the model as
  Pyodide) on the premise that dspy never invokes it. Nothing asserted that premise. 3.4.0 invokes
  it unconditionally, so the guard fired in production rather than in CI. A premise about
  upstream's CONTROL FLOW needs a test that exercises the control flow, not a raise that documents
  the assumption: `tests/test_dspy_compat.py::test_dspy_ACTUALLY_INVOKES_the_factory_it_is_given`
  is that test. And a lossy fallback must be LOUD: `_build_rlm`'s `except
  TypeError` drops all three budget caps to dspy's defaults, so it `logger.warning`s rather than
  `logger.debug`s.
- **An LM error dspy itself calls non-retryable fails the task fast, with one carve-out.**
  `_dspy_compat.is_fast_fail_lm_error(exc)` mirrors dspy's own `is_retryable_lm_error`: an
  auth/billing/configuration failure, an invalid request, or an unsupported model/feature is not
  worth burning `run_with_retry`'s budget on, since every attempt re-sends the same doomed call.
  `_retry.py:run_with_retry` gained an `is_fast_fail` predicate hook alongside the existing
  `non_retryable` type allowlist: a predicate because "isinstance(exc, LMError) and NOT
  ContextWindowExceededError and NOT is_retryable_lm_error(exc)" cannot be expressed as a static
  `except (A, B, C):` tuple. A match propagates the ORIGINAL exception verbatim, consumes NO
  attempt, and is never wrapped in `RLMTaskError`: the same contract as `non_retryable`, checked
  second because a type match there is cheaper. `RLMTask.arun()` wires
  `is_fast_fail=_dspy_compat.is_fast_fail_lm_error` in; `_retry.py` itself stays dspy-free, the
  predicate is supplied by the dspy-aware caller. **The one carve-out:**
  `ContextWindowExceededError` is a non-retryable `LMInvalidRequestError` by dspy's classification,
  but that classification assumes a retry re-sends the identical request: `run_with_retry`
  instead re-runs the WHOLE trajectory, which can genuinely produce a shorter prompt that fits on
  a later attempt, so it is excluded and keeps retrying like any other exception. This was the
  contested, previously-unresolved half of the design noted in CHANGELOG 1.2.0; it is resolved
  now and must not be re-litigated by hardcoding `ContextWindowExceededError` back into the
  fast-fail set. Reached through the PUBLIC `dspy.is_retryable_lm_error`, never the private
  `dspy.utils.exceptions._RETRYABLE_LM_ERRORS` tuple it is built from: the same rule as every
  other shim in `_dspy_compat.py`. Degrades to `False` (never fast-fail, i.e. today's pre-1.2.1
  behavior) whenever `dspy.LMError` or `dspy.is_retryable_lm_error` is missing on the installed
  dspy, so a future rename fails safe rather than over-eagerly killing a retryable run.
- **Keep the dspy-free modules dspy-free.** `config.py`, `_retry.py`, `sandbox.py`, `tools/`,
  `trace.py`, `skills.py`, `replay.py`, `dataset.py`, `serving.py`, `harness_serve.py`,
  `_dspy_compat.py`, `metrics.py`, `rubric.py`, `_toolname.py`, `atomic.py`, `isolation.py`,
  `memory.py`, `memory_organise.py`, `memory_vectors.py` must NOT import `dspy` at module top:
  that keeps their logic testable without dspy. Only `task.py`, `runtime.py`, `sub_lm.py` (lazily), `mcp.py`,
  `container_interpreter.py`, `testing.py`, `claude_agent_lm.py` and `memory_tasks.py` touch dspy:
  the last five live outside the dspy-free set and are lazily imported (by `__getattr__` / by
  `sandbox.build_interpreter`'s `"container"` branch / inside `testing.py`'s functions), so
  `sandbox.py`'s module top and `import rlm_harness` stay dspy-free. `claude_agent_lm.py` (optional
  `rlm-harness[subscription]`) additionally keeps its `claude-agent-sdk` import out of module top:
  deferred to instance construction, so `rlm_harness.ClaudeAgentLM` is gettable without the extra,
  like `mcp_tools`.

  **`ClaudeAgentLM` is an ENGINE, not a `forward()` subclass, and the suite must stay free of that
  DeprecationWarning.** It was a `dspy.BaseLM` implementing `forward`/`aforward` until 1.17.0, on a
  documented exemption dated by the warning dspy emitted for it. dspy 3.4 deprecated that interface
  in favour of an engine with `complete(Request) -> Response` passed as `dspy.LM(engine=...)` and
  scheduled removal for 3.5, so the exemption was resolved rather than renewed: the class is now a
  `dspy.LM` holding `_ClaudeAgentEngine` / `_AsyncClaudeAgentEngine`.

  Two things that go with it, each with its own pin. The LM is **managed** now, so dspy WOULD
  consult a cache for it: `__init__` defaults `cache=False` so a subscription call keeps touching
  none, because an injected LM keeps its own flag (1.15.1) and the kit's own default could not
  reach it. And a **copy must keep its engine**, or `copy_lm`'s `rollout_id` variant silently calls
  litellm; `tests/test_dspy_compat.py` pins that, replacing the pin that used to assert the
  ABSENCE of `_engine_spec`.

  The instrument is a TEST, not a habit:
  `tests/test_dspy_compat.py::test_the_kits_own_LMs_are_off_the_interface_dspy_removes_in_3_5`
  drives a `forward()`-style subclass first and demands it raise, THEN demands the kit's own LMs
  raise nothing. **Both DIRECTIONS, because dspy decides "legacy" per direction:**
  `dspy/clients/execution.py` computes `method = "aforward" if asynchronous else "forward"` and
  asks whether the LM overrides that one. The first version of this test drove only the sync
  `__call__`, which left `aforward` free to come back on `ClaudeAgentLM` with the suite green and
  dspy warning: it guarded the sub-LM seat and missed the planner seat, which is the primary RLM
  route. Without that control the second half would pass for an LM that is on the dead
  interface, and relying on remembering to pass `-W error::DeprecationWarning` is not an
  instrument: nothing in CI passes it, so a future edit reintroducing `forward()` would be green.
  `dspy-latest.yml` remains what goes red when 3.5 actually ships; the cap still cannot pass
  `<3.5.0` (see `pyproject.toml`) until then. `intercept_sub_lm`'s wrapper is NOT on the deprecated
  interface and needs no migration: it overrides `__call__`, not `forward`, and the same test
  covers it.
- **`ClaudeAgentLM`'s politeness ladder rests on dspy's error CLASSIFICATION, and that dependency
  is a borrowed table.** `dspy/clients/execution.py` computes
  `retries = lm.num_retries if call.managed else 0`. While the class was unmanaged that budget was
  structurally zero and the adapter's own single rate-limit retry was the whole ladder. Since
  1.17.0 it is managed and `num_retries` defaults to 3, so the budget is LIVE; what still keeps a
  personal subscription from paying four CLI spawns per failed call is only that the adapter's
  `RuntimeError`/`TimeoutError` surface as `LMUnexpectedError`, which is absent from dspy's
  `_RETRYABLE_LM_ERRORS` (measured: `is_retryable_lm_error` → `False`, one SDK call for one failed
  call). Nobody chose that; it is upstream's table.
  `tests/test_claude_agent_lm.py::test_dspys_retry_budget_is_live_but_never_multiplies_an_sdk_call`
  makes it visible, with its control inside it (a zero `num_retries` or an unmanaged LM would make
  the test pass for a build where the hazard does not exist). If it goes red, cap `num_retries` on
  the LM; do not relax the assertion. The module docstring's "One retry ladder, smallest at each
  rung" is the promise this holds up, and it is about someone's personal usage window, not a
  performance knob.

  **`n` is the second thing that became dspy's to multiply, and the first sweep missed it.**
  `execute` runs `count = call.n` engine calls once the request is canonical, where an unmanaged
  LM took its hardcoded `n=1` branch and dropped the kwarg inside `forward()`. Measured across the
  migration: `n=3` was ONE CLI spawn on 1.16.0 and is three now. The constructor warns above 1
  rather than refusing, because a caller asking for k candidates correctly gets k; the warning is
  for an `n` carried over from a litellm config on the belief this adapter ignores it, which is
  what the docstrings said. When reading this invariant for a NEW dspy behaviour, enumerate what
  `managed` turns on rather than checking the one knob that prompted the question: retries and `n`
  arrived together and only one was noticed.
- **`import rlm_harness` must not import dspy.** `RLMTask` and `configure` are lazy
  re-exports in `__init__.py` (PEP 562). Don't make them eager.
- **Resolve custom output types via `output_model`.** `RLMTask._build_rlm` passes
  the output model through dspy's `custom_types=`. dspy otherwise resolves a type
  *name* by walking the call stack's globals/locals, which works only while a
  caller frame holds the name and raises `Unknown name` for dynamic types or
  runner-driven paths. Do NOT reintroduce reliance on that call-stack resolution.
- **Tools passed to `RLMTask(tools=…)` MUST be sync.** dspy's interpreter invokes a tool with a
  plain synchronous call: `PythonInterpreter.invoke_tool` is
  `result = self.tools[name](**kwargs)`. **The consequence changed in dspy 3.4.0 and the rule did
  not, which is worth stating so a reader who checks the old mechanism and finds it false does not
  conclude the rule is obsolete.** 3.4.0 appends
  `return _await_in_sync(result) if asyncio.iscoroutine(result) else result`, and `_await_in_sync`
  calls `asyncio.run` when there is no running loop and `loop.run_until_complete` when there is. The
  kit always reaches the sandbox from inside a running loop, since `RLMTask.arun()` is async and
  `run()` wraps it, so an `async def` tool now raises
  `RuntimeError: This event loop is already running` rather than handing the model an un-awaited
  `"<coroutine object …>"`. Loud instead of silent, and still a broken tool. So `tools/` factories
  (`make_fetch_tool`, `make_web_search_tool`, …) and their `fetcher`/`searcher` inputs are sync.
  Don't make a tool `async`; wrap an async client into a sync call yourself.
- **A tool injected into the REPL MUST expose EXPLICIT params: never `*args`/`**kwargs`.** dspy.RLM
  builds the in-sandbox tool proxy from `inspect.signature(tool.func)` (NOT `dspy.Tool.args`), and this
  holds for BOTH backends: dspy's Deno `PythonInterpreter._extract_parameters` AND rlm-harness's
  `ContainerInterpreter._extract_parameters` read the wrapped func's signature. So a `**kwargs`/`*args`
  param is flattened into a single proxy param literally named `kwargs`/`args` (the model can only pass
  the value under that meaningless name: a strict MCP server rejects it, a plain tool mis-binds), and a
  required param placed AFTER a defaulted one makes the generated Deno `def` a SyntaxError that aborts the
  whole registration. When a wrapper must be `def call(**kwargs)` (e.g. `mcp._make_tool`, whose params
  come from a runtime JSON Schema), stamp `call.__signature__` from the schema: required-first,
  KEYWORD_ONLY, so the proxy exposes real names. Enforce it in a test with
  `rlm_harness.testing.assert_repl_safe(tool)` (see `tests/test_repl_safety.py`, which sweeps every shipped
  factory). Register a NEW factory in that file's `_REPL_FACTORIES` table, not only in the new tool's own
  test file: the table is hand-built because every factory takes different arguments, but
  `test_sweep_covers_every_shipped_repl_factory` fails the moment a `make_*` reaches
  `rlm_harness.tools.__all__` without an entry there (or an argued exemption in `_NOT_REPL_FACTORIES`).
  Six factories shipped in 1.3.0 before that guard existed and none reached the sweep. Each was safe,
  but only because its author happened to remember. A consumer exposing its own tools should assert the same.
- **A tool's NAME is derived data too: sanitize it, and keep the raw name for the wire and the
  trace.** dspy validates the name at `RLM(...)` construction (identifier, not a keyword, unique
  across the task) and a failure aborts registration for EVERY tool, so one bad name silently takes
  the rest down. Any name built from data the kit does not control: an MCP server's tool name, a
  model id, a `pydantic` model's `__name__`. Goes through `sanitize_tool_name` /
  `unique_tool_names`; uniqueness is a property of the SET, so resolve a whole tool list in one
  pass. All four such sites shipped broken; CHANGELOG 1.0.2. Both of those are PUBLIC since 1.1.0,
  along with the SHAPE half `signature_from_json_schema`, so a CONSUMER building its own tools
  (e.g. from `McpCatalog`'s raw names) uses the same derivation instead of writing a second one,
  and it needs BOTH halves: fixing only the name leaves a well-named `**kwargs` tool that
  `assert_repl_safe` still rejects. **Keep the three identities separate:**
  the WIRE name (what goes back to the server) and the TRACE name (`record_tool_call`) stay RAW:
  only the REPL-facing name is sanitized, with an optional `repl_name` payload field carrying the
  mapping when it differs. And `sanitize_tool_name` MUST stay a fixpoint on already-valid names,
  *including non-ASCII ones*: `str.isidentifier()` accepts them and so does dspy, so a
  `[^A-Za-z0-9_]` character class would rewrite names that work today (an all-CJK name collapses to
  `_`): test character validity with `str.isidentifier()`, never with a character class. Note dspy
  reads `dspy.Tool(name=…)` when given, NOT `func.__name__`; sanitizing only `__name__` on a tool
  built with an explicit `name=` is a placebo (`assert_repl_safe` resolves it dspy's way and catches
  exactly that).
- **A tool's DOCSTRING is prompt text, not documentation.** dspy takes `dspy.Tool.desc` from a
  tool's `__doc__` with no opt-in, renders one line per tool in `RLM._format_tool_docs`, appends
  the block to `generate_action.signature.instructions`, and the adapter puts that string in the
  SYSTEM message. dspy's own RLM guide says so in one sentence; nobody here had read it, and no
  upstream page states the `__doc__` mechanism. CHANGELOG 1.18.0, and `docs/PROMPT.md` for the
  whole picture.

  Measured on a two-tool kit task, where "instruction string" means
  `generate_action.signature.instructions` and "the tool-doc block" means dspy's
  `Additional tools available` header through the end of it:

  * The block was **about two thirds of the instruction string**, roughly 5000 characters
    of 7800, inside a system message of about 9000 whose user-message counterpart was a few
    hundred.

  * Of the **fourteen** tools the kit shipped then, **three** had no model-facing docstring at all, so
    dspy rendered the literal words `No description`. Of the eleven that had one, **eight**
    carried Sphinx markup and **ten** produced a mid-sentence gap or markup, because dspy flattens
    with `.replace("\n", "  ")` and a docstring's 8-space continuation indent becomes ten spaces.

  * **Two tools were four fifths of everything the kit said to a model**: `verify_quote`
    around 7000 characters and `grep_files` around 3800, out of some 13000 across all
    fourteen, re-sent on every turn of every run.

  Normalising at the seam and moving the developer halves out cut that task's instruction
  string by about a third, and the kit's share of it, the block plus the interpreter's
  `execution_instructions` fragment and NOT the task's own instructions, from roughly two
  thirds to under a half. Correctness cost characters back on the way, roughly a quarter of the
  catalogue, because FIVE descriptions had to be rewritten against the code. **A shorter
  instruction that misstates a return value is worse than a longer one.**

  **The exact per-tool numbers live in `tests/tool_description_baseline.json`, where a test
  compares them to the code on every run, and the figures above are rounded on purpose.** A
  number measuring something a change can edit does not belong in prose: the precise version
  of this paragraph went stale twice in one review, once silently, because a description grew
  by less than its own allowance.

  Four rules follow.

  * **Write the INNER closure's docstring for the model and the FACTORY's for the developer.** Only
    the inner one reaches the prompt, which is why a factory can carry version history and
    measured percentages safely. A bare public function handed over directly, `verify_quote`, has
    no factory to hide behind, so its developer half lives in the MODULE docstring.

  * **Never state a number a factory argument can change.** Build it from the argument, as
    `make_web_search_tool` does, and clamp it where the implementation can exceed it.

  * **Every fact a CALLER acts on survives the cut, and is copied from the code rather than from
    memory.** Shortening `verify_quote` dropped the one thing its old text had flagged as
    caller-critical, that a coordinate-verified MATCH returns neither an offset nor a snippet, and
    replaced it with a sentence saying the opposite. Sentinel strings, refusal conditions and
    model-settable parameters are the cut's blast radius.

  * **When a tool needs more than its baseline to be USED correctly, the surplus is
    developer-facing by definition.** Raising the gate is how the system message grows back.

  The normalisation is applied ONCE, at `RLMTask._build_rlm` via `with_model_facing_description`,
  for the same reason `_ensure_tool_timing` is: a rule every author must remember is missing for
  someone, a consumer whose tools delegate to these factories has no seam of its own, and MCP's
  description belongs to a remote server no docstring convention here can reach. Whitespace
  collapses BEFORE the markup substitutions, which is what lets them see a construct wrapped
  across two source lines. It rewrites only objects the kit owns, a wrapper marked `__rlm_timed__`
  or a pydantic copy of a `dspy.Tool`, never a caller's, and it therefore does not reach a
  `functools.partial`, a callable instance, a coroutine function, a `staticmethod` object or an
  `lru_cache` wrapper.

  `tests/test_tool_descriptions.py` is the pin. What it asserts:

  * A non-empty, markup-free, gap-free description for every tool the kit ships.

  * That every sentinel string a description quotes is one the code really returns, plus that a
    quoted EXAMPLE is a prefix of the real return and that each number in a description comes
    from the constant or factory argument it describes. The sentinels are ENUMERATED out of each
    rendered description rather than listed, so a new one arrives already needing a row, and the
    sweep's coverage is asserted against the catalogue because a hand-written table drifts from it
    in silence and did.

  * **The loop has three sides and two are asserted.** Quoted-in-a-description to
    returned-by-the-code, and in-a-row to named-in-the-description. The third,
    returned-by-the-code to quoted-in-a-description, is NOT asserted, which is why three tools sit
    in `_NO_SENTINEL` returning a sentinel their descriptions never name. Those are **exempted,
    not covered**: each exemption reason names the string it is blind to, and a test keeps that
    naming true, so the gap is recorded rather than invisible.

  * **The enumeration reads QUOTED SPANS, which bounds it twice over.** Three versions of the rule
    each missed the form not yet visible in the repo when it was written, and one could be
    disarmed by the same edit it policed; the forms it now covers are listed as data in
    `test_every_form_the_kit_writes_a_sentinel_in_is_enumerated`. But the larger blindness is not
    the form: a sentinel a description states WITHOUT quoting it is invisible whatever its shape,
    and no widening of the form list reaches that.

    The way out is to enumerate from the CODE, which is a closed set: an `ast` walk over
    `tools/*.py` collecting the literal head of every returned string needs no prose parsing, and
    would have caught the defect that shipped from the other side. Measured, it finds around ten
    returned prefixes that no description names at all, three of them success messages. That is
    the open follow-up and it is a test rather than a design change. The nearest precedent for
    `ast` in this kit is `rlm_harness/testing.py`'s mirror of dspy's own signature parsing, which
    is a shipped helper rather than a verified claim; the scan would be the first of those.

    Its own bound, stated with it: only a return whose head is an inline literal is visible, which
    was 41 of 137 returns under `tools/` when it was measured, before the memory tools added
    more.

  * That the seam reaches the three producers that come from no factory, asserted end to end on a
    built task rather than as a property of their docstrings. The SIZE gate covers every
    catalogue tool and not those three.

  * That the seam is wired in rather than merely correct, and that a caller's own object is never
    rewritten.

  * Size, as a checked-in per-tool BASELINE rather than a cap: an aggregate cap measures a task
    that never ships, and a per-tool cap is a number nobody can guess for a tool that does not
    exist yet.

  Several assertions carry a control that fails when they stop measuring anything. **The TOOL
  table is deliberately exhaustive rather than an exemption list**, because an exemption list is
  what a future contributor adds themselves to; the sentinel sweep's `_NO_SENTINEL` is the one
  place an exemption is allowed, and it is guarded by a control that reds when an exempted tool
  grows a sentinel.

- **A prompt-visible string is a measurement target, not a judgement call.** The reason the
  docstring rule above went unnoticed through every release before this one is that nothing
  printed the system message. Before claiming a prompt change helps, render the actual string the
  model receives and diff it; before claiming an optimizer helped, check that the baseline had
  headroom at all. A GEPA feasibility study on this repo scored its own 10-example trainset at a
  mean of 1.000 on the shipped instructions: no gradient existed, and a slightly noisier metric
  would have produced an improvement number that was pure noise. Cost varied 3.3x in turns and
  13.5x in wall clock across those same perfect-scoring runs, so the headroom was real and the
  metric was pointed at the wrong axis.

  **That diagnosis has since been measured rather than argued.** Re-pointed, with correctness as a
  hard GATE scoring zero and turn count as the gradient, the same ten examples score a mean of
  0.700 with a spread from 0.30 to 1.00, and a held-out set the optimizer never sees scores 0.513.
  So a saturated metric is a property of the metric and not of the task, and the gate that refuses
  to spend a budget at the ceiling is what makes the difference visible.

  **A number says whether to keep a compiled artifact; only reading the artifact says what you
  are keeping.** One GEPA run did two things at once: it raised a held-out set, and it wrote four
  of the ten trainset answers into the instruction. The score shows the first and is blind to the
  second. `docs/PROMPT.md` has the run, the table, and the substring check that called five of
  seven guardrails deleted when only one of them was.
- **A fresh `threading.Thread` starts with an empty `contextvars.Context`: contextvars are NOT
  inherited into it.** `tools/_async.py`'s `run_isolated` (a bridging primitive for a consumer's own
  in-process harness-delegation transport: see `tools/harness.py`'s `pointer_to_invocation` and
  `examples/harness_local_run.py`) always runs its coroutine on a dedicated new thread, which means
  contextvars are not inherited into it. **That is a CONTRAST with
  `dspy.RLM.llm_query_batched`'s `ThreadPoolExecutor` sub-LM workers, not an analogue**: dspy submits
  those with `contextvars.copy_context().run`, so they DO see the caller's context, which is exactly
  why `bind_recorder_to_sub_lm` works there and why nothing equivalent rescues a bare thread here. Concretely: a `TraceRecorder` entered AROUND a
  `run_isolated(...)` call is invisible to `current_recorder()` INSIDE the coroutine it runs, so any
  `record_tool_call`/`intercept_sub_lm` activity a delegated child triggers would go silently
  unrecorded. Any contextvar-scoped state needed inside the isolated call, notably a delegated
  child's OWN `TraceRecorder`, for its own separate rollout. Must be established INSIDE the
  coroutine `run_isolated` runs, never around the call to `run_isolated` itself. Tested directly in
  `tests/test_async.py` (both the single-call isolation and a sequential-calls case, so a future
  "simplification" that reused a thread/pool across calls: leaving a recorder's `_active.set()`
  unmatched by a `reset()` on a persistent `Context`. Would go red here).
- **MCP is CLIENT-ONLY, and its async SDK is bridged to sync (`mcp.py`, optional `rlm-harness[mcp]`).**
  `mcp_tools(server)` connects to an EXTERNAL MCP server (rlm-harness never IS a server, never bundles
  one: you point it at someone else's) and exposes that server's tools to `RLMTask`. The MCP SDK is
  async (`ClientSession.call_tool` is a coroutine) but RLM tools must be sync (above), so the session
  runs in a dedicated background thread + event loop kept alive for the `with` block, and each tool
  bridges one call via `run_coroutine_threadsafe(...).result(timeout)`. Do NOT reuse dspy's
  `dspy.Tool.from_mcp_tool` for this: it yields an ASYNC tool for `ReAct.acall`, unusable on the RLM
  sync path. MCP tools execute HOST-SIDE (outside the sandbox; a stdio server is a spawned
  subprocess): treat the server as a TRUSTED dependency and its output as untrusted LM context (a
  prompt-injection surface, like `fetch_url`). `mcp.py` lives OUTSIDE `tools/` so it may import
  dspy + mcp; `mcp_tools` is a lazy `__getattr__` export so `import rlm_harness` stays dspy/mcp-free.
  Each call records a `tool_call` (an existing event type, no schema change). Keep it client-only + sync-bridged.
- **The sub-LM intercept does deterministic transforms only** (validate / post-process).
  Agentic actions (external tool calls) stay LM-decided via `tools=`, so the
  decision lands in the trajectory: keeping the run an RLM and the RL data honest.
  The split is *structural*, not stylistic: a **sub-LM** (`sub_lm=`, e.g.
  `intercept_sub_lm`) is framework-invoked and is the recursion seat: reached only
  through dspy.RLM's built-in `llm_query`/`llm_query_batched` (the sole callers of
  `sub_lm`). Its output may only be touched by deterministic code, and it is recorded
  as a `sub_call`. A **tool-LM** (`tools=`, e.g. `model_as_tool`) is a leaf the main LM
  *chooses* to call, recorded as a `tool_call`. Do NOT smuggle a model-judgement (asking
  another model to grade the output) into the sub-LM intercept: that is an agentic
  decision and must be a tool. `intercept_sub_lm` is THE sub-LM interception hook (the
  only point dspy exposes); don't try to recompose it from `make_model_tool`, which is
  tool-side. Full consumer-facing explanation: the guide (`rlm_harness/README.md`) "Sub-LM vs. tool".
- **"Recorded as a `sub_call`" is AUTOMATIC since 1.7.0, and it had to become automatic.** Before
  that the event existed only if the consumer remembered `intercept_sub_lm`; a plain `dspy.LM` was
  called by dspy directly and recorded NOTHING. Surveyed across the fleet, four consumers never
  wrapped, and two of those had corpora, 141 traces, in which `sub_call` was identically zero and
  therefore indistinguishable from "measured, and the model never escalated". That ambiguity got
  into a design decision in this repo (the sPTC deferral cited the zero as evidence of no
  escalation; the real rate, re-derived from code content, was 0.15%). **An absent event is not a
  measurement**: before reading a zero count as a zero rate, check that the thing which emits the
  event was wired at all. `RLMTask` now wraps a plain `sub_lm` via `sub_lm._ensure_sub_call_recording`
  at the SAME seam that binds the recorder, with the bind OUTERMOST (it establishes `recorder_scope`;
  the interceptor reads `current_recorder()` at call time, so reversing them makes the record
  silently vanish). Do this at the per-run TASK seam, never in `configure`: consumers call
  `intercept_sub_lm(get_sub_lm(), …)`, so wrapping there would double-record for exactly the
  consumers who did it right. **A consumer with its OWN recording wrapper opts out by declaring
  `records_sub_call = True`** on it; the probe is `is True`, never truthiness, because `getattr` on
  a mock manufactures a truthy attribute for any name and would silently skip it: the same
  absent-event failure one layer up.
- **Before believing a count, ask what the reader would report if the writer were perfect, and
  whether the thing being counted is still moving.** The `sub_call` bullet above is one instance of
  a failure that has now cost this project and its consumers TEN times, and the general form is
  cheaper to check than any of the specific ones. Two questions, in order:
  1. **What number would this reader return if everything it measures were working?** If the answer
     is "zero either way", the reader is disqualified before any data is collected: no experiment
     needed. Four recorded cases: `sub_call` recorded nothing because nobody wrapped the LM; `ps`
     was absent from a container image so "0 processes" meant "no tool"; `pgrep -f` matched its own
     command so "2 running" meant "none"; a duration-coverage figure read as "83% of calls are
     untimed for an unknown reason" was **entirely** the share of a corpus predating the release
     that added the field: 111 runs at 0%, 55 at 100%, zero partial. **A rate that varies by
     "tool", "task" or "kind" is worth suspecting of being corpus composition first.** The
     signature: a component added AFTER an upgrade reads 100% while everything older reads ~15%.
  2. **Is the population still changing while I describe it?** The fifth case is the one where
     nothing was broken: two readers, both correct, both counts right at their moment, and a
     40-event discrepancy that was a job still writing traces between the two sentences. No check on
     the reader catches it. Quote a count WITH the corpus size and the moment, never a bare rate.

  **A null that is too clean is evidence about the instrument before it is evidence about the
  world**, and the tell is believability: "zero EVENTS" is a different claim from "zero DURATIONS",
  and the first should be disbelieved first. A related trap has no self-check at all: **a rule
  indexed by the SITUATION that triggers it will not fire in a situation that looks different, even
  when the underlying fact is identical.** Two notes already said the corpus is version-mixed and
  named `run_start.rlm_harness` as what separates the cohorts: a consumer's ("don't bucket traces by
  the version field when you mean 'this index'") and this project's own memory ("never average a rate
  across the upgrade boundary"). Both were known, both were correct, and NEITHER was retrieved:
  because "why does duration coverage vary by tool" presents as a question about TOOLS, and both
  rules are filed under "when you are bucketing by version". **Note which one missed harder: the
  memory file whose ENTIRE SUBJECT is the upgrade boundary, held by the reader who then suggested
  querying the wrong axis.** So "write it down more prominently" is not the remedy. It was already
  a whole note about exactly this. The fix cost one query once the right
  field was suggested from outside. **The fact was never missing; the retrieval was.** So when a rate
  varies by some dimension, enumerate what else differs between the groups before treating the
  dimension as the cause.
- **A sub-LM wrapper hands dspy back the SHAPE dspy handed it.** `RLM._query_lm` accepts dspy's
  TYPED response or the legacy `list[str | dict]`, and both reads and both rebuilds live in
  `_dspy_compat.sub_lm_response_text` / `sub_lm_response_with_text`: never at a call site. The
  wrapper used to collapse anything non-list into `[outputs]`, which turned a typed response into
  `[response]` and made dspy raise; invisible on the default path, fatal under
  `dspy.context(experimental=True)`. **The typed CLASS and its LAYOUT have both moved**, which is
  why neither is written down: 3.4.0 deleted `dspy.LMResponse` (a LIST of `.outputs`, pydantic) for
  `dspy.lm15.Response` (one `.message`, a frozen dataclass), so `_lm_response_cls` resolves the
  class dspy's own `isinstance` names, the rebuild probes the container off the OBJECT rather than
  assuming either layout, and `_copy_with` probes the copy-with-changes protocol instead of calling
  `model_copy`. Three rules this path encodes: a shape the shim does not recognise is returned
  UNTOUCHED by its caller (`sub_lm.py`, not the shim, which ends `return [text]`) so dspy raises its
  own error (rebuilding it as `[""]` converts a loud failure into a silent empty
  completion that reaches the planner and the RL data); substituting text drops the response's
  LATER text parts because its `.text` JOINS them (with `""` on 3.3, `"\n"` on 3.4, so a test must
  not assert the separator); and a text part is discriminated on its `type` tag, NEVER on
  `hasattr(part, "text")`, because a thinking part carries a `.text` field of its own on 3.4.0.
- **The JSONL trace is the source of truth** for replay and RL datasets. Langfuse
  is an optional mirror only; never make `dataset.py` depend on Langfuse export.
  `TraceRecorder.record` is **lock-guarded**: dspy.RLM's `llm_query_batched` fans the
  wrapped sub_lm across threads, so concurrent `sub_call`s would otherwise race
  `step_id` or interleave JSONL lines; keep the lock (the Langfuse mirror stays
  outside it). All `tool_call` emission goes through `trace.record_tool_call` so the
  canonical payload shape lives in one place: don't hand-roll `record("tool_call", …)`.
- **Skills are KNOWLEDGE-only, progressive disclosure.** `load_skills_as_tools`
  (`skills.py`) gives the LM `list_skills` (name+description) → `read_skill` (full body),
  over `SKILL.md`/`<name>.md` files with `name`/`description` frontmatter: Agent-Skills
  convention. `read_skill` returns the file's text and nothing else, which
  `tests/test_skills.py::test_read_skill_returns_the_file_and_nothing_else` asserts as an
  equality: no bundled script runs and no linked file is expanded, because the sandbox is the
  only place code runs. What no test can gate: a third-party skill is usable, but its text
  becomes LM context, so treat an untrusted skill as a prompt-injection surface. See the guide
  (`rlm_harness/README.md`) "Skills (progressive disclosure)".
- **Memory is never rewritten, a model writes only `inferred`, and only the host erases.**
  `MemoryStore` (`memory.py`) labels every item with one of `PROVENANCES`: `observed` (taken from a
  source by the host), `said` (written by a person) and `inferred` (written by a model). Eight
  properties hold and must keep holding:
  1. **The model's only write path cannot name a provenance.** `memory_note(text, revises)` has
     no such argument and stores `inferred`, so the guarantee comes from the tool's SHAPE, the way
     `HarnessInvoke`'s single long-text argument enforces its contract. Do not add a `provenance=`
     parameter to that tool, or a second model-facing write tool that takes one. A note written
     after reading untrusted text is replayed into every later run that searches for it, and the
     label is the only thing that lets a reader discount it.
  2. **Nothing written is changed, and an id never names different text.** `revise` adds a new row
     pointing at the old one, and `retire` sets `valid_to` once; that is the module's only UPDATE.
     Triggers on every row table (items, labels, failures, purge records, aliases, seen names)
     refuse a stray
     `UPDATE`, `DELETE` or `INSERT OR REPLACE` (REPLACE deletes without firing a DELETE trigger,
     which is why an INSERT trigger refuses an existing id; it guards on `NEW.id > 0` because an
     auto-assigned id reads as -1 there, and `CHECK (id > 0)` keeps a stray row at -1 from turning
     that guard against every later add), and the UPDATE trigger compares with
     `IS NOT`, because a `!=` guard reads `NULL != 'x'` as false and would let a NULL `revises`
     or `run_id` be filled in later. The FTS5 indexes are contentless with `contentless_delete=1`
     (SQLite 3.43 or newer, checked at construction), so they can forget; an index row deleted by
     other means only makes search miss an item that is still intact. These protect history from a
     host's SQL mistakes, not from someone with the file: `DROP TRIGGER` still works, and the
     docs must not claim otherwise.
  3. **Only `purge` erases, and only the host calls it.** A person may want back out what they put
     in, so erasure exists, but as one deliberate host-side method with no tool and no model-facing
     path. It deletes the item, its labels, failures, index rows and vector and writes a
     content-free `purged` record. A `contentless_delete` index keeps a deleted row's words in its segments
     behind a tombstone, so purge then merges both indexes (FTS5 `'optimize'`). Every store
     connection runs with `secure_delete` on, not only purge's: an ordinary write merges index
     segments too, and without it leaves an old copy of the text as slack in a live page that a
     later purge cannot reach (measured). A test scans the file's bytes for a purged word, because
     a search-only test passed while the words were still on disk. It passes the delete triggers through a
     `purge_flag` row inserted and removed inside its own transaction, which no other connection can
     see. Aliases and seen names are store-wide vocabulary and purge leaves them, so a name only a
     purged item used stays there (stated in the purge docstring). The table is ordinary rather
     than TEMP because a main-schema trigger cannot read the temp
     schema, so every open clears a row a host's stray SQL may have committed there. Do not add another
     way past the triggers, and do not expose `purge` to a model. Ids stay AUTOINCREMENT, so a purged
     id names nothing rather than something else: that, not "nothing is ever deleted", is what keeps
     an old trace or citation honest.
  4. **A person's label outranks a model's.** For each item and kind, the current label is the
     latest `said` row when its value is set, otherwise the latest `inferred` row; a `said` NULL is
     a person clearing their own label. This lives in the `current_labels` view and nowhere else,
     so every reader agrees. A model-backed pass writes `inferred` labels only. A label shown to a
     model carries its own provenance (`memory_search` rows have `title_provenance`): a review found
     a model-written title riding in a row whose only provenance said `observed`, which is the
     mislabelling rule 1 exists to prevent.
  5. **A person's decision about a name stands, an unmerge included.** Aliases are event rows read
     through the `current_aliases` view, and the rule differs from labels on purpose: for a label a
     `said` NULL lets the model's label show through, for an alias a `said` row decides whenever
     one exists, NULL meaning "not an alias". Written the label way, a person's unmerge of a model's
     merge would bring the merge straight back. An `inferred` merge or unmerge is refused for any
     name with a `said` row in ANY role it plays (alias, canonical given, canonical resolved to):
     checking the alias alone let a model rejoin a separated pair the other way round. It also
     needs the name in the batch the model was asked about. A label is never written with a name
     the alias table would give it: grounding keeps an alias as written and readers resolve it, so
     an unmerge undoes the merge in every label written while it was live. A canonical
     that is itself an alias is followed before a merge is recorded, and a name that is already a
     canonical or a live alias is refused, so the table never holds a chain and no later merge
     silently repoints an alias. `entity_seen` has no UNIQUE column: INSERT OR REPLACE against one
     deletes without firing the delete trigger, so "insert if absent" is a `WHERE NOT EXISTS`.
  6. **A person's decision about an item and a topic stands, a removal included.** Topic
     membership is read through `current_members` with the alias rule, not the label rule, for the
     same reason: a person's `unplace` of a model's placement must not let it show through again.
     A topic's name follows the label rule (a person's name outranks a model's). The organising
     pass is one `dspy.Predict` call per round, as in the design it was taken from, and the host
     judges every placement (`plan_from`): only items shown, each once, only live topics, an
     existing topic only with a "subject" or "part" fit, a new topic only for two or more items and
     never into a pair a person dismissed or took apart, and never under the name of a topic a
     PERSON retired (a retirement row carries provenance, so a pass retiring its own underfilled
     topic does not burn the name). Items shown but not placed are recorded against the topic
     set the model was SHOWN (taken before the call, so the next round re-checks them against the
     topics this one opened), and come back only beside a new arrival or after the topics change:
     a lone item gets its topic when a second item on its subject arrives, and an unchanged store
     asks nothing. Dropping the re-gather half left every lone item out for good. Rows that name an
     item go with it on purge; a topic and its names stay.
  7. **Vectors are a derived cache that a purge erases, and similarity stays a weak signal.**
     `memory_vectors.py` keeps plain float32 BLOBs in `<store path>.vectors.db`, compared with
     numpy, with its own migration ladder and no append-only triggers (every row can be rebuilt by
     `sync`, like the FTS indexes). No SQLite extension: with one, only a connection that loaded it
     could delete a vector, and `purge` erases a purged item's vector with the standard library
     alone, open index or not, because an embedding can be partly turned back into its text. `sync`
     re-checks the store AFTER writing a batch, so a purge that commits mid-batch leaves no vector.
     The vector file is a second database, so a crash between the store's commit and that erase
     leaves the vector until the next `sync` drops it, and a reader open on the file keeps freed
     pages in its WAL until the last connection closes (logged). A row keeps the SHA-256 of the
     model key and the exact text embedded, so a changed label, text or model re-embeds and a short
     item is not re-read until it changes. `DEFAULT_MODEL` (multilingual-e5-small int8, pinned
     revision, sizes and SHA-256 values) is the KIT's own choice for multilingual text, not a
     consumer's; `download_model` is never called implicitly and installs nothing that fails its
     checksum. Two items are similar only at or above the pair's threshold AND as mutual near
     neighbours. A pair of mostly CJK texts is held to `cjk_threshold` (0.90 for the default) and
     every other pair to `similar_threshold` (0.86): measured on the default model, unrelated
     Chinese pairs reached 0.892 while unrelated English pairs peaked at 0.819, and in a mixed store
     one 0.86 threshold drew 13 wrong links beside 15 right ones where the pair rule drew 14 right
     and none wrong. Do not fold the two back into one number. `MIN_TEXT` counts a CJK character 3
     times (`text_weight`), so a Chinese note is compared from about 14 characters. A similarity
     suggestion scores at `SUGGEST_THRESHOLD`, below any label match of the same score. numpy is a
     dev dependency so CI runs the real comparison; onnxruntime and tokenizers are faked.
  8. **A run reaches `observed` only through bytes the host holds: the host verifies the bytes,
     the model chose them.** `consolidate_run` stores a model's `fact` as `observed` only when its
     named source is trusted and `_find_span` finds the quote in that source's full text, and it
     stores the source's span, never the model's wording, with `meta.selected_by = "model"`.
     REPL output is never a source. A recorded tool result is trusted only for a tool the host
     names in `trusted_tools`, and a `model:` tool never: `model_as_tool` records another model's
     answer as its `result`, and a consumer's tools may too, so trusting every recorded result
     would let a model's words pass for a source, the mislabelling rule 1 exists to prevent. For
     the same reason a call that recorded `ok=False` is never trusted, and the docs tell the host
     to name only a tool whose result cannot echo the model's arguments; the kit cannot check that.
     Everything else a run proposes is `inferred`, with the reason a fact fell back in its
     `meta`. `compute_memory_use` reads the memory tools' `memory_op` field rather than their
     names, and calls an unreferenced read an upper bound on waste, never a measurement.

  What the kit cannot verify is the host's choice between `observed` and `said`; the host's code
  path assigns it, and the docs say so rather than implying a check. The schema changes only by
  appending to `memory._MIGRATIONS`, counted in `PRAGMA user_version`, and a shipped step is never
  edited: step 5 rebuilds the item index with an `origin` column rather than changing step 2's
  rebuild, which still writes the two columns that step created. A step that needs the opening
  store's settings (step 5 reindexes with its `max_index_chars` and `index_origins`) carries the
  `_REINDEX_ITEMS` marker, matched by identity, because a step's code receives only the connection;
  a version comparison would also fire on a test that cuts the ladder short. A memory call records
  an ordinary `tool_call` with optional `memory_ids` and `memory_op` fields; it does not add an
  event type. Pinned by `tests/test_memory.py`, `tests/test_memory_tools.py`,
  `tests/test_memory_labels.py`, `tests/test_memory_purge.py`, `tests/test_memory_tasks.py`,
  `tests/test_memory_aliases.py`, `tests/test_memory_topics.py`, `tests/test_memory_vectors.py`,
  `tests/test_memory_consolidate.py` and `tests/test_memory_lookups.py`.

  The labelling pass (`memory_tasks.distill_pending`) makes no model call the host did not ask for,
  caps a pass at 500 items, and keeps a short item to ONE `dspy.Predict` call; only a text past
  `SHORT_LIMIT` pays for an RLM run. An entity label is kept only when its name occurs in the item,
  in the span its quote was found at (`tools.grounding._find_span`, which returns the source's own
  bytes and refuses a quote under 20 non-space characters), or among the names in use the call was shown. Do
  not drop that last clause: a summary told to reuse a known name writes it in another script than
  the text uses, and grounding on the text alone would delete exactly the convergence it exists for.
  The labelling rules tell the model to write an entity exactly as the text or a known name writes
  it, even when `language` asks for another one. Do not let entities follow `language`: measured
  on an English text labelled in Chinese, every translated name failed grounding and the item kept
  no entities.
- **The trace is a VERSIONED wire format: additive-only within a version.** `SCHEMA =
  "rlm-harness/trace/v2"` + every `EVENT_*` type string + the `{schema, run_id, step_id, ts, type,
  payload}` envelope + the dataset-exporter record shapes are a CONTRACT that offline readers build
  on (replay, the `export_*` exporters, AND every consumer's report renderer / dataset / re-render).
  Within a version you MAY add an optional payload field; you may NOT remove, rename, or re-type an
  existing event type, envelope key, or established payload field: that silently breaks every
  downstream reader without a test failure here. A breaking change bumps `SCHEMA` to the next
  version with a migration (`v2` since 1.17.0, so `v3` next).
  `tests/test_contract.py` pins all of this: if it goes red you are about to break a consumer, not
  the test.

  **The version has moved exactly twice, and only one of those was a format change.** It read
  `rlm-kit/trace/v1` until the pre-publication rename to `rlm-harness` (CHANGELOG 1.0.0), which
  moved the vendor tag while leaving `v1` and the shape byte-identical: safe only because the
  format had zero external readers at the time. Then **1.17.0 took `v1` to `v2`**, for one key:
  `api_rounds` left the `run_end.payload.usage` call entries, where it had shipped in 1.11.0 as a
  field the kit set itself, so the additive-only rule covered it. The removal was forced, not
  chosen: an engine-backed LM's usage reaches dspy's tracker through `usage_dict`, which rebuilds
  a fixed key set off the typed `lm15.Usage`, and a tracker entry carries no per-call identity to
  re-attach it by (measured both ways). What was a CHOICE is the bump: the same release first
  described the whole thing as a "move" and called it additive, which is true of the new
  `EVENT_API_ROUNDS` and false of the old key, and a removal carried quietly inside a version is
  exactly what this rule exists to stop. Nothing else about v2 differs from v1.

  Two things that make a bump cheap here, and both are properties to KEEP rather than facts to
  assume. The kit writes `schema` and reads it back nowhere, so a v1 corpus still loads through
  `load_events` and every exporter: pinned by
  `tests/test_contract.py::test_a_v1_trace_still_reads_because_nothing_validates_the_schema`, and
  the obvious post-bump "improvement" of validating the string is what that test exists to refuse.
  And every reader here and in all ten consumers keys off `type`, never off `schema`: audited
  before the 1.0.0 rename, and re-audited across every consumer checkout before this bump (the
  only `schema ==` comparison downstream is `diff-sentry` checking its OWN `diff-sentry/` prefix).
  Re-audit it the same way next time rather than inheriting the conclusion; "no consumer validates
  the version" is a measurement with a date on it, and the first release that asserted it without
  measuring got the cost of a bump backwards.
- **rlm-harness produces TRAJECTORIES, never reward.** The kit runs the RLM, records the trace,
  and turns traces into datasets (`export_sft_turns` / `export_rl` / `export_actions`). It does
  NOT score them: `export_rl` and `export_actions` carry a `reward=` HOOK the downstream trainer
  fills and pass `reward=None` themselves, and `export_sft_turns` has no such parameter at all,
  which is the stronger version of the same property rather than a gap in it. Reward composition,
  credit assignment, and GRPO/SFT are a SEPARATE fine-tuning project: rlm-harness + its consumer
  are the ROLLOUT stage only. Emit raw labels/metrics; let the trainer score. (A prompt/policy
  convention that improves rollout QUALITY is in scope, and so is `dspy.GEPA`, which evolves
  instruction TEXT against a caller-supplied metric rather than composing reward from a trace;
  `optimize.py` and `docs/PROMPT.md` cover it: better rollouts ≠ reward.)
- **The public surface is `__all__`; consumers EXTEND, they don't fork.** `__init__.__all__` + the
  trace schema + `RLMTask`'s declaration fields are the API a consumer builds on; a `_`-prefixed name
  or module internal (`trace._active`, `_retry`) is private and may change without notice. A consumer
  extends three ways and only these: subclass `RLMTask` (declaration), add a tool the **base/wrap**
  way (generic base + syntactic guard + factory HERE, provider + tracing in the consumer: as
  `make_model_tool` / `make_fetch_tool` / `make_web_search_tool` / `make_harness_tool` do), and read
  results through the trace + exporters. It must NEVER fork the harness or re-implement tracing. If a consumer needs an
  internal seam the kit doesn't expose, ADD a named, documented hook here (how `recorder_scope` in
  `trace.py` + `bind_recorder_to_sub_lm` in `sub_lm.py` were born: the cross-thread sub-LM recording
  fix; both are importable public functions, though not in the top-level `__all__`). Do not reach
  into a `_private` name. Full walkthrough: the guide (`rlm_harness/README.md`) "Building a consumer".
- **`make_harness_tool` delegates a sub-task to ANOTHER rlm-harness harness. Long text IS the contract.**
  The promoted "wrap a downstream harness as a tool" shape (`tools/harness.py`), a THIN reuse of
  `make_model_tool`'s retry/validate/circuit-break core plus a child-rollout LINK. Its reason to exist is
  the RLM framework's native advantage: an input field holds near-unbounded text that dspy injects as the
  Root LM's REPL ENVIRONMENT. So a `HarnessInvoke` takes ONE long-text arg and nothing else (the contract
  enforced by SHAPE), and `harness_from_endpoint` binds that WHOLE context to the downstream harness's
  long-text input field: the child then runs a FULL RLM loop (REPL + its own MCP / skills / fetch) over
  it, not a one-shot completion. TRAJECTORY SEPARATION is load-bearing: the parent records ONE leaf
  tool_call + a `child_run_id` / `child_trace` link (additive: no new event type), while the child owns its
  OWN trace/rollout, exported separately (both reward-free). The kit ships NO transport and NAMES no
  harness: the consumer injects `call_endpoint` (subprocess / in-process / HTTP) and the harness's
  identity lives only in the consumer's runtime config, exactly as `make_command_tool` takes an injected
  `Runner`. A dead / slow / looping child degrades (`endpoint_error` / `circuit_broken`), never sinking
  the parent run. (The CLIENT side; its SERVER-side mirror is `serve_harness` below. Both sides are the
  consumer guide's "Delegate to another harness, or be one" step, `rlm_harness/README.md`.)
- **`serve_harness` is the SERVER-side mirror of `make_harness_tool`: so connecting a harness needs no
  bespoke glue.** `make_harness_tool` is the CLIENT (the parent wraps a harness as a tool via an injected
  `call_endpoint`); `serving.py`'s `serve_harness(run, to_pointer, …)` + the `python -m
  rlm_harness.harness_serve <module:run>` entry are the SERVER: they turn any RLMTask harness into a process
  that speaks the contract. The kit owns ALL generic plumbing (read stdin → the child's RLM env, run_id,
  CWD isolation, the `HarnessPointer` wire, exit codes: 0=ran / 1=infra, keep the harness's identity +
  tracebacks OFF stdout). The consuming HARNESS supplies only the one thing the kit can't know: mapping
  ITS result object into a `HarnessPointer` (`to_pointer`): in a ~5-line `serve` module in its OWN repo;
  the operator then points the client endpoint straight at `-m <harness>.serve`, no intermediate project.
  `HarnessPointer.to_json_line` flattens `meta` to top level so the wire is what the client's `read_output`
  parses. dspy-free. Anticipatory: written for a FUTURE downstream harness; the kit names none. **Worked
  example to copy: `examples/harness_serve.py`; both sides are step 6 ("Delegate to another harness, or
  be one") of the consumer guide (`rlm_harness/README.md`).**
- **Keep the public surface vendor-neutral.** rlm-harness's package, source, docs, and commit messages
  refer to downstream consumers GENERICALLY ("a consumer", "a downstream UI"): never by a specific
  project name, and never reproducing a consumer's product domain. A consumer's own concrete values
  (model names, schemas, product terms, paths) live in the consumer, not here. This keeps the kit
  decoupled from any one user and the published artifact free of third-party specifics. The ONE
  exception is a single, clearly-delimited **"Built with rlm-harness"** adopters section in the README: it
  MAY list real, PUBLIC downstream projects by name + link + a one-line description. That is an adopters
  list, not design coupling: the kit's mechanics, examples, API docs, and commit messages still describe
  consumers generically, and a consumer's domain specifics still never appear anywhere else. Only list a
  consumer that is public and whose maintainer wants the association; never a private or internal one.

