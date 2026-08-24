# rlm-harness — the guide

The deep documentation for `rlm-harness`: what each module owns, the harness-engineering
layer (sub-LM hook + trajectory tracing), the tool surfaces, the rollout conventions,
the consumer contract, and every configuration knob. For what the kit *is* — the
pitch, the quickstart, and installation — start at the
[top-level README](../README.md).

## Layout

| Module | Responsibility |
|---|---|
| `config.py` | Single source of truth; `RLMConfig.from_env()`. No dspy import. |
| `runtime.py` | `configure()` — wires dspy + optional observability once, including auto-routing a `claude-agent-sdk/`-prefixed model string onto `ClaudeAgentLM`. |
| `task.py` | `RLMTask` base class. |
| `_retry.py` | Validation + retry engine (dspy-free, unit-tested). |
| `sandbox.py` | Interpreter selection + the insecure-sandbox guard. |
| `atomic.py` | `atomic_write_text` — a same-directory temp file + `fsync` + `os.replace`, so a concurrent reader never sees a partial write. dspy-free. |
| `metrics.py` | `RunUtilization` / `compute_run_utilization` / `compute_utilization_by_run` — reward-free trace utilization metrics (how a run's activity split across root-LM turns, tool calls, sub-LM escalations), a pure derived read over already-recorded `trace/v1` events. dspy-free. |
| `tools/` | `make_schema_validator` (pydantic) + `make_json_schema_validator` (validate a parsed object against a vendored JSON Schema — the base for the "validate against an official, version-pinned upstream schema" pattern; needs `rlm-harness[jsonschema]`), SSRF-guarded `make_fetch_tool`, its filesystem-side analogue `make_read_file_tool` / `make_grep_files_tool` / `resolve_within_root` (needs `rlm-harness[grep]` for a wall-clock-bounded `grep_files` — see below) plus the write side `make_write_file_tool` / `make_edit_file_tool` in `tools/edit.py` (see below), provider-agnostic `make_web_search_tool`, `make_command_tool` — a traced `run_command` over a consumer-supplied *isolated* runner (the kit ships no executor) with an optional `refuse_broad_git_history` guard, `make_model_tool` — the generic "model-as-tool + transient-retry + validate" core (a project wraps it with its own endpoint/validator/messages), and the harness-delegation pieces `make_harness_tool` / `harness_from_endpoint` / `pointer_to_invocation` / `run_isolated` (see "Delegate to another harness" below). |
| `optimize.py` | GEPA harness — metric templates now, compile in Phase 2. |
| `sub_lm.py` | `intercept_sub_lm` — wrap the RLM's sub-LM to trace every escalation as a `sub_call` (+ optional validate/post-process); `model_as_tool` for LM-decided multi-model routing. |
| `skills.py` | `load_skills_as_tools` — expose a Skills directory to the RLM as tools. |
| `trace.py` | `TraceRecorder` — unified append-only JSONL trajectory (main steps + sub-LM + tool calls). |
| `replay.py` | Reconstruct/replay a recorded run using recorded tool outputs. |
| `dataset.py` | `export_sft_turns` / `export_rl` / `export_actions` — turn traces into training datasets (`export_sft_turns` = per-root-turn SFT, the RLM recipe of arXiv 2512.24601); `run_label_bundle` — carry per-run LABEL surfaces beside the trajectory. |
| `mcp.py` | MCP **client** (optional `rlm-harness[mcp]`): `mcp_tools` exposes an external server's tools as SYNC `dspy.Tool`s; `McpCatalog`/`McpConnection` for a progressive multi-server surface. |
| `testing.py` | Offline forward-path harness — `ScriptedInterpreter`, `scripted_lm`, and the REPL-safety assertions `assert_repl_safe` / `assert_task_repl_safe`. |
| `serving.py` / `harness_serve.py` | Server side of the harness-delegation contract — turn an `RLMTask` into a process that speaks it. |
| `container_interpreter.py` | The opt-in `interpreter="container"` REPL — model code runs inside an isolated Docker container so it can spawn subprocesses. |
| `_toolname.py` | REPL-safety rules for a tool's NAME and SIGNATURE. Mostly private, but `is_valid_tool_name` / `sanitize_tool_name` / `unique_tool_names` / `signature_from_json_schema` are public — a consumer building its own tools (e.g. from `McpCatalog`'s raw names) needs the same derivation the kit uses. |
| `_dspy_compat.py` | Private. Every cross-version dspy difference (kwarg renames, the interpreter seam, the recoverable/terminal error split) resolved by introspection in ONE place. |
| `rubric.py` | Reward-free rubric primitives: the `Criterion`/`RubricCriteria`/`CriterionFact` types, `rubric_to_meta`/`rubric_from_meta`, `validate_rubric`, and a pure `criteria_facts(criteria, facts, lens)`. `category` is an OPAQUE caller-defined label — the kit imposes no taxonomy. See "Building a consumer". |
| `claude_agent_lm.py` | `ClaudeAgentLM` — run rlm-harness on a Claude Pro/Max subscription: a `dspy.BaseLM` over the official Claude Agent SDK. `configure()` now auto-routes a `SUBSCRIPTION_PREFIX`-carrying (`"claude-agent-sdk/"`) `main_model`/`sub_model` string onto it — no explicit `main_lm=`/`sub_lm=` wiring needed unless you're overriding it. Opt-in `rlm-harness[subscription]`; pure completions (no tools), lazily exported so `import rlm_harness` stays dspy/SDK-free. |
| `examples/mini_run.py` | Minimal end-to-end live run — config + a tiny `RLMTask` through a real `dspy.RLM`, with the trajectory recorded and summarised. |
| `examples/claude_agent_lm.py` | Runnable demo of `ClaudeAgentLM` — a tiny `RLMTask` through a real `dspy.RLM` on a subscription login. |

## RLM as Harness Engineering (sub-LM hook + tracing)

`dspy.RLM` exposes no hook to intercept a sub-LLM response, and (as of dspy 3.3.0) no
multi-sub-model or depth>1 recursion. The clean lever is to **wrap a `dspy.LM`**:

```python
from rlm_harness import intercept_sub_lm, model_as_tool, get_sub_lm, TraceRecorder, RLMConfig, configure

configure(RLMConfig.from_env())
base = get_sub_lm()          # the configured base sub-LM — single source of truth
# intercept_sub_lm traces every escalation; validators/postprocessors are optional
# (deterministic only — agentic actions stay LM-decided tools):
smart_sub = intercept_sub_lm(base, validators=[...], postprocessors=[str.strip])

with TraceRecorder("traces/run.jsonl", run_id="r1"):
    finding = await MyTask(sub_lm=smart_sub).arun(evidence=blob)
```

`intercept_sub_lm` records a `sub_call` for every escalation and, if you pass them,
runs deterministic validate → post-process. `get_sub_lm()` hands back the base sub-LM
`configure` built — wrap THAT rather than reconstructing a `dspy.LM`, so it can't drift
from the configured model. External tools are exposed to the main
LM via `tools=` / `load_skills_as_tools` / `model_as_tool`, so the decision lands in
the trajectory. `TraceRecorder` records main steps (`Prediction.trajectory`), every
sub-LM call, and every tool call into one JSONL stream — replayable (`replay.py`) and
exportable as an RL/SFT dataset (`dataset.py`). Langfuse is an optional mirror; the
JSONL is the dataset's source of truth.

> **Reading a `sub_call`:** every `sub_call` event is exactly one sub-LM escalation,
> reached through `dspy.RLM`'s built-in `llm_query` / `llm_query_batched` (the only
> callers of `sub_lm`). The payload carries `kind:"sub_lm"` + the wrapper `name`. It
> does **not** record which built-in triggered it — dspy calls `sub_lm` identically for
> both. The planner's actual `llm_query(...)` call lives in the `main_step` `code`, so
> *that* is where a Root-LM trainer learns "call llm_query"; the `sub_call` is the inner
> view. `llm_query_batched` fans calls across threads — `TraceRecorder.record` is
> lock-guarded so concurrent `sub_call`s can't corrupt the JSONL.

> Depth is **1** by design here (main LM + one intercepted sub-LM layer). True
> depth>1 recursion is unsupported upstream and out of scope.

### Sub-LM vs. tool: which model goes where

`intercept_sub_lm` and `model_as_tool` both "wrap a model," which makes them easy
to confuse. They sit on **opposite sides of the RLM boundary**, and the choice is not
cosmetic — it decides what your RL data records.

- **A sub-LM is part of the machine.** Wired in as `sub_lm=`, the *framework* decides
  when to call it — it is the seat the RLM's recursion plugs into (depth-1 here, but
  structurally the recursive seat). The framework assembles its prompt/context and it
  carries the run's identity (tracing, budget). The main LM never *chooses* to call it.
  → recorded as a **`sub_call`**.
- **A tool-LM is a leaf the main LM picks up.** Passed via `tools=` (e.g.
  `model_as_tool`), the *main LM* decides, in the REPL, to call it — with whatever
  string it wrote. It takes a string, returns a string, and stops: it can't recurse and
  never becomes an RLM root. The call is the LM's own decision, so it lands in the
  trajectory. → recorded as a **`tool_call`**.

> At the lowest level both are "call an LM with text, get text back." The difference is
> **role, not mechanics**: a sub-LM is a structural seat (framework-invoked,
> recursion-capable); a tool-LM is an optional leaf the main LM reaches for.

**"Deterministic transform" = plain code, no AI.** Both sides may check their model's
output with ordinary functions — same input, same output: `intercept_sub_lm` runs
`validators`/`postprocessors` on the sub-LM output; `make_model_tool` runs a `validate`
callable on a generated artifact (a consumer's generator tool runs a `postprocess()` validator to
verify the artifact's shape — that lives on the **tool** side, not the sub-LM). What neither may
do is ask *another model* to judge the output: that is an *agentic* decision, and agentic
decisions must stay with the main LM as a `tools=` call so the choice is visible in the
trajectory (and honest as RL data). That is exactly why `model_as_tool` is a thin
pass-through with no validation baked in — **deterministic checks are fine on either side;
a model-judgement must be an LM-decided tool call.**

**Pick by question:**

| You want… | Use | Wire as |
|---|---|---|
| a smarter/cheaper *default* sub-model, traced, with optional deterministic checks | `intercept_sub_lm(base, validators=…, postprocessors=…)` | `sub_lm=` |
| the main LM to *choose*, mid-task, to consult another named model | `model_as_tool(name, lm)` | `tools=` |
| both (a chosen model that also self-checks) | compose them: `model_as_tool("expert", intercept_sub_lm(expert_lm, …))` | `tools=` |

**Escalate to the sub-LM when a tool WALLS — don't circle it.** A convention, not an API. When a
`make_model_tool` (or any model-backed tool) repeatedly fails on the SAME gap — declines, returns
INVALID, can't fill the hole — that IS the signal the main LM cannot specify its way out: escalate to
the sub-LM for that gap instead of re-spinning the tool. Circling a walled tool burns the iteration
budget and can hit the cap with the task still unfinished; one focused sub-LM question often unblocks
it in a single turn (the sub-LM is the recovery seat — its whole purpose; the "expensive" framing is no
reason to keep re-spinning a stuck tool). Like grounded completeness this lives in the consumer's task
INSTRUCTIONS, kept in the trajectory as honest RL data. A consumer can nudge its planner this way after
a few repeated tool declines on one gap — turning a hard run that would otherwise circle a stuck tool
until the cap into one that escalates once and converges.

The nudge is a PROMPT, which a weaker root LM can ignore (one may hammer a stuck tool dozens of times). For a
deterministic backstop, `make_model_tool(max_consecutive_invalid=N)` is a run-scoped CIRCUIT BREAKER:
after N consecutive validator declines the next call SHORT-CIRCUITS (no model call,
`circuit_broken=True`), capping the wasted calls and forcing the consumer's redirect (escalate /
finalize). It resets on any ok; an endpoint error doesn't count. The factory only flags the break —
the consumer owns the message — and builds one tool per run so the breaker state resets naturally.

**`ok=False` has THREE causes; read `result.cause` before naming one.** The validator rejected the
output (`"invalid"`), the endpoint failed after retries (`"endpoint"`), or the breaker
short-circuited without calling the model at all (`"circuit_broken"`). In the last two the validator
NEVER RAN, so a consumer that reads only `ok` and writes "failed validation" is blaming the model
for infrastructure. That has shipped in more than one consumer, in both directions that matter — a
per-run training label named `*_rejects` whose docstring said "the validator rejected" incrementing
on a 502, and a reviewer-facing string reading "failed its format check" shown for an endpoint
timeout. `result.validator_ran` is the direct form of the question:

```python
result = call(spec)
if not result.ok:
    reason = ("the output failed validation" if result.validator_ran
              else f"no output to validate ({result.cause})")
```

Two counting notes that follow from it, both observed downstream: a circuit-broken call carries
`ok=False` **too**, so a `sum(1 for r in calls if not r.ok)` metric silently includes every break —
filter on `cause == "invalid"` if you mean rejections. And an endpoint error deliberately does not
trip the breaker, so `circuit_broken` counts and endpoint counts never overlap with each other.

**Across the trace boundary the same distinction is `trace.payload_cause(payload)`**, reading the
same four words off a recorded `tool_call`. It exists because `ok` is often ABSENT on an endpoint
payload (consumers record `error=` alone there), so `payload.get("ok")` returns `None` — falsy —
and every `not payload.get("ok")` counter downstream absorbs infrastructure failures as content
declines without a word of warning. In the worst measured case that put 113 endpoint failures into
a metric named `generator_declines`, fed it to a scored rubric criterion about the planner's spec
quality, and printed "113 partial/retry" in a delivered report — for a run whose validator ran zero
times, and whose planner had correctly concluded the endpoint was unreachable.

`export_actions` now carries `outcome.cause` and `outcome.error` for exactly this reason: that
record is what reaches a trainer, and it previously carried neither, so the split could not be
reconstructed downstream even by hand.

Two write-side hazards worth stating, both observed:

- **Record the outcome ONCE, AFTER the branch.** A consumer that emits its event before checking
  `endpoint_error` destroys the distinction at write time, and no read-side fix recovers it.
- **Do not omit `ok` on the endpoint path.** Recording only `error=` is what makes
  `payload.get("ok")` return `None`. Passing `cause=result.cause` explicitly is the cheapest
  insurance: the derivation is then done once, by the code that knows.

## Skills (progressive disclosure)

`load_skills_as_tools(dir)` exposes a directory of knowledge as two tools, so the main LM
pulls reference **on demand** instead of carrying it all in the prompt:

- `list_skills()` → each skill's `name` + one-line `description` (cheap, always in view)
- `read_skill(name)` → the full skill body, fetched only when the LM judges it relevant

Skills follow the Agent-Skills convention: a `<name>.md` file (or a `<folder>/SKILL.md`)
with a leading `---` frontmatter block carrying `name` / `description`, then a plain-markdown
body. The list→read split is two-level **progressive disclosure**; and because the LM calls
these as tools inside the REPL, "which knowledge did I load" lands in the trajectory (and the
RL dataset).

For a larger catalog you can skip the discovery round-trip: `load_skills_as_tools(dir,
discovery="inject")` returns **only** `read_skill`, and you inject the catalog into the system
prompt yourself with `render_skills_manifest(dir)`. The LM then sees every skill's
`name` + `description` at startup (no `list_skills` call) and still pulls a full body
just-in-time with `read_skill`. The default `discovery="list"` keeps the `list_skills` tool
instead. See `examples/harness_run.py`.

Scope & caveats:
- **Knowledge only.** `read_skill` returns the markdown text — it does NOT execute bundled
  scripts or expand linked files. A "just instructions" skill works fully; a skill that ships
  runnable helpers gives you only its prose.
- **Third-party skills work** if they use the `SKILL.md` + `name`/`description` convention:
  drop them in the dir and they are discoverable. But a skill's text becomes the main LM's
  context — treat untrusted skills as a **prompt-injection surface** and vet them. Frontmatter
  beyond `name`/`description` is ignored.

## MCP tools (connect an external MCP server)

`mcp_tools(server)` exposes an **external** [MCP](https://modelcontextprotocol.io) server's tools to
an `RLMTask` as ready-to-use tools. rlm-harness is a **client only** — it never runs a server and bundles
none; you point it at someone else's (a local stdio command, or a remote streamable-HTTP URL):

```python
from rlm_harness import mcp_tools

with mcp_tools({"url": "https://mcp.example.com/mcp"}) as tools:        # or {"command": "npx", "args": [...]}
    finding = MyTask(tools=tools).run(...)                              # the server's tools are now callable
# `tools=` (1.1.0+) is the per-instance override — the sanctioned way to attach tools that exist
# ONLY inside this `with` block. It REPLACES the class-body `tools` declaration.
```

Needs the extra: `pip install "rlm-harness[mcp]"`.

- **The connection is live for the `with` block** and torn down on exit (a stdio subprocess is
  terminated). Each tool call is recorded as a `tool_call` in the trace, like any other tool.
- **Sync, despite an async SDK.** The MCP Python SDK is async, but dspy.RLM invokes tools
  synchronously, so rlm-harness runs the session in a background thread and bridges each call across.
  (dspy's own `Tool.from_mcp_tool` makes an *async* tool for `dspy.ReAct` — it does not work on the
  RLM sandbox path, which is why `mcp_tools` exists.)
- **Security: MCP tools run HOST-SIDE**, outside the sandbox — a stdio server is a subprocess this
  process spawns. Treat an MCP server as a **trusted dependency**, and its output as a
  **prompt-injection surface** (untrusted LM context), exactly like fetched web content.

### Many servers, progressively — `McpCatalog` + `McpConnection`

`mcp_tools` is the single-server convenience: it materializes one server's tools as self-tracing
`dspy.Tool`s up front. For a consumer building its OWN progressive tool surface over **several** servers
— list servers, `load` one on demand, read its tools, `call` — use `McpCatalog`:

```python
from rlm_harness import McpCatalog

cat = McpCatalog([{"name": "docs", "url": "https://mcp.example.com/mcp"},
                  {"name": "shell", "command": "npx", "args": ["-y", "some-mcp"]}])
try:
    cat.servers()                      # [(name, description), ...] — every declared server
    cat.load("docs")                   # connect one on demand (a no-op under the eager default)
    for tool in cat.tools("docs"):     # RAW mcp Tool objects (name / description / inputSchema)
        ...                            # map them onto YOUR own tool shape
    text = cat.call("docs", "search", {"q": "..."})   # flattened result text
finally:
    cat.close()
```

- **Raw, and records nothing.** `McpCatalog` returns the server's RAW MCP `Tool`s (not `dspy.Tool`s) and
  emits no trace events — the consumer maps each tool to its own shape, and its own tool wrapper owns the
  `tool_call`. That keeps the catalog dspy-free and leaves tracing where the consumer wants it.
- **`McpConnection`** is the public single-server bridge `McpCatalog` manages one of per server (and that
  `mcp_tools` is built on); **`result_text`** flattens a `CallToolResult` to text. Both are exported for a
  consumer driving a connection directly.
- **Eager by default** (connect host-side, before the run) — a subprocess spawn inside an async tool loop
  can hang asyncio; `connect="lazy"` defers each server's connect to its first `load` (opt-in). The same
  HOST-SIDE execution and prompt-injection notes as `mcp_tools` apply.

## Running local commands (an isolated runner)

`make_command_tool(runner)` gives an `RLMTask` a `run_command` tool — the reusable half of
letting the model run a local command (a build, a test, a git op) the way a coding agent does.
It enforces the sync contract, turns a failure into text the RLM reacts to, and records one
`tool_call` in the canonical shape. The kit ships **no** executor and picks **no** isolation: you
supply the `runner`.

```python
from rlm_harness.tools import make_command_tool

run_command = make_command_tool(my_isolated_runner)     # your runner; the kit ships none
finding = MyTask(tools=[run_command]).run(...)
```

**The runner's isolation IS the security boundary.** `run_command` executes model-CHOSEN commands
HOST-SIDE — outside the pyodide/deno sandbox that isolates the RLM's own REPL code (same as the
fetch/search providers and MCP servers). A naive `subprocess.run` runner is arbitrary code
execution steered by the model — the same class of danger as the refused `local` interpreter. For
anything processing untrusted input the runner MUST execute inside a disposable, network-restricted
container / VM / OS-sandbox; `examples/command_runner.py` is a reference Docker runner (`--rm
--network=none`, workspace mounted read-only). A command **allowlist is not a substitute** — a shell
allowlist is routinely bypassed (`make`/`npm run` script edits, `find -exec`, `git -c`, `$(...)`,
env-var injection), so the kit ships no allowlist primitive; the optional `guard` hook is a
shape-only pre-flight, never a security claim. One such guard ships ready-made:
`refuse_broad_git_history` refuses a `git log` invocation carrying a broad-history option (`--all`,
`--branches`, `--remotes`, `--tags`, `--glob`, `--reflog`, `--walk-reflogs`, `-g`,
`--alternate-refs`) — an eval/training-run convention for stopping a model from reading branches,
tags, or reflogs it should not have task-specific hints from, same shape-only honesty as any other
`guard` (it is not a shell parser — a chained shell string bypasses it, by design, same as any
other `guard`).

- On success the model receives a `{"exit_code", "stdout", "stderr"}` dict (dspy JSON-bridges a
  `dict` into a real REPL value it reads — `run_command("ls")["stdout"]`; a dataclass would arrive
  only as its unsliceable `repr`, so the tool returns a dict and the runner returns the typed
  `CommandResult`). The trace keeps only `exit_code` + lengths + a stderr preview + `duration_ms`,
  like `fetch_url` records size not body.
- Sync, like every RLM tool — wrap an async container/sandbox client into a sync call yourself.

**One-shot vs stateful — the runner decides.** `run_command` returns a single command's result and
holds no shell state; whether cwd, env, filesystem writes, and background processes persist across
calls is the *runner's* contract. The reference example is a fresh container per call — a stateless
**inspect** surface (read-only mount). An edit-build-test loop needs a **stateful** runner: a closure
over a long-lived sandbox — `docker create` + `docker exec`, an [E2B](https://e2b.dev) /
[Modal](https://modal.com/docs/guide/sandbox) / [Daytona](https://www.daytona.io) sandbox handle, or
a [SWE-ReX](https://github.com/SWE-agent/SWE-ReX) `BashSession` — which fits the same seam with no API
change. Interactive tools and tmux-style sessions are out of scope for a one-shot result; if a
consumer later needs model-managed sessions, that's the moment to add an additive `session_id` to the
payload, not before. (`dspy.RLM`'s own pyodide/deno interpreter is WASM Python and **cannot** spawn a
subprocess, so shell execution has to come from a host-side tool like this — there is no in-sandbox
alternative.)

## Reading and searching local files (a bounded directory, no shell)

`make_read_file_tool(root)` / `make_grep_files_tool(root, candidate_paths)` — the filesystem-side
analogue of `make_fetch_tool`'s SSRF-guarded `is_safe_url`, filling the gap between "no filesystem
access at all" and `run_command`'s full-shell escape hatch. `root` is not "a repo" — it's any
bounded local directory tree a consumer scopes it to: a source repository, a docs corpus, an
extracted archive, a dataset directory, a log directory. The single most common thing a
coding-adjacent consumer needs — let the model read or search a bounded directory tree — does not
require a shell: both tools are a pure-Python scan over `resolve_within_root`-guarded paths, no
subprocess, no `rg`/`grep` binary on `PATH`.

```python
from rlm_harness.tools import make_grep_files_tool, make_read_file_tool

read_file = make_read_file_tool(repo_root)
grep_files = make_grep_files_tool(repo_root, candidate_paths=my_file_list)   # a consumer-computed list
finding = MyTask(tools=[read_file, grep_files]).run(...)
```

- **`resolve_within_root(root, path)`** is the shared guard both factories build on (public, like
  `is_safe_url`/`parse_cidrs`, for a consumer building a third filesystem tool the kit doesn't
  ship): refuses a `..` traversal, an absolute path elsewhere, or a symlink INSIDE `root` pointing
  OUTSIDE it — via `os.path.realpath` (which follows symlinks) then a `commonpath` containment
  check, never `os.path.normpath` (purely lexical, which would miss the symlink case).
- **`candidate_paths` is REQUIRED, consumer-supplied** — no default directory walk, no built-in
  `.gitignore` handling. Same base/wrap split as `make_command_tool` demanding an injected
  `Runner`: the kit owns the safety guard, the consumer decides which files are even candidates
  (walk a directory, read a manifest, whatever fits).
- **`name=` on both factories** (default `"read_file"`/`"grep_files"`) fixes a real collision: a
  task with more than one bounded root (a source root AND a docs root, say) needs each one's tool
  to have a distinct REPL identity, since dspy keys its tool dict by name and two tools sharing a
  name abort registration for EVERY tool on the task, not just the second one. Validated at
  factory-build time (a valid, non-reserved identifier) — a bad name raises `ValueError`
  immediately rather than surfacing as an obscure construction failure later.
- **`make_read_file_tool`** additionally takes `encoding=` (default `"utf-8"`, for a non-UTF-8
  corpus), `max_output_chars=` (default `None` = unlimited; truncates with a visible, non-silent
  marker when set), and `line_numbers=` (default `False`; prefixes each returned line with its
  REAL 1-indexed file line number, so a model reading a slice starting mid-file doesn't have to
  compute one itself from `start_line` — removing exactly the off-by-one a model gets wrong when
  later asked to cite or edit that line). The last two are scoped to the successful-read branch
  only — a `Refused`/`Read error` string is never numbered or truncated.
- **`make_grep_files_tool` requires the optional `regex` package outright** (`pip install
  "rlm-harness[grep]"` — a friendly `ImportError` otherwise, no silent fallback to stdlib `re`).
  `pattern` is LM-controlled, unbounded regex, matched against real file lines with no wall-clock
  budget anywhere else in a tool's call path — a catastrophic-backtracking pattern (`(a+)+$`
  against a non-matching line) can hang the host process indefinitely on stdlib `re`, and stdlib
  `re` cannot be bounded by ANY pure-Python mechanism, not even `signal.alarm` (CPython's `re`
  engine doesn't yield to the signal dispatcher mid-match — one `re.search()` call is a single,
  uninterruptible C-level operation). `regex`'s own matching loop periodically checks elapsed
  wall-clock time internally and raises `TimeoutError` when exceeded — a real, working,
  pattern-structure-agnostic mechanism. Same "no silently-weaker substitute" posture
  `make_json_schema_validator` already takes for its own optional `jsonschema` extra.
- Two composed budgets, both factory (operator) parameters, never model-controlled:
  `per_match_timeout_s` (default `1.0`) bounds ONE line's match — a timeout skips just that line
  (counted, surfaced in the result, never silent); `max_total_time_s` (default `30.0`) bounds the
  WHOLE call, checked before EVERY line (not merely once per file — a per-file-only check would let
  a single large file with many timeout-tripping lines blow past the budget by an arbitrary
  multiple before it ever fired again).
- **`output_mode=` and `ignore_case=` on `grep_files`.** `output_mode` (default `"content"`,
  unchanged) adds `"files_with_matches"` (distinct matching file paths only, no line text) and
  `"count"` (`path: N`, files with zero matches omitted). `max_results` caps the number of MATCHES
  found (not total output lines — see `context_before`/`context_after` below). `ignore_case`
  (default `False`) is case-insensitive matching as a first-class flag rather than something baked
  into the pattern.
- **Per-file early-break, `"files_with_matches"` only.** That mode only needs to know "did this
  file have ≥1 match" — scanning stops the instant one is found, moving to the next candidate
  file. `"count"` mode cannot do this — it needs the file's exact total match count, so every line
  there is still scanned. Both `"files_with_matches"`/`"count"` additionally stop OPENING further
  candidate files once `max_results` qualifying files are already found (an outer-loop break — it
  never skips a line of a file already being scanned, only avoids starting further ones).
- **`context_before=`/`context_after=`** (default `0`, `"content"` mode only — a silent no-op,
  never an error, in the other two modes, which have no per-line text to attach context to): show
  that many unchanged lines immediately before/after each match, using grep's own convention — a
  match keeps `path:line: text` (colon); a context line uses `path-line- text` (hyphen); a `"--"`
  line separates two blocks that don't touch (a numbering gap) within the same file — never at a
  file boundary, since the path prefix already marks that. A line that itself matches is ALWAYS
  emitted as a match, never as leftover context from an earlier match's `context_after` window — a
  fresh match resets the after-context countdown outright, it never stacks with one still running.
  `max_results` counts MATCHES only — context/separator lines are supplementary and uncapped by
  it, mirroring real `grep -m`; when a match hits the cap, its own trailing context may be
  truncated if the file/budget ends first (an accepted, deliberate simplicity choice, the same one
  `max_total_time_s` already makes for whatever's in-flight). When both are `0` (the default),
  behavior — including the traced `result_count` — is byte-identical to a build of this tool with
  no context support at all.

## Writing and editing a bounded local directory (`tools/edit.py`)

`make_write_file_tool(root)` / `make_edit_file_tool(root)` — the write side of the read/search
pair above, kept in a separate module (`fs.py` is already the largest single file in `tools/`, so
"everything that can mutate the filesystem" stays physically distinct from "everything that only
reads it"). Same `resolve_within_root` guard, same `name=`/`encoding=` parameters and validation
as `make_read_file_tool`/`make_grep_files_tool` — including the same multi-root name-collision fix.

```python
from rlm_harness.tools import make_edit_file_tool, make_write_file_tool

write_file = make_write_file_tool(repo_root)
edit_file = make_edit_file_tool(repo_root)
finding = MyTask(tools=[read_file, grep_files, write_file, edit_file]).run(...)
```

- **`make_write_file_tool`**: creates or overwrites a whole file, atomically
  (`atomic.atomic_write_text`). Unconditional overwrite (no create-only mode — a consumer wanting
  one can call `read_file` first and check for its "missing file" error string) and no
  `max_content_chars` cap (the content was generated by the model itself, so it's already bounded
  by whatever produced it — this does NOT cover a model looping on the tool and filling the host's
  disk with many individually-bounded files, an accepted, out-of-scope-for-now gap).
- **`make_edit_file_tool`**: exact-string-anchor replacement — the same uniqueness-checked
  contract Claude Code's own `Edit` tool and `nano-rlm`'s `edit` skill both use independently.
  Refuses (never mis-edits) if `old_string` isn't found, or is found more than once and
  `replace_all` (a per-call flag the model sets, not a factory-level fixed behavior) is `False` —
  the file is left byte-for-byte untouched on every refusal path. `old_string == ""` and
  `old_string == new_string` are refused as degenerate inputs; `new_string == ""` (delete this
  text) is a legitimate operation and is NOT refused. **Known failure mode**: a file using
  different line endings than the `old_string` the model supplies fails closed with "not found"
  rather than mis-editing — not a safety bug, just worth knowing.
- **This is the kit's first file-mutation/data-loss-capable tool category** — every tool shipped
  before this was either read-only against the filesystem or delegated execution/network entirely
  to a consumer-supplied runner/fetcher. Both factories build on `atomic_write_text`, which is
  what makes an overwrite/edit crash-safe (never half-written) — but its guarantee is "no torn
  read," never "serializes concurrent read-modify-write across processes": two SEPARATE `RLMTask`
  runs (or two workers in a batch eval) racing an edit on the SAME file can silently lose one of
  the two updates. Accepted, out-of-scope-for-now — stated here rather than silently omitted.
  Building `make_write_file_tool`/`make_edit_file_tool` on `atomic_write_text` also surfaced a
  real bug in that primitive (fixed in `atomic.py` itself, not duplicated per-tool): overwriting
  an existing file through it used to silently reset the file's permission bits to `0600`
  (`tempfile.mkstemp` always creates its temp file at that mode, and `os.replace` doesn't carry
  the destination's mode across) — now the destination's existing mode is preserved across an
  overwrite.

## Environment interpreter (`interpreter="container"`)

The default `pyodide`/`deno` interpreter is WASM Python — it **cannot spawn a subprocess** (emscripten
has no processes). When a task needs the model to run real commands as part of its own reasoning — a
build, a test, `git`, a compiler — set `RLM_INTERPRETER=container` (or `RLMConfig(interpreter="container")`).
The RLM's REPL then runs **inside an isolated Docker container**, so the model's own Python can
`subprocess.run(...)` natively and hold real filesystem/process state.

```bash
docker pull python:3.11-slim
export RLM_INTERPRETER=container         # default stays pyodide; this is opt-in per run
```

- **One persistent container per run.** State persists across REPL turns within a run (the model can
  write a file in one cell and run it in the next), and the container is torn down at run end. This is
  the *environment* model — distinct from the [`run_command`](#running-local-commands-an-isolated-runner)
  tool, which is a model-*chosen* command per call against a runner you supply (and whose reference
  example is a fresh container per call). Use the container interpreter when the REPL itself needs a
  real environment; use `run_command` when commands are occasional, tool-like actions.
- **A stronger boundary than pyodide for this case, not a weaker one.** `--network=none` makes the
  host↔container stdio broker the only channel in or out (no egress); the LM credentials never enter
  the container — `llm_query` and tool callbacks execute **host-side**, only results cross the pipe;
  Linux capabilities are dropped (`--cap-drop=ALL`) and memory/pid are capped. It is the *opposite* of
  the refused `local` interpreter (host RCE), not a relaxation of it.
- **Config** (`RLM_CONTAINER_*`, all optional): `IMAGE` (default `python:3.11-slim`), `TIMEOUT`
  (per-cell sandbox-compute budget, s, default 120 — host tool time is not counted), `MEMORY`
  (`512m`), `PIDS_LIMIT` (`256`), `NETWORK` (`none`), `CPUS` (unset = uncapped), `CAP_DROP` (`true`),
  `READ_ONLY` (`false`; opt-in read-only rootfs for an inspect-only task, paired with a tmpfs `/tmp`),
  `WORKDIR` (a host dir mounted **read-only** at `/workspace`).
- **Needs the `docker` CLI** (checked at start; `import rlm_harness` stays docker-free). The `WORKDIR`
  mount resolves on the *daemon's* filesystem, so it won't work with a remote `DOCKER_HOST`.

## Sandbox turn timeout + cancellation (`pyodide`/`deno`)

The default `pyodide`/`deno` interpreter blocks on a plain subprocess pipe read with **no timeout
anywhere in dspy's own code** — a wedged Deno subprocess, or a model-written REPL cell that spins
forever, hangs the run with no recourse short of killing the whole process. `asyncio.Task.cancel()`
cannot help: the blocking call has no `await` inside it, so the event loop never gets a chance to
run cancellation machinery. rlm-harness closes this with the SAME timer-armed-before-blocking-read,
kill-to-unblock idiom the container interpreter already uses for its own `TIMEOUT` (above), ported
to `pyodide`/`deno`:

- **`RLM_SANDBOX_TURN_TIMEOUT`** (seconds; `RLMConfig.sandbox_turn_timeout_s`) — a per-`execute()`
  safety-net deadline. **Unset by default** (deliberately NOT matching the container interpreter's
  own `120.0`-default precedent: this budget has no hook to exclude host-side tool/sub-LM dispatch
  time, so a generous always-on value would misfire on legitimate multi-tool-call turns more often
  than the container analogy implies). Firing raises dspy's own RECOVERABLE
  `CodeInterpreterError` — the model sees an `"[Error] ..."` string and gets to retry next turn
  against a freshly-respawned sandbox.
- **`cancel_event`** (`RLMTask(cancel_event=a_threading.Event)`) — for a caller that wants to stop
  an in-flight run NOW (e.g. a "Cancel" button in a UI driving `arun()` from a worker thread). Set
  the event from another thread; the current sandbox turn is killed and `SandboxCancelled` (exported
  from `rlm_harness`) propagates all the way up through `arun()` as a genuine, NON-recoverable run-ending
  failure — never retried (see `run_with_retry`'s `non_retryable` below), never caught by dspy's own
  `except (CodeInterpreterError, SyntaxError)`.

```python
import threading
from rlm_harness import RLMTask, SandboxCancelled

cancel = threading.Event()
task = MyTask(cancel_event=cancel)
# from another thread: cancel.set()
try:
    result = await task.arun(q="…")
except SandboxCancelled:
    ...  # the run was stopped on purpose; not a failure to log as one
```

Both knobs are `None` by default and cost nothing when unset: no watcher thread is ever created, and
`execute()` is byte-identical to before either knob existed. `run_with_retry`'s `non_retryable`
parameter (a closed allowlist of exception types that propagate verbatim, consuming no attempt and
never wrapped in `RLMTaskError`) is what makes `SandboxCancelled` survive `RLMTask.arun()`'s own
retry engine untouched — a caller-driven cancellation must never be silently absorbed by a retry
that respawns the sandbox and restarts the whole trajectory from scratch.

## Grounded completeness — the sufficiency-critic recipe

A convention, not an API. When the RLM generates an artifact that must MATCH a retrieved
ground-truth (a spec, a contract, a source document), "am I done?" is the dangerous judgment:
a model asked to self-assess from memory will call a half-right artifact complete. There is
often no deterministic check for CONTENT correctness — a validator catches *structure/format*,
but not "this request is missing a required header" or "this answer skipped a clause".

The fix (the agentic-RAG *sufficient-context* pattern) is to GROUND the completeness judgment in
the retrieved source instead of the model's recall:

1. **Hold the ground-truth in REPL state.** Fetch the source once (a `fetch_url` tool, a skill)
   and keep it as a REPL variable — rlm-harness's interpreter persists variables across turns, so the
   ground-truth stays addressable without re-fetching or re-pasting.
2. **Diff the artifact against it, itemized.** Each turn, compare the generated artifact to the
   held ground-truth field-by-field and emit the SPECIFIC gaps ("missing header X, body field Y"),
   not a yes/no verdict.
3. **Regenerate on the gaps; finalize only when the diff is clean** (or the gap was escalated to a
   sub-LM and confirmed unobtainable). The itemized gap-list is a far stronger regeneration signal
   than a generic "make it complete".

This lives in the consumer's task INSTRUCTIONS (it is an LM-decided REPL action, kept in the
trajectory as honest RL data — same reasoning as keeping tools/skills LM-decided), and it needs no
new model: the main LM critiques cheaply against its own REPL state, reserving a sub-LM escalation
for a genuine knowledge gap. A consumer uses it so the planner stops finalizing a generated artifact
whose content only *looks* right — diffing it against the retrieved source held in the REPL.

## Judgement-only SUBMIT — assemble facts, don't let the policy report them

A convention, not an API — the companion to grounded completeness, for the *other* side of a
model-backed tool. When a `make_model_tool` (or any tool) is the AUTHORITATIVE producer of an
artifact, the root LM's final SUBMIT must not re-carry that artifact. Two failure modes if it does:

- **Mangling.** A root LM that re-types the tool's output into its result can corrupt it (re-indent,
  drop a nested block) — and nothing re-checks the re-typed copy, so a `valid=True` the LM *also*
  self-reports can label bytes that no longer pass the validator.
- **Trajectory poison.** The SUBMIT turn IS a training sample (`export_sft_turns`). If it re-authors
  the artifact, the policy learns to re-author it — exactly the job you gave the tool. And a
  self-reported validity flag becomes a label that can LIE: a downstream keep-filter
  (`complete and valid`) then keeps runs whose artifact is actually invalid.

The fix: keep DETERMINISTIC facts out of the policy's output type entirely.

1. **The `output_model` carries JUDGEMENT + a reference KEY, not the artifact.** The root LM SUBMITs
   its decisions (is this complete? what's missing? which variant?) and the producing tool-call's id —
   never the artifact bytes or a `valid` flag. With no field for it, the policy *cannot* re-type it,
   and the SFT turn stays clean.
2. **Assemble the artifact + its validity on READ, from the trace.** A small `assemble(result, events)`
   step re-sources each artifact VERBATIM from the matching tool-call event (by the id; last accepted
   wins) and DERIVES validity from the validator — never the policy's self-report. Run it everywhere
   the result is consumed: the live path, re-render, and the dataset exporters (`export_sft_turns` /
   `export_rl`), so the training labels read facts too.

   *Caveat when the validator CANONICALIZES, not just verdicts.* If the validator only returns
   pass/fail, verbatim re-sourcing is exactly right. But a validator that also REWRITES its input to a
   canonical form (stamps a fixed provenance field, strips a fabricated token) makes the raw draft and
   the canonical output DIVERGE — and re-sourcing the raw then ships the un-corrected bytes, so the
   deliverable silently misses a fix the run already applied (the root LM saw the corrected version, but
   the assembled artifact carries the raw one). Ship the validator's CANONICAL output as the artifact
   and derive validity from those same bytes; the tool-call still records the model's raw draft, so the
   trace stays faithful for RL while the deliverable stays canonical. Both are byte-identical no-ops when
   there is nothing to correct, so this costs nothing in the common case.

So the trace records the policy's real ACTION (its judgement), deterministic truth is COMPUTED (never
stored as if the policy produced it), and a self-reported flag can never drift from the bytes it
labels. Old traces heal on read: a pydantic `output_model` ignores the legacy artifact/validity keys
when coercing to the judgement-only type, and `assemble` re-sources them. A consumer does this — the
planner SUBMITs a per-artifact judgement keyed by the artifact's id; the system attaches the generator's
verbatim output and the validator verdict, so a re-typed/mangled artifact and a lying `valid` are both
structurally impossible.

**Corollary — the `run_start` meta must self-describe the run's CONFIG.** An OFFLINE, config-free
consumer (a dataset exporter, a re-renderer) can only read what the trace records. So any per-run
config it needs to INTERPRET the run — the expected value a validator enforced, the budget a
`hit_iteration_cap`-style metric compares against, the model roles — belongs in the `run_start` meta
the recorder writes, NOT a hardcoded default the reader guesses. Then an env override is honored
end-to-end (live AND in the offline labels), and an old trace lacking a key falls back gracefully.
This is the same principle as seeding `sft_turns` from the meta's initial state: the trace is the
sole source of truth for everything downstream of the run. A consumer records its canonical author
and `max_iterations` there so an offline reader reads the real per-run values, not the reference defaults.

## Building a consumer

`rlm-harness` is the ROLLOUT floor; a consumer is a thin declaration on top of it. `examples/harness_run.py`
is a minimal worked example — a task that wires the sub-LM hook, skills, tracing, and
RL export together. Five steps:

1. **Declare the task.** Subclass `RLMTask`: a `signature`, `output_field`, an `output_model`
   (judgement-only — see above), `instructions` (orchestration + a few hard safety rules), and
   `tools`. The retry/validation loop, sandbox, budget caps, and the trace are inherited. Put
   authoring KNOWLEDGE in a Skills directory (`load_skills_as_tools`), not the prompt — the prompt
   is for orchestration; skills are progressive-disclosure reference the LM pulls on demand.
2. **Add tools the base/wrap way.** Need a new capability (a model-as-tool producer, a fetcher, a
   searcher)? rlm-harness owns the GENERIC base + the syntactic guard + the async-safe factory
   (`make_model_tool`, `make_fetch_tool`, `make_web_search_tool`); the consumer owns the PROVIDER
   (the endpoint/validator/messages, or the httpx/vendor call) and the project-side TRACING. Tools
   passed to `RLMTask(tools=…)` MUST be sync — dspy's interpreter calls them with a plain `()`, so
   an `async def` tool returns an un-awaited coroutine and never runs.
3. **Pick the recursion seat deliberately.** A DETERMINISTIC transform of the sub-LM's output →
   `intercept_sub_lm` (the escalation seat, recorded as a `sub_call`). An action the main LM CHOOSES
   to take → a tool (`tools=`, recorded as a `tool_call`). Don't smuggle a model-judgement (asking
   another model to grade the output) into the sub-LM intercept — that is an agentic decision and
   must be a tool, so it lands in the trajectory as honest RL data. (See "Sub-LM vs. tool".)
4. **Record + read through the trace.** Run inside a `TraceRecorder` (`on_event` gives a live
   observer for streaming). EVERYTHING downstream — your report renderer, your dataset, a re-render
   of a past run — reads the JSONL trace, never the live objects. Carry any per-run config you'll
   need OFFLINE into the `run_start` meta (the corollary above), and assemble deterministic facts on
   READ (judgement-only SUBMIT), so a label can never drift from the bytes it describes.
5. **Export trajectories; score elsewhere.** `export_sft_turns` / `export_rl` / `export_actions`
   turn traces into training datasets. They are REWARD-FREE: each carries a `reward=` HOOK the
   trainer fills — rlm-harness never computes a reward.
6. **Delegate to another harness — or be one.** When a sub-task is better handled by a more
   specialized rlm-harness harness, delegate to it as a TOOL rather than reimplementing it. Two symmetric
   sides, both base/wrap, and NEITHER names the other harness in code — the identity lives only in the
   operator's runtime endpoint config:
   - **Client (you call another harness):** `make_harness_tool(invoke_fn, validate)` +
     `harness_from_endpoint(call_endpoint, read_output=…)`. The kit owns retry/validate/circuit-break +
     the child-rollout link; you own the transport (`call_endpoint` — a subprocess command / HTTP URL)
     and `read_output` (parse the child's reply). The single long-text arg becomes the child's RLM
     environment; the parent records ONE `tool_call` + a `child_run_id`/`child_trace` link while the
     child owns its own separate rollout (exported independently).
   - **Server (another harness calls YOU):** add a ~5-line `<pkg>/serve.py` that calls
     `serve_harness(run, to_pointer)`. The kit owns stdin→env, run_id, CWD isolation, the JSON-pointer
     wire, exit codes (0=ran / 1=infra→caller retries), and keeping your logs + tracebacks OFF stdout;
     you own only `to_pointer` — the mapping from YOUR result object into a `HarnessPointer`. The
     operator points the client at `python -m <pkg>.serve`. A FLAT result (`.artifact`/`.run_id`) needs
     NO file — `python -m rlm_harness.harness_serve <pkg.module>:run` uses the duck-typed default. Copy
     `examples/harness_serve.py`.
   - **Give the operator an ABSOLUTE `workdir_base`.** `serve_harness` isolates each run's CWD under
     it, and the default is RELATIVE — a child inherits the PARENT's working directory, so its run
     folders materialise inside the caller's project. Document an absolute path in your serve module
     (`python -m <pkg>.serve /var/tmp/<pkg>-harness-runs`), and have callers ignore `harness-runs/`.
   - **Your `run` must match the contract's shape, or adapt to it in `serve.py`.** `serve_harness`
     calls `run(source: str, run_id=…)`. A harness whose entry takes a domain object (a resolved lead,
     a parsed spec) or derives its own id needs a small adapter there: resolve the caller's text into
     that object through whatever PUBLIC seam the harness already exposes, and absorb the kit's
     `run_id` when the harness has a more meaningful one of its own — report the real id back through
     `to_pointer` so the parent's child-link still resolves. RAISE from the adapter when the text
     resolves to nothing: that is exit 1, which the caller retries then degrades. Returning an empty
     artifact instead is exit 0, and buys the caller a full run over nothing.
   - **A multi-file deliverable: use `bundle_artifact`, never a format of your own.** `artifact` is
     ONE string, but plenty of harnesses produce a FOLDER (a write-up + a PoC + a diff; a Dockerfile +
     a compose file + notes). `bundle_artifact({name: content})` packs it as `===== <name> =====`
     sections — escalating the marker when a file's own content contains one, so a report that QUOTES
     a bundle cannot truncate itself — and `parse_artifact_bundle` reads it back. Line endings are
     normalised to `\n` at pack time (both halves must agree on what a LINE is, or a header hides from
     escalation and still acts as a section break), and a filename that cannot round-trip raises
     rather than vanishing. Use them: a packing format invented per harness/client pair is a
     silent-failure generator, because the two sides agree until they don't and the mismatch then
     degrades into "the child returned junk" instead of surfacing as the wiring bug it is. A client
     that just wants the whole deliverable as CONTEXT needs no parser at all — the text is meant to be
     read as-is, by a Root LM and by a human debugging the wire.

   - **In-process transport (no subprocess).** The kit still ships no `call_endpoint` — subprocess
     (`serve_harness`) and HTTP remain the usual choices — but a THIRD option, for a trusted child
     harness in the same process/deployment where low latency matters more than OS-level isolation,
     is to await the child's `RLMTask.arun()` directly. Two small primitives make this easy to build
     correctly: `rlm_harness.tools.run_isolated(coro_factory)` bridges the sync tool-call contract
     into the child's `async arun()` — always on a dedicated new thread, so it works regardless of
     whether the calling thread already has a running loop (it will, whenever the parent task is
     itself mid-`arun()`) — and `rlm_harness.tools.pointer_to_invocation(pointer)` is the canonical
     `serving.HarnessPointer` → `tools.HarnessInvocation` mapping, reused unchanged from the
     subprocess/HTTP case. **Read `run_isolated`'s docstring before wrapping a traced delegation**: a
     fresh thread starts with an empty `contextvars.Context`, so the delegated child's OWN
     `TraceRecorder` must be entered INSIDE the coroutine `run_isolated` runs, never around the call
     to `run_isolated` itself — a `TraceRecorder` entered outside is invisible to `current_recorder()`
     inside, and the child's own tool_calls/sub_calls would go silently unrecorded. See
     `examples/harness_local_run.py` for the full worked pattern (protected offline by
     `tests/test_harness_tool.py::test_in_process_transport_wiring`, which exercises the identical
     composition with a stub child instead of a real `dspy.RLM`). Nothing here adds a NEW
     code-execution surface — the child still enforces its own `RLMConfig`/sandbox guard on its own
     REPL code, exactly as it would over any other transport; an in-process call only changes HOW a
     Python object gets invoked, not WHAT gets executed where. Keep the subprocess/HTTP transport
     instead when the child needs real process/OS isolation, a different runtime/language, or truly
     runs on a remote machine.

**Score your own rubric (optional).** To decompose "did this run succeed?" into observable per-run
LABELS, `rlm_harness.rubric` gives you the reward-free substrate — the `Criterion`/`RubricCriteria`/
`CriterionFact` types, `rubric_to_meta`/`rubric_from_meta` (carry the rubric in the `run_start` meta),
`validate_rubric`, and a pure `criteria_facts(criteria, facts, lens)`. `category` is an OPAQUE label YOU
define — the kit imposes no taxonomy. The pattern (all consumer-side except the primitives):
- define your own category set + a fixed (or per-task) criterion skeleton (`default_rubric`);
- write a `trace -> facts` function — reuse your OWN run labels/metrics so a criterion's facts can never
  drift from the export bundle — and a `category -> keys` lens choosing which facts each category surfaces;
- `criteria_facts(rubric_from_meta(events).criteria or default_rubric().criteria, trace_facts(events),
  LENS)` → per-criterion facts, reward-free. Emit them beside the trajectory via
  `run_label_bundle(runs, rubric=lambda ev: {...})`; a downstream trainer turns facts into a score.

An OPTIONAL model-graded EVAL is the same base/wrap shape — rlm-harness ships NO eval, only the pieces to
build one: wrap `make_model_tool` with YOUR judge prompt (100% your domain), a strict parser reading a
per-category 0–10 score dict, and a per-category means aggregation. Keep the prompt, the taskset, and the
category MEANINGS in your repo; the categories stay OPAQUE to the kit. This is a reference PATTERN, not a
shipped module — the valuable part of an eval is the domain prompt, which cannot be made generic without
emptying it.

**If you ship an in-repo `studio/` (or any workspace member that drives live runs), forward the
subscription extra.** A consumer's visual console is a uv workspace MEMBER with its own
`pyproject.toml`, kept behind a `live` optional extra so a replay-only deploy stays web-free. A
studio-scoped `uv` command (`uv run --package <consumer>-studio …`) resolves `--extra` against the
MEMBER, not the root — so the member must define BOTH `live = ["<consumer>"]` AND a forwarding
`subscription = ["<consumer>[subscription]"]`. Without the forward, `--extra subscription` is rejected on
the member and any sync that omits it prunes the Claude Agent SDK back out, so a subscription studio run
dies with `ImportError: ClaudeAgentLM requires the optional dependency`. The portable, cwd-independent
launch is therefore `uv run --package <consumer>-studio --extra live --extra subscription uvicorn
<consumer>_studio.app:app`. Every downstream studio carries this SAME pair, so one command is portable
across them.

**The promotion rule** keeps the boundary clean. When the consumer forces a workaround, ask "is this
GENERIC?" A reusable mechanic (the model-tool + retry + validate core, a new sandbox seam, a trace
hook) is PROMOTED into rlm-harness via the base/wrap split — the generic half here, the specific half in
the consumer. A consumer-specific VALUE (a model name, a schema, a validator, a path) stays in the
consumer. Never special-case the consumer inside the kit; never fork the harness or re-implement
tracing inside the consumer. If you need an internal seam the kit doesn't expose, ADD a public hook
here (that is how `recorder_scope` / `bind_recorder_to_sub_lm` / `get_sub_lm` were born) rather than
reaching into a `_private` name. The trace schema, the `EVENT_*` types, and the exporter record shapes
are a FROZEN v1 wire format — `tests/test_contract.py` pins them; adding an optional field is fine,
removing or re-typing one is a `v2` break. The `EVENT_*` type constants are exported from `rlm_harness`,
so a trace reader matches on `rlm_harness.EVENT_RESULT` instead of hardcoding the wire string `"result"`.

**The stage boundary** keeps the data honest. rlm-harness + your consumer are the ROLLOUT stage: they
produce trajectories (the trace) and turn them into datasets, emitting raw LABELS / METRICS, never a
reward scalar. SCORING (reward composition, credit assignment) and TRAINING (GRPO / SFT) are a
SEPARATE downstream project that installs the trainer. A prompt/policy rule that makes the rollouts
BETTER is in scope; a reward or penalty is not. Keep the trace clean training data and let the
trainer score it.

## Trace utilization metrics

`rlm_harness.metrics` answers "how was this run's activity distributed" — a sibling question to
`rubric.py`'s "does this run satisfy criterion X," equally reward-free, but a fixed COMPUTATION
over the raw events rather than a caller-supplied fact-slice:

```python
from rlm_harness import compute_run_utilization, group_by_run, load_events

runs = group_by_run(load_events("traces/run.jsonl"))
u = compute_run_utilization(runs["r1"])
print(u.main_steps, u.tool_calls_total, u.tool_calls_by_name, u.sub_calls_total)
print(u.tool_call_rate, u.sub_call_rate)   # per root-LM turn taken; None if main_steps == 0
```

`compute_utilization_by_run(events)` computes every run's `RunUtilization` in one call, for a
batch/dataset-level view. Reads ONLY already-frozen `trace/v1` fields (`event["type"]`,
`event["payload"]["tool"]`) — no new event type, no new payload field, nothing the trace contract
needs to change for.

Both rates are denominated over `main_steps` (root-LM turns) — "how many tool calls / sub-LM
escalations happened per root-LM turn taken." This is a judgment call, not a uniquely correct
answer: the raw counts are exposed alongside the rates, so a consumer wanting a different
denominator can recompute one from the same fields. A rate is `None` (not `0.0`) when
`main_steps == 0` — `0.0` would misleadingly read as "measured and found to be zero usage" rather
than "undefined," and a crashed/cancelled run that failed before its first `Prediction` ever
returned is a real, reachable example of this: it can carry live-recorded `tool_call`/`sub_call`
events with zero `main_step` events (`RLMTask.arun()` only records the main trajectory `if
"prediction" in captured`).

## Configuration

All via env (`RLMConfig.from_env()`): `RLM_MAIN_MODEL` (or `AI_MODEL_NAME`),
`RLM_SUB_MODEL` (or `SUB_AI_MODEL_NAME`), `RLM_API_KEY` (or `AI_API_KEY`),
`RLM_BASE_URL` (or `AI_BASE_URL`), `RLM_INTERPRETER`, `RLM_ADAPTER`,
`RLM_MAX_TOKENS`, `RLM_MAX_OUTPUT_CHARS`, `RLM_ALLOW_INSECURE_SANDBOX`,
`RLM_MAX_ITERATIONS`, `RLM_MAX_LLM_CALLS`, `RLM_MAX_RETRIES`, `RLM_OBSERVE`.

The `AI_*` fallbacks let the kit drop into projects already keyed on those vars
without re-keying env; the `RLM_*` form wins when both are set.

**Injecting a pre-built LM.** `configure(cfg, main_lm=…, sub_lm=…)` uses a supplied LM
verbatim instead of constructing one from `cfg` — a `dspy.utils.DummyLM` in tests, or a
cached / custom client in production. It's the public seam for a test double, so nothing
has to reach into private runtime state; read the active config back with `get_config()`.

**Claude-subscription auto-routing.** Set `RLM_MAIN_MODEL`/`RLM_SUB_MODEL` (or pass `main_model=`/
`sub_model=` on `RLMConfig`) to `claude_agent_lm.SUBSCRIPTION_PREFIX` + a model id — e.g.
`claude-agent-sdk/claude-sonnet-5` — and `configure()` builds a `ClaudeAgentLM` for that role
automatically; no explicit `main_lm=`/`sub_lm=` wiring needed. An explicit `main_lm=`/`sub_lm=`
kwarg still wins outright regardless of the model string — the prefix is only consulted for a role
left unset. This can raise, in addition to `configure()`'s own errors: `ValueError` for a bare
prefix with no model id; `RuntimeError` if `ANTHROPIC_API_KEY` is set (`ClaudeAgentLM` refuses to
start while it's set — the Claude Code CLI silently prefers it over subscription OAuth); or
`ImportError` with an install hint if the optional `rlm-harness[subscription]` extra isn't
installed. Needs the same setup as `ClaudeAgentLM` itself (see the module's own docstring) —
building one by hand via `main_lm=ClaudeAgentLM(...)` still works exactly as before, for anyone who
wants to pass its other constructor kwargs (`timeout_s`, `cwd`, …) explicitly.

**Model names with a custom endpoint.** When `RLM_BASE_URL` is set, `configure` pins
litellm's `custom_llm_provider="openai"`, so the model names are the **plain id your
endpoint serves** — e.g. `qwen/qwen3-next`, not `openai/qwen/qwen3-next`. (dspy.LM runs on
litellm, which otherwise reads the first path segment as a provider and fails on a bare id;
the pin routes everything via the OpenAI wire protocol to your `base_url`.) A prefixed
`openai/...` name still works. With no base_url, write litellm's own prefix (`openai/gpt-4o`,
`anthropic/claude-...`).

`RLM_ADAPTER` (default `json`) picks how structured fields are coaxed out of the
model:

- **`json`** (default) — schema-guided structured output, end-to-end. A lenient `JSONAdapter` always
  sends the `json_schema` response format directly — bypassing dspy's `supports_response_schema`
  gate, so no `litellm.register_model` poke is needed — and a brace-tolerant parse absorbs guided
  output that drops the outer `{ }`. Works on **any** structured-output endpoint — OpenAI-proper
  AND vLLM / NVIDIA NIM (which reject schema-less `json_object` but accept `json_schema`). The
  decoder enforces the schema, so it **yields valid output even from a weak / imperfectly-
  formatting model**.
- **`chat`** — `dspy.ChatAdapter` with the JSONAdapter fallback **off**: never sends
  `response_format`. For an endpoint with **no** structured-output support. Needs the model to
  follow dspy's text field-marker format reliably — the fallback is off (dspy's stock ChatAdapter
  recovers via bare `json_object`, which vLLM rejects), so a model that drops a field has no
  recovery. Not as portable as it looks.
- **`default`** — leave dspy's stock adapter (ChatAdapter *with* the json_object fallback):
  recovers on OpenAI-proper endpoints, but the fallback is rejected by vLLM/NIM.

`RLM_MAX_TOKENS` (default `8192`) is the per-call generation cap. It defaults generous rather
than deferring to the server: a **reasoning model** emits its chain-of-thought before the answer,
so a server's small default cap (e.g. 1000) truncates the thinking and returns **empty content**.
Set it higher for very verbose reasoning models, or `RLMConfig(max_tokens=None)` to defer to the server.

A **reasoning model can be the RLM root**, not just an instruct one: some reasoning servers emit the
*whole* structured turn into the `reasoning_content` channel and return `content` null. `_LenientJSONAdapter`
promotes `reasoning_content` to the answer when `content` is empty (guarded — a well-behaved model's
`content` always wins, so its native thinking stays discarded), which keeps the root's first turn from
dying on dspy's "empty or null response" check. The native chain-of-thought is still dropped from the
trajectory either way, so a reasoning root spends extra tokens the trace won't keep.

## Testing the forward path offline (`rlm_harness.testing`)

Construction tests (`task._build_rlm()`) catch signature/kwarg drift but never run the loop — where
wiring bugs actually hide (a prompt naming a tool `foo` while it registered as `foo_tool` is a
`NameError` no construction test sees). `rlm_harness.testing` drives the **real `dspy.RLM.aforward` loop
offline** — no model, no Deno, no network:

```python
from rlm_harness import RLMConfig, RLMTask, configure
from rlm_harness.testing import ScriptedInterpreter, call, scripted_lm, submit

configure(RLMConfig(main_model="x", sub_model="x", interpreter="mock"),
          main_lm=scripted_lm([                         # the planner's canned turns
              {"reasoning": "call the tool", "code": "print(my_tool(x=1))"},
              {"reasoning": "submit", "code": "SUBMIT(answer={'x': 5})"}]),
          sub_lm=scripted_lm([{"reasoning": "r", "answer": "{}"}]))

task = MyTask(interpreter=ScriptedInterpreter([          # one step per planner turn
    call("my_tool", x=1),                                # dispatch the REAL injected tool (traces a tool_call)
    submit({"answer": {"x": 5}})]))                      # SUBMIT — terminates the loop, coerces the result
result = await task.arun(q="…")                          # the whole planner→tools→result chain, offline
```

`dspy.RLM` injects the run's real tools onto the scripted interpreter's `.tools`, so a `call(...)` step
runs the actual tool (its tracing records a genuine `tool_call`); a `dict`/`submit(...)` step SUBMITs.
The `interpreter=` kwarg is an injection seam (like `sub_lm=`): an explicit interpreter OBJECT overrides
`config.interpreter` and — like an injected `DummyLM` — bypasses `build_interpreter` and its guard, so it
is a test/advanced seam; the default string path keeps the guard. `rlm_harness.testing` imports dspy lazily,
so it doesn't affect `import rlm_harness`. `cancel_event=` (above) has NO effect when `interpreter=` is also
given — a caller supplying their own interpreter object owns its cancellation behavior too, exactly like
`ScriptedInterpreter` owns its own.
