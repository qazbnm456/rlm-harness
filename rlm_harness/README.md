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
| `_retry.py` | Validation + retry engine (dspy-free, unit-tested). Mostly private, but `short_error` is public — head-and-tail elision for a caught exception, so a giant `AdapterParseError` (which embeds the whole raw completion) becomes one readable log line instead of thousands. |
| `sandbox.py` | Interpreter selection + the insecure-sandbox guard. |
| `atomic.py` | `atomic_write_text` / `atomic_write_stream` — a same-directory temp file + `fsync` + `os.replace`, so a concurrent reader never sees a partial write; `atomic_write_stream` takes an iterable of `bytes` chunks instead of one in-memory blob, aborting once a running total exceeds an optional `max_bytes`. dspy-free. |
| `metrics.py` | `RunUtilization` / `compute_run_utilization` / `compute_utilization_by_run` / `ToolWaste` / `compute_tool_waste` / `compute_tool_waste_by_run` — reward-free trace utilization metrics (how a run's activity split across root-LM turns, tool calls, sub-LM escalations), a pure derived read over already-recorded `trace/v1` events. dspy-free. |
| `isolation.py` | `run_in_subprocess` — a safe, isolated-subprocess primitive for a web-facing consumer: run one picklable callable in a fresh OS process, get its result or a clear error back, bounded by a timeout (see below). dspy-free. |
| `tools/` | `make_schema_validator` (pydantic) + `make_json_schema_validator` (validate a parsed object against a vendored JSON Schema — the base for the "validate against an official, version-pinned upstream schema" pattern; needs `rlm-harness[jsonschema]`), SSRF-guarded `make_fetch_tool`, its filesystem-side analogue `make_read_file_tool` / `make_grep_files_tool` / `resolve_within_root` (needs `rlm-harness[grep]` for a wall-clock-bounded `grep_files` — see below) plus the write side `make_write_file_tool` / `make_edit_file_tool` in `tools/edit.py` (see below), `list_candidate_paths` — a safe, `.gitignore`-aware default for building `candidate_paths` in `tools/discover.py` (needs `rlm-harness[gitignore]`; see below), `make_git_clone_tool` — safe git clone with fallback auth over a consumer-supplied isolated `cloner` in `tools/git_clone.py` (see below), `make_extract_archive_tool` — safe `zip`/tar extraction in `tools/archive.py` (see below), `verify_quote` — a deterministic quote/citation grounding check in `tools/grounding.py` (see "Grounded completeness" below), provider-agnostic `make_web_search_tool`, `make_command_tool` — a traced `run_command` over a consumer-supplied *isolated* runner (the kit ships no executor) with an optional `refuse_broad_git_history` guard, `make_model_tool` — the generic "model-as-tool + transient-retry + validate" core (a project wraps it with its own endpoint/validator/messages), and the harness-delegation pieces `make_harness_tool` / `harness_from_endpoint` / `pointer_to_invocation` / `run_isolated` (see "Delegate to another harness" below). |
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

`dspy.RLM` exposes no hook to intercept a sub-LLM response, and (as of dspy 3.3.1) no
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
    for tool in cat.tools("docs"):     # RAW mcp Tool objects (name / description / input schema)
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

### `list_candidate_paths` — the recommended way to build `candidate_paths` (`tools/discover.py`)

`candidate_paths` stays a plain, required list — no contract change. But safely walking a
directory tree to build one (respecting `.gitignore`, never escaping `root` via a symlink, never
treating VCS internals as candidates) is exactly the kind of mechanic every consumer would
otherwise reinvent. `list_candidate_paths(root)` is a plain host-side function (no factory, not a
REPL tool — called from a consumer's own setup code before wiring a tool, the same role
`resolve_within_root` already plays) returning `CandidatePaths(paths, truncated)`, pipeable
directly into `make_grep_files_tool`/`make_read_file_tool` with zero reshaping:

```python
from rlm_harness.tools import list_candidate_paths, make_grep_files_tool, make_read_file_tool

candidates = list_candidate_paths(repo_root, glob="*.py")
read_file = make_read_file_tool(repo_root)
grep_files = make_grep_files_tool(repo_root, candidate_paths=candidates.paths)
```

- **Needs the optional `rlm-harness[gitignore]` extra (`pathspec`) ONLY when there's an actual
  `.gitignore` pattern to compile** — a real root `.gitignore` present (with
  `respect_gitignore=True`, the default) or a non-empty `extra_ignore_patterns`. A caller with
  neither never needs the dependency installed. `.gitignore`'s own subtler syntax (negation,
  directory-only patterns, anchoring) is why this uses `pathspec` rather than a hand-rolled
  parser — slightly wrong parsing either leaks a file a consumer explicitly meant to exclude
  (`.env`, credentials) or wrongly excludes real source; same "don't reinvent a
  correctness-critical mechanic" reasoning behind `make_grep_files_tool`'s `regex` extra and
  `make_json_schema_validator`'s `jsonschema` extra.
- **Root-level `.gitignore` only, stated honestly** — no nested per-directory `.gitignore`
  merging, no global gitignore. `extra_ignore_patterns` (same `gitwildmatch` syntax) is the escape
  hatch for a consumer that needs more than the one root-level file covers.
- **`.git` is always excluded, unconditionally, by two distinct mechanisms**: a directory named
  `.git` is pruned from the walk (never descended into), and — independently — a plain FILE
  literally named `.git` is also always excluded, since that's git's REAL submodule gitlink shape
  (a one-line pointer file, not a directory); pruning the directory case alone would miss it.
- **Every candidate file is re-checked through `resolve_within_root`**, independent of
  `follow_symlinks` (which only gates whether `os.walk` descends into a symlinked directory) — a
  symlink pointing outside `root` never surfaces as a candidate either way.
- **`max_files`** (default `5000`) bounds the walk itself, stopping it outright rather than
  slicing an unbounded result after the fact; `CandidatePaths.truncated` makes a partial result
  visible, never a silent cutoff. Directory and file names are sorted at every level of the walk
  first — `os.walk`'s own order is filesystem/OS-dependent, which would otherwise make WHICH files
  survive a truncation non-reproducible across runs on the same tree.
- **No REPL-tool wrapper yet** — this is a setup-time helper a consumer's own code calls, not
  something the model calls mid-trajectory; a `make_list_files_tool` wrapper is a natural,
  separately-reviewable future addition.

### `make_git_clone_tool` — safe git clone with fallback auth (`tools/git_clone.py`)

Base/wrap, the same shape as `make_fetch_tool`/`make_command_tool` — the kit does NOT shell out to
`git` itself. `command.py`'s own module docstring already explains why for `run_command`: a
model-adjacent operation executed host-side needs ISOLATION, and a `git clone` is not meaningfully
safer to run un-isolated than an arbitrary command (a malicious git server can exploit a client
vulnerability; a cloned repo's own hooks can execute code unless disabled). So
`make_git_clone_tool(root, cloner, *, name="git_clone", get_credentials=None, default_depth=1)`
takes a CONSUMER-SUPPLIED, isolated `cloner`:

```python
from rlm_harness.tools import make_git_clone_tool

def my_cloner(url, dest_path, depth, creds):
    # run inside your OWN isolation (a disposable container, an E2B/Modal sandbox, ...) --
    # the kit ships no executor here either, same posture as make_command_tool's Runner.
    ...
    return CommandResult(exit_code=0)

git_clone = make_git_clone_tool(repo_root, my_cloner)
finding = MyTask(tools=[git_clone]).run(...)
```

- **URL safety reuses `is_safe_url` directly** — no reinvented SSRF check. Syntactic pre-flight
  only, same caveat `fetch.py` already documents for its own `fetcher`: a public hostname
  resolving to a private address at actual connect time is NOT caught here, since the real
  network connection happens inside the isolated `cloner`, not in this wrapper — the `cloner`
  should call `resolved_host_is_safe` internally at connect time if that matters for its
  deployment.
- **Destination confinement reuses `resolve_within_root` directly** — `dest_dir` is resolved
  exactly like `write_file`'s `path`.
- **Fallback auth — a two-attempt orchestration, never a retry loop.** Tries without credentials
  first (the common, public-repo case costs nothing extra); on failure — a nonzero exit code OR a
  raised exception, handled identically — if a `get_credentials` provider was configured, ONE
  retry with credentials. Never more than two `cloner` invocations per call.
- **Credential redaction, disclosed as best-effort, not absolute.** A `get_credentials(url)`
  provider returning a dict MUST include a `"secret"` key (the raw credential string); after a
  credentialed attempt, that exact string is redacted (plain string replacement) before it can
  reach the traced `stderr_preview` or the model-visible return string. `stdout` is redacted too
  before its (post-redaction) length feeds the trace's `stdout_len` — `stdout` content itself is
  never traced verbatim, only that length, matching `run_command`'s own existing "lengths + a
  preview, not the full stream" posture. This does NOT catch a derived or transformed leak
  (URL-encoding, case-folding, a truncated echo from a misconfigured credential helper). A
  malformed dict (missing/non-string/empty `"secret"`), or a provider that itself raises, fails
  CLOSED — treated exactly like a decline, no retry attempted, never crashes the call.
- **`default_depth=1`** (shallow clone by default) is passed through to the `cloner` as a plain
  argument — the "avoid being tricked into cloning an enormous repository" mitigation;
  `default_depth=None` opts out for a caller that explicitly wants full history.
- **On success**: `"Cloned {url!r} into {dest_dir!r}."` — terse, matching `write_file`'s own
  success-string convention. A consumer wanting to see what landed already has
  `list_candidate_paths`/`read_file`/`grep_files` for that.
- **Accepted, disclosed gap**: no cleanup of a partially-cloned directory on failure — making the
  whole clone atomic would require dictating how the `cloner` itself writes to disk, contradicting
  the base/wrap split this design is built on.

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
- **On success, a windowed snippet of the result is appended** (`show_snippet=`, default `True`;
  reuses `read_file`'s own `f"{lineno:>6}\t{line}"` numbering) so a model can confirm what its
  edit actually did without a separate `read_file` round-trip — `show_snippet=False` is the escape
  hatch back to the terse confirmation alone. `snippet_context_lines=` (default `3`) bounds each
  shown region; an edit larger than that shows only its own head/tail with a visible "N line(s)
  omitted" marker, never an unbounded dump regardless of how large `new_string` was.
  `max_snippet_occurrences=` (default `3`) caps how many `replace_all=True` occurrences get their
  own snippet block — the file is still fully edited regardless of the cap; a truncated result
  says so explicitly ("N more occurrence(s) not shown"). Overlapping windows for closely-spaced
  occurrences are shown independently, not merged. Scoped to the success path only — a
  `Refused`/`Read error`/`Write error` string never gets a snippet appended.
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

## Extracting archives safely (`tools/archive.py`)

`make_extract_archive_tool(root)` — a safe `zip`/tar extraction tool. `zipfile.extractall()`/
`tarfile.extractall()` are not safe by default: a malicious entry can carry an absolute path, a
`..`-traversal path, or (tar) a symlink/hardlink pointing outside the extraction target ("zip
slip") — the same `resolve_within_root` reasoning `read_file`/`write_file` already apply to a
single path argument, generalized here to every entry of an archive.

```python
from rlm_harness.tools import make_extract_archive_tool

extract_archive = make_extract_archive_tool(repo_root)
finding = MyTask(tools=[extract_archive, read_file, write_file]).run(...)
```

- Supports `.zip` and tar variants (`.tar`, `.tar.gz`/`.tgz`, `.tar.bz2`/`.tbz2`, `.tar.xz`/
  `.txz`), dispatched by extension — an unrecognized extension returns an error string, never
  raises, never sniffs content.
- **Two-pass extraction, matching this kit's "refuse outright, never partially mutate" posture**:
  Pass 1 validates every entry's metadata ONLY (name, type, declared size, and — zip-only — the
  encryption/compression-method header fields) and refuses the WHOLE operation upfront on any
  violation, before a single byte is written — a zip-slip path, a symlink/hardlink/device entry,
  an encrypted entry, or an unsupported compression method all fail here, with nothing landing on
  disk. Pass 2 (only reached once Pass 1 fully passes) streams each entry's real bytes via the new
  `atomic_write_stream` primitive (below), bounding peak memory to a small, fixed chunk size
  regardless of that entry's own size.
- **A declared size cannot smuggle more decompressed output than it promises** — confirmed
  empirically: both stdlib read APIs hard-ceiling their output at the entry's own declared size
  field, so a "lying header" memory bomb does not exist via either. Pass 1's cumulative
  declared-size check (`max_extracted_bytes`, default 200 MiB) and entry-count check
  (`max_entries`, default 10,000) are factory (operator) parameters, never model-controlled — same
  posture `make_grep_files_tool`'s timeouts and `list_candidate_paths`'s `max_files` already take.
- **No password-protected/encrypted archives** — refused upfront in Pass 1, with a clear reason,
  never a crash. **No nested-archive recursion** — an archive found inside the extracted output is
  not itself auto-extracted; call the tool again on it, subject to the same checks a second time.
  **Unconditional overwrite** of existing files at the destination, matching `make_write_file_tool`.
- **`atomic_write_stream(path, chunks, *, max_bytes=None)`** (new, in `atomic.py`) — a separate,
  additive primitive alongside `atomic_write_text` (not a refactor of it): the same
  same-directory-temp-file/`fsync`/`os.replace`/permission-preservation idiom, but for a caller
  with an iterable of `bytes` chunks rather than one already-in-memory blob, aborting the moment a
  running total exceeds `max_bytes` — checked after every chunk, not merely once at the end.

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
- **`RLM_REQUEST_TIMEOUT`** (seconds; `RLMConfig.request_timeout_s`) — a wall-clock cap on ONE
  model HTTP request ATTEMPT, handed to `dspy.LM(timeout=...)` and from there to litellm. This is
  `RLM_SANDBOX_TURN_TIMEOUT`'s sibling on the other side of a turn: the sandbox side was bounded
  and the model side was not settable at all.

  **Unset is not "no cap".** With nothing passed, litellm applies its own
  `COMPLETION_HTTP_FALLBACK_SECONDS` of **600 s** per attempt. So this setting *replaces* that
  number rather than introducing a bound where there was none — and a consumer whose turns
  legitimately run longer than ten minutes must set it **up**, not leave it alone.

  **It does not bound a run to its own value.** dspy passes `num_retries=3`, and litellm's first
  call hands the OpenAI SDK `max_retries=2`, so a dead endpoint is retried: the run-level wait is a
  MULTIPLE of this value plus backoff. Size a caller-side budget on the multiple.

  What prompted it, and an honest reading of it: against a self-hosted OpenAI-compatible endpoint
  one request never came back — the socket stayed `ESTABLISHED` with both queues empty and the
  worker slept in `epoll_wait` for 38 minutes at 0.3% CPU, while that same endpoint answered
  unrelated requests in half a second. 600 s across four attempts is about forty minutes, so that
  observation fits "litellm's own default, retried" at least as well as "nothing was watching";
  no attempt counter was captured at the time. Either way the consumer could not choose the
  number, and now it can.
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
`execute()` takes the same direct path it did before either knob existed. `run_with_retry`'s `non_retryable`
parameter (a closed allowlist of exception types that propagate verbatim, consuming no attempt and
never wrapped in `RLMTaskError`) is what makes `SandboxCancelled` survive `RLMTask.arun()`'s own
retry engine untouched — a caller-driven cancellation must never be silently absorbed by a retry
that respawns the sandbox and restarts the whole trajectory from scratch.

## Timeouts — what bounds what

Six different things can end a stuck run, on three different clocks, and no single one of them
bounds a run's wall time. This is the whole map; reach for it before adding a seventh.

| Knob | Env | Default | Bounds | On expiry |
|---|---|---|---|---|
| `RLMConfig.request_timeout_s` | `RLM_REQUEST_TIMEOUT` | unset → litellm's own 600s | **one HTTP request attempt** to a litellm-backed LM | `LMTimeoutError`, which dspy classifies as RETRYABLE — so it is retried, not fatal |
| `ClaudeAgentLM(timeout_s=…)` | — (constructor only) | 600s | **one whole call**, INCLUDING time queued behind the SDK's concurrency semaphore | `TimeoutError` |
| `RLMConfig.sandbox_turn_timeout_s` | `RLM_SANDBOX_TURN_TIMEOUT` | unset (disabled) | one whole `execute()` (`pyodide`/`deno`) — host-side tool and sub-LM dispatch time INCLUDED, which is why it is off by default | dspy's **recoverable** interpreter error — the model gets another turn |
| `ContainerConfig.timeout_s` | `RLM_CONTAINER_TIMEOUT` | 120s | one `execute()`'s sandbox compute (`container`; host tool time excluded) | recoverable, same as above |
| `RLMTask(cancel_event=…)` | — | unset | anything the caller decides | `SandboxCancelled` — **never** retried, never wrapped |
| `RLMConfig.max_retries` | `RLM_MAX_RETRIES` | 1 (no retry) | the whole trajectory, in attempts not seconds | `RLMTaskError` |

Three things about this table are easy to get wrong.

**`request_timeout_s` does not bound a run to its own value.** An attempt is not a request: dspy's
`LM` passes `num_retries=3` and litellm hands the OpenAI SDK `max_retries=2` of its own, so a dead
endpoint is retried and the run-level wait is a MULTIPLE of this number plus backoff. Size a
caller-side budget on the multiple. Leaving it unset is not "no cap" either — litellm then applies
its own `COMPLETION_HTTP_FALLBACK_SECONDS` of 600s, so this field REPLACES that number rather than
introducing a bound where none existed. A consumer whose turns legitimately run longer must set it
UP, not leave it alone.

**The two 600s defaults are not the same quantity.** `request_timeout_s` bounds one HTTP request
with retries around it; `ClaudeAgentLM`'s bounds one whole call including queueing, with nothing
around it. Which one you get depends on the model string, and `request_timeout_s` deliberately does
NOT drive the subscription route — under `llm_query_batched`'s thread fan-out, a value that is
generous per request would make queued sub-LM calls time out from waiting alone. `configure()` warns
when you set the knob and a role was auto-routed, and
`configure(main_lm=ClaudeAgentLM(model, timeout_s=…))` is how you choose that number.

**A sandbox timeout and a cancel are opposite outcomes, on purpose.** The turn timeout is a
safety net: it raises the error dspy CATCHES, so the model sees a dead turn and writes different
code. A `cancel_event` is a decision: it raises `SandboxCancelled`, which stands outside dspy's
exception hierarchy entirely and is on `run_with_retry`'s `non_retryable` list, so it ends the run.
Never let a cancel degrade into the first kind.

**Worked example.** `RLM_REQUEST_TIMEOUT=120`, `RLM_MAX_ITERATIONS=30`, `RLM_MAX_RETRIES=1`,
against a litellm-backed model on an endpoint that has stopped answering:

```
one attempt      120s              the value you set
  x ~3-4         dspy's LM(num_retries=3) re-sends the request
  x ~3           litellm hands the OpenAI SDK max_retries=2 of its own
one turn       ~20min + backoff    for ONE wedged model call
  x 30 turns   ~10 hours           the iteration cap is the only thing that ends the run
```

The two retry layers are counts of RETRIES, not attempts, so read those multipliers as lower
bounds. The point is the shape, not the digits: a "2-minute request timeout" is a wall-clock
worst case measured in hours, and lowering `max_iterations` moves that number far more than
lowering the timeout does. `max_retries` multiplies it again — which
is why it defaults to 1. If you need a real wall-clock bound on a run, put it in your own
driver (`asyncio.wait_for` around `arun`, or a `cancel_event`); no knob here is one.

> These semantics assume non-streaming model requests, which is what the kit issues today. A
> streamed request changes what a per-request HTTP timeout means — an inter-chunk read timeout does
> not bound a whole generation — so anything that turns streaming on owes this page a revision and
> a deadline of its own.

## `run_in_subprocess` — a safe, isolated-subprocess primitive (`isolation.py`)

A small PRIMITIVE only: safely run one picklable callable in an isolated OS process, get its
result or a clear error back, bounded by a timeout. Task-scheduling — how a web server actually
queues many of these (Celery, RQ, a plain thread/process pool) — is explicitly the consumer's own
concern, not shipped here, matching this kit's standing base/wrap posture (`make_command_tool`
ships no executor, `make_git_clone_tool` ships no cloner) applied to "run a whole task."

```python
import functools
from rlm_harness import run_in_subprocess

def run_my_task(**inputs):          # a plain, MODULE-LEVEL function -- see "must be picklable" below
    return MyTask().run(**inputs)

result = run_in_subprocess(functools.partial(run_my_task, q="…"), timeout_s=30.0)
```

**A genuinely different gap than three things in this kit that already sound similar** — worth
naming so this doesn't look redundant with them: `interpreter="container"` (above) isolates the
RLM's own REPL SANDBOX in a Docker container, but the ROOT process still runs the RLM's own
orchestration (LM calls, retries, tool dispatch) directly — a hang/crash there isn't covered.
`rlm_harness.tools.run_isolated` bridges an async coroutine into a sync call site on a dedicated
THREAD — same process, no OS-level isolation, solves an event-loop-nesting problem, not a
fault-isolation one. `cancel_event` (above) stops an IN-FLIGHT run the calling code already owns
and is watching — it doesn't hand a whole task off to a separate process in the first place. The
actual gap: a web-facing consumer whose request handler wants to run ONE task without that run's
own crash, hang, or resource usage taking down the request-handling process itself needs to run
it in a SEPARATE OS PROCESS.

- **Lives at the top level, not `rlm_harness.tools`** — that package's own module docstring
  scopes it as "tools RLM tasks can expose to the model inside the REPL"; nothing here is ever
  placed in a `tools=[...]` list or invoked by the model.
- **`factory` MUST be picklable** — a real, easy-to-get-wrong gotcha. A local closure or
  `lambda` is NOT picklable across the `"spawn"` boundary this uses; `functools.partial(
  module_level_function, **kwargs)` IS — but only if every value bound into `args`/`kwargs` is
  ALSO picklable, not merely the function reference: a live socket, open file handle, DB
  connection, or lock bound as one of the `partial`'s own arguments hits the same class of
  pickling failure the "avoid closures" advice was supposed to prevent.
- **Uses `multiprocessing.get_context("spawn")`, never `"fork"`** — the calling process is very
  plausibly a web server already running an event loop / thread pool / open file descriptors / a
  live LM client; forking that risks a corrupted child (inherited locks held by a thread that
  doesn't exist in the child, half-open sockets). `spawn` is itself still fork()+exec() on POSIX,
  but exec runs BEFORE any user code executes in the freshly-forked child, which is what actually
  avoids the corruption class a bare `fork` risks. `spawn` also requires `factory`'s entire import
  chain to be safely re-importable in the fresh interpreter — if `factory` is defined in a script
  run as `__main__`, guard it with `if __name__ == "__main__":`.
- **A real bug found and fixed during design review**: `multiprocessing.Queue.put()` does not
  pickle synchronously — a background feeder thread does, and a pickling failure there is logged
  and silently DROPPED, never raised back to `put()`'s caller, which would leave the parent
  hanging on `get()` forever. Fixed by having the child explicitly test-pickle its payload
  synchronously, in its own code, BEFORE ever calling `put()` — falling back to a plain-string
  `RuntimeError` only if that test-pickle itself fails. The parent's own `queue.get()` also
  carries its own bounded timeout, independent of the process-level timeout below, as a backstop
  against an out-of-band kill (e.g. the host OOM-killing the child) that bypasses this
  primitive's own signal-based escalation entirely.
- **`timeout_s`** (default `None` = no limit): on expiry, `process.terminate()` (SIGTERM), then a
  `grace_period_s` (default `5.0`) wait, then `process.kill()` (SIGKILL) if still alive, followed
  by a final reap so no zombie is left — raises `TimeoutError`. SIGTERM does NOT reliably let the
  child's `finally`/`atexit` code run — that's only true if the child itself installs a
  `signal.signal(SIGTERM, ...)` handler; with none (the default), the OS's default disposition
  terminates it immediately.
- **`max_memory_mb`/`cpu_time_limit_s`** (default `None` = no cap): opt-in, POSIX-only,
  best-effort resource caps applied inside the child via `resource.setrlimit` before `factory()`
  runs — silent no-ops on a platform without `resource` (Windows). `cpu_time_limit_s`
  (`RLIMIT_CPU`) is confirmed to enforce correctly on every POSIX platform tested, including
  macOS. `max_memory_mb` (`RLIMIT_AS`) bounds VIRTUAL address space, not resident/physical
  memory — and on macOS specifically, the kernel refuses to lower `RLIMIT_AS` from unlimited AT
  ALL (empirically confirmed: every attempted value, from 50 MB to 1000 MB, failed identically
  with `ValueError: current limit exceeds maximum limit`), so this parameter is effectively
  Linux-only in practice today. Either way it fails LOUDLY — relayed as a clear exception through
  the same test-pickle-then-relay path as any other error — never silently leaving the cap
  unenforced. **A second edge case, confirmed on real Linux CI, not just reasoned about**: an
  aggressively low `max_memory_mb` can starve the relay mechanism itself — the child correctly
  hits `MemoryError`, but by then is so memory-constrained that `multiprocessing.Queue.put()`'s
  own internal feeder thread fails to even start (`RuntimeError: can't start new thread`),
  crashing the child before anything can be relayed. No fallback is possible for this — a
  resource-exhausted process can't reliably report its own exhaustion through a mechanism
  (spawning a thread) that itself needs spare resources — so it degrades to the exact same
  safety net an external kill would (the parent's own bounded `queue.get()` times out and raises
  a generic "child exited without delivering a result" error). This is the correct, accepted
  outcome for this scenario, not a bug — the cap was still genuinely enforced.
- **No process pooling/reuse** — a fresh `multiprocessing.Process` per call, matching
  `run_isolated`'s own "one thread per call, by design — simplest correct answer" precedent,
  transplanted to the process level. A consumer wanting a persistent worker pool builds one using
  this primitive as the per-task unit.

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

### `verify_quote` — the deterministic half of step 2's diff

Step 2 above ("diff the artifact against it, itemized") is entirely model-judged — nothing backs
the claim that a specific quote/citation actually appears in the held source. `verify_quote(source,
quote)` (`rlm_harness.tools`) is the deterministic primitive for exactly that one checkable piece:
it returns a parseable `"MATCH: ..."` (with a line number and a context snippet) or `"MISMATCH:
..."` (with a bounded "closest line" hint when `quote` is single-line) — never a self-graded
guess. A single plain function, no factory, no `name=`, no trace call (it binds to nothing at
construction time and touches no filesystem/network, matching `make_schema_validator`'s own
precedent). Matching is whitespace-flexible by default (`normalize_whitespace=True`) and needs no
`regex` package — `quote` is literal text, every character either escaped or collapsed to a flat
`\s+`/`\s*`, so the built pattern can never exhibit catastrophic backtracking the way an
LM-controlled `grep_files` pattern could. Normalization is junction-aware: whitespace between two
word characters must be present in the source (so `foo bar` never verifies against `foobar`),
whitespace anywhere else — beside a bracket, a quote mark, punctuation — is optional (so a quote
that reflowed a line break still verifies). Call it from within the task's own instructions before finalizing
(closing this recipe's step-3 "regenerate on the gaps" loop with a real check instead of a
re-read), or reuse the same function host-side, outside the REPL, to re-derive whether a SUBMITted
citation was actually grounded — the same "derive facts from bytes, never trust the self-report"
posture the next section establishes for validity.

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

## Reading a trace — the ordering rules

Three facts a reader needs and cannot infer from the file. A downstream consumer got this wrong,
concluded "file order is unreliable, sort by `ts`", and reordered its turns.

- **`main_step` events are written in one block AFTER the run.** `record_main_trajectory` runs once
  `aforward()` has returned, because dspy.RLM only exposes its trajectory post-hoc. So a `tool_call`
  recorded mid-run appears EARLIER in the file than the `main_step` of the turn that made it, while
  being chronologically later. Measured at 70 of 76 real traces. By design, not a defect.
- **`payload["turn"]` is authoritative for ordering**, and file order among `main_step` events
  already matches it (72 of 72 traces). **Never sort `main_step` events by `ts`** — a `ts` is
  backfilled from a live stamp and, for a turn whose stamp could not be matched, falls back to the
  flush time, so sorting by it moves turns.
- **`ts` is for placing a turn against the tool calls around it**, and nothing else. A per-turn
  duration computed as `t[i] - t[i-1]` is an estimate that also contains the model's generation
  time; `exec_duration_s` (1.6.0) is the measured `execute()` half, and `duration_s` on a
  `tool_call` is the measured tool half. **`exec_duration_s` is `execute()` wall-clock, not sandbox
  CPU:** dspy dispatches tool calls and `llm_query` synchronously from inside `execute()`, so a cell
  that calls one blocks — and that whole round trip is inside the number. Read a large value as
  "the turn blocked", and try to cross-check it against the `tool_call` / `sub_call` events in the
  same run — but that check is often unavailable, and its absence is not the field lying: `llm_query`
  emits a `sub_call` only when the caller wrapped its `sub_lm` in `intercept_sub_lm`, and a
  `tool_call`'s `duration_s` is optional and unset for the local read/grep/edit tools.
  For scale, measured on a real workload: execution is ~1% of a turn's wall-clock; ~99% is the model
  generating. Prefer a measured field over a gap wherever one exists — and treat a
  NEGATIVE gap as unknown rather than as data (traces written before 1.6.1 can contain them; see
  that entry in `CHANGELOG.md`).

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
batch/dataset-level view. Reads only already-frozen `trace/v1` fields — `event["type"]`,
`payload["tool"]`, the optional `payload["duration_s"]`, and (through `payload_cause`, never
directly) `circuit_broken` / `endpoint_error` / `error` / `ok`. No new event type, nothing the
trace contract needs to change for.

### `compute_tool_waste` — which calls produced nothing usable, and what they cost

The sibling question, and the one nobody could answer before 1.6.0: **how much of a run's time went
into tool calls that produced nothing?**

```python
from rlm_harness import compute_tool_waste, group_by_run, load_events

runs = group_by_run(load_events("traces/run.jsonl"))
for name, w in compute_tool_waste(runs["r1"]).items():   # ...or compute_tool_waste_by_run(events)
    print(name, w.calls, w.invalid, w.endpoint_errors, w.circuit_broken)
    print("  declined:", w.invalid_rate, " time wasted:", w.wasted_seconds, w.wasted_share)
```

Two things it refuses to do, both deliberate:

- **It never reads `ok` directly.** Outcomes come from `payload_cause`, because `ok` is frequently
  ABSENT on an endpoint-failure payload — `payload.get("ok")` then returns `None`, which is falsy,
  so a naive counter silently absorbs infrastructure failures as content declines. That mistake has
  shipped four times. `invalid_rate` is likewise denominated over the calls that actually reached a
  validator, not over every call: a circuit break ran no validator and an endpoint failure produced
  no output to judge.
- **It never infers a duration from the gap between events.** `*_seconds` is `None` — meaning "not
  recorded" — for any tool that did not pass `duration_s`, including every trace written before
  1.6.0. `0.0` would read as "measured and found to be free", and inferring from event gaps charges
  a whole turn's model generation to that turn's first tool call. Every wall-clock attribution made
  against this kit's own corpus before 1.6.0 had exactly that error in it.

**Which shipped tools carry a duration.** The ones whose cost is a WAIT on something outside this
process: `fetch_url`, `web_search`, `run_command`, `git_clone`, the `model:<id>` tool from
`model_as_tool`, and every MCP tool. A local read/edit is sub-millisecond and its refusal paths
never touch anything, so timing them would add noise to the attribution rather than signal — those
record no `duration_s` and the metric says "not recorded" rather than "free".

`grep_files` is the honest exception to that reasoning and is documented rather than fixed: it
ships `per_match_timeout_s=1.0` per line and `max_total_time_s=30.0` per call, so it is *not*
sub-millisecond in principle, and `compute_tool_waste` is blind to exactly its worst case. It stays
untimed because it is sub-millisecond in PRACTICE on the corpora measured so far — n=146, median
0.029s, max 0.746s, not one call over a second. Two caveats stand against that number: it does not
identify which backend served those calls, and it contains no pathological regex over a large tree,
which is the only shape that would approach the budget. New evidence of either reopens this.

The tool that dominates a real run is usually the consumer's own model-backed one, and for the two
BASE factories the kit cannot record it: `make_model_tool` and `make_harness_tool` are deliberately
side-effect-free (no tracing, no messages), so the consumer's wrapper owns the `record_tool_call`.
(`model_as_tool` is the exception — a model-backed tool the kit *does* record — and it carries its
own duration.) **Pass
`duration_s=time.perf_counter() - t0` from it** (any monotonic clock) — that one line is what
makes the 80% of a run's
wall-clock visible.

### Denominators

Both `RunUtilization` rates are denominated over `main_steps` (root-LM turns) — "how many tool calls / sub-LM
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
`RLM_MAX_ITERATIONS`, `RLM_MAX_LLM_CALLS`, `RLM_MAX_RETRIES`, `RLM_SANDBOX_TURN_TIMEOUT`,
`RLM_REQUEST_TIMEOUT`, `RLM_OBSERVE`.

(`RLM_REQUEST_TIMEOUT` is this kit's own name. litellm separately reads a bare `REQUEST_TIMEOUT`
for its global default — setting that one moves litellm, not this.)

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
Set `RLMConfig(max_tokens=None)` to defer to the server.

**The default is a floor, not a ceiling, and running into it looks like a different bug.** 8192 has
to hold the chain-of-thought AND the structured answer of the same turn. When a long turn overruns
it, the reply is cut off *mid-JSON* — so the adapter cannot parse what came back and the run dies as

```
RLMTaskError: Failed to produce a valid '<field>' after N attempts
  — caused by AdapterParseError: ... failed to parse the LM response
```

That is a **truncation**, not a model that cannot follow the schema, and it is easy to misdiagnose
as one: the text in the error often looks like well-formed output right up to where it stops.
Tell the two apart by reading the END of the quoted `LM Response` — a truncated one simply stops,
mid-string or mid-object, with no closing brace.

Two things make it more likely, and they compound: a **reasoning** model (the thinking is spent
before the answer starts) and a task whose one turn is genuinely long (assembling a large
structured result, or a REPL turn that both reasons and writes a big code block). More than one
consumer has hit this and settled on **16384** for the planner, which is a reasonable first move
when the symptom above appears.

It is not free, though, so raise it deliberately rather than reflexively: an OpenAI-compatible
server commonly validates `prompt_tokens + max_tokens` against the context window, so a bigger cap
removes usable PROMPT budget — and an RLM planner's prompt grows every turn, which is exactly the
workload where that bites. You are trading one failure (a truncated answer) against another (a
context-window refusal), not buying headroom for nothing.

There is also a case where the 8192 default fails immediately rather than mid-run: OpenAI's own
reasoning models reject it outright, and `dspy.LM("openai/o3", max_tokens=8192)` raises
`LMConfigurationError: ... require passing temperature=1.0 or None and max_tokens >= 16000 or
None` before a single request is sent. That is the loudest instance of this whole class, and it is
why 16000+ is the right floor for those models specifically.

A deterministic confirmation, when you want one rather than eyeballing the quoted response: dspy
logs `LM response was truncated due to exceeding max_tokens=...` at WARNING whenever the provider
reports `finish_reason == "length"`.

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
