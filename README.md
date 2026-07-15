# rlm-kit

A clean, reusable harness for building **any task** on top of
[DSPy](https://dspy.ai)'s Recursive Language Model module (`dspy.RLM`).

RLMs ([Zhang & Khattab, MIT, arXiv:2512.24601](https://arxiv.org/abs/2512.24601))
let a model explore unbounded context by treating it as a variable in a sandboxed
Python REPL and recursively calling sub-LLMs over it. DSPy's `dspy.RLM` is the
first-party implementation (Khattab co-authored both DSPy and the RLM paper) — it
works with existing Signatures and is optimizer-compatible (GEPA/MIPRO). This kit
distills the boilerplate around it into one small, opinionated layer.

**rlm-kit is domain-agnostic** — anything `dspy.RLM` can do fits: multi-hop "deep
research", an RSS-digest agent that posts to a webhook, structured extraction,
detection authoring, you name it. Security happens to be the author's own first
use of it, but it isn't the kit's scope.

## Why this exists

Using `dspy.RLM` directly leaves you re-writing the same plumbing for every task:
model/sub-model config, a retry+validation loop, a sandbox choice, observability.
`rlm-kit` makes a task a *declaration*:

```python
from rlm_kit import RLMConfig, RLMTask, configure
from rlm_kit.tools import make_schema_validator
from pydantic import BaseModel

class Article(BaseModel):
    title: str
    summary: str

class Summarize(RLMTask):
    signature = "document: str -> article: Article"
    output_field = "article"
    output_model = Article
    instructions = "Read the document and produce a title and a one-paragraph summary."
    tools = [make_schema_validator(Article)]

configure(RLMConfig.from_env())
article = Summarize().run(document=long_text)   # validated Article
```

The retry loop, pydantic validation, sandbox selection, and budget caps are all
inherited.

## Installation

```bash
# from git (pre-release — not on PyPI yet):
pip install "git+https://github.com/qazbnm456/rlm-kit"
# or with uv:
uv add "git+https://github.com/qazbnm456/rlm-kit"
```

Once the first release is published, `pip install rlm-kit` will work too. `rlm-kit` needs Python ≥ 3.11
and pulls in `dspy` + `pydantic`; observability extras are opt-in (`pip install "rlm-kit[observe]"`). A
*live* `dspy.RLM` run additionally needs model credentials (see [Configuration](#configuration)) and a
Deno sandbox (`brew install deno`) — the logic and tests run without either.

## Layout

| Module | Responsibility |
|---|---|
| `config.py` | Single source of truth; `RLMConfig.from_env()`. No dspy import. |
| `runtime.py` | `configure()` — wires dspy + optional observability once. |
| `task.py` | `RLMTask` base class. |
| `_retry.py` | Validation + retry engine (dspy-free, unit-tested). |
| `sandbox.py` | Interpreter selection + the insecure-sandbox guard. |
| `tools/` | `make_schema_validator` (pydantic) + `make_json_schema_validator` (validate a parsed object against a vendored JSON Schema — the base for the "validate against an official, version-pinned upstream schema" pattern; needs `rlm-kit[jsonschema]`), SSRF-guarded `make_fetch_tool`, provider-agnostic `make_web_search_tool`, `make_command_tool` — a traced `run_command` over a consumer-supplied *isolated* runner (the kit ships no executor), and `make_model_tool` — the generic "model-as-tool + transient-retry + validate" core (a project wraps it with its own endpoint/validator/messages). |
| `optimize.py` | GEPA harness — metric templates now, compile in Phase 2. |
| `sub_lm.py` | `intercept_sub_lm` — wrap the RLM's sub-LM to trace every escalation as a `sub_call` (+ optional validate/post-process); `model_as_tool` for LM-decided multi-model routing. |
| `skills.py` | `load_skills_as_tools` — expose a Skills directory to the RLM as tools. |
| `trace.py` | `TraceRecorder` — unified append-only JSONL trajectory (main steps + sub-LM + tool calls). |
| `replay.py` | Reconstruct/replay a recorded run using recorded tool outputs. |
| `dataset.py` | `export_sft_turns` / `export_rl` / `export_actions` — turn traces into training datasets (`export_sft_turns` = per-root-turn SFT, the RLM recipe of arXiv 2512.24601). |
| `examples/mini_run.py` | Minimal end-to-end live run — config + a tiny `RLMTask` through a real `dspy.RLM`, with the trajectory recorded and summarised. |
| `examples/claude_agent_lm.py` | Run rlm-kit on a Claude Pro/Max subscription: a `dspy.BaseLM` adapter over the official Claude Agent SDK, injected via `configure(main_lm=…, sub_lm=…)` — personal use, pure completions (no tools), kit unchanged. |

## RLM as Harness Engineering (sub-LM hook + tracing)

`dspy.RLM` exposes no hook to intercept a sub-LLM response, and (as of 3.2.1) no
multi-sub-model or depth>1 recursion. The clean lever is to **wrap a `dspy.LM`**:

```python
from rlm_kit import intercept_sub_lm, model_as_tool, get_sub_lm, TraceRecorder, RLMConfig, configure

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
an `RLMTask` as ready-to-use tools. rlm-kit is a **client only** — it never runs a server and bundles
none; you point it at someone else's (a local stdio command, or a remote streamable-HTTP URL):

```python
from rlm_kit import mcp_tools

with mcp_tools({"url": "https://mcp.example.com/mcp"}) as tools:        # or {"command": "npx", "args": [...]}
    finding = MyTask(tools=tools).run(...)                              # the server's tools are now callable
```

Needs the extra: `pip install "rlm-kit[mcp]"`.

- **The connection is live for the `with` block** and torn down on exit (a stdio subprocess is
  terminated). Each tool call is recorded as a `tool_call` in the trace, like any other tool.
- **Sync, despite an async SDK.** The MCP Python SDK is async, but dspy.RLM invokes tools
  synchronously, so rlm-kit runs the session in a background thread and bridges each call across.
  (dspy's own `Tool.from_mcp_tool` makes an *async* tool for `dspy.ReAct` — it does not work on the
  RLM sandbox path, which is why `mcp_tools` exists.)
- **Security: MCP tools run HOST-SIDE**, outside the sandbox — a stdio server is a subprocess this
  process spawns. Treat an MCP server as a **trusted dependency**, and its output as a
  **prompt-injection surface** (untrusted LM context), exactly like fetched web content.

## Running local commands (an isolated runner)

`make_command_tool(runner)` gives an `RLMTask` a `run_command` tool — the reusable half of
letting the model run a local command (a build, a test, a git op) the way a coding agent does.
It enforces the sync contract, turns a failure into text the RLM reacts to, and records one
`tool_call` in the canonical shape. The kit ships **no** executor and picks **no** isolation: you
supply the `runner`.

```python
from rlm_kit.tools import make_command_tool

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
shape-only pre-flight, never a security claim.

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
- **Needs the `docker` CLI** (checked at start; `import rlm_kit` stays docker-free). The `WORKDIR`
  mount resolves on the *daemon's* filesystem, so it won't work with a remote `DOCKER_HOST`.

## Grounded completeness — the sufficiency-critic recipe

A convention, not an API. When the RLM generates an artifact that must MATCH a retrieved
ground-truth (a spec, a contract, a source document), "am I done?" is the dangerous judgment:
a model asked to self-assess from memory will call a half-right artifact complete. There is
often no deterministic check for CONTENT correctness — a validator catches *structure/format*,
but not "this request is missing a required header" or "this answer skipped a clause".

The fix (the agentic-RAG *sufficient-context* pattern) is to GROUND the completeness judgment in
the retrieved source instead of the model's recall:

1. **Hold the ground-truth in REPL state.** Fetch the source once (a `fetch_url` tool, a skill)
   and keep it as a REPL variable — rlm-kit's interpreter persists variables across turns, so the
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

`rlm-kit` is the ROLLOUT floor; a consumer is a thin declaration on top of it. `examples/harness_run.py`
is a minimal worked example — a task that wires the sub-LM hook, skills, tracing, and
RL export together. Five steps:

1. **Declare the task.** Subclass `RLMTask`: a `signature`, `output_field`, an `output_model`
   (judgement-only — see above), `instructions` (orchestration + a few hard safety rules), and
   `tools`. The retry/validation loop, sandbox, budget caps, and the trace are inherited. Put
   authoring KNOWLEDGE in a Skills directory (`load_skills_as_tools`), not the prompt — the prompt
   is for orchestration; skills are progressive-disclosure reference the LM pulls on demand.
2. **Add tools the base/wrap way.** Need a new capability (a model-as-tool producer, a fetcher, a
   searcher)? rlm-kit owns the GENERIC base + the syntactic guard + the async-safe factory
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
   trainer fills — rlm-kit never computes a reward.

**The promotion rule** keeps the boundary clean. When the consumer forces a workaround, ask "is this
GENERIC?" A reusable mechanic (the model-tool + retry + validate core, a new sandbox seam, a trace
hook) is PROMOTED into rlm-kit via the base/wrap split — the generic half here, the specific half in
the consumer. A consumer-specific VALUE (a model name, a schema, a validator, a path) stays in the
consumer. Never special-case the consumer inside the kit; never fork the harness or re-implement
tracing inside the consumer. If you need an internal seam the kit doesn't expose, ADD a public hook
here (that is how `recorder_scope` / `bind_recorder_to_sub_lm` / `get_sub_lm` were born) rather than
reaching into a `_private` name. The trace schema, the `EVENT_*` types, and the exporter record shapes
are a FROZEN v1 wire format — `tests/test_contract.py` pins them; adding an optional field is fine,
removing or re-typing one is a `v2` break. The `EVENT_*` type constants are exported from `rlm_kit`,
so a trace reader matches on `rlm_kit.EVENT_RESULT` instead of hardcoding the wire string `"result"`.

**The stage boundary** keeps the data honest. rlm-kit + your consumer are the ROLLOUT stage: they
produce trajectories (the trace) and turn them into datasets, emitting raw LABELS / METRICS, never a
reward scalar. SCORING (reward composition, credit assignment) and TRAINING (GRPO / SFT) are a
SEPARATE downstream project that installs the trainer. A prompt/policy rule that makes the rollouts
BETTER is in scope; a reward or penalty is not. Keep the trace clean training data and let the
trainer score it.

## Built with rlm-kit

Real projects using rlm-kit as their RLM scaffold:

- **[cve-reverser](https://github.com/qazbnm456/cve-reverser)**: reverses publicly disclosed CVEs from
  their patches into local-lab PoCs and Nuclei detection templates. A traced, trainable RLM harness.
- **[diff-sentry](https://github.com/qazbnm456/diff-sentry)**: classifies GitHub changes (PRs, issues,
  pushes) for malicious intent — the diff is read as untrusted data in the sandboxed REPL, emitting
  evidence-backed benign / suspicious / malicious verdicts into a SIEM.

Built something on rlm-kit? Open a PR to add it here.

## Security note — the sandbox is the boundary

RLM executes model-written code. When that code processes untrusted scraped
content, the interpreter choice is your attack surface. The default
(`pyodide`/`deno`) is the sandboxed DSPy interpreter. The `local` interpreter runs
code on the host and is **refused** unless you set
`allow_insecure_sandbox=True` / `RLM_ALLOW_INSECURE_SANDBOX=1`. Don't.

The default sandbox is built by the kit (not handed straight to dspy) so it can
pre-bind the JSON literals `true`/`false`/`null` to `True`/`False`/`None` in the
REPL namespace — a JSON-trained instruct model otherwise writes `SUBMIT({"ok":
true})` and the REPL raises `NameError: name 'true' is not defined`, which the model
tends to retry verbatim. Isolation is unchanged; `RLMTask` owns the teardown.

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

## Develop

```bash
uv sync --group dev
uv run pytest          # logic tests (no live LLM needed)
```

Tests cover config parsing, the retry/validation engine, the sandbox guard, the
tools, the sub-LM-hook/trace/replay/dataset layer, and a real-`dspy.RLM`
construction check (dspy-bearing tests use `DummyLM` or skip if dspy is absent).
A *live* run additionally needs real credentials and a Deno sandbox
(`brew install deno`); `examples/mini_run.py` shows it. See `CLAUDE.md` for
invariants when modifying the kit.

### Testing the forward path offline (`rlm_kit.testing`)

Construction tests (`task._build_rlm()`) catch signature/kwarg drift but never run the loop — where
wiring bugs actually hide (a prompt naming a tool `foo` while it registered as `foo_tool` is a
`NameError` no construction test sees). `rlm_kit.testing` drives the **real `dspy.RLM.aforward` loop
offline** — no model, no Deno, no network:

```python
from rlm_kit import RLMConfig, RLMTask, configure
from rlm_kit.testing import ScriptedInterpreter, call, scripted_lm, submit

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
is a test/advanced seam; the default string path keeps the guard. `rlm_kit.testing` imports dspy lazily,
so it doesn't affect `import rlm_kit`.

## Status

**v0.2.0** (in development — not yet tagged or published to PyPI; the version is the
target) — scaffold + harness-engineering layer (sub-LM hook, skills-as-tools,
trajectory recording, replay, dataset export). Hardened by dogfooding against a
real downstream consumer; the changes that surfaced are in [`CHANGELOG.md`](https://github.com/qazbnm456/rlm-kit/blob/main/CHANGELOG.md).

Next: enable `optimize.compile_task` against a labelled trainset to actually
GEPA-compile tasks (currently a documented stub).

## License

MIT © Boik Su ([@boik_su](https://x.com/boik_su)). See [`LICENSE`](https://github.com/qazbnm456/rlm-kit/blob/main/LICENSE).
