# rlm-harness: the guide

The deep documentation for `rlm-harness`: what each module owns, the harness-engineering
layer (sub-LM hook + trajectory tracing), the tool surfaces, the rollout conventions,
the consumer contract, and every configuration knob. For what the kit *is*: the
pitch, the quickstart, and installation: start at the
[top-level README](../README.md).

## Layout

| Module | Responsibility |
|---|---|
| `config.py` | Single source of truth; `RLMConfig.from_env()`. No dspy import. |
| `runtime.py` | `configure()`: wires dspy + optional observability once, including auto-routing a `claude-agent-sdk/`-prefixed model string onto `ClaudeAgentLM`. |
| `task.py` | `RLMTask` base class. |
| `_retry.py` | Validation + retry engine (dspy-free, unit-tested). Mostly private, but `short_error` is public: head-and-tail elision for a caught exception, so a giant `AdapterParseError` (which embeds the whole raw completion) becomes one readable log line instead of thousands. |
| `sandbox.py` | Interpreter selection + the insecure-sandbox guard. |
| `atomic.py` | `atomic_write_text` / `atomic_write_stream`: a same-directory temp file + `fsync` + `os.replace`, so a concurrent reader never sees a partial write; `atomic_write_stream` takes an iterable of `bytes` chunks instead of one in-memory blob, aborting once a running total exceeds an optional `max_bytes`. dspy-free. |
| `memory.py` | `MemoryStore` / `MemoryItem` / `MemoryLabels` / `Topic` / `PurgeRecord` / `PROVENANCES` / `MemoryUnavailable`: a provenance-typed memory in one SQLite file that outlives a run, with labels, CJK-aware full-text search and a host-only `purge` (see "Memory" below). dspy-free, stdlib only. |
| `memory_organise.py` | The model-free half of organising a store: `section_map`, `clean_language`, `known_listing`, `suggest_topics` / `TopicSuggestion`, `topic_graph` / `TopicGraph`, `run_sources` / `RunSource` / `render_run` (a finished run as text a model can read). dspy-free. |
| `memory_vectors.py` | Local similarity for a store (the `memory-vectors` extra): `EmbeddingModel` / `DEFAULT_MODEL`, `download_model`, `OnnxEmbedder`, `MemoryVectors` (`sync`, `similar`, `similar_each`, `similar_pairs`, `forget`). dspy-free; numpy and the model runtime are imported on use. Imported explicitly, not re-exported. |
| `memory_tasks.py` | The model-backed half: `SummariseShort` (one `dspy.Predict` call), `OrganiseBatch` (one `dspy.Predict` call), `DistillLongItem`, `AlignEntities` and `ConsolidateRun` (`RLMTask`s), `run_step` / `arun_step` / `subprocess_step_runner`, `distill_pending`, `align_pending`, `organize_pending` and `consolidate_run`. Lazily exported, since it imports dspy. |
| `metrics.py` | `RunUtilization` / `compute_run_utilization` / `compute_utilization_by_run` / `ToolWaste` / `compute_tool_waste` / `compute_tool_waste_by_run` / `RUN_FACT_KEYS` / `compute_run_facts` / `compute_run_facts_by_run` (the generic half of a rubric's facts) / `MemoryUse` / `compute_memory_use` / `compute_memory_use_by_run` (what a run read from and wrote to memory): reward-free trace utilization metrics (how a run's activity split across root-LM turns, tool calls, sub-LM escalations), a pure derived read over already-recorded trace events. dspy-free. |
| `large_text.py` | `LargeText`: a long input staged as a file the run's sandbox is granted, addressed by `sandbox_path`, instead of an inline literal rebuilt every turn. Measured at 0.71x the payload in peak RSS against the variable path's 66x (see "A long input as a file" below). The only thing that opens a sandbox read grant, and host code is the only place one can be built. dspy-free, stdlib only. |
| `isolation.py` | `run_in_subprocess`: a safe, isolated-subprocess primitive for a web-facing consumer: run one picklable callable in a fresh OS process, get its result or a clear error back, bounded by a timeout (see below). dspy-free. |
| `tools/` | `make_schema_validator` (pydantic) + `make_json_schema_validator` (validate a parsed object against a vendored JSON Schema: the base for the "validate against an official, version-pinned upstream schema" pattern; needs `rlm-harness[jsonschema]`), SSRF-guarded `make_fetch_tool`, its filesystem-side analogue `make_read_file_tool` / `make_grep_files_tool` / `resolve_within_root` (needs `rlm-harness[grep]` for a wall-clock-bounded `grep_files` (see below) plus the write side `make_write_file_tool` / `make_edit_file_tool` in `tools/edit.py` (see below), `list_candidate_paths`) a safe, `.gitignore`-aware default for building `candidate_paths` in `tools/discover.py` (needs `rlm-harness[gitignore]`; see below), `make_git_clone_tool` (safe git clone with fallback auth over a consumer-supplied isolated `cloner` in `tools/git_clone.py` (see below), `make_extract_archive_tool`) safe `zip`/tar extraction in `tools/archive.py` (see below), `verify_quote` (a deterministic quote/citation grounding check in `tools/grounding.py` (see "Grounded completeness" below), provider-agnostic `make_web_search_tool`, `make_command_tool`) a traced `run_command` over a consumer-supplied *isolated* runner (the kit ships no executor) with an optional `refuse_broad_git_history` guard, `make_model_tool`: the generic "model-as-tool + transient-retry + validate" core (a project wraps it with its own endpoint/validator/messages), the memory tools `make_memory_search_tool` / `make_memory_get_tool` / `make_memory_note_tool` in `tools/memory.py` (see "Memory" below), and the harness-delegation pieces `make_harness_tool` / `harness_from_endpoint` / `pointer_to_invocation` / `run_isolated` (see "Delegate to another harness" below). |
| `optimize.py` | GEPA harness: `compile_task`, `audit_instruction`, `trainset_from_traces`. |
| `sub_lm.py` | Every sub-LM escalation is traced as a `sub_call` automatically (1.7.0). `intercept_sub_lm`: wrap the RLM's sub-LM to ADD a deterministic validate/post-process pipeline; `model_as_tool` for LM-decided multi-model routing. |
| `skills.py` | `load_skills_as_tools`: expose a Skills directory to the RLM as tools. |
| `trace.py` | `TraceRecorder`: unified append-only JSONL trajectory (main steps + sub-LM + tool calls). |
| `replay.py` | Reconstruct/replay a recorded run using recorded tool outputs. |
| `dataset.py` | `export_sft_turns` / `export_rl` / `export_actions` (turn traces into training datasets (`export_sft_turns` = per-root-turn SFT, the RLM recipe of arXiv 2512.24601); `run_label_bundle`) carry per-run LABEL surfaces beside the trajectory. |
| `mcp.py` | MCP **client** (optional `rlm-harness[mcp]`): `mcp_tools` exposes an external server's tools as SYNC `dspy.Tool`s; `McpCatalog`/`McpConnection` for a progressive multi-server surface. |
| `testing.py` | Offline forward-path harness: `ScriptedInterpreter`, `scripted_lm`, and the REPL-safety assertions `assert_repl_safe` / `assert_task_repl_safe`. |
| `serving.py` / `harness_serve.py` | Server side of the harness-delegation contract: turn an `RLMTask` into a process that speaks it. |
| `container_interpreter.py` | The opt-in `interpreter="container"` REPL: model code runs inside an isolated Docker container so it can spawn subprocesses. |
| `_toolname.py` | REPL-safety rules for a tool's NAME and SIGNATURE. Mostly private, but `is_valid_tool_name` / `sanitize_tool_name` / `unique_tool_names` / `signature_from_json_schema` are public: a consumer building its own tools (e.g. from `McpCatalog`'s raw names) needs the same derivation the kit uses. |
| `_dspy_compat.py` | Private. Every cross-version dspy difference (kwarg renames, the interpreter seam, the recoverable/terminal error split) resolved by introspection in ONE place. |
| `rubric.py` | Reward-free rubric primitives: the `Criterion`/`RubricCriteria`/`CriterionFact` types, `rubric_to_meta`/`rubric_from_meta`, `validate_rubric`, and a pure `criteria_facts(criteria, facts, lens)`. `category` is an OPAQUE caller-defined label, the kit imposes no taxonomy. See "Building a consumer". |
| `claude_agent_lm.py` | `ClaudeAgentLM`: run rlm-harness on a Claude Pro/Max subscription: a `dspy.LM` over the official Claude Agent SDK. `configure()` now auto-routes a `SUBSCRIPTION_PREFIX`-carrying (`"claude-agent-sdk/"`) `main_model`/`sub_model` string onto it, no explicit `main_lm=`/`sub_lm=` wiring needed unless you're overriding it. Opt-in `rlm-harness[subscription]`; pure completions (no tools), lazily exported so `import rlm_harness` stays dspy/SDK-free. |
| `examples/mini_run.py` | Minimal end-to-end live run: config + a tiny `RLMTask` through a real `dspy.RLM`, with the trajectory recorded and summarised. |
| `examples/memory_run.py` | One live run over the whole memory arc: a host seeding `observed` and `said` items, the three tools in a task, `compute_memory_use`, `consolidate_run`, and `purge`. |
| `examples/claude_agent_lm.py` | Runnable demo of `ClaudeAgentLM`: a tiny `RLMTask` through a real `dspy.RLM` on a subscription login. |

## RLM as Harness Engineering (sub-LM hook + tracing)

`dspy.RLM` exposes no hook to intercept a sub-LLM response, and no multi-sub-model seat. The clean
lever is to **wrap a `dspy.LM`**:

```python
from rlm_harness import intercept_sub_lm, model_as_tool, get_sub_lm, TraceRecorder, RLMConfig, configure

configure(RLMConfig.from_env())
base = get_sub_lm()          # the configured base sub-LM: single source of truth
# escalations are traced with or without this; intercept_sub_lm ADDS the deterministic pipeline
# (deterministic only: agentic actions stay LM-decided tools):
smart_sub = intercept_sub_lm(base, validators=[...], postprocessors=[str.strip])

with TraceRecorder("traces/run.jsonl", run_id="r1"):
    finding = await MyTask(sub_lm=smart_sub).arun(evidence=blob)
```

`intercept_sub_lm` adds a validate/post-process pipeline (the `sub_call` event itself is automatic since 1.7.0) and, if you pass them,
runs deterministic validate → post-process. `get_sub_lm()` hands back the base sub-LM
`configure` built: wrap THAT rather than reconstructing a `dspy.LM`, so it can't drift
from the configured model. External tools are exposed to the main
LM via `tools=` / `load_skills_as_tools` / `model_as_tool`, so the decision lands in
the trajectory. `TraceRecorder` records main steps (`Prediction.trajectory`), every
sub-LM call, and every tool call into one JSONL stream: replayable (`replay.py`) and
exportable as an RL/SFT dataset (`dataset.py`). Langfuse is an optional mirror; the
JSONL is the dataset's source of truth.

> **A refused escalation is still an escalation (1.13.0).** When the provider raises, for instance
> a sub model pulled from a proxy, the `sub_call` is recorded with `raw: null`, `processed: null`,
> the error, and `cause: "endpoint"`, and the exception then propagates as before. Until 1.13.0 the
> event was written only after the call returned, so a run that escalated repeatedly against a dead
> provider recorded ZERO escalations and read as one where the planner never called `llm_query`.
> Read `cause` rather than `error` to separate that from an output a validator rejected
> (`"invalid"`), since only the second is the model's doing. A SUCCESSFUL escalation's usage does
> reach `run_end.payload.usage` under the sub model's key, `llm_query_batched`'s thread fan-out
> included, so after this release a corpus with neither a `sub_call` nor a sub-model usage key is a
> planner that genuinely did not escalate.

> **Reading a `sub_call`:** every `sub_call` event is exactly one sub-LM escalation,
> reached through `dspy.RLM`'s built-in `llm_query` / `llm_query_batched` (the only
> callers of `sub_lm`). The payload carries `kind:"sub_lm"` + the wrapper `name`. It
> does **not** record which built-in triggered it: dspy calls `sub_lm` identically for
> both. The planner's actual `llm_query(...)` call lives in the `main_step` `code`, so
> *that* is where a Root-LM trainer learns "call llm_query"; the `sub_call` is the inner
> view. `llm_query_batched` fans calls across threads: `TraceRecorder.record` is
> lock-guarded so concurrent `sub_call`s can't corrupt the JSONL.

> Depth is **1** by design here (main LM + one intercepted sub-LM layer). The wrapper does not
> recurse, and the interception it adds is per-response, not per-trajectory.

Since dspy 3.4.1, model code CAN reach a nested `dspy.RLM` through the sandbox dspy facade, so
"unsupported upstream" is no longer the reason depth stays 1 here. The reason is now a measured
property: a nested RLM built in the sandbox records as ONE `sub_call` carrying only its final
answer, so the inner run's turns, code and REPL output are absent from the trace entirely. When a
child's trajectory is the point, use `make_harness_tool`, which gives the child its own rollout and
the parent a pointer to it; see "Delegate to another harness". `RLMConfig.sandbox_sub_agents`
decides whether the facade is available at all.

### Sub-LM tracing is automatic (1.7.0)

You do not have to wrap anything to see sub-LM escalations in a trace. `RLMTask` wraps a plain
`sub_lm` for `sub_call` recording at the same per-run seam that binds the recorder, so an
escalation is recorded whether or not you asked.

That changed because the opposite default was quietly lossy. Before 1.7.0 the event existed only
if you called `intercept_sub_lm` yourself; a plain `dspy.LM` was invoked by dspy directly and
recorded nothing. A corpus with zero `sub_call` events is indistinguishable from one where the
model never escalated, and reading the first as the second is a mistake that has already been
made, in this kit's own design notes, on real data. **An absent event is not a measurement.**

Call `intercept_sub_lm` when you want the deterministic pipeline:

```python
sub = intercept_sub_lm(get_sub_lm(), validators=[...], postprocessors=[str.strip], name="lifeline")
task = MyTask(sub_lm=sub)     # the kit detects it and does not wrap it again
```

**If you have your own recording wrapper, declare `records_sub_call = True` on it** and the kit
leaves it alone:

```python
class MyTracingSubLM:
    records_sub_call = True   # kit-defined protocol: "I already emit sub_call myself"
    ...
```

The probe is `is True`, not truthiness: a `unittest.mock` double manufactures a truthy attribute
for any name, and a truthiness test would silently skip wrapping it. Auto-wrapping also never
raises: a sub-LM it cannot wrap is used bare with a warning, because an observability convenience
must not be why a run fails to start.

### Sub-LM vs. tool: which model goes where

`intercept_sub_lm` and `model_as_tool` both "wrap a model," which makes them easy
to confuse. They sit on **opposite sides of the RLM boundary**, and the choice is not
cosmetic: it decides what your RL data records.

- **A sub-LM is part of the machine.** Wired in as `sub_lm=`, the *framework* decides
  when to call it. It is the seat the RLM's recursion plugs into (depth-1 here, but
  structurally the recursive seat). The framework assembles its prompt/context and it
  carries the run's identity (tracing, budget). The main LM never *chooses* to call it.
  → recorded as a **`sub_call`**.
- **A tool-LM is a leaf the main LM picks up.** Passed via `tools=` (e.g.
  `model_as_tool`), the *main LM* decides, in the REPL, to call it, with whatever
  string it wrote. It takes a string, returns a string, and stops: it can't recurse and
  never becomes an RLM root. The call is the LM's own decision, so it lands in the
  trajectory. → recorded as a **`tool_call`**.

> At the lowest level both are "call an LM with text, get text back." The difference is
> **role, not mechanics**: a sub-LM is a structural seat (framework-invoked,
> recursion-capable); a tool-LM is an optional leaf the main LM reaches for.

**"Deterministic transform" = plain code, no AI.** Both sides may check their model's
output with ordinary functions: same input, same output: `intercept_sub_lm` runs
`validators`/`postprocessors` on the sub-LM output; `make_model_tool` runs a `validate`
callable on a generated artifact (a consumer's generator tool runs a `postprocess()` validator to
verify the artifact's shape: that lives on the **tool** side, not the sub-LM). What neither may
do is ask *another model* to judge the output: that is an *agentic* decision, and agentic
decisions must stay with the main LM as a `tools=` call so the choice is visible in the
trajectory (and honest as RL data). That is exactly why `model_as_tool` is a thin
pass-through with no validation baked in. **deterministic checks are fine on either side;
a model-judgement must be an LM-decided tool call.**

**Pick by question:**

| You want… | Use | Wire as |
|---|---|---|
| a smarter/cheaper *default* sub-model with deterministic validate/post-process checks (tracing needs no wrapper) | `intercept_sub_lm(base, validators=…, postprocessors=…)` | `sub_lm=` |
| the main LM to *choose*, mid-task, to consult another named model | `model_as_tool(name, lm)` | `tools=` |
| both (a chosen model that also self-checks) | compose them: `model_as_tool("expert", intercept_sub_lm(expert_lm, …))` | `tools=` |

**Escalate to the sub-LM when a tool WALLS: don't circle it.** A convention, not an API. When a
`make_model_tool` (or any model-backed tool) repeatedly fails on the SAME gap: declines, returns
INVALID, can't fill the hole: that IS the signal the main LM cannot specify its way out: escalate to
the sub-LM for that gap instead of re-spinning the tool. Circling a walled tool burns the iteration
budget and can hit the cap with the task still unfinished; one focused sub-LM question often unblocks
it in a single turn (the sub-LM is the recovery seat: its whole purpose; the "expensive" framing is no
reason to keep re-spinning a stuck tool). Like grounded completeness this lives in the consumer's task
INSTRUCTIONS, kept in the trajectory as honest RL data. A consumer can nudge its planner this way after
a few repeated tool declines on one gap: turning a hard run that would otherwise circle a stuck tool
until the cap into one that escalates once and converges.

The nudge is a PROMPT, which a weaker root LM can ignore (one may hammer a stuck tool dozens of times). For a
deterministic backstop, `make_model_tool(max_consecutive_invalid=N)` is a run-scoped CIRCUIT BREAKER:
after N consecutive validator declines the next call SHORT-CIRCUITS (no model call,
`circuit_broken=True`), capping the wasted calls and forcing the consumer's redirect (escalate /
finalize). It resets on any ok; an endpoint error doesn't count. The factory only flags the break:
the consumer owns the message, and builds one tool per run so the breaker state resets naturally.

**`ok=False` has THREE causes; read `result.cause` before naming one.** The validator rejected the
output (`"invalid"`), the endpoint failed after retries (`"endpoint"`), or the breaker
short-circuited without calling the model at all (`"circuit_broken"`). In the last two the validator
NEVER RAN, so a consumer that reads only `ok` and writes "failed validation" is blaming the model
for infrastructure. That has shipped in more than one consumer, in both directions that matter: a
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
`ok=False` **too**, so a `sum(1 for r in calls if not r.ok)` metric silently includes every break:
filter on `cause == "invalid"` if you mean rejections. And an endpoint error deliberately does not
trip the breaker, so `circuit_broken` counts and endpoint counts never overlap with each other.

**Across the trace boundary the same distinction is `trace.payload_cause(payload)`**, reading the
same four words off a recorded `tool_call`, plus a fifth the live side cannot produce. It exists
because `ok` is often ABSENT on an endpoint payload (consumers record `error=` alone there), so
`payload.get("ok")` returns `None`: falsy, and every `not payload.get("ok")` counter downstream
absorbs infrastructure failures as content declines without a word of warning. In the worst
measured case that put 113 endpoint failures into a metric named `generator_declines`, fed it to a
scored rubric criterion about the planner's spec quality, and printed "113 partial/retry" in a
delivered report: for a run whose validator ran zero times, and whose planner had correctly
concluded the endpoint was unreachable.

A payload carrying no `ok` at all is the fifth word, `"unknown"`, since 1.24.0. It used to read as
a decline, so a writer that forgot the field produced a confident wrong answer rather than an
unknown: three of this kit's own writers had, and a consumer measured its own `tool_declines` at 60
against a real 12. Those writers are fixed and `tests/test_record_tool_call_sites.py` holds them to
it; your own `record_tool_call` calls are not covered by that, and this is where an omission in one
becomes visible. Reading an older corpus therefore moves numbers: calls that counted as declines
now count as unknown. That is the correction rather than a regression, so split a corpus by kit
version before comparing a rate across one.

`export_actions` now carries `outcome.cause` and `outcome.error` for exactly this reason: that
record is what reaches a trainer, and it previously carried neither, so the split could not be
reconstructed downstream even by hand.

Three write-side hazards worth stating, all observed:

- **Record the outcome ONCE, AFTER the branch.** A consumer that emits its event before checking
  `endpoint_error` destroys the distinction at write time, and no read-side fix recovers it.
- **Do not omit `ok` on the endpoint path.** Recording only `error=` is what makes
  `payload.get("ok")` return `None`. Record `ok=` alongside the endpoint string and the breaker
  flag: those three keys are what `payload_cause` reads, and a `cause=` keyword is not among them.
  This paragraph recommended passing one until 1.24.0, which read as a decline and now reads as
  unknown, so a writer following that advice declared an outcome nothing would ever see.
- **The OUTERMOST callable the model invokes owns the event.** Every tool here records under its
  own name, so wrapping one and then deciding a different verdict writes two events for one call,
  the inner one contradicting what the model was told. The kit's composable pieces therefore record
  nothing and say so: `make_model_tool`, `make_harness_tool` and `make_schema_validator`. If you
  wrap a tool that does record, such as `read_file`, your wrapper's verdict is the one the trace
  needs, and the inner event is a second row under a name the model never called.

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
- **Knowledge only.** `read_skill` returns the markdown text. It does NOT execute bundled
  scripts or expand linked files. A "just instructions" skill works fully; a skill that ships
  runnable helpers gives you only its prose.
- **Third-party skills work** if they use the `SKILL.md` + `name`/`description` convention:
  drop them in the dir and they are discoverable. But a skill's text becomes the main LM's
  context: treat untrusted skills as a **prompt-injection surface** and vet them. Frontmatter
  beyond `name`/`description` is ignored.

## MCP tools (connect an external MCP server)

`mcp_tools(server)` exposes an **external** [MCP](https://modelcontextprotocol.io) server's tools to
an `RLMTask` as ready-to-use tools. rlm-harness is a **client only**: it never runs a server and bundles
none; you point it at someone else's (a local stdio command, or a remote streamable-HTTP URL):

```python
from rlm_harness import mcp_tools

with mcp_tools({"url": "https://mcp.example.com/mcp"}) as tools:        # or {"command": "npx", "args": [...]}
    finding = MyTask(tools=tools).run(...)                              # the server's tools are now callable
# `tools=` (1.1.0+) is the per-instance override, the sanctioned way to attach tools that exist
# ONLY inside this `with` block. It REPLACES the class-body `tools` declaration.
```

Needs the extra: `pip install "rlm-harness[mcp]"`.

- **The connection is live for the `with` block** and torn down on exit (a stdio subprocess is
  terminated). Each tool call is recorded as a `tool_call` in the trace, like any other tool.
- **Sync, despite an async SDK.** The MCP Python SDK is async, but dspy.RLM invokes tools
  synchronously, so rlm-harness runs the session in a background thread and bridges each call across.
  (dspy's own `Tool.from_mcp_tool` makes an *async* tool for `dspy.ReAct`. It does not work on the
  RLM sandbox path, which is why `mcp_tools` exists.)
- **Security: MCP tools run HOST-SIDE**, outside the sandbox. A stdio server is a subprocess this
  process spawns. Treat an MCP server as a **trusted dependency**, and its output as a
  **prompt-injection surface** (untrusted LM context), exactly like fetched web content.

### Many servers, progressively: `McpCatalog` + `McpConnection`

`mcp_tools` is the single-server convenience: it materializes one server's tools as self-tracing
`dspy.Tool`s up front. For a consumer building its OWN progressive tool surface over **several** servers,
list servers, `load` one on demand, read its tools, `call`: use `McpCatalog`:

```python
from rlm_harness import McpCatalog

cat = McpCatalog([{"name": "docs", "url": "https://mcp.example.com/mcp"},
                  {"name": "shell", "command": "npx", "args": ["-y", "some-mcp"]}])
try:
    cat.servers()                      # [(name, description), ...]: every declared server
    cat.load("docs")                   # connect one on demand (a no-op under the eager default)
    for tool in cat.tools("docs"):     # RAW mcp Tool objects (name / description / input schema)
        ...                            # map them onto YOUR own tool shape
    text = cat.call("docs", "search", {"q": "..."})   # flattened result text
finally:
    cat.close()
```

- **Raw, and records nothing.** `McpCatalog` returns the server's RAW MCP `Tool`s (not `dspy.Tool`s) and
  emits no trace events: the consumer maps each tool to its own shape, and its own tool wrapper owns the
  `tool_call`. That keeps the catalog dspy-free and leaves tracing where the consumer wants it.
- **`McpConnection`** is the public single-server bridge `McpCatalog` manages one of per server (and that
  `mcp_tools` is built on); **`result_text`** flattens a `CallToolResult` to text. Both are exported for a
  consumer driving a connection directly.
- **Eager by default** (connect host-side, before the run): a subprocess spawn inside an async tool loop
  can hang asyncio; `connect="lazy"` defers each server's connect to its first `load` (opt-in). The same
  HOST-SIDE execution and prompt-injection notes as `mcp_tools` apply.

## A tool's docstring is prompt text

Before you write a tool, know where its docstring ends up. dspy takes `dspy.Tool.desc` from the
function's `__doc__` with no opt-in, renders one line per tool, appends the block to
`generate_action.signature.instructions`, and the adapter puts that string in the **system
message**. On a kit task with just `read_file` and `grep_files`, the rendered tool-doc block was
about two thirds of the instruction string, against a few hundred characters of actual input. On
five real consumer tasks the same block ranges from nothing to about a third; see
`docs/PROMPT.md` for the spread and for what fills the rest.

A tool docstring is not documentation that happens to be nearby. It is instructions, re-sent on
every turn of every run.

Three rules, all of them learned the hard way (CHANGELOG 1.18.0):

- **The INNER closure's docstring is the model's; the FACTORY's is yours.** Only the inner one
  reaches the prompt, which is why a factory can carry version history and measured percentages
  safely. A bare public function handed over directly, like `verify_quote`, has no factory to hide
  behind: its developer half goes in the module docstring instead.
- **Say what a caller needs and stop.** Keep the parameters, the return shape, the sentinel strings
  a model has to branch on. Drop the rationale.
- **Never write a number a factory argument can change.** `make_web_search_tool` builds its result
  cap into the docstring from `max_results`, because a docstring that states the default while the
  build enforces something else is a wrong instruction, not a stale comment.

You do not have to hand-format any of this. `RLMTask._build_rlm` normalises tool descriptions on
the way to the model: Sphinx markup becomes plain text and the indentation Python put in your
docstring stops arriving as a ten-space gap.

It reaches a plain function or a bound method, which is almost everything, and a `dspy.Tool`
through a copy. It does NOT reach a `functools.partial`, a callable instance, a coroutine
function, a `staticmethod` object or an `lru_cache` wrapper, because those are the shapes the
timing seam skips too and there is no kit-owned object to write on. If you wrap an MCP tool
yourself, wrap it as a function.

What normalisation cannot do is invent a description or shorten one.
`tests/test_tool_descriptions.py` catches an empty one, and checks each sentinel string a
description quotes against what the code really returns. It finds those sentinels by reading the
description rather than from a list, in the forms this kit writes them in, so if you invent a new
shape of prefix, add it to the forms that test enumerates.

Two limits worth knowing before you rely on it. It reads QUOTED spans, so a prefix your
description mentions without quoting is invisible to it whatever its shape. And it only checks the
direction from the description outward: a sentinel your code returns and your description never
names is not caught, which is true of three of the kit's own tools today, each recorded as such.
`tests/tool_description_baseline.json` catches one that grew. If you add a tool, add it to
`_build_every_shipped_tool`'s table in `tests/test_tool_descriptions.py`, then re-run the suite
with `RLM_UPDATE_DESCRIPTION_BASELINE=1` and commit the regenerated baseline JSON, which is
generated and not edited by hand.

**Running an optimizer's instructions.** `RLMTask(compiled_instructions=…)` takes the path
`optimize.save_program` wrote and runs the optimized text. Only the text is taken, read out of the
state file rather than loaded by dspy, so no demos and no other model arrive with it and your
tools, your interpreter and the sandbox guard are still the ones this task builds. While an
artifact is in use it REPLACES the rendered tool documentation rather than sitting beside it, so
what the model is told about your tools is whatever the optimizer wrote about them.

Save with `save_program(program, path, task=task)`, passing the RLM itself rather than a module
wrapping it: that records what the text was compiled against, and a state whose record disagrees
is refused rather than quietly describing your tools as they used to work. Four things are
compared, the signature, each tool's model-facing docstring, the output model's schema and the
sandbox's own description, so a kit upgrade that leaves all four alone is not a disagreement. Your
task's own instructions and the budget caps are recorded and not compared, because a typo fix
and a raised cap should not invalidate an artifact. The run records a `compiled_instructions`
event naming the source, the versions that compiled it and the one running now, what moved
without refusing, which caps only one side recorded, which checks were skipped because one side
could not supply them, any predictor the file named that this build does not have, and which
predictors it replaced, so a reader of the trace knows the prompt was not the shipped one.

`compiled_instructions=""` runs what the build ships, which is the A/B you need to decide whether
to keep an artifact.

**[`docs/PROMPT.md`](../docs/PROMPT.md) is the full picture**: the five owners of that one string
and their measured shares, which parts of dspy's REPL protocol you cannot replace, the one seam
that lets you contradict it, where skills and MCP descriptions land, and what GEPA actually
rewrites and costs.

## Running local commands (an isolated runner)

`make_command_tool(runner)` gives an `RLMTask` a `run_command` tool: the reusable half of
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
HOST-SIDE: outside the pyodide/deno sandbox that isolates the RLM's own REPL code (same as the
fetch/search providers and MCP servers). A naive `subprocess.run` runner is arbitrary code
execution steered by the model: the same class of danger as the refused `local` interpreter. For
anything processing untrusted input the runner MUST execute inside a disposable, network-restricted
container / VM / OS-sandbox; `examples/command_runner.py` is a reference Docker runner (`--rm
--network=none`, workspace mounted read-only). A command **allowlist is not a substitute**: a shell
allowlist is routinely bypassed (`make`/`npm run` script edits, `find -exec`, `git -c`, `$(...)`,
env-var injection), so the kit ships no allowlist primitive; the optional `guard` hook is a
shape-only pre-flight, never a security claim. One such guard ships ready-made:
`refuse_broad_git_history` refuses a `git log` invocation carrying a broad-history option (`--all`,
`--branches`, `--remotes`, `--tags`, `--glob`, `--reflog`, `--walk-reflogs`, `-g`,
`--alternate-refs`): an eval/training-run convention for stopping a model from reading branches,
tags, or reflogs it should not have task-specific hints from, same shape-only honesty as any other
`guard` (it is not a shell parser: a chained shell string bypasses it, by design, same as any
other `guard`).

- On success the model receives a `{"exit_code", "stdout", "stderr"}` dict (dspy JSON-bridges a
  `dict` into a real REPL value it reads: `run_command("ls")["stdout"]`; a dataclass would arrive
  only as its unsliceable `repr`, so the tool returns a dict and the runner returns the typed
  `CommandResult`). The trace keeps only `exit_code` + lengths + a stderr preview + `duration_ms`,
  like `fetch_url` records size not body.
- **The trace's stdout/stderr asymmetry is deliberate, and it has one blind spot worth knowing.**
  `stdout` is bulk content, so it gets a LENGTH only: the same rule `fetch_url` applies to a body,
  and the one that keeps a credential out of the JSONL even when redaction misses. `stderr` is
  diagnostic rather than bulk, so it gets a 500-character preview. **The blind spot: a tool that
  writes its diagnostics to STDOUT leaves nothing readable in the trace when it fails**: you get
  `exit_code: 1`, an empty `stderr_preview`, and `stdout_len: 1234`. The model still sees
  everything (it holds the full dict), so this costs a trace READER, not the run. If your commands
  behave that way and you need the text offline, have your `runner` fold the relevant stdout into
  `CommandResult.stderr`, or record your own event alongside. Do not expect the length to
  reconstruct it.
- Sync, like every RLM tool: wrap an async container/sandbox client into a sync call yourself.

**One-shot vs stateful: the runner decides.** `run_command` returns a single command's result and
holds no shell state; whether cwd, env, filesystem writes, and background processes persist across
calls is the *runner's* contract. The reference example is a fresh container per call: a stateless
**inspect** surface (read-only mount). An edit-build-test loop needs a **stateful** runner: a closure
over a long-lived sandbox: `docker create` + `docker exec`, an [E2B](https://e2b.dev) /
[Modal](https://modal.com/docs/guide/sandbox) / [Daytona](https://www.daytona.io) sandbox handle, or
a [SWE-ReX](https://github.com/SWE-agent/SWE-ReX) `BashSession`, which fits the same seam with no API
change. Interactive tools and tmux-style sessions are out of scope for a one-shot result; if a
consumer later needs model-managed sessions, that's the moment to add an additive `session_id` to the
payload, not before. (`dspy.RLM`'s own pyodide/deno interpreter is WASM Python and **cannot** spawn a
subprocess, so shell execution has to come from a host-side tool like this. There is no in-sandbox
alternative.)

## Reading and searching local files (a bounded directory, no shell)

`make_read_file_tool(root)` / `make_grep_files_tool(root, candidate_paths)`: the filesystem-side
analogue of `make_fetch_tool`'s SSRF-guarded `is_safe_url`, filling the gap between "no filesystem
access at all" and `run_command`'s full-shell escape hatch. `root` is not "a repo": it's any
bounded local directory tree a consumer scopes it to: a source repository, a docs corpus, an
extracted archive, a dataset directory, a log directory. The single most common thing a
coding-adjacent consumer needs, let the model read or search a bounded directory tree, does not
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
  OUTSIDE it: via `os.path.realpath` (which follows symlinks) then a `commonpath` containment
  check, never `os.path.normpath` (purely lexical, which would miss the symlink case).
- **`candidate_paths` is REQUIRED, consumer-supplied**: no default directory walk, no built-in
  `.gitignore` handling. Same base/wrap split as `make_command_tool` demanding an injected
  `Runner`: the kit owns the safety guard, the consumer decides which files are even candidates
  (walk a directory, read a manifest, whatever fits).
- **`name=` on both factories** (default `"read_file"`/`"grep_files"`) fixes a real collision: a
  task with more than one bounded root (a source root AND a docs root, say) needs each one's tool
  to have a distinct REPL identity, since dspy keys its tool dict by name and two tools sharing a
  name abort registration for EVERY tool on the task, not just the second one. Validated at
  factory-build time (a valid, non-reserved identifier). A bad name raises `ValueError`
  immediately rather than surfacing as an obscure construction failure later.
- **`make_read_file_tool`** additionally takes `encoding=` (default `"utf-8"`, for a non-UTF-8
  corpus), `max_output_chars=` (default `None` = unlimited; truncates with a visible, non-silent
  marker when set), and `line_numbers=` (default `False`; prefixes each returned line with its
  REAL 1-indexed file line number, so a model reading a slice starting mid-file doesn't have to
  compute one itself from `start_line`: removing exactly the off-by-one a model gets wrong when
  later asked to cite or edit that line). The last two are scoped to the successful-read branch
  only: a `Refused`/`Read error` string is never numbered or truncated. **If you also verify
  citations, read "Line numbers and `verify_quote`" below BEFORE turning `line_numbers=` on**: the
  two must not see the same string, and getting it wrong silently fails correct citations.
- **`make_grep_files_tool` requires the optional `regex` package outright** (`pip install
  "rlm-harness[grep]"`, a friendly `ImportError` otherwise, no silent fallback to stdlib `re`).
  `pattern` is LM-controlled, unbounded regex, matched against real file lines with no wall-clock
  budget anywhere else in a tool's call path: a catastrophic-backtracking pattern (`(a+)+$`
  against a non-matching line) can hang the host process indefinitely on stdlib `re`, and stdlib
  `re` cannot be bounded by ANY pure-Python mechanism, not even `signal.alarm` (CPython's `re`
  engine doesn't yield to the signal dispatcher mid-match: one `re.search()` call is a single,
  uninterruptible C-level operation). `regex`'s own matching loop periodically checks elapsed
  wall-clock time internally and raises `TimeoutError` when exceeded: a real, working,
  pattern-structure-agnostic mechanism. Same "no silently-weaker substitute" posture
  `make_json_schema_validator` already takes for its own optional `jsonschema` extra.
- Two composed budgets, both factory (operator) parameters, never model-controlled:
  `per_match_timeout_s` (default `1.0`) bounds ONE line's match: a timeout skips just that line
  (counted, surfaced in the result, never silent); `max_total_time_s` (default `30.0`) bounds the
  WHOLE call, checked before EVERY line (not merely once per file. A per-file-only check would let
  a single large file with many timeout-tripping lines blow past the budget by an arbitrary
  multiple before it ever fired again).
- **`output_mode=` and `ignore_case=` on `grep_files`.** `output_mode` (default `"content"`,
  unchanged) adds `"files_with_matches"` (distinct matching file paths only, no line text) and
  `"count"` (`path: N`, files with zero matches omitted). `max_results` caps the number of MATCHES
  found (not total output lines: see `context_before`/`context_after` below). `ignore_case`
  (default `False`) is case-insensitive matching as a first-class flag rather than something baked
  into the pattern.
- **Per-file early-break, `"files_with_matches"` only.** That mode only needs to know "did this
  file have ≥1 match": scanning stops the instant one is found, moving to the next candidate
  file. `"count"` mode cannot do this. It needs the file's exact total match count, so every line
  there is still scanned. Both `"files_with_matches"`/`"count"` additionally stop OPENING further
  candidate files once `max_results` qualifying files are already found (an outer-loop break: it
  never skips a line of a file already being scanned, only avoids starting further ones).
- **`context_before=`/`context_after=`** (default `0`, `"content"` mode only: a silent no-op,
  never an error, in the other two modes, which have no per-line text to attach context to): show
  that many unchanged lines immediately before/after each match, using grep's own convention: a
  match keeps `path:line: text` (colon); a context line uses `path-line- text` (hyphen); a `"--"`
  line separates two blocks that don't touch (a numbering gap) within the same file: never at a
  file boundary, since the path prefix already marks that. A line that itself matches is ALWAYS
  emitted as a match, never as leftover context from an earlier match's `context_after` window: a
  fresh match resets the after-context countdown outright, it never stacks with one still running.
  `max_results` counts MATCHES only: context/separator lines are supplementary and uncapped by
  it, mirroring real `grep -m`; when a match hits the cap, its own trailing context may be
  truncated if the file/budget ends first (an accepted, deliberate simplicity choice, the same one
  `max_total_time_s` already makes for whatever's in-flight). When both are `0` (the default),
  behavior, including the traced `result_count`, is byte-identical to a build of this tool with
  no context support at all.

### `list_candidate_paths`: the recommended way to build `candidate_paths` (`tools/discover.py`)

`candidate_paths` stays a plain, required list: no contract change. But safely walking a
directory tree to build one (respecting `.gitignore`, never escaping `root` via a symlink, never
treating VCS internals as candidates) is exactly the kind of mechanic every consumer would
otherwise reinvent. `list_candidate_paths(root)` is a plain host-side function (no factory, not a
REPL tool: called from a consumer's own setup code before wiring a tool, the same role
`resolve_within_root` already plays) returning `CandidatePaths(paths, truncated)`, pipeable
directly into `make_grep_files_tool`/`make_read_file_tool` with zero reshaping:

```python
from rlm_harness.tools import list_candidate_paths, make_grep_files_tool, make_read_file_tool

candidates = list_candidate_paths(repo_root, glob="*.py")
read_file = make_read_file_tool(repo_root)
grep_files = make_grep_files_tool(repo_root, candidate_paths=candidates.paths)
```

- **Needs the optional `rlm-harness[gitignore]` extra (`pathspec`) ONLY when there's an actual
  `.gitignore` pattern to compile**: a real root `.gitignore` present (with
  `respect_gitignore=True`, the default) or a non-empty `extra_ignore_patterns`. A caller with
  neither never needs the dependency installed. `.gitignore`'s own subtler syntax (negation,
  directory-only patterns, anchoring) is why this uses `pathspec` rather than a hand-rolled
  parser: slightly wrong parsing either leaks a file a consumer explicitly meant to exclude
  (`.env`, credentials) or wrongly excludes real source; same "don't reinvent a
  correctness-critical mechanic" reasoning behind `make_grep_files_tool`'s `regex` extra and
  `make_json_schema_validator`'s `jsonschema` extra.
- **Root-level `.gitignore` only, stated honestly**: no nested per-directory `.gitignore`
  merging, no global gitignore. `extra_ignore_patterns` (same `gitwildmatch` syntax) is the escape
  hatch for a consumer that needs more than the one root-level file covers.
- **`.git` is always excluded, unconditionally, by two distinct mechanisms**: a directory named
  `.git` is pruned from the walk (never descended into), and, independently, a plain FILE
  literally named `.git` is also always excluded, since that's git's REAL submodule gitlink shape
  (a one-line pointer file, not a directory); pruning the directory case alone would miss it.
- **Every candidate file is re-checked through `resolve_within_root`**, independent of
  `follow_symlinks` (which only gates whether `os.walk` descends into a symlinked directory): a
  symlink pointing outside `root` never surfaces as a candidate either way.
- **`max_files`** (default `5000`) bounds the walk itself, stopping it outright rather than
  slicing an unbounded result after the fact; `CandidatePaths.truncated` makes a partial result
  visible, never a silent cutoff. Directory and file names are sorted at every level of the walk
  first: `os.walk`'s own order is filesystem/OS-dependent, which would otherwise make WHICH files
  survive a truncation non-reproducible across runs on the same tree.
- **No REPL-tool wrapper yet**: this is a setup-time helper a consumer's own code calls, not
  something the model calls mid-trajectory; a `make_list_files_tool` wrapper is a natural,
  separately-reviewable future addition.

### `make_git_clone_tool`: safe git clone with fallback auth (`tools/git_clone.py`)

Base/wrap, the same shape as `make_fetch_tool`/`make_command_tool`. The kit does NOT shell out to
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

- **URL safety reuses `is_safe_url` directly**: no reinvented SSRF check. Syntactic pre-flight
  only, same caveat `fetch.py` already documents for its own `fetcher`: a public hostname
  resolving to a private address at actual connect time is NOT caught here, since the real
  network connection happens inside the isolated `cloner`, not in this wrapper: the `cloner`
  should call `resolved_host_is_safe` internally at connect time if that matters for its
  deployment.
- **Destination confinement reuses `resolve_within_root` directly**: `dest_dir` is resolved
  exactly like `write_file`'s `path`.
- **Fallback auth: a two-attempt orchestration, never a retry loop.** Tries without credentials
  first (the common, public-repo case costs nothing extra); on failure: a nonzero exit code OR a
  raised exception, handled identically: if a `get_credentials` provider was configured, ONE
  retry with credentials. Never more than two `cloner` invocations per call.
- **Credential redaction, disclosed as best-effort, not absolute.** A `get_credentials(url)`
  provider returning a dict MUST include a `"secret"` key (the raw credential string); after a
  credentialed attempt, that exact string is redacted (plain string replacement) before it can
  reach the traced `stderr_preview` or the model-visible return string. `stdout` is redacted too
  before its (post-redaction) length feeds the trace's `stdout_len`: `stdout` content itself is
  never traced verbatim, only that length, matching `run_command`'s own existing "lengths + a
  preview, not the full stream" posture. This does NOT catch a derived or transformed leak
  (URL-encoding, case-folding, a truncated echo from a misconfigured credential helper). A
  malformed dict (missing/non-string/empty `"secret"`), or a provider that itself raises, fails
  CLOSED: treated exactly like a decline, no retry attempted, never crashes the call.
- **`default_depth=1`** (shallow clone by default) is passed through to the `cloner` as a plain
  argument: the "avoid being tricked into cloning an enormous repository" mitigation;
  `default_depth=None` opts out for a caller that explicitly wants full history.
- **On success**: `"Cloned {url!r} into {dest_dir!r}."`: terse, matching `write_file`'s own
  success-string convention. A consumer wanting to see what landed already has
  `list_candidate_paths`/`read_file`/`grep_files` for that.
- **Accepted, disclosed gap**: no cleanup of a partially-cloned directory on failure, making the
  whole clone atomic would require dictating how the `cloner` itself writes to disk, contradicting
  the base/wrap split this design is built on.

## A long input as a file, not a variable (`LargeText`)

An input field reaches the sandbox as an inline Python literal, rebuilt on every turn. For a
multi-megabyte text that is the most expensive thing a run does, and it is invisible until a host
runs out of memory. `LargeText` stages the text as a file the run's sandbox is granted instead.

```python
from rlm_harness import LargeText

with LargeText(huge_log) as log:
    answer = await Investigate().arun(log=log, question="What happened at 04:12?")
```

Pass one where the signature declares a `str`. The kit substitutes the path before dspy sees it,
adds a line to that run's instructions saying the variable is a path to a file of N characters and
to seek and read a slice, and grants the sandbox that one file. `examples/large_text_run.py` is the
whole shape on a live model.

**Measured, by a consumer on Linux aarch64 in its own deployed image**, peak RSS of the whole
process tree in MB, medians of three runs, a 20 ms sampler whose Deno peak equalled the kernel's
`VmHWM`:

| context | as a variable (1 / 4 turns) | as a file, read whole | as a file, read in 64 KiB slices |
|---|---|---|---|
| tiny | 496 | 481 | 456 |
| 11.6 MB plain text | 1194 / 1311 | 516 / 540 | 492 / 517 |
| 50 MB of control bytes | 2843 / 5189 | 795 / 889 | 671 / 712 |

Their worst case went from 5.2 GB to about 0.8 GB. The same consumer then measured the same
arms on an **x86_64** host through the same deployed path: 551 MB for the 11.6 MB text read whole
over four turns, within 2% of aarch64, and LOWER on both 50 MB rows, 835 MB read whole against 889
and 611 MB sliced against 712. So the ordering and the magnitude hold on both architectures, and
the sliced row is the cheapest on each. Every run submitted the staged file's length
and SHA-256 and every one matched the host's, so the cheap rows are reading the payload and not
reading nothing.

**The bytes still reach the sandbox.** dspy's Deno side reads the file and writes it into Pyodide's
filesystem, so the payload materialises once. The saving is one copy against tens, not one against
none, and anyone restating this as "the context never enters the sandbox" has it wrong.

**Model code must open the path it is given**, which is `/sandbox/<generated name>`. The host
location does not resolve inside the sandbox, so sandbox code handed `host_path` finds no file and
reads nothing. The failure is quiet and it looks like a win: a run that reads nothing costs
nothing, which is how the first attempt at measuring this approach reported no cost at all.

**`make_read_file_tool` is the wrong tool for a staged text.** Pointed at the permitted path it
showed no gain at all, about the same tree peak as the variable path. Two structural reasons. It
slices by LINE, so a context that is effectively one very long line comes back whole from a single
call. And it reads the file on the HOST per call, which pays the copy the staging exists to avoid.
Offset reads inside the sandbox are what the table above measures, and they need no tool.

**The one opening in the sandbox boundary.** Granting read access to a host file widens the
boundary this kit's invariants open with, so every part of the path is owned here. The filename is
generated; there is no name, prefix, directory or path argument, so no caller value and no model
output becomes part of it, and the text is the file's content rather than any part of its location.
The grant is that one file, read-only, and only for a run whose inputs carry one of these objects.
Constructing one takes host Python, so no model reply, no tool result and no string that looks like
a path can open a grant or widen one. A comma in the resolved path is refused outright, because
Deno's permission list is comma-separated and misparses an entry containing one. Any other
interpreter kind refuses a read path rather than accepting one it cannot honour, and a run that
stages nothing grants nothing, which is every run that existed before this.

**`$TMPDIR` decides whether the staged copy is disk or RAM.** The staging goes wherever
`tempfile` points, and a container's `/tmp` is often a tmpfs, so the one copy this approach costs
can be a RAM copy rather than a file on a disk. That does not change the saving, which is one copy
against the tens an inline literal costs, but it does change where the one lands: a consumer moved
`$TMPDIR` to a real volume for exactly that reason once they were running it in production. Set
`$TMPDIR` per process at startup rather than reassigning `tempfile.tempdir` mid-run, since that
setting is process-wide and captures every other thread's temporary file while it is in effect.

**There is no staging-directory argument, and none is needed for isolation.** The staging
directory comes from `tempfile.mkdtemp()`, which creates it at `0700`: group and other have nothing,
and the file inside is `0400`. A consumer reached for a process-wide `tempfile.tempdir` override to
get its own `0700` directory, which is both unnecessary and worse, since that setting is global and
any other thread's temporary file lands there for as long as it is in effect. If you need the
staging on a PARTICULAR filesystem, that is a different request and worth making; it has not come
up, and a directory argument is the shape that lets a caller value, and then an attacker-controlled
value, reach the path the sandbox is granted.

**Deleting the staged file is the caller's**, through `close()` or the context manager. The kit does
not, because the same object can serve a second run. The file is created read-only and the delete
clears that bit first, which matters on Windows, where a read-only file cannot be unlinked and the
removal would otherwise be skipped in silence.

**Where this has run.** The staging, the type-only discovery and every refusal are tested offline on
every platform CI covers. The MOUNT needs Deno: it was exercised by hand on macOS, and by a
consumer in production on Linux **aarch64 and x86_64**, through a `sudo` path where the staging and
the sandbox share a uid. No permission problem appeared there, which is one image rather than a
general result: `0400` plus a container that runs the sandbox as a different user than the one that
staged the file would still present as a mount failure rather than a permission error. A Windows
host has not been through it at all.

## Writing and editing a bounded local directory (`tools/edit.py`)

`make_write_file_tool(root)` / `make_edit_file_tool(root)`: the write side of the read/search
pair above, kept in a separate module (`fs.py` is already the largest single file in `tools/`, so
"everything that can mutate the filesystem" stays physically distinct from "everything that only
reads it"). Same `resolve_within_root` guard, same `name=`/`encoding=` parameters and validation
as `make_read_file_tool`/`make_grep_files_tool`, including the same multi-root name-collision fix.

```python
from rlm_harness.tools import make_edit_file_tool, make_write_file_tool

write_file = make_write_file_tool(repo_root)
edit_file = make_edit_file_tool(repo_root)
finding = MyTask(tools=[read_file, grep_files, write_file, edit_file]).run(...)
```

- **`make_write_file_tool`**: creates or overwrites a whole file, atomically
  (`atomic.atomic_write_text`). Unconditional overwrite (no create-only mode: a consumer wanting
  one can call `read_file` first and check for its "missing file" error string) and no
  `max_content_chars` cap (the content was generated by the model itself, so it's already bounded
  by whatever produced it. This does NOT cover a model looping on the tool and filling the host's
  disk with many individually-bounded files, an accepted, out-of-scope-for-now gap).
- **`make_edit_file_tool`**: exact-string-anchor replacement, the same uniqueness-checked
  contract Claude Code's own `Edit` tool and `nano-rlm`'s `edit` skill both use independently.
  Refuses (never mis-edits) if `old_string` isn't found, or is found more than once and
  `replace_all` (a per-call flag the model sets, not a factory-level fixed behavior) is `False`.
  the file is left byte-for-byte untouched on every refusal path. `old_string == ""` and
  `old_string == new_string` are refused as degenerate inputs; `new_string == ""` (delete this
  text) is a legitimate operation and is NOT refused. **Known failure mode**: a file using
  different line endings than the `old_string` the model supplies fails closed with "not found"
  rather than mis-editing: not a safety bug, just worth knowing.
- **On success, a windowed snippet of the result is appended** (`show_snippet=`, default `True`;
  reuses `read_file`'s own `f"{lineno:>6}\t{line}"` numbering) so a model can confirm what its
  edit actually did without a separate `read_file` round-trip: `show_snippet=False` is the escape
  hatch back to the terse confirmation alone. `snippet_context_lines=` (default `3`) bounds each
  shown region; an edit larger than that shows only its own head/tail with a visible "N line(s)
  omitted" marker, never an unbounded dump regardless of how large `new_string` was.
  `max_snippet_occurrences=` (default `3`) caps how many `replace_all=True` occurrences get their
  own snippet block. The file is still fully edited regardless of the cap; a truncated result
  says so explicitly ("N more occurrence(s) not shown"). Overlapping windows for closely-spaced
  occurrences are shown independently, not merged. Scoped to the success path only: a
  `Refused`/`Read error`/`Write error` string never gets a snippet appended.
- **This is the kit's first file-mutation/data-loss-capable tool category**: every tool shipped
  before this was either read-only against the filesystem or delegated execution/network entirely
  to a consumer-supplied runner/fetcher. Both factories build on `atomic_write_text`, which is
  what makes an overwrite/edit crash-safe (never half-written), but its guarantee is "no torn
  read," never "serializes concurrent read-modify-write across processes": two SEPARATE `RLMTask`
  runs (or two workers in a batch eval) racing an edit on the SAME file can silently lose one of
  the two updates. Accepted, out-of-scope-for-now: stated here rather than silently omitted.
  Building `make_write_file_tool`/`make_edit_file_tool` on `atomic_write_text` also surfaced a
  real bug in that primitive (fixed in `atomic.py` itself, not duplicated per-tool): overwriting
  an existing file through it used to silently reset the file's permission bits to `0600`
  (`tempfile.mkstemp` always creates its temp file at that mode, and `os.replace` doesn't carry
  the destination's mode across): now the destination's existing mode is preserved across an
  overwrite.

## Extracting archives safely (`tools/archive.py`)

`make_extract_archive_tool(root)`: a safe `zip`/tar extraction tool, applying the same
`resolve_within_root` reasoning `read_file`/`write_file` use for a single path argument to every
entry of an archive.

**The stdlib's own behaviour here is both format- and version-dependent, which is the reason to own
the check rather than inherit it.** Measured: `zipfile.extractall()` sanitises an absolute path and
a `..` traversal and never creates symlinks, on every supported version, so the classic "zip slip"
framing is the case it is least true of. `tarfile.extractall()` does escape on 3.11 and 3.13, and
stops on 3.14, whose default `data` filter raises `OutsideDestinationError`. This tool refuses by
its own containment regardless of format or version, which is a promise the stdlib does not make.

```python
from rlm_harness.tools import make_extract_archive_tool

extract_archive = make_extract_archive_tool(repo_root)
finding = MyTask(tools=[extract_archive, read_file, write_file]).run(...)
```

- Supports `.zip` and tar variants (`.tar`, `.tar.gz`/`.tgz`, `.tar.bz2`/`.tbz2`, `.tar.xz`/
  `.txz`), dispatched by extension. An unrecognized extension returns an error string, never
  raises, never sniffs content.
- **Two-pass extraction, matching this kit's "refuse outright, never partially mutate" posture**:
  Pass 1 validates every entry's metadata ONLY (name, type, declared size, and, zip-only, the
  encryption/compression-method header fields) and refuses the WHOLE operation upfront on any
  violation, before a single byte is written: a zip-slip path, a symlink/hardlink/device entry,
  an encrypted entry, or an unsupported compression method all fail here, with nothing landing on
  disk. Pass 2 (only reached once Pass 1 fully passes) streams each entry's real bytes via the new
  `atomic_write_stream` primitive (below), bounding peak memory to a small, fixed chunk size
  regardless of that entry's own size.
- **A declared size cannot smuggle more decompressed output than it promises**: confirmed
  empirically: both stdlib read APIs hard-ceiling their output at the entry's own declared size
  field, so a "lying header" memory bomb does not exist via either. Pass 1's cumulative
  declared-size check (`max_extracted_bytes`, default 200 MiB) and entry-count check
  (`max_entries`, default 10,000) are factory (operator) parameters, never model-controlled: same
  posture `make_grep_files_tool`'s timeouts and `list_candidate_paths`'s `max_files` already take.
- **No password-protected/encrypted archives**: refused upfront in Pass 1, with a clear reason,
  never a crash. **No nested-archive recursion**: an archive found inside the extracted output is
  not itself auto-extracted; call the tool again on it, subject to the same checks a second time.
  **Unconditional overwrite** of existing files at the destination, matching `make_write_file_tool`.
- **`atomic_write_stream(path, chunks, *, max_bytes=None)`** (new, in `atomic.py`): a separate,
  additive primitive alongside `atomic_write_text` (not a refactor of it): the same
  same-directory-temp-file/`fsync`/`os.replace`/permission-preservation idiom, but for a caller
  with an iterable of `bytes` chunks rather than one already-in-memory blob, aborting the moment a
  running total exceeds `max_bytes`: checked after every chunk, not merely once at the end.

## Memory: a provenance-typed store that outlives a run (`memory.py`, `tools/memory.py`)

A run's REPL forgets everything when it ends. `MemoryStore` is the part that does not: one SQLite
file of short texts that your host code puts things into and a model reads from inside a run. It
needs no extra and no service.

```python
from rlm_harness import MemoryStore
from rlm_harness.tools import make_memory_get_tool, make_memory_note_tool, make_memory_search_tool

memory = MemoryStore("state/memory.db")
memory.add(page_text, provenance="observed", origin=url, tags=["databases"])
memory.add("Prefer answers that cite the official documentation", provenance="said", origin="user")

tools = [make_memory_search_tool(memory), make_memory_get_tool(memory), make_memory_note_tool(memory)]
answer = MyTask(tools=tools).run(question=q)
```

`examples/memory_run.py` runs that on a live model and carries on through the rest of this
section: `compute_memory_use` over the finished run, `consolidate_run`, and `purge`.

**Every item carries a provenance, and there are exactly three** (`PROVENANCES`):

| Provenance | Meaning | Who writes it |
|---|---|---|
| `observed` | text taken from a tool output or an external source | your host code |
| `said` | text a person wrote | your host code |
| `inferred` | text a model wrote: a note, a summary, a revision | the note tool, or your host code |

The kit cannot check which of the first two a host-side `add` really is, so choose the label from
the code path the text arrived by, never from anything the text says about itself. What the kit
does guarantee is that a model writes only `inferred`. `memory_note(text, revises="")` has no
argument that names a provenance, so no prompt, injected page or confused plan can make a model's
words arrive as `observed` or `said`. The rule holds because of the tool's shape, so nobody has to
remember to check it.

That label is what makes a note safe to keep. A note written after reading a hostile page is
returned to every later run whose search matches it, which makes memory a persistent
prompt-injection path. `inferred` is how a reader, or a later task's instructions, tells a model's
own words from a source and declines to cite them.

**What is written is never changed, an id never names different text, and only the host erases.**
Four calls change what a reader sees, and none rewrites a row:

- `revise(item_id, text, *, origin, provenance="inferred")` adds a new item whose `revises` points
  at the old one. Pass `provenance="said"` when the revision is a person's correction.
- `retire(item_id, *, at=None)` sets `valid_to` once. It is the only UPDATE the module issues.
- `label(item_id, ...)` adds label rows (below); the latest shows, and a person's outranks a model's.
- `merge` / `unmerge` add alias rows that every reader applies to entity names (see "Merging
  names").

`purge` is the one way anything leaves the file; see "Removing data" below.

SQLite triggers refuse a stray `UPDATE`, `DELETE` or `INSERT OR REPLACE` against every table that
holds a row (items, labels, failures, purge records, aliases, seen names). This protects history from a host's own SQL mistakes; it is not a security boundary against
someone holding the file, since `DROP TRIGGER` still works. An index row deleted by other means only
makes search miss that item; it never changes the item.

**Search** (`search(query="", *, limit=10, provenance=None, tags=(), entities=(), at=None,
include_retired=False)`):

- A query matches an item's text, its origin when the store was opened with `index_origins=True` (so
  an address or a file name is searchable), or one of its current labels. An item that matches in a
  label ranks above one that matches only in its text or origin, because a label says what an item
  is about; within each group, by `bm25` and then newest first (among those matches, a hit in the
  item's own tags outranks one in its body, and one in its origin weighs no more than one in its
  body). A superseded label is not searched.
- A query is split into terms the way the index's tokenizer splits text (letters, digits and
  combining marks, so a Hindi or pointed Hebrew word stays whole). Thai, Lao, Khmer and Myanmar are
  written without spaces and are not split into words, so a query in them matches only a whole
  run.
- `tags` needs every given tag among the item's own tags or its current tag labels; `entities`
  needs every given name among its current entity labels, compared casefolded.
- An empty query lists the newest items. A query with no usable terms (punctuation, a lone letter
  or digit) matches nothing.
- Items retired as of now are left out unless `include_retired=True`; one retired at a future time
  stays visible until then. `at=<timestamp>` is a point-in-time view: only items whose `valid_from`
  has passed and that were not yet retired at that moment. Labels have no time window, so `at`
  always reads the current ones.
- Chinese, Japanese and Korean need no dictionary. That covers kana, the CJK ideograph blocks
  including extensions A to G with the marks `々 〆 〇 〻`, and Hangul syllables and both jamo
  blocks. An unspaced run is indexed as its overlapping character pairs and its single characters,
  so `索引` matches inside `資料庫與索引` and a one-character query still finds something. Text is
  NFKC-normalised, so `ＰＤＦ` matches `pdf`.

**The three tools.** Reading and writing are separate factories, the way `read_file` and
`write_file` are, so a task that should only consult memory leaves the note tool out.

- `make_memory_search_tool(store, *, max_results=8, max_chars=500)`: `memory_search(query, limit)`
  returns a JSON list, each row with the item's current `title` and its `title_provenance`, since a
  model-written title is a label and never evidence. `limit` is clamped to `max_results`, each text
  is cut at `max_chars` and marked ` …[truncated]`, and items retired as of the call never appear.
- `make_memory_get_tool(store, *, max_chars=4000)`: `memory_get(item_id)` returns one item, retired
  or not, with every field (including `origin`, `run_id` and your `meta`, so keep private data out
  of `meta`), plus `revised_by`.
- `make_memory_note_tool(store, *, max_chars=2000, tags=())`: `memory_note(text, revises)` stores an
  `inferred` item, with `origin` set to `run:<run_id>` inside a traced run. Its refusals are strings
  the model can read: an empty text, a text over `max_chars`, an unknown id, or a text the store
  itself refuses (over its `max_item_chars`, or not encodable).

Each takes `name=` so two stores can sit in one task. Each records exactly one `tool_call`,
failures included, with the raw arguments the model passed and a `memory_ids` field listing the ids
it returned or wrote; there is no new trace event type. An argument longer than its cap is
recorded cut, with `<name>_len` beside it, so an oversized argument cannot bloat the trace (a list
or dict is measured by its JSON form). Each returns a string and never raises on a model-controlled
argument, and a database failure (a lock held past the timeout, a full disk, a damaged file) comes
back as `Memory error: <type>`.

### Labels and distillation (`memory_tasks.py`, `memory_organise.py`)

A label says what an item is about: a `title`, a `summary`, `tags` and `entities`. Labels are rows
of their own, so the item itself never changes.

```python
from rlm_harness import distill_pending

memory.label("m12", title="Replication slots", tags=["databases"])    # a person's labels: said
report = await distill_pending(memory, limit=50, language="English")  # a model's: inferred
memory.labels("m12")   # MemoryLabels(title=..., summary=..., tags=(...), entities=(...), provenance={...})
```

- **Which label shows.** For each kind, the latest `said` label when it has a value, otherwise the
  latest `inferred` one. So a person's label outranks any later summary, and `label(id, title=None)`
  (a `said` clear) lets the model's show again. Labels are trimmed, never refused: title 80
  characters, summary 600, up to 8 tags of 40 (casefolded), up to 8 entities of 60 (case kept).
- **The queue.** `unlabelled()` lists items with no summary yet, newest first. It is derived from
  the labels, so nothing can be stuck in a "being summarised" state. An empty summary from a model
  counts as an answer, so a text too thin to summarise is asked about once; a person clearing their
  own summary does not, so the item is queued again. An item that failed three times
  (`note_distill_failure`, `last_distill_error`) leaves the queue unless named in `item_ids`.
- **The pass.** `distill_pending` runs only when you await it, at most 500 items per call. A text up
  to `SHORT_LIMIT` (12,000) characters is ONE `dspy.Predict` call (`SummariseShort`). A longer one is
  an RLM run (`DistillLongItem`) with the text in the REPL and a section map in the prompt, and only
  that path needs Deno; `defer_long=True` skips it. The labels already in use (`label_counts`) are
  shown to each call, so related items converge on one tag instead of two near-synonyms. The
  returned `DistillReport` lists what was labelled, what came back with an empty summary
  (`skipped_empty`, the model's judgement that the text was too thin), what failed and why, and
  whether `should_stop` ended the pass early.
- **What a label may say.** Coordinate markers are stripped. An entity is kept only when its name
  occurs in the item as a whole word (for a long item, in the span its quote was found at, the quote
  checked like `verify_quote` and at least 20 non-space characters), or when it is an entity name already in
  use that the call was shown, which is what lets a text in one script carry a name other items
  write in another. A name matched that way is stored in the spelling already in use, so one entity
  keeps one name.
- **Running the model elsewhere.** `run=` replaces the default step runner. It receives a
  JSON-serialisable payload, which names the item it labels in `item_id` (never passed to the
  model), and returns a dict, so a consumer that isolates model calls passes
  `run=subprocess_step_runner(timeout_s=300)`, which runs each step in a fresh process through
  `run_in_subprocess`, configured with this process's `RLMConfig`. Cancelling the pass does not stop
  a child already running (it runs until done or `timeout_s`), and a long item's steps run in the
  child, so they do not reach a `recorder_for` trace. A short item's one `Predict` call is never
  recorded, in-process or not, so its run holds only `run_start` and `run_end`.
  `recorder_for(item_id)` gives each item its own trace run, and the labels it writes carry that
  run's id. `should_stop` and `on_progress` make a long pass stoppable and observable. A database
  error while writing (a lock held past the timeout, as behind a long purge) fails that item without
  counting toward its three failures and never ends the pass. So does a `StepNotCounted` that a
  `run=` raises for a failure that is not the item's (a stopped call, an unreachable model); raise it
  with the original error's text as its message, which every pass reports as given. Two passes at
  once may label one item twice; keep one running.
- `optimize.compile_task` compiles `DistillLongItem`; compile `SummariseShort`'s `Predict` with
  `dspy.GEPA` directly.

### Merging names (`aliases`)

Labels are written one item at a time, so one entity arrives under several names: `Kubernetes`,
`k8s`, a translation. An alias table says which are the same, and every reader applies it, so no
label is ever rewritten and a merge is undone by a new row.

```python
memory.merge("k8s", "Kubernetes")          # a person's merge: said
report = await align_pending(memory)       # a model's merges: inferred, one call per batch
memory.resolve("K8S")                      # "Kubernetes"
memory.unmerge("k8s")                      # and apart again; no model can merge it back
```

- **Who decides.** A person's (`said`) decision about a name stands whenever one exists, an
  unmerge included: the model's earlier merge does not show through it, and no later `inferred`
  merge or unmerge can touch that name, in either role, so a model cannot rejoin a separated pair by
  merging it the other way round. `unmerge` on a name that is not an alias yet records "keep
  these apart" ahead of time.
- **What a merge may do.** Both names must be entity names in use (`entity_names()`); a canonical
  that is itself an alias is followed first, so the table never holds a chain; a name that is
  already a canonical or a live alias is refused; and a model's merge must name something it was
  asked about. A refusal is a `ValueError` whose message is the reason. Names are compared folded
  (NFKC, casefolded). A canonical keeps the spelling its group already has (else its most used
  one), and stays mergeable into even after the last item naming it is gone.
- **Who reads it.** `labels()` shows merged names as their canonical, `label_counts("entities")`
  counts a merged group as one name, once per item, `search(entities=...)` matches every name in the group, and a query
  that is an entity name searches all of its names, so `k8s` finds an item that only ever says
  `Kubernetes`. The labelling pass accepts an alias of a name in use as grounding and stores it as
  written, so readers resolve it and an unmerge restores every label written while the merge was
  live; the pass is not shown aliases, only each name in use once.
- **Reading names yourself.** `readings()` is the map those readers apply, `folded name -> the name
  it is read as` (aliases, and case or width variants of a canonical); look a name up by
  `fold_name(name)`, the form the store compares names in. `resolve(name)` is one lookup of it.
- **The pass.** `align_pending` runs only when you await it. It takes up to `limit` names nobody has
  considered yet (`unseen_entities()`, capped at 500) to ONE `AlignEntities` call (an `RLMTask`,
  with every name in use and its count in the REPL), applies what comes back through
  `merge(..., provenance="inferred", batch=...)`, and marks the batch seen whether or not anything
  merged, so a store that grows by ten names pays to align ten names. A failed call marks nothing
  seen, and a name whose merge hit a database error stays unseen. A chained answer is applied link
  by link, so a link the store refuses stops the chain there. `AlignReport` lists what was asked, accepted and refused (with reasons). It takes `run=`,
  `recorder_for` (called with `"align"`) and `should_stop` like the labelling pass; since it makes
  one call per batch, `on_progress` receives `{"batch", "ok", "error"}` once, after that call.
- `entity_graph(limit=200)` returns pairs of canonical entities named together by one item, with
  counts, strongest first: data to draw, not a drawing.

### Topics (`create_topic`, `suggest_topics`, `organize_pending`)

A topic is a named group of items a reader goes back to together. Topics, their names, their
retirement and their members are all event rows, so nothing is rewritten.

```python
topic = memory.create_topic("Container orchestration")      # said
memory.place("m12", topic.id)                                 # said
for s in suggest_topics(memory):                              # local, no model call
    print(s.item_id, s.topic_id, s.score, s.shared_entities)
report = await organize_pending(memory, max_new_topics=10)    # a model, inferred
```

- **Who decides.** A topic's name follows the label rule: a person's name outranks any model's.
  Membership follows the alias rule: a person's decision about an item and a topic stands whenever
  one exists, a removal included, so `unplace` keeps that pair apart for good and no later model
  placement can undo a person's. A retired topic leaves listings and suggestions and takes no new
  members; its old members stay readable, and `restore_topic` brings it back.
- **Reading.** `topics()`, `topic(id)`, `topic_items(id)`, `item_topics(id)`, `search(topic=...)`
  and `entity_graph(topic=...)`. `topic_graph()` returns pairs of live topics whose members name the
  same entities; a bridge stays when it is among the three strongest of either topic, so every topic
  keeps its strongest links, a small one is never cut off by a busy one, and the dropped bridges are
  counted per topic.
- **Local suggestions.** `suggest_topics` offers each item in no topic its best live topic, scored
  2 per entity and 1 per tag it shares with the topic's members (entities read through aliases),
  from a score of 3. A dismissed pair (`dismiss`) or one a person took apart is never offered.
  `landing=` names a catch-all topic that still counts as none and is never suggested. It reads
  every item's labels in one query and joins them in Python, so it is meant for stores of thousands
  of items, not millions.
- **The organising pass.** `organize_pending` runs only when you await it. Each round shows up to
  `limit` (40) labelled items in no topic, with the live topics and a few member titles each, to
  ONE `OrganiseBatch` call (a `dspy.Predict`), and the host decides what the answer may do: only
  items it was shown, each once; only live topics; an existing topic only with a fit of "subject"
  or "part"; a new topic only for two or more items, at most `max_new_topics` (10), never under the
  name of a topic a person retired; and never a pair a person dismissed or took apart (the item's
  next usable placement in the answer may still apply). Items shown but not placed are left
  against the topic set they were shown (`left_items`) and come back only beside a new arrival or
  after the topics change: a lone item on a subject gets its topic when a second one comes, and an
  unchanged store asks nothing. A topic the round opened that ends up with fewer than two members
  (its other placements refused) is retired again by the pass itself, which leaves the name free for
  a later pass, and its placements are reported as refused. It runs up to `rounds` (3) and stops
  after a round that placed nothing.
  `recorder_for` is called with `"organize:round<N>"`, and `on_progress` receives
  `{"round", "ok", "error"}` after each round.
- **A host's say in each placement.** A host that keeps its own copy of where things are filed (a
  folder per topic, say) passes `on_place(item_id, topic_id, new_topic)`, called before each
  placement once any new topic is open; False declines it: nothing is placed, `refused` says it was
  declined by the host, and the item is left like any other, so it is not asked about at every
  pass. A hook that raises gave no answer, so that item is not recorded as left.
  `on_unplace(item_id, topic_id)` hears of a placement the pass has taken back (a new topic left
  with one member, or a write that failed after the host accepted) and `on_retire(topic_id)` of a
  topic it retires. `should_stop` is checked before each placement too, hooks or not, and what the
  plan had not reached is not recorded as left, since nothing was decided about it; nor is a
  placement taken back because its new topic's sibling got no decision.

### Similarity (`memory_vectors.py`, the `memory-vectors` extra)

Labels need a summary, and a summary is a model call. An embedding is the free route: a small
model on this machine gives every item a vector, so an item nobody has summarised can still be
linked to what it is like. Install `rlm-harness[memory-vectors]` (onnxruntime, tokenizers,
numpy).

```python
from rlm_harness.memory_vectors import DEFAULT_MODEL, MemoryVectors, download_model

download_model(DEFAULT_MODEL, "models/e5")                 # ~135 MB, once, when you call it
vectors = MemoryVectors(memory, "models/e5")
vectors.sync(limit=200)                                      # local, no model call
vectors.similar("m12")                                       # [("m40", 0.91), ...]
suggest_topics(memory, vectors=vectors)
```

- **The model.** `DEFAULT_MODEL` is `multilingual-e5-small` quantised to int8, pinned to one
  upstream revision with each file's size and SHA-256. It is this kit's choice, made for
  multilingual text; pass your own `EmbeddingModel` to use another, and to run it with your own
  code, any object with an `embed(texts) -> vectors` method as `embedder=` (which then needs
  `model=`, since a vector's model key and similarity threshold come from it). `download_model` runs
  only when you call it, fetches only `https` addresses, resumes a partial file, keeps a file
  already in place only if it still matches, and deletes a file that does not match its size and
  checksum instead of installing it. On a CPU it embeds an item in roughly 70 to 100 milliseconds.
- **What is embedded.** An item's current title, its summary and the first 1,500 characters of
  its text. `sync` embeds, newest first and at most `limit` (clamped to 500) per call, the live
  items with no vector or whose text or model changed since theirs was made, and drops the
  vectors of retired and purged items. Text that weighs under 40 (`MIN_TEXT`, with each Chinese,
  Japanese or Korean character counting 3) is recorded as skipped and not read again until it
  changes: a vector of almost nothing sits near everything. A Chinese note is compared from about
  14 characters, the length from which, measured, it starts to link to its own subject.
- **Where it lives.** `<store path>.vectors.db`, a separate file of plain float32 BLOBs compared
  with numpy. It is a derived cache that `sync` can rebuild, so it has no append-only triggers;
  its schema has its own numbered steps. No SQLite extension is involved, which is why a purge
  erases a purged item's vector with the standard library alone, whether or not a
  `MemoryVectors` is open.
- **What similar means.** Two items are similar only at or above the pair's threshold AND when
  each is among the other's four nearest. The default model holds a pair of mostly Chinese,
  Japanese or Korean texts (`is_cjk`) to 0.90 and every other pair, a Chinese and an English text
  included, to 0.86, because unrelated Chinese texts sit much closer together than unrelated
  English ones: measured over eight topics, unrelated Chinese pairs reached 0.892, unrelated
  English pairs 0.819. In a store of those texts in both languages, one 0.86 threshold drew 13
  wrong links beside 15 right ones; the pair rule drew 14 right and none wrong. A missing link
  still means nothing. `similar(item_id, among=...)` counts only those items, both as candidates
  and as the crowd each side's nearest are counted in; `similar_each(ids, among=...)` answers it
  for many items from one load of the vectors, which a per-item loop would read once per call;
  `similar_pairs(ids)` returns every such pair among `ids`, strongest first. `forget(ids)` erases
  vectors and keeps those items out until their text changes.
- **Suggestions.** `suggest_topics(memory, vectors=...)` also offers up to 30 of the newest items
  with no entities or tags the topic of the filed item they are most like. Similarity is the
  weaker signal, so such a suggestion scores 3 and sorts after any label match of the same score;
  it names the item it resembles (`similar_to`) and the cosine (`similarity`).

### Runs into memory (`consolidate_run`, `compute_memory_use`)

A run's REPL forgets everything when it ends. `consolidate_run` asks a model once what a finished
run should leave for a later run on a related task, and stores what the host can stand by.

```python
events = group_by_run(load_events("traces/run.jsonl"))[run_id]
report = await consolidate_run(memory, events,
                               evidence={"page": page_text},     # texts the host kept
                               trusted_tools=["fetch_page"])      # your tools that record result=
report.observed, report.inferred, report.downgraded              # ids, and why a fact fell back
```

- **What the model sees.** `render_run` gives the run's meta, the final output, the evidence, and
  each tool call with its arguments and the head of any recorded result, each source labelled
  `step:<n>` or `evidence:<key>` and marked trusted or not. Past 60,000 characters the tool calls
  are cut first; evidence too large for the budget on its own is cut as well. The model's own REPL
  turns are left out, and REPL output is never a source: the model wrote it. Up to 40 related items
  already in memory are shown too, found by searching for the run's output, so it does not propose
  them again.
- **The host verifies the bytes; the model chose them.** A `fact` proposal names one source and
  quotes it. It is stored as `observed` only when that source is trusted and its full text
  contains the quote (whitespace aside, at least 20 non-space characters); the stored text is
  then the source's own span, the origin `trace:<run_id>#<step>` or `evidence:<key>`, and `meta`
  says `selected_by: model`. Every other fact, and every `note`, is stored as `inferred` in the
  model's words, and a downgraded fact's reason is in the report and in its `meta`.
- **Which sources are trusted.** Evidence you pass is, unless you pass it as a `(text, trusted)`
  pair whose flag is not `True`. A recorded tool result is only when its tool is in `trusted_tools`,
  because a result may be another model's words (`model_as_tool` records one, and so may a
  consumer's own tools) and the kit cannot tell; a `model:` tool never is, and neither is a call
  that recorded `ok=False`. Name a tool only if its result can never contain text the model
  supplied: one that echoes its arguments ("no page at <url>") would let a model quote its own words
  back as a source. Most of the kit's tools record no result at all, so in practice `evidence` is
  how a run's sources reach `observed`.
- **Limits.** One call per run, at most `limit` (20) proposals kept, each at most 2,000
  characters, and a text an item not retired already holds is skipped. `recorder_for` is called
  with `"consolidate:<run_id>"` and every stored row carries its run id; `should_stop` is
  checked before the call and before each proposal.
- **What a run did with memory.** `compute_memory_use(events)` returns the ids a run read, the
  ids it wrote, and the ids it read but never named afterwards, in a later step's code or
  reasoning, a later tool call's arguments or the final output (on a word boundary, so `m1` is not
  inside `m12`). It reads the `memory_op` field the memory tools record, so renamed tools still
  count. An unreferenced read is an upper bound on waste, not a measurement of it: a model can
  use what an item says without writing its id.

### Removing data (`purge`)

`purge(item_ids, *, reason="", include_revisions=False, secure=False)` erases items for good. It is
the host's deliberate exception to append-only: no tool and no model-facing path reaches it, and it
exists because a person may want back out what they put in.

- It deletes each item, its labels, its recorded failures, its topic memberships, dismissals and
  leftovers, its search-index entries and its vector (an embedding can be partly turned back
  into its text), and writes a `purged` record with the time and `reason`, never the text
  (`purged(item_id)` reads it back).
- Deleting from the index only writes a tombstone; the words stay in its segments until they are
  merged. So every purge merges both indexes, which rewrites them and costs time in proportion to
  the store's size, holding the write lock throughout: other writers, a model's `memory_note`
  included, wait up to the 30-second busy timeout and past it get `Memory error`. Every connection
  the store opens runs with SQLite's `secure_delete` on, so space freed by any write is zeroed
  (extra writes on ordinary adds and labels too); without that, an ordinary index merge long before
  the purge leaves an old copy of the text in a live page, where no later purge can reach it.
- Ids are never reused, so an id in an old trace or citation names nothing rather than something
  else; `get` returns `None` and `memory_get` says `Not found`.
- `include_revisions=True` also purges every item that revises a purged one; otherwise a revision
  survives, with a `revises` naming an id that no longer exists.
- Aliases and the names alignment has considered are store-wide vocabulary, not tied to an item, so
  a purge leaves them; a name that only a purged item used stays in that vocabulary.
- `secure=True` also runs `VACUUM` to shrink the file. Last, the WAL is checkpointed. If another
  connection is inside a read transaction at that moment the checkpoint cannot finish and a warning
  is logged; the old pages then leave the main file and the WAL at the next full checkpoint, which
  SQLite runs when the last connection closes. `checkpoint()` runs one again without waiting and
  returns whether it finished, so a host can retry it after a purge. A `VACUUM` that cannot get the
  lock is logged too, and the purge, already committed, still returns its ids.

**Operational notes.**

- The store opens one connection per operation, so it is safe to share between the thread dspy
  calls a tool on and your own threads. Writes take `BEGIN IMMEDIATE` and queue rather than fail.
- The file runs in WAL mode, which needs shared memory. Keep it on a local disk; a network or
  VM-shared filesystem is unreliable for it.
- `":memory:"` is refused, because a per-operation connection would lose everything between calls.
- `max_item_chars` (default 200,000) caps one item, and a longer text raises rather than being cut.
- `max_index_chars` (default `None`, everything) caps how much of each item's text the search index
  reads: the item keeps its whole text, and a word past the cap is not found by `search`. It bounds
  the index of a store that keeps book-length texts.
- `index_origins` (default `False`) indexes each item's origin as well. It is off by default because
  the kit writes origins of its own (`model`, `run:<id>`, `trace:<run>#<step>`) and query words are
  ORed, so a search for "model" would return every note a model wrote; turn it on for a store whose
  origins are addresses and file names a reader searches by. Neither setting is stored in the file:
  each applies to items written from then on and to migration step 5's one reindex, and reopening
  with another value does not reindex.
- `find_meta(key, value)` returns the items whose top-level `meta[key]` equals `value` (a str only a
  JSON string, a number only a JSON number), oldest first and retired ones included, for a host that
  keeps its own key for an item in `meta` and must find it again (after a crash between adding an
  item and recording its id, for example). It scans every item's `meta`, so it suits occasional
  lookups, not a hot path.
- `search_ids(query, ...)` takes `search`'s query and filters and returns the ids `search` would, in
  its order, without loading any text. Its `limit` defaults to `None`, every match, for a host that
  bounds what it reads itself.
- A Python whose SQLite lacks FTS5, or is older than 3.43, raises `MemoryUnavailable` when the store
  is constructed, for a new file and an existing one alike. Every Python the kit's CI uses ships a
  newer one.
- The schema changes only by appending a step to `memory._MIGRATIONS`, counted in
  `PRAGMA user_version`. A file whose schema is newer than this kit knows is refused rather than
  opened. Step 5 rebuilds the item index with the origin column; it reindexes with the opening
  store's `max_index_chars` and `index_origins` (a marker in the step, since a step's own code sees only the
  connection).
- Timestamps (`valid_from`, `at`) must be finite numbers: SQLite stores NaN as NULL, which would
  read as "never retired".

**Not built yet.** No metric scores recall for `optimize.compile_task`: its metric sees an example
and a prediction, not a run's events, and a write-then-ask metric doubles the rollouts. One can be
built on `consolidate_run` and `compute_memory_use`; the kit will ship one once a consumer's real
metric exists to generalise from.

## Environment interpreter (`interpreter="container"`)

The default `pyodide`/`deno` interpreter is WASM Python: it **cannot spawn a subprocess** (emscripten
has no processes). When a task needs the model to run real commands as part of its own reasoning: a
build, a test, `git`, a compiler: set `RLM_INTERPRETER=container` (or `RLMConfig(interpreter="container")`).
The RLM's REPL then runs **inside an isolated Docker container**, so the model's own Python can
`subprocess.run(...)` natively and hold real filesystem/process state.

```bash
docker pull python:3.11-slim
export RLM_INTERPRETER=container         # default stays pyodide; this is opt-in per run
```

- **One persistent container per run.** State persists across REPL turns within a run (the model can
  write a file in one cell and run it in the next), and the container is torn down at run end. This is
  the *environment* model: distinct from the [`run_command`](#running-local-commands-an-isolated-runner)
  tool, which is a model-*chosen* command per call against a runner you supply (and whose reference
  example is a fresh container per call). Use the container interpreter when the REPL itself needs a
  real environment; use `run_command` when commands are occasional, tool-like actions.
- **A stronger boundary than pyodide for this case, not a weaker one.** `--network=none` makes the
  host↔container stdio broker the only channel in or out (no egress); the LM credentials never enter
  the container: `llm_query` and tool callbacks execute **host-side**, only results cross the pipe;
  Linux capabilities are dropped (`--cap-drop=ALL`) and memory/pid are capped. It is the *opposite* of
  the refused `local` interpreter (host RCE), not a relaxation of it.
- **Config** (`RLM_CONTAINER_*`, all optional): `IMAGE` (default `python:3.11-slim`), `TIMEOUT`
  (per-cell sandbox-compute budget, s, default 120. Host tool time is not counted), `MEMORY`
  (`512m`), `PIDS_LIMIT` (`256`), `NETWORK` (`none`), `CPUS` (unset = uncapped), `CAP_DROP` (`true`),
  `READ_ONLY` (`false`; opt-in read-only rootfs for an inspect-only task, paired with a tmpfs `/tmp`),
  `WORKDIR` (a host dir mounted **read-only** at `/workspace`).
- **Needs the `docker` CLI** (checked at start; `import rlm_harness` stays docker-free). The `WORKDIR`
  mount resolves on the *daemon's* filesystem, so it won't work with a remote `DOCKER_HOST`.

## Sandbox turn timeout + cancellation (`pyodide`/`deno`)

The default `pyodide`/`deno` interpreter blocks on a plain subprocess pipe read with **no timeout
anywhere in dspy's own code**: a wedged Deno subprocess, or a model-written REPL cell that spins
forever, hangs the run with no recourse short of killing the whole process. `asyncio.Task.cancel()`
cannot help: the blocking call has no `await` inside it, so the event loop never gets a chance to
run cancellation machinery. rlm-harness closes this with the SAME timer-armed-before-blocking-read,
kill-to-unblock idiom the container interpreter already uses for its own `TIMEOUT` (above), ported
to `pyodide`/`deno`:

- **`RLM_SANDBOX_TURN_TIMEOUT`** (seconds; `RLMConfig.sandbox_turn_timeout_s`): a per-`execute()`
  safety-net deadline. **Unset by default** (deliberately NOT matching the container interpreter's
  own `120.0`-default precedent: this budget has no hook to exclude host-side tool/sub-LM dispatch
  time, so a generous always-on value would misfire on legitimate multi-tool-call turns more often
  than the container analogy implies). Firing raises dspy's own RECOVERABLE interpreter error:
  `CodeExecutionError` on dspy >= 3.3.0, resolved at call time through
  `_dspy_compat.recoverable_interpreter_error()` rather than hardcoded, because 3.3.0 INVERTED the
  split: the base `CodeInterpreterError` used to be the recoverable one and is now the TERMINAL
  one. The model sees an `"[Error] ..."` string and gets to retry next turn against a
  freshly-respawned sandbox.
- **`RLM_REQUEST_TIMEOUT`** (seconds; `RLMConfig.request_timeout_s`): a wall-clock cap on ONE
  model HTTP request ATTEMPT, handed to `dspy.LM(timeout=...)` and from there to litellm. This is
  `RLM_SANDBOX_TURN_TIMEOUT`'s sibling on the other side of a turn: the sandbox side was bounded
  and the model side was not settable at all.

  **Unset is not "no cap".** With nothing passed the provider SDK applies its own. Measured on
  dspy 3.4.0 with openai 2.41.0, `DEFAULT_TIMEOUT` is
  `Timeout(connect=5.0, read=600, write=600, pool=600)`, so the effective default is **600 s** per
  attempt: this setting *replaces* that number rather than introducing a bound where there was none,
  and a consumer whose turns legitimately run longer than ten minutes must set it **up**, not leave
  it alone. (The 600 used to be attributed to litellm's `COMPLETION_HTTP_FALLBACK_SECONDS`, which is
  genuinely 600.0 but is not on dspy 3.4.0's default path: `select_backend` returns `native=True`
  for every LM shape this kit builds, so the vendored `lm15` engine handles the call. Same number,
  different owner, which is why checking the figure never surfaced the mistake.)

  **It does not bound a run to its own value.** dspy's `LM` passes `num_retries=3` and the OpenAI
  SDK's own `DEFAULT_MAX_RETRIES` is 2, so a dead endpoint is retried: the run-level wait is a
  MULTIPLE of this value plus backoff. Size a caller-side budget on the multiple.

  What prompted it, and an honest reading of it: against a self-hosted OpenAI-compatible endpoint
  one request never came back: the socket stayed `ESTABLISHED` with both queues empty and the
  worker slept in `epoll_wait` for 38 minutes at 0.3% CPU, while that same endpoint answered
  unrelated requests in half a second. 600 s across four attempts is about forty minutes, so that
  observation fits "litellm's own default, retried" at least as well as "nothing was watching";
  no attempt counter was captured at the time. Either way the consumer could not choose the
  number, and now it can.
- **`cancel_event`** (`RLMTask(cancel_event=a_threading.Event)`): for a caller that wants to stop
  an in-flight run NOW (e.g. a "Cancel" button in a UI driving `arun()` from a worker thread). Set
  the event from another thread; the current sandbox turn is killed and `SandboxCancelled` (exported
  from `rlm_harness`) propagates all the way up through `arun()` as a genuine, NON-recoverable run-ending
  failure: never retried (see `run_with_retry`'s `non_retryable` below), never caught by dspy's own
  `except (CodeExecutionError, SyntaxError)` (the class it catches is version-dependent: see the
  turn-timeout bullet above; `SandboxCancelled` stands outside dspy's hierarchy entirely, which is
  what makes it non-recoverable on EVERY version).

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
retry engine untouched. A caller-driven cancellation must never be silently absorbed by a retry
that respawns the sandbox and restarts the whole trajectory from scratch.

## Two dspy defaults the kit turns off

`configure()` sets `dspy.configure(disable_history=True)` and builds its LMs with `cache=False`.
Both are properties of the RLM WORKLOAD rather than of any one consumer, which is why they belong
here rather than in each project: a whole corpus as the input, retries that re-send identical
inputs, and a process that outlives many runs.

**History** (`RLMConfig.disable_history`, `RLM_DISABLE_HISTORY`, default on). dspy's `record_history`
appends every call's full entry, messages and response and outputs, to `GLOBAL_HISTORY` and
`lm.history` up to `MAX_HISTORY_SIZE = 10_000`. In an RLM each entry holds a whole turn's prompt plus
a reasoning-model reply, so the list grows with corpus size times turns. One consumer running a
repository's tasks in a single process measured 2,413 calls carrying 121.4M prompt and 12.6M
completion tokens, with the container's cgroup recording `oom_kill 2` and the process dying 17s into
run 81. The kit never reads that history, and turning it off does not change the trace: measured,
`run_end.payload` carries the same keys either way, because usage comes from dspy's separate tracker.

**Cache** (`RLMConfig.lm_cache`, `RLM_LM_CACHE`, default off). `dspy.LM` defaults `cache=True`.
**Every recovery an RLM consumer has is a retry with identical inputs**, so a live cache turns each
one into a replay of the failure it exists to escape. Two consumers found this independently, each
after a "fresh attempt" completed in seconds having made no model call.

**Why the per-LM flag and not `dspy.configure_cache`.** On dspy 3.4.0 the per-LM flag GATES access
to the global store: `clients/execution.py` computes `use_cache` as false unless the LM is `managed`,
so switching either one off prevents the replay and the per-LM one is the narrower change. It reaches
only the LMs `configure()` BUILDS. An injected `main_lm`/`sub_lm` is used verbatim, as every other
injected-LM rule here says, so pass `dspy.LM(…, cache=False)` yourself if you build one.
`ClaudeAgentLM` needs nothing either, though for a different reason since 1.17.0: it is engine-backed
and therefore MANAGED, so dspy would consult a cache for it, and its own constructor defaults
`cache=False` to keep a subscription call touching none. Pass `cache=True` if you want one.

**Opting back in, and the one thing to know first.** `RLM_DISABLE_HISTORY=0` restores dspy's history
and `RLM_LM_CACHE=1` restores the cache. The history one is a PROCESS-wide dspy setting, so a
consumer that calls `configure()` and then uses `dspy.inspect_history()` for its own debugging gets
nothing back until it opts in. A per-role passthrough still wins over the cache default, so
`RLM_MAIN_LM_KWARGS='{"cache": true}'` keeps dspy's behaviour for the planner only.

**What ran is in the trace.** `run_end.payload.dspy_settings` carries
`{"disable_history": bool, "cache": {"main": bool, "sub": bool}}`, read back off `dspy.settings` and
off the LMs rather than echoed from config, because an injected LM keeps its own `cache` and a
non-kit caller that configured dspy first keeps its own `disable_history`. Additive: a new optional
payload field.

## Sandbox sub-agents (`RLMConfig.sandbox_sub_agents`, `RLM_SANDBOX_SUB_AGENTS`, default on)

From dspy 3.4.1, `dspy.RLM` installs a `dspy` shim inside the sandbox and appends a paragraph to
every action prompt telling the model it may use it. Model code can then write `import dspy` and
build `Predict`, `ChainOfThought`, `ReAct`, `ReActV2`, `CodeAct`, `ProgramOfThought` or a nested
`RLM`. Those run on the HOST: they call this run's `sub_lm`, they charge `max_llm_calls`, and they
can be handed the task's own tools BY NAME (never a tool that was not injected, and never a
function authored inside the sandbox).

The default is on because that is what the installed dspy does; turning it off is the kit
overriding dspy, not a kit default. Two things to know before leaving it on:

- **Who drives your tools changes.** The answer stops being "the root LM" and becomes "the root LM,
  plus any sub-agent its code builds". If the sandbox is your security boundary, that is the
  sentence to re-read.
- **A nested `RLM` is invisible in the trace.** Measured: one `sub_call` carrying only the final
  answer, no `main_step` for the inner turns, no child run id. When a child's trajectory matters,
  `make_harness_tool` is the seam that keeps it.

`RLMConfig(sandbox_sub_agents=False)` refuses the shim install. The refusal goes through dspy's own
degradation path, so dspy drops its two host tools, logs that sub-agents are unavailable, and leaves
the sub-agent paragraph OUT of the prompt: the model is never offered a capability it does not have.
It applies to every interpreter kind the kit builds, and has no effect on an interpreter you supply
yourself via `RLMTask(interpreter=...)`, which bypasses `build_interpreter` entirely.

A turn-timeout restart replays the shim install, because a fresh sandbox would otherwise lose it
while the prompt still offered it.

## Subscription logins: this kit's adapter, and dspy's own providers

`ClaudeAgentLM` runs a Claude Pro/Max subscription through the official **Claude Agent SDK**, which
spawns the `claude` CLI per call. Anthropic's support article covers that path explicitly: "You can
still use the Claude Agent SDK, `claude -p`, and third-party apps with your subscription limits"
([support.claude.com, "Use the Claude Agent SDK with your Claude plan"][claude-plan-sdk], read
2026-10-14; the same page's 2026-06-15 update says the previously-announced changes to Agent SDK
usage are PAUSED, so read it for the current state rather than trusting this quote's date).

One kwarg is a cost multiplier here. `n=k` asks dspy for k candidates, dspy runs one engine call
per candidate, and each of those is a separate `claude` CLI spawn charged to your own usage window:
measured, `n=3` was one spawn until 1.16.0 and is three since 1.17.0. The constructor warns when
`n` is above 1. `temperature`, `top_p` and `max_tokens` are tolerated and ignored as the docstring
says, because the SDK exposes no sampling controls.

### Spending your own credits without meaning to

A subscription run can start costing money mid-flight, and nothing in the call path asks first. If
your account has usage credits enabled, requests past the plan's included limit switch to
pay-as-you-go at API rates and keep going. The switch is silent: no prompt, no error, no different
return value. Read [Anthropic's own page on usage and costs][claude-costs] for the account side,
because that is where the setting lives and this kit cannot see it.

**What actually bounds the spend, in the order that matters.**

- **The account setting**, which this kit can neither read nor change. Whether usage credits are
  on, and whether a monthly cap is set, lives on the account, and the page above is where to look.
  No kit-side control beats it.
- **`ClaudeAgentOptions.max_budget_usd`**, which the SDK enforces: the query stops and returns an
  `error_max_budget_usd` result. This kit never passes it, so it is not protecting you today. Build
  your own options and you can set it.
- **Fewer calls.** `n=k` is k separate CLI spawns charged to your window, as the paragraph above
  says, and a long trajectory is a long bill.

**`ANTHROPIC_API_KEY` bills the key, not the subscription.** The CLI silently prefers the key when
it is set, so a login you believed was running on your plan quietly runs on API billing instead.
`ClaudeAgentLM` refuses to start when that variable is present, rather than warning, and
`allow_api_key=True` is how you say you meant it.

**Reading it back off a trace.** Since 1.22.0 a run records the rate-limit reports the SDK makes,
under `run_end.payload.rate_limits`, so a finished run can answer whether it was being charged past
the included limit:

```python
from rlm_harness import load_events

events = load_events("traces/run.jsonl", run_id="run-1")
end = [e for e in events if e["type"] == "run_end"][0]
for report in end["payload"].get("rate_limits", []):
    print(report["status"], report.get("rate_limit_type"), report.get("overage_status"))
```

A report carries `status` (`allowed` / `allowed_warning` / `rejected`), `rate_limit_type` (the one
to look for is `overage`), `utilization`, `resets_at`, and the `overage_*` fields when the SDK sets
them. An absent `rate_limits` key means the SDK reported nothing, which is the ordinary case: it is
not a promise that nothing was charged. The list is capped per run, with the overflow counted under
`rate_limits_dropped`, so a long run cannot grow its own trace without bound.

**What this is not.** It records; it does not stop anything. Nothing here refuses a call, raises, or
ends a run early, and the report arrives while the call that triggered it is already in flight, so
even a guard built on it later would bound the NEXT call rather than that one. A stop condition also
needs the real event sequence, which nobody has observed yet: whether the switch arrives as
`status="allowed"` with `rate_limit_type="overage"`, whether an `allowed_warning` precedes it, and
whether it repeats per call. Recording them is what makes that observable. Until then, the account
setting is the control.


dspy ships its own subscription providers, and the kit needs no code for them: `configure()` takes a
pre-built LM verbatim, so they go in through the same seam as any other.

```python
from dspy.clients.engines import AsyncLM15Engine, LM15Engine

lm = dspy.LM("claude-code:sonnet", engine=LM15Engine(), async_engine=AsyncLM15Engine(), cache=False)
configure(RLMConfig.from_env(), main_lm=lm, sub_lm=lm)
```

The model string takes `provider:model` with a COLON; a slash is read as a litellm prefix and
refused. Registered providers include `claude-code:` (the local `claude` CLI login),
`openai-codex:` (the local `codex` CLI login) and `xai:` (which lm15 logs in itself, by device
code; its own module docstring gives the reason as xAI shipping no CLI credential file
convention, which is lm15's claim rather than one checked here). Verified from here: all three construct and route,
and the kit's own readers (`applied_lm_budget`, `lm_cache_is_live`, `copy_lm`) work on the result.

**`claude-code:` does not work on macOS with a Keychain login.** Two halves, settled differently.

The lm15 half is executable from here: it reads the credential only from
`~/.claude/.credentials.json` (`dspy/_vendor/lm15/auth.py`) and has no Keychain path. Settle it with
`grep -rin 'keychain\|find-generic-password\|SecItem' "$(python -c 'import dspy,os;print(os.path.dirname(dspy.__file__))')"`,
which returns nothing.

Where the credential actually is, is two commands away on any macOS machine, and on the machine
this was written on they answered: `ls ~/.claude/.credentials.json` → absent, and
`security find-generic-password -s "Claude Code-credentials"` → present. So lm15 looks in the one
place the credential is not.

The rest is Claude Code's own documented behaviour and not settleable from here: its
authentication docs say macOS credentials go to the Keychain and that it falls back to that file
when the Keychain refuses the write ([code.claude.com/docs/en/authentication][cc-auth]), and an
open issue asks for an opt-in file store, reporting that no setting forces one while the Keychain
works ([anthropics/claude-code#96310][cc-96310]). Copying the Keychain entry into the file is a
community workaround this guide does not recommend: it is undocumented, the CLI can overwrite it,
and a `CLAUDE_CODE_OAUTH_TOKEN` variant is reported to delete the Keychain credential on exit
([anthropics/claude-code#37512][cc-37512]). On macOS, `ClaudeAgentLM` is the path that works,
because the CLI reads the Keychain itself.

[claude-plan-sdk]: https://support.claude.com/en/articles/15036540-use-the-claude-agent-sdk-with-your-claude-plan
[claude-costs]: https://code.claude.com/docs/en/costs
[cc-auth]: https://code.claude.com/docs/en/authentication
[cc-96310]: https://github.com/anthropics/claude-code/issues/96310
[cc-37512]: https://github.com/anthropics/claude-code/issues/37512

`openai-codex:` and `xai:` are capability this kit has no adapter for, and they read files rather
than the Keychain, so they are expected to work where `claude-code:` does not. That expectation is
NOT verified from this repo, for either of them. Settle the codex one with:

```bash
codex login   # then:
python -c "import dspy; from dspy.clients.engines import AsyncLM15Engine, LM15Engine; \
print(dspy.LM('openai-codex:gpt-5', engine=LM15Engine(), async_engine=AsyncLM15Engine(), cache=False)('say OK'))"
```

A failure naming a credential file means routing worked and only the login is missing. `xai:` has
no equivalent one-liner here because its device-code flow is interactive.

## Timeouts: what bounds what

Six different things can end a stuck run, on three different clocks, and no single one of them
bounds a run's wall time. This is the whole map; reach for it before adding a seventh.

| Knob | Env | Default | Bounds | On expiry |
|---|---|---|---|---|
| `RLMConfig.request_timeout_s` | `RLM_REQUEST_TIMEOUT` | unset → the provider SDK's own 600s | **one HTTP request attempt** to a litellm-backed LM | `LMTimeoutError`, which dspy classifies as RETRYABLE, so it is retried, not fatal |
| `ClaudeAgentLM(timeout_s=…)` |: (constructor only) | 600s | **one whole call**, INCLUDING time queued behind the SDK's concurrency semaphore | `TimeoutError` |
| `RLMConfig.sandbox_turn_timeout_s` | `RLM_SANDBOX_TURN_TIMEOUT` | unset (disabled) | one whole `execute()` (`pyodide`/`deno`) (host-side tool and sub-LM dispatch time INCLUDED, which is why it is off by default | dspy's **recoverable** interpreter error) the model gets another turn |
| `ContainerConfig.timeout_s` | `RLM_CONTAINER_TIMEOUT` | 120s | one `execute()`'s sandbox compute (`container`; host tool time excluded) | recoverable, same as above |
| `RLMTask(cancel_event=…)` |, | unset | anything the caller decides | `SandboxCancelled`, **never** retried, never wrapped |
| `RLMConfig.max_retries` | `RLM_MAX_RETRIES` | 1 (no retry) | the whole trajectory, in attempts not seconds | `RLMTaskError` |

Three things about this table are easy to get wrong.

**`request_timeout_s` does not bound a run to its own value.** An attempt is not a request: dspy's
`LM` passes `num_retries=3` and the OpenAI SDK's own `DEFAULT_MAX_RETRIES` is 2, so a dead endpoint
is retried and the run-level wait is a MULTIPLE of this number plus backoff. Size a caller-side
budget on the multiple. Leaving it unset is not "no cap" either: the provider SDK then applies
its own `COMPLETION_HTTP_FALLBACK_SECONDS` of 600s, so this field REPLACES that number rather than
introducing a bound where none existed. A consumer whose turns legitimately run longer must set it
UP, not leave it alone.

**The two 600s defaults are not the same quantity.** `request_timeout_s` bounds one HTTP request
with retries around it; `ClaudeAgentLM`'s bounds one whole call including queueing, with nothing
around it. Which one you get depends on the model string, and `request_timeout_s` deliberately does
NOT drive the subscription route: under `llm_query_batched`'s thread fan-out, a value that is
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
  x ~3           the OpenAI SDK's own DEFAULT_MAX_RETRIES is 2
one turn       ~20min + backoff    for ONE wedged model call
  x 30 turns   ~10 hours           the iteration cap is the only thing that ends the run
```

The two retry layers are counts of RETRIES, not attempts, so read those multipliers as lower
bounds. The point is the shape, not the digits: a "2-minute request timeout" is a wall-clock
worst case measured in hours, and lowering `max_iterations` moves that number far more than
lowering the timeout does. `max_retries` multiplies it again, which
is why it defaults to 1. If you need a real wall-clock bound on a run, put it in your own
driver (`asyncio.wait_for` around `arun`, or a `cancel_event`); no knob here is one.

> These semantics assume non-streaming model requests, which is what the kit issues today. A
> streamed request changes what a per-request HTTP timeout means: an inter-chunk read timeout does
> not bound a whole generation, so anything that turns streaming on owes this page a revision and
> a deadline of its own.

## `run_in_subprocess`: a safe, isolated-subprocess primitive (`isolation.py`)

A small PRIMITIVE only: safely run one picklable callable in an isolated OS process, get its
result or a clear error back, bounded by a timeout. Task-scheduling: how a web server actually
queues many of these (Celery, RQ, a plain thread/process pool). Is explicitly the consumer's own
concern, not shipped here, matching this kit's standing base/wrap posture (`make_command_tool`
ships no executor, `make_git_clone_tool` ships no cloner) applied to "run a whole task."

```python
import functools
from rlm_harness import run_in_subprocess

def run_my_task(**inputs):          # a plain, MODULE-LEVEL function -- see "must be picklable" below
    return MyTask().run(**inputs)

result = run_in_subprocess(functools.partial(run_my_task, q="…"), timeout_s=30.0)
```

**A genuinely different gap than three things in this kit that already sound similar**: worth
naming so this doesn't look redundant with them: `interpreter="container"` (above) isolates the
RLM's own REPL SANDBOX in a Docker container, but the ROOT process still runs the RLM's own
orchestration (LM calls, retries, tool dispatch) directly: a hang/crash there isn't covered.
`rlm_harness.tools.run_isolated` bridges an async coroutine into a sync call site on a dedicated
THREAD: same process, no OS-level isolation, solves an event-loop-nesting problem, not a
fault-isolation one. `cancel_event` (above) stops an IN-FLIGHT run the calling code already owns
and is watching: it doesn't hand a whole task off to a separate process in the first place. The
actual gap: a web-facing consumer whose request handler wants to run ONE task without that run's
own crash, hang, or resource usage taking down the request-handling process itself needs to run
it in a SEPARATE OS PROCESS.

- **Lives at the top level, not `rlm_harness.tools`**: that package's own module docstring
  scopes it as "tools RLM tasks can expose to the model inside the REPL"; nothing here is ever
  placed in a `tools=[...]` list or invoked by the model.
- **`factory` MUST be picklable**: a real, easy-to-get-wrong gotcha. A local closure or
  `lambda` is NOT picklable across the `"spawn"` boundary this uses; `functools.partial(
  module_level_function, **kwargs)` IS, but only if every value bound into `args`/`kwargs` is
  ALSO picklable, not merely the function reference: a live socket, open file handle, DB
  connection, or lock bound as one of the `partial`'s own arguments hits the same class of
  pickling failure the "avoid closures" advice was supposed to prevent.
- **Uses `multiprocessing.get_context("spawn")`, never `"fork"`**: the calling process is very
  plausibly a web server already running an event loop / thread pool / open file descriptors / a
  live LM client; forking that risks a corrupted child (inherited locks held by a thread that
  doesn't exist in the child, half-open sockets). `spawn` is itself still fork()+exec() on POSIX,
  but exec runs BEFORE any user code executes in the freshly-forked child, which is what actually
  avoids the corruption class a bare `fork` risks. `spawn` also requires `factory`'s entire import
  chain to be safely re-importable in the fresh interpreter: if `factory` is defined in a script
  run as `__main__`, guard it with `if __name__ == "__main__":`.
- **A real bug found and fixed during design review**: `multiprocessing.Queue.put()` does not
  pickle synchronously: a background feeder thread does, and a pickling failure there is logged
  and silently DROPPED, never raised back to `put()`'s caller, which would leave the parent
  hanging on `get()` forever. Fixed by having the child explicitly test-pickle its payload
  synchronously, in its own code, BEFORE ever calling `put()`: falling back to a plain-string
  `RuntimeError` only if that test-pickle itself fails. The parent's own `queue.get()` also
  carries its own bounded timeout, independent of the process-level timeout below, as a backstop
  against an out-of-band kill (e.g. the host OOM-killing the child) that bypasses this
  primitive's own signal-based escalation entirely.
- **`timeout_s`** (default `None` = no limit): on expiry, `process.terminate()` (SIGTERM), then a
  `grace_period_s` (default `5.0`) wait, then `process.kill()` (SIGKILL) if still alive, followed
  by a final reap so no zombie is left. Raises `TimeoutError`. SIGTERM does NOT reliably let the
  child's `finally`/`atexit` code run: that's only true if the child itself installs a
  `signal.signal(SIGTERM, ...)` handler; with none (the default), the OS's default disposition
  terminates it immediately.
- **`max_memory_mb`/`cpu_time_limit_s`** (default `None` = no cap): opt-in, POSIX-only,
  best-effort resource caps applied inside the child via `resource.setrlimit` before `factory()`
  runs: silent no-ops on a platform without `resource` (Windows). `cpu_time_limit_s`
  (`RLIMIT_CPU`) is confirmed to enforce correctly on every POSIX platform tested, including
  macOS. `max_memory_mb` (`RLIMIT_AS`) bounds VIRTUAL address space, not resident/physical
  memory, and on macOS specifically, the kernel refuses to lower `RLIMIT_AS` from unlimited AT
  ALL (empirically confirmed: every attempted value, from 50 MB to 1000 MB, failed identically
  with `ValueError: current limit exceeds maximum limit`), so this parameter is effectively
  Linux-only in practice today. Either way it fails LOUDLY: relayed as a clear exception through
  the same test-pickle-then-relay path as any other error: never silently leaving the cap
  unenforced. **A second edge case, confirmed on real Linux CI, not just reasoned about**: an
  aggressively low `max_memory_mb` can starve the relay mechanism itself: the child correctly
  hits `MemoryError`, but by then is so memory-constrained that `multiprocessing.Queue.put()`'s
  own internal feeder thread fails to even start (`RuntimeError: can't start new thread`),
  crashing the child before anything can be relayed. No fallback is possible for this: a
  resource-exhausted process can't reliably report its own exhaustion through a mechanism
  (spawning a thread) that itself needs spare resources, so it degrades to the exact same
  safety net an external kill would (the parent's own bounded `queue.get()` times out and raises
  a generic "child exited without delivering a result" error). This is the correct, accepted
  outcome for this scenario, not a bug. The cap was still genuinely enforced.
- **No process pooling/reuse**: a fresh `multiprocessing.Process` per call, matching
  `run_isolated`'s own "one thread per call, by design: simplest correct answer" precedent,
  transplanted to the process level. A consumer wanting a persistent worker pool builds one using
  this primitive as the per-task unit.

## Grounded completeness: the sufficiency-critic recipe

A convention, not an API. When the RLM generates an artifact that must MATCH a retrieved
ground-truth (a spec, a contract, a source document), "am I done?" is the dangerous judgment:
a model asked to self-assess from memory will call a half-right artifact complete. There is
often no deterministic check for CONTENT correctness: a validator catches *structure/format*,
but not "this request is missing a required header" or "this answer skipped a clause".

The fix (the agentic-RAG *sufficient-context* pattern) is to GROUND the completeness judgment in
the retrieved source instead of the model's recall:

1. **Hold the ground-truth in REPL state.** Fetch the source once (a `fetch_url` tool, a skill)
   and keep it as a REPL variable: rlm-harness's interpreter persists variables across turns, so the
   ground-truth stays addressable without re-fetching or re-pasting.
2. **Diff the artifact against it, itemized.** Each turn, compare the generated artifact to the
   held ground-truth field-by-field and emit the SPECIFIC gaps ("missing header X, body field Y"),
   not a yes/no verdict.
3. **Regenerate on the gaps; finalize only when the diff is clean** (or the gap was escalated to a
   sub-LM and confirmed unobtainable). The itemized gap-list is a far stronger regeneration signal
   than a generic "make it complete".

This lives in the consumer's task INSTRUCTIONS (it is an LM-decided REPL action, kept in the
trajectory as honest RL data: same reasoning as keeping tools/skills LM-decided), and it needs no
new model: the main LM critiques cheaply against its own REPL state, reserving a sub-LM escalation
for a genuine knowledge gap. A consumer uses it so the planner stops finalizing a generated artifact
whose content only *looks* right: diffing it against the retrieved source held in the REPL.

### Line numbers and `verify_quote`: feed them different text

`make_read_file_tool(line_numbers=True)` and `verify_quote` **must not see the same string**, and
getting that wrong costs real accuracy in both directions:

    line_numbers=True  ->  the MODEL, so it can cite a coordinate without counting lines
    the raw file text  ->  verify_quote, NOT the tool's rendered output

The rendered output prefixes each line with `f"{n:>6}\t"`, and so does `edit_file`'s success
snippet, which is a second source of guttered text and reaches the model on every edit. A model
quoting from either naturally carries that gutter into its quote.

**Until 1.9.0 that cost the citation.** The quote is not in the file. The gutter is not file
content, so `verify_quote` refused a perfectly correct citation, and a consumer who turned line
numbers on to fix coordinates started failing verification while one who turned them off made the
model count lines itself. Since 1.9.0 the gutter is READ rather than searched: 83.16% of those
citations now resolve to the line they name, and the rest keep the old verdict rather than getting a
wrong one. The rule and its limits are under "`verify_quote` reads a gutter as a COORDINATE CLAIM"
below.

**The second failure is the expensive one and it has been measured**: on a live deployment with line
numbers OFF, **59 of 411 stored citations (14.4%) quoted the text verbatim at the wrong line**, and
without a host-side coordinate correction the unverified rate would have been 21.9% instead of 7.5%.
Every artifact in that population had at least two.

The same consumer then turned line numbers back on and re-ran the task: coordinate corrections went
**21.4% to 0.0%** (15 of 70 citations, then 0 of 39). Zero corrections at the prior rate is
P = 8.2e-05, so it is not sample size. It is two independent generations rather than a paired
re-run: the model planned a different outline the second time, 11 artifacts against 6, so the
citation counts are not comparable and only the proportion is.

**A second consumer sized this failure on its own corpora, with a different denominator.**
Re-running its current verifier over four artifact sets built before line numbers were on corrected
**96 of 556 coordinates (17.3%)**, reaching **35.6%** on the worst of the four. Read that as a size,
not a rate: one deployment, one model family, re-verified after the fact rather than observed live.

**Do not reconcile it with the 14.4% above, or average them.** They count different things over
related corpora: 14.4% is stored CITATIONS observed to quote the right text at the wrong line, live;
17.3% is COORDINATES a later verifier corrected, after the fact. That consumer holds at least one
further cut on the same family of artifacts: coordinate errors as a share of the UNVERIFIED
citations, which is a share of FAILURES and lands near 60%, and none of the three is the working
for another. The spread is what to carry: a factor of two across four corpora from one system, so a
single measurement does not bound the next, and neither of these bounds the other.

The fix is not a kit setting, it is which string goes where. Your verifier holds the source anyway.
it has to, to verify, so pass it the raw text and let the tool's rendered form exist only for the
model.

**A BLANK line renders as nothing but its gutter**, and that used to defeat the guard entirely.
`"     7\t"` strips to `"7"`: non-empty, so it passed the empty-quote check and then matched any
source containing that digit. A citation of nothing verified, reporting a line the citation never
claimed. Since 1.8.2 a quote whose every non-blank line is a bare number is refused with its own
message, which also covers the shapes a model actually emits: the trailing tab trimmed (`"     2"`),
a space written for it (`"     2 "`), and the render's real trailing newline (`"     2\t\n"`).

Two things to know about that guard. It refuses any all-digit quote, so a citation of a genuine
number (`"8080"`, or a column from a numeric file) is refused as well; cite the line's text, not
its coordinate. And it closes the fully-blank case only. A guttered quote that carries content is
not refused, and the gutter NUMBER is then searched as literal content, so it matches wherever that
number happens to precede the line's text, across a mandatory `\s+` and therefore across blank
lines. A full line of code is reachable that way, not just a line of punctuation: a quote claiming
line 42 of this repo's `tests/test_async.py` verifies at line 39, because line 39 ends `== 42`.
Rendering every non-blank line of every `.py` here and verifying it against its own file leaves 2
such false matches in 18,761 quotes. **1.9.0 resolves both**, by the mechanism below.

**`verify_quote` reads a gutter as a COORDINATE CLAIM, and never strips it.** Stripping the
leading `spaces + digits + tab` and searching the remainder repairs most guttered quotes and
accepts a citation naming the WRONG line whenever that remainder appears anywhere else: an
invented claim passing verification, which is why it was refused for four releases.

Using the gutter is stronger than discarding it: the content must sit at exactly the line it names,
and that block must occur exactly once in the source. Both halves are load-bearing. Without
uniqueness a bare position check is WORSE than searching: 16.84% of non-blank lines in this repo
recur in their own file, so a fabricated coordinate verifies at roughly 0.15% against 0.000% for a
plain search. An unrelated Python tree measures 16.39% of 52,941, so budget for it as a property of
the language rather than of this repo. Without exactness a fabricated INDENTATION level verifies, since 2.16% of lines here
are exact-unique but identical after stripping, and indentation is semantics in Python.

With both, fabrication is closed by construction rather than by rate: exact content at line `n` plus
a block occurring once means `n` is the only line that can hold it. What remains is honest citations
that stay uncoordinated: 16.84% of non-blank lines, or 32.8% of everything `read_file` renders once
blank-line citations are counted. Those keep exactly today's verdict, and that class contains no
wrong-line matches to inherit.

A coordinate-verified match says so in its text, because it means something different from an
ordinary one: the source holds the CONTENT at that line, but the quoted bytes, gutter included.
are not a substring of it. Branch on that if you re-derive grounding host-side with `quote in
source`. `normalize_whitespace=False` skips the path entirely: byte-exact mode means no
interpretation, and reading a gutter as a coordinate is an interpretation. That is also the lever
if `source` is itself a numbered listing, where a correct literal match would otherwise be
overridden: a shape found in none of the tens of thousands of text files scanned locally, though
`cat -n` emits it.

### `verify_quote`: the deterministic half of step 2's diff

Step 2 above ("diff the artifact against it, itemized") is entirely model-judged: nothing backs
the claim that a specific quote/citation actually appears in the held source. `verify_quote(source,
quote)` (`rlm_harness.tools`) is the deterministic primitive for exactly that one checkable piece:
it returns a parseable `"MATCH: ..."` (with a line number, plus a character offset and context
snippet on the literal path; a coordinate-verified match has no offset to give) or
`"MISMATCH: ..."` (with a bounded "closest line" hint when `quote` is single-line), never a
self-graded guess. A single plain function, no factory, no `name=`, no trace call (it binds to nothing at
construction time and touches no filesystem/network, matching `make_schema_validator`'s own
precedent). Matching is whitespace-flexible by default (`normalize_whitespace=True`) and needs no
`regex` package: `quote` is literal text, every character either escaped or collapsed to a flat
`\s+`/`\s*`, so the built pattern can never exhibit catastrophic backtracking the way an
LM-controlled `grep_files` pattern could. Normalization is junction-aware: whitespace between two
word characters must be present in the source (so `foo bar` never verifies against `foobar`),
whitespace anywhere else (beside a bracket, a quote mark, punctuation) is optional (so a quote
that reflowed a line break still verifies). Call it from within the task's own instructions before finalizing
(closing this recipe's step-3 "regenerate on the gaps" loop with a real check instead of a
re-read), or reuse the same function host-side, outside the REPL, to re-derive whether a SUBMITted
citation was actually grounded: the same "derive facts from bytes, never trust the self-report"
posture the next section establishes for validity.

## Judgement-only SUBMIT: assemble facts, don't let the policy report them

A convention, not an API: the companion to grounded completeness, for the *other* side of a
model-backed tool. When a `make_model_tool` (or any tool) is the AUTHORITATIVE producer of an
artifact, the root LM's final SUBMIT must not re-carry that artifact. Two failure modes if it does:

- **Mangling.** A root LM that re-types the tool's output into its result can corrupt it (re-indent,
  drop a nested block), and nothing re-checks the re-typed copy, so a `valid=True` the LM *also*
  self-reports can label bytes that no longer pass the validator.
- **Trajectory poison.** The SUBMIT turn IS a training sample (`export_sft_turns`). If it re-authors
  the artifact, the policy learns to re-author it: exactly the job you gave the tool. And a
  self-reported validity flag becomes a label that can LIE: a downstream keep-filter
  (`complete and valid`) then keeps runs whose artifact is actually invalid.

The fix: keep DETERMINISTIC facts out of the policy's output type entirely.

1. **The `output_model` carries JUDGEMENT + a reference KEY, not the artifact.** The root LM SUBMITs
   its decisions (is this complete? what's missing? which variant?) and the producing tool-call's id:
   never the artifact bytes or a `valid` flag. With no field for it, the policy *cannot* re-type it,
   and the SFT turn stays clean.
2. **Assemble the artifact + its validity on READ, from the trace.** A small `assemble(result, events)`
   step re-sources each artifact VERBATIM from the matching tool-call event (by the id; last accepted
   wins) and DERIVES validity from the validator: never the policy's self-report. Run it everywhere
   the result is consumed: the live path, re-render, and the dataset exporters (`export_sft_turns` /
   `export_rl`), so the training labels read facts too.

   *Caveat when the validator CANONICALIZES, not just verdicts.* If the validator only returns
   pass/fail, verbatim re-sourcing is exactly right. But a validator that also REWRITES its input to a
   canonical form (stamps a fixed provenance field, strips a fabricated token) makes the raw draft and
   the canonical output DIVERGE, and re-sourcing the raw then ships the un-corrected bytes, so the
   deliverable silently misses a fix the run already applied (the root LM saw the corrected version, but
   the assembled artifact carries the raw one). Ship the validator's CANONICAL output as the artifact
   and derive validity from those same bytes; the tool-call still records the model's raw draft, so the
   trace stays faithful for RL while the deliverable stays canonical. Both are byte-identical no-ops when
   there is nothing to correct, so this costs nothing in the common case.

So the trace records the policy's real ACTION (its judgement), deterministic truth is COMPUTED (never
stored as if the policy produced it), and a self-reported flag can never drift from the bytes it
labels. Old traces heal on read: a pydantic `output_model` ignores the legacy artifact/validity keys
when coercing to the judgement-only type, and `assemble` re-sources them. A consumer does this: the
planner SUBMITs a per-artifact judgement keyed by the artifact's id; the system attaches the generator's
verbatim output and the validator verdict, so a re-typed/mangled artifact and a lying `valid` are both
structurally impossible.

**Corollary: the `run_start` meta must self-describe the run's CONFIG.** An OFFLINE, config-free
consumer (a dataset exporter, a re-renderer) can only read what the trace records. So any per-run
config it needs to INTERPRET the run: the expected value a validator enforced, the budget a
`hit_iteration_cap`-style metric compares against, the model roles: belongs in the `run_start` meta
the recorder writes, NOT a hardcoded default the reader guesses. Then an env override is honored
end-to-end (live AND in the offline labels), and an old trace lacking a key falls back gracefully.
This is the same principle as seeding `sft_turns` from the meta's initial state: the trace is the
sole source of truth for everything downstream of the run. A consumer records its canonical author
and `max_iterations` there so an offline reader reads the real per-run values, not the reference defaults.

## Building a consumer

`rlm-harness` is the ROLLOUT floor; a consumer is a thin declaration on top of it. `examples/harness_run.py`
is a minimal worked example: a task that wires the sub-LM hook, skills, tracing, and
RL export together. Six steps:

1. **Declare the task.** Subclass `RLMTask`: a `signature`, `output_field`, an `output_model`
   (judgement-only: see above), `instructions` (orchestration + a few hard safety rules), and
   `tools`. The retry/validation loop, sandbox, budget caps, and the trace are inherited. Put
   authoring KNOWLEDGE in a Skills directory (`load_skills_as_tools`), not the prompt: the prompt
   is for orchestration; skills are progressive-disclosure reference the LM pulls on demand.
2. **Add tools the base/wrap way.** Need a new capability (a model-as-tool producer, a fetcher, a
   searcher)? rlm-harness owns the GENERIC base + the syntactic guard + the async-safe factory
   (`make_model_tool`, `make_fetch_tool`, `make_web_search_tool`); the consumer owns the PROVIDER
   (the endpoint/validator/messages, or the httpx/vendor call) and the project-side TRACING. Tools
   passed to `RLMTask(tools=…)` MUST be sync: dspy's interpreter calls them with a plain `()`, and
   on dspy 3.4.0 an `async def` tool then raises `RuntimeError: This event loop is already running`
   (it silently returned an un-awaited coroutine before 3.4.0).
3. **Pick the recursion seat deliberately.** A DETERMINISTIC transform of the sub-LM's output →
   `intercept_sub_lm` (the escalation seat, recorded as a `sub_call`). An action the main LM CHOOSES
   to take → a tool (`tools=`, recorded as a `tool_call`). Don't smuggle a model-judgement (asking
   another model to grade the output) into the sub-LM intercept: that is an agentic decision and
   must be a tool, so it lands in the trajectory as honest RL data. (See "Sub-LM vs. tool".)
4. **Record + read through the trace.** Run inside a `TraceRecorder` (`on_event` gives a live
   observer for streaming). EVERYTHING downstream: your report renderer, your dataset, a re-render
   of a past run. Reads the JSONL trace, never the live objects. Carry any per-run config you'll
   need OFFLINE into the `run_start` meta (the corollary above), and assemble deterministic facts on
   READ (judgement-only SUBMIT), so a label can never drift from the bytes it describes.
5. **Export trajectories; score elsewhere.** `export_sft_turns` / `export_rl` / `export_actions`
   turn traces into training datasets. They are REWARD-FREE: each carries a `reward=` HOOK the
   trainer fills: rlm-harness never computes a reward.
6. **Delegate to another harness, or be one.** When a sub-task is better handled by a more
   specialized rlm-harness harness, delegate to it as a TOOL rather than reimplementing it. Two symmetric
   sides, both base/wrap, and NEITHER names the other harness in code: the identity lives only in the
   operator's runtime endpoint config:
   - **Client (you call another harness):** `make_harness_tool(invoke_fn, validate)` +
     `harness_from_endpoint(call_endpoint, read_output=…)`. The kit owns retry/validate/circuit-break +
     the child-rollout link; you own the transport (`call_endpoint`: a subprocess command / HTTP URL)
     and `read_output` (parse the child's reply). The single long-text arg becomes the child's RLM
     environment; the parent records ONE `tool_call` + a `child_run_id`/`child_trace` link while the
     child owns its own separate rollout (exported independently).
   - **Server (another harness calls YOU):** add a ~5-line `<pkg>/serve.py` that calls
     `serve_harness(run, to_pointer)`. The kit owns stdin→env, run_id, CWD isolation, the JSON-pointer
     wire, exit codes (0=ran / 1=infra→caller retries), and keeping your logs + tracebacks OFF stdout;
     you own only `to_pointer`: the mapping from YOUR result object into a `HarnessPointer`. The
     operator points the client at `python -m <pkg>.serve`. A FLAT result (`.artifact`/`.run_id`) needs
     NO file: `python -m rlm_harness.harness_serve <pkg.module>:run` uses the duck-typed default. Copy
     `examples/harness_serve.py`.
   - **Give the operator an ABSOLUTE `workdir_base`.** `serve_harness` isolates each run's CWD under
     it, and the default is RELATIVE: a child inherits the PARENT's working directory, so its run
     folders materialise inside the caller's project. Document an absolute path in your serve module
     (`python -m <pkg>.serve /var/tmp/<pkg>-harness-runs`), and have callers ignore `harness-runs/`.
   - **Your `run` must match the contract's shape, or adapt to it in `serve.py`.** `serve_harness`
     calls `run(source: str, run_id=…)`. A harness whose entry takes a domain object (a resolved lead,
     a parsed spec) or derives its own id needs a small adapter there: resolve the caller's text into
     that object through whatever PUBLIC seam the harness already exposes, and absorb the kit's
     `run_id` when the harness has a more meaningful one of its own: report the real id back through
     `to_pointer` so the parent's child-link still resolves. RAISE from the adapter when the text
     resolves to nothing: that is exit 1, which the caller retries then degrades. Returning an empty
     artifact instead is exit 0, and buys the caller a full run over nothing.
   - **A multi-file deliverable: use `bundle_artifact`, never a format of your own.** `artifact` is
     ONE string, but plenty of harnesses produce a FOLDER (a write-up + a PoC + a diff; a Dockerfile +
     a compose file + notes). `bundle_artifact({name: content})` packs it as `===== <name> =====`
     sections: escalating the marker when a file's own content contains one, so a report that QUOTES
     a bundle cannot truncate itself, and `parse_artifact_bundle` reads it back. Line endings are
     normalised to `\n` at pack time (both halves must agree on what a LINE is, or a header hides from
     escalation and still acts as a section break), and a filename that cannot round-trip raises
     rather than vanishing. Use them: a packing format invented per harness/client pair is a
     silent-failure generator, because the two sides agree until they don't and the mismatch then
     degrades into "the child returned junk" instead of surfacing as the wiring bug it is. A client
     that just wants the whole deliverable as CONTEXT needs no parser at all. The text is meant to be
     read as-is, by a Root LM and by a human debugging the wire.

   - **In-process transport (no subprocess).** The kit still ships no `call_endpoint`: subprocess
     (`serve_harness`) and HTTP remain the usual choices, but a THIRD option, for a trusted child
     harness in the same process/deployment where low latency matters more than OS-level isolation,
     is to await the child's `RLMTask.arun()` directly. Two small primitives make this easy to build
     correctly: `rlm_harness.tools.run_isolated(coro_factory)` bridges the sync tool-call contract
     into the child's `async arun()`, always on a dedicated new thread, so it works regardless of
     whether the calling thread already has a running loop (it will, whenever the parent task is
     itself mid-`arun()`), and `rlm_harness.tools.pointer_to_invocation(pointer)` is the canonical
     `serving.HarnessPointer` → `tools.HarnessInvocation` mapping, reused unchanged from the
     subprocess/HTTP case. **Read `run_isolated`'s docstring before wrapping a traced delegation**: a
     fresh thread starts with an empty `contextvars.Context`, so the delegated child's OWN
     `TraceRecorder` must be entered INSIDE the coroutine `run_isolated` runs, never around the call
     to `run_isolated` itself: a `TraceRecorder` entered outside is invisible to `current_recorder()`
     inside, and the child's own tool_calls/sub_calls would go silently unrecorded. See
     `examples/harness_local_run.py` for the full worked pattern (protected offline by
     `tests/test_harness_tool.py::test_in_process_transport_wiring`, which exercises the identical
     composition with a stub child instead of a real `dspy.RLM`). Nothing here adds a NEW
     code-execution surface: the child still enforces its own `RLMConfig`/sandbox guard on its own
     REPL code, exactly as it would over any other transport; an in-process call only changes HOW a
     Python object gets invoked, not WHAT gets executed where. Keep the subprocess/HTTP transport
     instead when the child needs real process/OS isolation, a different runtime/language, or truly
     runs on a remote machine.

**Score your own rubric (optional).** To decompose "did this run succeed?" into observable per-run
LABELS, `rlm_harness.rubric` gives you the reward-free substrate: the `Criterion`/`RubricCriteria`/
`CriterionFact` types, `rubric_to_meta`/`rubric_from_meta` (carry the rubric in the `run_start` meta),
`validate_rubric`, and a pure `criteria_facts(criteria, facts, lens)`. `category` is an OPAQUE label YOU
define: the kit imposes no taxonomy. The pattern (all consumer-side except the primitives):
- define your own category set + a fixed (or per-task) criterion skeleton (`default_rubric`);
- write a `trace -> facts` function: **start from `compute_run_facts(events)`, which gives you the
  generic half for free (see "Rubric facts" below), and merge your own domain labels/metrics into it**
  so a criterion's facts can never drift from the export bundle, and a `category -> keys` lens
  choosing which facts each category surfaces;
- `criteria_facts(rubric_from_meta(events).criteria or default_rubric().criteria, trace_facts(events),
  LENS)` → per-criterion facts, reward-free. Emit them beside the trajectory via
  `run_label_bundle(runs, rubric=lambda ev: {...})`; a downstream trainer turns facts into a score.

An OPTIONAL model-graded EVAL is the same base/wrap shape: rlm-harness ships NO eval, only the pieces to
build one: wrap `make_model_tool` with YOUR judge prompt (100% your domain), a strict parser reading a
per-category 0-10 score dict, and a per-category means aggregation. Keep the prompt, the taskset, and the
category MEANINGS in your repo; the categories stay OPAQUE to the kit. This is a reference PATTERN, not a
shipped module: the valuable part of an eval is the domain prompt, which cannot be made generic without
emptying it.

**If you ship an in-repo `studio/` (or any workspace member that drives live runs), forward the
subscription extra.** A consumer's visual console is a uv workspace MEMBER with its own
`pyproject.toml`, kept behind a `live` optional extra so a replay-only deploy stays web-free. A
studio-scoped `uv` command (`uv run --package <consumer>-studio …`) resolves `--extra` against the
MEMBER, not the root, so the member must define BOTH `live = ["<consumer>"]` AND a forwarding
`subscription = ["<consumer>[subscription]"]`. Without the forward, `--extra subscription` is rejected on
the member and any sync that omits it prunes the Claude Agent SDK back out, so a subscription studio run
dies with `ImportError: ClaudeAgentLM requires the optional dependency`. The portable, cwd-independent
launch is therefore `uv run --package <consumer>-studio --extra live --extra subscription uvicorn
<consumer>_studio.app:app`. Every downstream studio carries this SAME pair, so one command is portable
across them.

**The promotion rule** keeps the boundary clean. When the consumer forces a workaround, ask "is this
GENERIC?" A reusable mechanic (the model-tool + retry + validate core, a new sandbox seam, a trace
hook) is PROMOTED into rlm-harness via the base/wrap split: the generic half here, the specific half in
the consumer. A consumer-specific VALUE (a model name, a schema, a validator, a path) stays in the
consumer. Never special-case the consumer inside the kit; never fork the harness or re-implement
tracing inside the consumer. If you need an internal seam the kit doesn't expose, ADD a public hook
here (that is how `recorder_scope` / `bind_recorder_to_sub_lm` / `get_sub_lm` were born) rather than
reaching into a `_private` name. The trace schema, the `EVENT_*` types, and the exporter record shapes
are a FROZEN v1 wire format: `tests/test_contract.py` pins them; adding an optional field is fine,
removing or re-typing one is a `v2` break. The `EVENT_*` type constants are exported from `rlm_harness`,
so a trace reader matches on `rlm_harness.EVENT_RESULT` instead of hardcoding the wire string `"result"`.

**The stage boundary** keeps the data honest. rlm-harness + your consumer are the ROLLOUT stage: they
produce trajectories (the trace) and turn them into datasets, emitting raw LABELS / METRICS, never a
reward scalar. SCORING (reward composition, credit assignment) and TRAINING (GRPO / SFT) are a
SEPARATE downstream project that installs the trainer. A prompt/policy rule that makes the rollouts
BETTER is in scope; a reward or penalty is not. Keep the trace clean training data and let the
trainer score it.

## Rubric facts: the kit computes the generic half (1.8.0)

`rubric.py` gives you the SHAPE of a rubric and stays deliberately empty of meaning: the category
label is opaque, and `criteria_facts(criteria, facts, lens)` is pure: it slices a facts dict
through your lens and knows nothing about traces. What was missing was the facts.

```python
from rlm_harness import compute_run_facts, criteria_facts, load_events

facts = {**compute_run_facts(load_events(path, run_id)), **my_domain_facts(events)}
per_criterion = criteria_facts(my_criteria, facts, MY_LENS)
```

`compute_run_facts` emits exactly `RUN_FACT_KEYS`: import it rather than hand-copying names:

| key | |
|---|---|
| `main_steps`, `tool_calls`, `sub_calls`, `tool_call_rate`, `sub_call_rate` | what the run did |
| `tool_declines`, `tool_endpoint_errors`, `tool_circuit_breaks`, `tool_ok` | the four outcomes, summed |
| `tool_unknown` | calls whose writer declared no outcome, so the four above are a SUBSET of `tool_calls` |
| `tool_wasted_seconds`, `tool_total_seconds`, `tool_measured_calls` | wall-clock OCCUPIED by tool calls, and how much of it was measurable |
| `fence_refused_turns` | turns dspy refused over a fence tag: read the note below |
| `budget_exhausted` | `True`/`False`/`None`; the ITERATION cap only |

`tool_calls_by_name` is deliberately absent: an open key space cannot be a closed set. Call
`compute_run_utilization` / `compute_tool_waste` for per-tool detail. Use
`compute_run_facts_by_run` for a file holding several runs.

**Three readings that are easy to get wrong.** `tool_wasted_seconds` and `tool_total_seconds` are
`None`, never `0.0`, when nothing carried a duration. Unmeasured is not zero, and
`tool_measured_calls` is there so one measured call in fifty does not look like fifty in fifty.
And since 1.9.1 both measure the **union** of the tool calls' intervals rather than their sum, so
neither equals `sum(w.total_seconds for w in compute_tool_waste(...))` once one tool call nests
inside another. A sum double-counts the nested call, which is two correct events describing one
stretch of wall clock; on a real trace that reported **136.5% of the run's own span**. It takes both
halves to happen: the kit times the outer tool for you, and the inner call is recorded by the tool
itself, so a flat tool graph never sees the double-count.

**Do not assert `new <= old` across the upgrade.** The relation is strictly smaller only when
something nests; otherwise the two agree just to within reconstruction error, and that error has two
terms. Float quantisation contributes ~1e-7 s (`ts - (ts - d)` at epoch scale), in EITHER direction:
a 24-call run with nothing nested came out 1.6e-7 s *larger*. Clock rate difference contributes
`duration_s x drift`, which is bigger and grows with the call: on one run with nothing
double-counted, a 0.9 s call and a 298 s call reconstructed as overlapping by 7.8 ms. They had run
back to back: moving the total by −0.00208% of its own value.
And `budget_exhausted` is `None` whenever the answer is unknown; it never guesses `False`.
`fence_refused_turns` is `0` on a run with no turns at all (unmeasured, not measured-zero) so read
it beside `main_steps`, which rides in the same dict for exactly that reason. In that same zero-turn
case **both rates are `None`** (there is no denominator), and a crashed run really can carry
live-recorded tool calls with no `main_step`, so a lens doing arithmetic on a rate must handle it.

**`fence_refused_turns` is named for a mechanism, and the obvious cause is wrong.** Across three
real corpora, **60 of 60** refusals had the fence BURIED rather than leading: 55 of the 60 cells
are valid Python assigning a documentation page whose text contains a fenced example: `markdown = """# Overview … ```bash … """`. dspy's fence stripper
scans the whole cell including string literals. So this counts environment friction, not a model
failing to follow a format; a consumer read it the other way and spent two prompt generations
suppressing the code blocks its own pages needed.

**Writing the facts into the trace.** Off by default. `TraceRecorder(..., record_metrics=True)`, or
`RLM_TRACE_METRICS=1`, folds them into `run_end.payload["metrics"]` at teardown, computed from the
file just written. A killed run has no `run_end` and therefore no snapshot: `compute_run_facts` on
the events is the authority in every case; the snapshot is a convenience for a reader that will not
run it.

### When a run fails, the trace says why

`run_end` carries `ok: False` and `error`: the outer exception's `repr`. Since 1.8.4 it also
carries **`error_chain`** when there is one: the causes BELOW that exception, innermost last, each
rendered as `Type: message` by `short_error`.

That matters because `RLMTaskError`'s message is deliberately generic. `run_with_retry` raises it
`from` the last real failure, so the cause is on the exception object, and the `repr` that was all
`run_end` recorded drops it. Across a nine-consumer fleet, 15 of 15 recorded failures carried
nothing but the generic sentence, on the one artifact that outlives an intermittent failure you
cannot reproduce.

    "error": "RLMTaskError(\"Failed to produce a valid 'result' after 2 attempts: ValueError: adapter could not parse the completion\")",
    "error_chain": ["ValueError: adapter could not parse the completion",
                    "ConnectionError: proxy: read timeout on POST /v1/chat/completions"]

**`run_end` also carries `budgets` and `usage` since 1.10.0: because a TRUNCATED completion and a
MALFORMED one raise the same exception.** dspy detects truncation (`finish_reason == "length"`) and
only logs a warning, so the trace recorded nothing to tell them apart, and the misdiagnosis it
enables is one this kit has already documented once, under `max_tokens`.

    "budgets": {"main": {"cap": 16384, "key": "max_tokens"},
                "sub":  {"cap": 4096,  "key": "max_tokens"},
                "iterations": {"max_iterations": 10, "max_llm_calls": 30,
                               "max_output_chars": 10000, "dropped": false}},
    "usage":   [{"attempt": 0, "turns_recorded": true,
                 "calls": {"openai/gpt-4o": [{"prompt_tokens": 1234, "completion_tokens": 2100}]}},
                {"attempt": 1, "turns_recorded": false,
                 "calls": {"openai/gpt-4o": [{"prompt_tokens": 1234, "completion_tokens": 16384}]}}]

**A truncation has THREE causes once a thinking budget is in play, and the fix differs per cause.**
Read them off `completion_tokens`, `reasoning_tokens` and the two recorded caps, per CALL, which
needs no mapping from a call to a turn. Reasoning AT the thinking ceiling and completion AT the cap
means the budget is working and the TOTAL is too small: raise `max_tokens`, or lower the thinking
budget to leave the answer room. LOW reasoning and completion at the cap is a content runaway the
thinking budget cannot reach, so raising it will not help. No thinking budget at all is the original
case below. Measured on one deployment: of 6 cap hits in 1762 calls, 3 were the first kind and 1 the
second.

**Token counts cannot tell you WHAT the model emitted. Only the response text can, and the trace
carries it.** This paragraph asserted a "no content" signature twice and both versions were wrong, so
the rule it leaves behind is the useful part. First it said `reasoning_tokens == completion_tokens ==
cap` means the model emitted nothing, generalised from three observations that all happened to sit at
the cap. Then a call at `(completion 1384, reasoning 1383)`, far below both limits, was taken as
proof that the cap was incidental and the EQUALITY was the signature. Reading that call's actual response
settled it: the model had returned a full paragraph of prose, and the provider had counted the
visible content as reasoning tokens. **So `reasoning == completion` is consistent with a complete
reply, and neither version of the claim survives.** The earlier cap-level observations may well
have been empty, but the counts were never what showed it.

What to read instead: `run_end.payload.error_chain`, whose `AdapterParseError` frame embeds the head
of `LM Response:`. That is what identified the call above, and `short_error`'s ~600-character cap
keeps the beginning, which is the part that distinguishes an empty reply from a full one in the wrong
shape. Usage answers "how many tokens"; the error chain answers "what came back". Do not substitute
one for the other, which is what both retracted versions of this paragraph did.

**One failure class that IS worth recognising, from the same call.** The reply was thinking-style
prose opened with a bare `{`, sitting where the JSON envelope's `reasoning` and `code` fields belong,
so `Actual output fields parsed from the LM response: []`. That is the model's thinking leaking into
the CONTENT channel and breaking the envelope, which is close to an ordinary wrong-shape reply and
nowhere near a truncation: no cap, no ceiling, nothing to raise or lower. One instance. Distinct from
the reasoning-model case `_LenientJSONAdapter` already handles, where the whole structured turn
arrives in `reasoning_content` and `content` is empty; here the channels are the other way round.

**Do not read the three branches as predicting the OUTCOME. Position does that.** On the same corpus
a branch-1 call (reasoning at the ceiling, completion at the cap) landed mid-run at call 24 and the
run RECOVERED, while a branch-2 call (content runaway) landed on the FINAL call and the run failed.
That is the 1.10.1 finding restated: a truncated CODE cell comes back as a turn the model can retry,
and a truncated FINAL answer has no handler. The branch tells you which knob to reach for; where the
call sits in the run tells you whether it was survivable. Bucketing outcomes by branch alone will
attribute to the branch what position caused.

**What a thinking budget actually buys on the committing turn is ROOM FOR THE ANSWER, not better
reasoning.** This is the sharper reading of the same knob and it came from a paired rerun on that
deployment: three questions, same model, same wiki, same 12-turn fast-mode cap, with only the budget
unset. With it, 3 of 3 finished. Without it, 3 of 3 FAILED, every one at the last iteration, every
one with the identical shape `(completion 32768, reasoning 32768)`: the whole cap spent reasoning,
zero content, dspy's truncation warning, then `RLMTaskError`. The budgeted runs reached that same
turn, were cut at the ceiling, and dspy's forced extract had tokens left to write an answer with.

Two things follow that a rate would not tell you. **Both conditions used all 12 turns**, so a budget
does not cause or prevent iteration-cap pressure: that came from the questions, each a 2,000-2,900
character five-part audit. And the failure concentrates on the turn that has to COMMIT, the one with
the most to synthesise, which is why looking at a distribution over all turns misses it and why the
consecutive-ceiling precursor below points at the same place. In the failing runs one of the earlier
turns also reasoned past 16384 and completed anyway, so cutting a normal turn at the ceiling is not
what was at stake.

n=3, paired, one variable, and one disclosed confound: the reruns carried no page context that the
originals may have had, which the trace cannot settle. Read it as a mechanism, not a rate.

**Repeated hits at the thinking ceiling look like a precursor, and what it detects is a STUCK
model rather than a hard task.** Without a budget the ratio below has a hole and gives no early
warning. With one, calls pile up AT the ceiling instead of passing through it. On one deployment
both failing runs showed 10+ consecutive calls at the ceiling before dying, and the same page then
SUCCEEDED on a later attempt with 2 ceiling hits and never two in a row. That pair is what says the
pattern is not the page needing more thinking, since the budget was identical on all three
attempts: it is the trajectory looping. Still one page, so watch it rather than alert on it.

Read `completion_tokens == cap` as a truncation. **Whether the RATIO gives early warning depends on
how generous your cap is, and 1.10.0 shipped claiming it always does.** On the first production
corpus: one model, cap 32768, **385 runs reaching `run_end`: 379 that succeeded and 6 that
failed**. The distribution has a HOLE. The bins are the **379 successes**: 363 below 0.6, ZERO
between 0.6 and 1.0, 16 at the cap. Adding the 6 failures (1 below 0.6, 5 at the cap) gives
**364 / 0 / 21 across all 385**. A turn stays under ~0.55 or blows straight through, and a
proximity meter has nothing to point at.

But the hole is an artifact of that cap, not of the model. Converting the bins back to absolute
tokens and re-dividing by a 16384 cap moves **64 of the 379 into the empty band, 16.9%**, with
the median max-turn going 0.209 → 0.418 and p90 0.378 → 0.756. Quoted over the successes because
that is the population the bins cover; the all-385 figure is not derivable without knowing whether
the one sub-cap failure sits in [0.3, 0.5), which nothing here records. So a cap set at roughly twice
what the model needs produces the hole; a tighter one produces a real gradient, and there the
warning reading works. Measure your own before building either. (The transposition assumes token
counts do not change with the cap: `max_tokens` is a hard stop rather than a hint, so that should
hold, but nothing here settles it.)

That corpus also measured what truncation costs, which was invisible before this release: **21 runs
of 385 (5.5%) hit the cap, and 16 of those 21 (76%) finished anyway.** Not luck. It is WHERE in the
turn the truncation landed. A truncated CODE cell raises a `SyntaxError` that dspy hands back to the
model as another turn, and the model repairs it. A truncated FINAL answer is an adapter parse failure
with no handler, and kills the run. So the visible failures are the ~24% that land in the wrong
place, and the rest self-heal without anyone knowing they happened.

**The recovery is NOT the diagnostic doing the work, and this sentence used to say it was.** On the
default `pyodide`/`deno` interpreter, the message dspy hands back for a syntax error is EMPTY. Not
sometimes: measured against a real Deno sandbox, the raw JSON-RPC carries `{"message": "",
"data": {"type": "SyntaxError", "args": []}}`, because dspy's `runner.js` excludes `SyntaxError`
from its args extraction on the grounds that "only python exceptions have args" while a Python
`SyntaxError`'s args are exactly where the line and offset live. A consumer counted 1029 of these
across four kit versions and not one carried a non-empty message, and filed it upstream as
[dspy#10509](https://github.com/stanfordnlp/dspy/issues/10509) with the two-line fix; until that
lands, assume the feedback carries nothing. So the model sees `[Error] Invalid
Python syntax. message:` and nothing else, and the 76% is what it recovers with NO information about
what was wrong, which makes the figure more impressive and the feedback path worth fixing upstream
rather than relied on. The kit's own `container` interpreter does not share the hole: its agent sends
`str(e)`, so the same failure reads `invalid syntax (<sandbox>, line 1)` there.

Five things to know:

- **`budgets` is the cap the LM CARRIES, read off the LM, not `RLMConfig`.** An injected
  `main_lm`/`sub_lm` is used verbatim, so the configured value can be one the call never used. It
  is the cap APPLIED for every LM that honours its own kwargs, which is the assumption dspy's own
  truncation check makes too, but an LM that accepts a kwarg and ignores it reports a cap nothing
  enforced, and `completion_tokens` can then exceed it. `ClaudeAgentLM` is the shipped case: the
  subscription SDK exposes no sampling controls. **The trace can FALSIFY that a cap was applied and
  can never CONFIRM it**, but only against the largest of the per-role token caps (`main`/`sub`),
  never one role's. `budgets` records no model name and `usage` is keyed by MODEL, so nothing in
  the trace maps a usage entry to a role at all: sharing a model string makes it vivid (the calls
  land in one list nothing can key apart, see the `usage` bullet below) but differing model
  strings do not rescue it either. So a `completion_tokens` above `sub.cap` may simply be a main-LM
  call: in the example above, 16384 exceeds `sub`'s 4096 while being an ordinary main truncation at
  exactly `main`'s 16384. Above every per-role cap, no role can account for it and the equality is
  broken. (An operator who knows their own config can attribute out of band, and a per-role
  comparison is valid for them; "never" is about what the trace ALONE licenses.) For this LM
  specifically the model key carries `SUBSCRIPTION_PREFIX`, so a reader can recognise it by name
  alone, with no attribution needed. And note which direction the exposure runs: the auto-routed
  path builds this LM with no kwargs at all, so a cap appears for it only when you constructed the
  LM yourself and passed `max_tokens=`. `key` says which name held it: dspy rewrites `max_tokens`
  to `max_completion_tokens` for OpenAI reasoning models, and a
  reader of the first name alone sees nothing for exactly that class of model. An absent
  `main`/`sub` means no cap was set on that role: never `0`, and never `False` for the derived truncation. (`budgets` itself is
  present on any task-driven run, since `iterations` is unconditional; its absence means nothing was
  staged at all.)
- **`usage` is per ATTEMPT, and `turns_recorded` marks the one whose turns are in the trace.**
  `run_with_retry` re-runs the whole trajectory, and the attempt that reached the trace is NOT
  always the last: a run whose final attempt RAISES keeps an earlier attempt's turns. Scoping usage
  to the attempt with the turns would have discarded the fatal call's tokens, which is the number
  worth having. A run that never produced a prediction: `main_steps: 0`, the shape of the incident
  behind this. Records every attempt with none flagged.
- **It is NOT per turn, and that is a limit, not an omission.** `sub_model` falls back to
  `main_model`, and dspy propagates the tracker into `llm_query_batched`'s workers, so planner
  turns, sub-LM escalations and same-model tool-LM calls land in one flat list under one key with
  no call id and no timestamp. Nothing in dspy's tracker can key them apart. For a distribution
  over runs use `max(completion_tokens)` per run, never the run's SUM: summing fifteen turns'
  tokens against a per-completion cap answers a cost question, not this one.
- **Usage may be absent**, and absent is not zero: not every provider returns a usage block, dspy
  drops an empty one entirely, and streaming needs `dspy.settings.track_usage` for the provider to
  send it at all. If your own code opens `dspy.track_usage()` around a run, the kit REUSES your
  tracker rather than shadowing it, and records only the calls it made. One cost of installing a
  tracker at all, if you had none: dspy attaches per-prediction usage (`get_lm_usage()`) only when
  no tracker is installed, so a `dspy.Module` YOU call from inside a kit run returns `None` there
  while the kit's scope is open. The numbers are not lost. They are in the tracker, and in the
  trace, but that one accessor stops answering.
- **An `api_rounds` EVENT answers a question the token fields cannot.** One LM call can make more
  than one API request, and each request can run several *sampling iterations* server-side. The
  event's payload is `{"model": ..., "rounds": [ ... ]}`: the provider's own per-iteration
  breakdown, recorded verbatim when a call reports one and absent when it does not, one fact with
  one meaning. Only `ClaudeAgentLM` produces it.

  **It moved in 1.17.0**, from `run_end.payload.usage[...]["api_rounds"]` to its own event, and the
  move was forced rather than chosen. An engine-backed LM's usage reaches dspy's tracker through
  `usage_dict`, which rebuilds a fixed key set off the typed `lm15.Usage` and passes nothing else
  through, so the breakdown could no longer ride along (measured: present on the old `forward()`
  path, absent on the engine path). Nor could the kit put it back beside the right entry
  afterwards, because a tracker entry carries no per-call identity and only POSITION could match
  it, which is unsound once dspy owns retries and `llm_query_batched` fans calls across threads.
  The payload keeps the `rounds` nesting so a reader that already parsed `api_rounds["rounds"]`
  parses it unchanged. The old key is GONE rather than deprecated: a reader still looking at
  `run_end.payload.usage[...]["api_rounds"]` finds nothing, not a stale value. That half is a
  removal, not an addition, so **it is what took the wire format from `trace/v1` to
  `rlm-harness/trace/v2`** in the same release: the additive-only rule asks for a version bump with
  a migration rather than a key quietly disappearing, and this is the whole migration. Nothing else
  about v2 differs from v1, the kit validates the string nowhere, so an existing v1 corpus still
  loads through `load_events` and every exporter; a consumer that does validate it should accept
  both. The field's purpose is the CONTEXT WINDOW SIZE: the provider's documentation says to calculate
  it "from the last `message` entry". **The totals give you that too, but only conditionally.**
  When a call made one API request running one SAMPLING ITERATION, the top-level fields and that
  single entry are the same numbers and `prompt_tokens` already IS the context size. ("One
  `message` entry" is a different and weaker condition: a server-side fallback puts a `message`
  entry for the declined hop and a `fallback_message` for the serving one in the same request, so
  the totals cover both. That last step is an inference: only `compaction` is documented as
  excluded from the top-level fields, and it errs safe: the tighter condition can only decline an
  equality that happens to hold, never assert one that does not.) What the totals cannot say is WHICH call they were, so read a missing
  `api_rounds` as "the provider reported no breakdown", never as "no context number here". Three
  things to know, each a way this will otherwise be misread:
  - **It is NOT a decomposition of the call's `prompt_tokens`.** The two do not cover the same
    span. A call's token fields accumulate across every API request it made, while `api_rounds`
    holds only the LAST request's iterations, so on a call that took more than one request, the
    totals span all of them and the rounds do not. Independently: a `compaction` entry's tokens
    are excluded from the top-level fields entirely. Summing the rounds does not reconstruct the
    call, and the difference is not an error in either number.
  - **For context size, take the last `message` or `fallback_message` entry: never a
    `compaction` one, even when it is last.** There are four entry types, discriminated by `type`.
    A `compaction` entry reports what the summarisation itself cost, NOT the size of the context it
    closed, and the provider warns those differ by orders of magnitude: a compaction closing a
    ~200k context can report a few thousand tokens. Filtering by `type` is the whole difference
    between a context reading and a wrong one, and it matters exactly in the long-context regime
    where you would reach for this field.
  - **Order by `step_id`, and take the LAST event.** That ordering is the reason this is an event:
    the question the field exists for is what the latest breakdown says, and `step_id` answers it
    without correlating anything. The hazard its predecessor carried is gone with the move: a
    reader who went through `dspy.track_usage()`'s `get_total_tokens()` used to get every call's
    rounds merged into one flat list with the per-CALL boundaries lost, and dspy performs that
    merge ITSELF from `dspy.Module.__call__` under `dspy.configure(track_usage=True)`. The
    breakdown no longer passes through that aggregator at all, so there is nothing left to merge.

    The event is recorded mid-call, so its `step_id` comes BEFORE the `main_step` or `sub_call`
    for that same call, not after. Taking the last breakdown is unaffected, which is the question
    this answers. Correlating a breakdown to one specific call is not available: by adjacency you
    would attach each one to the call that follows it, and nothing would say so.

  A nested `*_details` / `cache_creation` object inside a round is a DECOMPOSITION of a field
  already counted, never an addition: add up every integer in a round and you count the cached
  half twice. Written for the SHAPE rather than today's field names, because the provider says
  `output_tokens` "remains the inclusive, authoritative total" of its own breakdown too.

**`budgets.thinking` is the THINKING ceiling, and it is its own key for a compatibility reason.**
`budgets.main` / `.sub` mean "this role carried a token cap" and their ABSENCE means no cap was set,
so a `main` holding only a thinking budget would change what an existing reader's
`budgets["main"]["cap"]` may assume. `budgets.thinking.main` sits beside `budgets.main` instead,
purely additive. It carries `{"value": int, "key": str}` where `key` is the dotted PATH the value
was found at (`extra_body.thinking_token_budget`), because a top-level litellm parameter and a raw
`extra_body` key are different mechanisms. **Absent means NOT RECOGNISED, never "no budget"**: the
kit owns no vocabulary on the wire, so a name it does not know is still SENT and simply not
annotated here. And a recognised key proves only what was sent: pair it with `usage`'s
`reasoning_tokens` to find out whether the server honoured it.

**`budgets` lives in `run_end`, which a KILLED run never writes.** A run that dies by exception
still records it, since `__exit__` runs during propagation, but a process stopped by a signal does
not get there at all, and a thinking runaway is exactly the kind of run an external timeout kills.
`applied_lm_budget(lm)` and `applied_thinking_budget(lm)` are public for that case (1.13.0): read
them off `dspy.settings.lm` and `get_sub_lm()` after `configure()`, and put the result in the
`run_start` meta, which is written before the run begins.

```python
from rlm_harness import applied_lm_budget, applied_thinking_budget, get_sub_lm, configure
import dspy

configure(RLMConfig.from_env())
meta = {"budgets": {role: {"tokens": applied_lm_budget(lm),        # {"cap": int, "key": str}
                           "thinking": applied_thinking_budget(lm)}  # {"value": int, "key": str}
                    for role, lm in (("main", dspy.settings.lm), ("sub", get_sub_lm()))}}
with TraceRecorder("traces/run.jsonl", run_id="r1", meta=meta):
    ...
```

`get_sub_lm()` is the LM a task will use unless you hand it an explicit `sub_lm=`, so both roles are
readable before the task exists. Use the kit's readers rather than a key list of your own: the names
a server may use for a thinking ceiling live in one place on purpose, and a second copy is what
drifts. The shapes are the same ones `budgets.main` and `budgets.thinking` publish, so the two
records cannot disagree.

**`budgets` covers a DIFFERENT exhaustion from `budget_exhausted`.** The `metrics` snapshot's
`budget_exhausted` reports the ITERATION cap: the run used up its turns. `budgets.iterations` shows
what those caps were, `budgets.main`/`.sub` show the TOKEN cap, and `budgets.iterations.dropped`
says whether dspy rejected the iteration kwargs and reverted them all to its own defaults, without
which the three numbers would read as applied when they were not. A run can exhaust either budget,
and `max_output_chars` is a third, independent truncation: dspy head+tail-caps each REPL output, so
a reader diagnosing "the output was cut off" has three mechanisms to rule out, not one.

**Read it as absent-or-populated, never as empty.** The key is written only when a chain exists, so
a reader can tell "there was no cause" from "we could not build one". `error` itself is unchanged.

**Three things it is not.** It is not a traceback: a traceback names every frame's file and line,
which multiplies the paths on offer; a message carries only what the raiser put in it. That is a
difference of degree, not of kind: `ValueError("cannot open /data/private/x.txt")` is one message
and one absolute path, so do not read "no traceback" as "no paths". It is capped at five frames,
truncating the OUTER end so the root cause survives on a deep chain. And each frame goes through
`short_error`, which bounds it just over 600 characters. Its contract is "never longer than the limit
plus an elision marker", so treat it as a bound, not an exact number: because dspy's
`AdapterParseError` embeds the entire raw completion.

**Two cautions if you forward it anywhere.** Unlike `error`'s `repr`, a frame is not guaranteed to
be single-line, pydantic and dspy adapter errors are multi-line, so a bare SSE `data:` field or a
markdown table cell will break on one where `error` never did. And an exception message can carry a
URL with a query-string token; this kit has no scrubber, and `short_error`'s cap is the whole
mitigation. That exposure already existed for `error`; the chain widens it to third-party frames,
which is where such a token most often is.

**If you already walk `__cause__` yourself**, the recorded chain is the authority. It is deeper
than a one-level walk and honours `__suppress_context__`, but each frame is shorter. Prefer it over
a second derivation, or the two will disagree about the same failure.

**If you are starting a rubric from scratch**, four categories that work for most agent runs. The
kit ships none of this. The names are yours to copy and the sentences after the dash are yours to
write:

```python
CATEGORY_MEANING = {
    "TF": "Task Fulfillment      - <did the run produce what was asked, within its budget>",
    "TA": "Tool Appropriateness  - <did it reach for the right capabilities, without waste>",
    "TG": "Tool Grounding        - <do its claims rest on what it actually observed>",
    "PA": "Parameter Accuracy    - <were its calls well-formed enough to succeed>",
}
```

A lens is a VIEW, not a partition: one fact may legitimately appear under two categories.
`budget_exhausted` answers the same question as a hand-rolled `hit_iteration_cap` and answers it
better: the `steps >= cap` form is a false positive on a run that submits on its last allowed turn,
so keep one, not both.

## Reading a trace: the ordering rules

Three facts a reader needs and cannot infer from the file, plus one about how they interact. A
downstream consumer got the first two wrong, concluded "file order is unreliable, sort by `ts`",
and reordered its turns; a second one got the fourth wrong and was out by four orders of magnitude.

**ORDER is not the hard part, and `step_id` is a perfectly good key for it.** `record()` assigns
`step_id` and stamps `ts` inside ONE critical section and writes the line in the same one, so for
the live events (`tool_call`, `sub_call`) `ts` is SAMPLED in `step_id` order, which is also file
order. The two therefore agree on real hardware and are not guaranteed to: `ts` comes from
`time.time`, and a wall clock that steps backwards breaks the agreement while `step_id` cannot.
So `step_id` is the sturdier key for pure ordering. `main_step` is the exception
and has its own key, `payload["turn"]` (see the bullets below).

**The hard part is ADJACENCY, and that is where the four orders of magnitude came from.**
Contiguous `step_id`s among live events do NOT mean "these happened in the same turn". Because
every `main_step` is written in one batch at the END of the run, the live events of a whole run
carry an unbroken `step_id` range with nothing between the turns, so grouping by runs of
consecutive `step_id` collapses an entire run into one burst. A consumer computing which tool calls
shared a turn that way measured a burst-parallelism ceiling of **90.5%** where the answer was
**0.0037%**, and the alarming number is the wrong one. To group live events into turns, use the
GAPS between their timestamps, or place them against a turn with the `ts` bullet below: never
`step_id` adjacency.

**And never mix the two families in one sort.** Sorting `main_step` together with `tool_call` /
`sub_call` by `step_id` puts EVERY turn after EVERY tool call of the same attempt, because the
batch lands last. `dataset.export_actions` did exactly this until 1.11.2, and its `state` field:
"the ordered list of prior actions". Was wrong in that specific way: a `tool_call` record's prior
actions contained no turns at all, and a turn's contained every tool call. It now interleaves on
`ts`; `dataset._sequenced_actions` is the worked version of this rule, including what it cannot do
when a turn's stamp was never matched. `replay.py`'s single-family `step_id` sort is unaffected and
deliberately unchanged.

- **`main_step` events are written in one block AFTER the run.** `record_main_trajectory` runs once
  `aforward()` has returned, because dspy.RLM only exposes its trajectory post-hoc. So a `tool_call`
  recorded mid-run appears EARLIER in the file than the `main_step` of the turn that made it, while
  being chronologically later. Measured at 70 of 76 real traces. By design, not a defect.
- **`payload["turn"]` is authoritative for ordering**, and file order among `main_step` events
  already matches it (72 of 72 traces). **Never sort `main_step` events by `ts`**: a `ts` is
  backfilled from a live stamp and, for a turn whose stamp could not be matched, falls back to the
  flush time, so sorting by it moves turns.
- **`ts` is for placing a turn against the tool calls around it**, and nothing else. A per-turn
  duration computed as `t[i] - t[i-1]` is an estimate that also contains the model's generation
  time; `exec_duration_s` (1.6.0) is the measured `execute()` half, and `duration_s` on a
  `tool_call` is the measured tool half. **`exec_duration_s` is `execute()` wall-clock, not sandbox
  CPU:** dspy dispatches tool calls and `llm_query` synchronously from inside `execute()`, so a cell
  that calls one blocks, and that whole round trip is inside the number. Read a large value as
  "the turn blocked", and try to cross-check it against the `tool_call` / `sub_call` events in the
  same run, but that check is often unavailable, and its absence is not the field lying: a `sub_call` carries no
  duration of its own, and a `tool_call`'s `duration_s` is filled since 1.8.3 for most: not all:
  of what a task hands the model. Before 1.8.3 it was present only for the six tool sources that
  timed themselves (`fetch_url`, `web_search`, `run_command`, `git_clone`, `model:<id>`, MCP) and
  absent for everything else; before 1.6.0 it did not exist at all. It is still absent for a tool
  called outside a task and for the shapes listed under "Which shipped tools carry a duration"
  below. Treat it as optional and read `None` as "not measured", never as zero.
  On a trace written before 1.7.0 there may be no `sub_call` at all: the
  event became automatic in that release, so check `run_start.rlm_harness` before reading a zero.
  For scale, measured on a real workload: execution is ~1% of a turn's wall-clock; ~99% is the model
  generating. Prefer a measured field over a gap wherever one exists, and treat a
  NEGATIVE gap as unknown rather than as data (traces written before 1.6.1 can contain them; see
  that entry in `CHANGELOG.md`).

## Trace utilization metrics

`rlm_harness.metrics` answers "how was this run's activity distributed": a sibling question to
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
batch/dataset-level view. Reads only already-frozen trace fields: `event["type"]`,
`payload["tool"]`, the optional `payload["duration_s"]`, and (through `payload_cause`, never
directly) `circuit_broken` / `endpoint_error` / `error` / `ok`. No new event type, nothing the
trace contract needs to change for.

### `compute_tool_waste`, which calls produced nothing usable, and what they cost

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
  ABSENT on an endpoint-failure payload: `payload.get("ok")` then returns `None`, which is falsy,
  so a naive counter silently absorbs infrastructure failures as content declines. That mistake has
  shipped four times. The four outcome counts can therefore sum to less than `tool_calls`: a
  payload with no `ok` and no endpoint string is `tool_unknown` rather than a decline.
  `invalid_rate` is likewise denominated over the calls that actually reached a
  validator, not over every call: a circuit break ran no validator and an endpoint failure produced
  no output to judge.
- **It never infers a duration from the gap between events.** `*_seconds` is `None`: meaning "not
  recorded": whenever the events carry no `duration_s`. Since 1.8.3 that is narrower than it was:
  a task fills the field for nearly all of what it hands the model, so `None` now means a pre-1.8.3
  trace, a tool called outside a task, or one of the shapes named below; before 1.6.0 the field did
  not exist at all. `0.0` would read as "measured and found to be free", and inferring from event gaps charges
  a whole turn's model generation to that turn's first tool call. Every wall-clock attribution made
  against this kit's own corpus before 1.6.0 had exactly that error in it.

**Which shipped tools carry a duration: since 1.8.3, nearly all of them.** `RLMTask` wraps every tool it
hands the model, and `record_tool_call` fills `duration_s` from that wrapper whenever the tool did
not measure itself. A tool that DOES measure itself keeps its own number, because it scopes the
window more precisely: `fetch_url` starts its clock after the SSRF check, `run_command` keeps a
runner-reported figure alongside. Your own tools are covered too, with no second event: the wrapper
publishes a start time and records nothing.

**Four shapes are deliberately NOT wrapped**, and a fifth is wrapped but records nothing, each falling back to an absent field rather than a
wrong one. A `dspy.Tool` OBJECT is passed through untouched (`mcp_tools` returns these, and they
record their own duration; a `dspy.Tool` you build yourself does not, so pass `duration_s`
yourself). So are a callable class instance and a `functools.partial`. `functools.wraps` does not refuse
either, it skips the attributes they lack, which is the problem: the wrapper keeps its own
`__name__` and dspy then registers `timed` with the arg types dropped. A coroutine function is
skipped for a different reason: `wraps` handles it fine, but dspy branches on
`inspect.iscoroutinefunction`, which does not follow `__wrapped__`, so wrapping one turns a run
that completes into an error. A GENERATOR FUNCTION is wrapped like any other, but records
nothing: the wrapper hands back the generator object and releases the start time before the body
has run, so the `record_tool_call` inside that body finds none. (A plain function that merely
returns a generator expression is timed normally. Its body does run inside the window.)

And the fill is matched on the tool's NAME, so a call recorded under a name other than the wrapped
function's `__name__` is left unmeasured rather than charged your window, including the tools your
own composite tool calls inside itself.

**The one shape that gets a wrong number rather than none**, disclosed because this function's own
bar is that a confident wrong answer is worse than an honest unknown: a composite tool whose
`__name__` MATCHES the tool it calls inside itself. A consumer wrapper named `read_file` that
delegates to `make_read_file_tool`'s `read_file` collides on the name, and every inner call is
charged the outer window. Measured with 3 inner reads inside a 0.15 s wrapper: four events of
~0.151 s each: the outer tool's own record lands in the same bucket, by construction, since the
collision IS the shape, so `compute_tool_waste["read_file"].total_seconds` reads 0.605 s against
0.15 s real, **4.0x**. Before 1.8.3 that shape produced `None`. Give your wrapper a distinct
name, or pass `duration_s` yourself. (Returning a kit factory's tool AS-IS is not this shape.
there is only one function, and it is timed correctly.)

This reverses two rules the kit used to state, and both reversals have the same cause.

*A refusal now carries a duration.* It used to record none, on the argument that a blocked URL
never touched the network so a ~0 would be noise. But `None` in `ToolWaste` means "nobody
measured", so using it for "measured, and it was instant" makes those two indistinguishable,
which is the structural-zero mistake 1.7.0 shipped a release to fix for `sub_call`.

*`grep_files` is no longer exempt*, and it was its own exemption that said so: the text here used
to justify leaving it untimed with n=146, median 0.029s, max 0.746s, and to name "a pathological
regex over a large tree" as the shape that would reopen it. Re-measured on a consumer deployment
across nine real repositories, 7 patterns x 3 runs each: **median 744 ms and max 6.3 s on a
2,110-file repository**, and it does not scale with file count: a 102-file repo measured slower
than a 1,210-file one, so the driver is bytes and match count, not files. Against the same corpus
that tool alone is about 40% of all sandbox execution time. The old number was not wrong; it was
taken on a corpus without a big repository in it.

The tool that dominates a real run is usually the consumer's own model-backed one, and for the two
BASE factories the kit cannot record it: `make_model_tool` and `make_harness_tool` are deliberately
side-effect-free (no tracing, no messages), so the consumer's wrapper owns the `record_tool_call`.
(`model_as_tool` is the exception, a model-backed tool the kit *does* record, and it carries its
own duration.) **Pass `duration_s=time.perf_counter() - t0` from it** (any monotonic clock):
that one line is what makes the 80% of a run's wall-clock visible.

### Denominators

Both `RunUtilization` rates are denominated over `main_steps` (root-LM turns): "how many tool calls / sub-LM
escalations happened per root-LM turn taken." This is a judgment call, not a uniquely correct
answer: the raw counts are exposed alongside the rates, so a consumer wanting a different
denominator can recompute one from the same fields. A rate is `None` (not `0.0`) when
`main_steps == 0`: `0.0` would misleadingly read as "measured and found to be zero usage" rather
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
`RLM_REQUEST_TIMEOUT`, `RLM_MAIN_LM_KWARGS`, `RLM_SUB_LM_KWARGS`, `RLM_DISABLE_HISTORY`,
`RLM_LM_CACHE`, `RLM_SANDBOX_SUB_AGENTS`, `RLM_OBSERVE`.

(`RLM_REQUEST_TIMEOUT` is this kit's own name. litellm separately reads a bare `REQUEST_TIMEOUT`
for its global default: setting that one moves litellm, not this.)

The `AI_*` fallbacks let the kit drop into projects already keyed on those vars
without re-keying env; the `RLM_*` form wins when both are set.

**Injecting a pre-built LM.** `configure(cfg, main_lm=…, sub_lm=…)` uses a supplied LM
verbatim instead of constructing one from `cfg`: a `dspy.utils.DummyLM` in tests, or a
cached / custom client in production. It's the public seam for a test double, so nothing
has to reach into private runtime state; read the active config back with `get_config()`.

**Claude-subscription auto-routing.** Set `RLM_MAIN_MODEL`/`RLM_SUB_MODEL` (or pass `main_model=`/
`sub_model=` on `RLMConfig`) to `claude_agent_lm.SUBSCRIPTION_PREFIX` + a model id: e.g.
`claude-agent-sdk/claude-sonnet-5`, and `configure()` builds a `ClaudeAgentLM` for that role
automatically; no explicit `main_lm=`/`sub_lm=` wiring needed. An explicit `main_lm=`/`sub_lm=`
kwarg still wins outright regardless of the model string. The prefix is only consulted for a role
left unset. This can raise, in addition to `configure()`'s own errors: `ValueError` for a bare
prefix with no model id; `RuntimeError` if `ANTHROPIC_API_KEY` is set (`ClaudeAgentLM` refuses to
start while it's set: the Claude Code CLI silently prefers it over subscription OAuth); or
`ImportError` with an install hint if the optional `rlm-harness[subscription]` extra isn't
installed. Needs the same setup as `ClaudeAgentLM` itself (see the module's own docstring):
building one by hand via `main_lm=ClaudeAgentLM(...)` still works exactly as before, for anyone who
wants to pass its other constructor kwargs (`timeout_s`, `cwd`, …) explicitly.

**Model names with a custom endpoint.** When `RLM_BASE_URL` is set, `configure` pins
litellm's `custom_llm_provider="openai"`, so the model names are the **plain id your
endpoint serves**: e.g. `qwen/qwen3-next`, not `openai/qwen/qwen3-next`. (dspy.LM runs on
litellm, which otherwise reads the first path segment as a provider and fails on a bare id;
the pin routes everything via the OpenAI wire protocol to your `base_url`.) A prefixed
`openai/...` name still works. With no base_url, write litellm's own prefix (`openai/gpt-4o`,
`anthropic/claude-...`).

`RLM_ADAPTER` (default `json`) picks how structured fields are coaxed out of the
model:

- **`json`** (default): schema-guided structured output, end-to-end. A lenient `JSONAdapter` always
  sends the `json_schema` response format directly: bypassing dspy's `supports_response_schema`
  gate, so no `litellm.register_model` poke is needed, and a brace-tolerant parse absorbs guided
  output that drops the outer `{ }`. Works on **any** structured-output endpoint: OpenAI-proper
  AND vLLM / NVIDIA NIM (which reject schema-less `json_object` but accept `json_schema`). The
  decoder enforces the schema, so it **yields valid output even from a weak / imperfectly-
  formatting model**. An EMPTY or whitespace-only completion is not absorbed: since 1.20.1 it
  raises, rather than being wrapped into the `{}` that dspy reads as an output field left unset,
  then filling that field from a default, a default factory, or an annotation admitting `None`.
  That reached a task only when EVERY output field was one dspy would supply: a single plain
  required field made the parse raise before 1.20.1 as well.
- **`chat`**: `dspy.ChatAdapter` with the JSONAdapter fallback **off**: never sends
  `response_format`. For an endpoint with **no** structured-output support. Needs the model to
  follow dspy's text field-marker format reliably. The fallback is off (dspy's stock ChatAdapter
  recovers via bare `json_object`, which vLLM rejects), so a model that drops a field has no
  recovery. Not as portable as it looks.
- **`default`**: leave dspy's stock adapter (ChatAdapter *with* the json_object fallback):
  recovers on OpenAI-proper endpoints, but the fallback is rejected by vLLM/NIM.

`RLM_MAX_TOKENS` (default `8192`) is the per-call generation cap. It defaults generous rather
than deferring to the server: a **reasoning model** emits its chain-of-thought before the answer,
so a server's small default cap (e.g. 1000) truncates the thinking and returns **empty content**.
Set `RLMConfig(max_tokens=None)` to defer to the server.

**The default is a floor, not a ceiling, and running into it looks like a different bug.** 8192 has
to hold the chain-of-thought AND the structured answer of the same turn. When a long turn overruns
it, the reply is cut off *mid-JSON*, so the adapter cannot parse what came back and the run dies as

```
RLMTaskError: Failed to produce a valid '<field>' after N attempts:
    AdapterParseError: ... failed to parse the LM response
```

The cause is in that message rather than only on `__cause__`, so a consumer that
catches `RLMTaskError` and prints `str(e)` shows it too. `error_chain` in the trace is unchanged and
still carries the full chain.

That is a **truncation**, not a model that cannot follow the schema, and it is easy to misdiagnose
as one: the text in the error often looks like well-formed output right up to where it stops.
Tell the two apart by reading the END of the quoted `LM Response`: a truncated one simply stops,
mid-string or mid-object, with no closing brace.

Two things make it more likely, and they compound: a **reasoning** model (the thinking is spent
before the answer starts) and a task whose one turn is genuinely long (assembling a large
structured result, or a REPL turn that both reasons and writes a big code block). More than one
consumer has hit this and settled on **16384** for the planner, which is a reasonable first move
when the symptom above appears.

It is not free, though, so raise it deliberately rather than reflexively: an OpenAI-compatible
server commonly validates `prompt_tokens + max_tokens` against the context window, so a bigger cap
removes usable PROMPT budget, and an RLM planner's prompt grows every turn, which is exactly the
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
promotes `reasoning_content` to the answer when `content` is empty (guarded: a well-behaved model's
`content` always wins, so its native thinking stays discarded), which keeps the root's first turn from
dying on dspy's "empty or null response" check. The native chain-of-thought is still dropped from the
trajectory either way, so a reasoning root spends extra tokens the trace won't keep.

### Per-role LM request parameters (`RLM_MAIN_LM_KWARGS` / `RLM_SUB_LM_KWARGS`)

A JSON object of extra `dspy.LM` kwargs, merged over what `configure()` builds **for that role
only** (`RLMConfig.main_lm_kwargs` / `sub_lm_kwargs`). Unset sends nothing. The default is
byte-identical to not having the field, down to putting no new key on the wire.

```bash
# bound a reasoning model's THINKING without touching the sub-LM's model
export RLM_MAIN_LM_KWARGS='{"extra_body":{"thinking_token_budget":16384}}'
```

**What it is for.** A reasoning model whose thinking runs away does not look like a long answer:
it looks like a parse failure. Measured on a consumer's vLLM deployment after a model swap: 53% of
attempts (33 of 62, as of one snapshot) had at least one call at the 32768 `max_tokens` cap, and
inside those calls the median `reasoning_tokens` was the ENTIRE budget with no content at all. One
such call kills the attempt with `AdapterParseError`, which reads as a model that cannot follow the
schema. Raising `max_tokens` does not help; it buys a longer runaway. A thinking ceiling does: the
same pages rerun with `thinking_token_budget: 16384` produced 0 of 155 calls at the cap, and the 15
calls that WERE cut at the budget each still produced a usable turn.

**The kit ships the mechanism and no vocabulary, on purpose.** The key is server-specific:
vLLM reads `thinking_token_budget`, Anthropic `thinking.budget_tokens`, llama.cpp has only a server
flag, OpenAI only `reasoning_effort`. A named kit knob would be a promise that one word means the
same thing everywhere, and the measurement that prompted this falsifies that inside a single model
family. On that model `reasoning_effort` moved reasoning the WRONG way (`low` 2443-2562 tokens and
`medium` 2778-2837 against 2237 at the default) while breaking the output-format instruction,
which is the last thing you want in front of a JSON adapter. So what you write is what reaches
the server, and the kit owns no name on the wire.

**Two levels, and the difference is where silent no-ops come from.** A TOP-LEVEL key is a litellm
parameter and goes through litellm's own per-provider mapping. `reasoning_effort` is one of these,
so the passthrough reaches it with litellm's semantics rather than the kit's. A key inside
`extra_body` is passed RAW into the request body with no mapping, which is what a server-specific
name needs. On that same deployment `max_thinking_tokens`, `thinking_budget`, `reasoning_budget`
and `chat_template_kwargs.thinking_budget` were all accepted and silently ignored.

**The limit worth stating: a passthrough cannot tell you the server dropped your key.** Nothing
here can, since an ignored key and an honoured one are identical on the wire. The check is after
the fact, in the trace: `run_end.payload.budgets.thinking` records what the LM carried, and
`run_end.payload.usage` carries the provider's own `reasoning_tokens`. A run supposedly capped at
16384 that reports 32768 of reasoning is the key doing nothing, and that comparison is available on
every run rather than in a one-off script.

**A budget set for one task binds on every task in the process, and that has been measured.** The
env var is read once per process, so a ceiling chosen for the task that runs away also applies to
tasks that never did. On the deployment above, the pre-budget baseline had 2 of 29 planner calls in
one task and 6 of 84 in another reasoning past 16384, all of them finishing well under the 32768
cap: ordinary long reasoning, not a runaway. A 16384 ceiling now cuts exactly that population.
Every cut call still produced a usable turn there (988-1413 tokens of content after the cut), and
whether the RESULT is worse is a question an ok rate cannot answer, so measure the output rather
than the completion. If it costs you, set the variable per task rather than per deployment, which
is available wherever a task gets its own process.

**Keys `configure()` owns are refused at parse time**, not silently dropped and not silently
winning:

| Refused | Because | Instead |
|---|---|---|
| `model` | `dspy.LM` takes it positionally | `RLM_MAIN_MODEL` / `RLM_SUB_MODEL` |
| `api_key`, `base_url`, `custom_llm_provider` | nothing records an override, so a silent one leaves a trace that reads like a run that went somewhere else | `RLM_API_KEY` / `RLM_BASE_URL`; a per-role ENDPOINT is a connection-identity change: build that role's LM and inject it with `configure(sub_lm=…)` |
| `timeout` | same, and it is `request_timeout_s`'s own spelling | `RLM_REQUEST_TIMEOUT` |

`max_tokens` is deliberately NOT refused, and the line is not "everything the kit sets". It is
whether the TRACE can see the override. `max_tokens` is read back off the LM into `budgets`
(`_dspy_compat.applied_lm_budget`), so a per-role cap is self-documenting, and that is how you get
one without a second config field:

```bash
export RLM_MAX_TOKENS=32768                    # the main model needs the room
export RLM_SUB_LM_KWARGS='{"max_tokens":4096}' # the sub model does not
```

**Quoting: there is no single form that survives every layer**, measured rather than reasoned:
one JSON value, four ways of getting it into the process:

| value as written in the file | `set -a; . ./file` (bash + zsh) | `docker run --env-file` | compose `env_file` | python-dotenv |
|---|---|---|---|---|
| `X={"extra_body": {"thinking_token_budget": 16384}}` | NOT SET: the space and `{` break the assignment | OK | OK | OK |
| `X={"extra_body":{"thinking_token_budget":16384}}` | UNPARSEABLE: the shell strips the `"` | OK | OK | OK |
| `X='{"extra_body":{"thinking_token_budget":16384}}'` | OK | UNPARSEABLE: quotes kept literally | OK | OK |

The two that disagree do so in OPPOSITE directions: a shell-sourced file needs the quotes and
`docker run --env-file` never strips them, so it receives them as part of the value. Compose's
`env_file` and python-dotenv both strip matching quotes and accept all three. **Single-quoted is the
right default**. It is correct everywhere except `docker run --env-file`, and compose is what
almost everyone actually uses. A double-quoted-and-escaped value is the trap to avoid: under
`docker run --env-file` it parses as a JSON *string* rather than an object, which is why a non-object
raises `TypeError` naming the variable instead of being quietly treated as empty.

## Testing the forward path offline (`rlm_harness.testing`)

Construction tests (`task._build_rlm()`) catch signature/kwarg drift but never run the loop: where
wiring bugs actually hide (a prompt naming a tool `foo` while it registered as `foo_tool` is a
`NameError` no construction test sees). `rlm_harness.testing` drives the **real `dspy.RLM.aforward` loop
offline**: no model, no Deno, no network:

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
    submit({"answer": {"x": 5}})]))                      # SUBMIT: terminates the loop, coerces the result
result = await task.arun(q="…")                          # the whole planner→tools→result chain, offline
```

`dspy.RLM` injects the run's real tools onto the scripted interpreter's `.tools`, so a `call(...)` step
runs the actual tool (its tracing records a genuine `tool_call`); a `dict`/`submit(...)` step SUBMITs.
The `interpreter=` kwarg is an injection seam (like `sub_lm=`): an explicit interpreter OBJECT overrides
`config.interpreter` and, like an injected `DummyLM`, bypasses `build_interpreter` and its guard, so it
is a test/advanced seam; the default string path keeps the guard. `rlm_harness.testing` imports dspy lazily,
so it doesn't affect `import rlm_harness`. `cancel_event=` (above) has NO effect when `interpreter=` is also
given: a caller supplying their own interpreter object owns its cancellation behavior too, exactly like
`ScriptedInterpreter` owns its own.
