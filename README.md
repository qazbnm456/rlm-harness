# rlm-harness

A clean, reusable harness for building **any task** on top of
[DSPy](https://dspy.ai)'s Recursive Language Model module (`dspy.RLM`).

RLMs ([Zhang & Khattab, MIT, arXiv:2512.24601](https://arxiv.org/abs/2512.24601))
let a model explore unbounded context by treating it as a variable in a sandboxed
Python REPL and recursively calling sub-LLMs over it. DSPy's `dspy.RLM` is the
first-party implementation (Khattab co-authored both DSPy and the RLM paper) — it
works with existing Signatures and is optimizer-compatible (GEPA/MIPRO). This kit
distills the boilerplate around it into one small, opinionated layer.

**rlm-harness is domain-agnostic** — anything `dspy.RLM` can do fits: multi-hop "deep
research", an RSS-digest agent that posts to a webhook, structured extraction,
detection authoring, you name it. Security happens to be the author's own first
use of it, but it isn't the kit's scope.

## Why this exists

Using `dspy.RLM` directly leaves you re-writing the same plumbing for every task:
model/sub-model config, a retry+validation loop, a sandbox choice, observability.
`rlm-harness` makes a task a *declaration*:

```python
from rlm_harness import RLMConfig, RLMTask, configure
from rlm_harness.tools import make_schema_validator
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
pip install rlm-harness
# or with uv:
uv add rlm-harness
```

`rlm-harness` needs Python ≥ 3.11
and pulls in `dspy` + `pydantic`; extras are opt-in — observability (`pip install "rlm-harness[observe]"`)
and running on a Claude Pro/Max subscription instead of an API key
(`pip install "rlm-harness[subscription]"` → `rlm_harness.ClaudeAgentLM`, injected via `configure(main_lm=…)`). A
*live* `dspy.RLM` run additionally needs model credentials (see the guide's
[Configuration](https://github.com/qazbnm456/rlm-harness/blob/main/rlm_harness/README.md#configuration)) and a
Deno sandbox — the logic and tests run without either. dspy requires Deno `>=2.0.0,<3.0.0`:
`brew install deno`, or let dspy manage it with `pip install "dspy[deno]"`.

## What's in the box

- **Tasks as declarations.** Subclass `RLMTask` — the retry+validation loop, sandbox
  selection, budget caps, and observability are inherited.
- **The whole trajectory, recorded.** `TraceRecorder` writes main steps, every sub-LM
  call, and every tool call into one append-only JSONL stream — replayable and
  exportable as SFT/RL datasets (reward-free: scoring belongs to your trainer).
- **The recursion seat, interceptable.** Every sub-LM escalation is traced as a
  `sub_call` automatically — no wrapper needed. `intercept_sub_lm` adds a deterministic
  validate/post-process pipeline on top; `model_as_tool` lets the main LM choose to
  consult another named model, in the trajectory.
- **Tools, the base/wrap way.** Pydantic/JSON-Schema validators, an SSRF-guarded
  `fetch_url`, provider-agnostic web search, the generic model-as-tool core, a
  `run_command` seam over your isolated runner, an MCP client bridge, and
  skills-as-tools progressive disclosure.
- **Sandboxed by default.** The pyodide/deno interpreter; the `local` interpreter is
  refused unless explicitly opted into; an opt-in Docker `container` interpreter for
  when the REPL itself needs real subprocesses.
- **Offline-testable.** `rlm_harness.testing` drives the real `dspy.RLM` forward loop
  with no model, no Deno, no network.

## Documentation — the guide

The deep documentation lives in
[**`rlm_harness/README.md`**](https://github.com/qazbnm456/rlm-harness/blob/main/rlm_harness/README.md):

- [Layout](https://github.com/qazbnm456/rlm-harness/blob/main/rlm_harness/README.md#layout) — what each module owns.
- [RLM as harness engineering](https://github.com/qazbnm456/rlm-harness/blob/main/rlm_harness/README.md#rlm-as-harness-engineering-sub-lm-hook--tracing) — the sub-LM hook + trajectory tracing.
- [Sub-LM vs. tool](https://github.com/qazbnm456/rlm-harness/blob/main/rlm_harness/README.md#sub-lm-vs-tool-which-model-goes-where) — which model goes where; the choice decides what your RL data records.
- [Skills](https://github.com/qazbnm456/rlm-harness/blob/main/rlm_harness/README.md#skills-progressive-disclosure), [MCP tools](https://github.com/qazbnm456/rlm-harness/blob/main/rlm_harness/README.md#mcp-tools-connect-an-external-mcp-server), [running local commands](https://github.com/qazbnm456/rlm-harness/blob/main/rlm_harness/README.md#running-local-commands-an-isolated-runner), and the [container interpreter](https://github.com/qazbnm456/rlm-harness/blob/main/rlm_harness/README.md#environment-interpreter-interpretercontainer) — the tool & environment surfaces.
- [Grounded completeness](https://github.com/qazbnm456/rlm-harness/blob/main/rlm_harness/README.md#grounded-completeness--the-sufficiency-critic-recipe) and [judgement-only SUBMIT](https://github.com/qazbnm456/rlm-harness/blob/main/rlm_harness/README.md#judgement-only-submit--assemble-facts-dont-let-the-policy-report-them) — the rollout conventions.
- [Building a consumer](https://github.com/qazbnm456/rlm-harness/blob/main/rlm_harness/README.md#building-a-consumer) — the five-step extension contract.
- [Configuration](https://github.com/qazbnm456/rlm-harness/blob/main/rlm_harness/README.md#configuration) — every env var, adapter selection, model naming.
- [Testing the forward path offline](https://github.com/qazbnm456/rlm-harness/blob/main/rlm_harness/README.md#testing-the-forward-path-offline-rlm_harnesstesting) — the scripted offline harness.

## Built with rlm-harness

Real projects using rlm-harness as their RLM scaffold:

- **[ctx-distillery](https://github.com/qazbnm456/ctx-distillery)**: distils an AI coding agent's
  session transcripts and memory store into a judgement-only distillation plan — what to prune,
  cross-reference, or promote into durable memory or a reusable Skill. It proposes; it writes nothing.
- **[cve-reverser](https://github.com/qazbnm456/cve-reverser)**: reverses publicly disclosed CVEs from
  their patches into local-lab PoCs and Nuclei detection templates. A traced, trainable RLM harness.
- **[diff-sentry](https://github.com/qazbnm456/diff-sentry)**: classifies GitHub changes (PRs, issues,
  pushes) for malicious intent — the diff is read as untrusted data in the sandboxed REPL, emitting
  evidence-backed benign / suspicious / malicious verdicts into a SIEM.
- **[toolscout](https://github.com/qazbnm456/toolscout)**: an ATLAS-style rollout harness — a small
  planner progressively discovers a large MCP toolspace and computes over tool results as code, emitting
  reward-free trajectories + per-criterion facts for a downstream trainer.

Built something on rlm-harness? Open a PR to add it here.

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

## Develop

```bash
uv sync --group dev
uv run pytest          # logic tests (no live LLM needed)
```

Tests cover config parsing, the retry/validation engine, the sandbox guard, the
tools, the sub-LM-hook/trace/replay/dataset layer, and a real-`dspy.RLM`
construction check (dspy-bearing tests use `DummyLM` or skip if dspy is absent).
A *live* run additionally needs real credentials and a Deno sandbox
(`brew install deno`, or `pip install "dspy[deno]"` for dspy's managed binary; it requires Deno
`>=2.0.0,<3.0.0`); `examples/mini_run.py` shows it. To drive the real forward
loop offline (no model, no Deno), see the guide's
[Testing the forward path offline](https://github.com/qazbnm456/rlm-harness/blob/main/rlm_harness/README.md#testing-the-forward-path-offline-rlm_harnesstesting).
See `CLAUDE.md` for invariants when modifying the kit.

## Status

**v1.3.0** — the current release. Scaffold + harness-engineering layer (sub-LM hook,
skills-as-tools, trajectory recording, replay, dataset export, harness delegation on both the client
and server side), plus fast-failing non-retryable LM errors and a filesystem/process layer: reading,
searching, writing and editing a bounded directory, safe git clone and archive extraction, an
isolated-subprocess primitive, and deterministic quote grounding. Hardened by dogfooding against real
downstream consumers; the changes that surfaced are in
[`CHANGELOG.md`](https://github.com/qazbnm456/rlm-harness/blob/main/CHANGELOG.md).

1.0.0 means the public surface is a contract: `__init__.__all__`, the `rlm-harness/trace/v1` wire format,
and `RLMTask`'s declaration fields are frozen under
[SemVer](https://semver.org/) and pinned by `tests/test_contract.py`. Additions ship in a minor
release; a rename or removal ships with an alias and a `DeprecationWarning` first, and the removal
itself waits for the next major. The trace format carries its own version and evolves
additive-only within v1. A `_`-prefixed name or module internal is not part of that promise.

Next: enable `optimize.compile_task` against a labelled trainset to actually
GEPA-compile tasks (currently a documented stub).

## License

MIT © Boik Su ([@boik_su](https://x.com/boik_su)). See [`LICENSE`](https://github.com/qazbnm456/rlm-harness/blob/main/LICENSE).
