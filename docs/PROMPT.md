# The prompt: who owns what reaches the model

Read this before writing anything a model will read: a task's `instructions`, a tool's docstring,
a skills manifest, an MCP server's tool description, or a metric for an optimizer that rewrites
all of them.

Every rule here was measured by running dspy rather than inferred from prose, including the parts
upstream states plainly. **dspy's RLM guide does document the central fact**, in one sentence:
*"The task instructions, output-field descriptions, and tool docs are formatted into the action
prompt when you construct the module, which is the surface an optimizer tunes."* Its API page adds
that those instructions are what *"DSPy adapters place in the system prompt"* and that
*"Optimizers such as GEPA may therefore adapt the execution guidance along with the rest of the
action policy."*

So the honest version of how this file came to exist is not that upstream hid anything. It is that
the sentence sat in a guide nobody here had read, while the kit shipped docstrings written for
Sphinx into a system message for every release from its first. What upstream genuinely does not
say is the mechanism: no page states that the text comes from a tool's `__doc__`, and no release
note has ever mentioned it. That gap is one line of code wide, and it was enough.

Where a number appears, [Checking these numbers yourself](#checking-these-numbers-yourself) gives
the command for it or says plainly that it needs a task you would have to build.

## One flat string, and most of it is yours

`dspy.RLM` builds the planner's instructions by concatenation (`dspy/predict/rlm.py`, the
`action_sig` assignment):

```python
dspy.Signature({}, task_instructions + ACTION_INSTRUCTIONS_TEMPLATE.format(...) + tool_docs)
```

That string becomes the **system message**. Measured on five real consumer tasks, built offline
against a mock interpreter:

| task | total | your `instructions` | dspy's fixed words | interpolated | tool descriptions |
|---|---|---|---|---|---|
| 1 tool | 4972 | 42.5% | 35.8% | 17.5% | 4.3% |
| no tools | 6819 | 34.8% | 26.1% | **39.1%** | 0.0% |
| 2 tools | 10477 | 41.7% | 17.0% | 9.4% | **31.8%** |
| 2 tools | 6712 | 47.7% | 26.5% | 13.3% | 12.6% |
| 2 tools, skills | 7571 | 47.7% | 23.5% | 23.6% | 5.3% |

Four things that table is worth reading for.

**dspy's own words are a constant, 1780 characters.** Identical on every task and every dspy from
3.3.1 on. Its *share* moves (17.0% to 35.8% above) only because your text around it grows. Measure
the share on your own task before deciding it matters.

**Between half and three quarters of the string is text you originated.** Your `instructions` plus
the part dspy renders from your signature, output model and config. The surprise in this file is
not that dspy takes up room; it is that so much of the rest arrives without anyone writing it as
a prompt.

**A rich output model is prompt cost that nothing counts.** In the task with no tools at all the
interpolated column is the largest block, and almost all of it is `{output_fields}`: dspy's
rendering of the signature and `output_model`. Measured on that row: 2669 characters interpolated,
of which **2515 (36.9%) is schema-derived** and 154 is the interpreter's own
`execution_instructions`. A nested pydantic model with long `Field(description=...)` strings pays
for itself on every turn of every run, and it never looks like a prompt at the call site. The same
goes for a skills manifest folded into `instructions`: in the last row that is 1224 characters,
16.2% of the whole message.

**Read the interpolated column with its apparatus in mind.** Those 154 characters are the test
double's `execution_instructions`; a `container` or `pyodide` run puts different text there, so
the column moves with how you built the task, not only with your schema. The snippet at the end
of this file is what injects it.

**Their tool descriptions carried no Sphinx markup; they did carry collapsed whitespace.** Across
the consumer-owned tools in those five tasks, zero had `` ``literals`` `` or roles, and every one
produced mid-sentence gaps after dspy's flatten, the worst losing 129 characters to normalisation
across 32 separate runs of spaces. So the markup half of the defect really was the kit's alone,
and the whitespace half was everybody's. That is the argument for normalising at the kit's build
seam instead of writing a style rule for consumers: a rule would have fixed the half they were
already getting right.

Two consequences of the ordering, and the second is the useful one.

**Your text does not arrive as "the system prompt".** It arrives as the opening of a longer string
whose middle you did not write. A rule you state can be followed by a dspy rule that points the
other way, and the model sees both.

**You own the first word and the last word; dspy owns the middle.** Tool documentation is appended
*after* dspy's protocol. So the only place a consumer can have the last word on a dspy rule is a
tool docstring. That is an odd place for policy, and it is where the position actually is.

## A tool's docstring is prompt text

`dspy.Tool` takes its `desc` from the function's `__doc__` with no opt-in, and `RLM` renders one
line per tool into the block above, where it reached 31.8% of one real task's instruction string.
Verified directly: a throwaway function whose docstring reads `A docstring that is really prompt
text.` comes back as exactly that in `dspy.Tool(fn).desc`.

This is not a dspy invention. Deriving a tool description from a docstring is what every Python
agent framework does. What is specific to `dspy.RLM` is that the descriptions are **prose in the
system message** rather than a structured `tools=[...]` API parameter, because RLM's tools are
functions inside a Python REPL, not API-level tool calls. dspy's own header for the block says so:
`Additional tools available (use these instead of standard library equivalents)`.

**dspy flattens each description onto one line**, and has since the first version that had an RLM
at all:

```python
# Get description with newlines escaped
desc = (tool.desc or "No description").replace("\n", "  ")
```

Three things follow, all of which this kit got wrong first.

**Write the inner closure's docstring for the model, the factory's for the developer.** Only the
inner one reaches the prompt. Three of the kit's tools had their whole description on the factory
and nothing on the closure, so dspy rendered the literal words `No description` into the prompt
for a tool it was offering the model. A bare public function handed over directly has no factory
to hide behind: its developer half belongs in the module docstring.

**Say what a caller needs and stop.** Two of the kit's tools shipped 6936 and 3816 characters of
design rationale and implementation notes: 52.2% and 28.7% of the 13280 characters of
model-facing text across the fourteen tools the kit ships, 81% between them,
re-sent in the system message on every turn of every run. Keep the parameters, the return shape,
and the sentinel strings a model has to branch on. Drop the reasoning. When a tool seems to need
more than its baseline to be *used* correctly, that surplus is developer-facing by definition.

**Never write a number that a factory argument can change.** A docstring stating a default while
the build enforces something else is a wrong instruction, not a stale comment. Build it from the
argument instead.

You do not have to format any of this by hand. `RLMTask._build_rlm` normalises tool descriptions
on the way to the model: Sphinx roles and inline literals become plain text, and the indentation
Python put in your docstring stops arriving as a ten-space gap mid-sentence. It rewrites only
objects the kit owns, never your function.

**What it reaches, and what it does not.** A plain function or a bound method is covered, which is
every tool the kit ships and almost every tool a consumer writes. A `dspy.Tool` is covered through
a copy, which is how `mcp_tools` hands over a server's description. Not covered, and silently so:
a `functools.partial`, a callable class instance, a coroutine function, a `staticmethod` object,
an `lru_cache`-wrapped callable. Those are the shapes the timing seam skips too, so there is no
kit-owned object to write on. If you wrap an MCP tool yourself, wrap it as a function.

One mechanism detail, since this file is pointed at by `AGENTS.md`: on the MCP path the kit
normalises the `desc` that dspy actually reads, and leaves the raw copy on the callable's
`__doc__` alone. Nothing in dspy or the sandbox stub reads that copy today.

What normalisation cannot do is shorten anything or invent a description you did not write;
`tests/test_tool_descriptions.py` and `tests/tool_description_baseline.json` are what catch those.
Growing a description on purpose is allowed and costs one line in a reviewed diff.

## dspy's REPL protocol is not replaceable

`ACTION_INSTRUCTIONS_TEMPLATE` is a module-level constant, interpolated when `RLM` is constructed.
No constructor argument turns it off or replaces it. It contains the `Available:` list, the
`IMPORTANT: This is ITERATIVE` paragraph, six numbered operating rules (`EXPLORE FIRST`, `ITERATE`,
`VERIFY BEFORE SUBMITTING`, `USE llm_query FOR SEMANTICS`, `MINIMIZE RETYPING`,
`SUBMIT ONLY AFTER SEEING OUTPUTS`) and the closing
`You have max {max_llm_calls} sub-LLM calls. When done, call SUBMIT() with your output.`

It has been there since the module existed, not added by a later dspy:

| dspy | `dspy.predict.rlm` | `ACTION_INSTRUCTIONS_TEMPLATE` | `{interpreter_rules}` |
|---|---|---|---|
| 3.1.0 | absent | absent | absent |
| 3.1.2 | **present** | 1902 chars | absent |
| 3.2.0 | present | 1902 chars | absent |
| 3.3.0 | present | 1902 chars | absent |
| 3.3.1 | present | 1865 chars | **present** |
| 3.4.0 | present | 1865 chars | present |
| 3.4.1 | present | 1865 chars | present |

The module arrives in **3.1.2** (there is no 3.1.1), already carrying the whole template. 3.2.0 is
where the notes first describe RLM at length, under the heading "Hardened RLM and
PythonInterpreter", which is a different release from the one that shipped it: easy to fuse into a
single wrong sentence, and worth keeping apart.

**The one seam is `{interpreter_rules}`**, filled from the interpreter factory's
`execution_instructions` attribute and rendered under an `Execution environment:` heading. It
arrived in 3.3.1, which is why the kit's floor moved to `dspy>=3.3.1` at the time; the declared
bound is now `>=3.4.0,<3.5.0`. Before 3.3.1 there was no seam at all. The kit already feeds it
from the live interpreter, so a `container` run is described as a container rather than as Pyodide.
It is the only sanctioned place to say something about the environment from the host side.

**Know where it lands, though: index 288 of the 1865-character template**, under
`Execution environment:`, which is 538 characters BEFORE the first of the six rules and about
1300 before the last. So it is in the middle of dspy's text, not after it. For saying what the
environment is, that is the right place. For contradicting a rule, it is the weaker of the two
seams, and a tool docstring is the stronger one.

No release note introduces the template, and none explains why it is worded as it is. The
documentation covers the REPL's purpose (*"Instead of feeding huge contexts directly into the
prompt"*, *"treats context as external data that the LLM examines via code execution"*) and, as
quoted at the top of this file, says where the action prompt ends up and that an optimizer tunes
it. What it does not do is justify the six rules, or connect them to a docstring.

### Three conflicts, reproduced

**A hardcoded sub-LM capacity, argued on the wrong axis.** The `Available:` list advertises
`llm_query(prompt) - query a sub-LLM (~500K char capacity) for semantic analysis`. That number is
a literal, identical on every dspy version above, and unrelated to the `sub_model` you configured.
Two consumers independently tell the model, in their own fetch tools' descriptions, to slice what
it needs and **not** pass bulk text to `llm_query`. Their reason is cost: `sub_model` is commonly
pointed at a pricier third-party API than the planner's own model, so bulk text sent to the
lifeline is money. dspy is advertising exactly what they forbid.

**The two statements are not even about the same thing**, which is what makes this worth a rule of
its own. dspy's is a *capability* claim and carries a number. The consumer's is a *cost* policy and
carried no reason, because its author had no idea anything would be arguing the other side. A model
reading both sees a quantified permission and an unexplained prohibition.

**So state the reason.** When your instruction contradicts something in dspy's protocol, assume the
model has already read dspy's version, and say why yours wins: *"`llm_query` can take very large
inputs, but each call bills a more expensive model, so slice first"* beats *"do NOT pass bulk text
to `llm_query`"* while being the same instruction. A prohibition with a reason competes; a bare one
loses to a number. And because the capacity claim sits in dspy's protocol while your tool
docstring lands after it, yours does at least get the last word.

**A sub-LM budget of zero still advertises the sub-LM.** With `RLM_MAX_LLM_CALLS=0` the prompt
reads:

```
- `llm_query(prompt)` - query a sub-LLM (~500K char capacity) for semantic analysis
You have max 0 sub-LLM calls. When done, call SUBMIT() with your output.
```

A tool is offered and withdrawn in the same message.

**Preconditions on SUBMIT.** dspy's rule 6 already carries one, in its title and its body:
`SUBMIT ONLY AFTER SEEING OUTPUTS - SUBMIT ends the current run immediately. If you need to
inspect printed output, run it in one step, review the result, then call SUBMIT in a later step.`
A consumer adding a different precondition (one pins the sentence
`Never SUBMIT while the last generation was format-INVALID` in its own tests) is not contradicting
that, but it is stating its rule from the position *before* dspy's six rather than alongside them,
which is a weaker position than it looks.

**What to do about all three.** A tool docstring is the only consumer-owned text that lands after
dspy's rules, so a correction that has to win an argument goes there. `execution_instructions` is
the sanctioned host-side seam but sits before them, so use it for what the environment IS rather
than for overruling a rule. Do not monkeypatch the module constant: it is global to the process,
and it silently changes every other task in the same run.

## Skills and MCP descriptions land in the same string

Two more paths feed the one system message, and both are easy to forget because neither looks like
a prompt at the call site.

**Skills.** `render_skills_manifest` produces text you fold into `RLMTask.instructions`, and
`load_skills_as_tools` hands over `list_skills` and `read_skill`, whose docstrings are prompt text
under the rules above. `read_skill`'s is reassigned per discovery mode, because what it should
tell the model differs between them; `list_skills` keeps an ordinary one. Both are plain prose
already, which is what you want.

**MCP.** `rlm_harness.mcp` takes the server's `description`, assigns it to the generated
callable's `__doc__`, and passes it to `dspy.Tool(..., desc=...)`. So an **external server's prose
becomes part of your system message**. dspy reads `desc` there and ignores `__doc__`, so that is
the one the kit normalises; the raw copy stays on the callable, where nothing reads it. Two things
follow: a verbose MCP server costs you prompt on every turn of every run, and the text is not
yours to edit. If you need it shorter or different, wrap the tool rather than hoping the server
changes.

## GEPA: what it optimises, and what it costs

`dspy.GEPA` is an evolutionary optimizer over **text**. It captures full execution traces, isolates
the slice belonging to one predictor, reflects on it, and proposes a replacement instruction.
Mutation is wholesale: the candidate's string replaces the predictor's instructions outright.

Five facts worth having before you plan a run.

**An `RLM` has exactly two optimizable predictors**, `generate_action` (the REPL planner) and
`extract`. There is no finer seam: GEPA rewrites the whole flat string from the table at the top of
this file, or none of it.

**So a candidate can silently delete dspy's protocol.** Verified offline: replacing the planner's
instructions removes `SUBMIT ONLY AFTER SEEING OUTPUTS` along with everything else, and dspy repairs
only its `interpreter_rules` fragment, only on the path where the runtime differs from
construction. Loud damage is self-correcting, since a candidate that drops the SUBMIT contract
scores near zero and dies. Quiet damage is the hazard: a candidate that keeps SUBMIT and weakens
`VERIFY BEFORE SUBMITTING`, or drops one tool's description, can score *better* on a small
trainset. Diff a compiled program's instructions against a freshly built baseline and check that
the text you depend on is still there.

**One metric call is one full rollout**, with a real sandbox and real sub-LM calls. dspy's own
`auto_budget` resolves `auto="light"` on a two-predictor program to 790 metric calls at a
10-example valset, 850 at 20, and 1030 at 50, dominated by a `minibatch_size` of 35 that `auto`
hardcodes. Pass `max_metric_calls=` instead of `auto=` for anything exploratory. Do not point a
compile at a subscription-backed LM adapter: a few hundred rollouts is a batch eval, which is what
such an adapter exists not to be.

**Labels are optional; a useful metric is not.** GEPA's only check on the trainset is that it is
non-empty, so an inputs-only trainset compiles. What you optimise is then decided entirely by the
metric. Label-free metrics that work: a schema validator, a host-side quote verifier against the
real source, declined tool calls, budget exhaustion. Returning
`dspy.Prediction(score=..., feedback=...)` is where the leverage is, since the feedback text is
what the reflection LM reads; a bare float gets you a synthesised *"This trajectory got a score
of N."*

**A saturated metric has no gradient, and looks exactly like a working one, and re-pointing it
is cheap.** A study on this
repository scored a 10-example trainset at a mean of **1.000** on the shipped instructions, so the
budget was never spent. The same runs varied **3.3x in turns and 13.5x in wall clock** at identical
scores. The headroom was real; the metric was pointed at the wrong axis.

Re-pointed, it works: correctness as a hard GATE that scores zero whatever the run cost, turn
count as the gradient, and the same ten examples give a mean of **0.700** spread from 0.30 to
1.00, with a held-out set at 0.513. So measure your baseline before you buy an optimizer run, keep
the gate that refuses to spend at the ceiling, and put correctness above the gradient rather than
beside it: an instruction that reaches a wrong answer faster must score zero, or that is what you
will get. Relatedly, do not aim a metric at
`fence_refused_turns`: `metrics.py` records why that counter is not prompt-fixable, and a consumer
already spent two prompt generations suppressing content its own output needed.

**There is no return path yet.** `RLMTask` takes no compiled program, and `_build_rlm` rebuilds the
signature from `self.instructions` on every run, so GEPA's output has nowhere to go inside the
kit's run path today. `optimize.save_program` / `load_program` act on a dspy program the kit's own
`arun` never uses. Compile and measure if you want the number; wiring the result back is still
open work.

## Checking these numbers yourself

Nothing here should be taken on trust, including the parts that are currently true.

**Who owns how much of YOUR task's instructions.** The one measurement worth running on your own
code, because the shares in the table above are other people's:

```python
import re, rlm_harness
from dspy.predict.rlm import ACTION_INSTRUCTIONS_TEMPLATE as T
from rlm_harness.testing import ScriptedInterpreter

rlm_harness.configure(rlm_harness.RLMConfig(
    main_model="openai/x", sub_model="openai/x", api_key="k",
    interpreter="mock", max_retries=1))
task = YourTask(interpreter=ScriptedInterpreter([]))
s = task._build_rlm().generate_action.signature.instructions
own, fixed = len(type(task).instructions or ""), len(re.sub(r"\{\w+\}", "", T))
i = s.find("\nAdditional tools available")
tools = (len(s) - i) if i > 0 else 0
print(f"total {len(s)} | yours {own} | dspy {fixed} | interpolated "
      f"{len(s) - own - fixed - tools} | tools {tools}")
print(s)      # read it. Nobody had.
```

**Every shipped tool's model-facing description, and its share.** Run as-is this prints the
CURRENT catalogue, with nothing empty; `tests/tool_description_baseline.json` holds the same
numbers and a test compares them to the code on every run. The 13280, the 52.2% and 28.7%, and
the three empty descriptions are the BEFORE column, which needs the older checkout in the
paragraph after this one. `tests/test_tool_descriptions.py` builds the same table:

```bash
uv run --python 3.12 --group dev --extra mcp --extra grep python -c "
import sys; sys.path.insert(0, 'tests')
from test_tool_descriptions import _build_every_shipped_tool
from rlm_harness._toolname import model_facing_description as m
import tempfile
with tempfile.TemporaryDirectory() as r: built = _build_every_shipped_tool(r)
sizes = {k: len(m(v.__doc__)) for k, v in built.items()}
tot = sum(sizes.values())
for k, v in sorted(sizes.items(), key=lambda kv: -kv[1]):
    print(f'{k:<22}{v:>6}{v/tot:>8.1%}')
print(f'{\"TOTAL\":<22}{tot:>6}')"
```

To get the BEFORE column, run the same thing against an older checkout:
`git archive <ref> | tar -x -C <dir>` and put that directory first on `PYTHONPATH`. Do not stash.

**What a given dspy version's protocol says, and where its one seam lands.**

```bash
uv run --no-project --with "dspy==3.4.1" python -c "
import re
from dspy.predict.rlm import ACTION_INSTRUCTIONS_TEMPLATE as T
print(len(T), 'total,', len(re.sub(r'\{\w+\}', '', T)), 'fixed words')
print('{interpreter_rules} at', T.find('{interpreter_rules}'), 'vs rule 1 at', T.find('1. EXPLORE'))
print(T)"
```

**What a GEPA budget really costs.** `AUTO_RUN_SETTINGS` alone does not give the numbers in this
file; the resolver does:

```bash
uv run --python 3.12 --group dev python -c "
from dspy.teleprompt.gepa.gepa import GEPA, AUTO_RUN_SETTINGS
print(AUTO_RUN_SETTINGS)
for v in (10, 20, 50):
    print(v, GEPA.auto_budget(GEPA, num_preds=2, num_candidates=6, valset_size=v))"
```

**Which message role carries it.** Subclass dspy's `DummyLM`, record the `messages` it is handed,
and run one scripted turn: `tests/` does this, and `DummyLM.history` stays empty, so recording in
`__call__` is the only way to see the prompt.

**What is NOT reproducible from this file.** The 8938-character system message and its 448-char
user counterpart, and the before-and-after instruction strings, come from a two-tool task written
a one-off study rather than shipped here, so a reader would have to build an equivalent. The GEPA
study's own figures (a mean of 1.000, 3.3x turns, 13.5x wall clock) came from a live run whose
script is not in the repo. Both are labelled where they appear.

Related: [`INVARIANTS.md`](INVARIANTS.md) for the rules that fire when you edit the kit,
[`CONVENTIONS.md`](CONVENTIONS.md) for versioning and the consumer loop, and
[`../rlm_harness/README.md`](../rlm_harness/README.md) for the guide.
