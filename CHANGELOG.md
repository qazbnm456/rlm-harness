# Changelog

All notable changes to `rlm-harness`. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/). Versions track
`rlm_harness/__init__.__version__` and `pyproject.toml` (kept in sync).

## [1.8.3] - 2026-09-01

Every tool a task hands the model now records how long it took, without its author doing anything.
Two documented rules are reversed to make that true, and both reversals have the same cause.

### Fixed

- **`duration_s` existed only where its author remembered, which is the `sub_call` failure one
  field over.** Six of the kit's tool sources measured themselves; the filesystem and knowledge tools —
  27 `record_tool_call` sites across `fs.py`, `edit.py`, `archive.py` and `skills.py` — did not. So
  `metrics.compute_tool_waste`'s `*_seconds` read `None` for them everywhere, and `ToolWaste` is
  explicit that `None` means "nothing measured" rather than zero.

  **A consumer could not fix it either.** One whose `read_file` / `grep_repo` / `read_skill` are
  pure delegation to these factories has no seam of its own; wrapping the callable to add a
  duration would emit a SECOND `tool_call` and double `tool_calls`, `tool_ok` and everything
  derived from them. It was right to refuse, and the only place to fix it was here.

  `RLMTask._build_rlm` now wraps every tool it hands the model — the same seam and the same reason
  as 1.7.0's automatic `sub_call` wrapper. The wrapper publishes a start time and **records nothing
  itself**, so it cannot double-count; `record_tool_call` fills `duration_s` from it only when the
  caller passed none. A tool that measures itself keeps its own figure, because it scopes the window
  more tightly — `fetch_url` starts its clock after the SSRF check, `run_command` keeps a
  runner-reported number alongside. A consumer's own tools are covered with no work.

  **What it deliberately does not reach**, each failing back to an absent field rather than a wrong
  one. A `dspy.Tool` OBJECT is passed through untouched — `mcp._make_tool` returns one, and it is a
  pydantic model with no `__name__`, so wrapping it would leave the wrapper called `timed`: two MCP
  tools would abort the task with "Duplicate tool name", one would register as `timed` with its
  args collapsed to `{"kwargs": {}}`. MCP records its own duration anyway. A coroutine function is
  passed through too, because dspy branches on `inspect.iscoroutinefunction`, which does not follow
  `__wrapped__` — deleting that one line turns a run that completes into `RLMTaskError: You are
  calling __call__ on an async tool`. So are a callable class instance and a `functools.partial`,
  neither of which `functools.wraps` can wrap without changing what dspy registers. And a GENERATOR FUNCTION -- one whose body
  contains a `yield` -- is wrapped like any other yet records nothing: calling it only builds the
  generator object, and the wrapper releases the start time before the body, and the
  `record_tool_call` inside it, ever runs. (A plain function that merely RETURNS a generator
  expression is a different shape and is timed normally; the distinguishing word is *function*,
  not *returns*.)

  Note the `dspy.Tool` passthrough is safe for MCP because MCP records its own duration; a
  `dspy.Tool` a CONSUMER builds does not, and must pass `duration_s` itself.

  **The fill is matched on the tool's name, and that is load-bearing.** Only what the task hands
  the model is wrapped, so a COMPOSITE tool — a consumer's tool calling a kit tool inside itself —
  would otherwise charge its whole window to every event recorded beneath it. Measured before the
  check existed: two zero-cost `read_file` calls inside a 0.25 s tool each reported 0.25 s, which
  triples `compute_tool_waste.total_seconds`. That is worse than the `None` it replaces — `None` is
  an honest unknown, that was a confident wrong answer — so a mismatch fills nothing.

  Applied at the task seam rather than inside each factory, which is safe for the annotations
  DESPITE the distance rather than because of it: every `tools/*.py` uses
  `from __future__ import annotations`, so the annotations `functools.wraps` copies are strings
  that only resolve in the defining module — and they survive only because `typing.get_type_hints`
  walks `__wrapped__` to find those globals. A factory-local wrapper would have resolved them
  trivially; this one depends on a CPython behaviour, which is why it is tested rather than assumed.
  Verified against dspy 3.3.1 on the 3.11 floor, where a `Tool` construction failure would abort
  registration for every tool on the task, not just the wrapped one.

### Changed

- **A refused call now carries a duration.** It used to record none, on the argument that a blocked
  URL never touched the network so a ~0 would be noise. `None` means "nobody measured", so spending
  it on "measured, and it was instant" makes the two indistinguishable — the mistake 1.7.0 shipped
  a release to correct. The reversal is stated in the guide, in `make_fetch_tool` and
  `make_web_search_tool`, and in the test that used to pin the old rule.

- **`grep_files` is no longer exempt from timing, reopened by its own terms.** Its exemption rested
  on a measurement — n=146, median 0.029s, max 0.746s — and named "a pathological regex over a
  large tree" as what would reopen it. Re-measured on a consumer deployment across nine real
  repositories, 7 patterns x 3 runs each: **median 744 ms, p95 4.8 s, max 6.3 s on a 2,110-file
  repository.** It does not scale with file count — a 102-file repo measured slower than a
  1,210-file one — so the driver is bytes and match count. Against that corpus the tool alone is
  about 40% of all sandbox execution time. The old number was not wrong; it was taken on a corpus
  with no large repository in it.

- **Do not average a `compute_tool_waste` figure across this upgrade.** A tool that reported `None`
  before reports a real number after, so a corpus spanning the boundary mixes "unmeasured" with
  "measured" in the same denominator — the same warning the 1.7.0 `sub_call` note carries, and
  `run_start.rlm_harness` is what separates the cohorts.

## [1.8.2] - 2026-09-01

One correctness fix in shipped code. It is model-visible: dspy builds a tool's description from
`func.__doc__`, and `verify_quote` is registered directly as a REPL tool, so its refusal text and
its docstring reach every consumer's model on every call.

### Fixed

- **A quote carrying only a line number verified against almost anything.**
  `make_read_file_tool(line_numbers=True)` renders a line as `f"{n:>6}\t{line}"`, so a BLANK line
  renders as `"     7\t"` — and `"     7\t".strip()` is `"7"`. Non-empty, so it passed the
  empty-quote guard, and the search then reduced to the bare pattern `7`, matching any source
  containing that digit:

      verify_quote("x = 42\n\ny = 1\n", "     2\t")
      -> MATCH: found at line 1 (char 5)

  A citation of nothing verified, at a line the citation never claimed. The guard's own stated
  reason for existing — "trivially matches almost any real text, a meaningless confirmation, not a
  real check" — describes that case exactly and did not fire on it.

  The guard now covers a second shape: a quote whose every non-blank line is a bare number, refused
  before any search, with its own message rather than the empty-quote one. The rule reads the whole
  line loosely on purpose, because the render is not what arrives — a model trims the trailing tab
  (`"     2"`), writes a space for it (`"     2 "`), or keeps the newline the renderer emits
  (`"     2\t\n"`), and a tighter pattern catches none of those. It matches ASCII digits only;
  `\d` accepts fullwidth and Arabic-Indic numerals, which are content here, not coordinates.

  It encodes no line-number format, so `grounding.py` stays independent of the tools that render
  one. `edit_file`'s success snippet uses the same convention and was never documented as a source
  of guttered text; it is now.

  **This closes the fully-blank case, not the whole class.** A guttered quote that carries content
  is not refused, and the gutter NUMBER is then searched as literal content — so the quote matches
  wherever that number happens to precede the line's text, including across a mandatory `\s+` that
  spans blank lines. A full line of code is reachable that way, not only a line of punctuation.
  From this repo's own suite: a quote claiming line 42 of `tests/test_async.py` verifies at line
  39, because line 39 ends `== 42` and the pattern becomes `42\s+def\s+test_run_...`. Measured by
  rendering every non-blank line of every `.py` here and verifying it against its own file, that is
  2 false matches in 17,412 quotes, about 1 in 8,700.

  Closing them needs the gutter-stripping repair, which is deferred: a position-checked design was
  built and audited, and it splits lines differently from the renderer in a way that verifies
  FABRICATED citations on any file containing a form feed. That needs its own release.

  **What this costs.** An all-digit quote no longer verifies even when the digits are genuinely in
  the source — `verify_quote("port = 8080", "8080")` is refused, and so is a multi-line quote whose
  every line is a number, such as a column lifted from a numeric file. For a short number that is
  the point, since the match was never evidence of anything. For a long one it is a real loss: an
  18-digit identifier appearing verbatim IS strong evidence, and it is refused too. The rule is
  blunt on purpose — it cannot tell a coordinate from a datum, and the failure it prevents is worse
  than the one it causes. It was 0 of 1,363 citations in a consumer corpus.

### Docs

- **Line numbers and `verify_quote` are complementary, not alternatives** — a consumer read them as
  a choice, turned numbers off to keep verification passing, and paid for it: 59 of 411 stored
  citations quoted the text verbatim at the wrong line. Turning them back on and re-running the
  same task moved coordinate corrections from 21.4% to 0.0% (15 of 70, then 0 of 39; P = 8.2e-05
  against the prior rate). Read that as a proportion, not a paired experiment — the second run
  planned a different outline, so it is 11 artifacts against 6 and the citation counts differ. The
  rule is which string goes where — the rendered text to the model, the raw file to the verifier —
  and the guide now says so under its own heading.

- **The README's Status section no longer restates the current release.** It had fallen five
  versions behind while claiming to describe the current one. It now points at the Releases page
  and this file and keeps only what does not change with a version number. That file is the PyPI
  long description, so the stale text was the first thing a reader saw.

## [1.8.1] - 2026-08-30

Documentation and test only. No behaviour change, no API change — every 1.8.0 call path is
byte-identical.

### Docs

- **`budget_exhausted` now documents what actually establishes it, and what it cannot answer.**
  1.8.0 shipped the field with its evidence resting entirely on `ScriptedInterpreter` — which
  proves dspy writes the forced-final marker and the kit reads it back, and says nothing about
  whether the real sandbox path reaches that branch the same way. A consumer ran all three states
  against `dspy.PythonInterpreter` on deno 2.8.2, scripting only the LM since the interpreter is
  the seam that matters, and the field discriminated. The docstring records that, the two traps
  that fake a negative result (the forced-final path makes a SECOND LM call for the task's output
  field, so a scripted LM one turn short dies in `extract` and the marker looks lost; and a `True`
  without a submitting control run is not a measurement), and the boundary that is a product fact
  rather than a defect — the trajectory is written after `aforward()` returns, so a SIGKILLed job,
  the case an operator actually asks about, is exactly the one this reports `None` for.

  **These notes were one commit past the `v1.8.0` tag**, so anybody who installed 1.8.0 and ran
  `help()` on the field saw none of it. Caught by a consumer that checked the installed package
  before writing "the kit documents this" into its own README.

### Tests

- **The forced-final test had no control run.** It asserted only that a run which never submits
  carries the marker — which a dspy that wrote that string on every run would also satisfy. It now
  drives a submitting run and requires the marker to be absent there. Verified additive: deleting
  the control leaves the suite green, so it guards something no other assertion reaches.

## [1.8.0] - 2026-08-30

The kit now computes the generic half of a rubric's facts, so a consumer supplies only its domain
half. Three new public names, one optional payload field, nothing removed or re-typed.

### Added

- **`compute_run_facts(events)` / `compute_run_facts_by_run` / `RUN_FACT_KEYS`.** `rubric.py` has
  always given you the SHAPE of a rubric — `criteria_facts(criteria, facts, lens)` is pure and knows
  nothing about traces — while every consumer hand-derived the facts to feed it. Those two modules
  are consecutive stages of one pipeline and the middle stage was missing, for a mundane reason:
  `rubric.py` predates 1.0.0 and `metrics.py` landed in 1.3.0, five weeks after the rubrics that
  would have used it. `RUN_FACT_KEYS` is a closed, public tuple the dict is BUILT against, so a new
  key cannot appear without a diff to a SemVer-governed name — which is the mechanism keeping a
  reward-shaped scalar out of the source of truth, rather than a promise in a docstring.

- **`budget_exhausted`** — did the run stop because its ITERATION budget ran out? Read from the
  marker dspy writes on its own fall-through branch, which the kit has always recorded. It needs no
  configured cap staged into the trace, works on every trace ever written, and avoids the
  `main_steps >= cap` false positive on a run that submits successfully on its last allowed turn.
  Tri-state: `None`, never `False`, when there is no `final` event or its reasoning is absent.

- **`fence_refused_turns`** — turns dspy refused to execute over a markdown fence tag.
  **Named for the mechanism, because the obvious cause is wrong**, and shipping the wrong name would
  have been the expensive part. Running dspy's own stripper over three real corpora found 60
  refusals and **zero** that start with a fence; 55 of the 60 are valid Python assigning a
  documentation page whose text contains a fenced example. dspy's stripper scans the whole cell including string
  literals. A consumer read the same number as format non-compliance and spent two prompt
  generations suppressing the code blocks its own pages needed.

- **Optional snapshot into the trace, OFF by default.** `TraceRecorder(record_metrics=True)` or
  `RLM_TRACE_METRICS=1` folds the facts into `run_end.payload["metrics"]` — additive within
  `trace/v1`, no new event type or envelope key. Computed by re-reading the file just written and
  filtered by `run_id`, so the snapshot is consistent-by-construction with the bytes beside it.
  **Emitted only when that re-read finds this run's own `run_start`:** `load_events` returns `[]` for
  a rotated file or a mismatched id, and an all-zero dict would be indistinguishable from a measured
  zero — and streamed live to every `on_event`. `run_end` is recorded from a `finally`, so a
  `BaseException` during the snapshot cannot lose it.

### Internal

- `_dspy_compat` gains `python_fence_langs`, `forced_final_marker` and `dspy_refuses_fence`. The
  three are not symmetric and the tests say so: the fence-tag SET is a dspy module constant, so its
  test asserts the introspection path resolves rather than the value; the forced-final marker is a
  bare literal with no constant behind it, so its test drives a real forced-final run instead of
  asserting the kit against itself; and `dspy_refuses_fence` is a declared verbatim mirror of a
  `_`-private dspy parser, cross-checked against that function itself. A regex shortcut was measured
  at 1,764 disagreements and 3,855 crashes over 20,016 cells — it dies on a bare fence, the
  commonest shape — so the mirror is not optional and the mutation test says so.

## [1.7.0] - 2026-08-29

A sub-LM escalation now records itself whether or not the consumer asked, and the sub-LM wrapper
hands dspy back the response SHAPE dspy handed it. No new `__all__` entry; `trace/v1` gains no
event type, envelope key, or payload field.

### Fixed

- **`intercept_sub_lm` broke a sub-LM that returns dspy's typed `LMResponse`.** The wrapper
  collapsed anything non-list into `[outputs]`, so an `LMResponse` became `[LMResponse]` and dspy
  raised `Sub-LM response must contain text, got LMResponse`. Invisible on the default path, fatal
  under `dspy.context(experimental=True)` — and dspy's own source dates the shape the wrapper
  assumed: *"In DSPy 3.3 and 3.4, ordinary calls preserve the legacy public return value."* The
  wrapper is shape-preserving now, reading and rebuilding through two new `_dspy_compat` shims
  rather than encoding dspy's convention at the call site, which is why no test could see this
  expire. A shape the shim does not RECOGNISE is returned untouched so dspy raises its own error:
  rebuilding it converts a loud failure into a silent empty completion that would reach the planner
  and then the RL data as a real escalation answer.
- **`model_as_tool` had the same defect** sixty lines away — `outputs[0]` on an `LMResponse` handed
  the model `str(LMResponse)`, the whole repr, and wrote it to the trace as the tool's result.
- **Substituting text into an `LMResponse` no longer duplicates it.** `LMOutput.text` JOINS every
  text part, so replacing only the first left the rest appended (`"AB"` round-tripping to `"ABB"`).
  dspy emits one text part per content item, so any provider returning a content array produces
  several. Thinking, tool-call, citation and refusal parts and every sibling field survive.

### Added

- **Every sub-LM escalation is traced automatically.** `CLAUDE.md` has always stated that a sub-LM
  call "is recorded as a `sub_call`"; it was true only when the consumer remembered to call
  `intercept_sub_lm` itself. A plain `dspy.LM` was invoked by dspy directly and recorded nothing.
  `RLMTask` now wraps a plain `sub_lm` for tracing at the same per-run seam that binds the recorder.

  **This is not a hypothetical gap.** Surveyed across the consumer fleet, four projects never
  wrapped; two of those had corpora — 141 traces — in which `sub_call` was identically zero, which
  is indistinguishable from "measured, and the model never escalated". That ambiguity got into a
  design decision in this repo: the speculative-tool-calling deferral cited the zero as evidence
  the model never escalates. Re-derived from code content, the real rate was 0.15%, and one of
  those escalations later proved to be a 235.5s call. **An absent event is not a measurement.**

  `intercept_sub_lm` keeps its purpose: pass `validators`/`postprocessors` for a deterministic
  validate/post-process pipeline. A consumer with its OWN recording wrapper opts out by declaring
  `records_sub_call = True` on it — a duck-typed protocol, probed with `is True` rather than
  truthiness, because `getattr` on a mock manufactures a truthy attribute for any name and a
  truthiness probe would silently skip it.

  Auto-wrapping never raises: a sub-LM it cannot wrap is used bare with a warning. It is an
  observability convenience the caller did not ask for and must never be why a run fails to start.

### Changed

- **Traces from a consumer that never wrapped gain `sub_call` events they did not have**, carrying
  the escalation prompt (`input`, truncated to 4,000 chars) into the JSONL. A corpus spanning the
  upgrade is not homogeneous — `run_start.rlm_harness` (1.6.0) separates it.
  `metrics.compute_run_utilization`'s `sub_calls_total` moves from an unmeasurable zero to a real
  count, and `export_actions` gains `kind="sub"` records, which is the point: an RL trainer doing
  credit assignment over actions previously could not see an escalation that happened.
  `export_sft_turns` / `export_rl` read no `sub_call` and are unaffected.
- On the automatic path `attempt` is structurally always `1` (no validators, so no second
  iteration), and `raw` is now always a string or `None` — `None` meaning the response shape was
  not recognised. Previously a legacy dict output was written to the JSONL verbatim.
- **A duck-typed sub-LM returning a bare `str` now fails where it used to work.** The old
  `[outputs]` normalisation turned `"hello"` into `["hello"]`, which dspy accepts; an unrecognised
  shape is now handed back untouched and dspy raises. That is the same rule that stops a silent
  empty completion, and it is the right default — but for this one input it converts a working call
  into a hard failure. A sub-LM deriving from `dspy.LM` is unaffected: `BaseLM.__call__` already
  returns one of the two shapes dspy accepts.
- A consumer driving `dspy.RLM` directly, without `RLMTask`, gets none of this.

### Upgrading — check the upgrade actually took

This release is only visible as a change in what traces CONTAIN, which makes a silent no-op
upgrade produce exactly the wrong conclusion: "the events did not appear" rather than "I am still
running the old version". One way that happens, verified rather than assumed:

    # pyproject.toml bumped to 1.7.0, lockfile still pinning the old version
    uv sync --frozen   ->  exit 0, installs the OLD version, no warning anywhere
    uv sync --locked   ->  exit 1

`--frozen` means "sync without updating the lock" — it checks nothing. `--locked` asserts the lock
agrees with the manifest. The same shape exists for any lockfile workflow (`poetry lock`,
`pip-compile`): bumping the manifest is not the upgrade, re-resolving is. Reported by a consumer
that lost a deploy to it, whose Dockerfile comment beside the line already claimed the property the
flag did not have — which is why reading the file could not catch it.

**Confirm at runtime, not from the manifest** — `run_start.rlm_harness` in any new trace, or
`python -c "import rlm_harness; print(rlm_harness.__version__)"` **from inside the deployed
environment**.

**Do not check it with `uv run`.** `uv run` re-locks implicitly, so `uv run python -c "import
rlm_harness; …"` repairs the drift it is being used to measure and reports the version you were
hoping for. That is worse than the `--frozen` behaviour it would be chasing: the flag fails
silently, but the probe actively manufactures the reassuring answer. The only version that a probe
cannot manufacture is the one already loaded in the process that is running.

### Retracted

- The 1.6.0 entry stated that a consumer "records **zero `sub_call` events, in every run** — the
  model never escalates to the sub-LM at all." **That inference was wrong**, and it is the exact
  failure this release removes: the zero measured that consumer's wiring, not its model.

## [1.6.1] - 2026-08-29

Two correctness fixes in shipped code. No new public name, no new payload field, no schema change.

### Fixed

- **A root turn's recorded `ts` could come from an earlier turn, and it reached a rendered UI.**
  `Adapter.__init_subclass__` re-wraps `format` and `parse` with `with_callbacks` for EVERY
  subclass — unconditionally, whether or not that subclass redefines them — so each subclass level
  adds a callback fire that a `super()` call then traverses. Measured, one root turn:

      stock `JSONAdapter`                          1 fire
      `runtime._LenientJSONAdapter` (the DEFAULT)  2   (it calls `super().parse`)
      a consumer subclass overriding NOTHING       3
      ...that also calls `super().parse`           4

  **The three-fire row is the one to read.** Subclassing the kit's adapter to set a single class
  attribute — overriding no method at all — was enough to add a fire, which is not what anyone
  would predict from "it calls `super().parse`", and gives that consumer no reason to suspect the
  adapter. A fix that divided by two would have left them broken while passing every test.

  So under the DEFAULT adapter every root turn fired `task._MainStepTimer` TWICE with identical
  outputs. `trace.record_main_trajectory` matches a turn
  to its live stamp by `reasoning`, so the surplus stamp was claimable by any LATER turn repeating
  that string — which a retry loop does (`"Retrying tool call - …"`, `"Placeholder before real …"`).
  Measured across 85 real traces: all 12 traces with a `ts` inversion had a duplicated reasoning
  and none of the 58 with unique reasoning did; 2.1% of per-turn deltas came out NEGATIVE. A
  consumer rendered one as a **-338.7s** turn duration.

  `_MainStepTimer` now stages the OUTERMOST parse only, via a per-thread depth from the public
  `on_adapter_parse_start`/`on_adapter_parse_end` pair — one stamp per turn, which makes the match
  an identity map. Fixing it in the matcher instead was tried and rejected: it cannot repair the
  case where the duplicate turns are ADJACENT, where it merely stops the delta going negative while
  the stamp stays ~0.1s wrong — trading a loud failure for a silent one. If a future dspy stops
  firing `on_adapter_parse_start` the depth never rises and behaviour degrades to exactly what it
  was before this change, never to staging nothing.

- **`record_main_trajectory`'s two matchers now scan forward only** (defence in depth, not the fix
  above). Trajectory order is chronological order, so a cursor parked past the previous match makes
  that an enforced property. `_match_exec` had no demonstrated defect — it stages one entry per
  `execute()` from dspy's sequential loop — but gains the same rule for symmetry and for one real
  case: dspy runs a setup `execute()` before the turn loop whose duration a turn with a colliding
  code string could otherwise claim.

- **`verify_quote` refused correct citations at non-word junctions.** Whitespace runs were joined
  with `\s+` uniformly, so a quote that reflowed a line break beside a delimiter failed:

      source  x = """One line.\nAnd another."""
      quote   """One line.\nAnd another.\n"""      -> MISMATCH, wrongly

  The joiner is now junction-aware: `\s+` between two word characters (so `foo bar` still cannot
  verify against `foobar` — the false-positive direction is the worse one and stays closed), `\s*`
  elsewhere. Word-ness uses Python's UNICODE `\w`, deliberately: `你好 世界` keeps requiring its
  space against `你好世界`, which an ASCII character class would have silently started accepting.
  The pattern's ReDoS-safety argument survives and has been restated rather than left to imply the
  old mechanic. Found by inspection, not by a failure: across ~479 real citations the old and new
  rules never disagreed, so this is a confirmed-real but confirmed-harmless defect on today's
  corpora — it ships here because it is small and saves a second release.

### Changed

- **`exec_duration_s` coverage may move slightly.** Under the forward-only cursor a turn whose only
  matching entry sits before the cursor loses its match, so the field can be ABSENT where it was
  previously present, and a `ts` can fall back to flush time where it was a live stamp. Both fields
  are optional in `trace/v1` and no reader breaks — but a consumer tracking coverage will see the
  number change.

- **`verify_quote` can now report an EARLIER occurrence.** Loosening a junction lets a match start
  where it previously could not, so the character offset in the `MATCH: found at line N (char M)`
  string moves — `a . b` against `a.b ... a . b` reported char 8 and now reports char 0. A consumer
  parsing that string, or re-deriving grounding host-side from a recorded tool call, sees it
  directly. The line number is often unchanged, so this is not visible from a line-level check.

### Docs

- **New "Reading a trace — the ordering rules" section** in the guide. `main_step` events are
  written in one block after the run, so a `tool_call` precedes them in file order while being
  chronologically later (70 of 76 traces); `payload["turn"]` is authoritative and file order already
  matches it (72 of 72); `ts` places turns against tool calls and nothing else. Written because a
  consumer reading these traces concluded "sort by `ts`", which reorders turns.
- **Corrected the claim that every local tool is sub-millisecond.** `make_grep_files_tool` ships
  `per_match_timeout_s=1.0` and `max_total_time_s=30.0`, so it is not sub-millisecond in principle
  and `compute_tool_waste` is blind to its worst case. It stays untimed because it is
  sub-millisecond in practice where measured (n=146, median 0.029s, max 0.746s, zero calls over a
  second), with the two caveats that number carries now written down beside it.

## [1.6.0] - 2026-08-27

Three new public names and three new optional `trace/v1` payload fields, all additive. No behaviour
change to any existing call path: every field is written only when there is something to write, and
every reader that has never seen them keeps working.

**This release is the groundwork a measurement asked for before the feature it was scoping.**
1.6.0 was going to be speculative programmatic tool calling — parse the REPL code as the Root LM
streams it, pre-launch the tool calls it contains, serve them from cache. Before writing any of it,
the idea was measured against ~400 real runs from the downstream fleet, and an independent audit
re-derived the numbers. Both said the same thing: on THIS fleet's current workloads there is almost
nothing to speculate, so the engine is deferred rather than dropped — and what it needs first is
the ability to tell whether that is still true.

- The fleet's dominant tool — 80% of its wall-clock — is a **serial repair loop**: one call per
  turn, each spec built from the previous call's result. 913 call sites in the corpus; exactly
  **one cell** ever contains two independent ones. Speculation ceiling there: **0.06–0.62%**.
- The newest and heaviest consumer (76 runs, 17.9 turns/run) records **zero `sub_call` events, in
  every run** — the model never escalates to the sub-LM at all.

Neither workload has parallel calls to collapse, which was the entire premise. These are numbers
about two workloads at one point in time, not a verdict on the technique: a workload that fans out
— the map-reduce-over-chunks shape the RLM paper describes — would read completely differently, and
the fields added below are what would show it.

What the same investigation found instead is that **this kit cannot be measured from its own
traces**, and that is this release:

- 99.8% of the heaviest consumer's wall-clock landed in one opaque bucket — the gap before a
  `main_step` — which mixes root-LM GENERATION with sandbox EXECUTION. Those have completely
  different fixes and nothing separated them.
- Not one of the 3,329 `tool_call` payloads in the corpus carried a duration. The only clock was
  the envelope `ts`, stamped when the event is *recorded*, so every wall-clock attribution had to
  infer durations from inter-event gaps — which charges a whole turn's model generation to that
  turn's first tool call. Every number produced against this corpus before 1.6.0 had that error.
- No trace said which kit wrote it. `schema` is the FORMAT version. Most of the corpus predates the
  fleet's move to a released version and there is no way to tell which parts.

### Added

- **`run_start.payload["rlm_harness"]`** — the kit version that wrote the trace, beside `meta`
  rather than inside it (`meta` is the caller's namespace; `rubric_to_meta` writes there). Resolved
  by a deferred import because `trace.py` is imported *by* `__init__.py`.

- **`tool_call.payload["duration_s"]`** — an explicit optional parameter on `record_tool_call`,
  written only when given, so the name and unit are documented in one place. The shipped tools whose
  cost is a WAIT on something outside this process now pass it: `fetch_url`, `web_search`,
  `run_command`, `git_clone` (spanning both attempts when the credentialed fallback runs), the
  `model:<id>` tool from `model_as_tool`, and every MCP tool. A local read/grep/edit does not — sub-millisecond work and
  refusal paths that never touch anything would add noise, not signal, and the metric says "not
  recorded" rather than "free". `run_command` keeps its runner-owned `duration_ms` alongside; they
  are different quantities and only the wrapper's wall-clock is comparable across tools.

  Enforced, not remembered: `tests/test_tool_durations.py` requires every `make_*` in
  `rlm_harness.tools.__all__` to be classified as outbound-and-timed or exempt WITH a written
  reason, so a new outbound tool that forgets fails at the moment it ships. That table exists
  because `git_clone` and `model_as_tool` both shipped without a duration in this release's own
  first draft — the same failure mode `tests/test_repl_safety.py`'s sweep was built for.

  The tool that dominates a real run is usually the consumer's own model-backed one, and for the
  two BASE factories the kit cannot record it: `make_model_tool` / `make_harness_tool` are
  deliberately side-effect-free, so the consumer's wrapper owns the `record_tool_call`. One
  `duration_s=` from that wrapper is what makes the 80% visible. (`model_as_tool` is the exception
  — a model-backed tool the kit does record — and it now carries its own duration.)

- **`main_step.payload["exec_duration_s"]`** — how long the sandbox spent on that turn, so the
  biggest bucket in a real trace stops being opaque. Reuses the mechanism that already backfills
  per-turn timestamps: a `note_exec_duration` sibling to `note_main_step`, staged by the two
  interpreter wrappers the kit owns (`sandbox.py`'s guarded `execute` — on both branches, leaving
  its isolated disabled-by-default guard literally first — and `ContainerInterpreter.execute`) and
  matched onto turns by the CODE that ran, in `record_main_trajectory`. An interpreter a caller injects
  directly is not wrapped, so its turns carry the key ABSENT rather than a wrong zero. Resets per
  attempt with the timestamp buffer.

  **Matched by the CODE that ran, never by position.** A turn does not always reach the sandbox:
  dspy raises `SyntaxError` out of `_strip_code_fences` for an explicitly non-Python fence tag
  (` ```json `, ` ```bash `) and records that turn from the unstripped text without calling
  `execute()`. A positional zip therefore credits the skipped turn with the NEXT turn's time and
  shifts every one after it — a confidently wrong attribution with nothing to signal it, which is
  worse than having none. Caught by a pre-merge review after the first implementation shipped
  exactly that bug; pinned by a test now.

- **`compute_tool_waste` / `compute_tool_waste_by_run` / `ToolWaste`** — per tool: calls split by
  outcome, and what each outcome cost. The number nobody had: on this corpus **57% of all tool
  wall-clock produced output the consumer's own validator rejected**. For scale, a
  `max_consecutive_invalid` of 2 instead of 5 — a knob `make_model_tool` already ships — would have
  saved 4.53h of 20.6h there, which is ~2000× the speculation ceiling, available today by changing
  one number.

  Two things it refuses to do. It never reads `ok` directly — outcomes come from `payload_cause`,
  because `ok` is frequently ABSENT on an endpoint-failure payload and `payload.get("ok")` then
  returns `None`, so a naive counter absorbs infrastructure failures as content declines (a mistake
  that has shipped four times). And it never infers a duration from event gaps: `*_seconds` is
  `None` — "not recorded" — for anything unmeasured, because `0.0` would read as "measured and
  found to be free" and inferring is the exact error this release exists to stop.

### Deferred to 1.7.0, with what was learned

Speculation itself, and the `speculatable()` marker that goes with it: a public marker with no
engine is dead surface, and post-1.0 it would be SemVer-frozen from the day it ships.

**Scope note, because an earlier draft of this entry got it wrong.** The upstream technique is NOT
sub-LM-only. Its gate is SIDE EFFECTS — "as long as all inputs are safe (i.e. pure functions, no
side effects), they can be speculated" — and it names *search APIs* alongside sub-agents as the
canonical high-latency target, with a mail-sending tool as the counter-example. So `fetch_url` and
`web_search` are squarely in the intended set on the author's own terms.

What excluded them HERE was a stricter, separate concern: a speculation that is launched and then
discarded has already been observed by the remote end, and nothing at commit time unsends it. That
is a real cost — it is someone else's server, someone else's rate limit, and a request the policy
never chose — but it is a POLICY judgement that belongs to the consumer, expressed through an
explicit opt-in, not a technical impossibility. `make_harness_tool` is the one genuine permanent
exclusion: its own source documents "one-call-at-a-time, so the slot is race-free", a cached result
would replay a `child_run_id` pointing at a different child rollout, and each call spawns a whole
child rollout rather than a cache lookup.

Streaming is deferred with it.

Four defects in that path were verified while scoping it, two of
which silently break documented invariants: `dspy.streamify` runs the program under an anyio task
group and anyio does not unwrap a single child exception, so `SandboxCancelled` arrives as an
`ExceptionGroup`, `_retry.py`'s `except non_retryable:` stops matching, and a caller-driven cancel
gets RETRIED; `is_fast_fail_lm_error`'s `isinstance` fails the same way. Also: `StreamListener`
defaults `allow_reuse=False` and so streams turn 1 only; setting `send_stream` hijacks a consumer's
own `dspy.streamify`; and `streamify` captures `settings.callbacks` at construction, dropping
`_MainStepTimer` so every `main_step` silently reverts to a flush timestamp. All four are written
down here so 1.7.0 starts from them rather than rediscovering them.

Suite: 791 passed, 1 skipped on dspy 3.3.1, on all three CI interpreter cells (3.11, 3.12, 3.13). Plan reviewed by an
independent agent before implementation; its objection to shipping the engine on these numbers is
why this release is the measurement rather than the feature.

## [1.5.0] - 2026-08-27

One new public name and two corrected dependency floors. **The headline is a correctness fix, not a
feature**: on the MCP SDK major a fresh install resolves today, a FAILED MCP tool call was reported
to the model — and recorded in the trace — as a SUCCESS. Every other item here is a gap the same
investigation exposed. No trace-format change; `trace/v1` is untouched.

Both floors move: `dspy>=3.3.1` (was `>=3.3.0`) and `mcp>=1.8.1` (was `>=1.0`). Neither is a
breaking change for a consumer — nothing pins dspy or mcp directly — but both are deliberate; see
below for why `>=1.0` was never a floor the code could actually run on.

### Fixed

- **An MCP tool failure read as a success (mcp SDK 2.x).** The SDK renamed its model fields
  camelCase -> snake_case at 2.0 and kept the old spellings only as pydantic *serialization*
  aliases, so attribute access under the old names no longer resolves. `rlm_harness/mcp.py` read
  three of them:

  | read | on 2.x | consequence |
  |---|---|---|
  | `Tool.inputSchema` | `input_schema` | `AttributeError` — `mcp_tools()` died outright |
  | `getattr(result, "isError", False)` | `is_error` | **every failed tool call reported `ok`** |
  | `getattr(result, "structuredContent", None)` | `structured_content` | structured results dropped |

  Only the first failed loudly. The second is the one that matters: the model was told a failing
  tool had succeeded, and `record_tool_call` wrote `ok=True` into the trace, so a dataset built
  from those rollouts learned from them.

  A fourth shape had to be handled with them: 2.x widened `call_tool`'s return type to
  `CallToolResult | InputRequiredResult | Result`, and only the first carries `content` or an
  error flag. `_is_tool_result` recognises the other arms BEFORE the flag is consulted, so they
  are neither flattened to `""` nor recorded as `ok=True` — which would have been the same
  "a call that did not succeed reads as a success" defect, reintroduced by the fix for it.

  All three renames now go through one `_sdk_field` accessor that reads both spellings. It returns a
  `_MISSING` sentinel rather than a default, deliberately: `structured_content` is typed `Any` on
  2.x, so `{}` / `0` / `False` are legitimate VALUES an `or`-chain would silently discard — and,
  more importantly, a `getattr(obj, name, False)` is exactly what turned the NEXT rename into a
  wrong answer instead of an error. A result carrying no error flag under either spelling now
  logs a warning once instead of being assumed successful. `result_text` also names a
  non-`CallToolResult` rather than flattening it to `""`, since 2.x widened `call_tool`'s return
  type to a union whose other arms carry no `content`.

  **Why no test caught it:** the three fixture servers in `tests/test_mcp.py` imported
  `mcp.server.fastmcp.FastMCP`, which does not exist on 2.x, so the whole suite could only ever
  run against 1.x — and the error flag was covered exclusively by `SimpleNamespace` fakes
  spelling it `isError`. Both are fixed: the fixtures build against whichever major is installed,
  and a live server that actually raises now pins the error path end to end. `structured_content`
  is the one half a live server still cannot exercise — a dict-returning tool is serialised to
  text content on both majors, so `result_text` returns before it looks — so it is pinned the
  only way that is honest instead: a live result must carry each field this module reads, under
  at least one of the two spellings it knows.

- **`RLM_REQUEST_TIMEOUT` was a silent no-op on the Claude-subscription route.** It becomes
  `dspy.LM(timeout=...)`, which an auto-routed `ClaudeAgentLM` never sees, so a consumer could set
  it, see no error, and believe a route was bounded to that number. `configure()` now warns,
  naming `configure(main_lm=ClaudeAgentLM(model, timeout_s=...))` as the seam that does choose it.

  It is deliberately NOT forwarded as `ClaudeAgentLM(timeout_s=...)`, which was the obvious fix
  and is wrong: `request_timeout_s` bounds one HTTP request with dspy (`num_retries=3`) and
  litellm (`max_retries=2`) retrying around it, while `ClaudeAgentLM.timeout_s` is an end-to-end
  per-call deadline that INCLUDES time queued behind that SDK's concurrency semaphore. Under
  `llm_query_batched`'s thread fan-out, a value that is generous per request would make queued
  sub-LM calls time out from waiting alone. One number cannot mean both.

- **The action prompt described the wrong runtime for every non-Pyodide interpreter.** dspy 3.3.1
  renders an "Execution environment:" section sourced from `interpreter_factory.execution_instructions`.
  The kit supplies its interpreter POSITIONALLY (which is what keeps ownership), so dspy read the
  attribute off its own default `PythonInterpreter` and told every run that "subprocesses and
  native extensions are unavailable" — including a `container` run, whose entire reason to exist
  is that they are available. Nothing went red; the model simply stopped trying.

  `_dspy_compat.interpreter_instructions_kwargs` now passes an `interpreter_factory` that is a
  metadata CARRIER only: dspy reads one attribute off it and never invokes it (`_validate_interpreter_factory`
  validates without calling, `_interpreter_context` returns early for a caller-owned interpreter,
  and `_build_rlm` always resolves one). It raises if ever invoked, so a future dspy that moves
  the positional seam fails loudly rather than double-shutting-down the sandbox. This does not
  contradict the standing "`interpreter_factory=` is the wrong seam" rule — that rule is about
  SUPPLYING an interpreter — and `CLAUDE.md`, `_dspy_compat.py` and `task.py` all say so now.

### Added

- **`short_error`** — head-and-tail elision for a caught exception, so a giant `AdapterParseError`
  (which embeds the entire raw LM completion) becomes one readable log line. It already existed
  as `_retry._short_error`; it is public because two independent consumers had reached into the
  private module for it, which is this project's own stated trigger for promoting a named hook.
  Its BEHAVIOUR is frozen, not its output string — see the docstring for exactly which properties
  are part of the promise. The private spelling still resolves, as an undocumented alias.
  It also no longer raises when an exception's own `__str__` does: every call site is an `except`
  block, where blowing up replaces a diagnosable failure with an undiagnosable one. And the
  length bound it now promises is one it actually keeps — at `limit <= 1` the head/tail split
  left `tail == 0`, and `text[-0:]` slices the WHOLE string, so the smallest budgets produced the
  longest output. Found by holding the code to the contract the docstring had just frozen.

- **`mcp-latest`, a second job in `.github/workflows/dspy-latest.yml`** (now named "newest deps"),
  and **`mcp-major`, a pinned 2.x leg in `ci.yml`**. The uncapped `mcp` extra had no defence at
  all, which is why the bug above shipped; dspy has had one since 1.0.1. The assert in the new job
  reads `importlib.metadata`, not `mcp.__version__` — the package exposes no `__version__` on
  either major, so copying the dspy job verbatim would have crashed rather than verified.

- **`execution_instructions` on the kit's own interpreters** — `ContainerInterpreter`,
  the `mock` interpreter, and `testing.ScriptedInterpreter` each now describe their real runtime.
  The container's is DERIVED from its `ContainerConfig`, not a constant: `network`, `read_only`
  and `workdir` are all operator-configurable, and a fixed string would eventually tell the model
  a capability is absent when it is present — the same defect class as the Pyodide default it
  replaces, just quieter.

### Changed

- **Floor `dspy>=3.3.1`.** Required, not preferred: `execution_instructions` does not exist in
  3.3.0. 3.3.1 also re-parents `CodeInterpreterError` under `DSPyError`, the hierarchy the
  cancel-is-never-recoverable invariant reads — now pinned by a test that asserts
  `SandboxCancelled` is a subclass of neither the recoverable nor the terminal class, nor of the
  new root. Note 3.3.1 hard-gates Deno to `>=2.0.0,<3.0.0` and adds `pip install "dspy[deno]"`;
  the docs say so now.

- **Floor `mcp>=1.8.1`.** `>=1.0` was never runnable: `mcp.client.streamable_http` does not exist
  at 1.0.0, so the HTTP transport could not work at the declared floor. 1.8.1 rather than 1.8.0
  because a floor is only meaningful if the suite passes on it — on 1.8.0 a refused connect takes
  ~16s, against the <10s this kit asserts for fail-fast. Still no upper bound — a cap propagates
  through pinned consumers — with the two new jobs as the compensating control.

### Documentation

- **A new guide section, "Timeouts — what bounds what"** (`rlm_harness/README.md`). Six bounds on
  three clocks, and none of them bounds a run's wall time on its own. It states the three things
  that are easy to get wrong: `request_timeout_s` is per ATTEMPT and dspy/litellm retry around it,
  so the run-level wait is a multiple; the two 600s defaults are different quantities selected by
  a model string; and a sandbox turn timeout is recoverable while a cancel deliberately is not.
  It also records that these semantics assume non-streaming requests, so a future change that
  turns streaming on owes the page a revision.

Suite: 758 passed, 1 skipped on dspy 3.3.1 — on Python 3.11 and 3.13, and on mcp 1.8.1 (the declared
floor), 1.28.0 (the lock) and 2.1.1 (the newest).

Two independent agents reviewed this release. The PLAN was audited before any code was written:
six blocking defects came back, three of them places where this changelog's first draft would
have been wrong — the `interpreter_factory` kwarg surviving `_build_rlm`'s lossy fallback, three
tests claimed to go red that would not have, and a fixture assertion on a raw SDK object that no
accessor could fix. The IMPLEMENTATION was then reviewed before merge, which is where the
non-result-reads-as-success bug, the `short_error` length bound, the config-dependent container
text, and a floor of `mcp>=1.8` that the suite does not actually pass on all came from.

## [1.4.0] - 2026-08-27

One new public name, additive and off by default. **MINOR, not PATCH** — `RLMConfig.request_timeout_s`
is public, documented, consumer-facing surface, and this project's rule is that adding a public name
is a minor bump. (1.2.1 is the only prior patch with an `### Added` section, and it opens by saying
everything it added was private.) With the field unset every call site behaves exactly as in 1.3.0.
No public surface removal, no trace-format change.

### Added

- **`RLMConfig.request_timeout_s` / `RLM_REQUEST_TIMEOUT`** (seconds) — a wall-clock cap on ONE
  model HTTP request ATTEMPT, handed to `dspy.LM(timeout=...)` and from there to litellm. A
  pass-through rather than new machinery: `dspy.LM` keeps kwargs it does not recognise and merges
  them into the litellm call, and `litellm.completion` takes `timeout`.

  **Unset is not "no cap", and this is worth reading before you size anything.** With nothing
  passed, litellm applies its own `COMPLETION_HTTP_FALLBACK_SECONDS` of **600 s** per attempt. So
  this field REPLACES that number rather than introducing a bound where there was none, and a
  consumer whose turns legitimately exceed ten minutes must set it UP rather than leave it alone.

  **It also does not bound a run to its own value.** dspy passes `num_retries=3` and litellm's
  first call hands the OpenAI SDK `max_retries=2`, so a dead endpoint is retried and the run-level
  wait is a MULTIPLE of this plus backoff. Size a caller-side budget on the multiple.

  What prompted it: `sandbox_turn_timeout_s` bounded the sandbox side of a turn and the model side
  was not settable at all. Against a self-hosted OpenAI-compatible endpoint one request never came
  back — socket `ESTABLISHED`, both queues empty, the worker asleep in `epoll_wait` for 38 minutes
  at 0.3% CPU, while that same endpoint answered unrelated requests in half a second. Stated
  honestly: 600 s across four attempts is about forty minutes, so that observation fits "litellm's
  own default, retried" at least as well as "nothing was watching", and no attempt counter was
  captured at the time. Either way the consumer could not choose the number, and now it can.

  When unset the key is ABSENT from the LM kwargs rather than present-and-`None`, since clients
  differ on what an explicit null means. Both roles get it — the sub-LM as well as the planner,
  which matters because `dspy.RLM` fans the sub-LM across a thread pool where a wedged request is
  less visible.

### Documentation

- **`max_tokens` now names the failure consumers keep misdiagnosing.** Its docs covered only the
  `None` case (defer to the server; a small server-side cap truncates a reasoning model's
  chain-of-thought and `content` comes back empty). Running into the DEFAULT presents completely
  differently: 8192 must hold one turn's chain-of-thought AND its structured answer, and a long
  turn that overruns it is cut off mid-JSON, so the adapter cannot parse the reply and the run
  surfaces as `RLMTaskError: Failed to produce a valid '<field>'` caused by `AdapterParseError`.
  That is a truncation, not a model failing a schema, and it is repeatedly diagnosed as the latter
  because the quoted response looks well-formed right up to where it stops. The guide now names
  the symptom, how to tell the two apart (read the END of the quoted response — a truncated one
  has no closing brace, and dspy separately logs `LM response was truncated due to exceeding
  max_tokens=...` at WARNING), and a number to try.
- Also states what raising it costs, which the first draft of this text did not: an
  OpenAI-compatible server commonly validates `prompt_tokens + max_tokens` against the context
  window, so a bigger cap removes usable PROMPT budget — and an RLM planner's prompt grows every
  turn. And that OpenAI's own reasoning models reject the 8192 default outright at `configure()`
  time (`LMConfigurationError: ... max_tokens >= 16000 or None`), which is the loudest instance of
  this whole class and the reason 16000+ is the right floor for them specifically.

Suite: 726 passed, 1 skipped on dspy 3.3.0. Reviewed by an independent agent before release; its
findings are why this is 1.4.0 rather than 1.3.1 and why the two claims above are stated as they
are — the first draft of both said "unset means no cap at all", which execution disproved.

## [1.3.0] - 2026-08-25

Twenty new public names, plus two new optional extras (`grep`, `gitignore`; the pre-existing `subscription`
extra also gains new auto-routing behavior in `configure()`, but is not itself new). No
trace-format change; every existing call site behaves byte-for-byte identically to 1.2.1.

**This batch introduces the kit's first file-mutation/data-loss-capable tools**
(`make_write_file_tool` / `make_edit_file_tool`, described in their own section below) — every
tool shipped before this was either read-only against the filesystem or delegated
execution/network entirely to a consumer-supplied runner/fetcher. A bug in either of the two new
tools can destroy content, not just return wrong information; see that section for how this is
mitigated and what remains explicitly out of scope.

**Harness delegation, made lighter-weight and consumer-name-agnostic.**

- **`pointer_to_invocation`** — the one canonical mapping from a served harness's
  `serving.HarnessPointer` (`artifact`/`run_id`/`trace_path`/`reasoning`/`meta`) onto
  `tools.HarnessInvocation` (`content`/`child_run_id`/`child_trace`/`reasoning`/`child_meta`), the
  shape `make_harness_tool` reads. Previously every consumer's `read_output` callback re-derived
  this by hand; now it's a `read_output=pointer_to_invocation` one-liner, whether the pointer
  arrived over a subprocess's stdout, an HTTP reply, or an in-process call that built one directly.
- **`run_isolated`** — a small async-bridge primitive for a consumer building their OWN
  `call_endpoint` for `harness_from_endpoint` (e.g. an in-process transport that awaits a child
  `RLMTask.arun()` directly, instead of a subprocess or HTTP call). Runs a coroutine to completion
  on a dedicated new thread with its own fresh event loop, so it never raises "cannot be called
  from a running event loop" regardless of whether the calling thread already has one — which it
  will, whenever the parent task itself is mid-`arun()` (`RLMTask.run()` calls
  `asyncio.run(self.arun(...))`). **Read its docstring before wrapping a traced delegation**: a
  fresh thread starts with an empty `contextvars.Context` (the same non-inheritance
  `trace.recorder_scope` already documents for `dspy.RLM`'s own `ThreadPoolExecutor` sub-LM
  workers), so a `TraceRecorder` for the delegated child's own rollout must be entered *inside* the
  isolated call, never around it — see `examples/harness_local_run.py` for the worked pattern. The
  kit still ships no transport (`call_endpoint` stays consumer-supplied) — this is a bridging
  primitive underneath one, not a shipped transport itself.
- **`refuse_broad_git_history`** — an opt-in `guard` for `make_command_tool` that refuses a
  `git log` invocation carrying a broad-history option (`--all`, `--branches`, `--remotes`,
  `--tags`, `--glob`, `--reflog`, `--walk-reflogs`, `-g`, `--alternate-refs`). An eval/training-run
  convention — stop a model from reading branches/tags/reflogs it should not have task-specific
  hints from — not a security boundary, same honesty framing as every other `guard` (shape-only;
  a shell string chaining multiple commands is a documented non-goal, not silently swept under it).

Also new: `examples/harness_local_run.py`, a worked in-process delegation recipe built from the
two primitives above (protected offline by `tests/test_harness_tool.py::test_in_process_transport_wiring`,
which exercises the same composition with a stub child).

**Mechanics promoted from a consumer-driven audit of downstream consumers**, plus reward-free
trace utilization metrics.

- **`atomic_write_text`** (top-level) — write a file via a same-directory temp file + `fsync` +
  `os.replace`, so a concurrent reader never sees a partial write. Useful for any consumer building
  a resumable/checkpointed job on top of `RLMTask` (a manifest, a cache); two independent
  downstream consumers had each built the identical primitive on their own before this landed here.
- **Claude-subscription auto-routing in `configure()`** — a `main_model`/`sub_model` string carrying
  the `claude-agent-sdk/` sentinel (the same one `ClaudeAgentLM` already stamps its own model
  string with) is now automatically routed to a `ClaudeAgentLM` for that role, with no explicit
  `main_lm=`/`sub_lm=` wiring required. An explicit override still wins outright. **New exception
  surface**: `configure()` can now also raise `ValueError` (a bare `claude-agent-sdk/` with no model
  id), `RuntimeError` (a stale `ANTHROPIC_API_KEY` — `ClaudeAgentLM` refuses to start while it's
  set), or `ImportError` (the optional `subscription` extra isn't installed) for a role whose model
  string carries the prefix. `claude_agent_lm.SUBSCRIPTION_PREFIX` is exported (lazily, alongside
  `ClaudeAgentLM`) as the single source of truth for the sentinel spelling.
- **`RunUtilization` / `compute_run_utilization` / `compute_utilization_by_run`** (top-level,
  `rlm_harness.metrics`) — how a run's activity was distributed across the root LM's own turns,
  tool calls, and sub-LM escalations (inspired by prior "PTC/sub-agent utilization" metrics work
  elsewhere). Reward-free, like `rubric.py`: raw counts and rates derived purely from already-
  recorded `trace/v1` events, no new event type or payload field. `None` (not `0.0`) for a rate
  when `main_steps == 0` — a crashed/cancelled run that failed before its first `Prediction` ever
  returned can carry live-recorded `tool_call`/`sub_call` events with zero `main_step` events, and
  `0.0` would misleadingly read as "measured and found to be zero usage" rather than "undefined."

**Reading and searching a bounded local directory (not just "a repo") — the filesystem-side
`make_fetch_tool`, generalized beyond its first pass.**

- **`make_read_file_tool` / `make_grep_files_tool` / `resolve_within_root`** (`rlm_harness.tools`) —
  the filesystem-side analogue of `make_fetch_tool`'s SSRF-guarded `is_safe_url`: a safe, scoped,
  no-shell way to let a model read or search a bounded local directory tree — a source repository,
  a docs corpus, an extracted archive, a dataset directory, a log directory, whatever the consumer
  scopes `root` to — filling the gap between "no filesystem access" and `make_command_tool`'s
  full-shell escape hatch. `resolve_within_root` refuses `..`/absolute-path/symlink escapes via
  `realpath`+`commonpath` containment (never `normpath`, which would miss a symlink escape).
  `make_grep_files_tool` **requires the new optional `regex` package outright** (`pip install
  "rlm-harness[grep]"`, a friendly `ImportError` otherwise, no silent fallback to stdlib `re`) — an
  LM-controlled regex `pattern` matched with no bound is a catastrophic-backtracking DoS against
  the host, and stdlib `re` cannot be bounded by any pure-Python mechanism (not even
  `signal.alarm`); `regex`'s native `timeout=` is the only thing that actually works. Bounded
  per-line (`per_match_timeout_s`, default `1.0`) and for the whole call (`max_total_time_s`,
  default `30.0`, checked before every line so a single large file with many pathological lines
  can't exceed it either).
- **`name=` on both factories** (default `"read_file"`/`"grep_files"`) — fixes a real, reachable
  bug: a task wanting BOTH "read the source repo" AND "read the docs corpus" as two distinct tools
  hit a duplicate-tool-name collision at dspy's `RLM(...)` construction (dspy keys its tool dict by
  name; the collision aborts registration for EVERY tool on the task, not just the second one).
  Validated at factory-build time against both `is_valid_tool_name` and dspy's reserved-tool-name
  set — a bad name raises `ValueError` immediately rather than surfacing as an obscure construction
  failure later.
- **`make_read_file_tool`** additionally takes `encoding=` (default `"utf-8"`) for a non-UTF-8
  corpus; `max_output_chars=` (default `None` = unlimited) truncates with a visible, non-silent
  marker; `line_numbers=` (default `False`) prefixes each line with its real 1-indexed file line
  number, removing the off-by-one a model can introduce computing one itself from `start_line`.
  Both `max_output_chars`/`line_numbers` are scoped to the successful-read branch only — the
  `Refused`/`Read error` strings are never numbered or truncated.
- **`make_grep_files_tool`** additionally takes `output_mode=` (default `"content"`, unchanged) —
  `"files_with_matches"` (distinct matching file paths only) and `"count"` (`path: N` per file,
  zero-match files omitted); and `ignore_case=` (default `False`) for case-insensitive matching as
  a first-class flag. The regex-DoS mitigation above is verified, via a parametrized test, to hold
  across every `output_mode` × `ignore_case` combination.
- **Per-file early-break, `"files_with_matches"` only** — that mode only needs to know "did this
  file have ≥1 match," so scanning stops the instant one is found. `"count"` mode cannot do this
  (it needs the file's exact total, so every line there is still scanned). Both
  `"files_with_matches"`/`"count"` additionally stop OPENING further candidate files once
  `max_results` qualifying files are already found (never skips a line of a file already being
  scanned — only avoids starting new ones).
- **`context_before=`/`context_after=`** (default `0`, `"content"` mode only — a silent no-op in
  the other two modes): show that many unchanged lines immediately before/after each match, using
  grep's own convention — a match keeps `path:line: text` (colon); a context line uses
  `path-line- text` (hyphen); a `"--"` line separates two blocks that don't touch within the same
  file. A line that itself matches is always emitted as a match, never as leftover context from an
  earlier match's window — a fresh match resets the after-context countdown, it never stacks with
  one already running. `max_results` now counts MATCHES only (context/separator lines are
  supplementary and uncapped by it, mirroring real `grep -m`) — when both `context_before`/
  `context_after` are `0` (the default), behavior, including the traced `result_count`, is
  byte-identical to a build with no context support at all.

**`list_candidate_paths` — a safe, good-default way to compute `candidate_paths`, in a new
`tools/discover.py` module.** `candidate_paths` stays a plain, required list on
`make_read_file_tool`/`make_grep_files_tool` (no contract change) — this is purely an additive way
to build one, closing a gap the user explicitly flagged: safely walking a directory tree
(respecting `.gitignore`, never escaping `root` via a symlink, never treating VCS internals as
candidates) is exactly the kind of mechanic every consumer would otherwise reinvent for itself.

- **`list_candidate_paths(root)` → `CandidatePaths(paths, truncated)`** (a frozen dataclass) — a
  plain host-side function, no factory, no REPL exposure (called from a consumer's own setup code
  before wiring a tool, the same role `resolve_within_root` already plays).
- **New optional extra `gitignore` (`pathspec`)** for correct `.gitignore` (`gitwildmatch`) syntax
  — negation, directory-only patterns, anchoring — a hand-rolled parser risks either leaking files
  a consumer explicitly meant to exclude (`.env`, credentials) or wrongly excluding real source.
  Lazily imported ONLY when there's an actual pattern to compile (a real root `.gitignore` present
  and `respect_gitignore=True`, or `extra_ignore_patterns` given); a friendly `ImportError`
  otherwise, no silent "ignore nothing" fallback.
- **Root-level `.gitignore` only, stated honestly** — no nested per-directory merging, no global
  gitignore. `extra_ignore_patterns` is the escape hatch for a consumer that needs more.
- **`.git` is always excluded, unconditionally, by TWO distinct mechanisms**: a directory named
  `.git` is pruned from the walk, and a plain FILE named `.git` (git's real submodule gitlink
  shape — a one-line pointer file, not a directory) is also always excluded. Directory-only
  ignore patterns (`build/`) are matched with a required trailing slash, matching `pathspec`'s own
  `gitwildmatch` convention — omitting it would silently fail to prune such a pattern.
  Every candidate file is re-checked through `resolve_within_root` regardless of
  `follow_symlinks`, and `dirnames`/`filenames` are sorted at every level of the walk so which
  files survive a `max_files` truncation is deterministic, not filesystem/OS-order-dependent.
- `glob=` applies the same `fnmatch` semantics `make_grep_files_tool`'s own `glob` already uses,
  so the result is directly pipeable: `make_grep_files_tool(root,
  candidate_paths=list_candidate_paths(root).paths)`.

**`make_git_clone_tool` — safe git clone with fallback auth, in a new `tools/git_clone.py`
module.** Base/wrap, the same shape as `make_fetch_tool`/`make_command_tool`: the kit does not
shell out to `git` itself (a `git clone` is not meaningfully safer to execute un-isolated than an
arbitrary command), so a CONSUMER-SUPPLIED, isolated `cloner` does the actual clone.

- URL safety reuses `is_safe_url` directly (a syntactic pre-flight only — the same
  DNS-rebinding-at-connect-time caveat `fetch.py` already documents for its own `fetcher` applies
  here, delegated to the `cloner`). Destination confinement reuses `resolve_within_root` directly.
- **Fallback auth**: tries without credentials first; on failure (a bad exit code OR a raised
  exception, handled identically), if a credentials provider is configured, ONE retry with
  credentials — never more than two clone attempts, no retry loop.
- **Credential redaction, best-effort, disclosed as such** — the raw secret string a credentials
  provider supplies is redacted (exact-string match) before it can reach anything model/trace-
  visible: the returned string, and `stderr_preview` in the trace. `stdout` is redacted too before
  its (post-redaction) length feeds the trace's `stdout_len` — `stdout` content itself is never
  traced verbatim, only that length, matching `run_command`'s own existing "lengths + a preview,
  not the full stream" trace-size posture. This does NOT catch a derived/transformed leak
  (URL-encoding, case-folding, a truncated echo). A malformed credentials dict (missing or
  non-string `"secret"`) — or a credentials provider that itself raises — fails CLOSED: treated
  exactly like a decline, never crashes the tool call.
- **`default_depth=1`** (shallow clone by default) — the "avoid being tricked into cloning an
  enormous repository" mitigation, a factory parameter passed through to the `cloner`.
- No cleanup of a partially-cloned directory on failure — stated as an accepted gap, not silently
  assumed away (making the whole clone atomic would require dictating how the `cloner` writes to
  disk, contradicting the base/wrap split this design is built on).

**Writing and editing a bounded local directory — the write side of the read/search pair above,
in a new `tools/edit.py` module** (kept separate from `fs.py`, already the largest single file in
`tools/`, so "everything that can mutate the filesystem" stays physically distinct from
"everything that only reads it").

- **`make_write_file_tool`** — create or overwrite a whole file, atomically
  (`atomic.atomic_write_text`), scoped by the same `resolve_within_root` guard as the read side.
  Unconditional overwrite (no create-only mode) and no `max_content_chars` cap in this round —
  both are stated, considered choices, not oversights (see the module's own docstring for the
  reasoning and the one accepted gap: a model looping on this tool can still fill the host's disk
  with many individually-bounded files).
- **`make_edit_file_tool`** — exact-string-anchor replacement, mirroring the same
  uniqueness-checked contract Claude Code's own `Edit` tool and `nano-rlm`'s `edit` skill both use
  independently: refuses (never mis-edits) if `old_string` isn't found, or is found more than
  once and `replace_all` (a per-call, not factory-level, flag) is `False` — the file is left
  byte-for-byte untouched on every refusal path. `old_string == ""` and `old_string == new_string`
  are refused as degenerate inputs; `new_string == ""` (delete this text) is a legitimate
  operation and is NOT refused.
- **On success, a windowed snippet of the result is appended** (`show_snippet=`, default `True`;
  reuses `read_file`'s own numbered-line convention) so a model can confirm what its edit did
  without a separate `read_file` round-trip. `snippet_context_lines=` (default `3`) bounds each
  shown region — an oversized edit shows only its own head/tail with a visible "N line(s) omitted"
  marker, never an unbounded dump. `max_snippet_occurrences=` (default `3`) caps how many
  `replace_all=True` occurrences get their own snippet (the file is still fully edited regardless
  — this only caps what's echoed back; a capped result says so explicitly). Scoped to the success
  path only — refusal/error strings are never appended to.
- Both factories take the same `name=`/`encoding=` parameters as `make_read_file_tool`/
  `make_grep_files_tool`, including the same `name=` collision fix and factory-build-time
  validation (identifier + not dspy-reserved).
- **A real bug in `atomic_write_text` (added last round, its first tool-level consumer) was found
  and fixed while building these two tools**: `tempfile.mkstemp` always creates its temp file at
  mode `0600`, and `os.replace` does not carry the destination's mode across — so overwriting an
  existing file through `atomic_write_text` previously reset its permissions to `0600` silently
  (e.g. stripping the executable bit off a script). Fixed in the shared primitive itself: the
  destination's existing mode is preserved across an overwrite, `stat`-then-`chmod`-then-`replace`,
  so there is never a window where the file at the final path has the wrong mode.
- **Known, accepted, out-of-scope-for-this-round risk**: `atomic_write_text`'s guarantee is "no
  torn read," never "serializes concurrent read-modify-write across processes" — two SEPARATE
  `RLMTask` runs (or two workers in a batch eval) racing an edit on the same file can silently
  lose one of the two updates. Stated here rather than omitted; not mitigated this round.

**`verify_quote` — a deterministic host-side building block for the Grounded-completeness
recipe's itemized diff (see "Grounded completeness" in the guide), closing a gap the guide itself
already named: there was no deterministic check for CONTENT correctness, only structure/format
validators.**

- **`verify_quote(source, quote)`** (`rlm_harness.tools`) — verifies `quote` appears (verbatim, or
  under whitespace normalization) in `source`, returning a parseable `"MATCH: ..."`/
  `"MISMATCH: ..."` string with a line number and a context snippet on success, and a bounded
  "closest line" diagnostic on failure (single-line `quote` only). A single plain function, no
  factory, no `name=`, no trace call — it binds to nothing at construction time and has no
  filesystem/network side effect, matching `make_schema_validator`/`make_json_schema_validator`'s
  own precedent.
- **No `regex` package needed, unlike `make_grep_files_tool`** — `quote` is matched as literal
  text (every character either `re.escape()`d or collapsed to a flat, non-nested `\s+`/`\s*`), so the
  built pattern can never exhibit catastrophic backtracking regardless of `quote`'s content;
  verified empirically against a literal `"(a+)+"` quote, not just argued by design.
- **Leading/trailing whitespace on `quote` is stripped before matching** — a bug caught by direct
  testing during implementation: without stripping, incidental padding around a quote became a
  MANDATORY `\s+` at the pattern's own edges, spuriously refusing an exact-content match whenever
  `source` didn't happen to have matching whitespace immediately outside the quoted text.
- A whitespace-only `quote` (not just an empty one) is refused outright — it would otherwise
  reduce to the bare pattern `\s+`, a trivially-satisfiable, meaningless "match."

**`run_in_subprocess` — a safe, isolated-subprocess primitive, in a new top-level
`isolation.py` module (not under `tools/` — see below).** A small primitive only: safely run
one picklable callable in an isolated OS process, get its result or a clear error back, bounded
by a timeout. Task-scheduling (how a web server actually queues many of these — Celery, RQ, a
plain thread/process pool) is explicitly the consumer's own concern, not shipped here.

- **A genuinely different gap than three things in this kit that already sound similar**:
  `interpreter="container"` isolates the RLM's own REPL sandbox, but the root process still runs
  the RLM's own orchestration directly; `tools.run_isolated` bridges an async coroutine into a
  sync call site on a dedicated thread — same process, no OS-level isolation; `cancel_event`
  stops an in-flight run the calling code already owns and is watching. None hand a whole task
  off to a separate OS process in the first place.
- **Lives at the top level, not `rlm_harness.tools`** — that package's own module docstring
  scopes it as "tools RLM tasks can expose to the model inside the REPL"; nothing here is ever
  placed in a `tools=[...]` list.
- Uses `multiprocessing.get_context("spawn")`, never `"fork"` — the parent is very plausibly a
  web server already running an event loop / thread pool / a live LM client, and forking that
  risks inherited locks/half-open sockets. `factory` must be a picklable, module-level callable
  (a local closure/lambda is not); every value bound into a `functools.partial`'s own arguments
  must also be picklable, not just the function reference.
- **A real, load-bearing bug found and fixed during design review**: `multiprocessing.Queue.put()`
  does not pickle synchronously — a background feeder thread does, and a pickling failure there
  is logged and silently dropped, never raised back to `put()`'s caller. Fixed by having the
  child explicitly test-pickle its payload synchronously, in its own code, before ever calling
  `put()` — falling back to a plain-string `RuntimeError` only if that test-pickle itself fails.
  The parent's own `queue.get()` also carries its own bounded timeout, independent of the
  process-level timeout, as a backstop against an out-of-band kill (e.g. host OOM) that bypasses
  this primitive's own signal-based escalation entirely.
- **Timeout escalation**: `terminate()` (SIGTERM), a grace period, then `kill()` (SIGKILL) if
  still alive, followed by a final reap so no zombie is left — SIGTERM does NOT reliably let the
  child's `finally`/`atexit` code run unless the child itself installs a handler.
- **`max_memory_mb`/`cpu_time_limit_s`** — opt-in, POSIX-only, best-effort `resource.setrlimit`
  caps. `cpu_time_limit_s` (`RLIMIT_CPU`) is confirmed to enforce correctly on every POSIX
  platform tested, including macOS. `max_memory_mb` (`RLIMIT_AS`) bounds virtual address space,
  not physical memory — and on macOS specifically, the kernel refuses to lower `RLIMIT_AS` from
  unlimited at ALL (empirically confirmed: every attempted value failed identically with
  `ValueError: current limit exceeds maximum limit`), so this parameter is effectively Linux-only
  in practice today. Either way it fails loudly (relayed as a clear exception) rather than
  silently leaving the cap unenforced. **A second edge case, confirmed on real Linux CI**: an
  aggressively low `max_memory_mb` can starve the relay mechanism itself — the child correctly
  hits `MemoryError`, but is by then so memory-constrained that `multiprocessing.Queue.put()`'s
  own feeder thread fails to start, crashing the child before anything relays. No fallback is
  possible for this (a resource-exhausted process can't reliably report its own exhaustion
  through a mechanism that itself needs resources) — it degrades to the same safety net an
  external kill would (the parent's bounded `queue.get()` times out with a generic error), which
  is the correct, accepted outcome, not a bug.

**`make_extract_archive_tool` — safe zip/tar extraction, in a new `tools/archive.py`, plus a new
`atomic_write_stream` primitive in `atomic.py`.** `zipfile.extractall()`/`tarfile.extractall()`
are not safe by default — a malicious entry can carry an absolute path, a `..`-traversal path, or
(tar) a symlink/hardlink pointing outside the extraction target ("zip slip"). This is the same
`resolve_within_root` reasoning `read_file`/`write_file` already apply to a single path argument,
generalized to every entry of an archive.

- **Two-pass extraction, matching this kit's "refuse outright, never partially mutate" posture**:
  Pass 1 validates every entry's metadata ONLY (name, type, declared size, and — zip-only — the
  encryption/compression-method header fields) and refuses the WHOLE operation upfront on any
  violation, before a single byte is written; Pass 2 (only reached once Pass 1 fully passes)
  streams each entry's real bytes via `atomic_write_stream`, bounding peak memory to a small,
  fixed chunk size regardless of that entry's own size.
- **A declared size cannot smuggle more decompressed output than it promises** — confirmed
  empirically, not just reasoned about: `ZipExtFile.read()`/`TarFile.extractfile()`'s reader are
  both hard-ceilinged by the entry's own declared size field, so a "lying header" memory bomb does
  not exist via either stdlib read API. Pass 1's cumulative declared-size check is therefore what
  actually bounds a decompression-bomb-shaped archive; the streaming design exists for a separate,
  still-real reason — bounding peak memory for a single large (but honestly declared) entry.
- **Exception handling: one shared, broad-but-bounded tuple, wrapped at every point this design
  calls into the archive/compression machinery** (the initial open, Pass 1's entry-enumeration
  loop, and Pass 2's per-entry read), covering `zipfile.BadZipFile`/`tarfile.TarError`/`EOFError`/
  `RuntimeError`/`NotImplementedError`/`UnicodeError` — closing a whole category (encrypted
  entries, an unsupported `compress_type`, a local-vs-central-directory header field mismatch, a
  malformed version field raised during the archive's own constructor) rather than one exception
  type at a time. **`zipfile`/`tarfile` further delegate decompression to `zlib`/`bz2`/`lzma`
  internally, and neither module translates those libraries' own exception types on the way
  through** — confirmed empirically with a structurally intact archive whose compressed PAYLOAD
  (not header) is corrupted: `zlib.error`/`lzma.LZMAError` are also now in the shared tuple, and
  plain `OSError` (what `bz2` itself raises for corrupted data) is too — made safe to add only
  because the budget-exceeded signal from `atomic_write_stream` is its own dedicated
  `_ExtractionBudgetExceeded` (an `OSError` subclass), caught first and separately, so it can
  never collide with — or be misreported as — an ordinary `OSError` a compression library raises
  for corrupted data.
- **No password-protected/encrypted archives** — refused upfront in Pass 1 via the entry's own
  header flag bits, with a clear reason, never a crash. **No nested-archive recursion** — an
  archive found inside the extracted output is not itself auto-extracted. **Unconditional
  overwrite** of existing files at the destination, matching `make_write_file_tool`'s own posture.
- `atomic_write_stream(path, chunks, *, max_bytes=None)` — a new, additive primitive alongside
  the already-shipped `atomic_write_text` (not a refactor of it, zero regression risk to that
  primitive's own tested behavior): the same same-directory-temp-file/`fsync`/`os.replace`/
  permission-preservation idiom, but for a caller with an iterable of `bytes` chunks rather than
  one already-in-memory blob, aborting the moment a running total exceeds `max_bytes` — checked
  after every chunk, not merely once at the end.
- **A Python 3.11-only crash in that very refusal, caught by CI's own 3.11 job** — the 3.12/3.13
  jobs and a full local run were all green. Pass 1 read each zip entry's `is_dir()` *before*
  checking its name, and CPython 3.11's `ZipInfo.is_dir()` detects a trailing `/` with
  `filename[-1]`: an entry with an EMPTY name therefore raised a raw `IndexError` out of the tool
  instead of returning the intended `Refused:` string — the exact "a stdlib exception escapes as
  itself" class the two-pass design already closed twice elsewhere, reintroduced through an
  accessor rather than through a read. (3.12+ tests it with `endswith("/")` and returns `False`,
  which is why exactly one job in the matrix went red.) **Fixed by ORDERING, not by another entry
  in the caught-exception tuple**: an entry's name is now read and refused first, before any other
  piece of its metadata is touched, so no name-derived accessor can ever see a degenerate name.
  Pinned on EVERY version — not just the one whose stdlib raises — by a stub-entry test whose
  `is_dir()` raises unconditionally, so the ordering cannot regress silently on a developer machine
  running 3.12+. The test suite's own zip fixture builder needed the same treatment for the same
  underlying reason (`ZipFile.writestr()` indexes a string name identically on 3.11): it now writes
  an empty-name entry through an explicit `ZipInfo`, which skips that branch on every version.

## [1.2.1] - 2026-08-23

**Fast-failing non-retryable LM errors** — the item 1.2.0 left open. No public surface change, no
trace-format change; `_retry.py` and `_dspy_compat.py` are both private modules.

### ⚠️ Before you upgrade

An LM error dspy itself classifies as non-retryable (`LMAuthError`, `LMBillingError`,
`LMConfigurationError`, `LMUnsupportedModelError`, `LMUnsupportedFeatureError`,
`LMUnexpectedError`) now escapes `RLMTask.arun()`/`run()` **as that original dspy exception**,
after exactly one attempt. Previously it burned the full `max_retries` budget re-running the same
doomed trajectory and was then wrapped in `RLMTaskError`. If you catch `RLMTaskError` around a
task run expecting it to be the only failure type, add a matching `except dspy.LMError:` (or the
specific subclass you care about) alongside it — the two convey different things: `RLMTaskError`
now means "the model kept producing invalid output," while an `LMError` means "the call to the LM
itself was never going to succeed."

### Why

`run_with_retry` treated every exception the same: a validation failure worth another attempt and
an invalid API key worth zero more were both retried `max_retries` times, then both wrapped in the
same `RLMTaskError`. For an unrecoverable LM error, every one of those retries re-sends the exact
same doomed request — pure wasted latency and cost, and the wrapping erased the one piece of
information (which dspy exception it actually was) a caller needs to tell "fix your API key" apart
from "the model can't do this task."

dspy 3.3.0 already ships the classification this needed: `dspy.is_retryable_lm_error(exc)`, built
on a full `LMError` hierarchy (`LMAuthError`, `LMBillingError`, `LMConfigurationError`,
`LMProviderError`, `LMRateLimitError`, `LMServerError`, `LMTimeoutError`, `LMTransportError`,
`LMUnexpectedError`, `LMUnsupportedFeatureError`, `LMUnsupportedModelError`,
`ContextWindowExceededError`). 1.2.0's floor bump made this shim finally *writable* — the
3.2.x era had no such helper to build on — and `is_retryable_lm_error` calls only
`LMRateLimitError`/`LMTimeoutError`/`LMServerError`/`LMTransportError` retryable; everything else
in the hierarchy is not.

### The one carve-out: `ContextWindowExceededError`

dspy's classification assumes a retry re-sends the *identical* request, which is true for the
provider-level retries `is_retryable_lm_error` is documented for. It is not true here:
`run_with_retry` retries by re-running the **whole trajectory**, and a different turn sequence (or
a truncated tool result) can genuinely produce a shorter prompt that fits on a later attempt. So
`ContextWindowExceededError` — a non-retryable `LMInvalidRequestError` by dspy's own rule — is
excluded from the fast-fail set and keeps retrying like any other exception, consuming the full
`max_retries` budget as before. This was the exact residual question 1.2.0 left contested; it is
resolved now, in one place, rather than left for whoever next reads that note.

### Added

- `_dspy_compat.is_fast_fail_lm_error(exc) -> bool` — the classification above, resolved through
  the PUBLIC `dspy.is_retryable_lm_error`, never the private `dspy.utils.exceptions.
  _RETRYABLE_LM_ERRORS` tuple it is built from (the same "introspect the public seam" rule as
  every other shim in this module). Degrades to `False` (never fast-fail — today's pre-1.2.1
  behavior) if the installed dspy is missing `LMError` or `is_retryable_lm_error`, so a future
  dspy rename fails safe rather than over-eagerly killing a run that would have succeeded on
  retry.
- `_retry.py:run_with_retry` gained an `is_fast_fail: Callable[[BaseException], bool] | None`
  parameter, alongside the existing `non_retryable` type allowlist. A predicate, not another type
  tuple, because "is an `LMError`, is NOT `ContextWindowExceededError`, and
  `is_retryable_lm_error` says no" cannot be expressed as a static `except (A, B, C):` — it needs
  a runtime decision `_retry.py` itself must not know how to make (the module stays dspy-free).
  Matched exactly like `non_retryable`: the original exception propagates verbatim, consumes no
  attempt, and is never wrapped in `RLMTaskError`. Checked only for exceptions that fall through
  `non_retryable` first (a type match there is cheaper, and the two sets don't overlap). Default
  `None` never fires, so every existing caller of `run_with_retry` is unaffected.
- `RLMTask.arun()` wires `is_fast_fail=_dspy_compat.is_fast_fail_lm_error` into its
  `run_with_retry` call, alongside the existing `non_retryable=(SandboxCancelled,)`.

Suite: 456 passed, 1 skipped on dspy 3.3.0 (+20 tests: the classification matrix in
`test_dspy_compat.py`, the predicate mechanics in `test_retry.py`, and two end-to-end cases in
`test_integration_dspy.py` driving the real `RLMTask.arun()` → `run_with_retry` chain).

### Fixed (release pipeline, no package-content change)

- **The first attempt to publish this release failed before reaching PyPI.**
  `.github/workflows/release.yml`'s `pypa/gh-action-pypi-publish` was SHA-pinned to `v1.14.0`,
  whose bundled twine does not recognize `Metadata-Version: 2.5` — which `hatchling>=1.27`'s PEP
  639 SPDX license support has emitted for a while, so every wheel this kit builds now carries it.
  The publish job's metadata-verification step rejected the wheel outright with
  `InvalidDistribution: Invalid distribution metadata: '2.5' is not a valid metadata version`,
  before ever contacting PyPI — a CI/pipeline failure, not a defect in the 1.2.1 code above,
  which is unchanged. Fixed by bumping the pin to `v1.14.2` (twine v7, which added 2.5 support).
  Same SHA-pinning discipline as every other pin in that file: SHA + trailing version comment
  updated together.

## [1.2.0] - 2026-08-07

**Requires `dspy>=3.3.0`.** The 3.2.x compatibility branches are deleted.

### ⚠️ Before you upgrade

If you pin `dspy==3.2.x`, this upgrade will **fail to resolve** — pip/uv refuses the install.
Nothing misbehaves silently; you get an error at install time. Either unpin dspy (the kit's
documented model is "pin the KIT, not dspy" — the floor exists so you don't have to), or stay on
`rlm-harness~=1.1.0`.

Your Python floor is unaffected: dspy 3.2.1 and 3.3.0 both require `>=3.10,<3.15`. dspy 3.3.0 also
carries *fewer* exact pins than 3.2.1 (it drops `asyncer`, `typeguard`, `numpy`, `xxhash`), so for
most consumers the bump relaxes the transitive constraint graph rather than tightening it.

### Why

Supporting two dspy minors was not free, and 3.2.x is **strictly worse for users**:

- it accepts DUPLICATE tool names and silently keeps only one — the model is never told the other
  exists (3.3.x raises);
- it does not reject a keyword tool name at construction; that becomes a Deno `SyntaxError` at
  registration instead, aborting every tool on the task;
- it has no `CodeExecutionError`, so a recoverable REPL error cannot be distinguished from a
  terminal one as precisely.

So the kit's own guarantees differed by whichever dspy a consumer happened to resolve. They are
uniform now.

### What was deleted, and what deliberately was not

Gone: `_dspy_compat.rlm_accepts_interpreter_kwarg` and the `RLM(interpreter=…)` branch in
`_build_rlm`; the `max_iterations` budget alias; `_RESERVED_TOOL_NAMES` from the reserved-name
probe; the `getattr` fallback in `recoverable_interpreter_error`; and the 3.2.x rationale prose
across the package, the tests and `dspy-latest.yml`.

**`_dspy_compat` itself stays, and so do its shims** even where each now resolves a single answer.
Its value was never "supports two versions" — it is that every dspy fact lives at ONE introspected
call site, so the next rename is a one-line change plus a red test here, instead of a silent
behaviour change in someone's rollout. dspy has now renamed in a minor release once, and two of
those three renames were silent. Do not collapse a shim into its call site just because it
currently has one branch.

One consequence worth stating: the interpreter seam is now hardcoded to the `forward()`-positional
form. A dspy that moved it back to the constructor would fail LOUDLY (a `TypeError` from
`aforward`) rather than auto-adapting. That is the right trade, and `dspy-latest.yml` is what
catches it.

### `assert_task_repl_safe`'s reason changed

It used to be justified as "dspy 3.2.x enforces none of these". dspy 3.3.x enforces all four. The
helper's surviving value is that it moves the failure from run time to test time, names the rule
instead of surfacing a dspy-worded `ValueError`, and — with no dspy equivalent at all — runs
`assert_repl_safe` over every tool, covering the SHAPE rules (`*args`/`**kwargs`, required param
after a defaulted one) that dspy does not validate anywhere.

### CI

- **New `packaging (fresh install)` job** — builds the wheel, installs it into a clean environment
  and runs a task from it, with dspy PINNED to `uv.lock`'s version (it runs on `pull_request`, and a
  floating dspy would let an upstream release redden a contributor's unrelated PR — the reason
  `dspy-latest.yml` is kept off that trigger; varying dspy is that job's responsibility). That axis was genuinely untested: every other job runs
  from the source tree via `uv run`, so a module missing from the wheel would have shipped silently.
  It is explicitly **not** a substitute for the dspy axis — measured, an offline end-to-end run
  still returns the correct answer while a renamed dspy kwarg silently drops the caller's budget
  cap, so it catches a loud break and is blind to the silent ones.
- **`dspy-latest.yml` is now the only dspy axis.** With the floor and the lock both at 3.3.0 it
  tests the same version as `ci.yml` and says so in a `::notice::` — and it restores real
  two-version coverage automatically the day dspy 3.4 ships, with no human action.

Still open, not in this release: fast-failing non-retryable LM errors. The floor bump makes the
shim writable (`isinstance(exc, dspy.LMError) and not dspy.is_retryable_lm_error(exc)`), but the
residual set is contested — `ContextWindowExceededError` should probably still retry here, because
`run_with_retry` re-runs the whole trajectory and may produce a shorter context that fits.

### Fixed before release

- **`interpreter="mock"` could not run a task.** The floor bump was what broke it: dspy 3.2.x
  took the interpreter as a constructor kwarg and validated nothing, while 3.3.x
  `isinstance`-checks it against the `@runtime_checkable` `CodeInterpreter` protocol on EVERY
  forward pass — and `_MockInterpreter` was missing `tools` and `start`. It failed with
  `TypeError: interpreter must implement CodeInterpreter`. Invisible to the suite, because every
  mock test either stopped at `_build_rlm()` or injected a `ScriptedInterpreter` (which overrides
  the string path and does satisfy the protocol). Now implements the full surface, with a
  regression test that drives a real forward pass through the STRING `mock` path.

Suite: 436 passed, 1 skipped on dspy 3.3.0.

## [1.1.0] - 2026-08-07

Additive on the API surface: nothing removed, renamed or re-typed, and `rlm-harness/trace/v1` is
untouched. **Two behaviour changes to know about before upgrading**, both detailed below:

- `RecordedToolProvider.replay` now RAISES on a `preview`-only record where it previously returned
  `None`. That affects replaying a trace containing MCP or `read_skill` calls. It is the correct
  posture — it was silently serving nothing — but it is a change, not an addition.
- `make_model_tool` / `make_harness_tool`'s returned callables were renamed in 1.0.2; if you also
  declare `tools: ClassVar[...]` on an `RLMTask` subclass, note `tools` is no longer annotated
  `ClassVar` here, so a type checker will flag "cannot override instance variable with class
  variable". Drop the `ClassVar` on your side; runtime behaviour is unchanged either way.

### `RLMTask(tools=…)` — the kwarg the guide already documented

`tools` was a `ClassVar` only, so the guide's own **runnable** examples raised
`TypeError: RLMTask.__init__() got an unexpected keyword argument 'tools'`. For MCP that was
the ONLY documented attach path, and it cannot be a class-body list because the tools exist
only inside the `with mcp_tools(...)` block.

The override is resolved at BUILD time (`RLMTask.resolved_tools`, also new), not assigned in
`__init__`. That is what makes it order-independent: writing `self.tools = tools` in
`__init__` would make the winner depend on where the subclass calls `super().__init__()` —
assign-after-super, the more idiomatic ordering, would have silently clobbered the caller's
explicit kwarg. `tools=` REPLACES the declaration (never merges — merging would make the
effective list depend on inheritance depth); `tools=[]` is a deliberate "no tools", distinct
from the `None` default, which leaves the declaration path untouched. `tools` is no longer
annotated `ClassVar`: it was describing a rule the kit itself did not follow
(`examples/harness_run.py` already assigned per instance), and the package ships `py.typed`,
so that reached consumers' type checkers.

### Public REPL-safety rules — `sanitize_tool_name`, `unique_tool_names`, `is_valid_tool_name`, `signature_from_json_schema`

A consumer driving `McpCatalog` gets the server's RAW tool names and schemas and builds its
own `dspy.Tool`s — hitting exactly the defects 1.0.2 fixed inside `mcp.py`, with no sanctioned
remedy: both halves were private, and "consumers EXTEND, they don't fork" bars reaching into a
`_`-private name. 1.0.2's `assert_repl_safe` detects both and fixed neither.

**Both halves are exported, because either alone is a half-fix**: the NAME rule on its own
leaves a well-named tool whose `**kwargs` wrapper `assert_repl_safe` still rejects.
`signature_from_json_schema` is the SHAPE half, factored out of `mcp.py` so there is one
derivation. `unique_tool_names` gained `taken=` for progressive loading (servers load one at a
time, so server B's names must avoid server A's).

What is frozen is the PROPERTIES — the fixpoint, validity, non-collision — **not** the literal
output strings (`t_`, the `_2` suffix, the trailing `_`). The reserved set is read from the
INSTALLED dspy, so a sanitised name can differ across dspy versions under an unchanged
rlm-harness; don't persist these as long-lived keys.

### `assert_task_repl_safe`

`assert_repl_safe` checks ONE tool. Three of dspy's construction-time rules are properties of
the whole task and each aborts registration for EVERY tool: duplicate tool names (dspy 3.2.x
keeps only one, **silently**, with no error), an input field colliding with a tool name or a
reserved sandbox name, and an output field dspy's Prediction already owns. dspy 3.2.x enforces
none of them — and that is the version `uv.lock` pins, so it is what CI runs.

Prefer passing an INSTANCE: runtime-assembled tools (the MCP case) exist only there, and those
are the sets these rules bite on. The signature parser mirrors dspy's own two lines
(`signatures/signature.py:616`, `:649`) rather than calling `dspy.Signature`, which resolves
the output type off the call stack and raises `Unknown name` for a dynamically-built model.

### Fixes

- **`replay.py` served `None` for three of the four shipped tool families.** It read only
  `result`, while MCP and `read_skill` record under `preview`, `web_search` under `results`,
  and the `make_model_tool` convention under `raw` — and `dataset.py` already read the
  fallback, so two readers of the same trace disagreed. Now reads `raw → result → results`.
  `preview` is deliberately excluded: it is a TRUNCATED head, so serving it would hand a replay
  silently-wrong bytes; a `preview`-only record now raises the same loud `LookupError` the
  class already uses for drift.
- **`export_actions` dropped `repl_name`.** Carried through now — conditionally, mirroring
  `mcp._repl_alias`, so non-MCP records stay byte-identical — at top level beside `tool`,
  because both are identity.
- **`mcp.py` could ship a tool its own guard rejects.** The signature stamping was skipped for
  a schema-less tool, leaving the `**kwargs` proxy while the comment above it claimed
  otherwise. Now unconditional: a zero-parameter signature. (The SDK makes `inputSchema` a
  required dict, so this was unreachable through `mcp_tools`.) The remaining `**kwargs` case —
  a property named `from`, or `db.query`, which cannot be an `inspect.Parameter` — is
  knowingly degraded and now records why: sanitising the property name is NOT an option, since
  the proxy forwards it to the server as a JSON key.

### CI

- **`.github/workflows/dspy-latest.yml`** — runs the suite against the NEWEST published dspy, on
  a weekly cron + `workflow_dispatch` + push-to-main, never on a PR. This is the job that would
  have caught 1.0.1: `ci.yml` resolves dspy from `uv.lock`, so it tests a version nobody
  installing from PyPI necessarily gets, and the whole suite stayed green while the kit was
  completely unrunnable on a fresh install. A separate workflow because `schedule:` and
  `concurrency:` are workflow-scoped. No `continue-on-error` — that makes the run conclude
  `success`, so a failed scheduled run would notify nobody. The newest version is resolved from
  PyPI at run time and then **asserted** after install: a bare `--with dspy` silently resolves
  back to the locked version, so without the assert the job would be decorative.

Suite: 431 passed, 1 skipped on both dspy 3.2.1 and 3.3.0.

## [1.0.2] - 2026-08-06

**Four shipped tool-naming defects.** dspy validates a tool's NAME when `RLM(...)` is constructed —
it must be a Python identifier, must not be a keyword, and must be unique across the task — and a
failure aborts registration for **every** tool, so one bad name silently takes the rest down with it.
Four factories derived a name from data the kit does not control, and all four were broken. No public
surface change, no trace-format break; drop-in.

- **`mcp.py` — the worst of the four.** The external MCP server's own tool name went straight to
  dspy. Hyphens and dots are the MCP naming *norm*, and both are hard failures on 3.2.x and 3.3.x:
  `ValueError: Invalid tool name 'get-weather'`. A single such tool made the whole `RLMTask` fail to
  construct, taking every other tool from that server with it. Three identities are now kept
  separate: `mcp_tool.name` stays the WIRE name (`bridge.call`), `prefix + name` stays the TRACE
  identity (`record_tool_call`), and only the REPL-facing name is sanitised. Names are resolved in
  ONE pass over the server's full tool list, because uniqueness is a property of the set —
  `get-weather` and `get.weather` both clean to `get_weather`.
- **`sub_lm.py:model_as_tool`** built `f"query_{model_id}"`, so a real id gave
  `query_openai/gpt-4o-mini` — refused outright on both dspy versions. The tool could never be
  registered. `tests/test_repl_safety.py` had masked this by passing `name="m"`.
- **`tools/model.py` + `tools/harness.py`** both returned a closure literally named `call`. Using the
  two together — which CLAUDE.md's own invariant describes as the expected pattern — registered
  `['call']` on dspy 3.2.x, **silently dropping one tool with no error**, and raised
  `Duplicate tool name` on 3.3.x. Now `call_model` and `call_harness`. This is the only
  model-visible rename in the release; neither factory records a `tool_call`, so no trace payload
  moved.
- **`tools/validation.py:make_schema_validator`** built `f"validate_{model.__name__}"`; a
  `pydantic.create_model("bad-name")` carried the hyphen through. Dynamic output models are exactly
  what `RLMTask.output_model` exists for, so this is reachable.

Added:

- `rlm_harness/_toolname.py` (private) — one derivation of the naming rule for all four sites.
  `sanitize_tool_name` guarantees a **fixpoint**: an already-valid name is returned unchanged,
  *including a non-ASCII one*. This is the property that matters and the one an obvious
  `re.sub(r"[^A-Za-z0-9_]", "_", …)` silently violates — `str.isidentifier()` accepts non-ASCII
  letters and so does dspy, so `日本語ツール` and `café_search` are valid tool names that work today
  and that a character-class sanitiser would rewrite (an all-CJK name collapses to a bare `_`).
  Character validity is tested with `str.isidentifier()` itself. `unique_tool_names` reserves every
  already-valid name *first*, so a sanitised name can never displace one that was already fine.
- `_dspy_compat.reserved_tool_names()` — dspy's reserved sandbox names, probed newest-first
  (`_RESERVED_SANDBOX_NAMES` on 3.3.x, `_RESERVED_TOOL_NAMES` on 3.2.x) and **unioned** with a
  hardcoded floor. Union, not either-or: a stale fallback may then only over-reject (loud and local)
  rather than under-reject (silent here, a dspy `ValueError` in a consumer's rollout).
- `testing.assert_repl_safe` now also checks the NAME, resolving it the way dspy does —
  `getattr(tool, "name", …)` before `func.__name__`. `dspy.Tool(f, name=…)` overrides the function's
  name and dspy validates the *override*, so checking `__name__` alone would pass a tool dspy
  refuses. That is not hypothetical: an earlier draft of the `mcp.py` fix sanitised `__name__` only
  and was a placebo; this check is what catches that class of mistake.
- `tool_call` payloads gain an **optional** `repl_name`, emitted only when sanitising actually
  changed the name (additive within `rlm-harness/trace/v1`; the common payload stays byte-identical).
  It is needed because the mapping is otherwise unrecoverable offline — the sanitised name depends on
  the server's whole tool list at run time, which never enters the trace, so a reader correlating
  `main_step.payload.code` (the model typing `get_weather(...)`) against the tool events (recording
  `get-weather`) has no join key. Read it as `payload.get("repl_name") or payload["tool"]`, which
  degrades correctly for older traces and for every non-MCP tool.

Suite: 393 passed, 1 skipped on both dspy 3.2.1 and 3.3.0.

## [1.0.1] - 2026-08-06

**Compatibility fix: the kit did not run at all on `dspy` 3.3.0.** No public surface change, no
trace-format change — `__init__.__all__`, `rlm-harness/trace/v1`, and `RLMTask`'s declaration fields
are untouched, so this is a drop-in upgrade for every consumer.

The kit declares `dspy>=3.2.1` and consumers pin the KIT, not dspy. dspy 3.3.0 renamed three things
at once, so any consumer resolving dspy freshly got a kit that was broken in one loud way and two
silent ones:

- **`RLM(interpreter=…)` moved** to the first positional argument of `forward()`/`aforward()`.
  Every run raised `TypeError: RLM.__init__() got an unexpected keyword argument 'interpreter'`
  before the first LM call — including the default `pyodide` path, since the kit always constructs
  its own sandbox (to pre-bind the JSON literals) and hands it to dspy. Total failure, at least loud.
- **`max_iterations` → `max_iters`.** The old best-effort `try` around the budget kwargs was
  all-or-nothing, so the one renamed kwarg made dspy reject the call and the fallback dropped **all
  three** caps (`max_iterations`, `max_llm_calls`, `max_output_chars`) back to dspy's own defaults.
  Silent: runs simply ran to a budget nobody configured.
- **`CodeInterpreterError` stopped being the recoverable interpreter error.** dspy 3.3.0 added
  `CodeExecutionError` for that role and made the base class TERMINAL, inverting the meaning of every
  `raise CodeInterpreterError` in the kit. A `sandbox_turn_timeout_s` firing — a *safety net* whose
  entire point is to hand the model another turn — became a run-ending failure, and in the
  `container` interpreter so did **any exception the model's own code raised in the sandbox**, the
  most ordinary event in a REPL loop. Silent, and only visible under load.

Fixed by resolving each difference through a new private `rlm_harness/_dspy_compat.py` (introspection
+ `lru_cache`, dspy-free at module top), so one build works on 3.2.x and 3.3.x:

- `task.py:_build_rlm` — picks the interpreter seam (constructor kwarg vs. stashing it on
  `_forward_interpreter` for the forward call) and maps the budget caps onto the accepted names;
  `arun` prepends `_dspy_compat.forward_interpreter_args(...)` to `rlm.aforward(...)`. The `except
  TypeError` fallback survives as a backstop but now `logger.warning`s, because it is lossy.
  Interpreter OWNERSHIP is unchanged and remains the kit's: dspy shuts down only an interpreter it
  created, which 3.3.0 documents explicitly. Deliberately **not** using 3.3.0's
  `interpreter_factory=` — dspy *does* shut down whatever that factory returns, which would
  double-shutdown the kit's sandbox.
- `sandbox.py` — the turn-timeout raises `_dspy_compat.recoverable_interpreter_error()` instead of a
  hardcoded `CodeInterpreterError`. `SandboxCancelled` is unchanged and still stands outside dspy's
  hierarchy entirely, which is what keeps it non-recoverable across the rename.
- `container_interpreter.py` — the three execute-path raises that were always meant to be recoverable
  (execution timeout, an exception from the model's code, a sandbox death mid tool-reply) now use the
  resolved class. Setup/health/protocol failures keep the base class and are terminal by intent.
- `testing.py:ScriptedInterpreter` — gained a no-op `start()`. From 3.3.0 dspy's `CodeInterpreter` is
  a `@runtime_checkable` Protocol and a caller-supplied interpreter is `isinstance`-checked before
  every forward pass, so the missing method was a run-time `TypeError`.
- `tests/test_dspy_compat.py` (new) — asserts the shim's contract against whichever dspy is
  installed, so the next rename lands as a red test here rather than in a consumer's rollout.
  `tests/test_integration_dspy.py` stopped asserting on dspy internals (`rlm._interpreter`,
  `rlm.max_iterations`) and now checks the kit's own wiring and the caps by value.

Suite: 359 passed, 1 skipped on **both** dspy 3.2.1 and 3.3.0.

## [1.0.0] - 2026-08-04

**The first published release.** Everything below ships in it. Nothing was released before this:
there were no tags, no GitHub releases, and no PyPI distribution — the `0.1.0` / `0.2.0` numbers that
appeared in `pyproject.toml` and in consumers' lockfiles never corresponded to a published version,
so they are folded into this entry rather than kept as a fictional release history.

1.0.0 is a promise about the surface, not a claim that development is finished: `__init__.__all__`,
the `rlm-harness/trace/v1` wire format, and `RLMTask`'s declaration fields are frozen under
[SemVer](https://semver.org/) and pinned by `tests/test_contract.py`. Additions ship in a minor
release; a rename or removal ships with an alias and a `DeprecationWarning` first, and the removal
itself waits for the next major — the pre-1.0 habit of hard-renaming and updating consumers in
lockstep (see `make_middleware_lm` → `intercept_sub_lm` below) ends here. The trace format carries
its own version and evolves additive-only within v1. A `_`-prefixed name or module internal is
outside the promise.

Contents: the harness-engineering layer, plus the hardening surfaced by dogfooding nine real
downstream consumers.

### Renamed before first publish: `rlm-kit` → `rlm-harness`

The project was developed as `rlm-kit`. PyPI refused that name — it collapses to the same normalised
form as the existing, actively-maintained [`rlmkit`](https://pypi.org/project/rlmkit/) ("a
state-machine framework for recursive LLM agents"). The block is correct rather than bureaucratic:
two packages one separator apart, both about recursive LM agents, is a genuine confusion surface, and
the other project was there first.

Rather than publish under a name that differs from the repository, all three public identities were
moved together: the PyPI distribution, the GitHub repository, and the import name are now
`rlm-harness` / `rlm_harness`. A user checking that a package's PyPI page and its source repository
carry the same name is doing real phishing triage, and that check should not fail on our own package.
The name is not an invention — the codebase already called itself the harness throughout
(`make_harness_tool`, `serve_harness`, "the harness-engineering layer").

**The trace schema identity moved with it: `rlm-kit/trace/v1` → `rlm-harness/trace/v1`.** The `v1` is
deliberately unchanged — the *shape* is byte-identical, only the vendor tag differs, and every reader
in this kit and in all nine consumers keys off `type`, never off `schema` (verified by audit before
the change). A trace written before this rename is therefore still readable by every exporter and by
`replay`; the two strings denote the same format. This is a one-time, pre-publication change made
while the format had exactly zero external readers. It is not a precedent: `SCHEMA` is frozen from
1.0.0 onward, and a real breaking change is still a `v2` with a migration.

Environment variables were already `RLM_*` (not `RLM_KIT_*`) and are untouched.

### Added — `trace.payload_cause` + `export_actions` carries the cause: the same distinction, across the trace boundary

`ModelToolResult.cause` (below) fixes the LIVE side. A consumer reading a trace back has only the
payload, and an audit of four downstream repos found the read side is where this actually bites:

- The endpoint path is conventionally recorded as `error=<str>` **with no `ok` key at all**. So
  `payload.get("ok")` returns `None` — falsy — and every `not payload.get("ok")` counter silently
  absorbs infrastructure failures as content declines. Two of the four repos do this; two split it
  correctly, so the information was always on the wire.
- Measured in one real corpus: 113 of 116 `generator_declines` were endpoint failures, in a run
  whose validator ran **zero** times. That number feeds a scored PA rubric criterion whose own
  description attributes it to the planner's spec quality, rides the `metrics` surface into any
  trainer, and is printed in a delivered report as "113 partial/retry" — 113 times over. The
  planner itself had it right; only the deterministic fact layer was wrong.
- `_action_record` carried `{ok, output, errors}` and **dropped `error` entirely**, so the endpoint
  string reached no consumer's `actions`/`sft_turns` at all — the split could not be reconstructed
  downstream even by hand. That was a kit-level gap, not a consumer one.

`payload_cause(payload)` is the read-side mirror, reading `circuit_broken` → the endpoint string
(under `endpoint_error` OR `error`, since both conventions are in use) → `ok`, in the order that
cannot disagree with itself. It is safe on any tool_call: with no breaker and no endpoint string it
degrades to exactly the `ok` boolean. `export_actions` now emits `outcome.cause` and
`outcome.error`, preferring an explicitly recorded `cause=` over the derivation — the write side is
the code that knows.

The four `CAUSE_*` constants moved to `rlm_harness.trace`, which owns them, and are re-exported from
`rlm_harness.tools` unchanged: a live result and a recorded payload must answer this question with the
same four words, or the distinction gets collapsed again at the trace boundary.

`record_tool_call`'s docstring now states the two write-side hazards, both observed in shipped
consumers: recording the event BEFORE the endpoint check destroys the distinction irrecoverably
(226 events in one corpus are indistinguishable between "the harness was unreachable" and "the
harness returned nothing usable"), and omitting `ok` on the endpoint path is what creates the falsy
`None` in the first place.

### Added — `ModelToolResult.cause` / `.validator_ran`: `ok=False` has three causes, and now they have a name

The information was always on the result — `circuit_broken`, `endpoint_error`, and their absence —
but it had no NAME, so every consumer re-derived the distinction and several silently did not. A
cross-repo review of three downstream projects found the same collapse in each, reaching both
training data and user-facing text:

- a per-run training label named `*_rejects`, whose docstring said "the host-side validator
  rejected", incrementing on a 502 — the validator never ran;
- a reviewer-facing string reading "failed its format check" rendered in a web UI for an endpoint
  timeout, and another reading "not adoptable: patch failed validation" for a circuit break;
- a metric pair where `rejections` counted every `ok is False` while `circuit_breaks` counted the
  breaks separately — and since a break carries `ok=False` too, one real trace reported
  `calls=3, breaks=7, rejections=10`.

`cause` returns `"ok"` / `"invalid"` / `"endpoint"` / `"circuit_broken"`; `validator_ran` is the
direct form of the question a caller is usually asking ("may I say this failed validation?").
`HarnessToolResult` subclasses `ModelToolResult`, so a delegation client inherits both — and needs
them for the same reason, since a transport failure is not a content decline.

Purely additive: two properties and four constants, no field or behaviour changed. The tests drive
each case through the REAL factory rather than constructing a result, so the mapping is pinned
against how the outcomes are actually produced, and one of them asserts the three not-ok causes are
mutually distinguishable — without it, the other three could pass with `cause` hardcoded.

### Added

- **`bundle_artifact` / `parse_artifact_bundle` — one shared convention for a MULTI-FILE harness
  deliverable.** `HarnessPointer.artifact` is a single string, which fits a harness whose output is one
  file (a template, a patch, a document). Plenty produce a FOLDER instead — a write-up plus a PoC plus a
  diff; a Dockerfile plus a compose file plus notes — and until now every harness/client pair had to
  invent its own packing format. That is a silent-failure generator: the two sides agree until they
  don't, and the mismatch degrades into "the child returned junk" rather than surfacing as the wiring
  bug it is (a downstream consumer had already shipped exactly that class of bug against invented wire
  keys). The kit already owns the wire schema, so it owns this too.
  Sections are `===== <name> =====` — deliberately plain text, because the artifact's primary reader is
  a Root LM in a REPL and its second is a human debugging the wire; a client that just wants the whole
  deliverable as CONTEXT needs no parser at all. The marker ESCALATES (`======`, `=======`, …) when a
  file's own content already contains a header line, so a report that QUOTES a bundle cannot truncate
  itself at its own quotation. Round-trips modulo two normalisations applied at PACK time and stated
  in the docstring: line endings become `\n`, and leading/trailing blank lines within a file are not
  significant. A filename that cannot round-trip — empty, or holding a line separator — RAISES rather
  than vanishing silently; names come from whatever the harness authored, often an LM.
  Both halves split lines with one shared helper, deliberately. An earlier cut of this scanned for
  embedded headers with a `re.MULTILINE` regex while parsing with `str.splitlines()` — MULTILINE
  breaks on `\n`/`\r\n`, `splitlines()` on ELEVEN separators — so a header inside CRLF text (a PoC
  quoting an HTTP exchange, a Windows-authored file) escaped escalation and was then honoured as a
  real section break. The quoting file truncated at its own quotation and its tail was absorbed into
  a phantom section, while the key count, the names and the order all still looked correct, so
  nothing downstream could notice. Found by an independent adversarial audit; pinned by a test over
  every one of the eleven separators.
- **`rlm_harness/README.md`, "Delegate to another harness — or be one"** gains three rules dogfooding
  surfaced: use `bundle_artifact` rather than a private format; adapt in `serve.py` when your `run`
  does not already match `run(source: str, run_id=…)` (resolve the caller's text through your own
  public seam, absorb the kit's `run_id` when you have a more meaningful one, and RAISE rather than
  return an empty artifact when the text resolves to nothing — a raise is exit 1, which the caller
  retries then degrades, while an empty artifact is exit 0 and buys the caller a full run over
  nothing); and give the operator an ABSOLUTE `workdir_base`, because the default is relative and a
  child inherits the PARENT's CWD, materialising its run folders inside the caller's project.
- **A per-turn sandbox execution budget + a real cancellation seam for the `pyodide`/`deno`
  interpreter — the reusable gap behind a genuine downstream bug.** While a downstream consumer
  (`ctx-distillery`) was designing a live "Cancel" control for an in-flight run, it traced why the
  obvious `asyncio.Task.cancel()` pattern (already shipped, unexercised against this exact case, in
  a sibling consumer's own studio) does not actually work here: `dspy.RLM`'s sandbox call blocks on a
  plain subprocess pipe read with NO timeout anywhere in dspy's own code, and that call has no
  `await` inside it — the event loop never gets a chance to run cancellation machinery, so a wedged
  Deno subprocess or a spinning model-written REPL cell hangs the run with literally no recourse.
  This kit already solved the identical problem for a DIFFERENT interpreter kind
  (`container_interpreter.py`'s own timer-armed-before-blocking-read, kill-to-unblock watchdog for
  `container`); this change ports that exact idiom to `pyodide`/`deno`, which had no equivalent
  guard. Two independent knobs, one mechanism: `RLMConfig.sandbox_turn_timeout_s`
  (`RLM_SANDBOX_TURN_TIMEOUT`, a per-`execute()` safety-net deadline, `None`/disabled by
  default — deliberately NOT matching `ContainerConfig.timeout_s`'s own `120.0`, since this budget
  cannot exclude host-side tool/sub-LM dispatch time and would misfire more often as a result) and
  `RLMTask(cancel_event=a_threading.Event)` (an externally-set cancel for a caller with a "Cancel"
  UI). A fired timeout raises dspy's own recoverable `CodeInterpreterError` (the model retries next
  turn against a freshly-respawned sandbox); a fired cancel raises the new `SandboxCancelled`
  (exported from `rlm_harness`) — deliberately NOT a `CodeInterpreterError` subclass, so it is a genuine,
  non-recoverable run-ending failure. `_retry.py`'s `run_with_retry` gained a generic `non_retryable`
  allowlist parameter (dspy-free, matching its own existing design), and `RLMTask.arun()` passes
  `SandboxCancelled` through it — without that wiring, the retry engine's own blanket
  `except Exception` would have silently retried an explicit cancel, transparently respawning the
  sandbox and restarting the whole trajectory from scratch. Both knobs are `None`/unset by default
  and cost nothing when unset: no watcher thread is ever created, and every existing caller (all four
  of this kit's real downstream consumers today) is byte-identical to before this existed. Went
  through three rounds of adversarial design review before implementation — round 1 found the
  retry-engine interaction and a watchdog design that didn't survive dspy's own internal
  respawn-and-retry recovery; round 2 found the fix for the latter still didn't check the fired
  reason on a clean return, plus an accidentally-unconditional watcher thread; round 3 found one
  remaining exception type (`SyntaxError`) that could still race past the guard. See `CLAUDE.md`'s
  new sandbox-watchdog invariant and `rlm_harness/README.md`'s "Sandbox turn timeout + cancellation"
  section for the full mechanism.
- **`rlm_harness.rubric` — reward-free rubric primitives (consumer-driven promotion).** The pydantic types
  `Criterion` / `RubricCriteria` / `CriterionFact`, plus `rubric_to_meta` / `rubric_from_meta` /
  `validate_rubric` and a pure `criteria_facts(criteria, facts, lens)` assembly loop, lifted from the
  byte-identical rubric substrate several downstream consumers had each copied (and had begun to drift).
  In core (pure pydantic, dspy-free, eager-exported) like `run_label_bundle`. **`category` is an OPAQUE,
  caller-defined label** — the kit hardcodes no taxonomy and carries no domain vocabulary; a consumer
  supplies its own category set, criterion descriptions, `trace -> facts` function, and `category -> keys`
  lens, and re-exports the types so its call sites stay unchanged. Scoring stays downstream
  (trajectories-never-reward). The `run_start` meta `rubric` wire shape is unchanged, so existing traces
  re-render/re-export as before.
- **`serve_harness` — the SERVER-side mirror of `make_harness_tool`, so connecting a downstream harness
  needs no bespoke glue (`rlm_harness/serving.py` + `python -m rlm_harness.harness_serve`).** `make_harness_tool`
  is the CLIENT (a parent RLM wraps a harness as a tool via an injected `call_endpoint`); `serve_harness`
  is the SERVER — it turns any RLMTask harness into a process that speaks the delegation contract: reads
  the caller's long text from stdin (→ the child's RLM environment), runs the harness, and prints ONE
  `HarnessPointer` JSON line on stdout (artifact + `run_id`/`trace_path` link to the child's own rollout),
  with exit codes 0=ran / 1=infra so the caller can retry an infra failure. The kit owns all generic
  plumbing (stdin, run_id, CWD isolation, the wire schema, exit codes, keeping the harness's identity +
  tracebacks OFF stdout); the consuming harness supplies only a ~5-line `serve` module in its OWN repo
  mapping its result into a `HarnessPointer` (`to_pointer`) — or, for a flat result, uses the duck-typed
  `python -m rlm_harness.harness_serve <module:run>` with zero files. The operator points the client endpoint
  straight at the harness, no intermediate project. Exports: `serve_harness`, `HarnessPointer`. dspy-free;
  vendor-neutral (the kit names no harness). 10 offline tests. Discoverability: a copyable worked
  example (`examples/harness_serve.py`) + step 6 ("Delegate to another harness — or be one", covering
  BOTH the client and server sides) in the consumer guide (`rlm_harness/README.md`).
- **`make_harness_tool` — delegate a sub-task to ANOTHER rlm-harness harness, wrapped as a tool
  (`rlm_harness/tools/harness.py`).** The promoted, generic "wrap a downstream harness as a tool" shape
  (the base/wrap sibling of `make_model_tool`, which it thinly REUSES for retry/validate/circuit-break),
  adding only a child-rollout LINK. Its reason to exist is the RLM framework's native advantage: an
  input field holds near-unbounded text that dspy injects as the Root LM's REPL environment — so a
  `HarnessInvoke` takes ONE long-text arg and nothing else (contract enforced by shape), and
  `harness_from_endpoint(call_endpoint, *, read_output)` binds that whole context to the downstream
  harness's long-text input, which runs a FULL RLM loop (REPL + its own MCP/skills/fetch) over it. The
  parent records ONE leaf tool_call + a `child_run_id`/`child_trace` link (additive within trace/v1);
  the child owns its own trace/rollout (both reward-free). The kit ships NO transport and NAMES no
  harness — the consumer injects `call_endpoint` (subprocess / in-process / HTTP), so the harness's
  identity lives only in the consumer's runtime config (like `make_command_tool`'s injected `Runner`). A
  dead/slow/looping child degrades (`endpoint_error`/`circuit_broken`), never sinking the parent run.
  Exports: `make_harness_tool`, `harness_from_endpoint`, `HarnessInvoke`, `HarnessInvocation`,
  `HarnessToolResult`. dspy-free; anticipatory (no consumer wired in the kit yet). 11 offline tests.
- **Consumer guide: document the studio paired-extras convention.** "Building a consumer" now
  states that an in-repo `studio/` workspace member must define BOTH `live = ["<consumer>"]` and a
  forwarding `subscription = ["<consumer>[subscription]"]`, so a studio-scoped
  `uv run --package <consumer>-studio --extra live --extra subscription …` is portable across every
  downstream (a studio-scoped uv command resolves extras against the member, not the root).
- **`rlm_harness.testing.assert_repl_safe(tool)` — enforce the "REPL tools expose explicit params" invariant.**
  Any callable injected into the RLM REPL has its sandbox proxy built from `inspect.signature(func)` (both
  backends), so a `*args`/`**kwargs` param — or a required param after a defaulted one — silently breaks
  the model's ability to call it (the `_make_tool` kwargs bug). This convention was documented but never
  tested; the helper asserts it and `tests/test_repl_safety.py` sweeps every shipped factory. A consumer
  exposing its own tools should assert the same. New CLAUDE.md invariant records the rule.
- **`rlm_harness.testing` — drive the RLM forward path OFFLINE (`ScriptedInterpreter` + `scripted_lm`), plus
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
  Deno, no network**. `rlm_harness.testing` imports dspy LAZILY, so `import rlm_harness` / the dspy-free modules
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
  runtime check, not a Python dep — `import rlm_harness` stays dspy-free AND docker-free via a lazy import
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
  `make_command_tool` / `CommandResult` export from `rlm_harness.tools`. (Necessary shape, not premature:
  `dspy.RLM`'s pyodide/deno interpreter is WASM Python and cannot spawn a subprocess — verified — so
  shell execution can only come from a host-side tool.)
- **`make_json_schema_validator` — validate a parsed object against a JSON Schema (draft 2020-12).**
  The generic base for the "validate against an official, vendored, version-pinned upstream JSON
  schema" pattern: a consumer vendors the schema file + a refresh script (the provider-specific half)
  and layers its own bespoke checks on top; the kit owns only the validator wiring. Returns the
  violation messages (`"<path>: <reason>"`, truncated at `max_errors` so a huge invalid doc can't
  flood the trace) for a parsed dict — parsing stays the consumer's job, so it composes with any
  extract/parse step. `jsonschema` is an OPTIONAL dependency (`rlm-harness[jsonschema]`), imported lazily
  so `import rlm_harness` and the dspy-free `tools` package stay lean. Consumer-driven: a downstream
  consumer was hand-rolling structural gates that drift from the real upstream format; this is the
  reusable half of moving to authoritative schema validation.
- **`get_sub_lm` promoted to the public surface** (`rlm_harness.get_sub_lm`; lazy re-export, keeps
  `import rlm_harness` dspy-free). Returns the base sub-LM `configure` built — the instance a consumer
  wraps with `intercept_sub_lm` before passing as `RLMTask(sub_lm=...)`. Consumer-driven: TWO
  independent consumers were reaching into `rlm_harness.runtime.get_sub_lm` (a submodule internal) because
  the kit exposed no public way to get the configured sub-LM; per the "add a named hook, don't reach
  into a `_private` name" rule it is now that hook. Using it (vs reconstructing `dspy.LM(cfg.sub_model,
  …)`) keeps a single source of truth so the wrapped model can't drift from `configure`.
- **Public LM-injection seam + `get_config` accessor** (`configure(cfg, main_lm=…, sub_lm=…)`,
  `rlm_harness.get_config`). `configure` now accepts a pre-built `main_lm` / `sub_lm` and uses it
  verbatim instead of constructing one from config — a `dspy.utils.DummyLM` in tests, or a cached /
  custom client in production. `get_config` (lazy re-export) reads the active `RLMConfig` back.
  Consumer-driven: consumer test suites (and the kit's own) were poking private
  `rlm_harness.runtime._STATE` to inject a fake LM because there was no public path; this closes that
  reach-in. Backward-compatible (keyword-only, default `None`); no wire-format change.
- **The trace/v1 `EVENT_*` type constants are now exported** (`rlm_harness.EVENT_RUN_START`,
  `EVENT_MAIN_STEP`, `EVENT_SUB_CALL`, `EVENT_TOOL_CALL`, `EVENT_FINAL`, `EVENT_RESULT`,
  `EVENT_RUN_END`). A trace reader matches on these instead of hardcoding wire strings like `"result"`.
  Additive to `__all__`; the strings themselves are unchanged and still pinned by `test_contract.py`.
- **Reusable resolved-IP SSRF guard for the `direct`-fetch pattern** (`rlm_harness.tools.resolved_host_is_safe`
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
- **MCP client — connect an external MCP server's tools to an RLM** (`rlm_harness.mcp.mcp_tools`,
  optional `rlm-harness[mcp]`). `with mcp_tools(server) as tools:` connects to someone else's
  [MCP](https://modelcontextprotocol.io) server (a local stdio command, or a remote streamable-HTTP
  URL), discovers its tools, and yields them as ready-to-use `dspy.Tool`s for `RLMTask(tools=…)`;
  the connection is live for the block and torn down on exit. rlm-harness is a CLIENT only (never a
  server, bundles none). The crux: the MCP SDK is async but dspy.RLM invokes tools synchronously, so
  the session runs in a dedicated background thread + event loop and each call bridges via
  `run_coroutine_threadsafe(...).result(timeout)` — dspy's own `Tool.from_mcp_tool` yields an ASYNC
  tool for `ReAct.acall`, unusable on the RLM sync path. A hung tool call trips the `timeout` and is
  cancelled (so it can't wedge the serial session); a start failure still tears the bridge down (no
  leaked thread/subprocess). Both stdio and streamable-HTTP transports are integration-tested
  against a real server. Each call records a `tool_call` (trace/v1, no schema change). MCP tools run HOST-SIDE (outside the sandbox; a stdio server is a spawned
  subprocess) — treat the server as a trusted dependency and its output as a prompt-injection
  surface. `mcp.py` lives outside the dspy-free `tools/` and `mcp_tools` is a lazy export, so
  `import rlm_harness` stays dspy/mcp-free.
- **The extension contract is now documented AND guarded** (`CLAUDE.md`, `README.md`,
  `tests/test_contract.py`) — so the next consumer builds on rlm-harness without reverse-engineering it.
  A README **"Building a consumer"** section states the 5-step recipe, the promotion rule
  (generic → kit via the base/wrap split; specific → consumer), and the rollout-vs-reward stage
  boundary; three matching CLAUDE.md hard invariants pin the contract — the trace is a VERSIONED
  `rlm-harness/trace/v1` wire format (additive-only within v1; removing/renaming/re-typing an event type,
  envelope key, or established payload field is a `v2` break), the kit produces TRAJECTORIES not
  reward (every exporter carries a `reward=` hook the downstream trainer fills), and the public
  surface is `__all__` (consumers EXTEND via subclass + base/wrap + read-the-trace; if a seam is
  missing, ADD a public hook — how `recorder_scope` / `bind_recorder_to_sub_lm` were born — never
  reach into a `_private` name). **`tests/test_contract.py`** freezes that v1 surface — SCHEMA, the
  seven `EVENT_*` strings, the recorded-event envelope, and the `export_actions` / `export_sft_turns`
  / `export_rl` record shapes + the public `__all__` — so a kit change that would silently break a
  downstream reader (a consumer's report + RL export, a consumer UI's replay) fails HERE in
  the kit's own suite, not opaquely in the consumer. (+7 tests → 148.)
- **README "Built with rlm-harness" adopters section** (`README.md`, `CLAUDE.md`). A single,
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
- **`ClaudeAgentLM` — run rlm-harness on a Claude Pro/Max SUBSCRIPTION (no API key), now shipped in
  the kit.** `from rlm_harness import ClaudeAgentLM` behind the opt-in extra
  `pip install "rlm-harness[subscription]"`: a `dspy.BaseLM` adapter over the official Claude Agent SDK
  (the sanctioned path for individual subscribers), injected through the existing
  `configure(main_lm=…, sub_lm=…)` seam — zero kit-core changes. Every call is a pure completion
  (`tools=[]`, `setting_sources=[]`, no agent loop), the async SDK is bridged to dspy's sync/async
  seats via a background event loop (the `mcp.py` pattern), concurrency is capped at 2 with a single
  rate-limit backoff (politeness: ordinary individual use, not batch rollouts), the kit's default
  `json` adapter's `response_format` is translated to the SDK's native schema-validated
  `output_format` (with `max_turns` headroom for its validation step), and a leftover
  `ANTHROPIC_API_KEY` fails fast so a subscription run can't silently bill API credit. Lazily
  exported (PEP 562) with the `claude-agent-sdk` import deferred to construction, so `import rlm_harness`
  stays dspy/SDK-free (the `mcp_tools` pattern). Previously lived only under `examples/` (not in the
  wheel), which forced every downstream consumer to vendor a byte-identical copy;
  `examples/claude_agent_lm.py` now shrinks to the runnable demo.
- **`run_label_bundle(runs, /, **label_fns)` — reward-free per-run LABEL surfaces** (`dataset.py`,
  public + contract-pinned). A companion MAPPER to the exporters: `{surface: {run_id: fn(events)}}`,
  where each keyword is a consumer-supplied fn turning one run's events into a dict of intrinsic labels
  (validity flags, objective metrics, a rubric's deterministic per-criterion facts) that ride BESIDE the
  trajectory records — so a downstream trainer reads ONE canonical bundle shape instead of each consumer
  re-deriving it. `runs` is positional-only so a label surface may itself be named `runs`; `reward` is a
  REFUSED surface name (rlm-harness produces trajectories, never reward — the trainer composes reward from
  these labels plus its own credit assignment). Consumer-driven: promoted from a downstream consumer's
  per-run labelling so every consumer shares one bundle shape.
- **Public multi-server MCP catalog: `McpConnection` + `McpCatalog` + `result_text`** (`mcp.py`,
  optional `rlm-harness[mcp]`). Alongside the single-server `mcp_tools(...)` convenience (one server's tools
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

- **`discovery="inject"` no longer points `read_skill` at a `list_skills` tool it never registers.**
  A tool's docstring IS the description the planner reads, and `read_skill`'s said "Use list_skills
  first to see names" unconditionally — but under `inject` the catalog comes from the caller's
  injected manifest and `list_skills` is deliberately NOT returned. So the one mode built for
  progressive disclosure shipped an instruction naming a symbol that raises `NameError` in the REPL.
  The description is now set per mode, and an unknown skill name reports the available ones inline
  (under `inject` there is no `list_skills` to recover with, so the miss has to carry them). Found
  by dogfooding `inject` in a downstream consumer.

- **`mcp._make_tool` now exposes each MCP tool's REAL param names to the RLM REPL (was: a single
  `kwargs`).** dspy.RLM builds the in-sandbox tool proxy from `inspect.signature(tool.func)` — NOT
  `dspy.Tool.args` — so the old `def call(**kwargs)` wrapper registered a proxy whose only param was
  literally named `kwargs`. The model then called e.g. `get_vulnerability(kwargs="CVE-…")` and a strict
  server (`additionalProperties:false`, expecting `id`) rejected it as an unexpected property — EVERY
  in-REPL MCP tool call was broken (only a HOST-side call, which passes `tool(id=…)` directly, worked).
  `_make_tool` now stamps `call.__signature__` from the tool's `inputSchema` (properties → KEYWORD_ONLY
  params, REQUIRED-FIRST so the generated Deno `def` compiles, zero-arg tools included, malformed-schema
  fields tolerated), and drops `None`-valued args so an omitted optional isn't posted as JSON null. The
  fix is on the wrapped func, so it protects BOTH the Deno and container backends (each reads the func
  signature). Regression tests in `tests/test_mcp.py`; the class is now guarded kit-wide by
  `assert_repl_safe` (see Added).
- **The co-dev editable overlay no longer shadows a consumer's namespace `tests/`** (dropped
  `tests/__init__.py`; guard test in `tests/test_packaging.py`). A consumer co-develops rlm-harness by
  overlaying an editable install (`uv pip install -e ../rlm-harness`), whose bare-path `.pth` puts the repo
  ROOT on the consumer's `sys.path`. Because rlm-harness shipped `tests/__init__.py` (a REGULAR package), a
  consumer's `import tests` bound to rlm-harness's `tests/` and SHADOWED the consumer's own namespace
  `tests/` — regardless of `sys.path` order (PEP 420: a regular package at any later entry beats an
  earlier namespace portion) — breaking its `from tests.conftest import ...` collection. rlm-harness's
  `tests/` is now a namespace dir (the `__init__.py` was empty; the suite is unchanged), so `rlm_harness`
  is the only regular package in the repo and the shadow is gone; a guard test keeps it that way. Wheel
  users were never affected (the wheel ships `rlm_harness` only). Note: rlm-harness and a consumer may share a
  test basename (e.g. `tests/test_config.py`) — harmless under pytest (namespace-dir tests import by
  file, not package path), but an explicit `import tests.test_config` in consumer code could resolve to
  rlm-harness's copy under the overlay; keep test basenames project-unique if that ever matters.
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

- **`McpConnection.close` now reaps a WEDGED connect (two-phase close), and `McpCatalog(connect="lazy")`
  is per-transport** (`mcp.py`, surfaced by a consumer's large-toolspace path). A connect that wedged — a
  tarpit that accepts TCP but never completes the MCP handshake — left `close()`'s `_stop.set()` a no-op
  (nothing awaited it yet), so `close()` burned a SECOND full `timeout` and then LEAKED the background
  thread plus its socket / stdio child forever (this hit the *eager* path too, e.g. a bad server at
  startup). `close()` is now two-phase: a graceful stop bounded by a short grace, then — if still alive —
  it cancels the serve task to unwind the session/transport `__aexit__` (closing the httpx stream /
  terminating the stdio child) and reap the thread. Separately, `connect="lazy"` is now PER-TRANSPORT:
  URL (streamable-HTTP) servers defer to first `load()` (safe mid-run — the handshake runs on the
  connection's own thread+loop, the caller's wait is `timeout`-bounded, and a wedged connect is
  cancelled+reaped), while stdio servers still connect eagerly (deferring a local spawn buys nothing). The
  prior blanket "a subprocess spawn inside the loop can hang asyncio" framing was an overstatement — every
  caller wait was already bounded; the real defect was the un-reaped wedged close. Opt-in `"lazy"` stays
  experimental. HTTP-transport tests force a direct connection (`NO_PROXY=*`) so a machine's system proxy
  can't mask a tarpit/refused as a proxy response.
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
  **`rlm_harness/README.md` ("the guide")**, which GitHub renders when browsing the package folder and
  hatchling ships inside the wheel. Cross-references in `CLAUDE.md` / `CONTRIBUTING.md` now point at
  the guide; external deep links into the old top-README sections need re-pointing.

### Initial scaffold

Where it started, before the layers above were built on top:

- `RLMConfig` + `configure`, `RLMTask`, the retry/validation engine (`_retry.py`), the sandbox
  security guard (`sandbox.py`), tools (schema validator, SSRF-guarded fetch), examples, and tests.
