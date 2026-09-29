# Changelog

All notable changes to `rlm-harness`. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/). Versions track
`rlm_harness/__init__.__version__` and `pyproject.toml` (kept in sync).

## [Unreleased]

### Fixed

- **`RLMTaskError`'s message carried none of the cause.** Two user reports resolved to one line. A
  missing Deno and a `claude-agent-sdk` CLI failing under a different uid both arrive at `_retry.py`'s
  wrap site as an exception that already says what to do: dspy names `pip install "dspy[deno]"`, and the
  SDK's `ProcessError` folds in the CLI's exit code and stderr. Wrapping replaced both with a sentence
  naming only the output field. `__cause__` and `run_end.payload.error_chain` still carried it, but
  neither is what a consumer sees when its CLI catches `RLMTaskError` and prints `str(e)`, which is how
  an environment problem with a one-line fix reads as the harness going in circles. The cause now goes
  through the existing public `short_error`, so this path cannot flood a terminal either.
- **`atomic_write_text` translated newlines on Windows, and leaked `mkstemp`'s descriptor when
  a bad `encoding` substituted a bookkeeping error for the caller's own.** Without `newline=""` the
  text layer rewrites every `\n` as `\r\n` on Windows, so a tool whose contract is "write this
  content" did not, and `atomic_write_stream`, its sibling in the same module, opens `"wb"` and
  therefore disagreed with it on the same platform. A no-op on POSIX, which is why it survived; pinned on BYTES, since a
  text read-back translates the bug away and passes either way.

  The second half is a fix for this release's own first attempt at it. `os.fdopen` takes ownership of
  `mkstemp`'s descriptor only once it succeeds, so an `except BaseException: os.close(fd)` looked like
  the obvious repair. **CPython's `io.open` already closes the fd on most failure paths**, so that
  close was a DOUBLE close raising `OSError: [Errno 9]` over the real error and demoting it to
  `__context__`, and in a process with other threads it could close a descriptor one of them had since
  been handed at the same number. Reachable from the public surface, since `make_write_file_tool` and
  `make_edit_file_tool` pass a caller's `encoding` straight through. `codecs.lookup(encoding)` before
  `mkstemp` removes the failure instead of handling it: the caller now gets their own `LookupError`,
  with no descriptor and no temp file ever created.

- **`resolve_within_root` RAISED instead of refusing for a path with no shared anchor.**
  `os.path.commonpath` does not answer when two paths share no anchor, it raises, and on Windows
  `ntpath.commonpath(["C:\\root", "D:\\evil"])` is `ValueError: Paths don't have the same drive`.
  `os.path.join` lets an absolute candidate replace the root outright, so a `"D:/evil"` argument
  reaches it. Every caller treats only `None` as the refusal, and `make_write_file_tool` /
  `make_edit_file_tool` promise their tool returns a string "(never raises)" for a path that escapes
  the root, so on Windows the tool raised out of the REPL for an input the guard exists to reject
  calmly. A path on another drive is definitionally outside the root, so it is a refusal now.
  Pre-existing and POSIX-invisible; surfaced by adding the Windows axis rather than by running it.

### Added

- **`ci.yml` gains a `test-windows` job.** `pyproject.toml` has declared
  `Operating System :: OS Independent` since 1.0.0 and nothing checked it: every other job here is
  `ubuntu-latest` bar `install-check.yml`'s macOS leg, so the third platform the classifier promises had
  no reader. One interpreter, because a Windows break is OS-shaped rather than version-shaped.

  **Its first run answered the question it was added to ask: 12 failures, 1045 passes, and every failure
  was a POSIX assumption in a TEST rather than a defect in the library.** Mode bits do not exist on
  Windows, `os.kill(pid, 0)` is not an existence probe there, a backslash is a path separator so a
  "literal `pkg\util.py`" check contradicts its own neighbour, `terminate()` maps to an uncatchable
  `TerminateProcess`, and a daemon that answers `docker info` can still refuse every Linux image. All
  twelve are platform-aware now, and `rlm-harness[mcp]` works there: the stdio child spawns and is
  reaped. **A green still does not mean a live run works on Windows**: Deno starts lazily on the
  sandbox's first turn and no job in this repo starts one, so the WASM sandbox is unproven on every
  platform, and `pip install "dspy[deno]"` resolves a `win_amd64` wheel and no `win_arm64` one.

### Changed

- **`CLAUDE.md` is gone; the agent guide is `AGENTS.md`, and its rules moved into `docs/`.** Claude Code
  reads `AGENTS.md` natively, so a Claude-specific filename costs cross-agent portability and buys
  nothing. The split is the substance: 532 lines loaded on every request, about 13k tokens whether or not
  any of it applied, became a root that holds only what is true for every task, plus the trigger for
  each of [`docs/INVARIANTS.md`](docs/INVARIANTS.md) (the invariants),
  [`docs/VERIFY.md`](docs/VERIFY.md) (the CI axes and what each cannot see),
  [`docs/CONVENTIONS.md`](docs/CONVENTIONS.md) (versioning and the consumer loop) and
  [`docs/HANDOFF.md`](docs/HANDOFF.md), which was `.claude/rules/handoff.md` and which Claude Code
  auto-loaded. **No line count is quoted for the root on purpose**: the first draft of this entry said
  63, the review round after it made the file 66, and the round that fixed THAT made it 73, while this
  same entry records the change that grew it. A self-reported size is a claim the next edit falsifies.

  **What moved is the CONTENT; every TRIGGER stayed resident**, because progressive disclosure fails on
  a tripwire: an agent that has not read a rule does not know the rule applies. So the root names the
  moment for each document rather than describing it.

  A split creates one defect class that nothing in the diff looks like: a cross-reference that was true
  while one file held both ends. An independent review found four, including the one place the first
  pass repointed by pattern instead of by target. **The sweep a split needs is direction words and
  partial names, not the moved filename.** Two deliberate non-edits: the `CLAUDE.md` in
  `claude_agent_lm.py`'s `setting_sources=[]` comment means the END USER's own file, and this file's own
  mentions record a file that had that name at the time.
- **This file is no longer a running account.** Every release entry in it was rewritten to
  carry the settled facts, the measurements, and the one reason that makes each non-obvious, and to drop
  investigation narration, process steps and design deliberation, roughly halving it from the 3,174
  lines it held at `f1595ac`. The durable rules those entries used to re-teach live in
  `docs/INVARIANTS.md`, which is where a reader is sent instead.

  **An independent review then found the first pass had overclaimed itself.** It said "no claim and no
  number removed", and that was false on nine counts: a dangling reference where 1.2.1 cites what
  "1.2.0 left open" after the referent was deleted, two consumer-actionable warnings (that the pre-1.6.0
  corpus cannot be split by kit version at all, and that neither 1.5.0 floor breaks a consumer because
  nothing pins dspy or mcp directly), two verification runs, and the per-release version matrices that
  recorded which dspy, Python and mcp versions each release was actually verified against. All are
  restored in the entries they belong to. **A subtractive rewrite cannot audit itself**: the same pass that decides a sentence
  is narration is the one that would have to notice another entry quotes it, and it reads both with the
  same eye. Two of the three greps used to check this file also gave confident wrong answers, because
  re-wrapping moves a quoted phrase across a line boundary: check it whitespace-insensitively, and read
  cross-references per entry rather than by pattern.

  **Two more rules came out of the rounds that followed, and both are about writing rather than
  cutting.** A document must not assert a fact about ITSELF unless a test pins it: a line count, an
  entry count, an invariant count, a job count. Each of those was correct when written and wrong within
  a round, because the commit that states such a number is usually the one that changes it, and a
  reader cannot tell a stale one from a live one. `tests/test_contract.py` is the other exit where a
  count is genuinely load-bearing. **But a test MENTIONING the number is not a pin: it has to go RED
  when the count moves.** `test_event_type_strings_are_frozen` names all seven `EVENT_*` constants and
  asserts their values, which catches a rename and catches nothing about an eighth arriving, measured:
  adding `EVENT_FOO` to `trace.py` passed the entire suite. So "seven" was quoted in three documents on
  the strength of a test that did not pin it, inside the very rule written to stop that. There is a real
  pin now, `test_the_event_type_SET_is_closed`, in the strict shape `RUN_FACT_KEYS` already used, and it
  goes red on an eighth. Adding an event type stays legal under additive-only-within-v1; the visible
  diff to a contract test is the point, since every downstream reader has to learn the new type.

  **And the discriminator for which exit to take, which is the part that stops the mechanism being
  applied past the reason it worked: pin a count when drift imposes cost on someone OUTSIDE this
  repo, drop it when drift only makes a sentence wrong.** That is why the `EVENT_*` set is pinned and
  the CI job roster is not. An eighth event type is work every consumer reading `trace/v1` has to do.
  A ninth CI job costs nobody anything, so a YAML-parsing test would charge the same price and buy
  only doc accuracy, going red on a legitimate addition whose correct fix is to edit the test.
  `docs/VERIFY.md` states the two PROPERTIES that bullet actually rested on instead, since a property
  survives someone adding a job and a roster cannot. And RESTORING a dropped measurement is exactly as capable of inventing one as cutting
  was of losing it: this entry's own restoration pass wrote "verified on both dspy 3.2.1 and 3.3.0"
  into 1.2.0, a release whose headline is that 3.2.x no longer installs. Diff a restoration against the
  original text, never against memory of it.

  **And the rule the eight rounds actually earned, which is not "be more careful".** The asymmetry
  driving them: a FINDING is verified against reality, while a FIX is verified against the finding,
  so a fix inherits the finding's evidence and then writes new explanatory prose that carries none.
  More care lowers the per-sentence error rate; it does not change that ratio. The regularity in this
  range's own record is sharper than the argument: **every subtractive fix held, and every fix that
  replaced one explanation with another needed a follow-up.** Held with no follow-up ever: the root's
  line count, the `(4300)` quote, "the 24 invariants", "ten jobs across four workflows", "all 28
  releases", the CI job roster, `task.py`'s "four fields", and "the matrix midpoint". Needed one: the
  Deno bound (two more sites found later), the `is_dir` story (three files fixed, `ci.yml` missed),
  the litellm ownership (four more sites), and the restoration pass that invented a dspy 3.2.1
  verification. So **prefer the fix that deletes a claim to the fix that replaces it**: a deleted
  claim cannot go stale or be miscited, while a replacement is a fresh liability with the same failure
  modes as the one it replaced. The question before writing a replacement is not "is this true?" but
  **"will I find out when this stops being true?"** If no, delete it, or attach the command that
  settles it, which is what the third tier below does from the other side.

  **The general form of all of this is now a bullet in `docs/VERIFY.md`, beside the Deno bound it
  came from**, with a resident trigger in `AGENTS.md` because the moment it applies is writing a
  sentence. A claim about the world outside this repo is one of two kinds. An EXECUTABLE one, naming
  a version, symbol, path or behaviour of an installed package, is auditable forever, so it will
  eventually be checked and must be re-measured before being requoted. A CORPUS-ATTRIBUTED one
  ("385 runs", "141 traces") is auditable exactly ONCE, at the moment it is written, because no
  review of this repo can reach the corpus; its only control is quoting the corpus size and the
  moment, which "Before believing a count" already demanded. Naming the second half is what stops
  the rule being read as "sweep everything", when half of it can never be swept.
- **The README leads with the portable Deno install.** `brew install deno` was first and
  `pip install "dspy[deno]"` was the alternative, so a Windows or Linux reader met a package manager
  they may not have before the path that works everywhere. It also now states what makes a missing Deno
  confusing rather than merely inconvenient: it starts LAZILY on the sandbox's first turn, so it fails
  at neither import nor `configure()`.

### Documented

- **Four invariants described mechanisms that dspy 3.4.0 changed, while the rules they carry still
  bind.** Corrected because a reader who checks a stated mechanism, finds it false, and concludes the
  rule is obsolete is the failure mode a stale justification actually causes. An `async def` tool no
  longer hands the model an un-awaited `"<coroutine object …>"`: 3.4.0's `invoke_tool` appends
  `_await_in_sync`, which calls `run_until_complete` on the loop `RLMTask.arun` is already running,
  so it raises `RuntimeError` instead. Loud rather than silent, and still a broken tool. dspy's
  `llm_query_batched` submits with `contextvars.copy_context().run`, so its workers DO inherit the
  caller's context, which makes `run_isolated`'s bare thread a CONTRAST rather than the analogue the
  note claimed. The untouched-return-on-unrecognised-shape rule lives in `sub_lm.py`, not in the
  shim, which ends `return [text]`. And `export_sft_turns` has no `reward=` parameter at all, which
  is a stronger version of the reward-free property than "every exporter carries the hook", not a gap
  in it.
- **`AGENTS.md` under-counted the CI axes and left two invariants without a resident trigger.** The
  root now points at `docs/VERIFY.md` for the job list instead of counting them. The two measurement
  invariants fire when you are READING a
  corpus rather than editing a file, so "before editing anything under `rlm_harness/`" never reaches
  them; they have their own moment now. The "paste the output" half of the verify rule moved into the
  root too, since it applies to every done claim rather than only to a stdlib or platform change.

### Fixed (claims about the outside world, found by running them)

- **`make_grep_files_tool` told a consumer, in the `ImportError` they actually read, that stdlib `re`
  "cannot be bounded by ANY pure-Python mechanism, including `signal.alarm`". It can.** Measured: a
  `SIGALRM` handler broke out of `re.search(r"(a+)+$", "a"*40+"b")` at 1.04s, mid-match on a pattern
  that otherwise runs for minutes, because CPython's `sre` engine calls `PyErr_CheckSignals()`
  periodically. Requiring `regex` is still right, on grounds the text did not give and which are both
  true and checkable: `signal.alarm` is MAIN-THREAD-only, and this tool runs from a `dspy.RLM` REPL
  and can be dispatched on a `ThreadPoolExecutor` worker; and it is POSIX-only, so Windows has no
  `SIGALRM` at all.
- **The 600s request default and the retry ladder were attributed to litellm, which is not on dspy
  3.4.0's default path.** `select_backend` returns `native=True` for every LM shape this kit builds,
  plain, with `api_base`, and with `timeout`, and `_engine_spec` is `'auto'`, so the vendored `lm15`
  engine handles the call and there is no 600 anywhere in it. The real owner, measured on
  openai 2.41.0, is the provider SDK: `DEFAULT_TIMEOUT` is
  `Timeout(connect=5.0, read=600, write=600, pool=600)` and `DEFAULT_MAX_RETRIES` is 2. litellm's
  `COMPLETION_HTTP_FALLBACK_SECONDS` is genuinely 600.0, and dspy's `num_retries` is genuinely 3.
  **Every figure was right and the ownership was wrong**, which is exactly why "verified by execution,
  not read off the docs" did not catch it: execution verified the NUMBER. Corrected in `config.py`
  and at four places in the guide.
- **`_dspy_compat.py` asserted the seam dspy 3.4.0 deleted, and its own file contradicted it 90 lines
  down.** It said the interpreter seam is "HARDCODED to the `forward()`-positional form" and that a
  dspy moving it back to the constructor would raise from `aforward`. `aforward` is
  `(self, *, interpreter_factory=None, **input_args)`, keyword-only, the kit passes the factory to
  the CONSTRUCTOR, and `task.py` calls `await rlm.aforward(**inputs)` with nothing positional. The
  predicted failure mode was backwards.
- **`ci.yml` cited a stdlib boundary that never existed.** It said `ZipInfo.is_dir()` "moved between
  3.11 and 3.12". Measured: 3.11.0 indexes `filename[-1]`, 3.11.13 does not, so the move was a PATCH
  within 3.11 and the framing was wrong when written rather than stale now. It cites
  `ZipFile.writestr("")` instead, which still raises on 3.11 and returns fine on 3.12+.

  **This one survived the round that fixed its siblings because that round's report named three files
  and nobody swept for the claim.** Exactly the failure this range already recorded against
  `ca29731`, which named three Deno sites and fixed two. A finding without a sibling sweep is half a
  finding, in either direction: the reporter owes the sweep as much as the fixer does.

- **`task.py`'s `custom_types` workaround has a dead premise.** It said dspy "silently drops
  `custom_types` when `instructions is None`". It does not, on 3.4.0: measured, without
  `custom_types` both `None` and `""` raise `ValueError: Unknown name`, and with it both resolve.
  The line stays because it costs nothing; the comment no longer asserts the mechanism.

  **Getting the CONTROL right is the whole experiment, and it took three attempts.** dspy falls back
  to walking the call stack with `sys._getframe`, so a type reachable by NAME from any frame makes
  the control PASS, which proves nothing. Binding it at module level fails that way; so does creating
  it inside the function that calls `dspy.Signature`, because that frame holds the name too. Only a
  type minted in one function and passed to another under a different name is genuinely unreachable.
  A passing control is the signature of this mistake, and the invariant about not relying on
  call-stack resolution is exactly what the failing control demonstrates.
- **`make_extract_archive_tool`'s standing justification named zip as the example it is least true
  of.** "`zipfile.extractall()`/`tarfile.extractall()` are not safe by default: a malicious entry can
  carry an absolute path, a `..`-traversal path" is wrong for zip on every supported version:
  measured, entries `../evil` and `/abs_evil` both land INSIDE the destination, and zipfile never
  creates symlinks. `tarfile` does escape, on 3.11 and 3.13, and stops on 3.14, whose default `data`
  filter raises `OutsideDestinationError`, which is inside dspy's own `>=3.10,<3.15`.

  So the stdlib's behaviour is format-dependent AND version-dependent, which is a better argument
  for owning the check than the one it replaces: this tool refuses by its own containment on every
  format and every version, a promise the stdlib does not make. No security defect: the code always
  did its own `resolve_within_root` containment and is unchanged.

  Worth recording how this was found. The claim also sits in a dated 1.3.0 entry, and the instinct
  was to leave that alone, correctly, since this project's convention is to retract in the CURRENT
  entry rather than edit a historical one. **Sweeping first moved the finding off the historical
  entry and onto two LIVE ones**, which is where it mattered. Two further claims measured wrong in
  the same pass, PyPI's refusal of `rlm-kit` attributed to PEP 503 normalisation when
  `canonicalize_name` leaves the two names distinct, and litellm described as chaining with `from`
  when it chains through `__context__` only, have no live site at all and are left standing as the
  record of what was believed.

- **Two more stdlib claims went stale under the FLOOR rather than above it, which is a different
  failure from the one the docs teach.** `grounding.py` quoted the integer-conversion error as
  `Exceeds the limit (4300)`; measured, that is right on 3.11.0 and reads `(4300 digits)` from
  3.11.13 onward. Together with `ZipInfo.is_dir()`, which indexed `filename[-1]` on 3.11.0 and uses
  `endswith` by 3.11.13, that is two claims written against 3.11.0, invalidated by a 3.11 PATCH, and
  still cited in shipped files as live 3.11-versus-3.12 differences. `docs/VERIFY.md`'s 3.11 bullet
  was framed entirely around "the versions differ" and now names the one that actually happens:
  **re-measure a 3.11 claim against CURRENT 3.11.x, because the axis is the patch level.**
- **"The project's default interpreter (3.12 today)" describes nothing, and a CI decision rested on
  it.** `requires-python` is `>=3.11` with no `.python-version`, so `uv run` takes the newest
  interpreter present: this checkout resolves to **3.14.3**, which the CI matrix (3.11, 3.12, 3.13)
  does not test at all. **So every "N passed" reported from a local run here is qualified by that**,
  and `ci.yml`'s "3.12 because it is the project's default interpreter" was resting a real choice on
  a non-existent fact. It now says the matrix midpoint, which stays true. `AGENTS.md` says to check
  `uv run python -V` before trusting a local pass.
- **The three claims that cannot be checked from here now carry the command that settles each**, per
  the middle tier of the two-kinds-of-claim rule: `@runtime_checkable` in dspy 3.2.1, hatchling
  1.27's own default metadata version, and the absent `win_arm64` `deno` wheel. Writing the command
  beside the claim is what keeps them out of the corpus bucket, where they would be treated as
  permanently on trust while being one command away.
- **Two reported leads were dropped as false positives rather than passed on**, which is worth
  recording because the cost of a false finding is a pointless edit to correct prose: `discover.py`
  already distinguishes pathspec's deprecated `gitwildmatch` FACTORY NAME from the `gitwildmatch`
  SYNTAX the docs name, and every live `UsageTracker` site is already version-scoped, with 3.4.0
  still ADDING same-typed values and only the mixed case becoming first-wins.

### Known, not fixed here

- **The edit tool's READ side still translates newlines** (`tools/edit.py:200` opens without
  `newline=""`). Unlike the write side this is a judgement, not a bug: with translation a model's `\n`
  search string matches a CRLF file but the edit rewrites every line ending in it; without it the file
  is preserved and the search misses.
- **`requires-python` permits 3.14, and nothing else in the project acknowledges it.** Measured:
  `requires-python` is `>=3.11` with no upper bound, the classifiers stop at 3.13, the CI matrix is
  3.11/3.12/3.13, and `import rlm_harness` works on 3.14.3. So a consumer can install on an
  interpreter this project neither declares nor tests, and the only evidence it works is that a local
  venv happened to resolve there during this review. **Recorded rather than decided**, because the
  three exits are a support-policy call and not a defect fix: cap `requires-python` at `<3.14`, or add
  3.14 to the matrix and the classifiers, or leave it open and say in `AGENTS.md` that the floor is
  tested and the ceiling is not.
- **A THIRD flake, observed once and previously unrecorded**:
  `tests/test_mcp.py::test_mcp_catalog_lazy_is_per_transport`, a `TimeoutError`. It spawns a stdio
  MCP subprocess under `McpCatalog(..., timeout=5)`, the tightest budget in that file, and it failed
  in the slowest full-suite run recorded during this work (196.7s against a normal 80-116s) while
  passing 5 of 5 in isolation. **Not reachable from this range's diff**: `rlm_harness/mcp.py` is not
  in it at all, and the only change to `tests/test_mcp.py` is a `_process_is_alive` helper used by a
  different test. Recorded rather than left for the next person to rediscover, because this review
  spent real effort twice on flakes whose written characterisation was wrong, and an unrecorded one
  costs strictly more than a wrongly-described one. One observation is not a rate.
- **Two further pre-existing test flakes, both outside 1.14.0's diff, both worth one issue.**
  `tests/test_isolation.py::test_sigterm_ignoring_factory_escalates_to_sigkill` is not a slow-host
  timing flake: it reproduced 8 times in 10 on a loaded machine against a LOWER-bound assertion
  (`assert 1.5 < elapsed`), observed failing at 1.10s. A fast failure means the child died on
  `terminate()`, so the escalation the test is named for did not fire because the child had not yet
  installed its handler: a child-startup race and the hollow-green direction. Not macOS-specific, since
  `isolation.py` pins the `spawn` context everywhere, so a shared runner is the reproducing condition.
  `_patch_process_capture` assigns `ctx.Process` on the `get_context("spawn")` SINGLETON while
  `monkeypatch` restores only `get_context`, so that override leaks into the rest of the session. The
  fix is to observe the escalation directly rather than through wall-clock, and to restore
  `ctx.Process`. **Record the SIGNATURE, not a rate: every failure lands UNDER the 1.5s bound and
  every pass lands at 2.07s or more.** Measured across 28 isolated runs by two readers: 10 here gave
  9 passes at 2.07-2.53s and one failure at 1.113s; 18 elsewhere gave failures at 1.49-1.65s against
  passes at 2.07-2.73s. Bimodal, not a continuum, which is exactly a race resolving two ways and
  confirms the child-startup cause above. **It is neither load-dependent nor suite-dependent**: one
  batch failed three times with nothing else running and the next passed eight times immediately
  after, and it reproduces with only its own test selected.

  Its recorded "8 times in 10 on a loaded machine" is the third flake characterisation in this file to
  be wrong, and all three were wrong the same way: **a condition fitted to a small sample.** At a
  roughly bimodal 20%, six runs reads as 50% and eight reads as 0%, so "never reproduced for me",
  "8 in 10" and "only under load" are the same event sampled too few times. A signature is checkable
  in ONE run and cannot be wrong the way a rate can. `tests/test_tool_durations.py::test_the_fill_keeps_sub_millisecond_resolution` is the
  mirror shape: a 1 ms UPPER bound over `sum(range(20000))` plus the recording plumbing.

  **It has no reproducing condition. It has a heavy tail, and three attempts to name a trigger were
  all fitting a story to a coin flip.** The recorded reading was "the cold or loaded first run of a
  batch"; it then failed a full-suite run at 96s, an ordinary unloaded time here, so that was replaced
  with "the full-suite CONTEXT"; and that is wrong too. Measured three ways, independently:

  - **The timed work alone exceeds the ceiling unconditionally.** `sum(range(20000))`, sampled 600
    times with nothing else running, gives p50 0.507 ms, **p95 1.437 ms**, p99 3.619 ms, max 18.2 ms,
    and 10.0% of samples at or above the test's 1 ms bound. The assertion's window is wider than that,
    since `record_tool_call`'s plumbing is inside it.
  - **It fails in ISOLATION**, which is what settles it: 1 failure in 20 runs of that test alone on a
    quiet host, and 3 in 25 on a busy one. No suite, no accumulation, no order effect (there is no
    `pytest-randomly` here, so order is deterministic, and a deterministic order cannot produce
    intermittent failure).
  - **A handful of passes is not evidence against it.** At these rates, 8 consecutive passes is about
    a 1-in-3 event and 3 is about 3-in-4. Both were read as pointing at a condition; neither was
    informative. That is the "what would this reader return if everything were working" check in
    `docs/INVARIANTS.md`, failed on a sample of three.

  So a 1 ms ceiling over ~0.5 ms of real work, against a distribution whose p99 is several times its
  median, fires at a rate that is a HOST property rather than a condition anyone can reproduce on
  request. Linux CI has been green on it throughout, so that tail is far tighter. Whether load raises
  the rate (5% quiet against 12% busy) is suggested and not established: small n, two setups.

  **A lead for whoever takes it, not a fix**: the property under test is that the fill is NOT rounded
  to milliseconds, and proving that by timing real work makes it a bet on host speed. The sibling test
  in the same file already proves precision the robust way, passing an explicit `duration_s=0.001` and
  asserting it round-trips at 6 dp. Asserting round-trip rather than magnitude would decouple the
  property from the host entirely.

## [1.14.0] - 2026-09-26

dspy 3.4.0 support. Requires dspy `>=3.4.0,<3.5.0`.

### Changed

- **dspy is now CAPPED, not just floored: `>=3.4.0,<3.5.0`, and pydantic `>=2.11.0`.** The cap
  reverses the no-upper-bound policy this file and `pyproject.toml` used to state, which was tested
  twice and broke a consumer both times (3.3.0's three renames, CHANGELOG 1.0.1; 3.4.0's removal of
  the interpreter seam). `<3.5.0` is upstream's own boundary rather than a guess: several 3.4.0
  deprecations say "removal in 3.5". `uv run --with "dspy==<newer>"` overrides the bound, measured,
  so `dspy-latest.yml` still tests the newest dspy. See `docs/INVARIANTS.md`, "Every dspy API
  difference is resolved in `_dspy_compat.py`".
- **Older releases stay broken on dspy 3.4.0 and the cap cannot reach back.** PyPI metadata cannot be
  amended after upload, so `pip install rlm-harness==1.13.0` still resolves dspy 3.4.0 and still
  fails, as does every earlier version. A consumer pinned to an older kit must pin `dspy<3.4.0`
  alongside it, or upgrade.
- **The interpreter is supplied through `interpreter_factory=`**, the only seam 3.4.0 left. A
  caller-supplied interpreter now goes out through new `sandbox.caller_owned`, a shutdown-suppressing
  view, so `RLMTask(interpreter=…)` keeps meaning what it says. Two consequences: a retry gets a
  FRESH sandbox on the string path, where every attempt used to re-enter the same dirty REPL
  namespace; and `dspy.settings.interpreter_factory`, new in 3.4.0, cannot substitute the runtime
  that executes model code.
- **An interpreter that describes nothing now yields NO "Execution environment:" prompt section**,
  where 1.13.0 gave it dspy's Pyodide text. A prompt change, so it is called out: only a consumer
  with an undescribed custom interpreter sees it. The default `pyodide` path is byte-identical.
- **The typed sub-LM response moved class AND layout**, `dspy.LMResponse` (a list of `.outputs`) to
  `dspy.lm15.Response` (one `.message`), so `_dspy_compat` assumes neither and probes both. `.text`
  now joins parts with `"\n"` instead of `""`.

### Fixed

- **`_InterceptedSubLM.copy()`.** 3.4.0's `dspy.LM.copy` reads `self._engine_spec`, which raised
  through the wrapper for any base that is not itself a `dspy.LM`, including `ClaudeAgentLM`. It also
  repairs something wrong on EVERY version: `copy(rollout_id=1)` updated the wrapper's decorative
  `kwargs` while `_base`, the object that makes the request, never saw it.

### Documented

- **dspy's usage merge changed and the kit did not have to.** 3.3.1 added same-named values blind;
  3.4.0 gates addition and silently keeps the FIRST non-summable value. The kit reads per-call
  entries, so nothing in `run_end.payload.usage` is lost either way.
- **`optimize.compile_task`'s documented blocker is void.** It assumed dspy would invoke the metadata
  carrier that deliberately raised. With a real factory that path works and each pass gets its own
  sandbox.

## [1.13.0] - 2026-09-24

1.12.0 records the budget a run carried in `run_end`. A run killed by a signal never writes one.

### Added

- **`applied_lm_budget` and `applied_thinking_budget` are PUBLIC**, re-exported from `rlm_harness`.
  A consumer calls them after `configure()` and records the result in its own `run_start` meta, which
  is written before the run begins and so survives a run that never reaches `run_end`. An exception
  still writes `run_end` via `TraceRecorder.__exit__`; a signal does not, and a runner enforcing a
  wall-clock timeout with `killpg` is exactly where a thinking runaway lands. A seam rather than a
  field because `run_start` is written before the task exists in that scope and the trace is
  append-only, so the recorder could write only a partial copy. The shapes were already frozen in
  trace/v1, so this adds no new promise.

### Fixed

- **An escalation the PROVIDER refused recorded nothing, so a run that escalated repeatedly read as
  one that never escalated.** `_InterceptedSubLM.__call__` wrote its `sub_call` after the base LM
  returned, so a raise skipped it. Found in production: 8 of 10 runs escalated 2-3 times each, every
  call failed with `no healthy deployments`, all 10 finished anyway, and the traces recorded zero
  escalations. The failed call now records `raw: null`, `processed: null`, the provider's error
  through `short_error`, and `cause: "endpoint"`. The exception propagates unchanged.
- **`sub_call` carries `cause`**, in the same `ok` / `invalid` / `endpoint` vocabulary a `tool_call`
  uses, because `error` alone could not separate a rejected output from a provider that never
  answered. Additive within trace/v1. A successful sub-LM call's usage does reach
  `run_end.payload.usage` keyed by the sub model, including through `llm_query_batched`'s thread
  fan-out, so an absent sub-model key means the planner did not escalate.

### Documented

- **A thinking budget set for one task binds on every task in the process**, since the env var is
  read once per process. Measured on the first production deployment of 1.12.0: 2 of 29 planner calls
  in one task and 6 of 84 in another reasoned past 16384, all finished well under the 32768 cap, and
  every cut call still produced a usable turn. Whether the RESULT is worse is a question an ok rate
  cannot answer, so the guide says to measure the output rather than the completion.

## [1.12.0] - 2026-09-24

A reasoning model whose THINKING runs away does not present as a long answer. It presents as a parse
failure, and nothing in the kit could bound it.

### Added

- **`RLMConfig.main_lm_kwargs` / `sub_lm_kwargs` (`RLM_MAIN_LM_KWARGS` / `RLM_SUB_LM_KWARGS`), a
  per-ROLE passthrough of extra `dspy.LM` kwargs**, merged over what `configure()` builds for that
  role only. Unset puts no new key on the wire. A top-level key is a litellm parameter and gets
  litellm's per-provider mapping; a key under `extra_body` goes RAW into the request body, which is
  what a server-specific name needs.

  Measured on a consumer's vLLM deployment after a model swap: 53% of attempts carried a call at the
  32768 `max_tokens` cap, and inside those the median `reasoning_tokens` was the whole budget with no
  content. One such call kills the attempt with `AdapterParseError`, which reads as a model that
  cannot follow the schema. A `thinking_token_budget` of 16384 gave 0 of 155 calls at the cap, and
  every call cut AT the budget still produced a usable turn.

  **The kit ships the mechanism and no vocabulary of its own.** A named `RLM_THINKING_BUDGET` would
  promise that one word means the same thing everywhere, and that fails inside a single model family:
  `reasoning_effort`, the one name litellm maps across providers, moved reasoning the WRONG way on
  that model (`low` 2443-2562 tokens against 2237 at the default) while breaking the output-format
  instruction in front of a JSON adapter.

  **Keys `configure()` owns are refused at parse time** (`model`, `api_key`, `base_url`,
  `custom_llm_provider`, `timeout`), never silently dropped and never silently winning. The line is
  whether the TRACE can see the override: those five are recorded nowhere. `max_tokens` IS readable
  off the LM, so `RLM_SUB_LM_KWARGS='{"max_tokens":4096}'` is how a per-role generation cap works.

- **`run_end.payload.budgets.thinking`**: `{"main"/"sub": {"value": int, "key": str}}`, where `key`
  is the dotted path the value was found at. Additive within trace/v1. Its own key rather than a
  field on `budgets.main`, whose PRESENCE means "this role carried a token cap".

  **The name table is on the READ path only**, so an unrecognised name still reaches the server and
  is merely not annotated: **absent means NOT RECOGNISED, never "no budget"**. This is also the
  answer to the feature's blind spot, that a passthrough cannot report a key the server ignored;
  pairing `budgets.thinking` with `usage`'s `reasoning_tokens` makes that checkable per run.

- **A warning when a thinking budget is `>=` the role's generation cap**, and another when a
  passthrough lands on a role `configure()` did not build. Both warn and never fail.

### Documented

- **The guide's "Per-role LM request parameters" section carries a measured quoting matrix.** The
  four layers disagree: a shell-sourced file needs the single quotes, `docker run --env-file` never
  strips them, while compose's `env_file` and python-dotenv strip them and accept every form. No
  single spelling survives all four; single-quoted is the right default.
- **Every em-dash is gone from the repo's prose**, docs, docstrings, comments, workflows and hook
  scripts alike. The 30 that sat inside runtime strings changed wording only.

### CI

- **`.githooks/` blocks a commit carrying a private project name, with the denylist kept OUTSIDE the
  repo.** The vendor-neutrality rule held for every file and every published release note and failed
  twice in commit MESSAGES, 128 and 176 commits deep. Enable in a clone with `git config
  core.hooksPath .githooks`; see `docs/VERIFY.md` for why a missing list skips rather than blocks.
- **`release.yml` records three things that only bite at publish time.** A `release: published` event
  runs the workflow at the TAG's commit, so fixing it on `main` does nothing until the tag is
  re-pointed. `requires = ["hatchling>=1.27"]` is a lower bound, so what gets uploaded can change
  with nothing here changing: a consumer's release failed on `InvalidDistribution: '2.5' is not a
  valid metadata version` when hatchling moved to core metadata 2.5 and the twine bundled in
  `gh-action-pypi-publish` below v1.14.2 rejected it. And `pypi.org/pypi/<name>/json` is CDN-cached:
  it reported a stale "latest" while publishing 1.11.0, where `/simple/` and
  `/pypi/<name>/<version>/json` were both correct.

## [1.11.2] - 2026-09-11

A trace is not written in the order it happened, and the RL exporter sorted by the order it was
written. Every `state` it produced was wrong, systematically, in opposite directions per kind.

### Fixed

- **`export_actions` sequenced by `step_id`, which is WRITE order, so every `state` was wrong.**
  `record_main_trajectory` flushes the whole trajectory once `aforward()` has returned, so a turn's
  `step_id` is higher than every live `tool_call`/`sub_call` of the same attempt. Sorting the three
  action types together by it put EVERY turn after EVERY tool call, and `state`, documented as "the
  ordered list of prior actions", came out wrong in one direction per kind: **a tool record's prior
  actions contained no turns at all, and a turn's contained every tool call.** Every record of every
  run that used tools, in every corpus, since the exporter shipped, because no test ever exported a
  MIXED-family run.

  Records now interleave on `ts`, whose documented purpose is placing a turn against the live events
  around it, and a turn's stamp is taken when its reasoning was PARSED, so it correctly precedes the
  tool calls that turn's code then makes. Turns keep `payload["turn"]` order among themselves, since
  an unmatched turn falls back to the flush time; a corpus with no timestamps falls back to
  `step_id`. The record SHAPE is unchanged.

  **A dataset previously exported from a tool-using run is worth re-exporting, with one limit**: it
  recovers nothing for a run whose turn stamps were never matched, because a flush time is later than
  every live event and the merge reproduces the old order exactly. `_sequenced_actions` emits a
  `logger.debug` on the detectable symptom, a run whose first turn is not stamped before its first
  live event. That is a hint, not a verdict: an earlier retry attempt reads the same way with nothing
  wrong.
- **`replay.reconstruct`'s `step_id` sort is the same write order**, and is left alone deliberately:
  every `Timeline` accessor filters to one event type, where write order IS causal order. Its comment
  now says so, and warns that iterating `Timeline.events` across types does not.
- **The guide's ordering rules left ADJACENCY to be inferred, and the inference a reader reaches for
  is out by four orders of magnitude.** A consumer building "which tool calls shared a turn" from
  runs of consecutive `step_id` measured a burst-parallelism ceiling of 90.5% where the answer was
  0.0037%, the alarming number being the wrong one: the `main_step` batch lands last, so a whole
  run's live events carry an unbroken `step_id` range and the grouping collapses the run into one
  burst. The section now separates ORDER from ADJACENCY: `step_id` is a fine order key for live
  events and sturdier than `ts`, while `step_id` ADJACENCY means nothing and turn grouping must come
  from timestamp gaps.

### CI

- **`install-check.yml`: the first job that installs the PUBLISHED artifact, and the first that runs
  on macOS.** Neither "what PyPI actually serves" nor "green on Linux, red on macOS" had an
  instrument before it. Prompted by a downstream consumer who hit a macOS-only install failure their
  all-Linux CI structurally could not see. Measured before adding: a fresh macOS install of the
  published 1.11.1 resolved dspy 3.3.1 + litellm 1.99.0 and passed, so nothing was broken; it is a
  tripwire rather than a fix. It chains off `Release` COMPLETING rather than `release: [published]`,
  demands the exact version just tagged, and is INFORMATIONAL, so a red is fix-forward. See
  `docs/VERIFY.md`.

## [1.11.1] - 2026-09-05

Documentation only: no public name, no behaviour, no schema change. It ships for the reason 1.10.1
did: the docstrings are IN the wheel, and four of them told a builder that `budgets` records the cap
a call APPLIED. It records the cap the LM CARRIES, and this kit ships one LM for which those differ.

### Fixed

- **Four shipped docstrings and the guide claimed `budgets` is the cap "APPLIED".** It is read off
  `lm.kwargs`, as dspy's own `_check_truncation` does, but that rests on an LM's kwargs being what it
  applied, and `ClaudeAgentLM` is the shipped counterexample: the subscription SDK exposes no output
  cap, so it tolerates and IGNORES a `max_tokens=`, which is then staged into `budgets` as a cap
  nothing enforced and which `completion_tokens` can exceed. `_dspy_compat.applied_lm_budget`,
  `task.py:_applied_budgets`, `trace.py:note_budgets` and the guide now all say CARRIES. (1.10.0's
  CHANGELOG entry keeps its original wording, as the record of what was believed at release.)
- **"Nothing can detect this from the trace side" was false.** A reader has two handles.
  `completion_tokens` above the largest per-role cap FALSIFIES the equality and never confirms it,
  which is the asymmetry the flat claim lost, while comparing against one role's cap proves nothing,
  since `budgets` records no model name and `usage` is keyed by model. The second needs no
  attribution: this LM's model key carries the exported `SUBSCRIPTION_PREFIX`. The advice attached to
  the false claim was inverted too: it told a reader to distrust a cap when they did not choose the
  LM, when the auto-routed path passes no kwargs at all and the only way to reach the trap is to
  construct the LM yourself.
- **`configure`'s docstring omitted `max_tokens` from the kwargs an auto-routed subscription role
  drops**, so `RLM_MAX_TOKENS` is silently inert for that role. The upside: no cap is recorded
  either, where a hand-built `ClaudeAgentLM(max_tokens=...)` stages one the call never applies.

## [1.11.0] - 2026-09-05

An LM call's token totals give the context size only when the call made a single API request with a
single sampling iteration, and they never say whether it did. Additive within trace/v1; no public
name moves.

### Added

- **`api_rounds` on a `run_end.payload.usage` call entry**: the provider's own per-iteration
  breakdown, verbatim, when it reports one. Its purpose is the CONTEXT WINDOW SIZE, which
  Anthropic's type documentation says to calculate from the last `message` entry. The field makes
  that answer UNCONDITIONAL rather than possible: the totals coincide with it on an ordinary
  single-request call, and the reading it displaces ("the totals are useless for context") throws
  away an exact number wherever that condition held. Motivating observation: a live run whose last
  of six planner calls recorded 91,514 prompt tokens against 10,591 for the one before, a shape the
  totals alone cannot explain because they cannot separate a grown context from a repeated request.

  **It is not a decomposition of the entry's `prompt_tokens`**, which is the reading it invites. A
  call's token fields accumulate across every API request it made, while `api_rounds` carries only
  the last request's iterations, and a `compaction` entry's tokens are excluded from the top-level
  fields entirely. So summing the rounds does not reconstruct the call, and a context reading takes
  the last `message`/`fallback_message` entry, never a `compaction` one, whose counts report what
  the summarisation cost.

  Two properties, each chosen against a specific failure. **Nested as `{"rounds": [...]}`**, because
  dspy's `UsageTracker` merges usage entries by ADDING same-named values, where a bare list
  concatenates and the mixed case raises `TypeError: int + list` out of `dspy.Module.__call__`
  itself, after a completed run. **Present only for a NON-EMPTY list of objects**, because `all(...)`
  over an empty list is `True` and `[]` is what the CLI's accumulator seeds itself with, so absent,
  null, empty and malformed collapse to one outcome with one meaning: no key. A rejected-but-present
  breakdown logs at DEBUG, so "no run has `api_rounds`" can be told apart from an upstream rename.

  Entries stay in the provider's own vocabulary. Normalising them would collapse the cache split,
  which is what made 1.10.2's bug invisible, and summing every integer in a round double-counts the
  cached half.

### Changed

- **`task.py` records WHY installing the usage tracker before `aforward` is load-bearing.**
  `dspy.Module.__call__`/`acall` auto-total a run when `settings.track_usage` is set and no tracker
  exists yet, so entering the kit's scope first means dspy never takes that branch, and `RLMTask`
  calls `aforward` directly rather than `acall`. Both were true by accident of ordering and are now
  true on purpose: a refactor to `rlm.acall(...)` would arm a crash that discards a finished run.

## [1.10.2] - 2026-09-05

`ClaudeAgentLM` recorded 0.1% of the prompt it actually sent, and manufactured a zero-token call
whenever the SDK reported no usage at all. Both are in the OPTIONAL subscription adapter
(`rlm-harness[subscription]`): no public name, no schema change, nothing else moves.

### Fixed

- **The prompt size read one of the three fields Anthropic splits it across.** The adapter mapped
  `result.usage["input_tokens"]` onto litellm's `prompt_tokens`, but that field is only the part of
  the prompt neither written to nor read from the cache, and the Agent SDK caches the system prompt
  and tool definitions by default. Measured on a live subscription call with a ~2k-token prompt:
  `input_tokens=2`, `cache_creation=2047` on first sight and `cache_read=2047` on the repeat, so the
  adapter recorded **2** where the prompt was **2049**. All three fields are now summed; a provider
  reporting no cache fields is unaffected. The sum is a SIZE and not a cost basis, since the three
  bill at different rates. It does not touch 1.10.0's truncation ratio, which never divides by this
  number.
- **An unreported usage was materialised as three zeroes.** The adapter always passed a `usage=`
  kwarg and litellm turns an empty one into `Usage(0, 0, 0)`; omitting it leaves the attribute off
  and both of dspy's reads then report absence rather than a count. So "the SDK reported nothing"
  reached the trace as "this call used zero tokens", a structural zero indistinguishable from a
  measurement, which `docs/INVARIANTS.md` names as a defect and the guide publishes a promise
  against.
- **Neither token field was read with an int guard**, so a `None` from the SDK reached
  `prompt_tokens + completion_tokens` and raised `TypeError`, failing the whole LM call over a
  missing count. Every field of `result.usage` now goes through one guard, which also excludes
  `bool`.

## [1.10.1] - 2026-09-05

Documentation only: no public name, no behaviour, no schema change. It ships because the docstrings
are IN the wheel: `pip install rlm-harness==1.10.0` gives a builder a `line_numbers=` warning that
argues the wrong way at the moment the flag is flipped.

### Fixed

- **`make_read_file_tool`'s `line_numbers=` warning cited the argument FOR the flag as the cost of
  turning it ON.** Its 17.3% figure was measured on corpora built with line numbers OFF, where no
  gutter existed: that is the SECOND failure, the one line numbers FIX. It now carries the first
  failure's own number (83.16% of guttered non-blank quotes resolving since 1.9.0, 67.2% of
  everything the tool renders once blank-line citations are counted) and points at the guide, where
  both failures can be weighed against each other.
- **The gutter warning was reachable only from where the consequence is explained, not from where
  the flag is flipped.** `make_read_file_tool`'s docstring, the guide's parameter list and
  `make_edit_file_tool`'s `show_snippet` each described their feature and stopped, while the
  interaction with `verify_quote` lived in the grounding section. `show_snippet` is the worst of the
  three because it defaults to **True**: a task using `edit_file` feeds the model guttered text
  whether or not it ever considered line numbers. All three now point at the explanation.
- **`_dspy_compat.py` reported a distribution summing to 384 over a stated 385.** Two populations
  mixed: 363 is successes-only, 21 spans successes and failures. The bins are the 379 successes
  (363 / 0 / 16), and adding the 6 failures gives 364 / 0 / 21 across all 385.

### Changed

- **`usage`'s token ratio is diagnostic after the fact, and whether it gives EARLY warning depends
  on the caller's cap.** 1.10.0's entry below says the ratio "shows a turn APPROACHING the cap" with
  no qualification, and that is left standing as the record of what was believed at release. The
  first production corpus qualified it: 385 runs on one model at a 32768 cap, and the distribution
  has a HOLE, 363 successes below 0.6, ZERO between 0.6 and 1.0, 16 at the cap. There a turn stays
  under ~0.55 or blows straight through and a proximity meter has nothing to point at. But the hole
  is an artifact of a cap set at roughly twice what that model needed: transposing the same bins onto
  a 16384 cap moves 64 of the 379 successes (16.9%) into the empty band, so a tighter cap gives a
  real gradient. Measure your own before building either reading.
- **What the field unambiguously bought, from the same corpus**: 21 of 385 runs (5.5%) hit the cap
  and **16 of those 21 (76%) finished anyway**, because a truncated CODE cell is a `SyntaxError` that
  dspy's own in-loop feedback repairs while a truncated FINAL answer has no handler and kills the
  run. Three quarters of truncation was invisible rather than absent. That also sizes the upstream
  request 1.10.0 refused: extending dspy's feedback path to a parse failure addresses 5 of 21.

## [1.10.0] - 2026-09-04

`run_end` records the token budget that was in force and what each attempt actually spent, so a
TRUNCATED completion stops being indistinguishable from a MALFORMED one.

### Added

- **`run_end.payload.budgets`: the generation cap that was APPLIED, per role.** A consumer's run died
  with `AdapterParseError: LM response cannot be serialized to a JSON object` and read as a small
  model failing to follow the format. It was a `max_tokens` truncation. dspy detects that,
  `_check_truncation` tests `finish_reason == "length"`, and then only `logger.warning`s it,
  discarding the datum before any caller can see it. The exception TYPE is identical either way, so
  this is the misdiagnosis 1.4.0 already documented under `max_tokens`, recurring because nothing
  recorded the one fact that separates them.

  Recorded alongside the token cap: the ITERATION caps (`max_iterations`, `max_llm_calls`,
  `max_output_chars`) and a `dropped` flag saying whether `_build_rlm`'s `except TypeError` fired,
  since that path reverts all three to dspy's defaults and the configured numbers would otherwise
  read as applied. `max_output_chars` matters in its own right: dspy head+tail-caps each REPL output,
  so "the output was cut off" has THREE independent mechanisms.

  Read off the LM, never from `RLMConfig`, because an injected `main_lm`/`sub_lm` is used verbatim so
  the configured cap can be one the call never used. The recorded `key` says which name held it,
  because dspy rewrites `max_tokens` to `max_completion_tokens` for OpenAI reasoning models and a
  reader of the first name alone gets `None` for exactly the thinking-model case. **Named keys
  only**: a trace is a shipped artifact and `lm.kwargs` carries `api_key`.

- **`run_end.payload.usage`: token counts, per ATTEMPT.** `completion_tokens == cap` is a truncation,
  and unlike a boolean the ratio also shows a turn APPROACHING the cap, which is the early warning
  nobody has ever been able to see. Collected through dspy's public usage API, whose tracker the kit
  HOLDS, so the counts survive the exception and the fatal call's tokens are recorded for a run that
  raised, which is the whole point.

  Per attempt, with `turns_recorded` marking the attempt whose turns are in the trace, because
  `run_with_retry` re-runs the whole trajectory and a run whose FINAL attempt raises keeps an EARLIER
  attempt's turns. **Not per TURN, and that is a limit rather than an omission**: `sub_model` falls
  back to `main_model` and dspy propagates the tracker into its sub-LM workers, so planner turns,
  escalations and same-model tool-LM calls land in one flat list under one key with no call id.
  For a distribution over runs use `max(completion_tokens)`, never the SUM, which answers a cost
  question instead.

- **A caller's own `dspy.track_usage()` is REUSED, not shadowed.** dspy installs a tracker only when
  none is set, so installing unconditionally would hand a consumer measuring cost around
  `task.arun(...)` ZERO entries for everything inside. One disclosed cost when the kit installs one
  because you had none: dspy attaches per-prediction usage only when no tracker is installed, so a
  `dspy.Module` a consumer calls from inside a kit run gets `None` from `get_lm_usage()`. The counts
  are still in the tracker and in the trace.

### Not done, deliberately

- **An in-loop recovery for a truncated or unparseable turn** was requested and is refused. dspy does
  expose the seam, `Adapter.__call__`/`acall`, which this kit already subclasses and deliberately
  strips of `JSONAdapter`'s re-call. But a recovery there is invisible to the trajectory: the RLM
  loop never sees it, so the corrective exchange is not a `main_step`, a second unrecorded turn added
  to a failure whose defining problem is an unrecorded deciding turn. The version worth having needs
  the loop, so it belongs upstream.
- **`RUN_FACT_KEYS` is unchanged.** The trace payload gains the fields; `compute_run_facts` does not.
  With per-turn attribution impossible, a `truncated_turns` count would be per-run and weaker than
  its name implies.

## [1.9.1] - 2026-09-02

`tool_total_seconds` measures the wall-clock tool calls OCCUPIED, instead of adding their durations
together and counting a nested call twice.

### Fixed

- **`compute_run_facts`'s `tool_total_seconds` / `tool_wasted_seconds` are the measure of the UNION
  of the tool calls' intervals, not a cross-tool sum.** A tool that records another `tool_call` from
  inside itself produces two CORRECT events describing one stretch of wall clock; adding them reports
  time that was never spent. On a real trace the sum reached **136.5% of the run's own span**,
  impossible for a wall-clock share and produced by this kit's own code. Five traces from one
  consumer moved 42.8 to 23.3%, 83.1 to 46.0% and 136.5 to 69.4%.

  It takes BOTH halves, which is why it went unseen: the key has been a sum since 1.8.0, but the
  double-count needs 1.8.3's auto-timing of the OUTER tool AND an inner call the tool records itself,
  so it could only ship in 1.8.3, 1.8.4 and 1.9.0. A flat tool graph never saw it, and the number
  stays under 100% until the nested call dominates the run.

  Each event contributes `[ts - duration_s, ts]`, which is sound across the two clocks involved
  (`ts` is `time.time`, `duration_s` a `time.perf_counter` delta) because a delta is clock-agnostic
  and fixes the interval's LENGTH, but nothing fixes its POSITION. The clocks differ in RATE, so a
  reconstructed start is displaced by roughly `duration_s x drift`: measured at >= 26.3 and >= 29.9
  ppm, i.e. 7.8 ms and 17.3 ms on calls of 298 s and 577 s. So the union can invent a small overlap
  between strictly sequential calls, in the fifth decimal place against a nesting effect measured in
  hundreds of seconds. `record_tool_call`'s docstring now states the precondition: the envelope `ts`
  must be the END of the measured window.
- **`compute_run_utilization` no longer raises on a `tool_call` whose `payload` is `None`.**
  `dict.get`'s default fires only on a MISSING key, never on a key present with a `None` value, so
  `event.get("payload", {}).get("tool", ...)` raised `AttributeError`. `compute_tool_waste` already
  handled it, and such an event now counts as tool `"?"` with cause *invalid*.

### Changed

- **`tool_total_seconds` no longer equals `sum(w.total_seconds for w in compute_tool_waste(...))`.**
  It is strictly SMALLER once any call nests, and otherwise equal only up to reconstruction error,
  where it can be a few 1e-7 s LARGER because `ts - (ts - d)` is quantised at epoch scale. Do not
  write `assert new <= old`: on a flat run with 24 calls it already fails, by +1.6e-7 s.
  `ToolWaste`'s own per-tool numbers stay SUMS deliberately; it is the cross-tool aggregate that must
  not double-count.
- **A consumer tracking `tool_total_seconds` across the upgrade sees a step**, downward on any run
  with a real nest. `tool_wasted_seconds` changes with it and its derived SHARE can move UP: an `ok`
  call nested inside an `invalid` one reads 0.556 before and 1.000 after. Latent for anyone whose
  failures wrap other calls.
- **The changed value is also written INTO traces** by `TraceRecorder(record_metrics=True)`. Nothing
  is added, removed or re-typed, so trace/v1's additive-only rule is intact, but a corpus spanning
  the upgrade holds two meanings of `payload.metrics.tool_total_seconds` in-file, distinguishable
  only by run date.
- **`compute_run_facts` on a MULTI-RUN list changes its conflation direction**: it already warns that
  a multi-run list conflates, and where that over-added, the union can now silently UNDER-add by
  merging two concurrently-executed runs' intervals. Use `compute_run_facts_by_run`.

### Removed

- `_sum_or_none` (private, unused once both keys move). Its `None`-vs-`0.0` discipline moves into the
  new helper: `None` when nothing carried a duration, decided over ALL tool calls and never over the
  wasted subset, since gating on the subset reports `None` where a healthy run should report `0.0`.

## [1.9.0] - 2026-09-02

`verify_quote` now reads a line-numbered citation as a coordinate claim and verifies it, instead of
searching the gutter's digits as if they were content.

### Added

- **A guttered quote is resolved by coordinate.** 1.8.2 closed the case where a citation of NOTHING
  verified; this closes the one it documented as open, a guttered quote carrying CONTENT searched with
  the gutter DIGITS as literal text, so it matched wherever that number happened to precede the
  line's text, across a mandatory whitespace junction and therefore across blank lines. Rendering
  every non-blank line of every `.py` in this repo and verifying it against its own file found **2
  such matches in 18,761**, both HONEST citations whose content really is at the gutter's line. Both
  now resolve correctly.

  A guttered quote verifies when four things hold: its gutters are consecutive, the line numbers are
  in range, the content sits at exactly the line the gutter names, and that block occurs exactly once
  in the source as a contiguous line sequence. Otherwise the ordinary search runs.

  **The gutter is used, never stripped.** Stripping and searching the remainder, the repair this
  function refused for four releases, accepts a citation naming the WRONG line whenever the remainder
  appears elsewhere. Both halves are load-bearing: without uniqueness a bare position check is WORSE
  than searching, because 16.84% of non-blank lines here recur in their own file, so a fabricated
  coordinate verifies at roughly 0.15% against 0.000% for a plain search; without exactness a
  fabricated INDENTATION level verifies, since 2.16% of lines here are exact-unique but identical
  after stripping, and in Python indentation is semantics. With both, fabrication is closed by
  construction. Uniqueness is a contiguous LINE SEQUENCE and not a substring count: one residual's
  content is `)`, with hundreds of substring hits and one whole-line hit.

### Changed

- **A coordinate-verified `MATCH` means something different, and says so.** The source holds the
  CONTENT at that line, but the quoted bytes, gutter included, are not a substring of it. A caller
  re-deriving grounding host-side with `quote in source` must branch on the text; the `MATCH:` prefix
  is unchanged for callers that branch on that.
- **`normalize_whitespace=False` skips the coordinate path.** Byte-exact mode means no
  interpretation, and reading a gutter as a coordinate is one. It is also the lever when `source` is
  itself a numbered listing, a shape found in none of the tens of thousands of local text files
  scanned though `cat -n` and `nl` emit it.
- **Two pinned test verdicts move, and that is the feature.** A quote of `"     1\tx = 42"` against a
  source whose line 1 is `x = 42` was a MISMATCH; it is now a coordinate-verified MATCH.

### Not a fix: a constraint on the new code

- **The gutter is bounded to nine digits, and that is why nothing raises.** `verify_quote` documents
  that it never raises, and on CPython 3.11 `int("9" * 4301)` raises `ValueError: Exceeds the limit
  (4300) for integer string conversion`. No released version ever converted a digit run, so this
  fixes no regression; an unbounded `[0-9]+` in the NEW recognizer would have introduced one.
- **The closest-line hint on a failed coordinate check was considered and left alone.** A guttered
  quote that falls through is diffed with its gutter attached, which depresses the similarity ratio
  for exactly this class. Hinting with the parsed content is a change to the MISMATCH path with its
  own verdict surface.

### What this does NOT close

16.84% of non-blank lines recur in their own file, and a citation of one of those stays uncoordinated:
32.8% of everything `read_file` renders, once blank-line citations refused by the 1.8.2 guard are
counted. Those keep today's verdict, and that class contains no wrong-line matches to inherit: every
one of the 18,759 non-residual honest quotes was already a MISMATCH.

## [1.8.4] - 2026-09-02

A failed run's trace now says why it failed, and, separately, stops losing the event that says so.

### Fixed

- **A lone surrogate anywhere in a payload lost the event.** `record()` json-dumps with
  `ensure_ascii=False` into a handle that had STRICT error handling, so a single unencodable character
  raised `UnicodeEncodeError` out of `record()` and nothing was written. Reproducible on shipped code
  with nothing exotic:

      rec.record("main_step", {"code": "x = '/data/caf\udce9'"})   -> UnicodeEncodeError

  `os.fsdecode`, a `surrogateescape` decode, and a model completion embedded in a dspy adapter error
  all produce them. The handle now opens with `errors="backslashreplace"`, which round-trips
  byte-exactly through `load_events` and leaves ordinary text alone, since the handler fires only on
  characters that cannot be encoded at all. `run_end`'s `error` survived this only because `repr()`
  happens to escape surrogates.

### Added

- **`run_end.payload["error_chain"]`: the causes below the outer exception.** `run_with_retry` raises
  `RLMTaskError(...)` **from** the last real failure, so the cause is chained on the object and the
  `repr()` that was all `run_end` recorded drops it, leaving a failed run's trace saying only that it
  had failed. **Across the nine-consumer fleet that was 15 of 15 recorded failures with no
  recoverable cause.**

  Each frame is `short_error(e)`: `Type: message`, head and tail kept, bounded near 600 characters.
  The outer frame is not repeated, since `error` already holds it. Written only when a chain exists,
  so a reader can tell "no cause" from "could not build one". Never a traceback, which would name
  every frame's file and line. Capped at five frames, truncating the OUTER end so the root cause
  survives a deep chain. The walk follows `__cause__`, else `__context__` unless
  `__suppress_context__`; the kit has zero of the latter (AST-verified) and dspy, litellm and httpx
  all raise with `from`.

  **Unconditional, not opt-in, and the `RLM_TRACE_METRICS` precedent does not apply.** That one
  defaults off because its facts are DERIVABLE from the events. A cause chain exists only on the live
  exception and is gone the moment `__exit__` returns. Gating it would also repeat what 1.7.0 and
  1.8.3 each shipped a fix for: a field that depends on someone opting in is missing for someone.

  **Two cautions before forwarding a frame.** Unlike `error`'s `repr`, a frame is not guaranteed
  single-line: pydantic and dspy adapter errors are multi-line. And an exception message can carry a
  URL with a query-string token; the kit has no scrubber and `short_error`'s cap is the whole
  mitigation. The exposure already existed for `error`; the chain widens it to third-party frames,
  which is where such a token most often lives.

## [1.8.3] - 2026-09-01

Every tool a task hands the model now records how long it took, without its author doing anything.
Two documented rules are reversed to make that true, and both reversals have the same cause.

### Fixed

- **`duration_s` existed only where its author remembered, which is the `sub_call` failure one field
  over.** Six of the kit's tool sources measured themselves; the filesystem and knowledge tools, 27
  `record_tool_call` sites across `fs.py`, `edit.py`, `archive.py` and `skills.py`, did not, so
  `compute_tool_waste`'s `*_seconds` read `None` for them everywhere. A consumer could not fix it
  either: wrapping the callable to add a duration would emit a SECOND `tool_call` and double
  everything derived from it, so the only place to fix it was here.

  `RLMTask._build_rlm` now wraps every tool it hands the model, the same seam and reason as 1.7.0's
  automatic `sub_call` wrapper. The wrapper publishes a start time and **records nothing itself**, so
  it cannot double-count; `record_tool_call` fills `duration_s` from it only when the caller passed
  none, so a tool that measures itself keeps its own tighter figure.

  **What it deliberately does not reach**, each failing back to an absent field rather than a wrong
  one: a `dspy.Tool` OBJECT (a pydantic model with no `__name__`, so wrapping would register it as
  `timed` and two would abort the task with "Duplicate tool name"; MCP records its own duration, but
  a `dspy.Tool` a CONSUMER builds does not and must pass `duration_s` itself); a coroutine function,
  because dspy branches on `inspect.iscoroutinefunction`, which does not follow `__wrapped__`; a
  callable class instance and a `functools.partial`, neither of which `functools.wraps` can wrap
  without changing what dspy registers; and a GENERATOR FUNCTION, wrapped like any other yet
  recording nothing, since calling it only builds the generator and the wrapper releases the start
  time before the body ever runs.

  **The fill is matched on the tool's name, and that is load-bearing.** Only what the task hands the
  model is wrapped, so a COMPOSITE tool would otherwise charge its whole window to every event
  recorded beneath it: measured before the check existed, two zero-cost `read_file` calls inside a
  0.25 s tool each reported 0.25 s, tripling `total_seconds`. That is worse than the `None` it
  replaces, so a mismatch fills nothing.

  Applied at the task seam, which is safe for the annotations DESPITE the distance rather than because
  of it: every `tools/*.py` uses `from __future__ import annotations`, so the annotations
  `functools.wraps` copies are strings that resolve only in the defining module, and they survive
  because `typing.get_type_hints` walks `__wrapped__`. A CPython behaviour, which is why it is tested
  rather than assumed.

### Changed

- **A refused call now carries a duration.** It used to record none, on the argument that a blocked
  URL never touched the network so a ~0 would be noise. `None` means "nobody measured", so spending
  it on "measured, and it was instant" makes the two indistinguishable: the mistake 1.7.0 shipped a
  release to correct.
- **`grep_files` is no longer exempt from timing, reopened by its own terms.** Its exemption rested on
  a measurement (n=146, median 0.029s, max 0.746s) and named "a pathological regex over a large tree"
  as what would reopen it. Re-measured on a consumer deployment across nine real repositories, 7
  patterns x 3 runs each: **median 744 ms, p95 4.8 s, max 6.3 s on a 2,110-file repository**, about
  40% of all sandbox execution time against that corpus. It does not scale with file count, so the
  driver is bytes and match count. The old number was not wrong; it was taken on a corpus with no
  large repository in it.
- **Do not average a `compute_tool_waste` figure across this upgrade.** A tool that reported `None`
  before reports a real number after, so a corpus spanning the boundary mixes "unmeasured" with
  "measured" in one denominator. `run_start.rlm_harness` is what separates the cohorts.

## [1.8.2] - 2026-09-01

One correctness fix in shipped code. It is model-visible: dspy builds a tool's description from
`func.__doc__`, and `verify_quote` is registered directly as a REPL tool, so its refusal text and its
docstring reach every consumer's model on every call.

### Fixed

- **A quote carrying only a line number verified against almost anything.**
  `make_read_file_tool(line_numbers=True)` renders a line as `f"{n:>6}\t{line}"`, so a BLANK line
  renders as `"     7\t"`, whose `.strip()` is `"7"`: non-empty, so it passed the empty-quote guard,
  and the search reduced to the bare pattern `7`.

      verify_quote("x = 42\n\ny = 1\n", "     2\t")
      -> MATCH: found at line 1 (char 5)

  A citation of nothing verified, at a line the citation never claimed, and the guard's own stated
  reason for existing describes that case exactly. It now covers a second shape: a quote whose every
  non-blank line is a bare number, refused before any search with its own message. The rule reads the
  whole line loosely on purpose, because the render is not what arrives: a model trims the trailing
  tab, writes a space for it, or keeps the newline. ASCII digits only, since `\d` accepts fullwidth
  and Arabic-Indic numerals, which are content here rather than coordinates. It encodes no
  line-number format, so `grounding.py` stays independent of the tools that render one.

  **This closes the fully-blank case, not the whole class.** A guttered quote carrying content is not
  refused, and the gutter NUMBER is then searched as literal content, so it matches wherever that
  number precedes the line's text, including across a mandatory `\s+` that spans blank lines. From
  this repo's own suite, a quote claiming line 42 of `tests/test_async.py` verifies at line 39,
  because line 39 ends `== 42`. Measured by rendering every non-blank line of every `.py` here: 2
  false matches in 17,412 quotes, about 1 in 8,700. Closing them needs the gutter-stripping repair,
  deferred because a position-checked design was built, audited, and found to verify FABRICATED
  citations on any file containing a form feed.

  **What this costs.** An all-digit quote no longer verifies even when the digits are genuinely in the
  source, so `verify_quote("port = 8080", "8080")` is refused, as is a column lifted from a numeric
  file. For a short number that is the point; for an 18-digit identifier it is a real loss. The rule
  is blunt on purpose: it cannot tell a coordinate from a datum, and the failure it prevents is worse
  than the one it causes. It was 0 of 1,363 citations in a consumer corpus.

### Docs

- **Line numbers and `verify_quote` are complementary, not alternatives**: a consumer read them as a
  choice, turned numbers off to keep verification passing, and paid for it with 59 of 411 stored
  citations quoting the text verbatim at the wrong line. Turning them back on and re-running the same
  task moved coordinate corrections from 21.4% to 0.0% (15 of 70, then 0 of 39; P = 8.2e-05 against
  the prior rate). Read that as a proportion and not a paired experiment: the second run planned a
  different outline, so it is 11 artifacts against 6 and the citation counts differ. The rule is which
  string goes where: the rendered text to the model, the raw file to the verifier.
- **The README's Status section no longer restates the current release.** It had fallen five versions
  behind while claiming to describe the current one, and that file is the PyPI long description, so
  the stale text was the first thing a reader saw.

## [1.8.1] - 2026-08-30

Documentation and test only. No behaviour change, no API change: every 1.8.0 call path is
byte-identical.

### Docs

- **`budget_exhausted` now documents what actually establishes it, and what it cannot answer.** 1.8.0
  shipped the field with its evidence resting entirely on `ScriptedInterpreter`, which says nothing
  about whether the real sandbox path reaches that branch the same way. A consumer ran all three
  states against `dspy.PythonInterpreter` on deno 2.8.2, scripting only the LM, and the field
  discriminated. The docstring now records that, the two traps that fake a negative result (the
  forced-final path makes a SECOND LM call for the task's output field, so a scripted LM one turn
  short dies in `extract` and the marker looks lost; and a `True` without a submitting control run is
  not a measurement), and the boundary that is a product fact rather than a defect: the trajectory is
  written after `aforward()` returns, so a SIGKILLed job, the case an operator actually asks about, is
  exactly the one this reports `None` for.

  **These notes were one commit past the `v1.8.0` tag**, so anybody who installed 1.8.0 and ran
  `help()` saw none of it. Caught by a consumer that checked the installed package before writing "the
  kit documents this" into its own README.

### Tests

- **The forced-final test had no control run.** It asserted only that a run which never submits carries
  the marker, which a dspy writing that string on every run would also satisfy. It now drives a
  submitting run and requires the marker to be absent there. Verified additive: deleting the control
  leaves the suite green.

## [1.8.0] - 2026-08-30

The kit now computes the generic half of a rubric's facts, so a consumer supplies only its domain
half. Three new public names, one optional payload field, nothing removed or re-typed.

### Added

- **`compute_run_facts(events)` / `compute_run_facts_by_run` / `RUN_FACT_KEYS`.** `rubric.py` always
  gave you the SHAPE of a rubric, while every consumer hand-derived the facts to feed it: two
  consecutive stages of one pipeline with the middle one missing. `RUN_FACT_KEYS` is a closed, public
  tuple the dict is BUILT against, so a new key cannot appear without a diff to a SemVer-governed
  name. That mechanism, rather than a promise in a docstring, is what keeps a reward-shaped scalar out
  of the source of truth.
- **`budget_exhausted`**: did the run stop because its ITERATION budget ran out? Read from the marker
  dspy writes on its own fall-through branch, which the kit has always recorded, so it needs no
  configured cap staged into the trace, works on every trace ever written, and avoids the
  `main_steps >= cap` false positive on a run that submits on its last allowed turn. Tri-state:
  `None`, never `False`, when there is no `final` event or its reasoning is absent.
- **`fence_refused_turns`**: turns dspy refused to execute over a markdown fence tag. **Named for the
  mechanism, because the obvious cause is wrong.** Running dspy's own stripper over three real corpora
  found 60 refusals and **zero** that start with a fence; 55 of the 60 are valid Python assigning a
  documentation page whose text contains a fenced example, because dspy's stripper scans the whole
  cell including string literals. A consumer read the same number as format non-compliance and spent
  two prompt generations suppressing the code blocks its own pages needed.
- **Optional snapshot into the trace, OFF by default.** `TraceRecorder(record_metrics=True)` or
  `RLM_TRACE_METRICS=1` folds the facts into `run_end.payload["metrics"]`, additive within `trace/v1`.
  Computed by re-reading the file just written and filtered by `run_id`, so the snapshot is
  consistent-by-construction with the bytes beside it. **Emitted only when that re-read finds this
  run's own `run_start`**, since `load_events` returns `[]` for a rotated file or a mismatched id and
  an all-zero dict would be indistinguishable from a measured zero.

### Internal

- `_dspy_compat` gains `python_fence_langs`, `forced_final_marker` and `dspy_refuses_fence`. The three
  are not symmetric and the tests say so: the fence-tag SET is a dspy module constant, so its test
  asserts the introspection path resolves rather than the value; the forced-final marker is a bare
  literal with no constant behind it, so its test drives a real forced-final run; and
  `dspy_refuses_fence` is a declared verbatim mirror of a `_`-private dspy parser, cross-checked
  against that function. A regex shortcut was measured at 1,764 disagreements and 3,855 crashes over
  20,016 cells, dying on a bare fence, the commonest shape, so the mirror is not optional.

## [1.7.0] - 2026-08-29

A sub-LM escalation now records itself whether or not the consumer asked, and the sub-LM wrapper hands
dspy back the response SHAPE dspy handed it. No new `__all__` entry; `trace/v1` gains no event type,
envelope key, or payload field.

### Fixed

- **`intercept_sub_lm` broke a sub-LM that returns dspy's typed `LMResponse`.** The wrapper collapsed
  anything non-list into `[outputs]`, so an `LMResponse` became `[LMResponse]` and dspy raised
  `Sub-LM response must contain text`. Invisible on the default path, fatal under
  `dspy.context(experimental=True)`. Both the read and the rebuild now live in `_dspy_compat` rather
  than encoding dspy's convention at the call site, which is why no test could see this expire. **A
  shape the shim does not RECOGNISE is returned untouched so dspy raises its own error**: rebuilding it
  converts a loud failure into a silent empty completion that would reach the planner and then the RL
  data as a real escalation answer.
- **`model_as_tool` had the same defect** sixty lines away: `outputs[0]` on an `LMResponse` handed the
  model `str(LMResponse)`, the whole repr, and wrote it to the trace as the tool's result.
- **Substituting text into an `LMResponse` no longer duplicates it.** `LMOutput.text` JOINS every text
  part, so replacing only the first left the rest appended (`"AB"` round-tripping to `"ABB"`). Thinking,
  tool-call, citation and refusal parts and every sibling field survive.

### Added

- **Every sub-LM escalation is traced automatically.** The invariant has always said a sub-LM call "is
  recorded as a `sub_call`"; it was true only when the consumer remembered to call `intercept_sub_lm`
  itself, since a plain `dspy.LM` is invoked by dspy directly and recorded nothing. `RLMTask` now wraps
  a plain `sub_lm` at the same per-run seam that binds the recorder.

  **Not a hypothetical gap.** Surveyed across the consumer fleet, four projects never wrapped, and two
  of those had corpora, 141 traces, in which `sub_call` was identically zero, indistinguishable from
  "measured, and the model never escalated". That ambiguity got into a design decision in this repo:
  the speculative-tool-calling deferral cited the zero as evidence the model never escalates. Re-derived
  from code content the real rate was 0.15%, and one of those escalations proved to be a 235.5s call.
  **An absent event is not a measurement.**

  A consumer with its OWN recording wrapper opts out by declaring `records_sub_call = True` on it,
  probed with `is True` rather than truthiness, because `getattr` on a mock manufactures a truthy
  attribute for any name and a truthiness probe would silently skip it. Auto-wrapping never raises: a
  sub-LM it cannot wrap is used bare with a warning, since it is an observability convenience the caller
  did not ask for.

### Changed

- **Traces from a consumer that never wrapped gain `sub_call` events they did not have**, carrying the
  escalation prompt (`input`, truncated to 4,000 chars). A corpus spanning the upgrade is not
  homogeneous, and `run_start.rlm_harness` (1.6.0) separates it.
  `metrics.compute_run_utilization`'s `sub_calls_total` moves from an unmeasurable zero to a real count,
  and `export_actions` gains `kind="sub"` records, which is the point: a trainer doing credit assignment
  previously could not see an escalation that happened. `export_sft_turns` / `export_rl` are unaffected.
- On the automatic path `attempt` is structurally always `1`, and `raw` is now always a string or `None`,
  `None` meaning the shape was not recognised, where a legacy dict output used to be written verbatim.
- **A duck-typed sub-LM returning a bare `str` now fails where it used to work.** The old `[outputs]`
  normalisation turned `"hello"` into `["hello"]`, which dspy accepts. That is the same rule that stops
  a silent empty completion and is the right default, but for this one input it converts a working call
  into a hard failure. A sub-LM deriving from `dspy.LM` is unaffected.
- A consumer driving `dspy.RLM` directly, without `RLMTask`, gets none of this.

### Upgrading: check the upgrade actually took

This release is only visible as a change in what traces CONTAIN, so a silent no-op upgrade produces
exactly the wrong conclusion: "the events did not appear" rather than "I am still running the old
version". One way that happens, verified rather than assumed:

    # pyproject.toml bumped to 1.7.0, lockfile still pinning the old version
    uv sync --frozen   ->  exit 0, installs the OLD version, no warning anywhere
    uv sync --locked   ->  exit 1

`--frozen` means "sync without updating the lock" and checks nothing; `--locked` asserts the lock agrees
with the manifest. The same shape exists for any lockfile workflow: bumping the manifest is not the
upgrade, re-resolving is. **Confirm at runtime**, via `run_start.rlm_harness` in a new trace or
`python -c "import rlm_harness; print(rlm_harness.__version__)"` from inside the deployed environment.
**Not with `uv run`**, which re-locks implicitly and so repairs the drift it is being used to measure,
manufacturing the reassuring answer.

### Retracted

- The 1.6.0 entry stated that a consumer "records **zero `sub_call` events, in every run**: the model
  never escalates to the sub-LM at all." **That inference was wrong**, and it is the exact failure this
  release removes: the zero measured that consumer's wiring, not its model.

## [1.6.1] - 2026-08-29

Two correctness fixes in shipped code. No new public name, no new payload field, no schema change.

### Fixed

- **A root turn's recorded `ts` could come from an earlier turn, and it reached a rendered UI.**
  `Adapter.__init_subclass__` re-wraps `format` and `parse` with `with_callbacks` for EVERY subclass,
  whether or not that subclass redefines them, so each subclass level adds a callback fire that a
  `super()` call then traverses. Measured, one root turn:

      stock `JSONAdapter`                          1 fire
      `runtime._LenientJSONAdapter` (the DEFAULT)  2   (it calls `super().parse`)
      a consumer subclass overriding NOTHING       3
      ...that also calls `super().parse`           4

  **The three-fire row is the one to read**: subclassing the kit's adapter to set a single class
  attribute, overriding no method at all, was enough to add a fire, which is not what anyone would
  predict from "it calls `super().parse`". A fix that divided by two would have left that consumer
  broken while passing every test.

  So under the DEFAULT adapter every root turn fired `task._MainStepTimer` TWICE with identical outputs,
  and `record_main_trajectory` matches a turn to its live stamp by `reasoning`, so the surplus stamp was
  claimable by any LATER turn repeating that string, which a retry loop does. Measured across 85 real
  traces: all 12 with a `ts` inversion had a duplicated reasoning and none of the 58 with unique
  reasoning did; 2.1% of per-turn deltas came out NEGATIVE, and a consumer rendered one as a
  **-338.7s** turn duration.

  `_MainStepTimer` now stages the OUTERMOST parse only, via a per-thread depth from the public
  `on_adapter_parse_start`/`on_adapter_parse_end` pair, which makes the match an identity map. Fixing it
  in the matcher was tried and rejected: it cannot repair ADJACENT duplicate turns, where it merely stops
  the delta going negative while the stamp stays ~0.1s wrong, trading a loud failure for a silent one. If
  a future dspy stops firing `on_adapter_parse_start` the depth never rises and behaviour degrades to
  exactly what it was before, never to staging nothing.
- **`record_main_trajectory`'s two matchers now scan forward only**, defence in depth rather than the fix
  above. Trajectory order is chronological order, so a cursor parked past the previous match makes that
  an enforced property. `_match_exec` had no demonstrated defect but gains the rule for one real case:
  dspy runs a setup `execute()` before the turn loop whose duration a turn with a colliding code string
  could otherwise claim.
- **`verify_quote` refused correct citations at non-word junctions.** Whitespace runs were joined with
  `\s+` uniformly, so a quote that reflowed a line break beside a delimiter failed:

      source  x = """One line.\nAnd another."""
      quote   """One line.\nAnd another.\n"""      -> MISMATCH, wrongly

  The joiner is now junction-aware: `\s+` between two word characters, so `foo bar` still cannot verify
  against `foobar` (the false-positive direction is the worse one and stays closed), and `\s*` elsewhere.
  Word-ness uses Python's UNICODE `\w` deliberately: `你好 世界` keeps requiring its space against
  `你好世界`, which an ASCII character class would have silently started accepting. Found by inspection,
  not by a failure: across ~479 real citations the old and new rules never disagreed, so this is a
  confirmed-real but confirmed-harmless defect on today's corpora.

### Changed

- **`exec_duration_s` coverage may move slightly.** Under the forward-only cursor a turn whose only
  matching entry sits before the cursor loses its match, so the field can be ABSENT where it was present
  and a `ts` can fall back to flush time. Both are optional in `trace/v1` and no reader breaks, but a
  consumer tracking coverage will see the number change.
- **`verify_quote` can now report an EARLIER occurrence.** Loosening a junction lets a match start where
  it previously could not, so the offset in `MATCH: found at line N (char M)` moves: `a . b` against
  `a.b ... a . b` reported char 8 and now reports char 0. The line number is often unchanged, so this is
  not visible from a line-level check.

### Docs

- **New "Reading a trace: the ordering rules" section** in the guide. `main_step` events are written in
  one block after the run, so a `tool_call` precedes them in file order while being chronologically
  later (70 of 76 traces); `payload["turn"]` is authoritative and file order already matches it (72 of
  72); `ts` places turns against tool calls and nothing else. Written because a consumer reading these
  traces concluded "sort by `ts`", which reorders turns.
- **Corrected the claim that every local tool is sub-millisecond.** `make_grep_files_tool` ships
  `per_match_timeout_s=1.0` and `max_total_time_s=30.0`, so it is not sub-millisecond in principle and
  `compute_tool_waste` is blind to its worst case. It stays untimed because it is sub-millisecond in
  practice where measured (n=146, median 0.029s, max 0.746s, zero calls over a second).

## [1.6.0] - 2026-08-27

Three new public names and three new optional `trace/v1` payload fields, all additive. No behaviour
change to any existing call path.

**This release is the groundwork a measurement asked for before the feature it was scoping.** 1.6.0 was
going to be speculative programmatic tool calling: parse the REPL code as the Root LM streams it,
pre-launch the tool calls it contains, serve them from cache. Measured against ~400 real runs from the
downstream fleet before any of it was written, and re-derived by an independent audit, there was almost
nothing to speculate:

- The fleet's dominant tool, 80% of its wall-clock, is a **serial repair loop**, each spec built from
  the previous result. 913 call sites in the corpus; exactly **one cell** ever contains two independent
  ones. Speculation ceiling there: **0.06-0.62%**.
- The newest and heaviest consumer (76 runs, 17.9 turns/run) records **zero `sub_call` events, in every
  run**: the model never escalates to the sub-LM at all.

These are numbers about two workloads at one point in time, not a verdict on the technique: a workload
that fans out, the map-reduce-over-chunks shape the RLM paper describes, would read completely
differently, and the fields below are what would show it. What the same investigation found instead is
that **this kit cannot be measured from its own traces**:

- 99.8% of the heaviest consumer's wall-clock landed in one opaque bucket, the gap before a
  `main_step`, which mixes root-LM GENERATION with sandbox EXECUTION. Different fixes, nothing
  separating them.
- Not one of the 3,329 `tool_call` payloads in the corpus carried a duration, so every wall-clock
  attribution had to infer durations from inter-event gaps, which charges a whole turn's model
  generation to that turn's first tool call. **Every number produced against this corpus before 1.6.0
  had that error.**
- No trace said which kit wrote it. `schema` is the FORMAT version. **Most of the corpus predates
  the fleet's move to a released version and there is no way to tell which parts**, so anyone
  re-analysing it cannot split it by kit version at all.

### Added

- **`run_start.payload["rlm_harness"]`**: the kit version that wrote the trace, beside `meta` rather
  than inside it, since `meta` is the caller's namespace.
- **`tool_call.payload["duration_s"]`**: an explicit optional parameter on `record_tool_call`, written
  only when given. The shipped tools whose cost is a WAIT on something outside this process pass it:
  `fetch_url`, `web_search`, `run_command`, `git_clone` (spanning both attempts when the credentialed
  fallback runs), `model_as_tool`, and every MCP tool. A local read/grep/edit does not, so the metric
  says "not recorded" rather than "free". Enforced, not remembered: `tests/test_tool_durations.py`
  requires every `make_*` in `rlm_harness.tools.__all__` to be classified as outbound-and-timed or
  exempt WITH a written reason, because `git_clone` and `model_as_tool` both shipped without a duration
  in this release's own first draft. For the two BASE factories the kit cannot record it:
  `make_model_tool` / `make_harness_tool` are deliberately side-effect-free, so the consumer's wrapper
  owns the `record_tool_call`, and one `duration_s=` from there is what makes the 80% visible.
- **`main_step.payload["exec_duration_s"]`**: how long the sandbox spent on that turn. Staged by the two
  interpreter wrappers the kit owns and **matched onto turns by the CODE that ran, never by position.**
  A turn does not always reach the sandbox: dspy raises `SyntaxError` out of `_strip_code_fences` for an
  explicitly non-Python fence tag and records that turn without calling `execute()`, so a positional zip
  credits the skipped turn with the NEXT turn's time and shifts every one after it, a confidently wrong
  attribution with nothing to signal it. An interpreter a caller injects directly is not wrapped, so its
  turns carry the key ABSENT rather than a wrong zero.
- **`compute_tool_waste` / `compute_tool_waste_by_run` / `ToolWaste`**: per tool, calls split by outcome
  and what each outcome cost. The number nobody had: on this corpus **57% of all tool wall-clock produced
  output the consumer's own validator rejected**. For scale, a `max_consecutive_invalid` of 2 instead of
  5, a knob `make_model_tool` already ships, would have saved 4.53h of 20.6h there, roughly 2000x the
  speculation ceiling, available today by changing one number.

  Two things it refuses to do. It never reads `ok` directly, since outcomes come from `payload_cause`
  because `ok` is frequently ABSENT on an endpoint-failure payload, so a naive counter absorbs
  infrastructure failures as content declines, **a mistake that has shipped four times**. And it never
  infers a duration from event gaps: `*_seconds` is `None` for anything unmeasured, because `0.0` would
  read as "measured and found to be free".

### Deferred to 1.7.0, with what was learned

Speculation itself, and the `speculatable()` marker that goes with it: a public marker with no engine is
dead surface, and post-1.0 it would be SemVer-frozen from the day it ships.

**Scope note, because an earlier draft of this entry got it wrong.** The upstream technique is NOT
sub-LM-only. Its gate is SIDE EFFECTS, and it names search APIs alongside sub-agents as the canonical
high-latency target, so `fetch_url` and `web_search` are squarely in the intended set. What excluded them
HERE is a stricter, separate concern: a speculation launched and then discarded has already been observed
by the remote end, and nothing at commit time unsends it. That is a POLICY judgement belonging to the
consumer, through an explicit opt-in, not a technical impossibility. `make_harness_tool` is the one
genuine permanent exclusion: a cached result would replay a `child_run_id` pointing at a different child
rollout.

Streaming is deferred with it, and four defects in that path were verified while scoping it, two of which
silently break documented invariants. `dspy.streamify` runs the program under an anyio task group and
anyio does not unwrap a single child exception, so `SandboxCancelled` arrives as an `ExceptionGroup`,
`_retry.py`'s `except non_retryable:` stops matching, and **a caller-driven cancel gets RETRIED**;
`is_fast_fail_lm_error`'s `isinstance` fails the same way. Also `StreamListener` defaults
`allow_reuse=False` so it streams turn 1 only; setting `send_stream` hijacks a consumer's own
`dspy.streamify`; and `streamify` captures `settings.callbacks` at construction, dropping
`_MainStepTimer` so every `main_step` silently reverts to a flush timestamp.

## [1.5.0] - 2026-08-27

One new public name and two corrected dependency floors. **The headline is a correctness fix, not a
feature**: on the MCP SDK major a fresh install resolves today, a FAILED MCP tool call was reported to
the model, and recorded in the trace, as a SUCCESS. `trace/v1` is untouched.

Both floors move, `dspy>=3.3.1` (was `>=3.3.0`) and `mcp>=1.8.1` (was `>=1.0`). **Neither is a
breaking change for a consumer: nothing pins dspy or mcp directly.** Verified on Python 3.11 and
3.13, and on mcp 1.8.1 (the declared floor), 1.28.0 (the lock) and 2.1.1 (the newest).

### Fixed

- **An MCP tool failure read as a success (mcp SDK 2.x).** The SDK renamed its model fields
  camelCase to snake_case at 2.0 and kept the old spellings only as pydantic *serialization* aliases, so
  attribute access under the old names no longer resolves. `rlm_harness/mcp.py` read three of them:

  | read | on 2.x | consequence |
  |---|---|---|
  | `Tool.inputSchema` | `input_schema` | `AttributeError`: `mcp_tools()` died outright |
  | `getattr(result, "isError", False)` | `is_error` | **every failed tool call reported `ok`** |
  | `getattr(result, "structuredContent", None)` | `structured_content` | structured results dropped |

  Only the first failed loudly. The second is the one that matters: the model was told a failing tool had
  succeeded, and `record_tool_call` wrote `ok=True` into the trace, so a dataset built from those
  rollouts learned from them. A fourth shape had to be handled with them, since 2.x widened
  `call_tool`'s return type to a union whose other arms carry no `content` or error flag;
  `_is_tool_result` recognises those BEFORE the flag is consulted, which would otherwise have
  reintroduced the same defect through its own fix.

  All three renames now go through one `_sdk_field` accessor reading both spellings. It returns a
  `_MISSING` sentinel rather than a default, deliberately: `structured_content` is typed `Any` on 2.x so
  `{}` / `0` / `False` are legitimate VALUES an `or`-chain would discard, and a
  `getattr(obj, name, False)` is exactly what turned the NEXT rename into a wrong answer instead of an
  error. A result carrying no error flag under either spelling now logs a warning once instead of being
  assumed successful.

  **Why no test caught it:** the three fixture servers imported `mcp.server.fastmcp.FastMCP`, which does
  not exist on 2.x, so the suite could only ever run against 1.x, and the error flag was covered
  exclusively by `SimpleNamespace` fakes spelling it `isError`. The fixtures now build against whichever
  major is installed and a live server that actually raises pins the error path end to end.
- **`RLM_REQUEST_TIMEOUT` was a silent no-op on the Claude-subscription route.** It becomes
  `dspy.LM(timeout=...)`, which an auto-routed `ClaudeAgentLM` never sees, so a consumer could set it,
  see no error, and believe a route was bounded. `configure()` now warns, naming
  `configure(main_lm=ClaudeAgentLM(model, timeout_s=...))` as the seam that does choose it. It is
  deliberately NOT forwarded as `ClaudeAgentLM(timeout_s=...)`, which was the obvious fix and is wrong:
  `request_timeout_s` bounds one HTTP request with dspy and litellm retrying around it, while
  `ClaudeAgentLM.timeout_s` is an end-to-end per-call deadline that INCLUDES time queued behind that
  SDK's concurrency semaphore. Under `llm_query_batched`'s thread fan-out, a value generous per request
  would make queued sub-LM calls time out from waiting alone. One number cannot mean both.
- **The action prompt described the wrong runtime for every non-Pyodide interpreter.** dspy 3.3.1 renders
  an "Execution environment:" section from `interpreter_factory.execution_instructions`, and the kit
  supplied its interpreter POSITIONALLY, so dspy read the attribute off its own default
  `PythonInterpreter` and told every run that subprocesses and native extensions are unavailable,
  including a `container` run, whose entire reason to exist is that they are. Nothing went red; the model
  simply stopped trying. The kit now passes an `interpreter_factory` that is a metadata CARRIER only,
  which raises if ever invoked, on the premise that dspy never does.

### Added

- **`short_error`**: head-and-tail elision for a caught exception, so a giant `AdapterParseError` becomes
  one readable log line. It existed as `_retry._short_error` and is public because two independent
  consumers had reached into the private module for it, which is this project's stated trigger for
  promoting a named hook. Its BEHAVIOUR is frozen, not its output string. It also no longer raises when
  an exception's own `__str__` does, since every call site is an `except` block. And the length bound it
  promises is one it now keeps: at `limit <= 1` the head/tail split left `tail == 0`, and `text[-0:]`
  slices the WHOLE string, so the smallest budgets produced the longest output. Found by holding the code
  to the contract the docstring had just frozen.
- **`mcp-latest` in `dspy-latest.yml`** (now "newest deps") and **`mcp-major`, a pinned 2.x leg in
  `ci.yml`**. The uncapped `mcp` extra had no defence at all, which is why the bug above shipped. The
  assert reads `importlib.metadata`, not `mcp.__version__`, which the package exposes on neither major.
- **`execution_instructions` on the kit's own interpreters**: `ContainerInterpreter`, the `mock`
  interpreter and `testing.ScriptedInterpreter` each describe their real runtime. The container's is
  DERIVED from its `ContainerConfig` rather than a constant, since `network`, `read_only` and `workdir`
  are operator-configurable and a fixed string would eventually claim a capability is absent when it is
  present: the same defect class as the Pyodide default it replaces, just quieter.

### Changed

- **Floor `dspy>=3.3.1`**, required rather than preferred: `execution_instructions` does not exist in
  3.3.0. 3.3.1 also re-parents `CodeInterpreterError` under `DSPyError`, the hierarchy the
  cancel-is-never-recoverable invariant reads, now pinned by a test asserting `SandboxCancelled` is a
  subclass of neither the recoverable nor the terminal class, nor of the new root.
- **Floor `mcp>=1.8.1`.** `>=1.0` was never runnable: `mcp.client.streamable_http` does not exist at
  1.0.0, so the HTTP transport could not work at the declared floor. 1.8.1 rather than 1.8.0 because a
  floor is only meaningful if the suite passes on it: on 1.8.0 a refused connect takes ~16s against the
  <10s this kit asserts for fail-fast. Still no upper bound, with the two new jobs as the compensating
  control.

### Documentation

- **A new guide section, "Timeouts: what bounds what".** Six bounds on three clocks, and none of them
  bounds a run's wall time on its own. It states the three things that are easy to get wrong:
  `request_timeout_s` is per ATTEMPT and dspy/litellm retry around it, so the run-level wait is a
  multiple; the two 600s defaults are different quantities selected by a model string; and a sandbox turn
  timeout is recoverable while a cancel deliberately is not. It also records that these semantics assume
  non-streaming requests.

## [1.4.0] - 2026-08-27

One new public name, additive and off by default. **MINOR, not PATCH**: adding a public name is a minor
bump by this project's rule. With the field unset every call site behaves exactly as in 1.3.0.

### Added

- **`RLMConfig.request_timeout_s` / `RLM_REQUEST_TIMEOUT`** (seconds): a wall-clock cap on ONE model
  HTTP request ATTEMPT, handed to `dspy.LM(timeout=...)` and from there to litellm. A pass-through
  rather than new machinery.

  **Unset is not "no cap".** With nothing passed, litellm applies its own
  `COMPLETION_HTTP_FALLBACK_SECONDS` of **600 s** per attempt, so this field REPLACES that number
  rather than introducing a bound where there was none, and a consumer whose turns legitimately exceed
  ten minutes must set it UP rather than leave it alone.

  **It also does not bound a run to its own value.** dspy passes `num_retries=3` and litellm's first
  call hands the OpenAI SDK `max_retries=2`, so a dead endpoint is retried and the run-level wait is a
  MULTIPLE of this plus backoff. Size a caller-side budget on the multiple.

  What prompted it: `sandbox_turn_timeout_s` bounded the sandbox side of a turn and the model side was
  not settable at all. Against a self-hosted OpenAI-compatible endpoint one request never came back:
  socket `ESTABLISHED`, both queues empty, the worker asleep in `epoll_wait` for 38 minutes at 0.3% CPU,
  while that endpoint answered unrelated requests in half a second. Stated honestly: 600 s across four
  attempts is about forty minutes, so that observation fits "litellm's own default, retried" at least as
  well as "nothing was watching", and no attempt counter was captured. Either way the consumer could not
  choose the number, and now can.

  When unset the key is ABSENT from the LM kwargs rather than present-and-`None`, since clients differ on
  what an explicit null means. Both roles get it, which matters because `dspy.RLM` fans the sub-LM across
  a thread pool where a wedged request is less visible.

### Documentation

- **`max_tokens` now names the failure consumers keep misdiagnosing.** Its docs covered only the `None`
  case. Running into the DEFAULT presents completely differently: 8192 must hold one turn's
  chain-of-thought AND its structured answer, and a long turn that overruns it is cut off mid-JSON, so
  the run surfaces as `RLMTaskError: Failed to produce a valid '<field>'` caused by `AdapterParseError`.
  That is a truncation, not a model failing a schema, and it is repeatedly diagnosed as the latter
  because the quoted response looks well-formed right up to where it stops. The guide names the symptom
  and how to tell them apart: read the END of the quoted response, where a truncated one has no closing
  brace, and dspy separately logs `LM response was truncated due to exceeding max_tokens=...` at WARNING.
- Also states what raising it costs: an OpenAI-compatible server commonly validates
  `prompt_tokens + max_tokens` against the context window, so a bigger cap removes usable PROMPT budget,
  and an RLM planner's prompt grows every turn. And OpenAI's own reasoning models reject the 8192 default
  outright at `configure()` time (`LMConfigurationError: ... max_tokens >= 16000 or None`), which is why
  16000+ is the right floor for them specifically.

## [1.3.0] - 2026-08-25

Twenty new public names and two new optional extras (`grep`, `gitignore`). No trace-format change;
every existing call site behaves byte-for-byte identically to 1.2.1.

**This batch introduces the kit's first file-mutation, data-loss-capable tools**
(`make_write_file_tool` / `make_edit_file_tool`). Every tool shipped before it was either read-only
against the filesystem or delegated execution and network entirely to a consumer-supplied runner or
fetcher, so a bug in either of these two can destroy content rather than merely return wrong
information.

### Added

The local-filesystem and command surfaces, plus the isolated-subprocess primitive under them.
`rlm_harness/README.md` documents every parameter; this entry keeps the decisions and the bugs.

- **`make_read_file_tool` / `make_grep_files_tool` / `resolve_within_root`**: a bounded directory, no
  shell. Both take `name=` (defaulting to `"read_file"` / `"grep_files"`), because two roots on one
  task would otherwise collide on a duplicate tool name and abort registration for every tool.
  `list_candidate_paths(root)` returns a frozen `CandidatePaths(paths, truncated)`. A new optional
  `gitignore` extra (`pathspec`) gives correct `gitwildmatch` syntax; **root-level `.gitignore` only,
  stated honestly**, with no nested per-directory merging and no global excludes. **`.git` is always
  excluded, unconditionally, by two distinct mechanisms**, so neither one failing opens it.
- **`make_write_file_tool` / `make_edit_file_tool`**: whole-file write and exact-string-anchor
  replacement, both atomic, both over the same bounded root. On success a windowed snippet of the
  result is appended (`show_snippet=`, default **True**), which is how a task can feed the model
  guttered text without ever having considered line numbers.
- **`make_git_clone_tool`**: shallow by default (`default_depth=1`), because the alternative is being
  tricked into cloning an entire history. Auth is a fallback, trying without credentials first.
  **Credential redaction is best-effort and disclosed as such.**
- **`make_extract_archive_tool`** (`tools/archive.py`) plus `atomic_write_stream`.
  `zipfile.extractall()` / `tarfile.extractall()` are not safe by default: an entry can carry an
  absolute path, a `..` traversal, or a tar symlink pointing outside the target. **Two-pass
  extraction, matching the kit's refuse-outright-never-partially-mutate posture**: pass 1 validates
  every entry's metadata only and refuses the whole operation upfront. Password-protected archives are
  refused there. `atomic_write_stream` aborts the moment a running total exceeds `max_bytes`, checked
  after every chunk rather than once at the end.
- **`run_in_subprocess`** (`isolation.py`): a safe isolated-subprocess primitive with `spawn` pinned on
  every platform, since forking a host already running an event loop risks inherited locks and
  half-open sockets. `factory` must be a picklable module-level callable. Timeout escalation is
  `terminate()`, a grace period, then `kill()`, then a final reap; **SIGTERM does not reliably let the
  child's `finally`/`atexit` run unless the child installs a handler.**
- **`verify_quote(source, quote)`** (top level, not `rlm_harness.tools`): verifies a quote appears in a
  source, verbatim or whitespace-normalised. No `regex` extra needed, since the quote is literal text.
- **`refuse_broad_git_history`**: an opt-in `guard` for `make_command_tool`.
- **`atomic_write_text`** promoted to the top level: same-directory temp file, `fsync`, `os.replace`,
  permission preservation.
- **`pointer_to_invocation`** and **`run_isolated`**: the canonical mapping from a served harness's
  pointer back to an invocation, and a small async bridge for a consumer building its own in-process
  delegation transport.
- **`RunUtilization` / `compute_run_utilization` / `compute_utilization_by_run`.**
- **Claude-subscription auto-routing in `configure()`**: a `main_model`/`sub_model` string carrying the
  subscription prefix routes to `ClaudeAgentLM` without the caller constructing it.

### Fixed

Four bugs, each found by a different instrument, which is the part worth keeping.

- **`multiprocessing.Queue.put()` does not pickle synchronously**, found in design review. A background
  feeder thread does, and a pickling failure there is logged and silently dropped, never raised back to
  `put()`'s caller. The child now test-pickles its payload synchronously, in its own code, before ever
  calling `put()`, falling back to a plain-string `RuntimeError`. The parent's `queue.get()` also
  carries its own bounded timeout as a backstop against an out-of-band kill that bypasses this
  primitive's escalation entirely.
- **A Python 3.11-only crash in `make_extract_archive_tool`'s own refusal path, caught by CI's 3.11 job
  while 3.12, 3.13 and a full local run were green.** Pass 1 read each zip entry's `is_dir()` before
  checking its name, and CPython 3.11's `ZipInfo.is_dir()` detects a trailing `/` with `filename[-1]`,
  so an entry with an EMPTY name raised a raw `IndexError` out of the tool instead of the intended
  `Refused:` string. 3.12+ uses `endswith("/")` and returns `False`, which is why exactly one matrix
  cell went red. **Fixed by ORDERING rather than by another entry in the caught-exception tuple**: the
  name is read and refused first, so no name-derived accessor can see a degenerate name. Pinned on
  EVERY version by a stub whose `is_dir()` raises unconditionally, so the ordering cannot regress on a
  developer machine running 3.12+.
- **`max_memory_mb` can starve the relay it needs to report, confirmed on real Linux CI.** An
  aggressively low cap has the child correctly hit `MemoryError` while being too memory-constrained for
  `Queue.put()`'s feeder thread to start, crashing it before anything relays. No fallback is possible,
  since a resource-exhausted process cannot report its own exhaustion through a mechanism that needs
  resources; it degrades to the parent's bounded `queue.get()` timing out, which is the correct
  accepted outcome rather than a bug. Related and platform-shaped: `RLIMIT_AS` bounds virtual address
  space, not physical memory, and **the macOS kernel refuses to lower it from unlimited at all**, so
  `max_memory_mb` is effectively Linux-only. `cpu_time_limit_s` (`RLIMIT_CPU`) enforces correctly on
  every POSIX platform tested, macOS included. Either way it fails loudly.
- **`atomic_write_text` had a real bug, found by its first tool-level consumer one release after it
  shipped**, and **`verify_quote` stripped leading and trailing whitespace off the quote before
  matching**, found by direct exercise rather than by review.

### Known, accepted

- `atomic_write_text`'s guarantee is "no torn file", not "no lost update": two concurrent writers of
  the same path both complete and the last `os.replace` wins.

## [1.2.1] - 2026-08-23

**Fast-failing non-retryable LM errors**: the item 1.2.0 left open. No public surface change, no
trace-format change; `_retry.py` and `_dspy_compat.py` are both private modules.

### Before you upgrade

An LM error dspy itself classifies as non-retryable (`LMAuthError`, `LMBillingError`,
`LMConfigurationError`, `LMUnsupportedModelError`, `LMUnsupportedFeatureError`, `LMUnexpectedError`) now
escapes `RLMTask.arun()`/`run()` **as that original dspy exception**, after exactly one attempt, where it
previously burned the full `max_retries` budget re-running the same doomed trajectory and was then
wrapped in `RLMTaskError`. If you catch `RLMTaskError` expecting it to be the only failure type, add a
matching `except dspy.LMError:` alongside it. The two convey different things: `RLMTaskError` now means
"the model kept producing invalid output", while an `LMError` means "the call to the LM itself was never
going to succeed".

### Why

`run_with_retry` treated every exception the same, so an invalid API key was retried `max_retries` times,
re-sending the identical doomed request, and the wrapping then erased the one piece of information a
caller needs to tell "fix your API key" from "the model can't do this task". dspy 3.3.0 ships the
classification this needed, `dspy.is_retryable_lm_error(exc)`, which 1.2.0's floor bump made writable.

### The one carve-out: `ContextWindowExceededError`

dspy's classification assumes a retry re-sends the *identical* request, which is true for the
provider-level retries it is documented for. It is not true here: `run_with_retry` re-runs the **whole
trajectory**, and a different turn sequence can genuinely produce a shorter prompt that fits on a later
attempt. So `ContextWindowExceededError`, a non-retryable `LMInvalidRequestError` by dspy's own rule, is
excluded from the fast-fail set and keeps retrying. This was the residual question 1.2.0 left contested;
it is resolved rather than left for whoever next reads that note.

### Added

- `_dspy_compat.is_fast_fail_lm_error(exc)`: resolved through the PUBLIC `dspy.is_retryable_lm_error`,
  never the private tuple it is built from. Degrades to `False`, today's pre-1.2.1 behaviour, if the
  installed dspy is missing `LMError` or `is_retryable_lm_error`, so a future rename fails safe rather
  than over-eagerly killing a run that would have succeeded on retry.
- `_retry.py:run_with_retry` gained an `is_fast_fail` predicate alongside the existing `non_retryable`
  type allowlist. A predicate rather than another type tuple, because "is an `LMError`, is NOT
  `ContextWindowExceededError`, and `is_retryable_lm_error` says no" cannot be expressed as a static
  `except (A, B, C):`, and it needs a runtime decision `_retry.py` itself must not know how to make.
  Matched exactly like `non_retryable`: the original exception propagates verbatim, consumes no attempt,
  and is never wrapped. Checked second, since a type match is cheaper. Default `None` never fires.

### Fixed (release pipeline, no package-content change)

- **The first attempt to publish this release failed before reaching PyPI.**
  `pypa/gh-action-pypi-publish` was SHA-pinned to `v1.14.0`, whose bundled twine does not recognise
  `Metadata-Version: 2.5`, which `hatchling>=1.27`'s PEP 639 support emits for every wheel this kit
  builds. The publish job rejected the wheel with `InvalidDistribution` before contacting PyPI. Fixed by
  bumping the pin to `v1.14.2` (twine v7).

## [1.2.0] - 2026-08-07

**Requires `dspy>=3.3.0`.** The 3.2.x compatibility branches are deleted.

### Before you upgrade

If you pin `dspy==3.2.x`, this upgrade will **fail to resolve** at install time. Either unpin dspy (the
kit's documented model is "pin the KIT, not dspy") or stay on `rlm-harness~=1.1.0`. Your Python floor is
unaffected: dspy 3.2.1 and 3.3.0 both require `>=3.10,<3.15`. And dspy 3.3.0 carries FEWER exact pins
than 3.2.1, dropping `asyncer`, `typeguard`, `numpy` and `xxhash`, so for most consumers the bump
relaxes the transitive constraint graph rather than tightening it.

### Why

3.2.x is **strictly worse for users**: it accepts DUPLICATE tool names and silently keeps only one, so
the model is never told the other exists (3.3.x raises); it does not reject a keyword tool name at
construction, which becomes a Deno `SyntaxError` at registration instead, aborting every tool on the
task; and it has no `CodeExecutionError`, so a recoverable REPL error cannot be distinguished from a
terminal one. The kit's own guarantees differed by whichever dspy a consumer happened to resolve.

### What was deleted, and what deliberately was not

Gone: `rlm_accepts_interpreter_kwarg` and the `RLM(interpreter=…)` branch, the `max_iterations` budget
alias, `_RESERVED_TOOL_NAMES`, the `getattr` fallback in `recoverable_interpreter_error`, and the 3.2.x
rationale prose.

**`_dspy_compat` itself stays, and so do its shims** even where each now resolves a single answer. Its
value was never "supports two versions": it is that every dspy fact lives at ONE introspected call site,
so the next rename is a one-line change plus a red test here instead of a silent behaviour change in
someone's rollout. dspy has now renamed in a minor release once, and two of those three renames were
silent. **Do not collapse a shim into its call site just because it currently has one branch.**

### CI

- **New `packaging (fresh install)` job**: builds the wheel, installs it into a clean environment and
  runs a task from it, with dspy PINNED to `uv.lock`'s version, since it runs on `pull_request` and a
  floating dspy would let an upstream release redden a contributor's unrelated PR. That axis was
  genuinely untested: every other job runs from the source tree, so a module missing from the wheel would
  have shipped silently. It is explicitly **not** a substitute for the dspy axis: measured, an offline
  end-to-end run still returns the correct answer while a renamed dspy kwarg silently drops the caller's
  budget cap.
- **`dspy-latest.yml` is now the only dspy axis**, says so in a `::notice::`, and restores real
  two-version coverage by itself the day dspy 3.4 ships.

### Fixed before release

- **`interpreter="mock"` could not run a task**, and the floor bump is what broke it: 3.2.x took the
  interpreter as a constructor kwarg and validated nothing, while 3.3.x `isinstance`-checks it against
  the `@runtime_checkable` `CodeInterpreter` protocol on EVERY forward pass, and `_MockInterpreter` was
  missing `tools` and `start`. Invisible to the suite, because every mock test either stopped at
  `_build_rlm()` or injected a `ScriptedInterpreter`, which overrides the string path. Now implements the
  full surface, with a regression test driving a real forward pass through the STRING `mock` path.

Verified on dspy 3.3.0, which the floor bump makes the only version it runs on.

**Still open, not in this release: fast-failing non-retryable LM errors.** The floor bump makes the
shim writable (`isinstance(exc, dspy.LMError) and not dspy.is_retryable_lm_error(exc)`), but the
residual set is contested: `ContextWindowExceededError` should probably still retry here, because
`run_with_retry` re-runs the whole trajectory and may produce a shorter context that fits. 1.2.1
resolves it.

## [1.1.0] - 2026-08-07

Additive on the API surface: nothing removed, renamed or re-typed, and `rlm-harness/trace/v1` is
untouched. **Two behaviour changes to know about before upgrading:**

- `RecordedToolProvider.replay` now RAISES on a `preview`-only record where it previously returned
  `None`, which affects replaying a trace containing MCP or `read_skill` calls. It is the correct
  posture, since it was silently serving nothing, but it is a change rather than an addition.
- `tools` is no longer annotated `ClassVar`, so if you also declare `tools: ClassVar[...]` on an
  `RLMTask` subclass a type checker will flag "cannot override instance variable with class variable".
  Drop the `ClassVar` on your side; runtime behaviour is unchanged either way.

### `RLMTask(tools=…)`: the kwarg the guide already documented

`tools` was a `ClassVar` only, so the guide's own **runnable** examples raised
`TypeError: RLMTask.__init__() got an unexpected keyword argument 'tools'`. For MCP that was the ONLY
documented attach path, and it cannot be a class-body list because the tools exist only inside the
`with mcp_tools(...)` block.

The override is resolved at BUILD time (`RLMTask.resolved_tools`, also new) rather than assigned in
`__init__`, which is what makes it order-independent: `self.tools = tools` would make the winner depend
on where the subclass calls `super().__init__()`, and assign-after-super, the more idiomatic ordering,
would have silently clobbered the caller's explicit kwarg. `tools=` REPLACES the declaration and never
merges, since merging would make the effective list depend on inheritance depth; `tools=[]` is a
deliberate "no tools", distinct from the `None` default.

### Public REPL-safety rules: `sanitize_tool_name`, `unique_tool_names`, `is_valid_tool_name`, `signature_from_json_schema`

A consumer driving `McpCatalog` gets the server's RAW tool names and schemas and builds its own
`dspy.Tool`s, hitting exactly the defects 1.0.2 fixed inside `mcp.py` with no sanctioned remedy: both
halves were private, and "consumers EXTEND, they don't fork" bars reaching into a `_`-private name.
1.0.2's `assert_repl_safe` detects both and fixed neither.

**Both halves are exported, because either alone is a half-fix**: the NAME rule on its own leaves a
well-named tool whose `**kwargs` wrapper `assert_repl_safe` still rejects. `signature_from_json_schema`
is the SHAPE half, factored out of `mcp.py` so there is one derivation. `unique_tool_names` gained
`taken=` for progressive loading, since servers load one at a time and server B's names must avoid
server A's.

What is frozen is the PROPERTIES, the fixpoint, validity and non-collision, **not** the literal output
strings. The reserved set is read from the INSTALLED dspy, so a sanitised name can differ across dspy
versions under an unchanged rlm-harness; do not persist these as long-lived keys.

### `assert_task_repl_safe`

`assert_repl_safe` checks ONE tool. Three of dspy's construction-time rules are properties of the whole
task and each aborts registration for EVERY tool: duplicate tool names, an input field colliding with a
tool name or a reserved sandbox name, and an output field dspy's `Prediction` already owns. Prefer
passing an INSTANCE, since runtime-assembled tools exist only there and those are the sets these rules
bite on. The signature parser mirrors dspy's own two lines rather than calling `dspy.Signature`, which
resolves the output type off the call stack and raises `Unknown name` for a dynamically-built model.

### Fixes

- **`replay.py` served `None` for three of the four shipped tool families.** It read only `result`,
  while MCP and `read_skill` record under `preview`, `web_search` under `results`, and the
  `make_model_tool` convention under `raw`; `dataset.py` already read the fallback, so two readers of
  the same trace disagreed. Now reads `raw → result → results`. `preview` is deliberately excluded
  because it is a TRUNCATED head, so serving it would hand a replay silently-wrong bytes; a
  `preview`-only record raises the same loud `LookupError` the class already uses for drift.
- **`export_actions` dropped `repl_name`**, carried through now, conditionally, so non-MCP records stay
  byte-identical.
- **`mcp.py` could ship a tool its own guard rejects.** The signature stamping was skipped for a
  schema-less tool, leaving the `**kwargs` proxy while the comment above it claimed otherwise. Now
  unconditional. The remaining `**kwargs` case, a property named `from` or `db.query` that cannot be an
  `inspect.Parameter`, is knowingly degraded and records why: sanitising the property name is not an
  option, since the proxy forwards it to the server as a JSON key.

### CI

- **`.github/workflows/dspy-latest.yml`** runs the suite against the NEWEST published dspy, on a weekly
  cron plus `workflow_dispatch` plus push-to-main, never on a PR. This is the job that would have caught
  1.0.1: `ci.yml` resolves dspy from `uv.lock`, so it tests a version nobody installing from PyPI
  necessarily gets, and the whole suite stayed green while the kit was completely unrunnable on a fresh
  install. No `continue-on-error`, which would make the run conclude `success` so a failed scheduled run
  notifies nobody. The newest version is resolved from PyPI at run time and then **asserted** after
  install, because a bare `--with dspy` silently resolves back to the locked version and the job would
  be decorative.

Verified on **both dspy 3.2.1 and 3.3.0**.

## [1.0.2] - 2026-08-06

**Four shipped tool-naming defects.** dspy validates a tool's NAME when `RLM(...)` is constructed: it
must be a Python identifier, must not be a keyword, and must be unique across the task, and a failure
aborts registration for **every** tool, so one bad name silently takes the rest down with it. Four
factories derived a name from data the kit does not control, and all four were broken. No public surface
change, no trace-format break; drop-in.

- **`mcp.py`, the worst of the four.** The external server's own tool name went straight to dspy, and
  hyphens and dots are the MCP naming *norm*: `ValueError: Invalid tool name 'get-weather'`, taking every
  other tool from that server with it. Three identities are now kept separate: `mcp_tool.name` stays the
  WIRE name, `prefix + name` stays the TRACE identity, and only the REPL-facing name is sanitised. Names
  are resolved in ONE pass over the server's full tool list, because uniqueness is a property of the set:
  `get-weather` and `get.weather` both clean to `get_weather`.
- **`sub_lm.py:model_as_tool`** built `f"query_{model_id}"`, so a real id gave
  `query_openai/gpt-4o-mini`, refused outright, and the tool could never be registered.
  `tests/test_repl_safety.py` had masked this by passing `name="m"`.
- **`tools/model.py` + `tools/harness.py`** both returned a closure literally named `call`. Using the two
  together, which the kit's own invariant describes as the expected pattern, registered `['call']` on
  dspy 3.2.x, **silently dropping one tool with no error**, and raised `Duplicate tool name` on 3.3.x.
  Now `call_model` and `call_harness`, the only model-visible rename in the release.
- **`tools/validation.py:make_schema_validator`** built `f"validate_{model.__name__}"`, and a
  `pydantic.create_model("bad-name")` carried the hyphen through. Dynamic output models are exactly what
  `RLMTask.output_model` exists for, so this is reachable.

Added:

- `rlm_harness/_toolname.py` (private): one derivation for all four sites. `sanitize_tool_name`
  guarantees a **fixpoint**, an already-valid name returned unchanged, *including a non-ASCII one*. That
  is the property that matters and the one an obvious `re.sub(r"[^A-Za-z0-9_]", "_", …)` silently
  violates: `str.isidentifier()` accepts non-ASCII letters and so does dspy, so `日本語ツール` and
  `café_search` are valid names that work today and that a character-class sanitiser would rewrite, an
  all-CJK name collapsing to a bare `_`. Character validity is tested with `str.isidentifier()` itself.
  `unique_tool_names` reserves every already-valid name *first*, so a sanitised name can never displace
  one that was already fine.
- `_dspy_compat.reserved_tool_names()`: dspy's reserved sandbox names, probed newest-first and
  **unioned** with a hardcoded floor. Union rather than either-or, so a stale fallback may only
  over-reject, loud and local, rather than under-reject, silent here and a dspy `ValueError` in a
  consumer's rollout.
- `testing.assert_repl_safe` now also checks the NAME, resolving it the way dspy does:
  `getattr(tool, "name", …)` before `func.__name__`. `dspy.Tool(f, name=…)` overrides the function's name
  and dspy validates the *override*, so checking `__name__` alone would pass a tool dspy refuses. Not
  hypothetical: an earlier draft of the `mcp.py` fix sanitised `__name__` only and was a placebo.
- `tool_call` payloads gain an **optional** `repl_name`, emitted only when sanitising changed the name.
  It is needed because the mapping is otherwise unrecoverable offline: the sanitised name depends on the
  server's whole tool list at run time, which never enters the trace. Read it as
  `payload.get("repl_name") or payload["tool"]`.

Verified on **both dspy 3.2.1 and 3.3.0**.

## [1.0.1] - 2026-08-06

**Compatibility fix: the kit did not run at all on `dspy` 3.3.0.** No public surface change, no
trace-format change, so this is a drop-in upgrade for every consumer.

The kit declares `dspy>=3.2.1` and consumers pin the KIT, not dspy. dspy 3.3.0 renamed three things at
once, so any consumer resolving dspy freshly got a kit broken in one loud way and two silent ones:

- **`RLM(interpreter=…)` moved** to the first positional argument of `forward()`/`aforward()`. Every run
  raised `TypeError` before the first LM call, including the default `pyodide` path, since the kit always
  constructs its own sandbox. Total failure, at least loud.
- **`max_iterations` → `max_iters`.** The old best-effort `try` around the budget kwargs was
  all-or-nothing, so one renamed kwarg made dspy reject the call and the fallback dropped **all three**
  caps back to dspy's own defaults. Silent: runs simply ran to a budget nobody configured.
- **`CodeInterpreterError` stopped being the recoverable interpreter error.** 3.3.0 added
  `CodeExecutionError` for that role and made the base class TERMINAL, inverting the meaning of every
  `raise CodeInterpreterError` in the kit. A `sandbox_turn_timeout_s` firing, a safety net whose entire
  point is to hand the model another turn, became a run-ending failure, and in the `container` interpreter
  so did **any exception the model's own code raised in the sandbox**, the most ordinary event in a REPL
  loop. Silent, and only visible under load.

Fixed by resolving each difference through a new private `rlm_harness/_dspy_compat.py` (introspection
plus `lru_cache`, dspy-free at module top), so one build works on 3.2.x and 3.3.x. `task.py:_build_rlm`
picks the interpreter seam and maps the budget caps onto the accepted names, with the `except TypeError`
fallback surviving as a backstop that now `logger.warning`s because it is lossy. `sandbox.py`'s turn
timeout raises `_dspy_compat.recoverable_interpreter_error()` instead of a hardcoded class, while
`SandboxCancelled` is unchanged and still stands outside dspy's hierarchy entirely, which is what keeps
it non-recoverable across the rename. `container_interpreter.py`'s three execute-path raises that were
always meant to be recoverable now use the resolved class, while setup, health and protocol failures keep
the base class and are terminal by intent. `testing.py:ScriptedInterpreter` gained a no-op `start()`,
since from 3.3.0 `CodeInterpreter` is a `@runtime_checkable` Protocol checked before every forward pass.

`tests/test_dspy_compat.py` (new) asserts the shim's contract against whichever dspy is installed, so the
next rename lands as a red test here rather than in a consumer's rollout, and
`tests/test_integration_dspy.py` stopped asserting on dspy internals. Verified on **both dspy 3.2.1
and 3.3.0**.

## [1.0.0] - 2026-08-04

**The first published release.** Nothing was released before it: no tags, no GitHub releases, no PyPI
distribution. The `0.1.0` / `0.2.0` numbers that appeared in `pyproject.toml` and in consumers'
lockfiles never corresponded to a published version, so they are folded into this entry rather than
kept as a fictional release history.

1.0.0 is a promise about the surface, not a claim that development is finished. `__init__.__all__`,
the `rlm-harness/trace/v1` wire format and `RLMTask`'s declaration fields are frozen under
[SemVer](https://semver.org/) and pinned by `tests/test_contract.py`: additions ship in a minor
release, while a rename or removal ships with an alias and a `DeprecationWarning` first and waits for
the next major. **That ends the pre-1.0 habit of hard-renaming and updating consumers in lockstep**,
of which `make_middleware_lm` → `intercept_sub_lm` is the recorded case. The trace format carries its
own version and evolves additive-only within v1. A `_`-prefixed name or module internal is outside the
promise.

### Renamed before first publish: `rlm-kit` → `rlm-harness`

PyPI refused `rlm-kit`: it normalises to the same form as the existing, actively-maintained
[`rlmkit`](https://pypi.org/project/rlmkit/), a state-machine framework for recursive LLM agents. Two
packages one separator apart, both about recursive LM agents, is a genuine confusion surface. Rather
than publish under a name differing from the repository, all three public identities moved together:
the PyPI distribution, the GitHub repository and the import name are `rlm-harness` / `rlm_harness`.

**The trace schema identity moved with it: `rlm-kit/trace/v1` → `rlm-harness/trace/v1`, and the `v1`
is deliberately unchanged.** The shape is byte-identical and only the vendor tag differs, and every
reader in this kit and in all nine consumers keys off `type`, never off `schema`, verified by audit
before the change. A trace written before the rename is still readable by every exporter and by
`replay`. **This is a one-time, pre-publication change made while the format had exactly zero external
readers, and it is not a precedent**: `SCHEMA` is frozen from 1.0.0 onward and a real breaking change
is still a `v2` with a migration. Environment variables were already `RLM_*` and are untouched.

### Added

The surface at 1.0.0, grouped. `rlm_harness/README.md` is the reference for all of it; only the
decisions that are not readable off the API are spelled out here.

- **The harness core.** `RLMTask` as a declaration (`signature`, `output_field`, `output_model`,
  `instructions`, `tools`) with retry, validation, sandbox selection, budget caps and observability
  inherited. `configure(cfg, main_lm=…, sub_lm=…)` plus `get_config` / `get_sub_lm` as the public
  injection seam and accessors. The 8192 `max_tokens` default is a measured floor rather than a guess:
  16 calls at a 16384 cap produced 0 empty completions and 0 length-truncations, against an empty one at
  a 1000 cap, read off each call's `finish_reason`, token counts and `reasoning_len`.
- **The sandbox.** The default `pyodide`/`deno` interpreter, the refused `local` one, and an opt-in
  `interpreter="container"` that runs the REPL inside an isolated Docker container so model code can
  spawn subprocesses. A per-turn execution budget and a real cancellation seam, whose two outcomes
  must never be conflated: a timeout raises dspy's RECOVERABLE interpreter error, a cancel raises
  `SandboxCancelled`, which is deliberately outside dspy's hierarchy so it cannot be retried into
  silence. JSON-literal REPL aliases (`true`/`false`/`null`) because a model writes them.
- **Tracing and datasets.** `TraceRecorder` writing `trace/v1` JSONL, with `record_tool_call` as the
  single emission point, `payload_cause` as its read-side mirror, the `EVENT_*` constants exported,
  live per-turn `main_step` timestamps, an `on_event=` observer, and a failed run recording its
  trajectory too. `replay.reconstruct`, plus `export_sft_turns` / `export_actions` / `run_label_bundle`,
  every one of them reward-free with a `reward=` hook the trainer fills.
- **The sub-LM seat.** `intercept_sub_lm` for deterministic validate/post-process only, with
  `recorder_scope` + `bind_recorder_to_sub_lm` so a batched escalation across dspy's thread pool is
  still recorded, and the escalation INPUT captured on the event.
- **Tools.** `make_model_tool` (with a circuit breaker), `make_fetch_tool` and `make_web_search_tool`
  behind an SSRF guard that includes the resolved-IP check `resolved_host_is_safe` for the direct-fetch
  pattern, `make_command_tool` over a consumer-supplied isolated runner, `make_json_schema_validator`,
  and `ModelToolResult.cause` / `.validator_ran`, because `ok=False` has three causes and they needed
  names. What made that concrete: `rejections` counted every `ok is False` while `circuit_breaks`
  counted the breaks separately, and a break carries `ok=False` too, so one real trace reported
  `calls=3, breaks=7, rejections=10`.
- **Knowledge, not execution.** `load_skills_as_tools` over the Agent-Skills convention, progressive
  disclosure via `list_skills` → `read_skill`, which returns markdown TEXT and never runs a bundled
  script. `read_skill` records a content `preview`.
- **MCP, client-only and sync-bridged.** `mcp_tools`, `McpConnection`, `McpCatalog`, `result_text`.
  The SDK is async and RLM tools must be sync, so the session runs on a dedicated background loop and
  each call bridges one request.
- **Harness delegation, both sides.** `make_harness_tool` as the client and `serve_harness` (plus
  `python -m rlm_harness.harness_serve`) as its server-side mirror, with `bundle_artifact` /
  `parse_artifact_bundle` as the multi-file convention. Trajectory separation is load-bearing: the
  parent records one leaf `tool_call` and a child link, the child owns its own trace.
- **Testing the forward path offline.** `rlm_harness.testing` with `ScriptedInterpreter` and
  `scripted_lm`, plus `assert_repl_safe(tool)` to enforce the "REPL tools expose explicit params"
  invariant, which no amount of care enforces by itself.
- **Reward-free rubric primitives** (`rlm_harness.rubric`) and a GEPA skeleton (`optimize.py`), metric
  templates only.
- **`ClaudeAgentLM`** (`rlm-harness[subscription]`): run the kit on a Claude Pro/Max subscription with
  no API key, through the existing `configure(main_lm=…, sub_lm=…)` seam.

### Fixed

- **`discovery="inject"` pointed `read_skill` at a `list_skills` tool it never registers.**
- **`mcp._make_tool` collapsed every MCP tool's parameters into a single `kwargs`**, because dspy builds
  the in-sandbox proxy from `inspect.signature`, not from `dspy.Tool.args`. It now stamps
  `call.__signature__` from the server's JSON Schema.
- **The co-dev editable overlay shadowed a consumer's namespace `tests/`.**
- **litellm's "Unclosed connector" warning**, and a retry log that flooded the terminal with a
  degenerate LM completion.

### Changed and hardened, surfaced by dogfooding nine downstream consumers

- **`RLMConfig.max_retries` now defaults to `1`, was `3`**: no whole-RLM retry by default, since an
  attempt re-runs the entire trajectory.
- **`McpConnection.close` reaps a WEDGED connect** via a two-phase close, and `McpCatalog(connect="lazy")`
  defers.
- **`export_actions` reads a tool's output through a `raw → result → results → preview` fallback.**
- **`configure()` tolerates a non-owner thread or task.**
- **The extension contract is documented AND guarded.** A consumer extends three ways and only these:
  subclass `RLMTask`, add a tool the base/wrap way, and read results through the trace and exporters.
  It never forks the harness or re-implements tracing. The README carries a single, clearly-delimited
  "Built with rlm-harness" adopters section; everything else stays vendor-neutral.
- **Four prompt and rollout conventions promoted from consumers**: the sub-LM escalation convention,
  run-config-in-`run_start`-meta, the judgement-only SUBMIT recipe, and grounded completeness. All four
  improve rollout QUALITY, which is in scope; none of them is reward.
