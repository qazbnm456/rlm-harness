# Verifying a change to rlm-harness

Read this before pushing, and whenever a change leans on stdlib, platform, or dspy behaviour.
The short form lives in `AGENTS.md`; this file is the CI axes and what each one can and
cannot see.

## Verify

- Run what CI gates on, BOTH jobs, before pushing:
  - `uvx ruff check .`: lint (ruff defaults, line-length 110). CI fails the build on any
    violation; it is NOT part of the pytest suite, so a green `pytest` is not enough on its own.
  - `uv run --group dev --extra mcp --extra grep --extra gitignore python -m pytest -q`: the
    full suite (CI runs it on 3.11/3.12/3.13). `--extra mcp` so the MCP-client tests run instead of
    skipping; `--extra grep` so `make_grep_files_tool`'s timeout tests exercise a REAL `regex`
    timeout instead of skipping (the whole point of that suite is verifying an actual timeout
    fires, not that the code merely imports); `--extra gitignore` so `list_candidate_paths`'s
    `.gitignore`-parsing tests exercise the real `pathspec` package instead of skipping. No live
    LLM, network, or Deno needed: the dspy-bearing tests use `DummyLM` or are skipped if dspy is
    absent.
- **That local `pytest` run is ONE of CI's three interpreter axes. 3.11 is the one worth
  repeating by hand.** `uv run` without `--python` takes the project's default interpreter (3.12
  today; `requires-python` is `>=3.11`), so a stdlib behavior that changed between 3.11 and 3.12 is
  invisible locally and reddens exactly one matrix cell AFTER the push. Not hypothetical:
  `make_extract_archive_tool` let a raw `IndexError` escape its own refusal path because the 3.11
  `ZipInfo.is_dir()` of the day indexed `filename[-1]` where 3.12+ used `endswith("/")`: a green
  local run plus green 3.12/3.13 jobs said nothing about it (CHANGELOG 1.3.0). **That particular
  divergence is gone: CPython backported the `endswith` form into 3.11.x, so `is_dir` behaves
  identically on 3.11, 3.12 and 3.13 today, measured.** The axis is not stale, though, and the live
  example is one function over: `ZipFile.writestr("", b"x")` still raises `IndexError` on 3.11 and
  returns fine on 3.12+, which `tests/test_archive.py`'s fixture builder has to work around. Quote
  THAT one, not `is_dir`, and re-measure before quoting either. So when a change leans on
  stdlib behavior (`zipfile`/`tarfile`, `resource`, `multiprocessing`, `asyncio`), also run the
  suite with `--python 3.11`. The matrix FLOOR is where a "the stdlib does X" assumption breaks
  first. Then pin the lesson in a test that fails on EVERY version (a stub whose accessor raises
  the way the old stdlib does), never one that only reproduces on 3.11.
- **The OS axis a local run cannot see at all.** CI is Linux except for the two legs named below, a
  Windows one and a macOS one; a local run sees neither. `run_in_subprocess`'s
  `max_memory_mb` (`RLIMIT_AS`) genuinely enforces on Linux and is refused outright by the macOS
  kernel, so its edge cases are Linux-only by nature and the relay-starvation one was found by a
  red CI run, not by local testing or by an independent review (CHANGELOG 1.3.0). When the
  PLATFORM is what differs, widen the TEST's accepted outcomes and disclose it; don't bend the
  production behavior toward whichever host you happen to be on.
- **`ci.yml`'s `test-windows` job is the THIRD platform, and it exists because
  `pyproject.toml` declares `Operating System :: OS Independent`.** That claim went unchecked from
  1.0.0 until the job landed, and its first run answered the question it was added to ask: 12
  failures, 1045 passes, and every one of the twelve was a POSIX assumption in a TEST rather than a
  defect in the library. **The axis still produced two library fixes**, found while writing the job
  rather than by running it: `atomic_write_text` translated newlines on Windows, and the first
  attempt to fix its descriptor leak was worse than the leak. Windows has no POSIX mode bits; a backslash is a path separator, so a "literal
  `pkg\util.py`" check contradicts its own neighbour; a daemon that answers `docker info` can still
  refuse every Linux image; and `os.kill(pid, 0)` is not an existence probe there. **That last one is
  worth stating by mechanism, because the obvious mechanism is wrong and the wrong one implies the
  opposite behaviour.** `signal.CTRL_C_EVENT` IS 0, and CPython's Windows `os.kill` branches on
  `sig == CTRL_C_EVENT || sig == CTRL_BREAK_EVENT` FIRST (`Modules/posixmodule.c`), calling
  `GenerateConsoleCtrlEvent(sig, pid)` whose second parameter is a process-GROUP id, not a pid, hence
  `ERROR_INVALID_PARAMETER (87)`. `TerminateProcess` is the fallback for a signal OUTSIDE that set, so
  it is never reached here, and `TerminateProcess(handle, 0)` would have SUCCEEDED and killed the
  child, making the old test pass by destroying its subject. The list above is the classes, not all
  twelve failures.
  **Two lessons worth more than the fixes.** A platform difference exposes a test that was weaker
  than its own docstring: `test_non_ascii_still_reaches_the_file_raw` documented a raw-BYTE check
  and asserted on `read_text()`, which decodes with the platform's preferred encoding and therefore
  could never have distinguished the UTF-8 file it pins from a locale-encoded one. And a diagnosis
  built from an error CODE plus nearby source, rather than from the traceback, invented an
  architecture problem that did not exist: `WinError 87` was read as asyncio refusing to spawn a
  subprocess from `mcp.py`'s background-thread loop, concluding stdio MCP could not work on Windows
  at all. The traceback said the child had already spawned and the raise was the test's own probe.
  **Read the frame, not the code around the error.**
- **`.github/workflows/ci.yml` also has a `packaging` job**: builds the wheel, installs it into a
  clean environment with NO lockfile, and runs a task from it. Every other job runs from the source
  tree via `uv run`, so a module missing from the wheel would ship silently. It is the PACKAGING
  axis only: an offline end-to-end run still returns the right answer while a renamed dspy kwarg
  silently drops the caller's budget cap, so it is blind to exactly the failures `_dspy_compat`
  exists for. Never let it stand in for the job below. It PINS dspy to `uv.lock`'s version on
  purpose: it runs on `pull_request`, and a floating dspy would let an upstream release redden
  a contributor's unrelated PR, which is the very reason the job below is kept off that trigger.
- **`.github/workflows/dspy-latest.yml` runs the same suite against the NEWEST published dspy,
  and since the 1.2.0 floor bump it is the ONLY dspy axis** (the floor and `uv.lock` are both on
  3.4.0 since 1.14.0, so `ci.yml` no longer covers a second version; the workflow says so in a
  `::notice::` and restores real two-version coverage by itself the day a newer dspy ships).
  **It carries a SECOND job on the same triggers, `mcp-latest`**, for the same reason and for the
  dependency that had no such defence: the `mcp` extra is uncapped, `uv.lock` held 1.x, and SDK
  2.0's camelCase→snake_case field rename made every failed MCP tool call read as a SUCCESS to the
  model (CHANGELOG 1.5.0). `ci.yml` additionally carries a PINNED 2.x leg, so a major that is
  already published stays covered on every PR without an upstream release being able to redden one.
  Both jobs live in that separate workflow, on a weekly cron + `workflow_dispatch` +
  push-to-main, never on a PR (an upstream break is not a contributor's problem). It exists because the jobs above
  resolve dspy from `uv.lock`, so they test a version nobody installing from PyPI necessarily gets:
  dspy 3.3.0 renamed three things at once and the whole suite stayed green while the kit was
  completely unrunnable on a fresh install: one break loud, two silent (CHANGELOG 1.0.1). It
  resolves the newest version from PyPI at run time and ASSERTS it actually installed that; a
  hardcoded floor goes stale, and "fail if resolved == locked" false-alarms right after every lock
  bump. `--with "dspy==<exact>"` is what overrides the lock: a bare `--with dspy` (and
  `--isolated --with dspy`) resolve back to the LOCKED version and would make the job decorative.
  **When it goes red, the kit is broken for every fresh install: a release blocker, not a flake.**
  But do NOT read green as "a fresh install works": the overlay upgrades ONLY dspy (plus whatever
  transitive it forces), so everything else stays locked and a break from, say, the newest
  `pydantic` is invisible to it. Reproduce locally with
  `uv run --group dev --extra mcp --extra grep --extra gitignore --with "dspy==<newest>" python -m pytest -q`. It leaves
  `uv.lock` untouched.
- **`.github/workflows/install-check.yml` is the only job that touches the PUBLISHED artifact, and
  the only one that runs on macOS.** Those two properties are the load-bearing part; the roster of
  every other job is deliberately NOT enumerated here, because an enumeration goes stale the day
  someone adds a legitimate job while the properties survive it. Every job that builds anything else
  builds it from this tree (`release.yml`'s `publish` is the exception that proves it: it has no
  checkout, because it uploads what `build` handed it). So two axes had no reader: what PyPI actually
  serves, and a break that is green on Linux and red on macOS. That second direction is not
  symmetric with the `RLIMIT_AS` case ABOVE: that one is Linux-only behaviour a Linux CI caught,
  while nothing here installs the published wheel on a Mac at all, which is what a consumer on that
  platform actually does. It runs weekly, on demand, and CHAINED off `Release` completing: never on
  `release: [published]` itself, the event `release.yml` keys on, which would race the upload and
  fail against an index that has nothing yet. On that chained path it demands the exact version just
  tagged rather than "latest", and FAILS rather than falling back if the tag is missing: a stale CDN
  would otherwise resolve the PREVIOUS release, pass, and report green on an artifact it never
  installed. It carries NO `actions/checkout` on purpose, with no source on the runner,
  "accidentally installed from the tree" is unavailable rather than merely discouraged. Its smoke
  body is byte-identical to `packaging`'s but for one print; if you change one, change both.
  **It is INFORMATIONAL, and a red is fix-forward, never a rollback**: it runs after the upload,
  and PyPI never lets a version be reused, so there is no undo to go looking for; a red means yank
  and ship `X.Y.Z+1`. A red also does not always mean broken-for-everyone: CDN lag past the retry
  window, a PyPI 5xx or a runner hiccup redden a healthy artifact. And green is narrow: it tests
  the RELEASED artifact on ONE interpreter, so a break on `main`, or one at the 3.11 floor, is
  invisible to it.
- **`.githooks/` refuses a commit carrying a private project name, and the denylist is NOT in this
  repo.** The vendor-neutrality rule (`docs/INVARIANTS.md`, "Keep the public surface
  vendor-neutral") held for every FILE and every published release note and
  failed twice in commit MESSAGES: two prose mentions of a private consumer, 128 and 176 commits
  deep, found only because someone thought to look. Removing them cost a history rewrite and a
  force-push of every tag. So it is a machine check now: `pre-commit` scans ADDED lines and staged
  paths, `commit-msg` scans the message, and `check-private-names all` audits the whole tree, every
  commit message and every tag message on demand. Enable it in a fresh clone with **`git config
  core.hooksPath .githooks`**. Hooks are not versioned state, so a clone does not inherit it.
  **The list lives at `~/.claude/private-names.txt` (override with `PRIVATE_NAMES_FILE`), outside
  every repo on purpose: putting the names into a public repo's own checker would publish exactly
  what the checker exists to keep private.** No file means no check: a contributor without one is
  told once and not blocked. Two limits that follow from that design and are not defects: **CI
  cannot run it** (the runner has no list, so it would skip), and `--no-verify` bypasses it. It
  guards the moment the mistake is actually made, which is local.
- A *live* `dspy.RLM` run needs real model credentials **and** a Deno sandbox
  (`brew install deno`, or `pip install "dspy[deno]"`). **Two Deno bounds exist and they are not
  the same number; quote the one that applies.** dspy's RUNTIME gate is
  `MIN_DENO_VERSION = (2, 0, 0)` in `dspy/primitives/python_interpreter.py`, which raises at startup
  outside `>=2.0.0,<3.0.0`; the `dspy[deno]` EXTRA pins the pip-installed `deno` package at
  `>=2.4.5,<3.0.0`. Both are byte-identical on 3.3.1 and 3.4.0: **3.4.0 changed nothing about Deno.**
  This bullet claimed for three releases that 3.4.0 raised the gate to 2.4.5 and broke a working
  Deno 2.1, and it reached a consumer's upgrade advice before anyone measured it. Both numbers were
  real and read off real files; the DELTA between them was invented by putting an extra's pin and a
  runtime constant in one sentence. A version bound is worth nothing without the file it came from.
- **Before writing down a fact about anything OUTSIDE this repo, know which of two kinds it is,
  because only one of them can ever be checked again.** The Deno bound above is the instance this
  rule came from, twice, and it is not the only one: a dspy version a release was verified on, and a
  CPython behaviour, each went into a shipped file and each was false when someone finally ran it.

  An **executable** claim names a version, a symbol, a file path or a behaviour of an installed
  package, so anyone can run it. It is auditable forever, which cuts both ways: it WILL eventually
  be checked, and a reader who checks it and finds it false concludes the rule it justifies is
  obsolete. `ZipInfo.is_dir()` is the worked example: the claim was true when written, CPython
  backported `endswith` into 3.11.x, and the sentence became the only evidence for a CI axis that is
  still entirely justified by a different function. **Re-measure before requoting, and never requote
  from memory when the original text is one command away.**

  A **corpus-attributed** claim ("385 runs", "141 traces", "57% of tool wall-clock") can be audited
  exactly ONCE, at the moment it is written, and never again by anyone. No review of this repo can
  reach the corpus. So its only control is at write time, which is what "Before believing a count"
  in `docs/INVARIANTS.md` already demands: quote the corpus size and the moment, not a bare rate.
  Say this half out loud, or the rule reads as "sweep everything", and half of it cannot be swept.
  Don't run a live run in CI; it costs money. `examples/` show it.
- Before claiming done, actually run the two commands above and paste the output. (The
  newest-dspy workflow is NOT one of them. It needs network, and CI runs it for you.)

