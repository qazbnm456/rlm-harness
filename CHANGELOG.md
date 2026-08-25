# Changelog

All notable changes to `rlm-harness`. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/). Versions track
`rlm_harness/__init__.__version__` and `pyproject.toml` (kept in sync).

## [1.3.0] - 2026-08-24

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
  text (every character either `re.escape()`d or collapsed to a flat, non-nested `\s+`), so the
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
