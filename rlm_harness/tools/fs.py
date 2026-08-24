"""``resolve_within_root`` / ``make_read_file_tool`` / ``make_grep_files_tool`` — the filesystem-side
analogue of ``fetch.py``'s SSRF-guarded ``is_safe_url``/``make_fetch_tool``: a safe, scoped, no-shell
way to let a model read or search a bounded local directory tree.

**``root`` is not "a repo" — it is any bounded local directory tree** a consumer scopes it to: a
source repository, a docs corpus, an extracted archive, a dataset directory, a log directory,
whatever the consumer's task needs. Today the only postures ``tools/`` offers for anything
filesystem/execution-shaped are "no access at all" or ``make_command_tool`` (full shell, consumer
supplies the isolation). This fills the gap in between — the single most common thing a
coding-adjacent consumer needs, and one that does NOT require a shell escape hatch: a pure-Python
regex scan over a root-confined, consumer-supplied file list. ``candidate_paths`` stays REQUIRED
(no default directory walk, no built-in ignore-file handling) — the base/wrap split: the kit owns
the safety guard, the consumer decides which files are even candidates, same as ``make_command_tool``
demands an injected ``Runner``.

**A task with more than one bounded root needs distinguishable tool names.** A task that wants
BOTH "read from the source repo" AND "read from the docs corpus" as two distinct tools hits a real
bug if both default to the same name: dspy requires unique tool names across one task, and a second
``make_read_file_tool(other_root)`` in the same ``tools=[...]`` list collides on the shared default
name and aborts registration for EVERY tool on the task, not just the second one. The ``name=``
parameter on both factories exists to let each bounded root get its own tool identity.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence

from ..trace import record_tool_call

_DEFAULT_MAX_RESULTS = 50

# How much of a read/search result is echoed into the trace — bounded because a trace is shipped
# for replay/observability and a whole-file read or a wide grep can return a lot.
_PREVIEW_CHARS = 1200
_PREVIEW_HITS = 8

_OUTPUT_MODES = frozenset({"content", "files_with_matches", "count"})


def _validate_tool_name(name: str) -> None:
    """Fail fast (``ValueError``, at factory-BUILD time) on a ``name=`` a consumer typed
    explicitly, rather than surfacing as an obscure dspy-construction failure later.

    Deliberately a DEVIATION from `sub_lm.py:model_as_tool` / `tools/validation.py:
    make_schema_validator`'s ``sanitize_tool_name`` (silent rewrite): those sanitize UNCONTROLLED
    derived data (a model id, a dynamic pydantic class name) where a best-effort rewrite is the
    right default. ``name=`` here is explicit, consumer-typed input — silently mangling a name the
    caller chose on purpose would be more surprising than refusing it outright.

    Checks BOTH halves dspy validates at ``RLM(...)`` construction: a valid Python identifier
    (not a keyword), and not one of dspy's reserved sandbox names. Both imports are lazy — this
    file, like the rest of ``tools/``, stays dspy-free at module top.
    """
    from .._toolname import is_valid_tool_name

    if not is_valid_tool_name(name):
        raise ValueError(
            f"{name!r} is not a valid tool name: must be a Python identifier and not a keyword."
        )
    from .._dspy_compat import reserved_tool_names

    if name in reserved_tool_names():
        raise ValueError(
            f"{name!r} is reserved by dspy's sandbox and cannot be used as a tool name."
        )


def resolve_within_root(root: str, path: str) -> str | None:
    """Resolve ``path`` (relative to ``root``) to a real, absolute path, or ``None`` if it escapes
    ``root`` via ``..``, an absolute path elsewhere, or a symlink pointing outside.

    **Must use ``os.path.realpath`` (which follows symlinks), never ``os.path.normpath`` (purely
    lexical) — this is the one thing that makes the check actually defeat a symlink escape.** A
    symlink INSIDE ``root`` pointing OUTSIDE it resolves, via ``realpath``, to its real target
    before the containment check runs; a ``normpath``-only check would never see past the symlink
    and would wrongly allow it. Do not "simplify" this to ``normpath`` — that would silently reopen
    exactly the escape this guard exists to close.
    """
    root_real = os.path.realpath(root)
    candidate = os.path.realpath(os.path.join(root, path))
    if os.path.commonpath([root_real, candidate]) != root_real:
        return None
    return candidate


def make_read_file_tool(
    root: str,
    *,
    name: str = "read_file",
    encoding: str = "utf-8",
    max_output_chars: int | None = None,
    line_numbers: bool = False,
) -> Callable[..., str]:
    """Build a ``read_file``-shaped tool scoped to ``root`` — wired in a task's ``__init__``
    (per-run state, never a classvar).

    ``name`` (default ``"read_file"``): the REPL-facing tool identity and the trace's ``tool``
    field — override it to give a second bounded root (a docs corpus alongside a source repo, say)
    a distinct name in the same task's ``tools=[...]`` list; see the module docstring. Validated at
    factory-build time (identifier + not reserved) — ``rlm_harness.testing.assert_repl_safe``
    remains the recommended proactive per-tool check, but is no longer the only line of defense.

    ``encoding`` (default ``"utf-8"``): point this at a non-UTF-8 corpus if needed.

    ``max_output_chars`` (default ``None`` = unlimited): truncates the returned text at that
    length with a VISIBLE marker appended — never a silent shortening. Scoped to the
    successful-read branch only: the ``Refused``/``Read error`` strings below are never truncated.

    ``line_numbers`` (default ``False``): prefix each returned line with its REAL 1-indexed file
    line number, so a model reading a slice starting mid-file doesn't have to compute one itself
    from ``start_line`` — removing exactly the kind of off-by-one a model gets wrong when later
    asked to cite or edit that line. Also scoped to the successful-read branch only.
    """
    _validate_tool_name(name)

    def read_file(path: str, start_line: int = 1, end_line: int | None = None) -> str:
        """Read lines ``start_line``..``end_line`` (1-indexed, inclusive; ``end_line=None`` means
        "to the end of the file") of a file at ``path``, relative to the root. Returns a
        "Refused: ..." string (never raises) for a path that escapes the root, or a short error
        string for a missing/unreadable/directory path."""
        resolved = resolve_within_root(root, path)
        if resolved is None:
            record_tool_call(
                name, args={"path": path}, ok=False, note="refused: escapes root"
            )
            return f"Refused: {path!r} is not a path inside this root."
        try:
            with open(resolved, encoding=encoding) as fh:
                lines = fh.readlines()
        except (OSError, UnicodeDecodeError) as exc:
            # IsADirectoryError is an OSError subclass — a directory-shaped `path` degrades the
            # same way a missing/unreadable file does, never a raised, unhandled exception.
            record_tool_call(
                name, args={"path": path}, ok=False, note=f"error: {type(exc).__name__}"
            )
            return f"Read error for {path!r}: {type(exc).__name__}"

        # `lo` is the CLAMPED start index — read_file already tolerates start_line <= 0 (that is
        # what max(0, ...) is FOR). Any line-number prefix must derive from `lo`, never from the
        # raw `start_line` parameter directly: for start_line <= 0, lo == 0 always, so the real
        # first line number is 1 — `start_line + offset` would be off by one (or more, for a
        # negative start_line) in exactly the inputs this clamp exists to tolerate.
        lo = max(0, start_line - 1)
        hi = len(lines) if end_line is None else min(len(lines), end_line)
        selected = lines[lo:hi]

        if line_numbers:
            result = "".join(
                f"{lo + 1 + offset:>6}\t{line}" for offset, line in enumerate(selected)
            )
        else:
            result = "".join(selected)

        # Truncation runs AFTER line-numbering, on the string the model actually receives — the
        # only order under which the char budget reflects real output (numbering prefixes count
        # against the cap) and avoids numbering an already-truncated fragment inconsistently.
        truncated = max_output_chars is not None and len(result) > max_output_chars
        if truncated:
            result = (
                result[:max_output_chars]
                + f"\n... [truncated at {max_output_chars} chars — narrow "
                "start_line/end_line to read more]"
            )

        record_tool_call(
            name,
            args={"path": path, "start_line": start_line, "end_line": end_line},
            ok=True,
            result_len=len(result),
            preview=result[:_PREVIEW_CHARS] or None,
            truncated=truncated or None,
        )
        return result

    read_file.__name__ = name
    read_file.__qualname__ = name
    return read_file


def make_grep_files_tool(
    root: str,
    candidate_paths: Sequence[str],
    *,
    name: str = "grep_files",
    per_match_timeout_s: float = 1.0,
    max_total_time_s: float = 30.0,
) -> Callable[..., str]:
    """Build a ``grep_files``-shaped tool scoped to ``root``, searching only ``candidate_paths``
    (typically a consumer-computed file list) — per-run state, wired in a task's ``__init__``.

    ``name`` (default ``"grep_files"``): same rationale and mechanism as
    :func:`make_read_file_tool`'s ``name`` — lets a second bounded root coexist in one task's
    ``tools=[...]`` list without a duplicate-name collision.

    **Requires the optional ``regex`` package** (``pip install "rlm-harness[grep]"``) — raises a
    friendly ``ImportError`` at factory-BUILD time if it's missing, with NO silent fallback to
    stdlib ``re``. This is deliberate, not a convenience gap: ``pattern`` is LM-controlled,
    unbounded regex, matched against real file lines with no wall-clock budget anywhere else in
    this kit's tool-call path — and stdlib ``re`` cannot be bounded by ANY pure-Python mechanism,
    including ``signal.alarm`` (CPython's ``re`` engine does not yield to the signal dispatcher
    mid-match; one ``re.search()`` call is a single, uninterruptible C-level operation from the
    interpreter's point of view). A catastrophic-backtracking pattern (e.g. ``(a+)+$`` against a
    non-matching line) can hang the host process indefinitely on stdlib ``re``. ``regex`` is
    different: its own matching loop periodically checks elapsed wall-clock time internally and
    raises ``TimeoutError`` when exceeded — a real, working, pattern-structure-agnostic mechanism.
    This mirrors ``make_json_schema_validator``'s existing posture for its own optional
    ``jsonschema`` extra: no silently-weaker substitute mode, because a pattern-matching tool whose
    LM-controlled pattern isn't actually wall-clock-bounded is worse than a tool that flatly
    refuses to run unbound.

    ``per_match_timeout_s`` (default ``1.0``) bounds ONE line's match; a ``TimeoutError`` skips
    just that line (counted, surfaced in the result, never silent) and the scan continues.
    ``max_total_time_s`` (default ``30.0``) bounds the WHOLE call — checked before EVERY line's
    match (not merely once per file: a per-file-only check would let a single large file with many
    timeout-tripping lines blow past this budget by an arbitrary multiple before the check ever
    fires again). Both are factory (operator/deployment) parameters, never model-controlled,
    matching ``make_command_tool``'s factory-level-not-call-level configuration precedent.
    """
    _validate_tool_name(name)

    try:
        import regex
    except ImportError as exc:
        raise ImportError(
            "make_grep_files_tool needs the optional 'regex' package for a wall-clock-bounded "
            "match (stdlib `re` has no way to bound catastrophic-backtracking cost — not even "
            "via signal.alarm). Install it with:  pip install \"rlm-harness[grep]\""
        ) from exc

    def grep_files(
        pattern: str,
        glob: str = "*",
        max_results: int = _DEFAULT_MAX_RESULTS,
        output_mode: str = "content",
        ignore_case: bool = False,
    ) -> str:
        """Search for a regex ``pattern`` across files matching ``glob`` (an ``fnmatch``-style
        glob against each file's path, default ``"*"`` = every file).

        ``output_mode`` (default ``"content"``): ``"content"`` returns up to ``max_results``
        matching lines as ``path:line: text``, one per line — the original, only behavior.
        ``"files_with_matches"`` returns up to ``max_results`` DISTINCT file paths that had ≥1
        match, one per line, no line text. ``"count"`` returns up to ``max_results`` ``path: N``
        lines (N = matching-line count in that file); a file with zero matches is omitted.
        All three modes scan every line of every candidate file IDENTICALLY (this round adds no
        per-file early-break) — the two non-``"content"`` modes are cheaper in OUTPUT/TRACE SIZE
        only, never in scan cost. An unrecognized ``output_mode`` returns an
        ``"Invalid output_mode ..."`` error string, never raises.

        ``max_results`` caps the number of OUTPUT ROWS returned (a row = a matching line in
        ``"content"`` mode, a file in the other two modes). In ``"content"`` mode a row is
        complete the instant a match is found, so the cap is checked per line, exactly as before.
        In ``"files_with_matches"``/``"count"`` mode a row (one file's aggregated result) is only
        complete once that file's entire line loop finishes, so the cap is checked once per
        COMPLETED file — the same "cap on output rows" contract, just checked at the granularity
        where a row actually becomes final.

        ``ignore_case`` (default ``False``): case-insensitive matching as a first-class flag,
        rather than something the caller has to bake into the pattern itself.

        A per-line match that exceeds the configured timeout is skipped (never raises, in any
        mode); the whole call is additionally bounded by a wall-clock budget, after which it
        returns whatever partial results it found so far, in whichever mode was requested."""
        import fnmatch

        if output_mode not in _OUTPUT_MODES:
            record_tool_call(
                name, args={"pattern": pattern, "output_mode": output_mode}, ok=False,
                note="invalid output_mode",
            )
            return f"Invalid output_mode {output_mode!r} (expected one of {sorted(_OUTPUT_MODES)})."

        try:
            compiled = regex.compile(pattern, flags=regex.IGNORECASE if ignore_case else 0)
        except regex.error as exc:
            record_tool_call(name, args={"pattern": pattern}, ok=False, note=str(exc))
            return f"Invalid regex {pattern!r}: {exc}"

        started = time.monotonic()
        # `hits` (content mode) is one entry per matching LINE, capped at max_results directly.
        # `file_matches` (the other two modes) accumulates one entry per COMPLETED file that had
        # >=1 match; its own max_results cap is applied after the scan, since a file's row isn't
        # final until its whole line loop finishes (no early-break this round).
        hits: list[str] = []
        file_matches: list[tuple[str, int]] = []  # (rel_path, match_count), files with >=1 match
        timed_out_lines = 0
        budget_exceeded = False

        for rel_path in candidate_paths:
            if budget_exceeded:
                break
            if output_mode == "content" and len(hits) >= max_results:
                break
            if not fnmatch.fnmatch(rel_path, glob):
                continue
            resolved = resolve_within_root(root, rel_path)
            if resolved is None:
                continue
            file_match_count = 0
            try:
                with open(resolved, encoding="utf-8") as fh:
                    for lineno, line in enumerate(fh, start=1):
                        # Checked before EVERY line's match, in every output_mode identically —
                        # not just once per file: a per-file-only check would let a single large
                        # file with many timeout-tripping lines blow past max_total_time_s by an
                        # arbitrary multiple before the check ever fired again.
                        if time.monotonic() - started > max_total_time_s:
                            budget_exceeded = True
                            break
                        try:
                            matched = compiled.search(line, timeout=per_match_timeout_s)
                        except TimeoutError:
                            timed_out_lines += 1
                            continue
                        if not matched:
                            continue
                        file_match_count += 1
                        if output_mode == "content":
                            hits.append(f"{rel_path}:{lineno}: {line.rstrip(os.linesep)}")
                            if len(hits) >= max_results:
                                break
            except (OSError, UnicodeDecodeError):
                continue
            if file_match_count > 0:
                file_matches.append((rel_path, file_match_count))

        if output_mode != "content":
            file_matches = file_matches[:max_results]
            if output_mode == "files_with_matches":
                hits = [rel_path for rel_path, _ in file_matches]
            else:  # "count"
                hits = [f"{rel_path}: {count}" for rel_path, count in file_matches]

        record_tool_call(
            name,
            args={"pattern": pattern, "glob": glob, "output_mode": output_mode},
            ok=True,
            result_count=len(hits),
            preview="\n".join(hits[:_PREVIEW_HITS])[:_PREVIEW_CHARS] or None,
            timed_out_lines=timed_out_lines or None,
        )
        if not hits:
            suffix = (
                f" ({timed_out_lines} line(s) skipped: match exceeded {per_match_timeout_s}s)"
                if timed_out_lines
                else ""
            )
            return f"No matches for {pattern!r} (glob {glob!r}).{suffix}"
        return "\n".join(hits)

    grep_files.__name__ = name
    grep_files.__qualname__ = name
    return grep_files
