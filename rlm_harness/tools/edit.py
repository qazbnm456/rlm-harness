"""``make_write_file_tool`` / ``make_edit_file_tool`` — the write side of the filesystem tools,
sitting alongside ``fs.py``'s read side (``make_read_file_tool`` / ``make_grep_files_tool``).

Kept in a SEPARATE module from ``fs.py`` (which is already the largest single file in `tools/` —
355 lines, ~1.6× the next-largest single file, ``command.py`` at 221 lines) so that "everything in
this package that can mutate the filesystem" stays physically distinct from "everything that only
reads it" — a real audit benefit, since this module introduces a genuinely new risk category the
read-only tools don't carry: DATA LOSS. `read_file`/`grep_files` returning wrong information is a
bug; `write_file`/`edit_file` doing the wrong thing can destroy content. That's the one dimension
of risk this module adds — it does NOT add a new SECURITY-boundary category: every tool in this
kit already executes host-side outside the sandbox (true of `fetch_url`, `run_command`, and
`fs.py`'s own read-side pair), and writing a file host-side inside an already-`resolve_within_root`
-guarded root is the natural write-side mirror of `read_file` reading host-side inside the same
guard.

Both factories reuse `fs.py`'s `resolve_within_root` guard and `_validate_tool_name` helper
(the first same-package private-name cross-import in this package — a deliberate, new internal
seam, not an existing pattern being followed; justified on its own terms, since the "consumers
don't reach into private names" rule is about code OUTSIDE this package, not sibling modules the
kit itself maintains). Both build on `atomic_write_text` (`rlm_harness.atomic`), which is what
makes an overwrite/edit crash-safe — the write itself is either fully visible or not visible at
all, never half-written.

**Known, accepted, out-of-scope-for-this-round risk**: `atomic_write_text`'s guarantee is "no torn
read" — it does NOT serialize a read-modify-write across two SEPARATE `RLMTask` runs (or two
workers in a batch eval) sharing the same `root`. Two concurrent `edit_file` calls on the same file
can both read the same original content, compute independent replacements, and the second
`os.replace` silently wins — a lost update, invisible to either caller (both report success). This
is a genuinely new hazard the read-only tools next to these are immune to; it is not addressed
here.
"""

from __future__ import annotations

from collections.abc import Callable

from ..atomic import atomic_write_text
from ..trace import record_tool_call
from .fs import _validate_tool_name, resolve_within_root

_PREVIEW_CHARS = 1200


def _format_edit_snippet(
    lines: list[str], start_line: int, end_line: int, context_lines: int
) -> str:
    """Render a numbered window (``read_file``'s own ``f"{lineno:>6}\\t{line}"`` convention)
    around ``lines[start_line-1:end_line]`` (1-indexed, inclusive), padded with up to
    ``context_lines`` unchanged lines on each side, clamped to ``[1, len(lines)]``.

    Sharing that convention makes this the SECOND source of gutter-bearing text a model can copy
    into a citation — ``read_file(line_numbers=True)`` is the documented one, and this reaches the
    model on every successful edit. :func:`~rlm_harness.tools.grounding.verify_quote` must
    therefore be given the raw file text here too. Since 1.8.2 it refuses a quote carrying only a
    line number (what a BLANK line in this window renders as); since 1.9.0 it resolves one carrying
    CONTENT against the line the gutter names, rather than searching the gutter's digits as text.

    If the edited region ITSELF spans more than ``2 * context_lines + 1`` lines, only its own
    head and tail (each ``context_lines`` long) are shown, with a visible ``"... N line(s)
    omitted ..."`` marker between them — bounds the rendered size to a small, fixed multiple of
    ``context_lines`` regardless of how large the edit was.
    """
    n = len(lines)
    lo = max(1, start_line - context_lines)
    hi = min(n, end_line + context_lines)

    def numbered(lineno: int) -> str:
        return f"{lineno:>6}\t{lines[lineno - 1]}"

    region_span = end_line - start_line + 1
    if region_span > 2 * context_lines + 1:
        head_hi = start_line + context_lines - 1
        tail_lo = end_line - context_lines + 1
        omitted = tail_lo - head_hi - 1
        rendered = [numbered(i) for i in range(lo, head_hi + 1)]
        rendered.append(f"       ... {omitted} line(s) omitted ...\n")
        rendered.extend(numbered(i) for i in range(tail_lo, hi + 1))
    else:
        rendered = [numbered(i) for i in range(lo, hi + 1)]
    return "".join(rendered)


def make_write_file_tool(
    root: str,
    *,
    name: str = "write_file",
    encoding: str = "utf-8",
) -> Callable[..., str]:
    """Build a ``write_file``-shaped tool scoped to ``root`` — wired in a task's ``__init__``
    (per-run state, never a classvar).

    ``name`` (default ``"write_file"``): same rationale and mechanism as
    :func:`rlm_harness.tools.make_read_file_tool`'s ``name`` — lets a second bounded root coexist
    in one task's ``tools=[...]`` list without a duplicate-tool-name collision at dspy's
    ``RLM(...)`` construction. Validated at factory-build time (identifier + not reserved).

    ``encoding`` (default ``"utf-8"``): point this at a non-UTF-8 corpus if needed.

    **Unconditional overwrite** — there is no "refuse if the file already exists" mode. A consumer
    wanting a create-only guarantee can call the sibling ``read_file`` tool first and check for its
    "missing file" error string. Kept deliberately minimal for v1 rather than adding an
    untested create-only/overwrite-only flag nobody has asked for yet.

    **No ``max_content_chars`` cap.** Unlike ``read_file``'s ``max_output_chars`` (which protects
    the MODEL's own context budget against an unexpectedly huge file it didn't write), the content
    here was generated by the model itself as part of its own output, so it's already bounded by
    whatever generated it — capping it further protects nothing new AGAINST A SINGLE CALL. This
    does NOT cover disk exhaustion from a model in a loop calling this repeatedly: many
    individually-bounded files can still fill the host's disk. Accepted as a known,
    out-of-scope-for-v1 gap, not mitigated this round.
    """
    _validate_tool_name(name)

    def write_file(path: str, content: str) -> str:
        """Create or overwrite the file at ``path`` (relative to the root) with ``content``,
        atomically. Returns a short confirmation, or a "Refused"/error string (never raises) for a
        path that escapes the root or can't be written."""
        resolved = resolve_within_root(root, path)
        if resolved is None:
            record_tool_call(
                name, args={"path": path}, ok=False, note="refused: escapes root"
            )
            return f"Refused: {path!r} is not a path inside this root."
        try:
            atomic_write_text(resolved, content, encoding=encoding)
        except OSError as exc:
            record_tool_call(
                name, args={"path": path}, ok=False, note=f"error: {type(exc).__name__}"
            )
            return f"Write error for {path!r}: {type(exc).__name__}"
        record_tool_call(name, args={"path": path}, ok=True, content_len=len(content))
        return f"Wrote {len(content)} char(s) to {path!r}."

    write_file.__name__ = name
    write_file.__qualname__ = name
    return write_file


def make_edit_file_tool(
    root: str,
    *,
    name: str = "edit_file",
    encoding: str = "utf-8",
    show_snippet: bool = True,
    snippet_context_lines: int = 3,
    max_snippet_occurrences: int = 3,
) -> Callable[..., str]:
    """Build an ``edit_file``-shaped tool scoped to ``root`` — wired in a task's ``__init__``
    (per-run state, never a classvar).

    ``name`` (default ``"edit_file"``): same rationale and mechanism as
    :func:`rlm_harness.tools.make_read_file_tool`'s ``name``.

    ``encoding`` (default ``"utf-8"``): point this at a non-UTF-8 corpus if needed.

    **Known failure mode, stated explicitly**: if the file on disk uses different line endings
    than the ``old_string`` the model supplies (e.g. a ``\\r\\n``-normalized file matched against a
    bare-``\\n`` anchor), the match fails CLOSED with "not found" rather than mis-editing. Not a
    safety bug — it never silently edits the wrong thing — but worth knowing so an unexpected
    refusal on an otherwise-correct anchor isn't a surprise.

    **On success, a windowed snippet of the RESULT is appended** (reusing ``read_file``'s own
    ``f"{lineno:>6}\\t{line}"`` numbering convention) so the model can confirm what its edit
    actually did without a separate ``read_file`` round-trip. ``show_snippet`` (default ``True``)
    is the escape hatch back to the terse ``"Replaced N occurrence(s) in {path!r}."`` alone.
    ``snippet_context_lines`` (default ``3``) bounds each shown region to a small window around it
    — if the edited region itself spans more, only its own head/tail are shown with an "... N
    line(s) omitted ..." marker, so a huge insertion never dumps an unbounded block back. With
    ``replace_all=True`` producing many replaced occurrences, ``max_snippet_occurrences`` (default
    ``3``) caps how many get their own snippet — the FILE is still fully edited regardless; this
    only caps how much of it is echoed back, and the returned string says explicitly when some
    were omitted. Scoped to the SUCCESS path only — ``Refused``/``Read error``/``Write error``
    strings are never appended to. Overlapping windows for closely-spaced occurrences are shown
    independently, not merged — the edited regions themselves never overlap, only their
    surrounding context can.
    """
    _validate_tool_name(name)

    def edit_file(
        path: str, old_string: str, new_string: str, replace_all: bool = False
    ) -> str:
        """Replace ``old_string`` with ``new_string`` in the file at ``path`` (relative to the
        root). Refuses (returns a string, never raises) if ``old_string`` is not found, or is
        found more than once and ``replace_all`` is False — supply more surrounding context to
        make it unique, or pass ``replace_all=True`` to replace every occurrence. ``old_string``
        and ``new_string`` must differ, and ``old_string`` must be non-empty."""
        resolved = resolve_within_root(root, path)
        if resolved is None:
            record_tool_call(
                name, args={"path": path}, ok=False, note="refused: escapes root"
            )
            return f"Refused: {path!r} is not a path inside this root."
        try:
            with open(resolved, encoding=encoding) as fh:
                content = fh.read()
        except (OSError, UnicodeDecodeError) as exc:
            # IsADirectoryError is an OSError subclass — a directory-shaped `path` degrades the
            # same way a missing/unreadable file does, never a raised, unhandled exception.
            record_tool_call(
                name, args={"path": path}, ok=False, note=f"error: {type(exc).__name__}"
            )
            return f"Read error for {path!r}: {type(exc).__name__}"

        if old_string == "":
            record_tool_call(
                name, args={"path": path}, ok=False, note="refused: empty old_string"
            )
            return "Refused: old_string must be non-empty (an empty anchor is ambiguous)."
        if old_string == new_string:
            record_tool_call(
                name, args={"path": path}, ok=False, note="refused: old_string == new_string"
            )
            return "Refused: old_string and new_string are identical — nothing to edit."

        occurrences = content.count(old_string)
        if occurrences == 0:
            record_tool_call(
                name, args={"path": path}, ok=False, note="old_string not found"
            )
            return f"Refused: old_string not found in {path!r}."
        if occurrences > 1 and not replace_all:
            record_tool_call(
                name, args={"path": path}, ok=False, occurrences=occurrences,
                note="ambiguous: multiple occurrences",
            )
            return (
                f"Refused: old_string appears {occurrences} times in {path!r} — supply more "
                f"surrounding context to make it unique, or pass replace_all=True to replace "
                f"every occurrence."
            )

        # Collected BEFORE the replace, using the same non-overlapping stride str.count()/
        # str.replace() both use (advance by len(old_string), not by 1) -- offsets computed with
        # any other stride would desynchronize from what .replace() actually does for a
        # self-overlapping old_string (e.g. "aa" inside "aaaa").
        replaced = occurrences if replace_all else 1
        shown = min(replaced, max_snippet_occurrences) if show_snippet else 0
        old_offsets: list[int] = []
        if shown:
            pos = 0
            for _ in range(shown):
                pos = content.find(old_string, pos)
                old_offsets.append(pos)
                pos += len(old_string)

        count = -1 if replace_all else 1
        new_content = content.replace(old_string, new_string, count)
        try:
            atomic_write_text(resolved, new_content, encoding=encoding)
        except OSError as exc:
            record_tool_call(
                name, args={"path": path}, ok=False, note=f"error: {type(exc).__name__}"
            )
            return f"Write error for {path!r}: {type(exc).__name__}"

        record_tool_call(
            name,
            args={"path": path, "replace_all": replace_all},
            ok=True,
            occurrences_replaced=replaced,
            old_preview=old_string[:_PREVIEW_CHARS],
            new_preview=new_string[:_PREVIEW_CHARS],
        )
        summary = (
            f"Replaced 1 occurrence in {path!r}."
            if replaced == 1
            else f"Replaced {replaced} occurrences in {path!r}."
        )
        if not show_snippet:
            return summary

        # k is 0-indexed: 0 prior replacements for the FIRST occurrence, matching how many of the
        # earlier occurrences (strictly before this one) have already shifted everything after
        # them by `delta` characters.
        delta = len(new_string) - len(old_string)
        new_lines = new_content.splitlines(keepends=True)
        blocks = []
        for k, old_offset in enumerate(old_offsets):
            new_start = old_offset + k * delta
            start_line = new_content.count("\n", 0, new_start) + 1
            if new_string:
                # The END line derives from the LAST character actually inside new_string
                # (new_start + len(new_string) - 1), never from the one-past-the-end offset --
                # using the one-past-the-end offset directly overcounts by one line whenever
                # new_string ends with "\n" (it would land on the start of the next, untouched
                # line and wrongly include it in the region).
                end_line = new_content.count("\n", 0, new_start + len(new_string) - 1) + 1
            else:
                end_line = start_line  # a deletion: zero-length region, nothing to convert
            blocks.append(
                (start_line, end_line, _format_edit_snippet(
                    new_lines, start_line, end_line, snippet_context_lines
                ))
            )

        parts = [summary, ""]
        for i, (start_line, end_line, text) in enumerate(blocks, start=1):
            parts.append(f"--- snippet {i}/{shown} (lines {start_line}-{end_line}) ---\n{text}")
        if replaced > shown:
            parts.append(f"... {replaced - shown} more occurrence(s) not shown.")
        return "\n".join(parts)

    edit_file.__name__ = name
    edit_file.__qualname__ = name
    return edit_file
