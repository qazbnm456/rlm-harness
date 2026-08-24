"""``verify_quote`` — deterministic host-side quote/citation grounding.

Closes a real gap this kit's own guide already names. `README.md`'s **"Grounded completeness —
the sufficiency-critic recipe"** states the problem outright: *"There is often no deterministic
check for CONTENT correctness — a validator catches structure/format, but not 'this answer
skipped a clause'."* Its step 2, *"Diff the artifact against it, itemized,"* is entirely
MODEL-JUDGED today — the main LM compares its own draft to a held ground-truth from memory or a
re-read, with nothing deterministic backing that judgment. This is exactly the failure mode the
neighboring **"Judgement-only SUBMIT"** section already warns about for a different seam ("a
self-reported validity flag becomes a label that can LIE") — except here it's the completeness
DIFF itself that's self-reported, with no tool standing behind it the way ``make_schema_validator``
stands behind structural validation.

``verify_quote`` is the deterministic primitive for the one thing that IS checkable without a
domain-specific parser: does this specific claimed quote/citation actually appear (verbatim, or
under whitespace normalization) in the held source text? It does not replace the model-judged
itemized diff — it gives that diff a deterministic building block a model can call BEFORE
finalizing (closing the loop in-trajectory, matching the recipe's "regenerate on the gaps" step),
and that a consumer can ALSO call AFTER the fact, host-side, to re-derive whether a SUBMITted
citation was actually grounded — the same "derive facts from bytes, never trust the self-report"
posture "Judgement-only SUBMIT" already establishes for validity.

**A single plain function, no factory, no trace call** — a deliberate difference from `fs.py`/
`edit.py`: those bind to a `root` (a security boundary) at construction time, so a factory and a
``name=`` override exist to give each bound instance its own REPL identity. ``verify_quote`` binds
to nothing — ``source``/``quote`` are call-time arguments — so there's no state to close over and
no multi-instance collision to solve, matching ``make_schema_validator``/
``make_json_schema_validator``'s own precedent of neither taking a `name=` nor calling
``record_tool_call``: a pure function with no filesystem/network side effect has nothing for a
host-side audit trail to capture beyond what the RLM's own trajectory already records as this
function's return value.
"""

from __future__ import annotations

import difflib
import re

_DEFAULT_SNIPPET_CHARS = 120
_CLOSE_MATCH_CUTOFF = 0.4


def verify_quote(
    source: str,
    quote: str,
    normalize_whitespace: bool = True,
    snippet_chars: int = _DEFAULT_SNIPPET_CHARS,
) -> str:
    """Verify that ``quote`` appears in ``source`` — the deterministic half of the
    Grounded-completeness recipe's itemized diff. Returns a ``"MATCH: ..."``/``"MISMATCH: ..."``
    string (never raises), a deliberately parseable prefix mirroring this kit's existing sentinel
    strings (``"Refused:"``, ``"Invalid regex:"``) — a REPL-side model branches on it directly,
    and a consumer reusing this function host-side (at assembly/export time) can do the same
    check against a recorded tool-call's return value with no separate parsing.

    **Matching mechanic — a whitespace-flexible LITERAL search**, not a normalize-then-search
    (which would lose position info this doesn't need to lose): ``quote`` is split on whitespace
    runs, each literal chunk is ``re.escape()``d (any metacharacter it contains — ``.``, ``(``,
    ``$`` — is matched as a literal, never as regex syntax) and each whitespace run becomes
    ``\\s+`` (when ``normalize_whitespace=True``, the default) or its own escaped self (when
    ``False``, for a caller that wants byte-exact whitespace too). The resulting pattern runs
    directly against the ORIGINAL, un-normalized ``source`` — a real match has a real ``.start()``
    offset in ``source``, no position-remapping needed.

    **This needs no ``regex`` package / no timeout budget**, unlike ``make_grep_files_tool``'s
    ``pattern``: that tool's catastrophic-backtracking risk exists because the model supplies the
    PATTERN'S STRUCTURE directly. Here, the model supplies ``quote`` — literal text, not regex
    syntax — and every character of it is either escaped or collapsed to a flat, non-nested
    ``\\s+``. The resulting pattern can never contain nested/adjacent quantifiers or overlapping
    alternation — the shapes that exhibit catastrophic backtracking — so stdlib ``re`` is provably
    safe here regardless of what ``quote`` contains.

    On a match, the return includes the 1-indexed line number and a ``snippet_chars``-wide window
    of ``source`` centered on the match, clamped to the string's bounds — enough to sanity-check
    the RIGHT occurrence was found if ``quote`` is generic enough to match more than one place
    (this reports the FIRST match only, via ``re.search`` — a presence check, not an enumeration;
    a model wanting every occurrence in a FILE already has ``grep_files`` for that).

    On no match, a cheap, bounded fallback tries one "closest line" diagnostic — only when
    ``quote`` (stripped) is single-line — via ``difflib.get_close_matches`` against
    ``source.splitlines()``. This is polynomial, not exponential (no catastrophic-blowup risk),
    but it is NOT as cheap per-line as a single regex match; acceptable only because it runs
    solely on the rare mismatch path, over one in-memory string. A multi-line ``quote`` or no
    candidate clearing the similarity cutoff skips the hint entirely rather than forcing a
    low-quality one.

    A ``quote`` that is empty OR whitespace-only is refused outright (before any search): a
    whitespace-only ``quote`` would otherwise reduce to the bare pattern ``\\s+``, which trivially
    matches almost any real text — a meaningless confirmation, not a real check. Same reasoning
    ``edit_file`` already applies to refusing an empty ``old_string``.

    **Leading/trailing whitespace on ``quote`` is stripped before matching** — it's incidental
    padding, not a claim about what precedes/follows the quoted text in ``source``. Without this,
    it would turn into a MANDATORY ``\\s+`` at the pattern's own edges, wrongly requiring
    ``source`` to also have whitespace immediately before/after the quoted content (a real bug
    caught by direct testing: ``verify_quote("the answer is 42", "  the answer is 42  ")`` would
    otherwise spuriously ``MISMATCH`` even though the content is an exact match).
    """
    stripped_quote = quote.strip()
    if not stripped_quote:
        return "MISMATCH: quote must be non-empty (an empty or whitespace-only quote is not a claim)."

    # Leading/trailing whitespace in `quote` is incidental padding, not a claim about what
    # precedes/follows the quoted text in `source` -- stripped BEFORE pattern-building so it never
    # turns into a mandatory `\s+` at the pattern's own edges (which would wrongly require `source`
    # to also have whitespace immediately before/after the quoted content, even when the quote is
    # simply the entire source or sits at a string boundary with nothing there at all).
    pieces = re.split(r"(\s+)", stripped_quote)
    pattern_parts = []
    for i, piece in enumerate(pieces):
        is_whitespace_run = i % 2 == 1  # re.split with a capturing group alternates literal/delim
        if is_whitespace_run and normalize_whitespace:
            pattern_parts.append(r"\s+")
        else:
            pattern_parts.append(re.escape(piece))
    pattern = "".join(pattern_parts)

    match = re.search(pattern, source)
    if match is not None:
        line = source.count("\n", 0, match.start()) + 1
        center = (match.start() + match.end()) // 2
        half = snippet_chars // 2
        lo = max(0, center - half)
        hi = min(len(source), lo + snippet_chars)
        lo = max(0, hi - snippet_chars)
        snippet = source[lo:hi]
        return f"MATCH: found at line {line} (char {match.start()}). Context: {snippet!r}"

    hint = ""
    if "\n" not in stripped_quote:
        candidates = source.splitlines()
        close = difflib.get_close_matches(
            stripped_quote, candidates, n=1, cutoff=_CLOSE_MATCH_CUTOFF
        )
        if close:
            ratio = difflib.SequenceMatcher(None, stripped_quote, close[0]).ratio()
            hint = f" Closest line: {close[0]!r} (~{ratio:.0%} similar)."
    return f"MISMATCH: quote not found in source (even allowing flexible whitespace).{hint}"
