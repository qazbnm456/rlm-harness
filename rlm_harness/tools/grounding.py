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
under whitespace normalization) in the held source text? "Normalization" is junction-aware rather
than uniform: whitespace between two word characters must be present in the source, everywhere else
it may be absent — so a quote that reflowed a line break beside a bracket or a quote mark still
verifies, while ``foo bar`` can never verify against ``foobar``. See ``_whitespace_joiner``.
It does not replace the model-judged
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


def _whitespace_joiner(before: str, after: str) -> str:
    """The pattern a whitespace run in the quote becomes: ``\\s+`` or ``\\s*``.

    ``\\s+`` at every junction is what a naive normalization does, and it REFUSES a correct
    citation wherever the source has no whitespace at that point — e.g. quoting a closing ``\"\"\"``
    onto its own line when the source keeps it on the previous one. Demonstrated on shipped code:

        source  x = \"\"\"One line.\\nAnd another.\"\"\"
        quote   \"\"\"One line.\\nAnd another.\\n\"\"\"      -> MISMATCH, wrongly

    ``\\s*`` everywhere would fix that and introduce the opposite, worse failure: whitespace
    DELETION, so ``foo bar`` would verify against ``foobar`` — an invented claim passing. The
    junction decides. Whitespace between two WORD characters is load-bearing (removing it glues
    two words into one that was never written), so it stays mandatory; anywhere else — beside a
    quote mark, a bracket, an operator, punctuation — it is layout the model reflowed, and is
    optional.

    Word-ness is tested with Python's UNICODE ``\\w``, never an ASCII class: CJK characters are word
    characters, so ``你好 世界`` keeps requiring the space against ``你好世界`` exactly as it does
    today. An ASCII class would silently start accepting it — the same trap CLAUDE.md's
    ``sanitize_tool_name`` rule names for identifier validity.

    Both neighbours are guaranteed non-empty: the quote is stripped before splitting, so no empty
    literal can sit at either edge, and ``\\s+`` is greedy so no two delimiters are adjacent. That
    is also what keeps the pattern free of adjacent or nested quantifiers.
    """
    tight = bool(before) and bool(after) and _is_word(before[-1]) and _is_word(after[0])
    return r"\s+" if tight else r"\s*"


def _is_word(ch: str) -> bool:
    return re.match(r"\w", ch, re.UNICODE) is not None


_NUMBERS_ONLY_LINE = re.compile(r"^\s*[0-9]+\s*$")


def _carries_only_numbers(quote: str) -> bool:
    """Is every non-blank line of ``quote`` a bare number and nothing else?

    Such a quote asserts no content, so verifying it confirms nothing — the same reason an empty
    or whitespace-only quote is refused. It reaches this function by a route the kit itself opens:
    ``make_read_file_tool(line_numbers=True)`` and ``edit_file``'s confirmation window both render
    a line as ``f"{n:>6}\\t{line}"``, so a BLANK line renders as ``"     7\\t"`` — whose ``.strip()``
    is ``"7"``, non-empty, and which then matches any source containing that digit.

    Three details are load-bearing and each has a named failure:

    * **It reads the RAW ``quote``, and under THIS rule that is defensive rather than
      load-bearing.** Verified equivalent to reading ``quote.strip()`` across 14,550 exhaustive
      combinations, and necessarily so: ``strip()`` removes exactly the leading/trailing whitespace
      the rule already ignores on both edges. The equivalence is a property of the rule's
      PERMISSIVENESS, not of the call — tighten either edge (require the separator, say) and the
      strip starts destroying the evidence it is meant to weigh, because ``"     7\\t".strip()`` is
      ``"7"``. Reading raw costs nothing and survives that change.
    * **Blank lines are SKIPPED, not disqualifying.** ``read_file``'s real output for a blank line
      keeps its trailing newline (``"     2\\t\\n"``), and a model that pads or wraps a quote emits
      ``"\\n     2\\t\\n"`` or ``"     2\\t\\n\\n"``. Requiring EVERY split element to match lets one
      empty element drop all three through.
    * **``\\s*`` on both sides and ``[0-9]``, not ``[ ]*`` + a literal tab and not ``\\d``.** A model
      trims the trailing tab (``"     2"``) or writes a space for it (``"     2 "``), and a quote may
      arrive with a leading tab; the tighter form misses all three and buys nothing, since no
      pattern requiring ``[0-9]+`` can match text starting with a letter. ``\\d`` is UNICODE — it
      accepts ``"     \u0662\\t"`` and ``"     \uff12\\t"``, which no renderer here emits.

    This encodes NO line-number format. The claim is only that a quote carrying nothing but digits
    is not a claim, which stands on its own — so ``grounding.py`` stays independent of the tools
    that render line numbers.

    **It closes the fully-blank case, not the whole class, and the residue is not only blank-ish
    lines.** A guttered quote that carries content is not refused here, and the GUTTER NUMBER is
    then searched as literal content — so the quote matches wherever that number happens to precede
    the line's text, including across a MANDATORY ``\\s+`` junction that spans blank lines. A full
    line of code is therefore reachable, not merely a line that is only punctuation. Live example
    in this repo's own suite:

        quote   "    42\\tdef test_run_isolated_does_not_see_an_outer_recorder(tmp_path):"
        source  line 39 is "    assert asyncio.run(outer()) == 42", then two blank lines
        result  MATCH at line 39 -- the pattern is `42\\s+def\\s+test_run_...`

    Measured by rendering every non-blank line of every ``.py`` here and verifying it against its
    own file: 2 false matches in 17,412 quotes (~1 in 8,700), one of them a full line of code.
    Closing these needs the gutter-stripping repair, which is deferred — see CHANGELOG 1.8.2. Do
    NOT describe this guard as refusing every guttered quote.
    """
    lines = [line for line in quote.split("\n") if line.strip()]
    return bool(lines) and all(_NUMBERS_ONLY_LINE.match(line) for line in lines)


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
    ``\\s+`` or ``\\s*`` by junction (when ``normalize_whitespace=True``, the default — see
    ``_whitespace_joiner``) or its own escaped self (when ``False``, for a caller that wants
    byte-exact whitespace too). So the match is whitespace-INSENSITIVE where whitespace is layout
    and whitespace-REQUIRING between two word characters, which is the only place its absence
    would change what was written. The resulting pattern runs
    directly against the ORIGINAL, un-normalized ``source`` — a real match has a real ``.start()``
    offset in ``source``, no position-remapping needed.

    **This needs no ``regex`` package / no timeout budget**, unlike ``make_grep_files_tool``'s
    ``pattern``: that tool's catastrophic-backtracking risk exists because the model supplies the
    PATTERN'S STRUCTURE directly. Here, the model supplies ``quote`` — literal text, not regex
    syntax — and every character of it is either escaped or collapsed to a flat, non-nested
    ``\\s+``/``\\s*``. Neither can ever end up adjacent to the other or nested: the quote is
    stripped before splitting (so no empty literal sits at either edge) and ``\\s+`` is greedy (so
    no two whitespace delimiters are adjacent), which means every quantifier is separated by a
    non-empty escaped literal that cannot itself contain whitespace — disjoint first-sets, no
    ambiguity. The resulting pattern can never contain nested/adjacent quantifiers or overlapping
    alternation — the shapes that exhibit catastrophic backtracking — so stdlib ``re`` is provably
    safe here regardless of what ``quote`` contains.

    **Pass the RAW file text as ``source``, not a line-numbered render.** ``read_file``'s
    ``line_numbers=True`` and ``edit_file``'s success snippet both prefix a line with
    ``f"{n:>6}\\t"``, so both are gutter-bearing text easy to copy by accident. A quote carrying a
    gutter usually fails here, because the gutter is not file content. The two are complementary:
    numbers so the MODEL can cite a coordinate without counting lines, raw text so the VERIFIER can
    check the claim.

    **This function will not strip a gutter for you, and that is a measured decision.** Removing a
    leading ``spaces + digits + tab`` repairs 98.1% of guttered quotes — and on a document whose
    line numbers ARE content (a stored line-numbered listing) it accepts a quote citing the WRONG
    number, because the remaining text still appears elsewhere in the file. That is an invented
    claim passing verification, the exact direction this function keeps closed.

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

    **A ``quote`` that asserts no content is refused before any search**, because a match on it
    would confirm nothing. Two shapes, each with its own message:

    * **Empty or whitespace-only.** It would reduce to the bare pattern ``\\s+``, matching almost
      any real text. Same reasoning ``edit_file`` applies to an empty ``old_string``.
    * **Only digits and whitespace**, on every non-blank line — which is what a BLANK line of a
      line-numbered render is (``"     7\\t"``). It would reduce to that digit and match wherever
      the digit occurs, reporting a line the citation never claimed.

    The second rule is deliberately blunt, so it also refuses an all-digit quote whose digits ARE
    in the source (``"8080"``, or a column of numbers from a numeric file). Quote the surrounding
    text instead, so that a match means something.

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

    # On the RAW `quote` -- equivalent to `stripped_quote` under today's rule, and deliberately
    # not relying on that. See `_carries_only_numbers`.
    if _carries_only_numbers(quote):
        return (
            "MISMATCH: quote carries only digits and whitespace, so there is no text to check "
            "(a blank line in a line-numbered render is just its number). Quote the line's TEXT "
            "instead; if the line is blank, quote the nearest non-blank line on its own."
        )

    # Leading/trailing whitespace in `quote` is incidental padding, not a claim about what
    # precedes/follows the quoted text in `source` -- stripped BEFORE pattern-building.
    # NOTE the ORIGINAL reason no longer applies: it was that an edge whitespace run would become a
    # mandatory `\s+` at the pattern's own edges, wrongly requiring `source` to have whitespace
    # around the quoted content. Since 1.6.1 `_whitespace_joiner` would give an edge run `\s*`
    # instead (its outward neighbour is the empty string), which is harmless. The strip stays
    # load-bearing for a DIFFERENT reason: it is what guarantees the joiner never sees an empty
    # neighbour on the inside, the premise this function's ReDoS-safety argument rests on.
    pieces = re.split(r"(\s+)", stripped_quote)
    pattern_parts = []
    for i, piece in enumerate(pieces):
        is_whitespace_run = i % 2 == 1  # re.split with a capturing group alternates literal/delim
        if is_whitespace_run and normalize_whitespace:
            pattern_parts.append(_whitespace_joiner(pieces[i - 1], pieces[i + 1]))
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
