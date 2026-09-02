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
import itertools
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
    own file: 2 false matches in 18,761 quotes, one of them a full line of code. **1.9.0 resolves
    BOTH** -- see :func:`_coordinate_match`, which reads a guttered quote as a coordinate claim and
    verifies it rather than searching its digits as content. Do NOT describe THIS guard as doing
    that: it refuses a quote that carries no content at all, and nothing more.
    """
    lines = [line for line in quote.split("\n") if line.strip()]
    return bool(lines) and all(_NUMBERS_ONLY_LINE.match(line) for line in lines)


_GUTTER_LINE = re.compile(r"^\s*([0-9]{1,9})\t(.*)$")


def _coordinate_match(source: str, quote: str) -> int | None:
    """A guttered quote is a COORDINATE CLAIM. Return the line it verifies at, or ``None``.

    1.8.2 closed the case where a citation of NOTHING verified. The other half stayed open: a
    guttered quote carrying CONTENT is searched with the gutter DIGITS as literal content, so it
    matches wherever that number precedes the line's text. Rendering every non-blank line of every
    ``.py`` in this repo and verifying it against its own file found 2 such matches in 18,761 --
    and BOTH are honest citations whose content really is at the gutter's line. The function was
    reporting a coordinate the citation never claimed.

    A coordinate claim can be verified more strongly than a substring can, so when this verifies it
    wins; when it does not, nothing changes and the caller falls through to the literal search.
    Four conditions, each closing a specific shape rather than a hypothetical one:

    * **Consecutive gutters.** Two exact-unique lines 92 apart in this very file verify
      individually; without this they would return a MATCH asserting an adjacency that does not
      exist -- a NEW false-match class, in the direction this function keeps closed.
    * **Bounds -- BOTH belt-and-braces, and the docs said otherwise until a review checked.**
      Deleting either leaves the suite green. The slice refuses every overhang on its own (a short
      slice is unequal), and a gutter of ``0`` cannot verify either, because ``lines[-1:0]`` is the
      EMPTY list -- ``lines[0 - 1]`` is the last line as a SUBSCRIPT, and this code slices. An
      earlier version of this comment asserted the lower bound was load-bearing on exactly that
      confusion, and named a test as its guardian which was green under the mutation it claimed to
      catch. But they are not JOINTLY redundant: delete the upper bound AND replace the slice
      with a ``zip`` and a block claiming lines 6-7 of a six-line source verifies against its
      content at lines 3-4 -- a fabricated coordinate past EOF, coordinate-verified. Uniqueness
      does not save it, having found that block exactly once. Pinned by
      ``test_removing_BOTH_the_bound_and_the_slice_would_verify_a_fabricated_coordinate``, which is
      the reason both stay: "each is deletable alone" is exactly the shape that invites deleting
      both.
    * **EXACT equality, never whitespace-tolerant.** ``container_interpreter.py``'s line 285 and
      line 320 are the same statement at 12 and 16 spaces. Each is exact-unique, so uniqueness
      alone passes, and a tolerant check would coordinate-VERIFY 285's content under a claimed
      gutter of 320 -- a fabricated indentation level in a language where indentation is semantics.
      406 of 18,761 lines here (2.16%, across 61 files) are exact-unique but strip-duplicate.
    * **The block occurs exactly once as a contiguous line SEQUENCE**, not as a substring. The
      second residual's content is ``)``: hundreds of substring hits, one whole-line hit, so a
      substring criterion would miss it entirely.

    **Together the last two close fabrication by construction, not by luck**: exact content at
    ``n`` plus a block occurring once means ``n`` is the ONLY line that can hold it. Without
    uniqueness a bare position check is WORSE than today -- measured at ~0.15% of random fabricated
    coordinates verifying, against 0.000% for the shipped function, because 16.84% of non-blank
    lines here recur in their own file.

    Two parsing details carry their own measurements:

    * **``strip("\n")``, not "strip one trailing newline".** ``make_read_file_tool`` renders from
      ``readlines()`` keepends, so one numbered line arrives as ``"     2\tdef foo():\n"``.
      Stripping both ends cannot destroy a claimed line -- a blank SOURCE line renders as
      ``"     8\t"``, carrying its own gutter, never a bare ``""`` -- and it additionally parses
      the padded and code-fence-unwrapped shapes ``_NUMBERS_ONLY_SHAPES`` already documents models
      emitting. Treating the trailing empty element as a CLAIMED blank line instead drops the fire
      rate from 83.16% to 14.70% and fixes NEITHER residual.
    * **``{1,9}`` on the digit run is a contract fix.** ``verify_quote`` documents that it never
      raises, and on CPython 3.11 ``int("9" * 4301)`` raises ``ValueError: Exceeds the limit
      (4300)``. An unbounded ``[0-9]+`` matched such a gutter. Nine digits covers 999,999,999 lines.

    Note ``split("\n")``, never ``splitlines()``: the latter breaks on eight separators
    universal-newline ``readlines()`` does not, and the renderer uses ``readlines()``. The mismatch
    refuses honest citations and verifies fabricated ones on any file containing a form feed.
    """
    pairs: list[tuple[int, str]] = []
    for element in quote.strip("\n").split("\n"):
        m = _GUTTER_LINE.match(element)
        if m is None:
            return None
        pairs.append((int(m.group(1)), m.group(2)))
    if any(b - a != 1 for (a, _), (b, _) in itertools.pairwise(pairs)):
        return None

    start = pairs[0][0]
    block = [content for _, content in pairs]
    lines = source.split("\n")
    if start < 1 or start - 1 + len(block) > len(lines):
        return None
    if lines[start - 1 : start - 1 + len(block)] != block:
        return None
    occurrences = sum(
        1 for i in range(len(lines) - len(block) + 1) if lines[i : i + len(block)] == block
    )
    return start if occurrences == 1 else None


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
    gutter is READ AS A COORDINATE CLAIM since 1.9.0 and verified as one — see
    :func:`_coordinate_match`. The two remain complementary: numbers so the MODEL can cite a
    coordinate without counting lines, raw text so the VERIFIER can check the claim.

    **It still does not STRIP the gutter, and that distinction is the whole design.** Removing a
    leading ``spaces + digits + tab`` and searching the remainder repairs most guttered quotes and
    accepts a citation naming the WRONG line whenever the remainder appears anywhere else — an
    invented claim passing verification. Instead the gutter is used: the content must be at exactly
    the line the gutter names, and that block must occur exactly once. Without the uniqueness half a
    bare position check is WORSE than searching, because 16.84% of non-blank lines in this repo
    recur in their own file. With it, fabrication is closed by construction rather than by rate.

    A coordinate-verified MATCH says so in its text, because it means something different: the
    source holds the CONTENT at that line, but the quoted bytes — gutter included — are not a
    substring of it. A caller re-deriving grounding with ``quote in source`` must branch on that.
    Pass ``normalize_whitespace=False`` to skip this path entirely; byte-exact mode means no
    interpretation, and reading a gutter as a coordinate is an interpretation.

    On a LITERAL match the return includes the 1-indexed line number, a character offset, and a
    ``snippet_chars``-wide window of ``source`` centered on the match, clamped to the string's
    bounds. **A coordinate-verified match (see** :func:`_coordinate_match` **) returns neither an
    offset nor a snippet**, because there is no match position to report: the quoted bytes are not
    in ``source`` at all, only their content is, and the line number IS the answer. The design
    considered giving it the offset of the content at that line and dropped it -- an offset only a
    coordinate match ever produces would invite a caller to treat the two returns as one shape,
    which is the thing the wording of that return exists to prevent. ``snippet_chars`` is unused on
    that path.

    The literal path's snippet is there to sanity-check that the RIGHT occurrence was found when
    ``quote`` is generic enough to match more than one place — it reports the FIRST match only, via
    ``re.search``, a presence check rather than an enumeration; a model wanting every occurrence in
    a FILE already has ``grep_files``. A coordinate match needs no such check: uniqueness is one of
    its four conditions, so there is exactly one place it could be.

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

    # A COORDINATE CLAIM outranks a substring, so this runs BEFORE the literal search -- the
    # residual it closes IS that search matching on the gutter's digits. Skipped under
    # `normalize_whitespace=False`: that mode means byte-exact matching with no interpretation, and
    # reading a gutter as a coordinate claim is an interpretation. It is also the one lever a caller
    # has if `source` itself contains numbered listings, where a correct literal match could
    # otherwise be overridden -- a shape not observed in tens of thousands of local text files, but
    # `cat -n` emits it.
    if normalize_whitespace:
        line = _coordinate_match(source, quote)
        if line is not None:
            return (
                f"MATCH: line {line} verified by line number. The source holds this content at "
                f"that line, and nowhere else. NOTE the quote's gutter is not in the source, so "
                f"unlike an ordinary MATCH the quoted text is not a substring of it."
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
