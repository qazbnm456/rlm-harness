"""verify_quote -- deterministic host-side quote/citation grounding. Plain function, no factory,
offline, dspy-free.
"""
import time

import pytest

from rlm_harness.testing import assert_repl_safe
from rlm_harness.tools import verify_quote


def test_verify_quote_exact_verbatim_match(tmp_path):
    source = "line one\nThe quick brown fox jumps over the lazy dog.\nline three\n"
    result = verify_quote(source, "quick brown fox")
    assert result.startswith("MATCH:")
    assert "line 2" in result
    assert "quick brown fox" in result  # snippet includes the matched text


def test_verify_quote_whitespace_flexible_by_default():
    source = "The quick   brown\nfox jumps."
    result = verify_quote(source, "quick brown fox")
    assert result.startswith("MATCH:")


def test_verify_quote_byte_exact_when_normalize_whitespace_false():
    source = "The quick   brown\nfox jumps."
    result = verify_quote(source, "quick brown fox", normalize_whitespace=False)
    assert result.startswith("MISMATCH:")


def test_verify_quote_regex_metacharacters_are_escaped_not_interpreted():
    source = "Total cost: $5.00 (tax incl.) due today."
    result = verify_quote(source, "cost: $5.00 (tax incl.)")
    assert result.startswith("MATCH:")


def test_verify_quote_not_found_with_similar_line_hint():
    source = "line one here\nThe quick brown fox jumps\nline three here\n"
    result = verify_quote(source, "The quikc brown fox jumps")
    assert result.startswith("MISMATCH:")
    assert "Closest line" in result
    assert "The quick brown fox jumps" in result


def test_verify_quote_not_found_with_nothing_similar_has_no_hint():
    source = "line one here\nThe quick brown fox jumps\nline three here\n"
    result = verify_quote(source, "zzz totally unrelated qqq")
    assert result.startswith("MISMATCH:")
    assert "Closest line" not in result


def test_verify_quote_multiline_quote_not_found_skips_hint_without_crashing():
    source = "line one\nline two\nline three\n"
    result = verify_quote(source, "nonexistent line a\nnonexistent line b")
    assert result.startswith("MISMATCH:")
    assert "Closest line" not in result


def test_verify_quote_empty_quote_is_refused():
    result = verify_quote("hello world", "")
    assert result.startswith("MISMATCH: quote must be non-empty")


def test_verify_quote_whitespace_only_quote_is_refused_not_a_spurious_match():
    # The Revision 1 regression: a naive `quote == ""`-only guard would let "   " through, and it
    # would reduce to the bare pattern \s+ -- matching almost any real text at its first
    # whitespace run. Must be refused with the SAME dedicated message, not "MATCH:".
    result = verify_quote("hello world", "   ")
    assert result.startswith("MISMATCH: quote must be non-empty")


# --- a quote carrying only line numbers -------------------------------------------------------
#
# `make_read_file_tool(line_numbers=True)` and `edit_file`'s snippet both render a line as
# f"{n:>6}\t{line}", so a BLANK line renders as "     7\t" -- whose .strip() is "7". That is
# non-empty, so it passed the guard above and then matched any source containing that digit:
# a citation of NOTHING verified, reporting a line the citation never claimed. Shipped in 1.8.1.

# Every shape below returned "MATCH: found at line 1" before the fix. Source line 2 is blank.
_NUMBERS_ONLY_SHAPES = [
    "     2\t",        # the render, verbatim
    "     2",          # a model trimmed the trailing tab
    "     2\t\n",       # read_file's ACTUAL output -- the trailing newline is kept
    "     2\t\n\n",      # padded
    "\n     2\t\n",      # wrapped, e.g. out of a code fence
    "     2 ",         # a space written for the tab
    "2",               # the coordinate alone
    "\t     2\t",       # a leading tab
]


@pytest.mark.parametrize("quote", _NUMBERS_ONLY_SHAPES)
def test_a_quote_carrying_only_line_numbers_is_refused(quote):
    assert verify_quote("x = 42\n\ny = 1\n", quote).startswith(
        "MISMATCH: quote carries only digits"
    )


def test_the_numbers_only_refusal_is_not_the_empty_quote_message():
    # Two different failures with two different remedies: pad the quote vs. quote the TEXT. The
    # model reads these, so they must not be interchangeable.
    empty = verify_quote("x = 42\n\ny = 1\n", "   ")
    numbers = verify_quote("x = 42\n\ny = 1\n", "     2\t")
    assert empty.startswith("MISMATCH: quote must be non-empty")
    assert numbers.startswith("MISMATCH: quote carries only digits")
    assert empty != numbers


@pytest.mark.parametrize(
    ("quote", "verdict"),
    [
        ("x = 42", "MATCH:"),                                  # ordinary content
        ("     1\tx = 42", "MATCH: line 1 verified"),            # a gutter WITH content -- 1.9.0
        ("42 is the answer", "MISMATCH: quote not found"),     # starts with a number
        ("3.14", "MISMATCH: quote not found"),                 # not all digits
        ("     1\tx = 42\n     2\t", "MATCH: line 1 verified"),      # content + blank -- 1.9.0
    ],
)
def test_a_quote_that_carries_content_is_untouched(quote, verdict):
    # Each verdict was what the PRE-1.8.2 function returned, and 1.8.2's guard moved none of them.
    # 1.9.0 moves exactly two, ON PURPOSE: a gutter carrying CONTENT is a coordinate claim, and both
    # of these verify at line 1. That is the feature, not a regression -- the old comment here said
    # "the guard must not move any of them", which was 1.8.2's promise about 1.8.2's guard.
    assert verify_quote("x = 42\n\ny = 1\n", quote).startswith(verdict)


def test_the_rule_uses_ascii_digits_so_a_unicode_numeral_is_real_content():
    # \d is UNICODE: it accepts fullwidth and Arabic-Indic numerals, which no renderer here emits.
    # Treating those as a gutter would refuse a citation of text that IS in the source.
    assert verify_quote("count = \uff12\uff10\n", "\uff12\uff10").startswith("MATCH:")


def test_verify_quote_incidental_padding_on_a_real_quote_is_not_over_refused():
    # Only the CONTENT matters -- leading/trailing padding around a meaningful quote must not be
    # treated as if the quote were empty.
    result = verify_quote("the answer is 42", "  the answer is 42  ")
    assert result.startswith("MATCH:")


def test_verify_quote_empty_source_is_a_clean_mismatch():
    result = verify_quote("", "hello")
    assert result.startswith("MISMATCH:")


def test_verify_quote_match_at_the_very_start_of_source():
    source = "X" * 5 + " rest of the text here"
    result = verify_quote(source, "XXXXX")
    assert result.startswith("MATCH: found at line 1 (char 0)")


def test_verify_quote_match_at_the_very_end_of_source_clamps_snippet_safely():
    source = "A" * 190 + "UNIQUE_TAIL"
    result = verify_quote(source, "UNIQUE_TAIL")
    assert result.startswith("MATCH:")
    assert "UNIQUE_TAIL" in result  # snippet clamped to source bounds, no IndexError


def test_verify_quote_redos_shaped_literal_text_completes_instantly():
    # The literal text "(a+)+" -- the textbook catastrophic-backtracking PATTERN -- appears here
    # only as LITERAL QUOTE TEXT, never as regex syntax (every char is re.escape()d). Proves the
    # escaping actually neutralizes it, empirically, not just by design argument.
    redos_shaped_quote = "(a+)+" * 5
    big_source = "a" * 200_000  # non-matching: no literal "(a+)+..." substring anywhere in it
    started = time.monotonic()
    result = verify_quote(big_source, redos_shaped_quote)
    elapsed = time.monotonic() - started
    assert elapsed < 1.0, f"took {elapsed:.3f}s -- escaping may not be neutralizing the pattern"
    assert result.startswith("MISMATCH:")


def test_verify_quote_is_repl_safe():
    assert_repl_safe(verify_quote)


def test_verify_quote_importable_from_tools_package():
    from rlm_harness.tools import verify_quote as vq

    assert vq is verify_quote


# ---- junction-aware whitespace normalization ---------------------------------------------------
#
# Uniform `\s+` REFUSES correct citations wherever the source has no whitespace at a junction;
# uniform `\s*` would accept invented ones by gluing words. The junction decides which applies.


def test_a_quote_that_reflowed_a_break_beside_a_delimiter_verifies():
    """The reproducer, found in shipped code. The quote puts the closing delimiter on its own
    line; the source keeps it on the previous one. Nothing about the claim is wrong."""
    source = 'x = """One line.\nAnd another."""'
    quote = '"""One line.\nAnd another.\n"""'
    assert verify_quote(source, quote).startswith("MATCH")


def test_whitespace_between_two_word_characters_stays_mandatory():
    """The false-positive direction, which matters more than the false-negative one: an invented
    claim must never verify. `\\s*` everywhere would make each of these pass."""
    assert verify_quote("foobar", "foo bar").startswith("MISMATCH")
    assert verify_quote("returnvalue", "return value").startswith("MISMATCH")
    assert verify_quote("a1", "a 1").startswith("MISMATCH")


def test_word_ness_is_unicode_so_cjk_still_requires_its_whitespace():
    """Pinned deliberately: an ASCII `[A-Za-z0-9_]` class would treat CJK as non-word and silently
    start accepting a space the source never had. Same trap CLAUDE.md names for tool-name validity.
    """
    assert verify_quote("你好世界", "你好 世界").startswith("MISMATCH")
    assert verify_quote("你好 世界", "你好 世界").startswith("MATCH")


def test_junctions_beside_punctuation_and_brackets_are_optional():
    for source, quote in [
        ("f(a, b)", "f( a, b )"),
        ("x=1", "x = 1"),
        ("[1,2]", "[ 1, 2 ]"),
        ("end.\nNext", "end.\n\nNext"),
    ]:
        assert verify_quote(source, quote).startswith("MATCH"), (source, quote)


def test_normalize_whitespace_false_is_unaffected():
    """The junction rule lives entirely inside the normalizing branch."""
    assert verify_quote("a  b", "a  b", normalize_whitespace=False).startswith("MATCH")
    assert verify_quote("a  b", "a b", normalize_whitespace=False).startswith("MISMATCH")


def test_reported_match_position_is_the_leftmost_occurrence():
    """Loosening a junction makes an EARLIER occurrence match where only a later one did before,
    which MOVES the reported character offset — `a . b` used to be found at char 8 and is now
    found at char 0. A consumer parsing "found at line N (char M)" sees that directly, so it is
    pinned. Note the LINE number is unchanged here, which is why asserting on it alone would pass
    both before and after and pin nothing."""
    out = verify_quote("a.b ... a . b", "a . b")
    assert out.startswith("MATCH")
    assert "char 0" in out, out


# --- a guttered quote is a COORDINATE CLAIM (1.9.0) -------------------------------------------
#
# 1.8.2 closed the case where a citation of NOTHING verified and documented what it left open: a
# guttered quote carrying CONTENT is searched with the gutter DIGITS as literal content, so it
# matches wherever that number precedes the line's text. Rendering every non-blank line of every
# `.py` here and verifying it against its own file found 2 such matches in 18,761 -- and both are
# HONEST citations whose content really is at the gutter's line.

import pathlib


def _residual(path, claimed):
    src = pathlib.Path(path).read_text()
    return src, f"{claimed:>6}\t" + src.split("\n")[claimed - 1]


def test_the_live_residual_in_this_repos_own_async_suite_now_resolves():
    """Reason-to-exist #1, pinned by VALUE from this tree rather than by "no longer 39".

    `tests/test_async.py` line 39 ends `== 42` and two blank lines later line 42 begins a `def`,
    so the pattern `42\\s+def\\s+test_run_...` matches at 39 for a quote claiming 42."""
    src, quote = _residual("tests/test_async.py", 42)
    assert verify_quote(src, quote).startswith("MATCH: line 42 verified")


def test_the_live_residual_in_this_repos_own_mcp_suite_now_resolves():
    """Reason-to-exist #2. Its content is `)` -- hundreds of SUBSTRING hits, one whole-line hit,
    which is why the uniqueness criterion has to be a contiguous line SEQUENCE. A substring
    criterion misses this one entirely."""
    src, quote = _residual("tests/test_mcp.py", 25)
    assert verify_quote(src, quote).startswith("MATCH: line 25 verified")


@pytest.mark.parametrize(
    ("label", "wrap"),
    [
        ("read_file's real output", "{q}\n"),          # renders from readlines() KEEPENDS
        ("padded", "{q}\n\n"),                          # both shapes are verbatim entries in
        ("unwrapped from a fence", "\n{q}\n"),          # _NUMBERS_ONLY_SHAPES above
    ],
)
def test_the_shapes_a_model_actually_pastes_all_parse(label, wrap):
    """`quote.strip("\\n")`, not "strip one trailing newline". Stripping both ends cannot destroy a
    claimed line -- a blank SOURCE line renders as `"     8\\t"`, carrying its own gutter, never a
    bare `""` -- and it recovers the two shapes 1.8.2 already documents models emitting.

    Treating the trailing empty element as a CLAIMED blank line instead drops the fire rate from
    83.16% to 14.70% and fixes NEITHER residual, because `tests/test_async.py:43` is a comment."""
    src, quote = _residual("tests/test_async.py", 42)
    assert verify_quote(src, wrap.format(q=quote)).startswith("MATCH: line 42 verified")


def test_a_fabricated_indentation_level_is_refused():
    """EXACT equality, never whitespace-tolerant. `container_interpreter.py` holds the same
    statement at 12 spaces (line 285) and 16 (line 320). Each is exact-unique, so uniqueness alone
    passes -- a tolerant check would coordinate-VERIFY 285's content under a claimed gutter of 320,
    fabricating an indentation level in a language where indentation is semantics. 406 of 18,761
    lines here (2.16%, 61 files) are exact-unique but strip-duplicate."""
    src = pathlib.Path("rlm_harness/container_interpreter.py").read_text()
    lines = src.split("\n")
    assert lines[284].strip() == lines[319].strip() and lines[284] != lines[319]
    assert verify_quote(src, f"   320\t{lines[284]}").startswith("MISMATCH:")


def test_a_fabricated_coordinate_over_duplicated_content_is_refused():
    """Uniqueness is what closes fabrication, and it closes it BY CONSTRUCTION: exact content at n
    plus a block occurring once means n is the only line that can hold it.

    Without it a bare position check is WORSE than the shipped function -- ~0.15% of random
    fabricated coordinates verify against 0.000% today, because 16.84% of non-blank lines in this
    repo recur in their own file."""
    src = "dup\nother\ndup\n"
    assert verify_quote(src, "     3\tdup").startswith("MISMATCH:")      # true, but unprovable
    assert verify_quote(src, "     1\tdup").startswith("MISMATCH:")      # ...and so is the honest one


def test_non_adjacent_gutters_over_adjacent_content_are_refused():
    """Contiguity, and reaching it takes the RIGHT source.

    The quote claims lines 1 and 3. If the source's lines 1 and 3 are not adjacent in content the
    slice comparison already refuses it, so a source like `"alpha\nfiller\nbeta"` tests nothing --
    dropping the contiguity check keeps that green. What needs the check is content that HAPPENS to
    be contiguous at 1-2 while the gutters say 1 and 3: the slice matches, and without contiguity
    the function reports line 1, asserting an adjacency the citation never claimed."""
    src = "alpha\nbeta\ngamma\n"
    assert verify_quote(src, "     1\talpha\n     3\tbeta").startswith("MISMATCH:")
    # ...and the shape the slice alone already handles, kept so both paths are visible
    assert verify_quote("alpha\nfiller\nbeta\n", "     1\talpha\n     3\tbeta").startswith("MISMATCH:")


def test_a_block_running_past_EOF_is_refused():
    """Two inputs, because a trailing newline in `source` changes WHICH check fires.

    Getting this docstring right took three tries and the first two shipped a wrong mechanism, so
    the accurate account: the bounds and the slice are each individually REDUNDANT — deleting
    either leaves the whole suite green — but they are not JOINTLY redundant. Remove both and a
    fabricated coordinate one line past EOF verifies:

        verify_quote("x\ny\nb\nc\nz\nb", "     6\tb\n     7\tc")  ->  MATCH at line 6

    That is the false-positive class this feature exists to close, so neither is decoration. And it
    is not the uniqueness scan that refuses an overhang, which an earlier version of this docstring
    claimed — uniqueness found that block exactly once, at lines 3-4, which is why it verified."""
    # WITH a trailing newline there is no overhang at all: `split("\n")` leaves a phantom empty
    # last element, so `2-1+2 == 3 == len(lines)` passes the bound and the CONTENT comparison is
    # what refuses (`["b", ""] != ["b", "c"]`).
    assert verify_quote("a\nb\n", "     2\tb\n     3\tc").startswith("MISMATCH:")
    # WITHOUT one it genuinely runs past the end, which is the case the bound is written for.
    assert verify_quote("a\nb", "     2\tb\n     3\tc").startswith("MISMATCH:")


def test_removing_BOTH_the_bound_and_the_slice_would_verify_a_fabricated_coordinate():
    """The reason two individually-redundant checks both stay.

    Each can be deleted alone with the suite still green, which is exactly the shape that invites
    someone to delete both. Together they are load-bearing: with the upper bound gone AND the slice
    replaced by a `zip`, a block claiming lines 6-7 of a six-line source verifies against its
    content at lines 3-4. Pinned as behaviour here so the pair cannot erode one half at a time."""
    src = "x\ny\nb\nc\nz\nb"
    assert verify_quote(src, "     6\tb\n     7\tc").startswith("MISMATCH:")


def test_a_gutter_too_long_to_convert_does_not_raise():
    """`verify_quote` documents that it never raises. On CPython 3.11 `int("9" * 4301)` raises
    `ValueError: Exceeds the limit (4300) for integer string conversion`, and an unbounded
    `[0-9]+` matched such a gutter. 20 digits is NOT the boundary and would not have caught this."""
    assert verify_quote("a\nb\n", " " * 4 + "9" * 4301 + "\tfoo").startswith("MISMATCH:")
    assert verify_quote("a\nb\n", " " * 4 + "9" * 20 + "\tfoo").startswith("MISMATCH:")


def test_an_empty_source_is_refused_by_the_numbers_only_guard_not_by_bounds():
    """Pinning the REASON, because the guards' ORDER is what makes this safe.

    In isolation the coordinate check accepts `("", "     1\\t")`: 1 <= 1, the slice `[""][0:1]`
    equals `[""]`, and it occurs once. Only 1.8.2's numbers-only guard, which runs first, stops it.
    Reordering the guards would silently open it -- the same shape of mistake as assuming a bound
    protects a loop it is applied after."""
    assert verify_quote("", "     1\t").startswith("MISMATCH: quote carries only digits")


def test_normalize_whitespace_False_is_the_escape_hatch():
    """A source that itself contains numbered listings is the one shape where a CORRECT literal
    match gets overridden. Not observed across tens of thousands of local text files, but `cat
    -n` and `nl` emit it and
    `source` is arbitrary caller text, so there is a lever: byte-exact mode means no interpretation,
    and reading a gutter as a coordinate claim is an interpretation.

    Uses the zero-leading-whitespace shape, which is what a real `.tsv` line looks like -- `^\\s*`
    matches both, and this is the form that occurs outside a renderer."""
    lines = [f"pad{i}" for i in range(41)] + ["UNIQUE_LINE"] + [f"z{i}" for i in range(51)]
    src = "\n".join([*lines, "42\tUNIQUE_LINE"]) + "\n"
    quote = "42\tUNIQUE_LINE"
    assert verify_quote(src, quote).startswith("MATCH: line 42 verified")
    assert verify_quote(src, quote, normalize_whitespace=False).startswith("MATCH: found at line 94")


def test_a_repair_match_says_the_quoted_bytes_are_not_in_the_source():
    """`MATCH` acquires a second meaning here: the source holds the CONTENT, not the gutter. Two
    consumers read this field by name and one forwards it into a browser-facing body, so the two
    must be distinguishable -- while keeping the `MATCH:` prefix they branch on."""
    src, quote = _residual("tests/test_async.py", 42)
    repair = verify_quote(src, quote)
    literal = verify_quote("hello world", "hello")
    assert repair.startswith("MATCH:") and literal.startswith("MATCH:")
    assert "verified by line number" in repair and "verified by line number" not in literal
    assert quote.strip() not in src                  # the claim the note is about


def test_a_non_unique_guttered_quote_falls_through_unchanged():
    """The degrade path: 16.84% of non-blank lines here recur in their own file. Those keep exactly
    today's verdict rather than getting a wrong one -- and today's verdict for this class contains
    ZERO wrong-line MATCHes, so what a consumer loses is a coordinate they never had."""
    src = "same\nsame\n"
    assert verify_quote(src, "     2\tsame").startswith("MISMATCH:")


def test_a_form_feed_does_not_verify_a_fabricated_coordinate():
    """`split("\n")`, never `splitlines()`. The latter breaks on eight separators universal-newline
    `readlines()` does not, and the renderer uses `readlines()` -- so on a file containing `\x0c`
    the two disagree about which line is which.

    Pinned on the DANGEROUS half. Under `splitlines()` this source's third line is `c`, so a quote
    claiming line 3 holds `c` VERIFIES -- a fabricated coordinate, coordinate-verified. An earlier
    version of this test used a source where the mutation only broke the HONEST case, which is the
    safe direction: it went red, so the sweep looked clean, while the failure its own docstring
    described was demonstrated by nothing.

    15 of 9,401 stdlib `.py` files on this interpreter contain a form feed."""
    src = "a\nb\x0cc\nd\n"
    assert src.split("\n")[2] == "d" and src.splitlines()[2] == "c"     # the two disagree HERE
    assert verify_quote(src, "     3\tc").startswith("MISMATCH:")        # green only under split()
    assert verify_quote(src, "     3\td").startswith("MATCH: line 3 verified")   # the honest one


def test_a_gutter_of_zero_is_refused():
    """Pins the behaviour; the lower bound is not what produces it.

    `lines[0 - 1]` is the last line as a SUBSCRIPT, but this code SLICES, and `lines[-1:0]` is the
    empty list -- so a gutter of 0 cannot verify with or without the bound. This docstring used to
    assert the opposite and name itself the bound's guardian."""
    assert verify_quote("a\nlast\n", "     0\tlast").startswith("MISMATCH:")
