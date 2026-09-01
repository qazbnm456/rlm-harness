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
        ("     1\tx = 42", "MISMATCH: quote not found"),        # a gutter WITH content
        ("42 is the answer", "MISMATCH: quote not found"),     # starts with a number
        ("3.14", "MISMATCH: quote not found"),                 # not all digits
        ("     1\tx = 42\n     2\t", "MISMATCH: quote not found"),  # content line + blank
    ],
)
def test_a_quote_that_carries_content_is_untouched(quote, verdict):
    # Each verdict is what the PRE-1.8.2 function returned. The guard must not move any of them.
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
