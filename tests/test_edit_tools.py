"""make_write_file_tool / make_edit_file_tool — the write side of the filesystem tools. All
offline, dspy-free.
"""
import os
import stat
import types

import pytest

from rlm_harness.testing import assert_repl_safe, assert_task_repl_safe
from rlm_harness.tools import make_edit_file_tool, make_write_file_tool
from rlm_harness.trace import EVENT_TOOL_CALL, TraceRecorder, load_events


def _duck_task(signature="q: str -> a: str", tools=()):
    """Minimal duck-typed stand-in `assert_task_repl_safe` accepts (mirrors the same-shaped
    helper already independently defined in tests/test_repl_safety.py and tests/test_tools.py —
    each test file keeps its own local copy, matching existing convention; there is no shared
    conftest.py fixture for it)."""
    return types.SimpleNamespace(signature=signature, tools=list(tools), output_field="a")


# ---- make_write_file_tool --------------------------------------------------------------------

def test_write_file_creates_a_new_file(tmp_path):
    tool = make_write_file_tool(str(tmp_path))
    tool("out.txt", "hello")
    assert (tmp_path / "out.txt").read_text() == "hello"


def test_write_file_overwrites_an_existing_file(tmp_path):
    (tmp_path / "out.txt").write_text("old")
    tool = make_write_file_tool(str(tmp_path))
    tool("out.txt", "new")
    assert (tmp_path / "out.txt").read_text() == "new"


def test_write_file_refuses_a_path_that_escapes_root(tmp_path):
    tool = make_write_file_tool(str(tmp_path))
    result = tool("../../etc/passwd", "pwned")
    assert result.startswith("Refused")


def test_write_file_creates_a_not_yet_existing_nested_subdirectory(tmp_path):
    # A real behavioral difference from the read-only tools, which only ever touch pre-existing
    # paths — exercises atomic_write_text's own os.makedirs, and confirms resolve_within_root's
    # containment check still holds for a path whose intermediate directories don't exist yet.
    tool = make_write_file_tool(str(tmp_path))
    tool("a/b/c/out.txt", "nested")
    assert (tmp_path / "a" / "b" / "c" / "out.txt").read_text() == "nested"


def test_write_file_overwrite_preserves_permission_bits(tmp_path):
    path = tmp_path / "script.sh"
    path.write_text("#!/bin/sh\necho hi\n")
    os.chmod(path, 0o755)
    tool = make_write_file_tool(str(tmp_path))
    tool("script.sh", "#!/bin/sh\necho bye\n")
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o755
    assert path.read_text() == "#!/bin/sh\necho bye\n"


def test_write_file_name_override_sets_repl_identity_and_trace_tag(tmp_path):
    tool = make_write_file_tool(str(tmp_path), name="write_docs")
    assert tool.__name__ == "write_docs"
    trace_path = str(tmp_path / "t.jsonl")
    with TraceRecorder(trace_path, run_id="r1"):
        tool("out.txt", "x")
    tc = [e for e in load_events(trace_path) if e["type"] == EVENT_TOOL_CALL][0]
    assert tc["payload"]["tool"] == "write_docs"


def test_write_file_name_override_fixes_the_real_multi_root_collision():
    a = make_write_file_tool("/tmp/source-root")
    b = make_write_file_tool("/tmp/docs-root")
    with pytest.raises(AssertionError, match="duplicate REPL tool name"):
        assert_task_repl_safe(_duck_task(tools=[a, b]))

    a2 = make_write_file_tool("/tmp/source-root", name="write_source")
    b2 = make_write_file_tool("/tmp/docs-root", name="write_docs")
    assert_task_repl_safe(_duck_task(tools=[a2, b2]))  # must not raise


def test_write_file_name_invalid_identifier_raises_value_error():
    with pytest.raises(ValueError, match="not a valid tool name"):
        make_write_file_tool("/tmp/x", name="write-file")


def test_write_file_name_reserved_by_dspy_raises_value_error():
    from rlm_harness._dspy_compat import reserved_tool_names

    reserved = next(iter(reserved_tool_names()))
    with pytest.raises(ValueError, match="reserved by dspy's sandbox"):
        make_write_file_tool("/tmp/x", name=reserved)


def test_write_file_is_repl_safe(tmp_path):
    assert_repl_safe(make_write_file_tool(str(tmp_path)))


def test_write_file_encoding_round_trips_non_utf8(tmp_path):
    tool = make_write_file_tool(str(tmp_path), encoding="latin-1")
    tool("latin1.txt", "café")
    assert (tmp_path / "latin1.txt").read_bytes() == "café".encode("latin-1")


# ---- make_edit_file_tool ----------------------------------------------------------------------

def _make_file(tmp_path, content):
    path = tmp_path / "file.py"
    path.write_text(content)
    return path


def test_edit_file_single_occurrence_succeeds(tmp_path):
    _make_file(tmp_path, "def foo():\n    return 1\n")
    tool = make_edit_file_tool(str(tmp_path))
    result = tool("file.py", "return 1", "return 2")
    assert "Replaced 1 occurrence" in result
    assert (tmp_path / "file.py").read_text() == "def foo():\n    return 2\n"


def test_edit_file_multiple_occurrences_default_refuses_and_leaves_file_untouched(tmp_path):
    original = "x = 1\nx = 1\nx = 1\n"
    path = _make_file(tmp_path, original)
    tool = make_edit_file_tool(str(tmp_path))
    result = tool("file.py", "x = 1", "x = 2")
    assert "3 times" in result
    assert "Refused" in result
    # The file must be BYTE-FOR-BYTE unchanged -- not just "the return string says refused".
    assert path.read_text() == original


def test_edit_file_multiple_occurrences_replace_all_replaces_every_one(tmp_path):
    _make_file(tmp_path, "x = 1\nx = 1\nx = 1\n")
    tool = make_edit_file_tool(str(tmp_path))
    result = tool("file.py", "x = 1", "x = 2", replace_all=True)
    assert "Replaced 3 occurrences" in result
    assert (tmp_path / "file.py").read_text() == "x = 2\nx = 2\nx = 2\n"


def test_edit_file_old_string_not_found_refuses_and_leaves_file_untouched(tmp_path):
    original = "hello\n"
    path = _make_file(tmp_path, original)
    tool = make_edit_file_tool(str(tmp_path))
    result = tool("file.py", "goodbye", "hi")
    assert "not found" in result.lower()
    assert path.read_text() == original


def test_edit_file_empty_old_string_is_refused(tmp_path):
    original = "hello\n"
    path = _make_file(tmp_path, original)
    tool = make_edit_file_tool(str(tmp_path))
    result = tool("file.py", "", "x")
    assert "Refused" in result
    assert path.read_text() == original


def test_edit_file_old_string_equals_new_string_is_refused(tmp_path):
    original = "hello\n"
    path = _make_file(tmp_path, original)
    tool = make_edit_file_tool(str(tmp_path))
    result = tool("file.py", "hello", "hello")
    assert "Refused" in result
    assert path.read_text() == original


def test_edit_file_new_string_empty_is_a_legitimate_deletion(tmp_path):
    # new_string == "" (delete this text) is a well-behaved operation and must NOT be refused --
    # only old_string=="" and old_string==new_string are refused.
    _make_file(tmp_path, "hello world\n")
    tool = make_edit_file_tool(str(tmp_path))
    result = tool("file.py", "world", "")
    assert "Replaced 1 occurrence" in result
    assert (tmp_path / "file.py").read_text() == "hello \n"


def test_edit_file_refuses_a_path_that_escapes_root(tmp_path):
    tool = make_edit_file_tool(str(tmp_path))
    result = tool("../../etc/passwd", "root:x", "hacked")
    assert result.startswith("Refused")


def test_edit_file_missing_file_is_an_error_string_not_an_exception(tmp_path):
    tool = make_edit_file_tool(str(tmp_path))
    result = tool("does/not/exist.py", "x", "y")
    assert "error" in result.lower()


def test_edit_file_directory_path_is_an_error_string_not_a_raised_exception(tmp_path):
    (tmp_path / "adir").mkdir()
    tool = make_edit_file_tool(str(tmp_path))
    result = tool("adir", "x", "y")
    assert "error" in result.lower()


def test_edit_file_preserves_permission_bits(tmp_path):
    # edit_file ALWAYS operates on a pre-existing file (unlike write_file, which can create a
    # brand-new one) -- making this arguably the more important of the two permission tests.
    path = tmp_path / "script.sh"
    path.write_text("#!/bin/sh\necho hi\n")
    os.chmod(path, 0o755)
    tool = make_edit_file_tool(str(tmp_path))
    tool("script.sh", "hi", "bye")
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o755
    assert path.read_text() == "#!/bin/sh\necho bye\n"


def test_edit_file_name_override_sets_repl_identity_and_trace_tag(tmp_path):
    _make_file(tmp_path, "hello\n")
    tool = make_edit_file_tool(str(tmp_path), name="edit_docs")
    assert tool.__name__ == "edit_docs"
    trace_path = str(tmp_path / "t.jsonl")
    with TraceRecorder(trace_path, run_id="r1"):
        tool("file.py", "hello", "hi")
    tc = [e for e in load_events(trace_path) if e["type"] == EVENT_TOOL_CALL][0]
    assert tc["payload"]["tool"] == "edit_docs"


def test_edit_file_name_override_fixes_the_real_multi_root_collision():
    a = make_edit_file_tool("/tmp/source-root")
    b = make_edit_file_tool("/tmp/docs-root")
    with pytest.raises(AssertionError, match="duplicate REPL tool name"):
        assert_task_repl_safe(_duck_task(tools=[a, b]))

    a2 = make_edit_file_tool("/tmp/source-root", name="edit_source")
    b2 = make_edit_file_tool("/tmp/docs-root", name="edit_docs")
    assert_task_repl_safe(_duck_task(tools=[a2, b2]))  # must not raise


def test_edit_file_name_invalid_identifier_raises_value_error():
    with pytest.raises(ValueError, match="not a valid tool name"):
        make_edit_file_tool("/tmp/x", name="edit-file")


def test_edit_file_name_reserved_by_dspy_raises_value_error():
    from rlm_harness._dspy_compat import reserved_tool_names

    reserved = next(iter(reserved_tool_names()))
    with pytest.raises(ValueError, match="reserved by dspy's sandbox"):
        make_edit_file_tool("/tmp/x", name=reserved)


def test_edit_file_is_repl_safe(tmp_path):
    assert_repl_safe(make_edit_file_tool(str(tmp_path)))


def test_edit_file_encoding_round_trips_non_utf8(tmp_path):
    path = tmp_path / "latin1.txt"
    path.write_bytes("café".encode("latin-1"))
    tool = make_edit_file_tool(str(tmp_path), encoding="latin-1")
    tool("latin1.txt", "café", "thé")
    assert path.read_bytes() == "thé".encode("latin-1")


# ---- windowed snippet on success (show_snippet=/snippet_context_lines=/max_snippet_occurrences=)
# ---- -- additive to the return string only; every existing test above this point still passes
# ---- unmodified (all check the confirmation line via substring, never exact equality).

def _numbered_lines(tmp_path, start=1, end=10):
    return "\n".join(f"line{i}" for i in range(start, end + 1)) + "\n"


def test_edit_file_snippet_shows_context_around_the_change(tmp_path):
    _make_file(tmp_path, _numbered_lines(tmp_path))
    tool = make_edit_file_tool(str(tmp_path), snippet_context_lines=2)
    result = tool("file.py", "line5", "CHANGED")
    assert "--- snippet 1/1 (lines 5-5) ---" in result
    assert "     3\tline3" in result
    assert "     4\tline4" in result
    assert "     5\tCHANGED" in result
    assert "     6\tline6" in result
    assert "     7\tline7" in result
    assert "line1" not in result  # outside the context window


def test_edit_file_snippet_on_line_1_has_no_negative_line_numbers(tmp_path):
    _make_file(tmp_path, _numbered_lines(tmp_path))
    tool = make_edit_file_tool(str(tmp_path), snippet_context_lines=3)
    result = tool("file.py", "line1\n", "CHANGED\n")  # "line1\n", not "line10"'s own prefix
    assert "--- snippet 1/1 (lines 1-1) ---" in result
    assert "     1\tCHANGED" in result
    assert "-1\t" not in result and "0\t" not in result  # no negative/zero line numbers


def test_edit_file_snippet_at_last_line_ends_cleanly_at_eof(tmp_path):
    _make_file(tmp_path, _numbered_lines(tmp_path))
    tool = make_edit_file_tool(str(tmp_path), snippet_context_lines=3)
    result = tool("file.py", "line10", "CHANGED")
    assert "--- snippet 1/1 (lines 10-10) ---" in result
    assert "    10\tCHANGED" in result  # simply ends here, no crash, no line 11/12/13


def test_edit_file_snippet_trailing_newline_end_line_is_not_off_by_one(tmp_path):
    # The exact Revision 1 counterexample: new_string ends in "\n", inserting whole lines.
    # The reported end line must be the LAST line actually inside new_string (3), never one past
    # it (4, the next, untouched line) -- proves the end-offset math doesn't use the
    # one-past-the-end offset directly.
    path = _make_file(tmp_path, "AAA\nBBB\nCCC\n")
    tool = make_edit_file_tool(str(tmp_path), snippet_context_lines=2)
    result = tool("file.py", "BBB\n", "XXX\nYYY\n")
    assert "--- snippet 1/1 (lines 2-3) ---" in result
    assert path.read_text() == "AAA\nXXX\nYYY\nCCC\n"
    assert "     1\tAAA" in result
    assert "     2\tXXX" in result
    assert "     3\tYYY" in result
    assert "     4\tCCC" in result


def test_edit_file_snippet_replace_all_gives_each_occurrence_its_own_block(tmp_path):
    content = "\n".join("x = 1" if i in (2, 5, 8) else f"line{i}" for i in range(1, 11)) + "\n"
    _make_file(tmp_path, content)
    tool = make_edit_file_tool(str(tmp_path), snippet_context_lines=1)
    result = tool("file.py", "x = 1", "x = 2", replace_all=True)
    assert "--- snippet 1/3 (lines 2-2) ---" in result
    assert "--- snippet 2/3 (lines 5-5) ---" in result
    assert "--- snippet 3/3 (lines 8-8) ---" in result
    assert "not shown" not in result  # all 3 occurrences fit under the default cap of 3


def test_edit_file_snippet_replace_all_exceeding_cap_shows_a_note(tmp_path):
    content = "\n".join(
        "x = 1" if i % 2 == 0 else f"line{i}" for i in range(1, 11)
    ) + "\n"  # 5 occurrences
    path = _make_file(tmp_path, content)
    tool = make_edit_file_tool(str(tmp_path), max_snippet_occurrences=2)
    result = tool("file.py", "x = 1", "x = 2", replace_all=True)
    assert "Replaced 5 occurrences" in result
    assert result.count("--- snippet") == 2
    assert "... 3 more occurrence(s) not shown." in result
    # The FILE itself is still fully edited regardless of how many snippets were shown.
    assert path.read_text().count("x = 2") == 5
    assert "x = 1" not in path.read_text()


def test_edit_file_snippet_oversized_region_shows_head_tail_and_omitted_marker(tmp_path):
    _make_file(tmp_path, "AAA\nBBB\nCCC\n")
    tool = make_edit_file_tool(str(tmp_path), snippet_context_lines=2)
    big_insert = "\n".join(f"new{i}" for i in range(1, 21)) + "\n"  # 20 lines, well over 2*2+1=5
    result = tool("file.py", "BBB\n", big_insert)
    assert "line(s) omitted" in result
    assert "new1\n" in result or "\tnew1" in result  # head shown
    assert "\tnew20" in result  # tail shown
    assert "new10" not in result  # middle genuinely omitted, not just visually collapsed
    # Bounded: total snippet size stays small regardless of the 20-line insertion.
    assert len(result) < 1000


def test_edit_file_snippet_deletion_renders_a_sensible_window(tmp_path):
    path = _make_file(tmp_path, "hello world\n")
    tool = make_edit_file_tool(str(tmp_path))
    result = tool("file.py", "world", "")
    assert "--- snippet 1/1 (lines 1-1) ---" in result
    assert "hello " in result  # the space survives; "world" is gone
    assert path.read_text() == "hello \n"


def test_edit_file_show_snippet_false_is_byte_identical_to_the_terse_confirmation(tmp_path):
    _make_file(tmp_path, "hello\n")
    tool = make_edit_file_tool(str(tmp_path), show_snippet=False)
    result = tool("file.py", "hello", "bye")
    assert result == "Replaced 1 occurrence in 'file.py'."


def test_edit_file_snippet_never_appended_to_refusal_or_error_paths(tmp_path):
    _make_file(tmp_path, "hello\n")
    tool = make_edit_file_tool(str(tmp_path))
    assert tool("file.py", "nonexistent", "x") == "Refused: old_string not found in 'file.py'."
    assert tool("file.py", "", "x") == (
        "Refused: old_string must be non-empty (an empty anchor is ambiguous)."
    )
    assert tool("file.py", "hello", "hello") == (
        "Refused: old_string and new_string are identical — nothing to edit."
    )
    assert tool("../../etc/passwd", "x", "y").startswith("Refused")
    assert "error" in tool("does/not/exist.py", "x", "y").lower()
    (tmp_path / "adir").mkdir()
    assert "error" in tool("adir", "x", "y").lower()
