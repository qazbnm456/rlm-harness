"""`LargeText`: the staging type, and the one read grant it opens.

The security properties are asserted rather than described, because the grant widens the boundary
`docs/INVARIANTS.md` opens with. Every test here is offline: no Deno, no model.
"""

from __future__ import annotations

import os
import re
import stat
import tempfile

import pytest

from rlm_harness.large_text import (
    LargeText,
    LargeTextRefused,
    stage_inputs,
    staged_instructions,
)
from rlm_harness.sandbox import build_interpreter


def test_nothing_a_caller_passes_can_steer_the_path():
    """The constructor takes the TEXT and nothing else, so no value chooses the location."""
    import inspect

    params = list(inspect.signature(LargeText.__init__).parameters)
    assert params == ["self", "text"], (
        f"LargeText grew a parameter besides the text: {params}. A name, prefix, directory or "
        "path argument is how a caller value, and then an attacker-controlled value, reaches the "
        "path that gets granted to the sandbox."
    )


def test_hostile_content_reaches_the_file_body_and_never_the_file_name():
    # Traversal, a NUL, a comma and a newline: everything that would matter in a path, supplied as
    # the payload. The basename is generated, so none of it lands anywhere that is parsed as a path.
    hostile = "../../etc/passwd\x00,/etc,--allow-read=/\nanything"
    with LargeText(hostile) as staged:
        name = os.path.basename(staged.host_path)
        assert re.fullmatch(r"context_[0-9a-f]{16}\.txt", name), name
        assert staged.sandbox_path == f"/sandbox/{name}"
        assert "," not in staged.host_path


def test_the_staged_file_is_not_writable_and_close_removes_it():
    """Asserted as "not writable", not as an octal, because `chmod` means different things.

    POSIX records the exact mode; Windows has only a read-only bit, so `0o400` reads back as
    `0o444` there and an exact comparison fails for a reason that is not about this file at all.
    The DELETION is the half that matters on every platform, and it is the half a Windows run had
    never reached: the mode assertion failed first, and `rmtree(ignore_errors=True)` would have
    left a read-only file behind in silence.
    """
    staged = LargeText("x" * 64)
    mode = os.stat(staged.host_path).st_mode
    assert mode & stat.S_IRUSR, oct(mode)
    assert not mode & stat.S_IWUSR, oct(mode)
    with pytest.raises(OSError), open(staged.host_path, "a", encoding="utf-8"):
        pass
    staged.close()
    assert not os.path.exists(staged.host_path)
    assert not os.path.exists(os.path.dirname(staged.host_path))
    staged.close()  # idempotent: a consumer closing twice is not an error


def test_a_comma_in_the_resolved_path_is_refused_rather_than_granted(monkeypatch, tmp_path):
    """Deno's `--allow-read` splits on commas, so one entry with a comma grants something else.

    dspy's own note and `stanfordnlp/dspy#9749` record the misparse. The generated basename cannot
    contain a comma, but `$TMPDIR` can, so the assembled path is what gets checked.
    """
    comma_dir = tmp_path / "has,comma"
    comma_dir.mkdir()
    monkeypatch.setattr(tempfile, "mkdtemp", lambda *a, **k: str(comma_dir))
    with pytest.raises(LargeTextRefused) as refused:
        LargeText("x")
    assert "comma" in str(refused.value)


def test_repr_carries_no_payload():
    # A trace line, a log, or an exception repr must not become the leak the file avoided.
    with LargeText("SUPERSECRET" * 32) as staged:
        assert "SUPERSECRET" not in repr(staged)
        assert str(staged.chars) in repr(staged)


def test_staging_is_by_type_and_leaves_every_other_input_alone():
    with LargeText("payload") as staged:
        clean, found = stage_inputs({"log": staged, "question": "why", "n": 3})
        assert found == {"log": staged}
        assert clean == {"log": staged.sandbox_path, "question": "why", "n": 3}
        # The value dspy receives is the SANDBOX path: the host path does not resolve in there.
        assert clean["log"].startswith("/sandbox/")


def test_an_input_that_merely_looks_like_one_is_not_staged():
    """Discovery by type is the whole guarantee, so a lookalike must not open a grant."""

    class Impostor:
        host_path = "/etc/passwd"
        sandbox_path = "/sandbox/passwd"
        chars = 1

        def preview(self, max_chars=500):
            return ""

    clean, found = stage_inputs({"log": Impostor(), "s": "/sandbox/context_dead.txt"})
    assert found == {}, "a duck-typed object opened a read grant"
    assert clean["s"] == "/sandbox/context_dead.txt"   # a plain string is still just a string


def test_no_staged_input_means_no_grant_and_no_added_instruction():
    clean, found = stage_inputs({"question": "hi"})
    assert clean == {"question": "hi"} and found == {}
    assert staged_instructions(found) == ""


def test_the_instruction_names_the_path_the_size_and_what_to_do():
    with LargeText("alpha beta gamma " * 10) as staged:
        text = staged_instructions({"log": staged}, preview_chars=11)
        assert "`log` holds the PATH" in text
        assert f"{staged.chars:,} characters" in text
        assert "in slices" in text
        assert "alpha beta" in text          # the head, so the model can recognise the shape
        assert staged.host_path not in text  # never the host location


def test_the_sandbox_is_granted_exactly_the_staged_file_and_nothing_by_default():
    plain = build_interpreter("pyodide")
    assert plain.enable_read_paths == [], (
        "a run that stages nothing must grant nothing: this is the default path every existing "
        "consumer takes"
    )
    with LargeText("x") as staged:
        granted = build_interpreter("pyodide", read_paths=(staged.host_path,))
        assert granted.enable_read_paths == [staged.host_path]


def test_a_read_grant_is_refused_for_a_kind_that_cannot_honour_it():
    # Accepting one for `mock` would report success while granting nothing.
    with pytest.raises(ValueError, match="read_paths is only supported"):
        build_interpreter("mock", read_paths=("/tmp/whatever.txt",))


def test_build_rlm_stays_callable_with_no_arguments():
    """`_build_rlm` is an override point, so staging must not reach it as a PARAMETER.

    A first attempt passed the staged map as a keyword argument. Every subclass that overrides
    `_build_rlm(self)`, which this repo's own tests do, then raised `TypeError`, and a surrounding
    `suppress(Exception)` turned that into a run which silently produced no trace fields at all.
    So the staged map travels on the instance and the call keeps taking nothing.
    """
    import inspect

    from rlm_harness.task import RLMTask

    params = list(inspect.signature(RLMTask._build_rlm).parameters)
    assert params == ["self"], (
        f"_build_rlm grew a parameter: {params}. Subclasses override it with a zero-argument "
        "signature, so a new parameter breaks them, and a caller that swallows exceptions makes "
        "that break silent."
    )
