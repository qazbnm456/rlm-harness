"""atomic_write_text / atomic_write_stream — write-without-partial-file. All offline, dspy-free."""
from __future__ import annotations

import os
import stat

import pytest

from rlm_harness import atomic_write_stream, atomic_write_text
from rlm_harness.atomic import _ExtractionBudgetExceeded


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _read_bytes(path):
    with open(path, "rb") as fh:
        return fh.read()


def test_atomic_write_text_writes_the_file(tmp_path):
    path = str(tmp_path / "out.txt")
    atomic_write_text(path, "hello")
    assert _read(path) == "hello"


def test_atomic_write_text_creates_a_nested_directory(tmp_path):
    path = str(tmp_path / "a" / "b" / "c" / "out.txt")
    atomic_write_text(path, "nested")
    assert _read(path) == "nested"


def test_atomic_write_text_bare_relative_filename(tmp_path, monkeypatch):
    # os.path.dirname("checkpoint.json") == "" — os.makedirs("", exist_ok=True) raises
    # FileNotFoundError without the `dirname or "."` guard. An entirely ordinary usage pattern
    # (write relative to the cwd) must not crash.
    monkeypatch.chdir(tmp_path)
    atomic_write_text("checkpoint.json", "bare")
    assert _read("checkpoint.json") == "bare"


def test_atomic_write_text_overwrites_atomically(tmp_path):
    path = str(tmp_path / "out.txt")
    atomic_write_text(path, "first")
    atomic_write_text(path, "second")
    assert _read(path) == "second"


def test_atomic_write_text_leaves_no_partial_or_temp_file_on_failure(tmp_path, monkeypatch):
    path = str(tmp_path / "out.txt")

    real_fdopen = os.fdopen

    def boom(fd, *a, **kw):
        fh = real_fdopen(fd, *a, **kw)
        fh.write("partial")
        raise RuntimeError("simulated failure mid-write")

    monkeypatch.setattr(os, "fdopen", boom)
    with pytest.raises(RuntimeError, match="simulated failure"):
        atomic_write_text(path, "should never land")

    assert not os.path.exists(path)
    leftovers = [f for f in os.listdir(tmp_path) if f.startswith(".tmp-")]
    assert leftovers == []


def test_atomic_write_text_replace_only_called_after_content_is_complete(tmp_path, monkeypatch):
    path = str(tmp_path / "out.txt")
    seen = {}

    real_replace = os.replace

    def spy_replace(src, dst):
        seen["content_at_replace_time"] = _read(src)
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy_replace)
    atomic_write_text(path, "complete-content")
    assert seen["content_at_replace_time"] == "complete-content"
    assert _read(path) == "complete-content"


def test_atomic_write_text_preserves_permission_bits_on_overwrite(tmp_path):
    # tempfile.mkstemp always creates its temp file at mode 0600 regardless of umask, and
    # os.replace does NOT carry the destination's mode across -- without the fix, overwriting an
    # existing file through atomic_write_text silently resets it to 0600, stripping e.g. the
    # executable bit off a script. This is the exact regression the fix in atomic.py exists to
    # prevent.
    path = str(tmp_path / "script.sh")
    atomic_write_text(path, "#!/bin/sh\necho hi\n")
    os.chmod(path, 0o755)
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o755

    atomic_write_text(path, "#!/bin/sh\necho bye\n")

    assert stat.S_IMODE(os.stat(path).st_mode) == 0o755
    assert _read(path) == "#!/bin/sh\necho bye\n"


def test_atomic_write_stream_assembles_chunks_in_order(tmp_path):
    path = str(tmp_path / "out.bin")
    written = atomic_write_stream(path, [b"abc", b"def", b"ghi"])
    assert written == 9
    assert _read_bytes(path) == b"abcdefghi"


def test_atomic_write_stream_creates_a_nested_directory(tmp_path):
    path = str(tmp_path / "a" / "b" / "c" / "out.bin")
    atomic_write_stream(path, [b"nested"])
    assert _read_bytes(path) == b"nested"


def test_atomic_write_stream_preserves_permission_bits_on_overwrite(tmp_path):
    # Same regression this fix already covers for atomic_write_text -- tempfile.mkstemp always
    # creates its temp file at mode 0600 regardless of umask, and os.replace does not carry the
    # destination's mode across.
    path = str(tmp_path / "script.sh")
    atomic_write_stream(path, [b"#!/bin/sh\n"])
    os.chmod(path, 0o755)
    atomic_write_stream(path, [b"#!/bin/sh\necho hi\n"])
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o755


def test_atomic_write_stream_unbounded_by_default(tmp_path):
    path = str(tmp_path / "out.bin")
    written = atomic_write_stream(path, [b"x" * 10_000])
    assert written == 10_000


def test_atomic_write_stream_budget_exceeded_raises_dedicated_subclass(tmp_path):
    path = str(tmp_path / "out.bin")
    with pytest.raises(_ExtractionBudgetExceeded):
        atomic_write_stream(path, [b"x" * 100], max_bytes=10)
    assert isinstance(_ExtractionBudgetExceeded("x"), OSError)


def test_atomic_write_stream_budget_exceeded_leaves_no_temp_file_and_destination_untouched(
    tmp_path,
):
    path = str(tmp_path / "out.bin")
    with open(path, "wb") as fh:
        fh.write(b"pre-existing content")

    with pytest.raises(_ExtractionBudgetExceeded):
        atomic_write_stream(path, [b"x" * 100], max_bytes=10)

    assert _read_bytes(path) == b"pre-existing content"
    leftovers = [f for f in os.listdir(tmp_path) if f.startswith(".tmp-")]
    assert leftovers == []


def test_atomic_write_stream_budget_checked_after_every_chunk_not_only_at_the_end(tmp_path):
    path = str(tmp_path / "out.bin")
    consumed = []

    def chunks():
        # An effectively unbounded generator -- if atomic_write_stream only checked the running
        # total once at the very end (after fully draining the iterable), this would hang/consume
        # all 1000 chunks before ever raising. Checking `consumed` below proves the abort instead
        # happens DURING iteration, the moment the running total crosses max_bytes.
        for i in range(1000):
            consumed.append(i)
            yield b"a" * 5

    with pytest.raises(_ExtractionBudgetExceeded):
        atomic_write_stream(path, chunks(), max_bytes=8)
    assert len(consumed) < 1000
    assert not os.path.exists(path)
