"""atomic_write_text — write-without-partial-file. All offline, dspy-free."""
from __future__ import annotations

import os

import pytest

from rlm_harness import atomic_write_text


def _read(path):
    with open(path, encoding="utf-8") as fh:
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
