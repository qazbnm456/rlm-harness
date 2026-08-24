"""list_candidate_paths -- a safe, good-default candidate_paths builder for
make_read_file_tool/make_grep_files_tool. Plain host-side function, offline, dspy-free.
"""
import os
from unittest import mock

import pytest

from rlm_harness.tools import CandidatePaths, list_candidate_paths
from rlm_harness.tools.fs import make_grep_files_tool


def _write(path, content=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_no_gitignore_present_returns_every_file_and_needs_no_pathspec(tmp_path, monkeypatch):
    _write(tmp_path / "main.py", "x = 1\n")
    _write(tmp_path / "pkg" / "util.py", "y = 2\n")

    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "pathspec":
            raise ImportError("simulated: pathspec not installed")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    result = list_candidate_paths(str(tmp_path))  # must not raise -- no pattern to compile
    assert set(result.paths) == {"main.py", "pkg/util.py"}
    assert result.truncated is False


def test_gitignore_excludes_matching_files_and_includes_itself(tmp_path):
    pytest.importorskip("pathspec")
    _write(tmp_path / ".gitignore", "*.log\n")
    _write(tmp_path / "main.py", "x = 1\n")
    _write(tmp_path / "debug.log", "log\n")

    result = list_candidate_paths(str(tmp_path))
    assert "main.py" in result.paths
    assert "debug.log" not in result.paths
    assert ".gitignore" in result.paths  # not itself conventionally gitignored


def test_gitignore_negation_reincludes_a_file(tmp_path):
    pytest.importorskip("pathspec")
    _write(tmp_path / ".gitignore", "*.log\n!keep.log\n")
    _write(tmp_path / "debug.log", "log\n")
    _write(tmp_path / "keep.log", "log\n")

    result = list_candidate_paths(str(tmp_path))
    assert "keep.log" in result.paths
    assert "debug.log" not in result.paths


def test_missing_pathspec_with_real_gitignore_raises_friendly_import_error(tmp_path, monkeypatch):
    _write(tmp_path / ".gitignore", "*.log\n")

    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "pathspec":
            raise ImportError("simulated: pathspec not installed")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"rlm-harness\[gitignore\]"):
        list_candidate_paths(str(tmp_path))


def test_missing_pathspec_with_only_extra_ignore_patterns_also_raises(tmp_path, monkeypatch):
    # The SECOND, independent trigger -- no .gitignore file at all, but extra_ignore_patterns
    # alone still needs pathspec to compile.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "pathspec":
            raise ImportError("simulated: pathspec not installed")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"rlm-harness\[gitignore\]"):
        list_candidate_paths(str(tmp_path), extra_ignore_patterns=("*.secret",))


def test_extra_ignore_patterns_work_with_no_gitignore_file_present(tmp_path):
    pytest.importorskip("pathspec")
    _write(tmp_path / "main.py", "x = 1\n")
    _write(tmp_path / "creds.secret", "s3cr3t\n")

    result = list_candidate_paths(str(tmp_path), extra_ignore_patterns=("*.secret",))
    assert "main.py" in result.paths
    assert "creds.secret" not in result.paths


def test_directory_only_pattern_prunes_the_whole_directory(tmp_path):
    # The exact regression case for the trailing-slash matching fix: a bare directory-only
    # pattern like "build/" must prune the WHOLE directory, not just fail silently.
    pytest.importorskip("pathspec")
    _write(tmp_path / ".gitignore", "build/\n")
    _write(tmp_path / "build" / "output.bin", "bin\n")
    _write(tmp_path / "main.py", "x = 1\n")

    result = list_candidate_paths(str(tmp_path))
    assert "build/output.bin" not in result.paths
    assert "main.py" in result.paths


def test_git_directory_always_excluded_even_without_gitignore(tmp_path):
    _write(tmp_path / ".git" / "HEAD", "ref: refs/heads/main\n")
    _write(tmp_path / "main.py", "x = 1\n")

    result = list_candidate_paths(str(tmp_path), respect_gitignore=False)
    assert not any(p.startswith(".git/") for p in result.paths)
    assert "main.py" in result.paths


def test_submodule_shaped_git_file_always_excluded(tmp_path):
    # git's REAL submodule gitlink shape: a plain FILE (not a directory) literally named ".git"
    # inside the submodule's own working tree -- the direct regression test for the gap a
    # directory-only prune would miss.
    _write(tmp_path / "vendor" / "submod" / ".git", "gitdir: ../../.git/modules/submod\n")
    _write(tmp_path / "vendor" / "submod" / "real.py", "x = 1\n")

    result = list_candidate_paths(str(tmp_path), respect_gitignore=False)
    assert "vendor/submod/.git" not in result.paths
    assert "vendor/submod/real.py" in result.paths


def test_symlinked_file_escaping_root_is_excluded_regardless_of_follow_symlinks(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret\n")
    (tmp_path / "link_to_secret.txt").symlink_to(outside / "secret.txt")
    _write(tmp_path / "real.py", "x = 1\n")

    for follow in (False, True):
        result = list_candidate_paths(
            str(tmp_path), respect_gitignore=False, follow_symlinks=follow
        )
        assert "link_to_secret.txt" not in result.paths, f"follow_symlinks={follow}"
        assert "real.py" in result.paths, f"follow_symlinks={follow}"


def test_symlinked_directory_escaping_root_is_excluded_when_followed(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside-dir"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret\n")
    (tmp_path / "link_dir").symlink_to(outside, target_is_directory=True)
    _write(tmp_path / "real.py", "x = 1\n")

    # follow_symlinks=True lets os.walk descend into the symlinked directory, but
    # resolve_within_root must still catch that its CONTENTS resolve outside root.
    result = list_candidate_paths(str(tmp_path), respect_gitignore=False, follow_symlinks=True)
    assert not any("secret.txt" in p for p in result.paths)
    assert "real.py" in result.paths


def test_max_files_truncates_the_walk_and_reports_it(tmp_path):
    for i in range(10):
        _write(tmp_path / f"f{i}.py", "x\n")

    result = list_candidate_paths(str(tmp_path), respect_gitignore=False, max_files=5)
    assert len(result.paths) == 5
    assert result.truncated is True

    result_full = list_candidate_paths(str(tmp_path), respect_gitignore=False, max_files=100)
    assert len(result_full.paths) == 10
    assert result_full.truncated is False


def test_max_files_truncation_is_deterministic_regardless_of_walk_order(tmp_path):
    # Two calls where the UNDERLYING os.walk is monkeypatched to hand back dirnames/filenames in
    # deliberately DIFFERENT orders (ascending vs. descending) across the two calls, against the
    # SAME tree exceeding max_files. Without the sort-before-use fix, the two calls would keep
    # different SUBSETS of files (whichever the truncation cut off first in each order) -- not
    # just a different ordering of the same set. This is the regression test for the fix; two
    # plain consecutive calls with no monkeypatching would not catch this, since consecutive reads
    # of an unmodified directory almost always observe the same underlying order regardless of
    # whether the code sorts.
    for i in range(10):
        _write(tmp_path / f"f{i}.py", "x\n")

    real_walk = os.walk

    def make_walk(reverse):
        def walk(top, **kwargs):
            for dirpath, dirnames, filenames in real_walk(top, **kwargs):
                dirnames[:] = sorted(dirnames, reverse=reverse)
                filenames[:] = sorted(filenames, reverse=reverse)
                yield dirpath, dirnames, filenames

        return walk

    import rlm_harness.tools.discover as discover_mod

    with mock.patch.object(discover_mod.os, "walk", make_walk(reverse=False)):
        first = list_candidate_paths(str(tmp_path), respect_gitignore=False, max_files=5)
    with mock.patch.object(discover_mod.os, "walk", make_walk(reverse=True)):
        second = list_candidate_paths(str(tmp_path), respect_gitignore=False, max_files=5)

    assert first.paths == second.paths
    assert first.paths == sorted(first.paths)


def test_glob_narrows_the_result_like_grep_files_own_glob(tmp_path):
    _write(tmp_path / "main.py", "x = 1\n")
    _write(tmp_path / "notes.txt", "hi\n")

    result = list_candidate_paths(str(tmp_path), respect_gitignore=False, glob="*.py")
    assert result.paths == ["main.py"]


def test_round_trips_directly_into_make_grep_files_tool(tmp_path):
    pytest.importorskip("regex")
    _write(tmp_path / "main.py", "def target():\n    pass\n")

    result = list_candidate_paths(str(tmp_path), respect_gitignore=False)
    tool = make_grep_files_tool(str(tmp_path), candidate_paths=result.paths)
    assert "def target" in tool("def target")


def test_candidate_paths_is_a_frozen_dataclass():
    cp = CandidatePaths(paths=["a.py"], truncated=False)
    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError, but avoid importing it
        cp.paths = ["b.py"]
