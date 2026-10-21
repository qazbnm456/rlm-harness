"""Packaging guard: keep the co-dev editable overlay from shadowing a consumer's tests.

A consumer co-develops rlm-harness by overlaying an editable install (``uv pip install -e ../rlm-harness``),
which puts the repo ROOT on the consumer's ``sys.path`` via a bare-path ``.pth``. A regular package
(a directory with ``__init__.py``) at the repo root, or nested anywhere outside ``rlm_harness/``:
SHADOWS a consumer's same-named namespace package regardless of ``sys.path`` order (PEP 420: a regular
package at ANY later path entry beats an earlier namespace portion). ``tests/__init__.py`` once did
exactly that to a consumer's namespace ``tests/``, breaking its ``from tests.conftest import ...``
collection. So keep ``rlm_harness`` the ONLY regular package in the repo: ``tests/`` stays a namespace
dir (shared fixtures go in a ``conftest.py``, never an importable ``tests.*`` module backed by an
``__init__.py``). The scan is recursive: a nested ``tests/helpers/__init__.py`` re-creates the same
shadow one level down inside the merged namespace.
"""

from __future__ import annotations

import os
import re
import subprocess
import tomllib
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SKIP = {"__pycache__", "build", "dist", "node_modules"}  # artifacts, never repo package content


def test_rlm_harness_is_the_only_regular_package():
    offenders: list[str] = []
    for dirpath, dirnames, filenames in os.walk(_ROOT):
        rel = Path(dirpath).relative_to(_ROOT)
        # Don't descend into dotdirs (.venv/.git/caches), build artifacts, or the one allowed package.
        dirnames[:] = [
            d for d in dirnames
            if not d.startswith(".") and d not in _SKIP
            and not (rel == Path(".") and d == "rlm_harness")
        ]
        if rel != Path(".") and rel.parts[0] != "rlm_harness" and "__init__.py" in filenames:
            offenders.append(str(rel))
    assert offenders == [], (
        "regular packages outside rlm_harness/ shadow a consumer's namespace package under the editable "
        f"co-dev overlay; make these namespace dirs (drop __init__.py): {offenders}"
    )


#: hatchling's sdist default is every file git does not ignore, so whatever is lying in the working
#: tree goes to PyPI: three stray `tmp*/m.db` test stores reached a commit that way, and a build made
#: while two review worktrees existed would have carried 268 files more. The allowlist in
#: `[tool.hatch.build.targets.sdist]` names the top level instead, so an untracked directory beside
#: `docs/` or `tests/` is no longer a candidate; an untracked stray inside a named one still ships.
#: What the list can do is silently omit something NEW, which is what the three tests below make
#: loud.
def _declared_sdist_paths() -> set[str]:
    """The top-level paths `pyproject.toml` promises the sdist."""
    data = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    include = data["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]
    return {path.lstrip("/") for path in include}


def _tracked_top_level() -> set[str]:
    """The first path component of every file git tracks."""
    try:
        proc = subprocess.run(
            ["git", "ls-files", "-z"], cwd=_ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):  # an unpacked sdist is not a checkout
        pytest.skip("not a git checkout, so there is no list of tracked paths to compare against")
    return {name.split("/", 1)[0] for name in proc.stdout.split("\0") if name}


def test_the_sdist_include_list_covers_every_tracked_top_level_path():
    undeclared = sorted(_tracked_top_level() - _declared_sdist_paths())
    assert not undeclared, (
        f"tracked top-level paths absent from the sdist: {undeclared}. Add each to "
        "[tool.hatch.build.targets.sdist] `include` in pyproject.toml, or delete it from the "
        "repository. An sdist that silently drops a file is what this list costs if it drifts."
    )


def test_every_declared_sdist_path_exists():
    """A path renamed or removed without touching the list ships nothing, silently."""
    missing = sorted(path for path in _declared_sdist_paths() if not (_ROOT / path).exists())
    assert not missing, f"named in the sdist include list but not in the repo: {missing}"


def test_no_tracked_file_is_hidden_from_the_sdist_by_gitignore():
    """The other way a tracked file leaves the sdist silently.

    hatchling applies `.gitignore` on top of the `include` list, and those patterns match at any
    depth (`traces/`, `build/`, `dist/`, `*.py[cod]`), so a tracked file one of them matches is
    dropped from a directory the list names. The likely route is drift in `.gitignore` itself:
    widening a pattern over files that are already tracked needs no `git add` at all, and a
    force-added ignored path gets there too. Neither is a new top-level entry, which is why the
    first test cannot see either one.
    """
    try:
        proc = subprocess.run(
            ["git", "ls-files", "-i", "-c", "--exclude-standard"],
            cwd=_ROOT, capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):  # an unpacked sdist is not a checkout
        pytest.skip("not a git checkout, so there is no list of tracked paths to compare against")
    ignored = sorted(name for name in proc.stdout.splitlines() if name.strip())
    assert not ignored, (
        f"tracked but matched by .gitignore, so the sdist will not carry them: {ignored}. "
        "Either stop tracking the path or narrow the pattern in .gitignore."
    )


def test_the_two_declared_versions_agree():
    """`AGENTS.md` and `docs/CONVENTIONS.md` both state this rule and nothing enforced it.

    `pyproject.toml` `[project].version` is what PyPI publishes; `rlm_harness.__version__` is what
    a consumer reads back at runtime and what a trace records. A release that moves one and not the
    other ships an artifact that misreports itself, and the mismatch is invisible until someone
    compares a trace against an install. Writing them by hand is exactly where it goes wrong.
    """
    declared = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pyproject_version = declared["project"]["version"]

    # Read the literal rather than importing, so this holds without the package's dependencies.
    source = (_ROOT / "rlm_harness" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', source, re.MULTILINE)
    assert match, "rlm_harness/__init__.py has no __version__ = \"...\" literal to compare"

    assert match.group(1) == pyproject_version, (
        f"pyproject.toml says {pyproject_version} and rlm_harness.__version__ says "
        f"{match.group(1)}; bump both or neither."
    )


def _floor_checker():
    """Load the floor check by PATH.

    Deliberately not a package: an `__init__.py` under `.github/` would make it an importable
    top-level module under the editable co-dev overlay, which is the shadowing this file's first
    test exists to prevent.
    """
    import importlib.util

    path = _ROOT / ".github" / "scripts" / "check_declared_floors.py"
    spec = importlib.util.spec_from_file_location("_floor_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_floor_check_reads_every_place_a_floor_can_be_declared():
    """The network half of that check is a scheduled job; THIS half is what breaks in silence.

    If `declared_floors` stops seeing a table, the job keeps passing and simply stops looking, which
    is the failure mode the check was written against in the first place. So pin what it reads:
    base dependencies, every optional-dependency extra, and every dependency group.
    """
    floors = _floor_checker().declared_floors("""
        [project]
        dependencies = ["alpha>=1.2.3", "beta==4.5.6", "gamma"]
        [project.optional-dependencies]
        one = ["delta>=0.1"]
        two = ["epsilon>=2,<3"]
        [dependency-groups]
        dev = ["zeta>=9.9.9"]
    """)
    assert floors == {"alpha": "1.2.3", "delta": "0.1", "epsilon": "2", "zeta": "9.9.9"}, floors


def test_the_floor_check_sees_this_repo_s_own_floors():
    """A parser that works on a fixture and not on the real file is the usual way this rots."""
    floors = _floor_checker().declared_floors(
        (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    # Named rather than counted: a count passes while silently reading one table.
    for name in ("dspy", "pydantic", "mcp", "jsonschema", "claude-agent-sdk", "pytest"):
        assert name in floors, f"{name} has a >= floor in pyproject.toml but the check misses it"
