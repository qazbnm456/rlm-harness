"""Guards that have to run before any test writes its first file."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


#: `tempfile` reads `$TMPDIR`, `$TEMP` and `$TMP`, then the platform's own temp locations, and its
#: candidate list ends with the CURRENT DIRECTORY. On a machine where nothing earlier in that list
#: can be written, every `TemporaryDirectory()` in this suite lands in the repository instead.
#: pytest's own `tmp_path` reads the same `gettempdir()`, so using the fixture everywhere would not
#: help. It happened once and left no trace while it ran: three `tmp*/m.db` stores reached a commit,
#: from the `TemporaryDirectory()` blocks in `tests/test_tool_descriptions.py` that hand a
#: `MemoryStore` a path under them, and hatchling put them in an sdist, because nothing in the suite
#: had any reason to look. One session-start check covers every temp consumer, including pytest's;
#: a `.gitignore` entry would only have hidden the next one.
#:
#: Scoped to a checkout, because what this prevents is a stray file reaching a commit. An unpacked
#: sdist carries this file too, and a build system that points `$TMPDIR` inside the source it
#: unpacked has nothing to commit a stray into, so it gets its test run.
def pytest_sessionstart(session: pytest.Session) -> None:
    repo = Path(__file__).resolve().parent.parent
    if not (repo / ".git").exists():
        return
    temp_dir = Path(tempfile.gettempdir()).resolve()
    if temp_dir.is_relative_to(repo):
        raise pytest.UsageError(
            f"tempfile.gettempdir() resolves to {temp_dir}, inside the repository at {repo}. "
            "Every temporary file this suite writes would be created in the working tree. "
            "Point $TMPDIR at a writable directory outside the repository and run again."
        )
