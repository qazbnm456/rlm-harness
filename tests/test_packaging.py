"""Packaging guard — keep the co-dev editable overlay from shadowing a consumer's tests.

A consumer co-develops rlm-kit by overlaying an editable install (``uv pip install -e ../rlm-kit``),
which puts the repo ROOT on the consumer's ``sys.path`` via a bare-path ``.pth``. A regular package
(a directory with ``__init__.py``) at the repo root — or nested anywhere outside ``rlm_kit/`` —
SHADOWS a consumer's same-named namespace package regardless of ``sys.path`` order (PEP 420: a regular
package at ANY later path entry beats an earlier namespace portion). ``tests/__init__.py`` once did
exactly that to a consumer's namespace ``tests/``, breaking its ``from tests.conftest import ...``
collection. So keep ``rlm_kit`` the ONLY regular package in the repo — ``tests/`` stays a namespace
dir (shared fixtures go in a ``conftest.py``, never an importable ``tests.*`` module backed by an
``__init__.py``). The scan is recursive: a nested ``tests/helpers/__init__.py`` re-creates the same
shadow one level down inside the merged namespace.
"""

from __future__ import annotations

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SKIP = {"__pycache__", "build", "dist", "node_modules"}  # artifacts, never repo package content


def test_rlm_kit_is_the_only_regular_package():
    offenders: list[str] = []
    for dirpath, dirnames, filenames in os.walk(_ROOT):
        rel = Path(dirpath).relative_to(_ROOT)
        # Don't descend into dotdirs (.venv/.git/caches), build artifacts, or the one allowed package.
        dirnames[:] = [
            d for d in dirnames
            if not d.startswith(".") and d not in _SKIP
            and not (rel == Path(".") and d == "rlm_kit")
        ]
        if rel != Path(".") and rel.parts[0] != "rlm_kit" and "__init__.py" in filenames:
            offenders.append(str(rel))
    assert offenders == [], (
        "regular packages outside rlm_kit/ shadow a consumer's namespace package under the editable "
        f"co-dev overlay; make these namespace dirs (drop __init__.py): {offenders}"
    )
