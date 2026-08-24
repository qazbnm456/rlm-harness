"""``atomic_write_text`` — write a file such that a concurrent reader never sees a partial write.

A same-directory temp file + ``fsync`` + ``os.replace`` — the standard "never a half-written file
visible mid-write" idiom, useful for any consumer building a resumable/checkpointed job on top of
``RLMTask`` (a manifest, a cache, any "never let a reader see a torn write" need). dspy-free, stdlib
only.
"""

from __future__ import annotations

import os
import stat
import tempfile


def atomic_write_text(path: str, text: str, *, encoding: str = "utf-8") -> None:
    """Write ``text`` to ``path`` atomically: a same-directory temp file, ``fsync``, then
    ``os.replace`` — never a window where a concurrent reader sees a partial file.

    ``os.makedirs(dirname, exist_ok=True)`` creates the destination directory first if needed.
    ``dirname = os.path.dirname(path) or "."`` — load-bearing, not cosmetic:
    ``os.path.dirname("checkpoint.json")`` is ``""`` for a bare relative filename (an entirely
    ordinary way to call this, e.g. from a script whose cwd already is the target directory), and
    ``os.makedirs("", exist_ok=True)`` raises ``FileNotFoundError``. ``os.makedirs(".",
    exist_ok=True)`` is a safe, correct no-op when the target directory is already the cwd.

    **Preserves the destination's existing permission bits across an overwrite.**
    ``tempfile.mkstemp`` always creates its temp file at mode ``0600`` regardless of umask, and
    ``os.replace`` does NOT carry the destination's mode across — so without this, overwriting an
    existing file through this function would silently reset it to ``0600`` (confirmed
    empirically: a ``0o755`` script loses its executable bit). If ``path`` already exists, its
    mode is read via ``os.stat`` and applied to the temp file (``os.chmod``) BEFORE the final
    ``os.replace`` — so there is never a window where the file at the final path is visible with
    the wrong mode. A narrow stat-to-replace race (the destination's permissions changing in that
    exact window) is not solved and not worth solving for this: fail in the safe, common-case
    direction rather than over-engineer a race no caller of this function is exposed to in
    practice. If ``path`` doesn't exist yet, there's nothing to preserve — the temp file's own
    mode is used as-is.

    On any exception during the write, the temp file is best-effort removed and the exception is
    re-raised — ``path`` itself is never touched until the final, atomic ``os.replace``.
    """
    dirname = os.path.dirname(path) or "."
    os.makedirs(dirname, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dirname, prefix=".tmp-")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            existing_mode = os.stat(path).st_mode
        except FileNotFoundError:
            pass  # nothing to preserve — the temp file's own mode is fine
        else:
            os.chmod(tmp_path, stat.S_IMODE(existing_mode))
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
