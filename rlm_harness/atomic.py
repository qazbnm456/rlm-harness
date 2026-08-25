"""``atomic_write_text`` / ``atomic_write_stream`` — write a file such that a concurrent reader
never sees a partial write.

A same-directory temp file + ``fsync`` + ``os.replace`` — the standard "never a half-written file
visible mid-write" idiom, useful for any consumer building a resumable/checkpointed job on top of
``RLMTask`` (a manifest, a cache, any "never let a reader see a torn write" need). dspy-free, stdlib
only.

``atomic_write_stream`` is a SEPARATE, ADDITIVE primitive (not a refactor of ``atomic_write_text``)
for a different shape of caller: one that has an ITERABLE of ``bytes`` chunks (e.g. streamed out of
an archive entry, one chunk at a time) rather than one already-in-memory blob, and wants the
running total bounded and checked as it goes rather than only once at the end. Implemented as its
own function, re-doing the same temp-file/fsync/replace/permission-preservation idiom fresh, to
pose zero regression risk to ``atomic_write_text``'s own already-shipped, already-tested behavior.
"""

from __future__ import annotations

import os
import stat
import tempfile
from collections.abc import Iterable


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


class _ExtractionBudgetExceeded(OSError):
    """Raised by :func:`atomic_write_stream` when the running total of written bytes exceeds
    ``max_bytes``. A DEDICATED subclass, not a bare ``OSError`` — a caller that needs to tell
    "budget exceeded" apart from an ordinary ``OSError`` raised by whatever produced the chunks
    (e.g. a corrupted compressed stream) can catch this specifically, ahead of a broader catch
    that also matches plain ``OSError``. Still an ``OSError`` itself, so any caller that only
    catches the base type is unaffected.

    Module-level, not exported in ``__all__`` — a consumer of ``atomic_write_stream`` catches
    plain ``OSError`` like any other write failure; this specific subclass exists for
    ``rlm_harness.tools.archive``'s own Pass 2 to import and catch by name, ahead of its own
    broader archive-error catch, matching this kit's existing precedent for an internal-but-
    cross-module-imported name (``fs.py``'s ``_validate_tool_name``, imported by ``edit.py``).
    """


def atomic_write_stream(
    path: str, chunks: Iterable[bytes], *, max_bytes: int | None = None
) -> int:
    """Write the concatenation of ``chunks`` to ``path`` atomically — same same-directory temp
    file + ``fsync`` + ``os.replace`` idiom as :func:`atomic_write_text`, same
    permission-preservation-on-overwrite behavior, same directory-creation behavior —
    implemented as its own, separate function (not a refactor of ``atomic_write_text``) to avoid
    any regression risk to that already-shipped primitive.

    Aborts (raises :class:`_ExtractionBudgetExceeded`, an ``OSError`` subclass; the temp file is
    removed, ``path`` itself is never touched) the moment the running total of bytes written
    exceeds ``max_bytes`` (default ``None`` = unbounded) — checked after EVERY chunk, not merely
    once at the end, so the caller controls the memory/overshoot bound entirely via how large
    each chunk is. Returns the total number of bytes written on success.
    """
    dirname = os.path.dirname(path) or "."
    os.makedirs(dirname, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dirname, prefix=".tmp-")
    try:
        written = 0
        with os.fdopen(fd, "wb") as fh:
            for chunk in chunks:
                fh.write(chunk)
                written += len(chunk)
                if max_bytes is not None and written > max_bytes:
                    raise _ExtractionBudgetExceeded(
                        f"{path!r}: writing exceeded the {max_bytes}-byte budget"
                    )
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
    return written
