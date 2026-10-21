"""A long input handed to a run as a FILE the sandbox may read, instead of a REPL variable.

**Why this exists, measured.** dspy injects an input field into the REPL as an inline Python
literal, and rebuilds it on every turn. For a multi-megabyte text that costs tens of copies' worth
of allocator and WebAssembly-heap growth: one consumer measured about 66x the payload in peak
process-tree RSS for an 11.6 MB text, and an independent 11-run corpus fits peak tree RSS at
446.8 + 81.4 per corpus MB. Mounting the same bytes as a file instead costs about ONE copy: paired
here against a tiny control, same code path, 318.4 MB against 326.2 MB of peak Deno RSS for an
11.6 MB payload, which is 0.71x the payload.

**What it is NOT.** The bytes still reach the sandbox: dspy's Deno side reads the whole file and
writes it into Pyodide's filesystem, so the payload does materialise once. The saving is the
difference between one copy and tens, not between one and none. A consumer who reported "no
measurable growth" for this approach had in fact passed the HOST path to sandbox code, which
cannot see it (see `sandbox_path` below), so that turn read nothing at all.

**The opening this requires, and why it cannot be reached from outside.** Granting the sandbox read
access to a host file widens the boundary that `docs/INVARIANTS.md` opens with, so every part of
the path is owned here and none of it is reachable by anything a model or an input can influence:

* The path is GENERATED. There is no name, prefix, directory or path argument on this class, so no
  caller value and no model output reaches the filename. The only argument is the text itself, and
  the text becomes the file's CONTENT, never any part of its location.
* The grant covers that ONE file, read-only, and only for a run whose inputs carry one of these
  objects. A task that never takes one is built exactly as before, with no read paths at all.
* A model cannot create one. Constructing this requires host Python; nothing in the REPL, in a
  tool argument, or in a model reply can produce an instance, so the model can neither open a new
  grant nor widen an existing one. It can only read the file the host already decided to give it,
  whose content it would otherwise have received in full as a variable.
* **Where this has actually been exercised.** The staging, the type-only discovery and every
  refusal above are tested offline on every platform CI runs. The MOUNT itself needs Deno, so it
  was exercised by hand on macOS and never in CI, and a Windows host has not been through it: a
  drive letter and backslashes reach Deno's `--allow-read` there and nothing here has watched that
  happen.
* A comma anywhere in the resolved path is REFUSED rather than tolerated. Deno's `--allow-read`
  takes a comma-separated list and silently misparses an entry containing one (dspy's own note,
  and `stanfordnlp/dspy#9749`), which would turn one grant into two of something else. The
  generated basename cannot contain a comma; `$TMPDIR` can, so the assembled path is checked.
"""

from __future__ import annotations

import contextlib
import logging
import os
import secrets
import shutil
import stat
import tempfile
from pathlib import Path
from types import TracebackType
from typing import Self

#: What the sandbox sees. dspy mounts a read path at `/sandbox/<basename>` and the host path is
#: NOT visible inside the sandbox, so this prefix, not the host location, is what model code opens.
SANDBOX_MOUNT_DIR = "/sandbox"

logger = logging.getLogger(__name__)

#: How much of the text the model is shown up front. Enough to recognise the shape of the thing
#: (a log, a diff, a transcript) without paying for the whole of it in the prompt.
PREVIEW_CHARS = 500


class LargeTextRefused(RuntimeError):
    """Raised instead of granting a read path that Deno would misparse."""


class LargeText:
    """A long text staged as a file for the sandbox, addressed by `sandbox_path`.

    Pass one as an input value where the signature declares a `str`. The run reads the file by
    `sandbox_path`, and the kit substitutes that path for the text so the prompt never carries the
    payload. Delete the staged file with `close()`, or use it as a context manager.
    """

    __slots__ = ("_chars", "_dir", "_path")

    def __init__(self, text: str) -> None:
        if not isinstance(text, str):
            raise TypeError(f"LargeText takes a str, not {type(text).__name__}")
        self._chars = len(text)
        # mkdtemp, not a shared directory: the grant is per object, and a directory nothing else
        # writes into cannot have a sibling file appear in it between staging and mounting.
        self._dir = Path(tempfile.mkdtemp(prefix="rlm-largetext-"))
        self._path = self._dir / f"context_{secrets.token_hex(8)}.txt"
        resolved = str(self._path.resolve())
        if "," in resolved:
            shutil.rmtree(self._dir, ignore_errors=True)
            raise LargeTextRefused(
                f"the staged path would contain a comma ({resolved!r}), and Deno's --allow-read "
                "parses its entries as a comma-separated list, so granting it would not grant "
                "this one file. Point $TMPDIR at a directory whose path has no comma."
            )
        self._path.write_text(text, encoding="utf-8")
        os.chmod(self._path, 0o400)

    @property
    def host_path(self) -> str:
        """Where the file is on this machine. Granted to the sandbox; invisible INSIDE it."""
        return str(self._path)

    @property
    def sandbox_path(self) -> str:
        """The only path that works in sandbox code. The host path does not resolve there."""
        return f"{SANDBOX_MOUNT_DIR}/{self._path.name}"

    @property
    def chars(self) -> int:
        return self._chars

    def preview(self, max_chars: int = PREVIEW_CHARS) -> str:
        """The head of the text, for the variable description the model reads."""
        head = self._path.read_text(encoding="utf-8")[: max(0, max_chars)]
        return head

    def close(self) -> None:
        """Delete the staged file. Safe to call twice.

        The staged file is read-only, and on Windows a read-only file cannot be unlinked, so the
        bit is cleared before the remove. `shutil.rmtree(ignore_errors=True)` alone LEFT THE FILE
        THERE and said nothing, which for a file holding the payload is the wrong kind of quiet;
        the Windows CI leg is what opened that question, and only because the assertion above it
        had already failed for a different reason and never reached this.

        Still best-effort, because this runs from `__exit__` where raising would mask whatever
        sent us here, but a leftover is logged rather than swallowed.
        """

        # No `rmtree` error callback: `onexc` is 3.12+ and `onerror` is deprecated there, and
        # this package's floor is 3.11. Clearing the one file's read-only bit before the remove
        # covers the one case that needed a callback.
        with contextlib.suppress(OSError):
            os.chmod(self._path, stat.S_IWRITE | stat.S_IREAD)
        shutil.rmtree(self._dir, ignore_errors=True)
        if self._dir.exists():
            logger.warning(
                "could not delete the staged file at %s; it holds this run's input text", self._dir
            )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def __repr__(self) -> str:  # no content, so a trace or a log cannot leak the payload
        return f"LargeText(chars={self._chars}, sandbox_path={self.sandbox_path!r})"


def stage_inputs(inputs: dict[str, object]) -> tuple[dict[str, object], dict[str, LargeText]]:
    """Split a run's inputs into what dspy receives and which fields were staged as files.

    Discovery is BY TYPE and nothing else. A model reply, a tool result, a string that happens to
    look like a path: none of them can be an instance, so none of them can put a path into the
    grant. A task whose inputs carry no :class:`LargeText` returns the inputs unchanged and an
    empty map, which is the path every existing run takes.

    The value dspy receives is :attr:`LargeText.sandbox_path`, because that is the only path that
    resolves inside the sandbox, so model code can ``open()`` it directly.
    """
    staged = {name: value for name, value in inputs.items() if isinstance(value, LargeText)}
    if not staged:
        return inputs, {}
    clean = dict(inputs)
    for name, value in staged.items():
        clean[name] = value.sandbox_path
    return clean, staged


def staged_instructions(staged: dict[str, LargeText], *, preview_chars: int = PREVIEW_CHARS) -> str:
    """The lines appended to a run's instructions for each field staged as a file.

    Prompt text, so it names the mechanism and what to do with it rather than describing the
    kit: the model is told the variable holds a path, that the file is large, and to read it in
    pieces. The character count is the text's own, measured, not a budget the kit imposes.
    """
    if not staged:
        return ""
    lines = []
    for name, value in sorted(staged.items()):
        head = value.preview(preview_chars).replace("\n", " ")
        lines.append(
            f"`{name}` holds the PATH to a file of {value.chars:,} characters, not the text itself. "
            f"Open it and read the part you need, in slices, rather than reading it whole. "
            f"It begins: {head!r}"
        )
    return "\n".join(lines)
