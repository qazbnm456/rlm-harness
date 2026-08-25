"""``make_extract_archive_tool`` — safe zip/tar extraction into a bounded local directory.

Python's ``zipfile.extractall()``/``tarfile.extractall()`` are not safe by default: a malicious
archive entry can carry an absolute path, a ``..``-traversal path, or (tar) a symlink/hardlink
pointing outside the extraction target — "zip slip," a well-known vulnerability class. A task
that fetches an archive (``fetch_url`` + ``write_file``, or the model is simply handed one) and
wants to unpack it needs a safe way to do so. This mirrors ``resolve_within_root``'s exact
reasoning, applied to archive ENTRIES rather than a single path argument.

**A REPL tool, not a host-side helper** (unlike ``list_candidate_paths``) — extraction is
something a model plausibly does MID-TRAJECTORY (it fetched or was handed an archive as part of
the task), matching ``write_file``/``edit_file``'s own "mutates the filesystem, needs tracing"
category.

**Format dispatch, by extension, both stdlib**: ``.zip`` → ``zipfile.ZipFile``; a recognized
tar-shaped extension (``.tar``, ``.tar.gz``, ``.tgz``, ``.tar.bz2``, ``.tbz2``, ``.tar.xz``,
``.txz``) → ``tarfile.open(path, mode="r:*")``, whose auto-detect mode identifies the actual
compression from content rather than trusting the extension. An unrecognized extension returns an
error string, never raises, never attempts to sniff content type.

**Two-pass extraction — validate everything before writing anything**, matching this kit's
"refuse outright, never partially mutate" posture: Pass 1 inspects every entry's METADATA ONLY
(name, type, declared size, and — zip-only — the encryption/compression-method header fields) and
refuses the WHOLE operation upfront on any violation; Pass 2 (only reached once Pass 1 fully
passes) streams each entry's real bytes in bounded chunks via :func:`rlm_harness.atomic.
atomic_write_stream`, so peak memory for any single entry stays bounded by ``_CHUNK_SIZE``
regardless of that entry's own size.

**A declared size cannot be used to smuggle more decompressed output than it promises** — the
declared ``file_size``/``size`` field is a HARD CEILING on how much either stdlib read API can
ever return (``ZipExtFile.read()`` forces end-of-file once the declared size is reached, then
checks the CRC; ``TarFile.extractfile()``'s reader is bounded the same way by construction).
Pass 1's cumulative declared-size check is therefore what actually prevents a decompression-bomb-
shaped archive from being accepted — the streaming design in Pass 2 exists for a separate,
still-real reason: bounding PEAK MEMORY to a small, fixed constant while extracting any single
entry, rather than materializing a whole entry (which could legitimately be a large fraction of
the overall budget) in memory at once.

**Exception handling: one shared, broad-but-bounded tuple, wrapped at every point this design
calls into the archive/compression machinery** (the initial open, Pass 1's entry-enumeration
loop, and Pass 2's per-entry read) rather than chasing individual header fields or exception
types one at a time. ``zipfile``/``tarfile`` internally delegate decompression to ``zlib``/
``bz2``/``lzma``, and neither module translates or catches those libraries' own exception types
on the way through — so a structurally intact archive with a corrupted (not merely truncated)
compressed PAYLOAD reaches this exact tuple, not just a malformed header. The budget-exceeded
signal from ``atomic_write_stream`` is its own dedicated
:class:`rlm_harness.atomic._ExtractionBudgetExceeded` (an ``OSError`` subclass), caught FIRST and
separately, so it can never collide with — or be misreported as — a plain ``OSError`` a
compression library itself raises for corrupted data (confirmed reachable via ``bz2``, which,
unlike ``zlib``/``lzma``, raises a bare ``OSError`` rather than a dedicated exception type).

**No nested-archive recursion** — an archive found INSIDE the extracted output is not itself
auto-extracted; the model calls this tool again on it, subject to the identical safety checks a
second time.

**Unconditional overwrite of existing files at the destination** — same posture
``make_write_file_tool`` already takes (no create-only mode).

**No support for password-protected/encrypted archives** — refused upfront, in Pass 1, with a
clear reason (the entry's own header flag bits), never a crash.
"""

from __future__ import annotations

import lzma
import os
import stat
import tarfile
import zipfile
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..atomic import _ExtractionBudgetExceeded, atomic_write_stream
from ..trace import record_tool_call
from .fs import _validate_tool_name, resolve_within_root

_CHUNK_SIZE = 1024 * 1024  # 1 MiB -- bounds the in-memory working set per entry to this size

_TAR_EXTENSIONS = (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")

_SUPPORTED_ZIP_COMPRESSION = frozenset(
    {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED, zipfile.ZIP_BZIP2, zipfile.ZIP_LZMA}
)

# Every point this design calls into the archive/compression machinery is wrapped in this ONE
# shared tuple -- see the module docstring for why this is structural, not per-field. `OSError`
# is included because `bz2` raises a bare `OSError` for corrupted payload data (unlike `zlib`/
# `lzma`, which have their own dedicated types) -- safe to include alongside
# `_ExtractionBudgetExceeded` (itself an `OSError` subclass) ONLY because that budget signal is
# caught FIRST, in its own separate `except` clause, ahead of this tuple -- see `_pass2_extract`.
_ARCHIVE_ERROR_TYPES = (
    zipfile.BadZipFile,
    tarfile.TarError,
    EOFError,
    RuntimeError,
    NotImplementedError,
    UnicodeError,
    zlib.error,
    lzma.LZMAError,
    OSError,
)


def _detect_format(path: str) -> str | None:
    lower = path.lower()
    if lower.endswith(".zip"):
        return "zip"
    if any(lower.endswith(ext) for ext in _TAR_EXTENSIONS):
        return "tar"
    return None


@dataclass(frozen=True)
class _SafeEntry:
    """One Pass-1-approved entry, ready for Pass 2 — deliberately format-agnostic (``raw`` is
    a ``zipfile.ZipInfo`` or ``tarfile.TarInfo``, dispatched on ``fmt``) so Pass 2 can stay a
    single, small loop rather than two near-duplicate ones."""

    fmt: str  # "zip" or "tar"
    raw: Any
    target: str
    is_dir: bool


def _entry_name(entry: _SafeEntry) -> str:
    return entry.raw.filename if entry.fmt == "zip" else entry.raw.name


def _pass1_validate(
    archive: Any, fmt: str, dest_resolved: str, max_entries: int, max_extracted_bytes: int
) -> tuple[list[_SafeEntry], str | None]:
    """Metadata-only validation of EVERY entry — this function itself never deliberately opens or
    reads a single byte of payload (unlike Pass 2, which streams real content). **For a
    COMPRESSED tar, this is not quite the same as "zero payload bytes touched" in practice**: the
    underlying ``tarfile`` iteration decompresses-and-discards each prior member's payload as a
    side effect of seeking to the next header (see the module docstring's Pass-1-cost
    disclosure) — which is also why this loop is wrapped in ``_ARCHIVE_ERROR_TYPES`` rather than
    assumed exception-free. Returns ``(safe_entries, None)`` on success, or ``([],
    refusal_message)`` the moment any entry fails any check — refuse the WHOLE operation, never a
    partial list. Callable and testable independent of Pass 2 (see :func:`_pass2_extract`)."""
    safe_entries: list[_SafeEntry] = []
    total_declared = 0

    # zip's central directory is already fully materialized by the constructor (cheap, no
    # decompression) -- `infolist()` is a plain list. tar has no random-access central directory;
    # iterating the `TarFile` object itself (rather than calling `getmembers()`) yields members
    # ONE AT A TIME as the outer stream is sequentially decompressed, which is what lets the
    # max_entries/max_extracted_bytes checks below abort the decompression early instead of
    # always paying the full cost first.
    raw_iter = archive.infolist() if fmt == "zip" else archive

    for count, raw in enumerate(raw_iter, start=1):
        if fmt == "zip":
            raw_name = raw.filename
            declared_size = raw.file_size
            is_dir = raw.is_dir()
            mode = raw.external_attr >> 16
            is_special = bool(
                mode
                and (
                    stat.S_ISLNK(mode)
                    or stat.S_ISFIFO(mode)
                    or stat.S_ISCHR(mode)
                    or stat.S_ISBLK(mode)
                    or stat.S_ISSOCK(mode)
                )
            )
            if raw.flag_bits & 0x1:
                return [], (
                    f"Refused: {raw_name!r} is encrypted — password-protected archives are not "
                    "supported."
                )
            if raw.flag_bits & 0x60:
                return [], (
                    f"Refused: {raw_name!r} uses an unsupported compressed-patched-data/"
                    "strong-encryption flag."
                )
            if raw.compress_type not in _SUPPORTED_ZIP_COMPRESSION:
                return [], (
                    f"Refused: {raw_name!r} uses an unsupported compression method "
                    f"({raw.compress_type!r})."
                )
        else:
            raw_name = raw.name
            declared_size = raw.size
            is_dir = raw.isdir()
            is_special = raw.issym() or raw.islnk() or raw.isdev()

        normalized = raw_name.replace("\\", "/")
        if normalized in ("", "."):
            return [], "Refused: an archive entry has an empty or '.' name."
        if is_special:
            return [], (
                f"Refused: {raw_name!r} is a symlink, hardlink, device, or other non-regular, "
                "non-directory entry."
            )

        target = resolve_within_root(dest_resolved, normalized)
        if target is None:
            return [], f"Refused: {raw_name!r} would extract outside the destination directory."

        total_declared += declared_size
        if count > max_entries:
            return [], f"Refused: archive has more than {max_entries} entries."
        if total_declared > max_extracted_bytes:
            return [], (
                f"Refused: archive's declared total size exceeds the "
                f"{max_extracted_bytes}-byte budget."
            )

        safe_entries.append(_SafeEntry(fmt=fmt, raw=raw, target=target, is_dir=is_dir))

    return safe_entries, None


def _pass2_extract(
    archive: Any, fmt: str, safe_entries: list[_SafeEntry], max_extracted_bytes: int
) -> tuple[int, int, str | None]:
    """Real extraction, streaming each entry in bounded chunks. Takes ``safe_entries`` as a plain
    argument, independent of ``_pass1_validate`` — a test can construct one directly (bypassing
    Pass 1 entirely) to prove this function's own exception-handling backstop works on its own.
    Returns ``(written_bytes, written_count, None)`` on success, or ``(partial_bytes,
    partial_count, failure_message)`` the moment one entry fails — entries already written before
    the failing one stay on disk, a disclosed, accepted partial-extraction outcome."""
    written_bytes = 0
    written_count = 0

    for entry in safe_entries:
        if entry.is_dir:
            os.makedirs(entry.target, exist_ok=True)
            continue

        remaining_budget = max_extracted_bytes - written_bytes
        try:
            src_cm = archive.open(entry.raw) if fmt == "zip" else archive.extractfile(entry.raw)
            with src_cm as src:

                def _chunks() -> Any:
                    while True:
                        chunk = src.read(_CHUNK_SIZE)
                        if not chunk:
                            return
                        yield chunk

                written = atomic_write_stream(entry.target, _chunks(), max_bytes=remaining_budget)
        except _ExtractionBudgetExceeded:
            # Caught FIRST, specifically — a DEDICATED type, not a bare OSError, so it can never
            # collide with the plain OSError bz2 itself raises for corrupted payload data, which
            # falls through to the broader `_ARCHIVE_ERROR_TYPES` catch just below instead.
            return written_bytes, written_count, (
                f"Refused mid-extraction: {_entry_name(entry)!r} would exceed the "
                f"{max_extracted_bytes}-byte extraction budget."
            )
        except _ARCHIVE_ERROR_TYPES as exc:
            return written_bytes, written_count, (
                f"Archive error while extracting {_entry_name(entry)!r}: "
                f"{type(exc).__name__}: {exc}"
            )
        written_bytes += written
        written_count += 1

    return written_bytes, written_count, None


def make_extract_archive_tool(
    root: str,
    *,
    name: str = "extract_archive",
    max_extracted_bytes: int = 200 * 1024 * 1024,
    max_entries: int = 10_000,
) -> Callable[..., str]:
    """Build an ``extract_archive``-shaped tool scoped to ``root`` — wired in a task's
    ``__init__`` (per-run state, never a classvar).

    ``name`` (default ``"extract_archive"``): same rationale and mechanism as
    :func:`rlm_harness.tools.make_read_file_tool`'s ``name``. Validated at factory-build time.

    ``max_extracted_bytes`` (default ``200 MiB``)/``max_entries`` (default ``10_000``): factory
    (operator) parameters, never model-controlled — same posture ``make_grep_files_tool``'s
    ``per_match_timeout_s``/``max_total_time_s`` and ``list_candidate_paths``'s ``max_files``
    already take. Checked cumulatively across every entry BEFORE any extraction begins (see the
    module docstring) — the declared-size sum, not merely each entry individually, is what
    actually bounds a decompression-bomb-shaped archive.
    """
    _validate_tool_name(name)

    def extract_archive(archive_path: str, dest_dir: str = ".") -> str:
        """Extract every entry of the archive at ``archive_path`` (relative to the root) into
        ``dest_dir`` (relative to the root, default ``"."``). Supports ``.zip`` and tar variants
        (``.tar``, ``.tar.gz``/``.tgz``, ``.tar.bz2``/``.tbz2``, ``.tar.xz``/``.txz``). Returns a
        "Refused"/error string (never raises) for a path escaping the root, an unsupported
        extension, a corrupt/encrypted/oversized archive, or any entry that would land outside
        ``dest_dir``. Nothing is extracted unless every entry passes validation first."""
        resolved_archive = resolve_within_root(root, archive_path)
        if resolved_archive is None:
            record_tool_call(
                name, args={"archive_path": archive_path}, ok=False,
                note="refused: archive_path escapes root",
            )
            return f"Refused: {archive_path!r} is not a path inside this root."

        dest_resolved = resolve_within_root(root, dest_dir)
        if dest_resolved is None:
            record_tool_call(
                name, args={"dest_dir": dest_dir}, ok=False, note="refused: dest_dir escapes root"
            )
            return f"Refused: {dest_dir!r} is not a path inside this root."

        fmt = _detect_format(resolved_archive)
        if fmt is None:
            record_tool_call(
                name, args={"archive_path": archive_path}, ok=False, note="unsupported extension"
            )
            return (
                f"Unsupported archive extension for {archive_path!r} "
                "(expected .zip or a tar variant)."
            )

        try:
            # Not opened via `with` here on purpose -- the object must stay a plain variable
            # (either a ZipFile or a TarFile) reused across both Pass 1 and Pass 2 below, closed
            # by the single `with archive:` block that wraps both passes.
            if fmt == "zip":
                archive = zipfile.ZipFile(resolved_archive)
            else:
                archive = tarfile.open(resolved_archive, mode="r:*")  # noqa: SIM115
        except _ARCHIVE_ERROR_TYPES as exc:
            record_tool_call(
                name, args={"archive_path": archive_path}, ok=False,
                note=f"error: {type(exc).__name__}",
            )
            return f"Archive error opening {archive_path!r}: {type(exc).__name__}: {exc}"

        with archive:
            try:
                safe_entries, refusal = _pass1_validate(
                    archive, fmt, dest_resolved, max_entries, max_extracted_bytes
                )
            except _ARCHIVE_ERROR_TYPES as exc:
                record_tool_call(
                    name, args={"archive_path": archive_path}, ok=False,
                    note=f"error: {type(exc).__name__}",
                )
                return f"Archive error reading {archive_path!r}: {type(exc).__name__}: {exc}"

            if refusal is not None:
                record_tool_call(name, args={"archive_path": archive_path}, ok=False, note=refusal)
                return refusal

            written_bytes, written_count, failure = _pass2_extract(
                archive, fmt, safe_entries, max_extracted_bytes
            )

        if failure is not None:
            record_tool_call(name, args={"archive_path": archive_path}, ok=False, note=failure)
            return failure

        record_tool_call(
            name,
            args={"archive_path": archive_path, "dest_dir": dest_dir},
            ok=True,
            files_extracted=written_count,
            bytes_extracted=written_bytes,
        )
        return f"Extracted {written_count} file(s) ({written_bytes} bytes) to {dest_dir!r}."

    extract_archive.__name__ = name
    extract_archive.__qualname__ = name
    return extract_archive
