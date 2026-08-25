"""make_extract_archive_tool -- safe zip/tar extraction. All offline, dspy-free (real zipfile/
tarfile/zlib/bz2/lzma stdlib archives, no network)."""
from __future__ import annotations

import io
import os
import shutil
import stat
import struct
import subprocess
import tarfile
import zipfile

import pytest

from rlm_harness.atomic import _ExtractionBudgetExceeded, atomic_write_stream
from rlm_harness.testing import assert_repl_safe, assert_task_repl_safe
from rlm_harness.tools import make_extract_archive_tool
from rlm_harness.tools.archive import _pass1_validate, _pass2_extract, _SafeEntry
from rlm_harness.trace import EVENT_TOOL_CALL, TraceRecorder, load_events

# ---- archive-building helpers ---------------------------------------------------------------


def _make_zip(path, entries, compress_type=zipfile.ZIP_DEFLATED):
    """`entries`: a list of (name, bytes) pairs, or (name, bytes, is_dir) triples."""
    with zipfile.ZipFile(path, "w", compression=compress_type) as zf:
        for entry in entries:
            name, data = entry[0], entry[1]
            if name == "":
                # An empty name is exactly what the empty-name refusal test needs to build, but
                # CPython 3.11's ZipFile.writestr() detects a trailing "/" with `filename[-1]`
                # and so raises IndexError on an empty name before writing anything (3.12+ uses
                # `filename.endswith("/")` and accepts it). Handing writestr an explicit ZipInfo
                # skips that branch on every version; compress_type is carried over so the entry
                # still matches what the string path would have produced.
                info = zipfile.ZipInfo(filename=name)
                info.compress_type = compress_type
                zf.writestr(info, data)
            else:
                zf.writestr(name, data)


def _make_tar(path, entries, mode="w:gz"):
    with tarfile.open(path, mode) as tf:
        for name, data in entries:
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


def _add_zip_symlink(path, name, target):
    with zipfile.ZipFile(path, "a") as zf:
        zi = zipfile.ZipInfo(name)
        zi.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(zi, target)


def _add_tar_symlink(path, name, target, mode="w:gz"):
    # tarfile can't append to a compressed archive -- rebuild.
    entries_path = path + ".rebuild"
    with tarfile.open(path, "r:*") as src, tarfile.open(entries_path, mode) as dst:
        for member in src.getmembers():
            fh = src.extractfile(member) if member.isfile() else None
            dst.addfile(member, fh)
        link_info = tarfile.TarInfo(name=name)
        link_info.type = tarfile.SYMTYPE
        link_info.linkname = target
        dst.addfile(link_info)
    os.replace(entries_path, path)


def _patch_zip_compress_type(raw: bytes, new_type: int) -> bytes:
    data = bytearray(raw)
    i = data.find(b"PK\x03\x04")
    struct.pack_into("<H", data, i + 8, new_type)
    j = data.find(b"PK\x01\x02")
    struct.pack_into("<H", data, j + 10, new_type)
    return bytes(data)


def _lie_zip_declared_size(raw: bytes, new_size: int) -> bytes:
    """Patches ONLY the central directory's uncompressed/compressed size fields (what
    `zf.infolist()` reports, and what `ZipExtFile` uses as its read ceiling) -- leaves the local
    header and the real data bytes untouched, reproducing a genuine "declared size lies" archive
    without needing a from-scratch binary writer."""
    data = bytearray(raw)
    j = data.find(b"PK\x01\x02")
    struct.pack_into("<II", data, j + 20, new_size, new_size)  # compressed, uncompressed
    return bytes(data)


def _corrupt_zip_payload(raw: bytes, frac: float = 0.1, nbytes: int = 8) -> bytes:
    """Flips `nbytes` inside the FIRST entry's compressed payload, at `frac` of the way through
    it -- a structurally-intact archive with a corrupted (not truncated) payload, the exact shape
    that reaches zlib/bz2/lzma's own exception types rather than a header-level BadZipFile."""
    data = bytearray(raw)
    i = data.find(b"PK\x03\x04")
    fname_len, extra_len = struct.unpack_from("<HH", data, i + 26)
    comp_size = struct.unpack_from("<I", data, i + 18)[0]
    data_start = i + 30 + fname_len + extra_len
    pos = data_start + int(comp_size * frac)
    for k in range(nbytes):
        data[pos + k] ^= 0xFF
    return bytes(data)


def _corrupt_tar_xz_mid_stream(raw: bytes, frac: float = 0.6, nbytes: int = 8) -> bytes:
    """Flips bytes well past the outer .xz stream's own header/magic (so `tarfile.open`'s
    format auto-detection still succeeds), landing inside a non-final member's compressed
    payload -- surfaces during Pass 1's OWN member-enumeration loop, not just Pass 2's read."""
    data = bytearray(raw)
    pos = int(len(data) * frac)
    for k in range(nbytes):
        data[pos + k] ^= 0xFF
    return bytes(data)


def _read_bytes(path):
    with open(path, "rb") as fh:
        return fh.read()


def _write_bytes(path, data):
    with open(path, "wb") as fh:
        fh.write(data)


_HAS_ZIP_CLI = shutil.which("zip") is not None


# ---- correctness ---------------------------------------------------------------------------


def test_extracts_a_well_formed_zip(tmp_path):
    archive = tmp_path / "a.zip"
    _make_zip(archive, [("top.txt", b"hello"), ("nested/dir/deep.txt", b"world")])
    tool = make_extract_archive_tool(str(tmp_path))
    result = tool("a.zip", "out")
    assert result == "Extracted 2 file(s) (10 bytes) to 'out'."
    assert _read_bytes(tmp_path / "out" / "top.txt") == b"hello"
    assert _read_bytes(tmp_path / "out" / "nested" / "dir" / "deep.txt") == b"world"


def test_extracts_a_well_formed_tar_gz(tmp_path):
    archive = tmp_path / "a.tar.gz"
    _make_tar(archive, [("top.txt", b"hello"), ("nested/deep.txt", b"world")])
    tool = make_extract_archive_tool(str(tmp_path))
    result = tool("a.tar.gz", "out")
    assert result == "Extracted 2 file(s) (10 bytes) to 'out'."
    assert _read_bytes(tmp_path / "out" / "top.txt") == b"hello"
    assert _read_bytes(tmp_path / "out" / "nested" / "deep.txt") == b"world"


# ---- zip-slip / symlink refusal, nothing extracted ----------------------------------------


@pytest.mark.parametrize("bad_name", ["../../etc/passwd", "/etc/passwd"])
def test_zip_slip_refused_nothing_extracted(tmp_path, bad_name):
    archive = tmp_path / "a.zip"
    _make_zip(archive, [(bad_name, b"pwned")])
    dest = tmp_path / "out"
    dest.mkdir()
    tool = make_extract_archive_tool(str(tmp_path))
    result = tool("a.zip", "out")
    assert result.startswith("Refused:")
    assert os.listdir(dest) == []


@pytest.mark.parametrize("bad_name", ["../../etc/passwd", "/etc/passwd"])
def test_tar_slip_refused_nothing_extracted(tmp_path, bad_name):
    archive = tmp_path / "a.tar.gz"
    _make_tar(archive, [(bad_name, b"pwned")])
    dest = tmp_path / "out"
    dest.mkdir()
    tool = make_extract_archive_tool(str(tmp_path))
    result = tool("a.tar.gz", "out")
    assert result.startswith("Refused:")
    assert os.listdir(dest) == []


def test_tar_symlink_refused_nothing_extracted(tmp_path):
    archive = tmp_path / "a.tar.gz"
    _make_tar(archive, [("safe.txt", b"safe")])
    _add_tar_symlink(str(archive), "evil_link", "../../etc/passwd")
    dest = tmp_path / "out"
    dest.mkdir()
    tool = make_extract_archive_tool(str(tmp_path))
    result = tool("a.tar.gz", "out")
    assert result.startswith("Refused:")
    assert os.listdir(dest) == []


def test_zip_symlink_via_external_attr_refused_nothing_extracted(tmp_path):
    archive = tmp_path / "a.zip"
    _make_zip(archive, [("safe.txt", b"safe")])
    _add_zip_symlink(str(archive), "evil_link", "../../etc/passwd")
    dest = tmp_path / "out"
    dest.mkdir()
    tool = make_extract_archive_tool(str(tmp_path))
    result = tool("a.zip", "out")
    assert result.startswith("Refused:")
    assert os.listdir(dest) == []


# ---- encrypted / unsupported compression method ---------------------------------------------


@pytest.mark.skipif(not _HAS_ZIP_CLI, reason="requires the system 'zip' CLI")
def test_real_password_protected_zip_refused_before_open(tmp_path):
    plain_dir = tmp_path / "plain"
    plain_dir.mkdir()
    (plain_dir / "secret.txt").write_text("shh")
    archive = tmp_path / "a.zip"
    subprocess.run(
        ["zip", "-e", "-P", "hunter2", str(archive), "secret.txt"],
        cwd=str(plain_dir), check=True, capture_output=True,
    )
    tool = make_extract_archive_tool(str(tmp_path))
    result = tool("a.zip", "out")
    assert result.startswith("Refused:")
    assert "encrypted" in result
    assert not (tmp_path / "out").exists() or os.listdir(tmp_path / "out") == []


def test_zip_unsupported_compress_type_refused_in_pass1(tmp_path):
    archive = tmp_path / "a.zip"
    _make_zip(archive, [("bad.bin", b"x" * 100)], compress_type=zipfile.ZIP_STORED)
    patched = _patch_zip_compress_type(_read_bytes(archive), 1)  # PKZIP "shrink" -- unsupported
    _write_bytes(archive, patched)
    dest = tmp_path / "out"
    dest.mkdir()
    tool = make_extract_archive_tool(str(tmp_path))
    result = tool("a.zip", "out")
    assert result.startswith("Refused:")
    assert os.listdir(dest) == []


def test_pass2_backstop_catches_unsupported_compress_type_when_pass1_is_bypassed(tmp_path):
    # Direct unit test of Pass 2 in isolation -- proves the defense-in-depth catch works even
    # with Pass 1's own compress_type check never having run.
    archive = tmp_path / "a.zip"
    _make_zip(archive, [("bad.bin", b"x" * 100)], compress_type=zipfile.ZIP_STORED)
    _write_bytes(archive, _patch_zip_compress_type(_read_bytes(archive), 1))
    with zipfile.ZipFile(archive) as zf:
        raw = zf.infolist()[0]
        target = str(tmp_path / "out" / "bad.bin")
        safe_entries = [_SafeEntry(fmt="zip", raw=raw, target=target, is_dir=False)]
        _written_bytes, written_count, failure = _pass2_extract(zf, "zip", safe_entries, 10_000)
    assert written_count == 0
    assert failure is not None
    assert "Archive error" in failure
    assert not os.path.exists(target)


# ---- corrupted (not truncated) payload -- Revision 6 regression tests -----------------------


def test_zip_deflate_corrupted_payload_reports_archive_error_not_budget(tmp_path):
    archive = tmp_path / "a.zip"
    payload = b"the quick brown fox jumps over the lazy dog " * 500
    _make_zip(archive, [("f.bin", payload)], compress_type=zipfile.ZIP_DEFLATED)
    _write_bytes(archive, _corrupt_zip_payload(_read_bytes(archive), frac=0.1, nbytes=8))
    tool = make_extract_archive_tool(str(tmp_path))
    result = tool("a.zip", "out")
    assert result.startswith("Archive error")
    assert "extraction budget" not in result


def test_zip_lzma_corrupted_payload_reports_archive_error(tmp_path):
    archive = tmp_path / "a.zip"
    payload = b"the quick brown fox jumps over the lazy dog " * 200
    _make_zip(archive, [("f.bin", payload)], compress_type=zipfile.ZIP_LZMA)
    _write_bytes(archive, _corrupt_zip_payload(_read_bytes(archive), frac=0.5, nbytes=4))
    tool = make_extract_archive_tool(str(tmp_path))
    result = tool("a.zip", "out")
    assert result.startswith("Archive error")


def test_zip_bzip2_corrupted_payload_reports_archive_error_not_budget(tmp_path):
    # The direct regression test for the Revision 6 diagnostic-mislabeling finding: bz2 raises a
    # PLAIN OSError for corrupted data, which -- before the _ExtractionBudgetExceeded fix -- was
    # indistinguishable from (and misreported as) the budget-exceeded signal.
    archive = tmp_path / "a.zip"
    payload = b"the quick brown fox jumps over the lazy dog " * 200
    _make_zip(archive, [("f.bin", payload)], compress_type=zipfile.ZIP_BZIP2)
    _write_bytes(archive, _corrupt_zip_payload(_read_bytes(archive), frac=0.5, nbytes=4))
    tool = make_extract_archive_tool(str(tmp_path))
    result = tool("a.zip", "out")
    assert result.startswith("Archive error")
    assert "extraction budget" not in result


def test_tar_xz_corrupted_mid_stream_caught_during_pass1_enumeration(tmp_path):
    archive = tmp_path / "a.tar.xz"
    entries = [(f"m{i}.txt", (f"member {i} " * 5000).encode()) for i in range(5)]
    _make_tar(archive, entries, mode="w:xz")
    corrupted = _corrupt_tar_xz_mid_stream(_read_bytes(archive), frac=0.6, nbytes=8)
    _write_bytes(archive, corrupted)
    tool = make_extract_archive_tool(str(tmp_path))
    result = tool("a.tar.xz", "out")
    assert result.startswith("Archive error")


def test_budget_exceeded_still_correctly_reported_not_confused_with_archive_error(tmp_path):
    # Direct unit test of Pass 2's backstop: an intentionally tiny max_extracted_bytes passed
    # straight to _pass2_extract (bypassing Pass 1's own, always-sufficient cumulative check) --
    # confirms _ExtractionBudgetExceeded's dedicated type keeps this case correctly distinguished
    # from the plain-OSError bz2-corruption case above, now that both flow through code that
    # watches for OSError-shaped failures.
    archive = tmp_path / "a.zip"
    _make_zip(archive, [("big.bin", b"x" * 10_000)], compress_type=zipfile.ZIP_STORED)
    with zipfile.ZipFile(archive) as zf:
        raw = zf.infolist()[0]
        target = str(tmp_path / "out" / "big.bin")
        safe_entries = [_SafeEntry(fmt="zip", raw=raw, target=target, is_dir=False)]
        _written_bytes, written_count, failure = _pass2_extract(zf, "zip", safe_entries, 10)
    assert written_count == 0
    assert failure is not None
    assert "extraction budget" in failure
    assert not os.path.exists(target)


# ---- declared-size / entry-count budgets, nothing extracted on refusal ---------------------


def test_declared_size_sum_over_budget_refused_nothing_extracted(tmp_path):
    archive = tmp_path / "a.zip"
    _make_zip(archive, [("big.bin", b"x" * 10_000)], compress_type=zipfile.ZIP_STORED)
    dest = tmp_path / "out"
    dest.mkdir()
    tool = make_extract_archive_tool(str(tmp_path), max_extracted_bytes=100)
    result = tool("a.zip", "out")
    assert result.startswith("Refused:")
    assert "budget" in result
    assert os.listdir(dest) == []


def test_lied_declared_size_surfaces_as_a_clean_archive_error_not_a_bomb(tmp_path):
    archive = tmp_path / "a.zip"
    _make_zip(archive, [("f.bin", b"x" * 5000)], compress_type=zipfile.ZIP_STORED)
    _write_bytes(archive, _lie_zip_declared_size(_read_bytes(archive), 10))
    dest = tmp_path / "out"
    dest.mkdir()
    tool = make_extract_archive_tool(str(tmp_path))
    result = tool("a.zip", "out")
    # Either Pass 1 (declared size now looks tiny, so it PASSES the budget check) followed by a
    # Pass-2 CRC failure, or an equivalent clean refusal -- either way, never a raw traceback and
    # never a large payload actually landing on disk.
    assert not result.startswith("Extracted")
    assert os.listdir(dest) == []


def test_more_than_max_entries_refused_nothing_extracted(tmp_path):
    archive = tmp_path / "a.zip"
    _make_zip(archive, [(f"f{i}.txt", b"x") for i in range(10)])
    dest = tmp_path / "out"
    dest.mkdir()
    tool = make_extract_archive_tool(str(tmp_path), max_entries=5)
    result = tool("a.zip", "out")
    assert result.startswith("Refused:")
    assert "entries" in result
    assert os.listdir(dest) == []


# ---- corrupt/truncated archive at all three wrapped call sites ------------------------------


def test_truncated_zip_at_initial_open_reports_archive_error(tmp_path):
    archive = tmp_path / "a.zip"
    _make_zip(archive, [("f.txt", b"x" * 1000)])
    raw = _read_bytes(archive)
    _write_bytes(archive, raw[: len(raw) // 2])
    tool = make_extract_archive_tool(str(tmp_path))
    result = tool("a.zip", "out")
    assert result.startswith("Archive error")


def test_truncated_tar_gz_reports_archive_error(tmp_path):
    archive = tmp_path / "a.tar.gz"
    _make_tar(archive, [("f.txt", b"x" * 1000)])
    raw = _read_bytes(archive)
    _write_bytes(archive, raw[: len(raw) // 2])
    tool = make_extract_archive_tool(str(tmp_path))
    result = tool("a.tar.gz", "out")
    assert result.startswith("Archive error")


# ---- name normalization / empty-name refusal -------------------------------------------------


def test_empty_name_entry_refused_not_isadirectoryerror(tmp_path):
    archive = tmp_path / "a.zip"
    _make_zip(archive, [("", b"x")])
    dest = tmp_path / "out"
    dest.mkdir()
    tool = make_extract_archive_tool(str(tmp_path))
    result = tool("a.zip", "out")
    assert result.startswith("Refused:")
    assert os.listdir(dest) == []


class _EmptyNameZipInfo:
    """A zip entry whose name-derived accessor behaves the way CPython 3.11's `ZipInfo.is_dir()`
    does on an empty name (`filename[-1]` -> IndexError). 3.12+ returns False instead, so the
    real-archive test above only exercises this on 3.11 -- this stub pins the ordering on EVERY
    version, and would go red again the moment a metadata accessor is read before the name is
    refused."""

    filename = ""
    file_size = 1
    external_attr = 0
    flag_bits = 0
    compress_type = zipfile.ZIP_STORED

    def is_dir(self):
        raise IndexError("string index out of range")


class _StubZipArchive:
    def __init__(self, entries):
        self._entries = entries

    def infolist(self):
        return self._entries


def test_empty_name_refused_before_any_name_derived_accessor(tmp_path):
    safe_entries, refusal = _pass1_validate(
        _StubZipArchive([_EmptyNameZipInfo()]), "zip", str(tmp_path), 100, 10_000
    )
    assert safe_entries == []
    assert refusal == "Refused: an archive entry has an empty or '.' name."


def test_backslash_name_normalized_to_nested_path(tmp_path):
    archive = tmp_path / "a.zip"
    _make_zip(archive, [("pkg\\util.py", b"code")])
    tool = make_extract_archive_tool(str(tmp_path))
    result = tool("a.zip", "out")
    assert result.startswith("Extracted")
    assert _read_bytes(tmp_path / "out" / "pkg" / "util.py") == b"code"
    assert not (tmp_path / "out" / "pkg\\util.py").exists()


# ---- streaming correctness for a single large, honestly-declared entry ----------------------


def test_single_large_accurately_declared_entry_extracts_correctly(tmp_path):
    payload = os.urandom(5 * 1024 * 1024)  # 5 MiB, several multiples of _CHUNK_SIZE
    archive = tmp_path / "a.zip"
    _make_zip(archive, [("big.bin", payload)], compress_type=zipfile.ZIP_STORED)
    tool = make_extract_archive_tool(str(tmp_path))
    result = tool("a.zip", "out")
    assert result.startswith("Extracted")
    assert _read_bytes(tmp_path / "out" / "big.bin") == payload


def test_compressed_tar_over_max_entries_still_refused(tmp_path):
    # Confirms Pass 1 still correctly catches an over-max_entries archive despite needing to
    # decompress the outer stream to enumerate headers (the "bounded memory, real CPU cost"
    # disclosure) -- correctness, not a timing assertion.
    archive = tmp_path / "a.tar.gz"
    _make_tar(archive, [(f"f{i}.txt", b"x") for i in range(10)])
    dest = tmp_path / "out"
    dest.mkdir()
    tool = make_extract_archive_tool(str(tmp_path), max_entries=5)
    result = tool("a.tar.gz", "out")
    assert result.startswith("Refused:")
    assert os.listdir(dest) == []


# ---- unsupported extension -------------------------------------------------------------------


def test_unsupported_extension_returns_error_string(tmp_path):
    archive = tmp_path / "a.rar"
    archive.write_bytes(b"not really a rar")
    tool = make_extract_archive_tool(str(tmp_path))
    result = tool("a.rar", "out")
    assert result.startswith("Unsupported archive extension")


# ---- root containment, nested dirs, overwrite -----------------------------------------------


def test_archive_path_escaping_root_refused(tmp_path):
    outside = tmp_path.parent / "outside.zip"
    _make_zip(outside, [("f.txt", b"x")])
    root = tmp_path / "root"
    root.mkdir()
    tool = make_extract_archive_tool(str(root))
    result = tool("../outside.zip", "out")
    assert result.startswith("Refused:")


def test_dest_dir_escaping_root_refused(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    _make_zip(root / "a.zip", [("f.txt", b"x")])
    tool = make_extract_archive_tool(str(root))
    result = tool("a.zip", "../outside")
    assert result.startswith("Refused:")


def test_dest_dir_not_yet_existing_is_created(tmp_path):
    _make_zip(tmp_path / "a.zip", [("f.txt", b"x")])
    tool = make_extract_archive_tool(str(tmp_path))
    result = tool("a.zip", "new/nested/dir")
    assert result.startswith("Extracted")
    assert _read_bytes(tmp_path / "new" / "nested" / "dir" / "f.txt") == b"x"


def test_overwrite_existing_file_at_destination(tmp_path):
    _make_zip(tmp_path / "a.zip", [("f.txt", b"new content")])
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "f.txt").write_text("old content")
    tool = make_extract_archive_tool(str(tmp_path))
    result = tool("a.zip", "out")
    assert result.startswith("Extracted")
    assert _read_bytes(dest / "f.txt") == b"new content"


# ---- name= / collision / invalid / reserved / assert_repl_safe -----------------------------


def test_name_override_fixes_the_real_multi_root_collision(tmp_path):
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()
    a = make_extract_archive_tool(str(root_a))
    b = make_extract_archive_tool(str(root_b))
    task = type("T", (), {"signature": "q: str -> a: str", "tools": [a, b], "output_field": "a"})()
    with pytest.raises(AssertionError, match="duplicate REPL tool name"):
        assert_task_repl_safe(task)

    a2 = make_extract_archive_tool(str(root_a), name="extract_a")
    b2 = make_extract_archive_tool(str(root_b), name="extract_b")
    task2 = type(
        "T", (), {"signature": "q: str -> a: str", "tools": [a2, b2], "output_field": "a"}
    )()
    assert_task_repl_safe(task2)  # must not raise


def test_name_invalid_identifier_raises_value_error(tmp_path):
    with pytest.raises(ValueError, match="not a valid tool name"):
        make_extract_archive_tool(str(tmp_path), name="extract-archive")


def test_name_reserved_by_dspy_raises_value_error(tmp_path):
    from rlm_harness._dspy_compat import reserved_tool_names

    reserved = next(iter(reserved_tool_names()))
    with pytest.raises(ValueError, match="reserved by dspy's sandbox"):
        make_extract_archive_tool(str(tmp_path), name=reserved)


def test_is_repl_safe(tmp_path):
    assert_repl_safe(make_extract_archive_tool(str(tmp_path)))


def test_trace_records_success(tmp_path):
    _make_zip(tmp_path / "a.zip", [("f.txt", b"x")])
    tool = make_extract_archive_tool(str(tmp_path))
    trace_path = str(tmp_path / "t.jsonl")
    with TraceRecorder(trace_path, run_id="r1"):
        tool("a.zip", "out")
    tc = [e for e in load_events(trace_path) if e["type"] == EVENT_TOOL_CALL][0]
    assert tc["payload"]["ok"] is True
    assert tc["payload"]["tool"] == "extract_archive"


# ---- atomic_write_stream ----------------------------------------------------------------------


def test_atomic_write_stream_assembles_multiple_chunks_in_order(tmp_path):
    path = str(tmp_path / "out.bin")
    written = atomic_write_stream(path, [b"abc", b"def", b"ghi"])
    assert written == 9
    assert _read_bytes(path) == b"abcdefghi"


def test_atomic_write_stream_preserves_permission_bits_on_overwrite(tmp_path):
    path = str(tmp_path / "script.sh")
    atomic_write_stream(path, [b"#!/bin/sh\n"])
    os.chmod(path, 0o755)
    atomic_write_stream(path, [b"#!/bin/sh\necho hi\n"])
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o755


def test_atomic_write_stream_budget_exceeded_leaves_destination_untouched(tmp_path):
    path = str(tmp_path / "out.bin")
    _write_bytes(path, b"pre-existing content")
    with pytest.raises(_ExtractionBudgetExceeded):
        atomic_write_stream(path, [b"x" * 100], max_bytes=10)
    assert _read_bytes(path) == b"pre-existing content"
    leftovers = [f for f in os.listdir(tmp_path) if f.startswith(".tmp-")]
    assert leftovers == []


def test_atomic_write_stream_budget_exceeded_is_an_oserror_subclass(tmp_path):
    path = str(tmp_path / "out.bin")
    with pytest.raises(OSError):
        atomic_write_stream(path, [b"x" * 100], max_bytes=10)
