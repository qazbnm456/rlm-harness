"""Purge: the host's one deliberate way to erase. Offline, dspy-free, real SQLite in a temp dir."""

from __future__ import annotations

import contextlib
import os
import sqlite3

import pytest

from rlm_harness.memory import MemoryStore
from rlm_harness.tools import make_memory_get_tool


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path / "m.db")


def _raw(store: MemoryStore):
    return contextlib.closing(sqlite3.connect(store.path, isolation_level=None))


def test_purge_erases_the_item_its_labels_and_its_tokens(store):
    item = store.add("a zanzibarian phrase", provenance="observed", origin="o", tags=["quixotic"])
    store.label(item.id, title="Unrepeatable heading", tags=["ephemeral"], provenance="inferred")
    store.note_distill_failure(item.id, "boom")
    keep = store.add("something else", provenance="observed", origin="o")

    assert store.purge([item.id], reason="the person asked") == [item.id]

    assert store.get(item.id) is None and store.count() == 1
    assert store.labels(item.id).title == "" and store.last_distill_error(item.id) is None
    for word in ("zanzibarian", "quixotic", "unrepeatable", "ephemeral"):
        assert store.search(word) == [], f"{word!r} is still in the index"
    with _raw(store) as conn:
        assert conn.execute("SELECT COUNT(*) FROM labels WHERE item_id = ?", (1,)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM purge_flag").fetchone()[0] == 0
    assert store.get(keep.id) is not None


def test_the_purge_record_holds_no_content(store):
    item = store.add("secret text", provenance="observed", origin="o")
    store.purge([item.id], reason="asked")
    record = store.purged(item.id)
    assert record.item_id == item.id and record.reason == "asked" and record.purged_at > 0
    with _raw(store) as conn:
        row = conn.execute("SELECT * FROM purged").fetchone()
    assert "secret" not in repr(row)
    assert store.purged("m404") is None and store.purged("junk") is None


def test_ids_are_never_reused_after_a_purge(store):
    store.add("a", provenance="said", origin="o")
    second = store.add("b", provenance="said", origin="o")
    store.purge([second.id])
    assert store.add("c", provenance="said", origin="o").id == "m3"


def test_a_stray_delete_is_still_refused_after_a_purge(store):
    store.add("a", provenance="said", origin="o")
    victim = store.add("b", provenance="said", origin="o")
    store.purge([victim.id])
    with _raw(store) as conn:
        for sql in ("DELETE FROM items WHERE id = 1", "DELETE FROM purged"):
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(sql)
    assert store.get("m1") is not None


def test_purge_can_take_the_revisions_with_it(store):
    original = store.add("original", provenance="observed", origin="o")
    first = store.revise(original.id, "first revision", origin="m")
    second = store.revise(first.id, "second revision", origin="m")
    unrelated = store.add("unrelated", provenance="observed", origin="o")
    assert sorted(store.purge([original.id], include_revisions=True)) == sorted(
        [original.id, first.id, second.id])
    assert store.get(unrelated.id) is not None and store.count() == 1


def test_without_include_revisions_a_revision_survives(store):
    original = store.add("original", provenance="observed", origin="o")
    revision = store.revise(original.id, "revision", origin="m")
    store.purge([original.id])
    assert store.get(revision.id).revises == original.id
    assert store.get(original.id) is None


def test_unknown_and_malformed_ids_are_ignored(store):
    store.add("a", provenance="said", origin="o")
    assert store.purge(["m404", "junk", None, "m9999999999999999999"]) == []
    assert store.count() == 1


def test_the_wal_is_checkpointed_after_a_purge(store):
    store.add("x" * 50_000, provenance="said", origin="o")
    store.purge(["m1"])
    wal = f"{store.path}-wal"
    assert not os.path.exists(wal) or os.path.getsize(wal) == 0


def test_secure_purge_vacuums_the_free_pages_away(store):
    for _ in range(20):
        store.add("y" * 20_000, provenance="said", origin="o")
    store.purge([f"m{n}" for n in range(1, 21)], secure=True)
    with _raw(store) as conn:
        assert conn.execute("PRAGMA freelist_count").fetchone()[0] == 0


def test_a_purged_item_reads_as_not_found_through_the_tool(store):
    item = store.add("gone soon", provenance="observed", origin="o")
    store.purge([item.id])
    assert make_memory_get_tool(store)(item.id).startswith("Not found: ")


def _file_bytes(store: MemoryStore) -> bytes:
    data = b""
    for suffix in ("", "-wal"):
        path = f"{store.path}{suffix}"
        if os.path.exists(path):
            with open(path, "rb") as fh:
                data += fh.read()
    return data


@pytest.mark.parametrize("secure", [False, True])
def test_a_purged_word_is_gone_from_the_file_bytes(store, secure):
    """A search-only test passed while the words still sat in the index segments behind a
    tombstone; this one reads the file."""
    for n in range(30):
        store.add(f"ordinary filler text number {n}", provenance="observed", origin="o")
    victim = store.add("xylophonist quinquagenarian", provenance="observed", origin="o")
    store.label(victim.id, title="Brobdingnagian heading", provenance="inferred")
    assert b"xylophonist" in _file_bytes(store)
    store.purge([victim.id], secure=secure)
    data = _file_bytes(store)
    for word in (b"xylophonist", b"quinquagenarian", b"brobdingnagian", b"Brobdingnagian"):
        assert word not in data, f"{word!r} is still in the file"
    assert len(store.search("filler")) == 10, "the merged index still answers"


def test_a_busy_checkpoint_is_reported_not_hidden(store, caplog):
    store.add("x", provenance="said", origin="o")
    reader = sqlite3.connect(store.path)
    try:
        reader.execute("BEGIN")
        reader.execute("SELECT COUNT(*) FROM items").fetchone()
        with caplog.at_level("WARNING", logger="rlm_harness.memory"):
            store.purge(["m1"])
        assert "checkpoint" in caplog.text
    finally:
        reader.close()


def test_a_stray_committed_purge_flag_is_cleared_on_open(store, caplog):
    store.add("x", provenance="said", origin="o")
    with _raw(store) as conn:
        conn.execute("INSERT INTO purge_flag (x) VALUES (1)")
    with caplog.at_level("WARNING", logger="rlm_harness.memory"):
        reopened = MemoryStore(store.path)
    assert "purge_flag" in caplog.text
    with _raw(reopened) as conn, pytest.raises(sqlite3.IntegrityError):
        conn.execute("DELETE FROM items WHERE id = 1")


def test_a_vacuum_that_cannot_run_still_reports_the_committed_purge(store, caplog, monkeypatch):
    item = store.add("x", provenance="said", origin="o")
    real_connect = store._connect

    import contextlib as _cl

    @_cl.contextmanager
    def connect_failing_vacuum():
        with real_connect() as conn:
            class Wrapped:
                def __getattr__(self, name):
                    return getattr(conn, name)

                def execute(self, sql, *args):
                    if sql == "VACUUM":
                        raise sqlite3.OperationalError("database is locked")
                    return conn.execute(sql, *args)
            yield Wrapped()

    monkeypatch.setattr(store, "_connect", connect_failing_vacuum)
    with caplog.at_level("WARNING", logger="rlm_harness.memory"):
        assert store.purge([item.id], secure=True) == [item.id]
    assert "VACUUM failed" in caplog.text


def test_a_purge_of_more_ids_than_sqlite_binds_works(store):
    keep = store.add("kept", provenance="said", origin="o")
    victim = store.add("gone", provenance="said", origin="o")
    ids = [victim.id] + [f"m{n}" for n in range(10, 40_010)]
    assert store.purge(ids) == [victim.id]
    assert store.get(keep.id) is not None


def test_a_bare_string_is_refused_not_read_character_by_character(store):
    store.add("x", provenance="said", origin="o")
    with pytest.raises(TypeError):
        store.purge("m1")
    with pytest.raises(TypeError):
        store.unlabelled(item_ids="m1")
