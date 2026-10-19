"""Looking items up by a host's key, searching origins, capping the index, and the WAL checkpoint.
Offline, dspy-free, real SQLite in a temp dir."""

from __future__ import annotations

import contextlib
import json
import sqlite3

import pytest

from rlm_harness import memory as memory_module
from rlm_harness.memory import _MIGRATIONS, MemoryStore


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path / "m.db")


# ---- find_meta ----------------------------------------------------------------------------------


def test_find_meta_finds_items_by_a_hosts_key(store):
    first = store.add("one", provenance="observed", origin="o", meta={"host_key": "a-1", "n": 7})
    store.add("two", provenance="observed", origin="o", meta={"host_key": "b-2"})
    again = store.add("three", provenance="observed", origin="o", meta={"host_key": "a-1"})
    store.retire(again.id)
    assert [i.id for i in store.find_meta("host_key", "a-1")] == [first.id, again.id], \
        "oldest first, retired ones included"
    assert [i.id for i in store.find_meta("n", 7)] == [first.id]
    assert store.find_meta("host_key", "missing") == [] and store.find_meta("absent", "a-1") == []


def test_find_meta_reads_a_key_with_dots_or_quotes_literally(store):
    odd = store.add("x", provenance="observed", origin="o", meta={'a.b"c': "v", "a": {"b": "v"}})
    assert [i.id for i in store.find_meta('a.b"c', "v")] == [odd.id]
    assert store.find_meta("a.b", "v") == [], "a dot is part of the key, not a path step"


def test_find_meta_matches_only_a_value_of_the_same_kind(store):
    flagged = store.add("a", provenance="said", origin="o", meta={"k": True})
    listed = store.add("b", provenance="said", origin="o", meta={"k": ["v"]})
    nested = store.add("c", provenance="said", origin="o", meta={"k": {"v": 1}})
    real = store.add("d", provenance="said", origin="o", meta={"k": 7.0})
    assert store.find_meta("k", 1) == [], "a JSON true is not the number 1"
    assert store.find_meta("k", '["v"]') == [], "a list is not its JSON text"
    assert [i.id for i in store.find_meta("k", 7)] == [real.id], "7 finds 7.0"
    assert flagged and listed and nested


def test_find_meta_refuses_what_it_cannot_compare(store):
    with pytest.raises(ValueError):
        store.find_meta("", "v")
    with pytest.raises(TypeError):
        store.find_meta("k", True)
    with pytest.raises(TypeError):
        store.find_meta("k", ["v"])


# ---- origins are searchable ---------------------------------------------------------------------


def test_a_search_finds_an_item_by_its_origin_when_the_store_indexes_origins(tmp_path):
    store = MemoryStore(tmp_path / "m.db", index_origins=True)
    page = store.add("Quarterly deadlines for small firms.", provenance="observed",
                     origin="https://example.test/tax-guide")
    scan = store.add("掃描的文件內容", provenance="observed", origin="報稅指南.pdf")
    assert [i.id for i in store.search("tax-guide")] == [page.id]
    assert [i.id for i in store.search("example.test")] == [page.id]
    assert [i.id for i in store.search("報稅")] == [scan.id], "a CJK file name is searchable"


def test_origins_stay_out_of_the_index_by_default(store):
    """The kit writes origins of its own (`model`, `run:<id>`, `trace:...`), and query words are
    ORed: indexed, every note a model wrote would answer a search for "model"."""
    store.add("an unrelated note", provenance="inferred", origin="model")
    store.add("another unrelated note", provenance="inferred", origin="run:abc")
    assert store.search("which model should we use") == []
    assert store.search("run") == []


# ---- max_index_chars ----------------------------------------------------------------------------


def test_max_index_chars_caps_what_search_reads_not_what_is_kept(tmp_path):
    store = MemoryStore(tmp_path / "m.db", max_index_chars=20)
    item = store.add("early words here and then much later a hidden marker", provenance="said", origin="o")
    assert [i.id for i in store.search("early")] == [item.id]
    assert store.search("marker") == [], "past the cap"
    assert store.get(item.id).text.endswith("hidden marker"), "the item keeps its whole text"


@pytest.mark.parametrize("bad", [0, -1, 1.5, "10", True])
def test_max_index_chars_must_be_none_or_a_positive_int(tmp_path, bad):
    with pytest.raises(ValueError, match="max_index_chars"):
        MemoryStore(tmp_path / "m.db", max_index_chars=bad)


# ---- upgrading a step-4 file --------------------------------------------------------------------


def _write_old_item(path, text, *, origin="o"):
    """An item as step-4 code stored it: the index then had two columns, `tags` and `text`."""
    with contextlib.closing(sqlite3.connect(path, isolation_level=None)) as conn:
        cursor = conn.execute(
            "INSERT INTO items (text, provenance, origin, tags_json, created_at, valid_from, meta_json)"
            " VALUES (?, 'observed', ?, '[]', 0, 0, '{}')", (text, origin))
        conn.execute("INSERT INTO items_fts (rowid, tags, text) VALUES (?, '', ?)",
                     (cursor.lastrowid, memory_module._grams(text, unigrams=True)))


def test_a_step_four_file_is_upgraded_and_its_items_are_found_by_origin(tmp_path, monkeypatch):
    path = tmp_path / "m.db"
    with monkeypatch.context() as patch:
        patch.setattr(memory_module, "_MIGRATIONS", _MIGRATIONS[:4])
        patch.setattr(memory_module, "_SCHEMA_VERSION", 4)
        MemoryStore(path)
    _write_old_item(path, "an early phrase, then a word far past the cap", origin="https://example.test/a")
    upgraded = MemoryStore(path, max_index_chars=16, index_origins=True)
    with contextlib.closing(sqlite3.connect(path)) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == len(_MIGRATIONS)
        columns = [r[1] for r in conn.execute("PRAGMA table_info(items_fts)")]
    assert columns == ["tags", "text", "origin"]
    assert [i.id for i in upgraded.search("example.test")] == ["m1"]
    assert [i.id for i in upgraded.search("early")] == ["m1"]
    assert upgraded.search("cap") == [], "the reindex applied the store's cap"
    assert upgraded.purge(["m1"]) == ["m1"] and upgraded.search("early") == [], \
        "the rebuilt index still forgets what purge removes"


def test_the_reindex_marker_is_matched_by_identity_and_never_executed():
    step = _MIGRATIONS[-1]
    assert memory_module._REINDEX_ITEMS in step
    assert not callable(memory_module._REINDEX_ITEMS) and not isinstance(memory_module._REINDEX_ITEMS, str)


# ---- checkpoint ---------------------------------------------------------------------------------


def test_checkpoint_reports_whether_it_finished(store):
    store.add("x", provenance="said", origin="o")
    assert store.checkpoint() is True
    reader = sqlite3.connect(store.path, isolation_level=None)
    try:
        reader.execute("BEGIN")
        reader.execute("SELECT COUNT(*) FROM items").fetchone()
        store.add("y", provenance="said", origin="o")
        assert store.checkpoint() is False, "a reader inside a transaction keeps it from finishing"
    finally:
        reader.close()
    assert store.checkpoint() is True


def test_meta_json_round_trips_through_find_meta(store):
    item = store.add("x", provenance="said", origin="o", meta={"node_id": "nd-0123", "kind": "web"})
    found = store.find_meta("node_id", "nd-0123")
    assert found[0].meta == json.loads(json.dumps(item.meta))
