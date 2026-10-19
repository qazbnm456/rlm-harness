"""Labels on a MemoryStore: what an item is about. Offline, dspy-free, real SQLite in a temp dir."""

from __future__ import annotations

import contextlib
import json
import sqlite3

import pytest

from rlm_harness import memory as memory_module
from rlm_harness.memory import (
    _MIGRATIONS,
    MemoryLabels,
    MemoryStore,
    MemoryUnavailable,
)


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path / "m.db")


def _raw(store: MemoryStore):
    return contextlib.closing(sqlite3.connect(store.path, isolation_level=None))


# ---- writing and reading labels ----------------------------------------------------------------

def test_label_round_trip_and_normalisation(store):
    item = store.add("PostgreSQL vacuum tuning", provenance="observed", origin="o")
    got = store.label(item.id, title="  DB   upkeep ", summary="Keeps tables small.",
                      tags=["Databases", "databases", " ", "ℌistory"], entities=["PostgreSQL", "PostgreSQL"],
                      provenance="inferred", run_id="r1")
    assert isinstance(got, MemoryLabels)
    assert got.title == "DB upkeep" and got.summary == "Keeps tables small."
    # NFKC first, then casefold: the letterlike `ℌ` becomes `h`, not `H`.
    assert got.tags == ("databases", "history")
    assert got.entities == ("PostgreSQL",), "entities keep their case, deduplicated"
    assert got.provenance == {"title": "inferred", "summary": "inferred", "tags": "inferred",
                              "entities": "inferred"}


def test_labels_are_trimmed_never_refused(store):
    item = store.add("x", provenance="said", origin="o")
    got = store.label(item.id, title="t" * 500, summary="s" * 5000,
                      tags=[f"tag{n}" + "x" * 100 for n in range(20)], entities=[f"E{n}" for n in range(20)])
    assert len(got.title) == 80 and len(got.summary) == 600
    assert len(got.tags) == 8 and all(len(t) <= 40 for t in got.tags)
    assert len(got.entities) == 8


def test_an_omitted_kind_is_untouched(store):
    item = store.add("x", provenance="said", origin="o")
    store.label(item.id, title="first", tags=["a"])
    got = store.label(item.id, title="second")
    assert got.title == "second" and got.tags == ("a",)


@pytest.mark.parametrize("kwargs,error", [
    ({}, ValueError),
    ({"title": "t", "provenance": "observed"}, ValueError),
    ({"title": None, "provenance": "inferred"}, ValueError),
    ({"title": 5}, TypeError),
    ({"tags": "not-a-list"}, TypeError),
    ({"entities": [1]}, TypeError),
])
def test_label_refuses_what_it_cannot_record(store, kwargs, error):
    item = store.add("x", provenance="said", origin="o")
    with pytest.raises(error):
        store.label(item.id, **kwargs)
    assert store.labels(item.id) == MemoryLabels(provenance=dict.fromkeys(memory_module.LABEL_KINDS))


@pytest.mark.parametrize("bad", ["m404", "nope", None])
def test_label_refuses_an_unknown_item(store, bad):
    with pytest.raises(KeyError):
        store.label(bad, title="t")


def test_labels_of_an_unknown_or_unlabelled_item_are_empty(store):
    item = store.add("x", provenance="said", origin="o")
    assert store.labels(item.id).title == "" and store.labels(item.id).provenance["title"] is None
    assert store.labels("m404") == MemoryLabels() and store.labels(None) == MemoryLabels()


# ---- which label is current ---------------------------------------------------------------------

def test_a_said_label_outranks_a_later_inferred_one(store):
    item = store.add("x", provenance="said", origin="o")
    store.label(item.id, title="the person's", provenance="said")
    store.label(item.id, title="the model's", provenance="inferred")
    got = store.labels(item.id)
    assert got.title == "the person's" and got.provenance["title"] == "said"


def test_the_latest_inferred_label_wins_over_an_older_one(store):
    item = store.add("x", provenance="said", origin="o")
    store.label(item.id, title="old", provenance="inferred")
    store.label(item.id, title="new", provenance="inferred")
    assert store.labels(item.id).title == "new"


def test_a_said_clear_lets_the_inferred_label_show_through(store):
    item = store.add("x", provenance="said", origin="o")
    store.label(item.id, title="model", provenance="inferred")
    store.label(item.id, title="person", provenance="said")
    got = store.label(item.id, title=None, provenance="said")
    assert got.title == "model" and got.provenance["title"] == "inferred"


def test_a_said_clear_with_nothing_inferred_leaves_the_kind_empty(store):
    item = store.add("x", provenance="said", origin="o")
    store.label(item.id, title="person")
    got = store.label(item.id, title=None)
    assert got.title == "" and got.provenance["title"] is None


def test_a_said_label_after_a_clear_is_current_again(store):
    item = store.add("x", provenance="said", origin="o")
    store.label(item.id, title="model", provenance="inferred")
    store.label(item.id, title=None)
    assert store.label(item.id, title="person again").title == "person again"


# ---- append-only label tables -------------------------------------------------------------------

@pytest.mark.parametrize("sql", [
    "UPDATE labels SET value_json = '\"x\"' WHERE id = 1",
    "UPDATE labels SET run_id = 'r9' WHERE id = 1",
    "DELETE FROM labels WHERE id = 1",
    (
        "INSERT OR REPLACE INTO labels (id, item_id, kind, value_json, provenance, created_at)"
        " VALUES (1, 1, 'title', '\"evil\"', 'said', 0)"
    ),
    "UPDATE distill_failures SET error = 'x' WHERE id = 1",
    "DELETE FROM distill_failures WHERE id = 1",
    "INSERT OR REPLACE INTO distill_failures (id, item_id, error, created_at) VALUES (1, 1, 'e', 0)",
    "UPDATE purged SET reason = 'rewritten' WHERE id = 1",
    "INSERT OR REPLACE INTO purged (id, item_id, reason, created_at) VALUES (1, 9, 'evil', 0)",
    "DELETE FROM purged WHERE id = 1",
])
def test_a_stray_statement_cannot_rewrite_a_label_or_a_failure(store, sql):
    item = store.add("x", provenance="said", origin="o")
    store.label(item.id, title="kept", provenance="inferred")
    store.note_distill_failure(item.id, "boom")
    store.purge([store.add("gone", provenance="said", origin="o").id], reason="kept record")
    with _raw(store) as conn, pytest.raises(sqlite3.IntegrityError):
        conn.execute(sql)
    assert store.labels(item.id).title == "kept"
    assert store.last_distill_error(item.id)[0] == "boom"
    assert store.purged("m2").reason == "kept record"


_INSERT_LABEL = ("INSERT INTO labels (item_id, kind, value_json, provenance, created_at) "
                 "VALUES (1, ?, ?, ?, 0)")


@pytest.mark.parametrize("kind,value,provenance", [
    ("title", None, "inferred"),       # an inferred row may never be a clear
    ("title", '["a"]', "said"),         # a title is a string...
    ("tags", '"a"', "said"),            # ...and tags an array
    ("colour", '"a"', "said"),
    ("title", '"a"', "observed"),
])
def test_the_label_table_refuses_an_unreadable_row(store, kind, value, provenance):
    store.add("x", provenance="said", origin="o")
    with _raw(store) as conn, pytest.raises(sqlite3.IntegrityError):
        conn.execute(_INSERT_LABEL, (kind, value, provenance))


# ---- the distill queue --------------------------------------------------------------------------

def test_unlabelled_is_newest_first_and_skips_labelled_and_retired(store):
    a = store.add("a", provenance="said", origin="o")
    b = store.add("b", provenance="said", origin="o")
    c = store.add("c", provenance="said", origin="o")
    d = store.add("d", provenance="said", origin="o")
    store.label(b.id, summary="done", provenance="inferred")
    store.label(c.id, summary="", provenance="inferred")   # empty is still labelled: asked once
    store.retire(d.id)
    store.label(a.id, title="a title only")                 # no summary yet: still queued
    assert [i.id for i in store.unlabelled()] == [a.id]


def test_a_failing_item_leaves_the_queue_unless_named(store):
    item = store.add("x", provenance="said", origin="o")
    for _ in range(3):
        store.note_distill_failure(item.id, "boom")
    assert store.unlabelled(max_failures=3) == []
    assert [i.id for i in store.unlabelled(item_ids=[item.id])] == [item.id]
    assert [i.id for i in store.unlabelled(max_failures=4)] == [item.id]


def test_unlabelled_scopes_and_limits(store):
    short = store.add("x" * 10, provenance="said", origin="o")
    store.add("y" * 100, provenance="said", origin="o")
    assert [i.id for i in store.unlabelled(max_chars=50)] == [short.id]
    assert len(store.unlabelled(limit=1)) == 1
    assert store.unlabelled(item_ids=["m404", "junk"]) == []


def test_last_distill_error_is_the_latest(store):
    item = store.add("x", provenance="said", origin="o")
    assert store.last_distill_error(item.id) is None
    store.note_distill_failure(item.id, "first")
    store.note_distill_failure(item.id, "second")
    assert store.last_distill_error(item.id)[0] == "second"
    with pytest.raises(KeyError):
        store.note_distill_failure("m404", "x")


def test_label_counts_count_current_labels_of_current_items(store):
    a = store.add("a", provenance="said", origin="o")
    b = store.add("b", provenance="said", origin="o")
    c = store.add("c", provenance="said", origin="o")
    store.label(a.id, tags=["databases", "security"], provenance="inferred")
    store.label(b.id, tags=["old"], provenance="inferred")
    store.label(b.id, tags=["databases"], provenance="inferred")    # supersedes `old`
    store.label(c.id, tags=["databases"], provenance="inferred")
    store.retire(c.id)
    assert store.label_counts("tags") == [("databases", 2), ("security", 1)]
    with pytest.raises(ValueError):
        store.label_counts("title")


# ---- search over labels -------------------------------------------------------------------------

def test_a_label_hit_ranks_above_a_body_hit(store):
    body = store.add("a long text that mentions replication once", provenance="observed", origin="o")
    labelled = store.add("about copying data between servers", provenance="observed", origin="o")
    store.label(labelled.id, title="Replication", provenance="inferred")
    assert [i.id for i in store.search("replication")] == [labelled.id, body.id]


def test_a_superseded_label_is_not_searched(store):
    item = store.add("plain text", provenance="observed", origin="o")
    store.label(item.id, title="Obsolete heading", provenance="inferred")
    store.label(item.id, title="Current heading", provenance="inferred")
    assert store.search("obsolete") == []
    assert [i.id for i in store.search("current")] == [item.id]


def test_label_text_is_searched_with_the_cjk_rules(store):
    item = store.add("plain text", provenance="observed", origin="o")
    store.label(item.id, title="資料庫維護", provenance="inferred")
    assert [i.id for i in store.search("維護")] == [item.id]


def test_tags_filter_reads_label_tags_and_item_tags(store):
    tagged_item = store.add("a", provenance="observed", origin="o", tags=["Databases"])
    labelled = store.add("b", provenance="observed", origin="o")
    store.label(labelled.id, tags=["databases"], provenance="inferred")
    store.add("c", provenance="observed", origin="o")
    assert {i.id for i in store.search("", tags=["Databases"])} == {tagged_item.id, labelled.id}


def test_entities_filter_reads_current_entity_labels(store):
    item = store.add("a", provenance="observed", origin="o")
    store.label(item.id, entities=["PostgreSQL", "Linux"], provenance="inferred")
    other = store.add("b", provenance="observed", origin="o")
    store.label(other.id, entities=["PostgreSQL"], provenance="inferred")
    assert {i.id for i in store.search("", entities=["PostgreSQL"])} == {item.id, other.id}
    assert [i.id for i in store.search("", entities=["PostgreSQL", "Linux"])] == [item.id]


# ---- schema -------------------------------------------------------------------------------------


def _write_old_item(path, text, *, tags=()):
    """An item written the way the code of an earlier step did. `add` now indexes the origin as
    well, a column a ladder cut short of step 5 does not have, so an older file's rows are written
    here, in the shape that step stored them."""
    with contextlib.closing(sqlite3.connect(path, isolation_level=None)) as conn:
        cursor = conn.execute(
            "INSERT INTO items (text, provenance, origin, tags_json, created_at, valid_from, meta_json)"
            " VALUES (?, 'observed', 'o', ?, 0, 0, '{}')", (text, json.dumps(list(tags))))
        conn.execute("INSERT INTO items_fts (rowid, tags, text) VALUES (?, ?, ?)",
                     (cursor.lastrowid, memory_module._grams(" ".join(tags), unigrams=True),
                      memory_module._grams(text, unigrams=True)))

def test_a_step_one_database_is_upgraded_and_reindexed(tmp_path, monkeypatch):
    """A file written before labels existed opens, gains the new tables, and its items are still
    found: the index was rebuilt from `items` when it was recreated."""
    path = tmp_path / "m.db"
    with monkeypatch.context() as patch:
        patch.setattr(memory_module, "_MIGRATIONS", _MIGRATIONS[:1])
        patch.setattr(memory_module, "_SCHEMA_VERSION", 1)
        patch.setattr(memory_module, "_FTS_PROBE", "SELECT 1 FROM items_fts LIMIT 0")
        MemoryStore(path)
    _write_old_item(path, "a distinctive phrase", tags=["kept"])
    upgraded = MemoryStore(path)
    assert [i.id for i in upgraded.search("distinctive")] == ["m1"]
    assert [i.id for i in upgraded.search("kept")] == ["m1"]
    with contextlib.closing(sqlite3.connect(path)) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == len(_MIGRATIONS)
    assert upgraded.label("m1", title="t").title == "t"


def test_an_old_sqlite_is_named(tmp_path, monkeypatch):
    monkeypatch.setattr(memory_module.sqlite3, "sqlite_version_info", (3, 42, 0))
    with pytest.raises(MemoryUnavailable, match="3.43"):
        MemoryStore(tmp_path / "m.db")


def test_a_said_summary_clear_puts_the_item_back_in_the_queue(store):
    item = store.add("x", provenance="said", origin="o")
    store.label(item.id, summary="a person's summary")
    assert store.unlabelled() == []
    store.label(item.id, summary=None)
    assert [i.id for i in store.unlabelled()] == [item.id]


def test_a_list_label_must_hold_strings(store):
    store.add("x", provenance="said", origin="o")
    with _raw(store) as conn, pytest.raises(sqlite3.IntegrityError, match="array of strings"):
        conn.execute("INSERT INTO labels (item_id, kind, value_json, provenance, created_at) "
                     "VALUES (1, 'tags', '[1]', 'said', 0)")


def test_a_failed_step_two_leaves_a_step_one_file_as_it_was(tmp_path, monkeypatch):
    path = tmp_path / "m.db"
    with monkeypatch.context() as patch:
        patch.setattr(memory_module, "_MIGRATIONS", _MIGRATIONS[:1])
        patch.setattr(memory_module, "_SCHEMA_VERSION", 1)
        MemoryStore(path)
    _write_old_item(path, "kept")

    def boom(conn):
        raise RuntimeError("step two failed half way")

    broken = (_MIGRATIONS[0], (*_MIGRATIONS[1][:6], boom))
    with monkeypatch.context() as patch:
        patch.setattr(memory_module, "_MIGRATIONS", broken)
        with pytest.raises(RuntimeError, match="half way"):
            MemoryStore(path)
    with contextlib.closing(sqlite3.connect(path)) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
        assert conn.execute("SELECT name FROM sqlite_master WHERE name = 'purge_flag'").fetchone() is None
    assert [i.id for i in MemoryStore(path).search("kept")] == ["m1"]


def test_the_entities_filter_compares_folded(store):
    item = store.add("a", provenance="observed", origin="o")
    store.label(item.id, entities=["PostgreSQL"], provenance="inferred")
    assert [i.id for i in store.search("", entities=["postgresql"])] == [item.id]
    assert [i.id for i in store.search("", entities=["ＰｏｓｔｇｒｅＳＱＬ"])] == [item.id]
