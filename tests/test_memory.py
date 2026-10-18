"""MemoryStore: the provenance-typed memory. All offline, dspy-free, real SQLite in a temp dir."""

from __future__ import annotations

import contextlib
import math
import sqlite3
import subprocess
import sys
import threading
import time

import pytest

from rlm_harness.memory import (
    _MIGRATIONS,
    PROVENANCES,
    MemoryItem,
    MemoryStore,
    MemoryUnavailable,
    _query_terms,
)


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path / "m.db")


def _raw(store: MemoryStore):
    # `closing`, because `with sqlite3.connect(...)` never closes and an open handle keeps the WAL
    # files locked on Windows: the rule memory.py itself states.
    return contextlib.closing(sqlite3.connect(store.path, isolation_level=None))


# ---- writing and reading back ------------------------------------------------------------------

def test_add_returns_the_stored_item_and_get_reads_it_back(store):
    item = store.add("Sleep consolidates memory", provenance="observed", origin="https://x",
                     tags=["sleep", " sleep ", "", "memory"], meta={"page": 3}, run_id="r1")
    assert isinstance(item, MemoryItem)
    assert item.id == "m1"
    assert item.tags == ("sleep", "memory"), "tags are stripped, emptied out and de-duplicated"
    assert item.valid_from == item.created_at and item.valid_to is None and not item.retired
    assert store.get("m1") == item
    assert item.to_dict()["tags"] == ["sleep", "memory"] and item.to_dict()["meta"] == {"page": 3}


def test_an_item_survives_a_new_store_on_the_same_file(tmp_path):
    MemoryStore(tmp_path / "m.db").add("kept", provenance="said", origin="me")
    again = MemoryStore(tmp_path / "m.db")
    assert again.get("m1").text == "kept" and again.count() == 1


def test_the_parent_directory_is_created(tmp_path):
    MemoryStore(tmp_path / "a" / "b" / "m.db").add("x", provenance="said", origin="o")


def test_a_memory_item_hashes_despite_its_dict(store):
    item = store.add("x", provenance="said", origin="o", meta={"k": 1})
    assert hash(item) == hash(store.get(item.id))


@pytest.mark.parametrize("path", [":memory:", "", "file::memory:?cache=shared"])
def test_an_in_memory_database_is_refused(path):
    with pytest.raises(ValueError, match="file path"):
        MemoryStore(path)


@pytest.mark.parametrize("kwargs,match", [
    ({"text": "x", "provenance": "verified", "origin": "o"}, "provenance"),
    ({"text": "   ", "provenance": "said", "origin": "o"}, "non-empty"),
    ({"text": "x", "provenance": "said", "origin": "o", "meta": {"f": object()}}, "JSON"),
    # NaN is not JSON: the json_valid CHECK used to refuse it as an IntegrityError instead.
    ({"text": "x", "provenance": "said", "origin": "o", "meta": {"f": float("nan")}}, "JSON"),
])
def test_add_refuses_what_it_cannot_store_faithfully(store, kwargs, match):
    text = kwargs.pop("text")
    with pytest.raises(ValueError, match=match):
        store.add(text, **kwargs)
    assert store.count() == 0, "a refused add must leave nothing behind"


@pytest.mark.parametrize("kwargs,match", [
    ({"tags": [3]}, "tag"), ({"origin": 7}, "origin"),
    # A non-dict meta used to be stored and then break every read of that row for good.
    ({"meta": ["a"]}, "meta"), ({"meta": "x"}, "meta"),
])
def test_add_refuses_a_wrong_type_with_a_type_error(store, kwargs, match):
    with pytest.raises(TypeError, match=match):
        store.add("x", **{"provenance": "said", "origin": "o", **kwargs})
    assert store.count() == 0


def test_a_text_over_the_cap_is_refused_not_cut(tmp_path):
    small = MemoryStore(tmp_path / "m.db", max_item_chars=5)
    small.add("12345", provenance="said", origin="o")
    with pytest.raises(ValueError, match="max_item_chars of 5"):
        small.add("123456", provenance="said", origin="o")


def test_provenances_is_the_closed_set_of_three():
    assert PROVENANCES == ("observed", "said", "inferred")


# ---- append-only -------------------------------------------------------------------------------

@pytest.mark.parametrize("sql", [
    "UPDATE items SET text = 'rewritten' WHERE id = 1",
    "UPDATE items SET provenance = 'observed' WHERE id = 1",
    # A NULL column filled in later: the reason the trigger compares with IS NOT, not !=.
    "UPDATE items SET revises = 5 WHERE id = 1",
    "UPDATE items SET run_id = 'r9' WHERE id = 1",
    "UPDATE items SET id = 99 WHERE id = 1",
    "DELETE FROM items WHERE id = 1",
    # REPLACE deletes the old row without firing a DELETE trigger.
    (
        "INSERT OR REPLACE INTO items (id, text, provenance, origin, tags_json, created_at,"
        " valid_from, meta_json) VALUES (1, 'evil', 'observed', 'o', '[]', 0, 0, '{}')"
    ),
])
def test_a_stray_statement_cannot_rewrite_history(store, sql):
    store.add("original", provenance="inferred", origin="o")
    with _raw(store) as conn, pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute(sql)
    assert store.get("m1").text == "original" and store.get("m1").provenance == "inferred"


@pytest.mark.parametrize("column,value", [
    ("tags_json", "not json"), ("tags_json", '{"a": 1}'), ("meta_json", "[1]"), ("meta_json", "x"),
])
def test_a_raw_row_with_unreadable_json_is_refused(store, column, value):
    """One such row used to make every later search raise."""
    columns = {"tags_json": "'[]'", "meta_json": "'{}'"}
    columns[column] = "?"
    with _raw(store) as conn, pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO items (text, provenance, origin, tags_json, created_at, valid_from,"
            f" meta_json) VALUES ('x', 'said', 'o', {columns['tags_json']}, 0, 0,"
            f" {columns['meta_json']})",
            (value,),
        )
    store.add("fine", provenance="said", origin="o")
    assert [i.text for i in store.search("")] == ["fine"]


def test_valid_to_can_be_set_once_and_never_unset(store):
    store.add("x", provenance="said", origin="o")
    with _raw(store) as conn:
        conn.execute("UPDATE items SET valid_to = 5 WHERE id = 1")
        for sql in ("UPDATE items SET valid_to = 6 WHERE id = 1",
                    "UPDATE items SET valid_to = NULL WHERE id = 1"):
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(sql)
    assert store.get("m1").valid_to == 5


def test_the_search_index_refuses_deletion(store):
    store.add("x", provenance="said", origin="o")
    with _raw(store) as conn, pytest.raises(sqlite3.OperationalError, match="contentless"):
        conn.execute("DELETE FROM items_fts WHERE rowid = 1")


def test_a_stray_row_at_a_non_positive_id_is_refused_and_cannot_block_later_adds(store):
    """A BEFORE INSERT trigger sees -1 for an auto-assigned id, so a row a host managed to put at
    -1 would have made the REPLACE guard refuse every later add."""
    with _raw(store) as conn, pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO items (id, text, provenance, origin, tags_json, created_at, valid_from,"
            " meta_json) VALUES (-1, 'x', 'said', 'o', '[]', 0, 0, '{}')"
        )
    assert store.add("still works", provenance="said", origin="o").id == "m1"


def test_an_id_is_never_handed_out_twice(store):
    """Only meaningful with the delete trigger out of the way, which is the case it guards."""
    store.add("a", provenance="said", origin="o")
    store.add("b", provenance="said", origin="o")
    with _raw(store) as conn:
        conn.execute("DROP TRIGGER items_no_delete")
        conn.execute("DELETE FROM items WHERE id = 2")
    assert store.add("c", provenance="said", origin="o").id == "m3"


# ---- revise and retire -------------------------------------------------------------------------

def test_revise_adds_an_overlay_and_leaves_the_original_byte_identical(store):
    original = store.add("The meeting is on Monday", provenance="observed", origin="mail")
    fix = store.revise(original.id, "The meeting moved to Tuesday", origin="model")
    assert fix.provenance == "inferred" and fix.revises == original.id
    assert store.get(original.id) == original
    assert store.revisions(original.id) == [fix]


def test_a_host_can_record_a_persons_correction_as_said(store):
    original = store.add("x", provenance="observed", origin="o")
    assert store.revise(original.id, "y", origin="me", provenance="said").provenance == "said"


@pytest.mark.parametrize("bad", ["m999", "nope", "", "m0", "m9999999999999999999"])
def test_revise_refuses_an_unknown_id(store, bad):
    with pytest.raises(KeyError):
        store.revise(bad, "y", origin="o")


def test_retire_sets_valid_to_once(store):
    item = store.add("x", provenance="said", origin="o")
    retired = store.retire(item.id, at=123.0)
    assert retired.valid_to == 123.0 and retired.retired and retired.text == "x"
    with pytest.raises(ValueError, match="already retired"):
        store.retire(item.id)
    with pytest.raises(KeyError):
        store.retire("m999")
    with pytest.raises(KeyError):
        store.retire("m9999999999999999999")


@pytest.mark.parametrize("bad", [math.nan, math.inf, "soon"])
def test_a_timestamp_must_be_a_finite_number(store, bad):
    """SQLite stores NaN as NULL, so a NaN `valid_to` would read as 'never retired'."""
    item = store.add("x", provenance="said", origin="o")
    with pytest.raises(ValueError):
        store.retire(item.id, at=bad)
    with pytest.raises(ValueError):
        store.add("y", provenance="said", origin="o", valid_from=bad)
    with pytest.raises(ValueError):
        store.search("x", at=bad)
    assert not store.get(item.id).retired


def test_get_and_revisions_treat_a_malformed_id_as_absent(store):
    # The last one is well-formed but above SQLite's INTEGER ceiling, which raised OverflowError.
    for bad in ("", "1", "m", "m-1", "x9", None, 5, "m9999999999999999999"):
        assert store.get(bad) is None
        assert store.revisions(bad) == []


# ---- search -----------------------------------------------------------------------------------

def test_a_tag_hit_outranks_a_body_hit(store):
    body = store.add("a long note that mentions sleep once", provenance="observed", origin="o")
    tagged = store.add("about rest", provenance="observed", origin="o", tags=["sleep"])
    assert [i.id for i in store.search("sleep")] == [tagged.id, body.id]


def test_a_two_character_cjk_word_matches_inside_a_longer_run(store):
    hit = store.add("睡眠與記憶的關係", provenance="observed", origin="o")
    store.add("完全無關的內容", provenance="observed", origin="o")
    assert [i.id for i in store.search("記憶")] == [hit.id]
    assert [i.id for i in store.search("記")] == [hit.id], "a one-character query still matches"


def test_a_cjk_tag_is_found_by_part_of_it(store):
    hit = store.add("x", provenance="observed", origin="o", tags=["神經科學"])
    assert [i.id for i in store.search("科學")] == [hit.id]


@pytest.mark.parametrize("text,query", [
    ("मैं हिन्दी बोलता हूँ", "हिन्दी"),          # Devanagari: virama and vowel signs are marks
    ("מילה שָׁלוֹם כאן", "שָׁלוֹם"),               # pointed Hebrew
    ("𠀀𠀁𠀂", "𠀁𠀂"),                             # CJK extension B, routine in Cantonese text
    ("ㄱㄴㄷ", "ㄴㄷ"),                              # Hangul compatibility jamo
])
def test_words_outside_the_basic_blocks_are_found_whole(store, text, query):
    hit = store.add(text, provenance="observed", origin="o")
    store.add("unrelated words", provenance="observed", origin="o")
    assert [i.id for i in store.search(query)] == [hit.id]


def test_an_iteration_mark_keeps_its_word_together(store):
    """`々` outside the unspaced class left `人々` as the lone `人`, which matched every item
    carrying that ideograph."""
    hit = store.add("人々が集まる", provenance="observed", origin="o")
    store.add("一人で", provenance="observed", origin="o")
    assert [i.id for i in store.search("人々")] == [hit.id]


def test_a_katakana_middle_dot_does_not_leave_a_stray_kana(store):
    """Inside the unspaced class, `・` left `ン` and `ス` as one-character terms, so a dotted name
    matched any item sharing a single kana."""
    hit = store.add("ジョン・スミスの論文", provenance="observed", origin="o")
    store.add("パンとスープ", provenance="observed", origin="o")
    assert [i.id for i in store.search("ジョン・スミス")] == [hit.id]
    assert "ン" not in _query_terms("ジョン・スミス")


def test_a_lone_digit_is_not_a_term():
    assert _query_terms("3") == []


def test_query_terms_keep_combining_marks_inside_a_word():
    assert _query_terms("हिन्दी") == ["हिन्दी"], "split at a mark, the word matched nothing"


def test_full_width_and_half_width_are_one_word(store):
    hit = store.add("the ＰＤＦ parser", provenance="observed", origin="o")
    assert [i.id for i in store.search("pdf")] == [hit.id]


def test_a_query_that_reduces_to_no_terms_matches_nothing(store):
    store.add("x y z", provenance="said", origin="o")
    assert store.search("?!") == []
    assert store.search("a") == [], "a lone Latin letter is not a term"


def test_a_query_term_is_quoted_so_it_cannot_be_syntax(store):
    hit = store.add('he said "OR" NEAR the end', provenance="said", origin="o")
    assert [i.id for i in store.search('"OR" NEAR(')] == [hit.id]


def test_query_terms_are_capped_and_deduplicated():
    assert _query_terms("sleep sleep Sleep") == ["sleep"]
    assert len(_query_terms(" ".join(f"w{n}" for n in range(200)))) == 64


def test_an_empty_query_lists_the_newest_first(store):
    for n in range(3):
        store.add(f"n{n}", provenance="said", origin="o")
    assert [i.id for i in store.search("")] == ["m3", "m2", "m1"]
    assert [i.id for i in store.search("   ", limit=0)] == ["m3"], "limit is clamped to 1"


def test_the_filters(store):
    obs = store.add("sleep paper", provenance="observed", origin="o", tags=["a", "b"])
    said = store.add("sleep diary", provenance="said", origin="o", tags=["a"])
    note = store.add("sleep note", provenance="inferred", origin="o")
    assert {i.id for i in store.search("sleep", provenance="observed")} == {obs.id}
    assert {i.id for i in store.search("sleep", provenance=["said", "inferred"])} == {said.id, note.id}
    assert {i.id for i in store.search("sleep", tags=["a", "b"])} == {obs.id}
    assert {i.id for i in store.search("", tags="a")} == {obs.id, said.id}
    with pytest.raises(ValueError):
        store.search("sleep", provenance="verified")


def test_retired_items_are_hidden_unless_asked_for(store):
    old = store.add("sleep fact", provenance="observed", origin="o")
    store.retire(old.id, at=50.0)
    assert store.search("sleep") == []
    assert [i.id for i in store.search("sleep", include_retired=True)] == [old.id]


def test_an_item_retired_in_the_future_stays_visible_until_then(store):
    item = store.add("sleep fact", provenance="observed", origin="o")
    store.retire(item.id, at=time.time() + 3600)
    assert [i.id for i in store.search("sleep")] == [item.id]
    assert [i.id for i in store.search("sleep", at=time.time())] == [item.id]
    assert store.search("sleep", at=time.time() + 7200) == []


def test_at_is_a_point_in_time_view(store):
    early = store.add("sleep fact", provenance="observed", origin="o", valid_from=10.0)
    later = store.add("sleep fact two", provenance="observed", origin="o", valid_from=100.0)
    store.retire(early.id, at=50.0)
    assert {i.id for i in store.search("sleep", at=20.0)} == {early.id}
    assert {i.id for i in store.search("sleep", at=60.0)} == set()
    assert {i.id for i in store.search("sleep", at=200.0)} == {later.id}
    # A future valid_from is not hidden by the default view.
    assert later.id in {i.id for i in store.search("sleep")}


# ---- concurrency, schema, packaging -------------------------------------------------------------

def test_concurrent_writers_lose_nothing(tmp_path):
    path = tmp_path / "m.db"
    MemoryStore(path)
    errors: list[BaseException] = []

    def write(worker: int) -> None:
        try:
            own = MemoryStore(path)
            for n in range(25):
                own.add(f"w{worker} n{n}", provenance="said", origin="o")
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(w,)) for w in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    assert MemoryStore(path).count() == 200


def test_two_stores_opening_one_new_file_at_once_migrate_it_once(tmp_path):
    path = tmp_path / "new.db"
    errors: list[BaseException] = []
    barrier = threading.Barrier(6)

    def open_store() -> None:
        try:
            barrier.wait()
            MemoryStore(path)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=open_store) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    assert MemoryStore(path).add("x", provenance="said", origin="o").id == "m1"


def test_a_failed_migration_leaves_the_file_as_it_was(tmp_path, monkeypatch):
    """The step runs inside one transaction, so a failure part-way leaves nothing behind and the
    real migrations then open the same file cleanly."""
    from rlm_harness import memory

    path = tmp_path / "m.db"
    broken = ((*_MIGRATIONS[0][:2], "CREATE VIRTUAL TABLE items_fts USING fts5_missing(x)"),)
    with monkeypatch.context() as patch:
        patch.setattr(memory, "_MIGRATIONS", broken)
        patch.setattr(memory, "_SCHEMA_VERSION", 1)
        with pytest.raises(MemoryUnavailable):
            MemoryStore(path)
    with contextlib.closing(sqlite3.connect(path)) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 0
        assert conn.execute("SELECT name FROM sqlite_master WHERE name = 'items'").fetchone() is None
    assert MemoryStore(path).add("x", provenance="said", origin="o").id == "m1"


def test_a_missing_fts5_is_named_for_an_existing_file_too(tmp_path, monkeypatch):
    """An existing file runs no migration step, so only the probe can notice; without it the
    store constructed cleanly and the first search raised a bare OperationalError."""
    from rlm_harness import memory

    path = tmp_path / "m.db"
    MemoryStore(path)
    monkeypatch.setattr(memory, "_FTS_PROBE", "CREATE VIRTUAL TABLE temp.probe USING fts5_missing(x)")
    with pytest.raises(MemoryUnavailable, match="FTS5"):
        MemoryStore(path)


def test_a_database_from_a_newer_kit_is_refused(store):
    with _raw(store) as conn:
        conn.execute(f"PRAGMA user_version = {len(_MIGRATIONS) + 1}")
    with pytest.raises(RuntimeError, match="newer rlm-harness"):
        MemoryStore(store.path)


def test_the_schema_version_counts_the_migration_steps(store):
    with _raw(store) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == len(_MIGRATIONS)


def test_a_missing_fts5_is_named(tmp_path, monkeypatch):
    """The one failure the plan called fragile: it surfaces as MemoryUnavailable, not a bare
    OperationalError, so a consumer can tell 'this SQLite cannot' from a bug."""
    from rlm_harness import memory

    broken = ((*_MIGRATIONS[0][:2], "CREATE VIRTUAL TABLE items_fts USING fts5_missing(x)"),)
    monkeypatch.setattr(memory, "_MIGRATIONS", broken)
    monkeypatch.setattr(memory, "_SCHEMA_VERSION", 1)
    with pytest.raises(MemoryUnavailable, match="FTS5"):
        MemoryStore(tmp_path / "m.db")


def test_the_memory_modules_do_not_import_dspy():
    code = (
        "import sys, rlm_harness.memory, rlm_harness.tools.memory; "
        "assert 'dspy' not in sys.modules, 'the memory modules pulled in dspy'; print('ok')"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"
