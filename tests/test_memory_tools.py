"""The memory REPL tools: search, get, note. Offline, dspy-free except the REPL-safety check."""

from __future__ import annotations

import inspect
import json
import sqlite3

import pytest

from rlm_harness import TraceRecorder, load_events
from rlm_harness.memory import MemoryStore
from rlm_harness.tools import make_memory_get_tool, make_memory_note_tool, make_memory_search_tool


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path / "m.db")


def _calls(path):
    return [e["payload"] for e in load_events(str(path)) if e["type"] == "tool_call"]


# ---- memory_search ----------------------------------------------------------------------------

def test_search_returns_json_rows_with_provenance(store):
    store.add("sleep consolidates memory", provenance="observed", origin="paper", tags=["sleep"])
    rows = json.loads(make_memory_search_tool(store)("sleep"))
    assert rows == [{"id": "m1", "provenance": "observed", "origin": "paper", "tags": ["sleep"],
                     "text": "sleep consolidates memory", "revises": None}]


def test_search_cuts_long_text_and_marks_the_cut(store):
    store.add("x" * 50, provenance="observed", origin="o")
    rows = json.loads(make_memory_search_tool(store, max_chars=10)(""))
    assert rows[0]["text"] == "x" * 10 + " …[truncated]"


def test_search_never_returns_a_retired_item(store):
    item = store.add("sleep", provenance="observed", origin="o")
    store.retire(item.id)
    assert make_memory_search_tool(store)("sleep") == "No memories match 'sleep'."


def test_an_empty_query_tells_an_empty_store_from_an_all_retired_one(store):
    search = make_memory_search_tool(store)
    assert search("") == "Memory is empty."
    store.retire(store.add("x", provenance="said", origin="o").id)
    assert search("") == "No current memories.", "something WAS stored; saying otherwise is false"


@pytest.mark.parametrize("limit,expected", [(1, 1), (3, 3), (100, 4), (0, 1), (-1, 1),
                                            (None, 4), ("2", 2), (2.5, 2), ("many", 4)])
def test_search_clamps_any_limit_and_never_raises(store, limit, expected):
    for n in range(10):
        store.add(f"memory {n}", provenance="observed", origin="o")
    out = make_memory_search_tool(store, max_results=4)("memory", limit=limit)
    assert len(json.loads(out)) == expected


def test_a_query_with_no_terms_reports_no_match_rather_than_listing(store):
    store.add("anything", provenance="observed", origin="o")
    assert make_memory_search_tool(store)("?!") == "No memories match '?!'."


class _Hostile:
    def __str__(self):
        raise RuntimeError("no")

    __repr__ = __str__


@pytest.mark.parametrize("build,call", [
    (make_memory_search_tool, lambda t: t(_Hostile(), limit=_Hostile())),
    (make_memory_get_tool, lambda t: t(_Hostile())),
    (make_memory_note_tool, lambda t: t(_Hostile(), revises=_Hostile())),
])
def test_an_object_whose_str_raises_still_gets_a_string(store, build, call, tmp_path):
    """Under a recorder, which is where it failed: the trace's own json.dumps choked on the raw
    value after the tool had done its work, so a note was stored and the call still raised."""
    assert isinstance(call(build(store)), str)
    with TraceRecorder(str(tmp_path / "t.jsonl"), run_id="r"):
        assert isinstance(call(build(store)), str)


def _circular():
    loop: list = []
    loop.append(loop)
    return loop


class _BadInt:
    def __int__(self):
        raise RuntimeError("no")


@pytest.mark.parametrize("value", [_circular(), {(1, 2): "tuple key"}, _Hostile()])
def test_an_unserialisable_argument_is_recorded_as_text(store, tmp_path, value):
    trace = tmp_path / "t.jsonl"
    with TraceRecorder(str(trace), run_id="r"):
        assert isinstance(make_memory_search_tool(store)(value), str)
        assert isinstance(make_memory_get_tool(store)(value), str)
        assert isinstance(make_memory_note_tool(store)("note", revises=value), str)
    assert len(_calls(trace)) == 3
    assert all(isinstance(c["args"].get("query", c["args"].get("item_id", c["args"].get("revises"))),
                          str) for c in _calls(trace))


def test_a_limit_whose_int_raises_falls_back_to_the_default(store):
    store.add("memory", provenance="observed", origin="o")
    assert json.loads(make_memory_search_tool(store)("memory", limit=_BadInt()))


@pytest.mark.parametrize("query", [None, 5, ["a"], ""])
def test_search_never_raises_on_a_query_of_any_type(store, query):
    assert isinstance(make_memory_search_tool(store)(query), str)


def test_the_trace_records_the_raw_limit_and_the_ids_returned(store, tmp_path):
    store.add("sleep a", provenance="observed", origin="o")
    store.add("sleep b", provenance="observed", origin="o")
    trace = tmp_path / "t.jsonl"
    with TraceRecorder(str(trace), run_id="r"):
        make_memory_search_tool(store, max_results=8)("sleep", limit=99)
    call = _calls(trace)[-1]
    assert call["tool"] == "memory_search" and call["ok"] is True
    assert call["args"] == {"query": "sleep", "limit": 99}, "the model's own value, not the clamp"
    assert sorted(call["memory_ids"]) == ["m1", "m2"]


# ---- memory_get --------------------------------------------------------------------------------

def test_get_returns_the_item_and_what_revises_it(store):
    item = store.add("Monday", provenance="observed", origin="mail")
    store.revise(item.id, "Tuesday", origin="model")
    row = json.loads(make_memory_get_tool(store)(item.id))
    assert row["text"] == "Monday" and row["provenance"] == "observed"
    assert row["revised_by"] == ["m2"] and row["valid_to"] is None


def test_get_cuts_long_text_and_records_what_it_read(store, tmp_path):
    item = store.add("y" * 50, provenance="observed", origin="o")
    trace = tmp_path / "t.jsonl"
    with TraceRecorder(str(trace), run_id="r"):
        row = json.loads(make_memory_get_tool(store, max_chars=10)(item.id))
        make_memory_get_tool(store)("m404")
    assert row["text"] == "y" * 10 + " …[truncated]"
    hit, miss = _calls(trace)[-2:]
    assert hit["tool"] == "memory_get" and hit["ok"] is True and hit["memory_ids"] == [item.id]
    assert miss["ok"] is False and miss["memory_ids"] == [] and miss["args"] == {"item_id": "m404"}


def test_get_reads_a_retired_item(store):
    item = store.add("old", provenance="observed", origin="o")
    store.retire(item.id, at=5.0)
    assert json.loads(make_memory_get_tool(store)(item.id))["valid_to"] == 5.0


@pytest.mark.parametrize("item_id", ["m404", "", None, 3, " m404 ", "m9999999999999999999"])
def test_get_reports_an_unknown_id_without_raising(store, item_id):
    assert make_memory_get_tool(store)(item_id).startswith("Not found: ")


# ---- memory_note -------------------------------------------------------------------------------

def test_a_note_is_always_inferred_and_has_no_provenance_argument(store):
    note = make_memory_note_tool(store, tags=["from-model"])
    assert set(inspect.signature(note).parameters) == {"text", "revises"}
    out = note("the user prefers short answers")
    assert out == "Noted: m1"
    item = store.get("m1")
    assert item.provenance == "inferred" and item.origin == "model" and item.tags == ("from-model",)


def test_a_note_inside_a_run_names_the_run(store, tmp_path):
    trace = tmp_path / "t.jsonl"
    with TraceRecorder(str(trace), run_id="run-7"):
        make_memory_note_tool(store)("remember this")
    item = store.get("m1")
    assert item.origin == "run:run-7" and item.run_id == "run-7"
    call = _calls(trace)[-1]
    assert call["tool"] == "memory_note" and call["memory_ids"] == ["m1"] and call["ok"] is True


def test_a_note_can_revise_and_the_original_stays(store):
    original = store.add("Monday", provenance="observed", origin="mail")
    out = make_memory_note_tool(store)("moved to Tuesday", revises=original.id)
    assert out == "Noted: m2"
    assert store.get(original.id) == original
    assert store.get("m2").revises == original.id and store.get("m2").provenance == "inferred"


@pytest.mark.parametrize("kwargs,expected", [
    ({"text": ""}, "Refused: text must be non-empty."),
    ({"text": None}, "Refused: text must be non-empty."),
    ({"text": "x" * 11}, "Refused: text exceeds 10 characters."),
    ({"text": "ok", "revises": "m404"}, "Refused: no memory with id 'm404'."),
    ({"text": "ok", "revises": "m9999999999999999999"},
     "Refused: no memory with id 'm9999999999999999999'."),
])
def test_the_note_gate_refuses_in_words(store, kwargs, expected):
    assert make_memory_note_tool(store, max_chars=10)(**kwargs) == expected
    assert store.count() == 0


def test_a_store_cap_below_the_tool_cap_is_a_refusal_not_a_crash(tmp_path):
    small = MemoryStore(tmp_path / "m.db", max_item_chars=3)
    assert make_memory_note_tool(small, max_chars=100)("abcdef").startswith("Refused: ")


def test_a_refused_note_records_the_raw_arguments(store, tmp_path):
    trace = tmp_path / "t.jsonl"
    with TraceRecorder(str(trace), run_id="r"):
        make_memory_note_tool(store)(None)
    call = _calls(trace)[-1]
    assert call["ok"] is False and call["args"] == {"text": None, "revises": ""}


def test_a_refused_oversized_note_is_cut_in_the_trace(store, tmp_path):
    trace = tmp_path / "t.jsonl"
    with TraceRecorder(str(trace), run_id="r"):
        make_memory_note_tool(store, max_chars=10)("x" * 5000)
    call = _calls(trace)[-1]
    assert call["ok"] is False and call["text_len"] == 5000
    assert call["args"]["text"] == "x" * 10 + " …[truncated]"


def test_a_bad_factory_tag_fails_at_build_time(store):
    with pytest.raises(TypeError, match="tag"):
        make_memory_note_tool(store, tags=[1])


# ---- failures and the trace's bounds ----------------------------------------------------------

class _BrokenStore(MemoryStore):
    """Every store call fails the way a held lock, a full disk or a damaged file does."""

    def _fail(self, *a, **k):
        raise sqlite3.OperationalError("database is locked")

    search = get = revisions = add = revise = count = _fail


@pytest.mark.parametrize("build,call", [
    (make_memory_search_tool, lambda t: t("sleep")),
    (make_memory_get_tool, lambda t: t("m1")),
    (make_memory_note_tool, lambda t: t("a note")),
    (make_memory_note_tool, lambda t: t("a note", revises="m1")),
])
def test_a_database_failure_is_a_string_and_a_recorded_call(tmp_path, build, call):
    broken = _BrokenStore(tmp_path / "m.db")
    trace = tmp_path / "t.jsonl"
    with TraceRecorder(str(trace), run_id="r"):
        out = call(build(broken))
    assert out == "Memory error: OperationalError"
    recorded = _calls(trace)
    assert len(recorded) == 1, "exactly one tool_call, even when the store fails"
    assert recorded[0]["ok"] is False and recorded[0]["note"] == "error: OperationalError"


def test_a_lone_surrogate_note_is_refused_in_words(store):
    out = make_memory_note_tool(store)("bad \ud800 text")
    assert out.startswith("Refused: ") and store.count() == 0


def test_every_oversized_argument_is_cut_in_the_trace_and_the_echo(store, tmp_path):
    from rlm_harness.tools.memory import _ECHO_CHARS, _TRACE_ARG_CHARS

    trace = tmp_path / "t.jsonl"
    huge = "q" * 50_000
    with TraceRecorder(str(trace), run_id="r"):
        miss = make_memory_search_tool(store)(huge)
        make_memory_get_tool(store)(huge)
        make_memory_note_tool(store, max_chars=10)(" " * 50_000)
        make_memory_note_tool(store)("ok", revises=huge)
    search, get, empty_note, revise_note = _calls(trace)
    assert search["args"]["query"] == "q" * _TRACE_ARG_CHARS + " …[truncated]"
    assert search["query_len"] == 50_000
    assert get["item_id_len"] == 50_000 and len(get["args"]["item_id"]) < 2 * _TRACE_ARG_CHARS
    assert empty_note["text_len"] == 50_000 and len(empty_note["args"]["text"]) < 100
    assert revise_note["revises_len"] == 50_000
    assert len(miss) < _ECHO_CHARS + 60, "the sentinel echoes a cut query, not the whole thing"


def test_an_oversized_non_string_argument_is_cut_too(store, tmp_path):
    from rlm_harness.tools.memory import _TRACE_ARG_CHARS

    trace = tmp_path / "t.jsonl"
    with TraceRecorder(str(trace), run_id="r"):
        out = make_memory_search_tool(store)(["q" * 1_000_000])
    assert isinstance(out, str)
    call = _calls(trace)[-1]
    assert call["query_len"] > 1_000_000
    assert len(call["args"]["query"]) < _TRACE_ARG_CHARS + 50
    assert trace.stat().st_size < 50_000, "one list argument wrote megabytes into the trace"


# ---- names and REPL safety ---------------------------------------------------------------------

def test_two_stores_can_coexist_under_different_names(store, tmp_path):
    other = MemoryStore(tmp_path / "other.db")
    assert make_memory_search_tool(other, name="team_memory_search").__name__ == "team_memory_search"
    with pytest.raises(ValueError):
        make_memory_note_tool(store, name="not a name")


def test_every_memory_tool_is_repl_safe(store):
    pytest.importorskip("dspy")
    from rlm_harness.testing import assert_repl_safe

    for tool in (make_memory_search_tool(store), make_memory_get_tool(store),
                 make_memory_note_tool(store)):
        assert_repl_safe(tool)
