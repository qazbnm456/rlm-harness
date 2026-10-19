"""Topics: named groups of items, local suggestions, and the organising pass. Offline."""

from __future__ import annotations

import asyncio
import contextlib
import json
import sqlite3

import pytest

from rlm_harness import memory as memory_module
from rlm_harness.memory import _MIGRATIONS, MemoryStore, Topic
from rlm_harness.memory_organise import suggest_topics, topic_graph


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path / "m.db")


def _raw(store: MemoryStore):
    return contextlib.closing(sqlite3.connect(store.path, isolation_level=None))


def _item(store, *, entities=(), tags=(), summary="s", title="t"):
    item = store.add(title, provenance="observed", origin="o")
    store.label(item.id, title=title, summary=summary, entities=list(entities), tags=list(tags),
                provenance="inferred")
    return item.id


# ---- topics and their names ---------------------------------------------------------------------

def test_a_topic_is_created_named_and_listed(store):
    topic = store.create_topic("  Databases   and storage ")
    assert isinstance(topic, Topic) and topic.id == "t1" and topic.name == "Databases and storage"
    assert topic.name_provenance == "said" and not topic.retired
    assert store.topics() == [topic] and store.topic("t1") == topic
    assert store.topic("t404") is None and store.topic("junk") is None
    assert len(store.create_topic("x" * 200).name) == 60


def test_a_persons_name_outranks_a_models_whatever_the_order(store):
    topic = store.create_topic("Model name", provenance="inferred")
    assert store.rename_topic(topic.id, "Person name").name == "Person name"
    assert store.rename_topic(topic.id, "Later model name", provenance="inferred").name == "Person name"


def test_retire_hides_a_topic_and_restore_brings_it_back(store):
    topic = store.create_topic("Old")
    store.retire_topic(topic.id)
    assert store.topics() == [] and store.topics(include_retired=True)[0].retired
    assert store.restore_topic(topic.id).retired is False and store.topics()[0].id == topic.id


@pytest.mark.parametrize("call", [
    lambda s: s.rename_topic("t404", "x"),
    lambda s: s.retire_topic("t404"),
    lambda s: s.place("m1", "t404"),
])
def test_an_unknown_topic_is_a_key_error(store, call):
    store.add("x", provenance="said", origin="o")
    with pytest.raises(KeyError):
        call(store)


@pytest.mark.parametrize("name,error", [("", ValueError), ("   ", ValueError), (5, TypeError)])
def test_a_topic_name_must_be_text(store, name, error):
    with pytest.raises(error):
        store.create_topic(name)


# ---- membership ---------------------------------------------------------------------------------

def test_place_and_unplace(store):
    item = _item(store)
    topic = store.create_topic("T")
    store.place(item, topic.id)
    assert [i.id for i in store.topic_items(topic.id)] == [item]
    assert [t.id for t in store.item_topics(item)] == [topic.id]
    store.unplace(item, topic.id)
    assert store.topic_items(topic.id) == [] and store.item_topics(item) == []


def test_a_persons_removal_keeps_a_pair_apart_for_good(store):
    item = _item(store)
    topic = store.create_topic("T")
    store.place(item, topic.id, provenance="inferred")
    store.unplace(item, topic.id)
    assert store.topic_items(topic.id) == [], "the model's placement does not show through"
    with pytest.raises(ValueError, match="person already decided"):
        store.place(item, topic.id, provenance="inferred")
    assert store.removed_pairs() == {(item, topic.id)}


def test_a_persons_placement_cannot_be_undone_by_a_model(store):
    item = _item(store)
    topic = store.create_topic("T")
    store.place(item, topic.id)
    with pytest.raises(ValueError, match="person already decided"):
        store.unplace(item, topic.id, provenance="inferred")


def test_a_retired_topic_takes_no_new_members_but_keeps_its_old_ones(store):
    a, b = _item(store), _item(store)
    topic = store.create_topic("T")
    store.place(a, topic.id)
    store.retire_topic(topic.id)
    with pytest.raises(ValueError, match="retired"):
        store.place(b, topic.id)
    assert [i.id for i in store.topic_items(topic.id)] == [a]
    assert store.item_topics(a) == [], "a retired topic is not listed among an item's topics"


def test_search_and_the_entity_graph_take_a_topic(store):
    a = _item(store, entities=["Kubernetes", "Docker"])
    _item(store, entities=["Kubernetes", "Docker"])
    topic = store.create_topic("T")
    store.place(a, topic.id)
    assert [i.id for i in store.search("", topic=topic.id)] == [a]
    assert store.entity_graph(topic=topic.id) == [("Docker", "Kubernetes", 1)]
    assert store.entity_graph()[0] == ("Docker", "Kubernetes", 2)
    assert store.search("", topic="t404") == []


def test_dismissals_and_leftovers(store):
    item = _item(store)
    topic = store.create_topic("T")
    store.dismiss(item, topic.id)
    store.dismiss(item, topic.id)
    assert store.dismissed() == {(item, topic.id)}
    signature = store.topic_signature()
    store.record_left([item, item], signature)
    assert store.left_items(signature) == [item] and store.left_items() == [item]
    store.create_topic("Another")
    assert store.topic_signature() != signature and store.left_items(store.topic_signature()) == []
    with pytest.raises(TypeError):
        store.record_left(item, signature)


def test_a_purge_takes_an_items_topic_rows_and_leaves_the_topic(store):
    item = _item(store)
    topic = store.create_topic("T")
    store.place(item, topic.id)
    store.dismiss(item, topic.id)
    store.record_left([item], "sig")
    store.purge([item])
    with _raw(store) as conn:
        for table in ("topic_members", "topic_dismissed", "topic_left"):
            assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0, table
    assert store.topic(topic.id) is not None


@pytest.mark.parametrize("table,columns,values", [
    ("topics", "id, provenance, created_at", "1, 'said', 0"),
    ("topic_names", "id, topic_id, name, provenance, created_at", "1, 1, 'evil', 'said', 0"),
    ("topic_retired", "id, topic_id, retired, provenance, created_at", "1, 1, 1, 'said', 0"),
    ("topic_members", "id, item_id, topic_id, member, provenance, created_at", "1, 1, 1, 0, 'said', 0"),
    ("topic_dismissed", "id, item_id, topic_id, created_at", "1, 1, 9, 0"),
    ("topic_left", "id, item_id, signature, created_at", "1, 1, 'evil', 0"),
])
def test_a_stray_statement_cannot_rewrite_a_topic_row(store, table, columns, values):
    item = _item(store)
    topic = store.create_topic("T")
    store.retire_topic(topic.id)
    store.restore_topic(topic.id)
    store.place(item, topic.id)
    store.dismiss(item, topic.id)
    store.record_left([item], "sig")
    with _raw(store) as conn:
        for sql in (f"UPDATE {table} SET created_at = 5 WHERE id = 1",
                    f"DELETE FROM {table} WHERE id = 1",
                    f"INSERT OR REPLACE INTO {table} ({columns}) VALUES ({values})"):
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(sql)
    assert store.topic(topic.id).name == "T" and store.topic_items(topic.id)[0].id == item


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


def test_a_step_three_database_is_upgraded(tmp_path, monkeypatch):
    path = tmp_path / "m.db"
    with monkeypatch.context() as patch:
        patch.setattr(memory_module, "_MIGRATIONS", _MIGRATIONS[:3])
        patch.setattr(memory_module, "_SCHEMA_VERSION", 3)
        MemoryStore(path)
    _write_old_item(path, "x")
    upgraded = MemoryStore(path)
    upgraded.place("m1", upgraded.create_topic("T").id)
    with contextlib.closing(sqlite3.connect(path)) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == len(_MIGRATIONS)


# ---- local suggestions --------------------------------------------------------------------------

def test_suggest_topics_scores_shared_entities_and_tags(store):
    member = _item(store, entities=["Kubernetes", "Docker"], tags=["devops"])
    topic = store.create_topic("Containers")
    store.place(member, topic.id)
    strong = _item(store, entities=["Kubernetes", "Docker"])              # 2 + 2
    weak = _item(store, tags=["devops"])                                  # 1, below the threshold
    mixed = _item(store, entities=["Kubernetes"], tags=["devops"])        # 2 + 1
    suggestions = suggest_topics(store)
    assert [(s.item_id, s.score) for s in suggestions] == [(strong, 4), (mixed, 3)]
    assert suggestions[0].shared_entities == ("Docker", "Kubernetes") and suggestions[0].topic_id == topic.id
    assert weak not in {s.item_id for s in suggestions}


def test_suggest_topics_respects_dismissals_removals_and_landing(store):
    member = _item(store, entities=["Kubernetes", "Docker"])
    topic = store.create_topic("Containers")
    landing = store.create_topic("Inbox")
    store.place(member, topic.id)
    dismissed = _item(store, entities=["Kubernetes", "Docker"])
    removed = _item(store, entities=["Kubernetes", "Docker"])
    in_landing = _item(store, entities=["Kubernetes", "Docker"])
    store.dismiss(dismissed, topic.id)
    store.place(removed, topic.id)
    store.unplace(removed, topic.id)
    store.place(in_landing, landing.id)
    ids = {s.item_id for s in suggest_topics(store, landing=landing.id)}
    assert ids == {in_landing}, "landing alone counts as unfiled; dismissed and removed pairs are not offered"
    assert landing.id not in {s.topic_id for s in suggest_topics(store, landing=landing.id)}


def test_suggest_topics_reads_entities_through_aliases(store):
    member = _item(store, entities=["Kubernetes", "Docker"])
    topic = store.create_topic("Containers")
    store.place(member, topic.id)
    candidate = _item(store, entities=["k8s", "Docker"])
    store.merge("k8s", "Kubernetes")
    assert suggest_topics(store)[0].item_id == candidate and suggest_topics(store)[0].score == 4


def test_topic_graph_keeps_each_topics_strongest_bridges(store):
    hub = store.create_topic("Hub")
    spokes = [store.create_topic(f"Spoke {n}") for n in range(5)]
    hub_item = _item(store, entities=[f"E{n}" for n in range(5)])
    store.place(hub_item, hub.id)
    for n, spoke in enumerate(spokes):
        store.place(_item(store, entities=[f"E{n}"]), spoke.id)
    graph = topic_graph(store, per_topic=3)
    assert len(graph.bridges) == 5, "every spoke keeps its one bridge though the hub's cap is 3"
    assert graph.omitted == {}
    tight = topic_graph(store, per_topic=1)
    assert {b[1] for b in tight.bridges} == {s.id for s in spokes}


# ---- the organising pass ------------------------------------------------------------------------

from rlm_harness.memory_tasks import parse_plan, plan_from


def test_parse_plan_tolerates_fences_and_garbage():
    assert parse_plan('```json\n{"placements": [1]}\n```') == {"placements": [1]}
    assert parse_plan("no json here") == {"placements": []}
    assert parse_plan(None) == {"placements": []}
    assert parse_plan("[1, 2]") == {"placements": []}


def test_plan_from_keeps_only_what_the_host_allows():
    raw = {"placements": [
        {"item": "m1", "topic": "t1", "fit": "subject"},
        {"item": "m1", "topic": "t1", "fit": "subject"},          # twice: kept once
        {"item": "m2", "topic": "t1", "fit": "loose"},            # loose: left
        {"item": "m9", "topic": "t1", "fit": "part"},             # not shown
        {"item": "m3", "topic": "t404", "fit": "part"},           # not a live topic
        {"item": "m4", "new_topic": "databases", "fit": "part"},  # a live topic's name
        {"item": "m5", "new_topic": "Old Stuff"},                 # a retired topic's name
        {"item": "m6", "new_topic": "Lonely"},                    # only one item
        {"item": "m7", "new_topic": "Fresh"},
        {"item": "m8", "new_topic": "fresh "},
    ]}
    out = plan_from(raw, items={f"m{n}" for n in range(1, 9)}, topics={"t1": "Databases"},
                    retired_names={"old stuff"})
    assert out == [("m1", "topic", "t1"), ("m4", "topic", "t1"), ("m7", "new", "Fresh"),
                   ("m8", "new", "Fresh")]


def test_plan_from_caps_new_topics():
    raw = {"placements": [{"item": f"m{n}", "new_topic": f"Group {n // 2}"} for n in range(6)]}
    out = plan_from(raw, items={f"m{n}" for n in range(6)}, topics={}, max_new=2)
    assert {name for _, _, name in out} == {"Group 0", "Group 1"}


dspy = pytest.importorskip("dspy")

import rlm_harness.runtime as rt
from rlm_harness import RLMConfig, TraceRecorder
from rlm_harness.memory_tasks import OrganiseBatch, arun_step, organize_pending


def _answer(placements, *, seen=None, raises=None):
    async def run(payload):
        if seen is not None:
            seen.append(payload)
        if raises is not None:
            raise raises
        return {"plan_json": json.dumps({"placements": placements})}
    return run


def _organize(store, **kw):
    return asyncio.run(organize_pending(store, **kw))


def test_organize_opens_topics_places_items_and_records_leftovers(store):
    a, b = _item(store, title="A"), _item(store, title="B")
    c = _item(store, title="C")
    existing = store.create_topic("Databases")
    store.place(_item(store, title="D"), existing.id)
    seen: list = []
    report = _organize(store, rounds=1, run=_answer([
        {"item": a, "new_topic": "Containers"}, {"item": b, "new_topic": "Containers"},
    ], seen=seen))
    assert len(report.created) == 1 and sorted(report.placed) == sorted(
        [(a, report.created[0]), (b, report.created[0])])
    assert report.left == [c] and store.topic(report.created[0]).provenance == "inferred"
    payload = seen[0]
    assert {i["id"] for i in json.loads(payload["items"])} == {a, b, c}
    assert json.loads(payload["topics"])[0]["holds"] == ["D"]


def test_an_item_left_is_not_asked_about_until_the_topics_change(store):
    a = _item(store)
    _organize(store, rounds=1, run=_answer([]))
    seen: list = []
    _organize(store, rounds=1, run=_answer([], seen=seen))
    assert seen == [], "left against this topic set"
    store.create_topic("New")
    _organize(store, rounds=1, run=_answer([], seen=seen))
    assert json.loads(seen[0]["items"])[0]["id"] == a


def test_organize_never_places_a_pair_a_person_took_apart(store):
    a = _item(store)
    topic = store.create_topic("T")
    store.place(a, topic.id)
    store.unplace(a, topic.id)
    report = _organize(store, rounds=1, run=_answer([{"item": a, "topic": topic.id, "fit": "subject"}]))
    assert report.placed == [] and report.left == [a], "the pair is skipped before any write"


def test_organize_stops_after_a_round_that_places_nothing(store):
    _item(store)
    seen: list = []
    report = _organize(store, rounds=3, run=_answer([], seen=seen))
    assert len(seen) == 1 and report.rounds == 1


def test_a_failed_call_records_nothing_left(store):
    a = _item(store)
    events: list = []
    report = _organize(store, run=_answer([], raises=RuntimeError("down")), on_progress=events.append)
    assert "down" in report.error and store.left_items() == []
    assert events == [{"round": 1, "ok": False, "error": report.error}]
    assert report.left == [] and a


def test_organize_stops_before_a_call(store):
    _item(store)
    seen: list = []
    report = _organize(store, run=_answer([], seen=seen), should_stop=lambda: True)
    assert report.stopped and seen == []


def test_organize_only_shows_items_with_a_summary(store):
    item = store.add("labelled but not summarised", provenance="observed", origin="o")
    store.label(item.id, title="t", tags=["x"], provenance="inferred")
    seen: list = []
    _organize(store, run=_answer([], seen=seen))
    assert seen == [] and item


def test_organize_rows_carry_the_run_id(store, tmp_path):
    a, b = _item(store), _item(store)
    labels: list = []

    def recorder_for(label):
        labels.append(label)
        return TraceRecorder(str(tmp_path / "t.jsonl"), run_id=f"run-{label}")

    _organize(store, rounds=1, recorder_for=recorder_for, run=_answer([
        {"item": a, "new_topic": "Group"}, {"item": b, "new_topic": "Group"}]))
    assert labels == ["organize:round1"]
    with _raw(store) as conn:
        assert {r[0] for r in conn.execute("SELECT run_id FROM topic_members")} == {"run-organize:round1"}
        assert conn.execute("SELECT run_id FROM topics").fetchone()[0] == "run-organize:round1"


def test_organize_treats_landing_as_unfiled(store):
    landing = store.create_topic("Inbox")
    a = _item(store)
    store.place(a, landing.id)
    seen: list = []
    _organize(store, rounds=1, landing=landing.id, run=_answer([], seen=seen))
    assert json.loads(seen[0]["items"])[0]["id"] == a
    assert json.loads(seen[0]["topics"]) == [], "the landing topic is not offered"


def _configure(lm):
    rt.configure(RLMConfig(main_model="x", sub_model="x", interpreter="mock", observe=False),
                 main_lm=lm, sub_lm=lm)


def test_organise_batch_is_one_predict_call_through_arun_step():
    from dspy.utils.dummies import DummyLM

    plan = json.dumps({"placements": [{"item": "m1", "new_topic": "X"}]})
    _configure(DummyLM([{"plan_json": plan}], adapter=dspy.JSONAdapter()))
    result = asyncio.run(OrganiseBatch().arun(items="[]", topics="[]"))
    assert json.loads(result.plan_json)["placements"][0]["item"] == "m1"
    _configure(DummyLM([{"plan_json": plan}], adapter=dspy.JSONAdapter()))
    out = asyncio.run(arun_step({"step": "organise_batch", "items": "[]", "topics": "[]",
                                 "language": "", "max_new_topics": 3}))
    assert json.loads(out["plan_json"])["placements"][0]["new_topic"] == "X"


# ---- round-one review: behaviours pinned ------------------------------------------------------

def test_a_lone_item_gets_its_topic_when_a_second_one_arrives(store):
    """A leftover is shown again beside a new arrival; without that, items on one subject that
    arrive in separate passes were never grouped."""
    first = _item(store, title="Bread")
    _organize(store, rounds=1, run=_answer([]))
    assert store.left_items() == [first]
    second = _item(store, title="Sourdough")
    seen: list = []
    report = _organize(store, rounds=1, run=_answer(
        [{"item": first, "new_topic": "Baking"}, {"item": second, "new_topic": "Baking"}], seen=seen))
    assert {i["id"] for i in json.loads(seen[0]["items"])} == {first, second}
    assert sorted(i for i, _ in report.placed) == sorted([first, second])


def test_the_next_round_rechecks_leftovers_against_the_topics_this_one_opened(store):
    a, b, c = _item(store), _item(store), _item(store)
    seen: list = []
    answers = iter([
        [{"item": a, "new_topic": "Rocks"}, {"item": b, "new_topic": "Rocks"}],
        [],
    ])

    async def run(payload):
        seen.append(payload)
        return {"plan_json": json.dumps({"placements": next(answers)})}

    report = _organize(store, rounds=2, run=run)
    assert report.rounds == 2 and json.loads(seen[1]["items"])[0]["id"] == c
    assert report.left == [c], "one entry, though it was left in both rounds"
    assert [t["name"] for t in json.loads(seen[1]["topics"])] == ["Rocks"]


def test_a_retired_topics_name_is_never_reopened(store):
    out = plan_from({"placements": [{"item": "m1", "new_topic": "Old Stuff"},
                                    {"item": "m2", "new_topic": "old  stuff"}]},
                    items={"m1", "m2"}, topics={}, retired_names={"old stuff"})
    assert out == []
    old = store.create_topic("Old Stuff")
    store.retire_topic(old.id)
    a, b = _item(store), _item(store)
    report = _organize(store, rounds=1, run=_answer(
        [{"item": a, "new_topic": "Old Stuff"}, {"item": b, "new_topic": "Old Stuff"}]))
    assert report.created == [] and report.placed == []


def test_an_item_sent_elsewhere_does_not_count_toward_a_new_topic():
    out = plan_from({"placements": [
        {"item": "m1", "topic": "t1", "fit": "subject"},
        {"item": "m1", "new_topic": "X"},
        {"item": "m2", "new_topic": "X"},
    ]}, items={"m1", "m2"}, topics={"t1": "One"})
    assert out == [("m1", "topic", "t1")], "X would open with only m2"


def test_a_new_topic_left_with_one_member_is_retired(store):
    a, b = _item(store), _item(store)

    async def run(payload):
        store.purge([b])   # b vanishes while the model is thinking
        return {"plan_json": json.dumps({"placements": [
            {"item": a, "new_topic": "Pair"}, {"item": b, "new_topic": "Pair"}]})}

    report = _organize(store, rounds=1, run=run)
    assert len(report.created) == 1 and report.retired == report.created
    assert store.topics() == []


def test_a_persons_removal_wins_in_the_view_even_against_a_later_model_row(store):
    """Defence in depth: the write path refuses this, and the view must not show it either."""
    item = _item(store)
    topic = store.create_topic("T")
    store.unplace(item, topic.id)
    with _raw(store) as conn:
        conn.execute("INSERT INTO topic_members (item_id, topic_id, member, provenance, created_at) "
                     "VALUES (?, ?, 1, 'inferred', 9e9)", (int(item[1:]), int(topic.id[1:])))
    assert store.topic_items(topic.id) == []


def test_topic_graph_counts_what_it_drops(store):
    topics = [store.create_topic(f"T{n}") for n in range(4)]
    for topic in topics:
        store.place(_item(store, entities=["Shared"]), topic.id)
    graph = topic_graph(store, per_topic=1)
    assert len(graph.bridges) < 6 and sum(graph.omitted.values()) > 0


def test_a_pass_retiring_its_own_topic_does_not_burn_the_name(store):
    a, b = _item(store), _item(store)

    async def first(payload):
        store.purge([b])
        return {"plan_json": json.dumps({"placements": [
            {"item": a, "new_topic": "Baking"}, {"item": b, "new_topic": "Baking"}]})}

    report = _organize(store, rounds=1, run=first)
    assert report.placed == []
    assert any(item == a and "retired" in reason for item, _, reason in report.refused)
    assert store.said_retired_names() == set()
    c = _item(store)
    report = _organize(store, rounds=1, run=_answer(
        [{"item": a, "new_topic": "Baking"}, {"item": c, "new_topic": "Baking"}]))
    assert len(report.created) == 1 and len(report.placed) == 2


def test_only_a_persons_retirement_keeps_a_name_closed(store):
    mine = store.create_topic("Mine")
    theirs = store.create_topic("Theirs")
    store.retire_topic(mine.id)
    store.retire_topic(theirs.id, provenance="inferred")
    assert store.said_retired_names() == {"mine"}
    store.restore_topic(mine.id)
    assert store.said_retired_names() == set()


def test_candidates_skip_an_item_gone_before_it_is_read(store, monkeypatch):
    a = _item(store)
    real = store._labels_by_item

    def with_a_ghost():
        labels = real()
        labels["m999"] = labels[a]
        return labels

    monkeypatch.setattr(store, "_labels_by_item", with_a_ghost)
    seen: list = []
    _organize(store, rounds=1, run=_answer([], seen=seen))
    assert [i["id"] for i in json.loads(seen[0]["items"])] == [a]


def test_a_model_does_not_overturn_a_persons_restore(store):
    topic = store.create_topic("T", provenance="inferred")
    store.retire_topic(topic.id)
    store.restore_topic(topic.id)
    with pytest.raises(ValueError, match="person already decided"):
        store.retire_topic(topic.id, provenance="inferred")
    assert not store.topic(topic.id).retired


def test_a_retired_underfilled_topic_keeps_no_member(store):
    a, b = _item(store), _item(store)

    async def run(payload):
        store.purge([b])
        return {"plan_json": json.dumps({"placements": [
            {"item": a, "new_topic": "Pair"}, {"item": b, "new_topic": "Pair"}]})}

    report = _organize(store, rounds=1, run=run)
    store.restore_topic(report.retired[0])
    assert store.topic_items(report.retired[0]) == []


def test_organize_never_files_into_a_dismissed_pair(store):
    a = _item(store)
    topic = store.create_topic("T")
    other = store.create_topic("U")
    store.dismiss(a, topic.id)
    report = _organize(store, rounds=1, run=_answer([
        {"item": a, "topic": topic.id, "fit": "subject"},
        {"item": a, "topic": other.id, "fit": "part"},
    ]))
    assert report.placed == [(a, other.id)], "the dismissed pair is skipped, the next entry applies"


def test_an_item_placed_in_a_later_round_is_not_reported_as_left(store):
    a, b, c = _item(store), _item(store), _item(store)
    answers = iter([
        [{"item": a, "new_topic": "Rocks"}, {"item": b, "new_topic": "Rocks"}],
        None,
    ])

    async def run(payload):
        plan = next(answers)
        if plan is None:
            rocks = json.loads(payload["topics"])[0]["id"]
            plan = [{"item": c, "topic": rocks, "fit": "subject"}]
        return {"plan_json": json.dumps({"placements": plan})}

    report = _organize(store, rounds=2, run=run)
    assert {i for i, _ in report.placed} == {a, b, c} and report.left == []


def test_organise_batch_survives_a_bad_new_topic_cap():
    from dspy.utils.dummies import DummyLM

    _configure(DummyLM([{"plan_json": "{}"}], adapter=dspy.JSONAdapter()))
    assert asyncio.run(OrganiseBatch().arun(max_new_topics="x")).plan_json == "{}"
