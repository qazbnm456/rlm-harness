"""Names that mean the same thing: aliases, alignment and the entity graph. Offline."""

from __future__ import annotations

import asyncio
import contextlib
import json
import sqlite3

import pytest

from rlm_harness import memory as memory_module
from rlm_harness.memory import _MIGRATIONS, MemoryStore
from rlm_harness.memory_organise import known_listing


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path / "m.db")


def _raw(store: MemoryStore):
    return contextlib.closing(sqlite3.connect(store.path, isolation_level=None))


def _labelled(store, *entity_sets, text="x"):
    ids = []
    for entities in entity_sets:
        item = store.add(text, provenance="observed", origin="o")
        store.label(item.id, summary="s", entities=list(entities), provenance="inferred")
        ids.append(item.id)
    return ids


# ---- merge and resolve --------------------------------------------------------------------------

def test_a_merge_is_read_everywhere_an_entity_is(store):
    a, b = _labelled(store, ["Kubernetes", "Docker"], ["k8s"])
    _labelled(store, ["Kubernetes"])
    assert store.merge("k8s", "Kubernetes") == "Kubernetes"
    assert store.resolve("K8S") == "Kubernetes" and store.resolve("Docker") == "Docker"
    assert store.labels(b).entities == ("Kubernetes",)
    assert dict(store.label_counts("entities"))["Kubernetes"] == 3, "a merged pair counts as one"
    assert {i.id for i in store.search("", entities=["k8s"])} >= {a, b}
    assert store.aliases() == {"k8s": "Kubernetes"}


def test_the_canonical_is_followed_so_no_chain_forms(store):
    _labelled(store, ["Kubernetes"], ["k8s"], ["kube"])
    store.merge("k8s", "Kubernetes")
    assert store.merge("kube", "k8s") == "Kubernetes", "followed to the end, never a chain"
    assert set(store.aliases().values()) == {"Kubernetes"}


@pytest.mark.parametrize("alias,canonical,reason", [
    ("Unknown", "Kubernetes", "in use"),
    ("Kubernetes", "kubernetes", "already one name"),
])
def test_merge_refuses_with_a_reason(store, alias, canonical, reason):
    _labelled(store, ["Kubernetes"], ["k8s"])
    with pytest.raises(ValueError, match=reason):
        store.merge(alias, canonical)


def test_a_canonical_cannot_become_an_alias_and_an_alias_cannot_be_repointed(store):
    _labelled(store, ["Kubernetes"], ["k8s"], ["Nomad"])
    store.merge("k8s", "Kubernetes")
    with pytest.raises(ValueError, match="canonical of another"):
        store.merge("Kubernetes", "Nomad")
    with pytest.raises(ValueError, match="already an alias"):
        store.merge("k8s", "Nomad")


def test_merge_rejects_non_str_names(store):
    with pytest.raises(TypeError):
        store.merge(3, "x")


def test_the_canonical_is_recorded_in_its_most_used_spelling(store):
    _labelled(store, ["PostgreSQL"], ["PostgreSQL"], ["postgresql"], ["Postgres"])
    assert store.merge("Postgres", "postgresql") == "PostgreSQL"


# ---- who decides --------------------------------------------------------------------------------

def test_an_inferred_merge_needs_its_name_in_the_batch(store):
    _labelled(store, ["Kubernetes"], ["k8s"])
    with pytest.raises(ValueError, match="asked about"):
        store.merge("k8s", "Kubernetes", provenance="inferred", batch=["something else"])
    assert store.merge("k8s", "Kubernetes", provenance="inferred", batch=["k8s"]) == "Kubernetes"


def test_a_persons_unmerge_stays_unmerged(store):
    _labelled(store, ["Kubernetes"], ["k8s"])
    store.merge("k8s", "Kubernetes", provenance="inferred", batch=["k8s"])
    store.unmerge("k8s")
    assert store.aliases() == {}, "the model's earlier merge must not show through"
    with pytest.raises(ValueError, match="person already decided"):
        store.merge("k8s", "Kubernetes", provenance="inferred", batch=["k8s"])
    with pytest.raises(ValueError, match="person already decided"):
        store.unmerge("k8s", provenance="inferred")


def test_a_said_unmerge_ahead_of_time_keeps_two_names_apart(store):
    _labelled(store, ["Java"], ["JavaScript"])
    store.unmerge("JavaScript")
    with pytest.raises(ValueError, match="person already decided"):
        store.merge("JavaScript", "Java", provenance="inferred", batch=["JavaScript"])


def test_a_person_can_merge_again_after_unmerging(store):
    _labelled(store, ["Kubernetes"], ["k8s"])
    store.merge("k8s", "Kubernetes")
    store.unmerge("k8s")
    assert store.merge("k8s", "Kubernetes") == "Kubernetes"
    assert store.aliases() == {"k8s": "Kubernetes"}


def test_an_inferred_unmerge_needs_a_live_alias(store):
    _labelled(store, ["Kubernetes"], ["k8s"])
    with pytest.raises(ValueError, match="not an alias"):
        store.unmerge("k8s", provenance="inferred")


# ---- append-only --------------------------------------------------------------------------------

@pytest.mark.parametrize("sql", [
    "UPDATE entity_aliases SET canonical = 'x' WHERE id = 1",
    "DELETE FROM entity_aliases WHERE id = 1",
    ("INSERT OR REPLACE INTO entity_aliases (id, alias, canonical, provenance, created_at) "
     "VALUES (1, 'k8s', 'evil', 'said', 0)"),
    "UPDATE entity_seen SET name = 'x' WHERE id = 1",
    "DELETE FROM entity_seen WHERE id = 1",
    "INSERT OR REPLACE INTO entity_seen (id, name, created_at) VALUES (1, 'evil', 0)",
])
def test_a_stray_statement_cannot_rewrite_an_alias_or_a_seen_name(store, sql):
    _labelled(store, ["Kubernetes"], ["k8s"])
    store.merge("k8s", "Kubernetes")
    store.mark_seen(["k8s"])
    with _raw(store) as conn, pytest.raises(sqlite3.IntegrityError):
        conn.execute(sql)
    assert store.aliases() == {"k8s": "Kubernetes"}
    assert store.unseen_entities() == ["Kubernetes"]


def test_mark_seen_inserts_each_name_once(store):
    _labelled(store, ["Kubernetes"])
    store.mark_seen(["Kubernetes", "kubernetes"])
    store.mark_seen(["KUBERNETES"])
    with _raw(store) as conn:
        assert conn.execute("SELECT COUNT(*) FROM entity_seen").fetchone()[0] == 1
    with pytest.raises(TypeError):
        store.mark_seen("Kubernetes")


# ---- search and the graph -----------------------------------------------------------------------

def test_a_query_that_is_an_entity_name_searches_all_its_names(store):
    item = store.add("Deploying with Kubernetes", provenance="observed", origin="o")
    store.label(item.id, summary="s", entities=["Kubernetes"], provenance="inferred")
    other = store.add("notes", provenance="observed", origin="o")
    store.label(other.id, summary="s", entities=["k8s"], provenance="inferred")
    assert [i.id for i in store.search("k8s")] == [other.id], "before the merge, only its own name"
    store.merge("k8s", "Kubernetes")
    assert {i.id for i in store.search("k8s")} == {item.id, other.id}


def test_entity_graph_counts_pairs_through_aliases(store):
    _labelled(store, ["Kubernetes", "Docker"], ["k8s", "Docker"], ["Docker", "Linux"])
    store.merge("k8s", "Kubernetes")
    graph = store.entity_graph()
    assert graph[0] == ("Docker", "Kubernetes", 2)
    assert ("Docker", "Linux", 1) in graph
    assert len(store.entity_graph(limit=1)) == 1


def test_unseen_entities_skips_names_already_considered(store):
    _labelled(store, ["Kubernetes"], ["Kubernetes"], ["k8s"])
    assert store.unseen_entities() == ["Kubernetes", "k8s"]
    store.mark_seen(["kubernetes"])
    assert store.unseen_entities() == ["k8s"]


def test_known_listing_grounds_an_alias_as_written(store):
    _labelled(store, ["Kubernetes"], ["k8s"], ["kube"])
    store.merge("k8s", "Kubernetes")
    store.merge("kube", "Kubernetes")
    text, names = known_listing(store)
    assert names["k8s"] == "k8s" and names["kube"] == "kube", "kept as written, resolved on read"
    assert text == "Entities in use: Kubernetes", "each name in use once; aliases are not shown"


def test_a_purged_canonical_is_neither_shown_nor_grounded(store):
    jane, j = _labelled(store, ["Jane Roe"], ["J. Roe"])
    _labelled(store, ["Docker"])
    store.merge("J. Roe", "Jane Roe")
    store.purge([jane, j])
    text, names = known_listing(store)
    assert "Jane Roe" not in text and "j. roe" not in names and "jane roe" not in names


def test_every_alias_of_a_group_names_the_canonical_the_same_way(store):
    _labelled(store, ["Kubernetes"], ["k8s"], ["kubectl"])
    store.merge("k8s", "Kubernetes")
    _labelled(store, ["kubernetes"], ["kubernetes"], ["kubernetes"])
    assert store.merge("kubectl", "kubernetes") == "Kubernetes"
    counts = dict(store.label_counts("entities"))
    assert "kubernetes" not in counts and counts["Kubernetes"] == 6


def test_a_canonical_no_item_names_any_more_is_still_mergeable_into(store):
    alpha, _a1, _a2 = _labelled(store, ["Alpha"], ["A-1"], ["A one"])
    store.merge("A-1", "Alpha")
    store.purge([alpha])
    assert store.merge("A one", "Alpha") == "Alpha"


def test_a_step_two_database_is_upgraded(tmp_path, monkeypatch):
    path = tmp_path / "m.db"
    with monkeypatch.context() as patch:
        patch.setattr(memory_module, "_MIGRATIONS", _MIGRATIONS[:2])
        patch.setattr(memory_module, "_SCHEMA_VERSION", 2)
        old = MemoryStore(path)
        old.add("x", provenance="observed", origin="o")
        # Written as step-2 code would: `label()` now reads the alias table, which step 2 lacks.
        with _raw(old) as conn:
            conn.execute("INSERT INTO labels (item_id, kind, value_json, provenance, created_at) "
                         "VALUES (1, 'entities', '[\"Kubernetes\", \"k8s\"]', 'inferred', 0)")
    upgraded = MemoryStore(path)
    assert upgraded.merge("k8s", "Kubernetes") == "Kubernetes"
    with contextlib.closing(sqlite3.connect(path)) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == len(_MIGRATIONS)


# ---- align_pending ------------------------------------------------------------------------------

dspy = pytest.importorskip("dspy")

import rlm_harness.runtime as rt
from rlm_harness import RLMConfig, TraceRecorder
from rlm_harness.memory_tasks import AlignEntities, EntityMerges, align_pending
from rlm_harness.testing import (
    ScriptedInterpreter,
    assert_task_repl_safe,
    scripted_lm,
    submit,
)


def _answer(*pairs, seen=None, raises=None):
    async def run(payload):
        if seen is not None:
            seen.append(payload)
        if raises is not None:
            raise raises
        return {"merges": [{"alias": a, "canonical": c} for a, c in pairs]}
    return run


def _align(store, **kw):
    return asyncio.run(align_pending(store, **kw))


def test_align_merges_what_it_was_asked_about_and_reports_refusals(store):
    _labelled(store, ["Kubernetes"], ["Kubernetes"], ["k8s"], ["Nomad"])
    store.mark_seen(["Kubernetes", "Nomad"])
    seen: list = []
    report = _align(store, run=_answer(("k8s", "Kubernetes"), ("Nomad", "Kubernetes"), seen=seen))
    assert report.asked == ["k8s"]
    assert report.accepted == [("k8s", "Kubernetes")]
    assert report.refused[0][0] == "Nomad" and "asked about" in report.refused[0][2]
    assert json.loads(seen[0]["new_names"]) == ["k8s"]
    assert "Kubernetes\t2" in seen[0]["known"]
    assert store.unseen_entities() == []


def test_align_marks_the_batch_seen_even_when_nothing_merges(store):
    _labelled(store, ["Kubernetes"], ["Docker"])
    events: list = []
    report = _align(store, run=_answer(), on_progress=events.append)
    assert report.accepted == [] and report.asked and store.unseen_entities() == []
    assert events == [{"batch": 2, "ok": True, "error": ""}]


def test_align_makes_no_call_with_nothing_unseen(store):
    seen: list = []
    report = _align(store, run=_answer(seen=seen))
    assert seen == [] and report.asked == []


def test_a_failed_call_marks_nothing_seen(store):
    _labelled(store, ["Kubernetes"], ["k8s"])
    events: list = []
    report = _align(store, run=_answer(raises=RuntimeError("down")), on_progress=events.append)
    assert "down" in report.error and store.unseen_entities() == ["Kubernetes", "k8s"]
    assert events == [{"batch": 2, "ok": False, "error": report.error}]


def test_align_never_overrides_a_persons_decision(store):
    _labelled(store, ["Java"], ["JavaScript"])
    store.unmerge("JavaScript")
    report = _align(store, run=_answer(("JavaScript", "Java")))
    assert report.accepted == [] and "person already decided" in report.refused[0][2]
    assert store.aliases() == {}


def test_align_stops_before_the_call(store):
    _labelled(store, ["Kubernetes"], ["k8s"])
    seen: list = []
    report = _align(store, run=_answer(seen=seen), should_stop=lambda: True)
    assert report.stopped and seen == [] and store.unseen_entities() == ["Kubernetes", "k8s"]


def test_align_ignores_a_malformed_answer(store):
    _labelled(store, ["Kubernetes"], ["k8s"])

    async def run(payload):
        return {"merges": [{"alias": 3}, "nonsense"]}

    report = _align(store, run=run)
    assert report.accepted == [] and len(report.refused) == 2


def test_align_rows_carry_the_run_id(store, tmp_path):
    _labelled(store, ["Kubernetes"], ["k8s"])
    seen: list = []
    _align(store, run=_answer(("k8s", "Kubernetes"), seen=seen),
           recorder_for=lambda label: TraceRecorder(str(tmp_path / "t.jsonl"), run_id=f"run-{label}"))
    assert seen[0]["run_id"] == "run-align"
    with _raw(store) as conn:
        assert conn.execute("SELECT run_id FROM entity_aliases").fetchone()[0] == "run-align"


def _configure(lm):
    rt.configure(RLMConfig(main_model="x", sub_model="x", interpreter="mock", observe=False),
                 main_lm=lm, sub_lm=lm)


def test_align_entities_runs_through_the_rlm_loop():
    merges = {"merges": [{"alias": "k8s", "canonical": "Kubernetes"}]}
    _configure(scripted_lm([{"reasoning": "done", "code": f"SUBMIT(merges={merges!r})"}]))
    task = AlignEntities(interpreter=ScriptedInterpreter([submit({"merges": merges})]))
    assert task._config.max_iterations == AlignEntities.STEPS
    result = asyncio.run(task.arun(known="Kubernetes\t1", new_names='["k8s"]'))
    assert isinstance(result, EntityMerges) and result.merges[0].alias == "k8s"


def test_align_entities_is_repl_safe_and_carries_its_validator():
    _configure(scripted_lm([]))
    task = AlignEntities()
    assert_task_repl_safe(task)
    assert "validate_entitymerges" in [t.__name__ for t in task.resolved_tools]
    assert "validate_entitymerges" in AlignEntities.instructions


def test_a_model_cannot_rejoin_a_separated_pair_the_other_way_round(store):
    _labelled(store, ["Kubernetes"], ["k8s"])
    store.merge("k8s", "Kubernetes")
    store.unmerge("k8s")
    with pytest.raises(ValueError, match="person already decided"):
        store.merge("Kubernetes", "k8s", provenance="inferred", batch=["Kubernetes"])
    report = _align(store, run=_answer(("Kubernetes", "k8s")))
    assert report.accepted == [] and store.aliases() == {}


def test_an_unmerge_restores_labels_written_while_the_merge_was_live(store):
    from rlm_harness.memory_tasks import distill_pending

    _labelled(store, ["JavaScript"], ["Java"])
    store.merge("Java", "JavaScript")
    item = store.add("Notes on the Java virtual machine", provenance="observed", origin="o")

    async def run(payload):
        return {"title": "t", "summary": "s", "tags": [], "entities": ["Java"]}

    asyncio.run(distill_pending(store, item_ids=[item.id], run=run))
    assert store.labels(item.id).entities == ("JavaScript",), "read through the merge"
    store.unmerge("Java")
    assert store.labels(item.id).entities == ("Java",), "the wrong merge is undone in this label too"


def test_label_counts_count_an_item_once_per_entity_group(store):
    _labelled(store, ["Kubernetes", "k8s"])
    store.merge("k8s", "Kubernetes")
    assert store.label_counts("entities") == [("Kubernetes", 1)]


def test_the_query_expansion_keeps_the_term_cap(store, monkeypatch):
    names = [f"name{n} word{n} extra{n}" for n in range(40)]
    _labelled(store, names[:1], *[[n] for n in names[1:]])
    for n in names[1:]:
        store.merge(n, names[0])
    sent: list = []
    real = memory_module._match_expression
    monkeypatch.setattr(memory_module, "_match_expression",
                        lambda terms: sent.append(list(terms)) or real(terms))
    store.search(names[0])
    assert sent and len(sent[0]) == memory_module._MAX_QUERY_TERMS, "40 names x 3 words, capped"


def test_an_alias_grounds_in_its_most_used_spelling(store):
    _labelled(store, ["Kubernetes"], ["k8s"], ["k8s"], ["k8s"], ["K8s"])
    store.merge("k8s", "Kubernetes")
    assert known_listing(store)[1]["k8s"] == "k8s"


def test_a_group_stays_in_use_while_any_item_names_it(store):
    """Aliases are vocabulary a purge leaves (documented); while an item still carries the alias,
    the group is in use and reads as its canonical."""
    jane, j = _labelled(store, ["Jane Roe"], ["J. Roe"])
    store.merge("J. Roe", "Jane Roe")
    store.purge([jane])
    text, names = known_listing(store)
    assert text == "Entities in use: Jane Roe" and names["j. roe"] == "J. Roe"
    assert store.labels(j).entities == ("Jane Roe",)


def test_an_alias_no_item_writes_keeps_the_models_spelling(store):
    from rlm_harness.memory_tasks import distill_pending

    _kube, k8s_item = _labelled(store, ["Kubernetes"], ["k8s"])
    store.merge("k8s", "Kubernetes")
    store.label(k8s_item, entities=["Docker"], provenance="said")   # no item writes `k8s` now
    assert known_listing(store)[1]["k8s"] == ""
    item = store.add("Running K8S at home", provenance="observed", origin="o")

    async def run(payload):
        return {"title": "t", "summary": "s", "tags": [], "entities": ["K8S"]}

    asyncio.run(distill_pending(store, item_ids=[item.id], run=run))
    with _raw(store) as conn:
        stored = conn.execute("SELECT value_json FROM labels WHERE item_id = ? AND kind = 'entities'",
                              (int(item.id[1:]),)).fetchone()[0]
    assert json.loads(stored) == ["K8S"], "the model's own spelling, not the folded key"
    assert store.labels(item.id).entities == ("Kubernetes",)


def test_a_chained_answer_is_followed_to_its_end(store):
    _labelled(store, ["Kubernetes"], ["k8s"], ["kube"])
    store.mark_seen(["Kubernetes"])
    report = _align(store, run=_answer(("k8s", "kube"), ("kube", "Kubernetes")))
    assert sorted(report.accepted) == [("k8s", "Kubernetes"), ("kube", "Kubernetes")]
    assert store.resolve("k8s") == store.resolve("kube") == "Kubernetes"


def test_a_cycle_in_an_answer_does_not_loop(store):
    _labelled(store, ["a-one"], ["a-two"])
    report = _align(store, run=_answer(("a-one", "a-two"), ("a-two", "a-one")))
    assert len(report.accepted) + len(report.refused) == 2


def test_a_database_error_on_a_merge_leaves_that_name_unseen(store, monkeypatch):
    _labelled(store, ["Kubernetes"], ["k8s"], ["Docker"])
    real = store.merge

    def flaky(alias, canonical, **kw):
        if alias == "k8s":
            raise sqlite3.OperationalError("database is locked")
        return real(alias, canonical, **kw)

    monkeypatch.setattr(store, "merge", flaky)
    report = _align(store, run=_answer(("k8s", "Kubernetes")))
    assert "database error" in report.refused[0][2]
    assert store.unseen_entities() == ["k8s"], "asked about again next time"


def test_a_chain_never_merges_past_a_name_a_person_keeps_apart(store):
    """Resolving the answer ahead of the store skipped the refused middle link and merged
    `JVM Java` straight into `JavaScript`. A model touches no name a person decided about, in any
    role, so neither link lands."""
    _labelled(store, ["Java"], ["JavaScript"], ["JVM Java"])
    store.unmerge("Java")
    store.mark_seen(["JavaScript"])
    report = _align(store, run=_answer(("JVM Java", "Java"), ("Java", "JavaScript")))
    assert store.aliases() == {} and report.accepted == []
    assert all("person already decided" in reason for _, _, reason in report.refused)


def test_a_chain_stops_at_a_link_the_store_refuses(store):
    """The middle link is refused for its own reason (`Java` was not asked about), so the chain
    stops at `Java` instead of being carried on to `JavaScript`."""
    _labelled(store, ["Java"], ["JavaScript"], ["JVM Java"])
    store.mark_seen(["Java", "JavaScript"])
    report = _align(store, run=_answer(("JVM Java", "Java"), ("Java", "JavaScript")))
    assert store.aliases() == {"jvm java": "Java"}
    assert [a for a, _, _ in report.refused] == ["Java"]


def test_a_cycle_that_skips_the_alias_still_ends(store):
    _labelled(store, ["n-a"], ["n-b"], ["n-c"])
    report = _align(store, run=_answer(("n-a", "n-b"), ("n-b", "n-c"), ("n-c", "n-b")))
    assert len(report.accepted) + len(report.refused) == 3
