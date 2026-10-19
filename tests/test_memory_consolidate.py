"""A finished run into memory, and what a run did with memory. Offline: `run=` doubles and a
ScriptedInterpreter; hand-built traces for the measurement."""

from __future__ import annotations

import asyncio
import contextlib
import sqlite3

import pytest

from rlm_harness.memory import MemoryStore
from rlm_harness.memory_organise import RUN_CHARS, render_run, run_sources
from rlm_harness.metrics import MemoryUse, compute_memory_use, compute_memory_use_by_run

dspy = pytest.importorskip("dspy")

import rlm_harness.runtime as rt
from rlm_harness import RLMConfig, TraceRecorder
from rlm_harness.memory_tasks import (
    PROPOSAL_CHARS,
    ConsolidateRun,
    MemoryProposals,
    consolidate_run,
)
from rlm_harness.testing import ScriptedInterpreter, assert_task_repl_safe, scripted_lm, submit
from rlm_harness.tools.memory import (
    make_memory_get_tool,
    make_memory_note_tool,
    make_memory_search_tool,
)

PAGE = ("The quarterly filing deadline moved to the fifteenth of each month.\n"
        "Late filers pay a flat penalty plus interest on the unpaid balance.")


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path / "m.db")


def _event(step, kind, payload, *, ts=None, run_id="run-1"):
    return {"run_id": run_id, "step_id": step, "ts": float(step if ts is None else ts), "type": kind,
            "payload": payload}


def _trace(*, result=PAGE, tool="fetch_page"):
    return [
        _event(0, "run_start", {"meta": {"task": "tax deadlines"}}),
        _event(1, "tool_call", {"tool": tool, "args": {"url": "https://example.test"}, "ok": True,
                                "result": result}),
        _event(2, "main_step", {"turn": 0, "reasoning": "read it", "code": "print(page)",
                                "output": "REPL says: the deadline is the first of the month, always"}),
        _event(3, "result", {"output": "Filing is due on the fifteenth."}),
        _event(4, "run_end", {"ok": True}),
    ]


def _answer(*proposals, seen=None):
    async def run(payload):
        if seen is not None:
            seen.append(payload)
        return {"proposals": list(proposals)}
    return run


def _consolidate(store, events=None, **kw):
    kw.setdefault("run", _answer())
    return asyncio.run(consolidate_run(store, _trace() if events is None else events, **kw))


# ---- sources and rendering ----------------------------------------------------------------------


def test_a_tool_result_is_a_source_but_trusted_only_when_the_host_names_its_tool():
    assert run_sources(_trace())["step:1"].trusted is False
    assert run_sources(_trace(), trusted_tools=["fetch_page"])["step:1"].trusted is True


def test_a_model_backed_tool_is_never_trusted_even_when_named():
    events = _trace(tool="model:helper")
    assert run_sources(events, trusted_tools=["model:helper"])["step:1"].trusted is False


def test_evidence_is_trusted_unless_the_host_says_otherwise():
    sources = run_sources([], evidence={"page": PAGE, "forum": ("a post", False)})
    assert sources["evidence:page"].trusted is True and sources["evidence:forum"].trusted is False


def test_repl_output_is_never_a_source():
    sources = run_sources(_trace())
    assert set(sources) == {"step:1"}
    assert "REPL says" not in render_run(_trace())


def test_render_run_labels_each_source_and_keeps_the_meta_and_output():
    text = render_run(_trace(), evidence={"notes": "kept by the host"}, trusted_tools=["fetch_page"])
    assert "# run run-1" in text and "tax deadlines" in text
    assert "[step:1] (trusted) result:" in text and "[evidence:notes] (trusted):" in text
    assert "Filing is due on the fifteenth." in text


def test_render_run_is_cut_to_its_budget():
    long = "x" * (RUN_CHARS * 2)
    assert len(render_run([], evidence={"big": long}, max_chars=500)) < 600
    assert "more characters" in render_run([], evidence={"big": long}, max_chars=500)


# ---- what is stored -----------------------------------------------------------------------------


def test_a_fact_quoting_trusted_evidence_is_stored_as_observed_in_the_sources_own_words(store):
    quote = "Late filers  pay a flat penalty\nplus interest"   # the model's spacing
    report = _consolidate(store, evidence={"page": PAGE}, run=_answer(
        {"kind": "fact", "text": "Late filing costs a penalty and interest.", "quote": quote,
         "source": "evidence:page"}))
    item = store.get(report.observed[0])
    assert item.provenance == "observed" and item.text == "Late filers pay a flat penalty plus interest"
    assert item.origin == "evidence:page"
    assert item.meta == {"selected_by": "model", "run_id": "run-1", "source": "evidence:page"}


def test_a_fact_quoting_a_trusted_tool_names_its_step(store):
    report = _consolidate(store, trusted_tools=["fetch_page"], run=_answer(
        {"kind": "fact", "text": "t", "quote": "deadline moved to the fifteenth", "source": "step:1"}))
    assert store.get(report.observed[0]).origin == "trace:run-1#1"


@pytest.mark.parametrize(("proposal", "reason"), [
    ({"quote": "deadline moved to the fifteenth", "source": "step:1"}, "not a source the host trusts"),
    ({"quote": "fifteenth", "source": "evidence:page"}, "quote under 20"),
    ({"quote": "the deadline is the first of the month", "source": "evidence:page"}, "not found"),
    ({"quote": "the deadline is the first of the month", "source": "step:2"}, "no source labelled"),
    ({"quote": "deadline moved to the fifteenth"}, "no source named"),
])
def test_a_fact_the_host_cannot_stand_by_is_stored_as_inferred_with_its_reason(store, proposal, reason):
    report = _consolidate(store, evidence={"page": PAGE}, run=_answer(
        {"kind": "fact", "text": "The deadline is the fifteenth.", **proposal}))
    assert report.observed == [] and len(report.inferred) == 1
    item = store.get(report.inferred[0])
    assert item.provenance == "inferred" and item.text == "The deadline is the fifteenth."
    assert reason in report.downgraded[0][1] and reason in item.meta["downgraded"]


def test_repl_stdout_never_grounds_a_fact_even_when_quoted_exactly(store):
    report = _consolidate(store, trusted_tools=["fetch_page"], run=_answer(
        {"kind": "fact", "text": "t", "quote": "the deadline is the first of the month, always",
         "source": "step:2"}))
    assert report.observed == [] and report.downgraded


def test_a_note_is_inferred_and_a_proposal_cannot_choose_its_provenance(store):
    report = _consolidate(store, run=_answer(
        {"kind": "note", "text": "Check the filing calendar before quoting a date.", "provenance": "said"}))
    item = store.get(report.inferred[0])
    assert item.provenance == "inferred" and item.origin == "trace:run-1" and report.downgraded == []


def test_markers_are_stripped_and_empty_long_and_known_texts_are_skipped(store):
    store.add("Already known.", provenance="said", origin="o")
    report = _consolidate(store, run=_answer(
        {"kind": "note", "text": "A [[SRC:3|p1]]finding."},
        {"kind": "note", "text": "   "},
        {"kind": "note", "text": "y" * (PROPOSAL_CHARS + 1)},
        {"kind": "note", "text": "Already  known."}))
    assert store.get(report.inferred[0]).text == "A finding."
    assert [r for _, r in report.skipped] == ["empty", f"longer than {PROPOSAL_CHARS} characters",
                                              "already in memory"]


def test_a_fact_with_no_sentence_keeps_its_quote(store):
    report = _consolidate(store, run=_answer({"kind": "fact", "quote": "an unchecked claim of some length"}))
    assert store.get(report.inferred[0]).text == "an unchecked claim of some length"


def test_the_proposals_kept_are_capped(store):
    report = _consolidate(store, limit=2, run=_answer(*({"kind": "note", "text": f"note {n}"} for n in range(5))))
    assert len(report.inferred) == 2


def test_a_failed_call_stores_nothing(store):
    async def boom(payload):
        raise RuntimeError("model down")

    report = _consolidate(store, run=boom)
    assert report.error.startswith("RuntimeError") and store.search("note") == []


def test_events_from_two_runs_are_refused(store):
    with pytest.raises(ValueError, match="one run"):
        _consolidate(store, events=_trace() + [_event(9, "result", {"output": "x"}, run_id="run-2")])


def test_should_stop_before_the_call_makes_no_call(store):
    seen: list = []
    report = _consolidate(store, should_stop=lambda: True, run=_answer(seen=seen))
    assert report.stopped and seen == []


def test_on_progress_reports_each_proposal(store):
    events: list = []
    _consolidate(store, on_progress=events.append,
                 run=_answer({"kind": "note", "text": "one"}, {"kind": "note", "text": ""}))
    assert [e["proposal"] for e in events] == [0, 1] and all(e["ok"] for e in events)


def test_the_call_sees_the_run_and_related_memory_and_rows_carry_its_run(store, tmp_path):
    related = store.add("Filing deadlines move when the tax office says so.", provenance="said", origin="o")
    seen: list = []
    report = _consolidate(
        store, run=_answer({"kind": "note", "text": "Deadlines are set by the tax office."}, seen=seen),
        recorder_for=lambda label: TraceRecorder(str(tmp_path / "c.jsonl"), run_id=label))
    assert seen[0]["step"] == "consolidate_run" and seen[0]["run_id"] == "consolidate:run-1"
    assert "[step:1]" in seen[0]["run"] and related.id in seen[0]["known"]
    with contextlib.closing(sqlite3.connect(store.path)) as conn:
        assert conn.execute("SELECT run_id FROM items WHERE id = ?",
                            (int(report.inferred[0][1:]),)).fetchone()[0] == "consolidate:run-1"


# ---- the model step -----------------------------------------------------------------------------


def _configure(lm):
    rt.configure(RLMConfig(main_model="x", sub_model="x", interpreter="mock", observe=False),
                 main_lm=lm, sub_lm=lm)


def test_consolidate_run_task_runs_through_the_rlm_loop():
    answer = {"proposals": [{"kind": "note", "text": "keep this", "quote": "", "source": ""}]}
    _configure(scripted_lm([{"reasoning": "done", "code": f"SUBMIT(proposals={answer!r})"}]))
    task = ConsolidateRun(interpreter=ScriptedInterpreter([submit({"proposals": answer})]))
    assert task._config.max_iterations == ConsolidateRun.STEPS
    result = asyncio.run(task.arun(run="# run r", known=""))
    assert isinstance(result, MemoryProposals) and result.proposals[0].text == "keep this"


def test_consolidate_run_task_is_repl_safe():
    _configure(scripted_lm([]))
    assert_task_repl_safe(ConsolidateRun())


# ---- what a run did with memory -----------------------------------------------------------------


def test_the_memory_tools_say_whether_they_read_or_wrote(store, tmp_path):
    store.add("deadline notes", provenance="said", origin="o")
    recorder = TraceRecorder(str(tmp_path / "t.jsonl"), run_id="r")
    with recorder:
        make_memory_search_tool(store)("deadline")
        make_memory_get_tool(store)("m1")
        make_memory_note_tool(store)("a note")
        make_memory_note_tool(store)("")
    from rlm_harness import load_events

    calls = [e["payload"] for e in load_events(str(tmp_path / "t.jsonl")) if e["type"] == "tool_call"]
    assert [c["memory_op"] for c in calls] == ["read", "read", "write", "write"]


def _read(step, ids, *, ts=None):
    return _event(step, "tool_call", {"tool": "recall", "memory_op": "read", "memory_ids": ids}, ts=ts)


def test_memory_use_finds_reads_never_referenced_afterwards():
    events = [
        _read(1, ["m1", "m12"]),
        _event(2, "tool_call", {"tool": "remember", "memory_op": "write", "memory_ids": ["m30"]}),
        _event(3, "main_step", {"code": "print(get('m12'))", "reasoning": "m120 is something else"}),
        _event(4, "result", {"output": {"answer": "see m12"}}),
    ]
    use = compute_memory_use(events)
    assert use == MemoryUse(read=("m1", "m12"), written=("m30",), unreferenced=("m1",))
    assert use.unreferenced_share == 0.5


def test_an_id_must_be_named_after_its_read_and_on_a_word_boundary():
    events = [
        _event(1, "main_step", {"code": "", "reasoning": "m1 before it was read"}, ts=1.0),
        _read(2, ["m1"], ts=2.0),
        _event(3, "tool_call", {"tool": "x", "args": {"q": "m11 and xm1"}}, ts=3.0),
    ]
    assert compute_memory_use(events).unreferenced == ("m1",)


def test_a_main_step_stamped_at_flush_time_counts_for_every_read():
    events = [_read(1, ["m5"], ts=10.0), _event(2, "main_step", {"code": "use(m5)"}, ts=99.0)]
    assert compute_memory_use(events).unreferenced == ()


def test_memory_use_without_reads_has_no_share_and_groups_by_run():
    assert compute_memory_use([]).unreferenced_share is None
    by_run = compute_memory_use_by_run([_read(1, ["m1"]), {**_read(2, ["m2"]), "run_id": "run-2"}])
    assert by_run["run-1"].read == ("m1",) and by_run["run-2"].read == ("m2",)



# ---- edges --------------------------------------------------------------------------------------


def test_many_tool_calls_never_push_out_the_evidence_or_the_output():
    events = [_event(n, "tool_call", {"tool": "grep", "args": {"pattern": "x" * 2_500}})
              for n in range(1, 41)] + [_event(50, "result", {"output": "the final answer"})]
    text = render_run(events, evidence={"page": PAGE})
    assert len(text) <= RUN_CHARS and "[evidence:page] (trusted):" in text and "the final answer" in text
    assert "more characters" in text, "the tool calls were the part cut"


def test_a_tool_call_with_no_step_id_is_not_a_source():
    events = [{"type": "tool_call", "payload": {"tool": "fetch_page", "result": PAGE}},
              {"type": "tool_call", "payload": {"tool": "other", "result": "something else entirely"}}]
    assert run_sources(events, trusted_tools=["fetch_page"]) == {}


def test_only_true_makes_a_pair_of_evidence_trusted():
    sources = run_sources([], evidence={"a": (PAGE, "false"), "b": (PAGE, 1), "c": (PAGE, True)})
    assert [sources[k].trusted for k in ("evidence:a", "evidence:b", "evidence:c")] == [False, False, True]


def test_a_run_with_a_broken_or_missing_result_still_consolidates(store):
    store.add("null pointer notes", provenance="said", origin="o")
    seen: list = []
    events = [_event(1, "result", None) | {"payload": None}]
    report = _consolidate(store, events=events, run=_answer(seen=seen))
    assert report.error == "" and seen[0]["known"] == "", "no output means no search, not one for null"


def test_an_id_next_to_cjk_text_is_still_named():
    events = [_read(1, ["m3"]), _event(2, "main_step", {"code": "", "reasoning": "根據m3的內容回答"})]
    assert compute_memory_use(events).unreferenced == ()


def test_an_event_with_an_odd_timestamp_does_not_break_the_count():
    events = [_read(1, ["m4"], ts=1.0), {**_event(2, "main_step", {"code": "m4"}), "ts": "2026-10-19T10:00",
                                          "step_id": "a"}]
    assert compute_memory_use(events).read == ("m4",)


def test_a_failed_call_of_a_trusted_tool_is_not_trusted():
    events = [_event(1, "tool_call", {"tool": "fetch_page", "ok": False,
                                      "result": "No page at https://example.test/anything the model wrote"})]
    assert run_sources(events, trusted_tools=["fetch_page"])["step:1"].trusted is False


def test_evidence_past_the_budget_is_unseen_but_still_checked(store):
    evidence = {f"e{n}": f"filler {n} " * 400 for n in range(40)} | {"last": PAGE}
    assert "[evidence:last]" not in render_run([], evidence=evidence)
    report = _consolidate(store, evidence=evidence, run=_answer(
        {"kind": "fact", "text": "t", "quote": "deadline moved to the fifteenth", "source": "evidence:last"}))
    assert len(report.observed) == 1
