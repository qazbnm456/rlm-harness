"""Contract test — PIN the cross-project surface a consumer builds on, so a change that would
silently break a downstream reader (a consumer's report renderer + RL export, a trajectory-replay
UI, a future RL trainer) fails HERE, in rlm-kit's own suite, instead of in the consumer with no
clue why.

Three frozen things (see CLAUDE.md "The trace is a VERSIONED wire format"):
  1. the trace SCHEMA + the seven EVENT_* type strings,
  2. the recorded-event ENVELOPE shape,
  3. the dataset-exporter RECORD shapes (export_actions / export_sft_turns / export_rl / run_label_bundle),
plus the public ``__all__`` surface a consumer imports. ADDITIVE change is fine (a new optional
payload field, a new ``__all__`` entry); removing / renaming / re-typing any of these is a v1 break —
bump SCHEMA to ``rlm-kit/trace/v2`` with a migration instead of editing this test to be green.
"""

import rlm_kit
from rlm_kit import trace as T
from rlm_kit.dataset import export_actions, export_rl, export_sft_turns, run_label_bundle


def test_schema_id_is_frozen_at_v1():
    assert T.SCHEMA == "rlm-kit/trace/v1"


def test_event_type_strings_are_frozen():
    # downstream readers key on these literal STRINGS, not the constant names — so pin the values.
    assert (
        T.EVENT_RUN_START,
        T.EVENT_MAIN_STEP,
        T.EVENT_SUB_CALL,
        T.EVENT_TOOL_CALL,
        T.EVENT_FINAL,
        T.EVENT_RESULT,
        T.EVENT_RUN_END,
    ) == ("run_start", "main_step", "sub_call", "tool_call", "final", "result", "run_end")


def test_recorded_event_envelope_shape(tmp_path):
    # every event a recorder emits is exactly this envelope (replay + dataset readers depend on it).
    with T.TraceRecorder(str(tmp_path / "t.jsonl"), run_id="r1", clock=lambda: 1.0) as rec:
        ev = rec.record(T.EVENT_TOOL_CALL, {"tool": "x"})
    assert set(ev) == {"schema", "run_id", "step_id", "ts", "type", "payload"}
    assert ev["schema"] == "rlm-kit/trace/v1" and ev["run_id"] == "r1"
    assert ev["type"] == "tool_call" and ev["payload"] == {"tool": "x"}
    assert isinstance(ev["step_id"], int)


def _synthetic_run() -> dict:
    """A minimal but complete run, grouped as the exporters expect ({run_id: [events]}): run_start
    (meta = the initial state an RLM trajectory otherwise lacks) + a planner turn + a tool call + a
    sub-LM escalation. Events carry only the keys the exporters read (type/step_id/payload)."""
    return {
        "r": [
            {"type": "run_start", "step_id": 0,
             "payload": {"meta": {"source": "S", "instructions": "I"}}},
            {"type": "main_step", "step_id": 1,
             "payload": {"turn": 0, "reasoning": "why", "code": "x=1", "output": "ok"}},
            {"type": "tool_call", "step_id": 2,
             "payload": {"tool": "gen", "args": {"spec": "..."}, "ok": True, "raw": "yaml", "errors": []}},
            {"type": "sub_call", "step_id": 3,
             "payload": {"model": "m", "input": "q", "processed": "a"}},
        ]
    }


def test_export_actions_record_shape():
    recs = export_actions(_synthetic_run())
    assert [r["kind"] for r in recs] == ["planner", "tool", "sub"]   # one per action event, step order
    for r in recs:
        assert {"run_id", "state", "kind", "action", "outcome", "reward"} <= set(r)
        assert isinstance(r["state"], list)
        assert r["reward"] is None                                   # reward-free: a HOOK, not computed
    planner, tool, sub = recs
    assert {"reasoning", "code"} <= set(planner["action"])
    assert tool["tool"] == "gen" and {"ok", "output", "errors"} <= set(tool["outcome"])
    assert sub["model"] == "m"


def test_export_sft_turns_record_shape():
    recs = export_sft_turns(_synthetic_run())
    assert len(recs) == 1                                            # one sample per ROOT turn
    r = recs[0]
    assert {"run_id", "turn", "input", "output"} <= set(r)
    # input is SEEDED with the run_start meta (the "first user input" an RLM trace otherwise lacks)
    assert r["input"]["initial"] == {"source": "S", "instructions": "I"}
    assert isinstance(r["input"]["history"], list)
    assert {"reasoning", "code"} <= set(r["output"])


def test_export_rl_record_shape():
    recs = export_rl(_synthetic_run())
    assert len(recs) == 1
    r = recs[0]
    assert {"run_id", "turn", "state", "action", "outcome", "reward"} <= set(r)
    assert {"code", "reasoning"} <= set(r["action"])
    assert r["reward"] is None                                       # reward-free


def test_run_label_bundle_shape():
    # per-run LABEL surfaces a trainer reads BESIDE the reward-free records: {surface: {run_id: dict}}.
    # ADDITIVE to v1: a consumer's trainer keys on this exact bundle shape, so pin it.
    bundle = run_label_bundle(_synthetic_run(), labels=lambda ev: {"n": len(ev)})
    assert bundle == {"labels": {"r": {"n": 4}}}


def test_public_surface_includes_the_consumer_contract():
    # the load-bearing names a consumer imports — a representative subset, not the whole list (which
    # may GROW). Removing any breaks a downstream consumer / its UI / the trainer.
    must_export = {
        "RLMTask", "RLMConfig", "configure", "get_config", "RLMTaskError",
        "intercept_sub_lm", "model_as_tool", "get_sub_lm", "load_skills_as_tools",
        "TraceRecorder", "current_recorder", "record_tool_call", "load_events", "group_by_run",
        "export_sft_turns", "export_rl", "export_actions", "run_label_bundle",
        # reward-free rubric primitives (opaque category)
        "Criterion", "RubricCriteria", "CriterionFact",
        "rubric_to_meta", "rubric_from_meta", "validate_rubric", "criteria_facts",
        # trace/v1 contract constants — a consumer reads a trace against these, not raw strings
        "EVENT_RUN_START", "EVENT_MAIN_STEP", "EVENT_SUB_CALL", "EVENT_TOOL_CALL",
        "EVENT_FINAL", "EVENT_RESULT", "EVENT_RUN_END",
    }
    assert must_export <= set(rlm_kit.__all__)
