import pytest

from rlm_harness.dataset import export_rl, export_sft_turns, final_outputs
from rlm_harness.replay import RecordedToolProvider, load_timeline
from rlm_harness.trace import TraceRecorder, group_by_run, load_events


def _write_run(path, run_id):
    """Write a small but complete run: 2 main steps, 1 tool call, 1 result."""
    with TraceRecorder(path, run_id=run_id) as rec:
        rec.record("tool_call", {"tool": "read_skill", "args": {"name": "recon"}, "result": "SKILL BODY"})
        rec.record_main_trajectory(
            type("P", (), {
                "trajectory": [
                    {"reasoning": "r0", "code": "c0", "output": "o0"},
                    {"reasoning": "r1", "code": "c1", "output": "o1"},
                ],
                "final_reasoning": "fin",
            })()
        )
        rec.record_result({"answer": "done"})


def test_reconstruct_timeline(tmp_path):
    path = str(tmp_path / "t.jsonl")
    _write_run(path, "r1")
    tl = load_timeline(path, "r1")
    assert tl.run_id == "r1"
    assert len(tl.main_steps) == 2
    assert len(tl.tool_calls) == 1
    assert "2 main steps" in tl.summary()


def test_recorded_tool_provider_serves_in_order(tmp_path):
    path = str(tmp_path / "t.jsonl")
    with TraceRecorder(path, run_id="r1") as rec:
        rec.record("tool_call", {"tool": "fetch", "args": {}, "result": "A"})
        rec.record("tool_call", {"tool": "fetch", "args": {}, "result": "B"})
    tl = load_timeline(path, "r1")
    provider = RecordedToolProvider(tl)
    assert provider.replay("fetch") == "A"
    assert provider.replay("fetch") == "B"
    with pytest.raises(LookupError):
        provider.replay("fetch")  # recording exhausted -> loud failure


def test_export_sft_turns_per_turn_with_seeded_initial(tmp_path):
    # The RLM post-training recipe (arXiv 2512.24601 App. A): one SFT sample per root TURN,
    # input = full history seeded with the run's initial state (run_start meta), output = turn.
    path = str(tmp_path / "t.jsonl")
    with TraceRecorder(path, run_id="r1", meta={"source": "ADVISORY", "instructions": "SYS"}) as rec:
        rec.record_main_trajectory(
            type("P", (), {
                "trajectory": [
                    {"reasoning": "r0", "code": "c0", "output": "o0"},
                    {"reasoning": "r1", "code": "c1", "output": "o1"},
                ],
                "final_reasoning": "fin",
            })()
        )
        rec.record_result({"answer": "done"})
    runs = group_by_run(load_events(path))
    turns = export_sft_turns(runs)
    assert len(turns) == 2                                   # one sample per root turn
    # turn 0: history empty, but the initial state (source + instructions) IS the seed —
    # this is the "first user input" the bare trajectory otherwise lacks.
    assert turns[0]["input"]["initial"] == {"source": "ADVISORY", "instructions": "SYS"}
    assert turns[0]["input"]["history"] == []
    assert turns[0]["output"] == {"reasoning": "r0", "code": "c0"}
    # turn 1: full history now carries turn 0 (reasoning+code+the observed output o0)
    assert turns[1]["input"]["initial"]["source"] == "ADVISORY"   # seed rides every sample
    assert len(turns[1]["input"]["history"]) == 1
    assert turns[1]["input"]["history"][0] == {"reasoning": "r0", "code": "c0", "output": "o0"}
    assert turns[1]["output"] == {"reasoning": "r1", "code": "c1"}


def test_export_sft_turns_without_meta_seeds_empty(tmp_path):
    # No run_start meta (a trace that didn't capture the initial state) -> initial = {},
    # never raises; the per-turn split still works.
    path = str(tmp_path / "t.jsonl")
    _write_run(path, "r1")
    turns = export_sft_turns(group_by_run(load_events(path)))
    assert len(turns) == 2 and all(t["input"]["initial"] == {} for t in turns)


def test_export_rl_with_reward(tmp_path):
    path = str(tmp_path / "t.jsonl")
    _write_run(path, "r1")
    runs = group_by_run(load_events(path))

    def reward(events):
        return 1.0  # toy: every run scored 1

    rl = export_rl(runs, reward=reward)
    assert len(rl) == 2  # one per main step
    # First step's state is empty (no prior history); second has 1 prior turn.
    assert rl[0]["state"] == []
    assert len(rl[1]["state"]) == 1
    assert rl[0]["action"]["code"] == "c0"
    assert all(step["reward"] == 1.0 for step in rl)
    # Tool calls attached to the run's last step.
    assert "tool_calls" in rl[-1]


def test_export_rl_without_reward(tmp_path):
    path = str(tmp_path / "t.jsonl")
    _write_run(path, "r1")
    runs = group_by_run(load_events(path))
    rl = export_rl(runs)
    assert all(step["reward"] is None for step in rl)


def test_final_outputs(tmp_path):
    path = str(tmp_path / "t.jsonl")
    _write_run(path, "r1")
    outs = final_outputs(load_events(path))
    assert outs == [{"answer": "done"}]
