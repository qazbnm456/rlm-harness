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


# ---- the multi-key output fallback (1.1.0) -------------------------------
#
# `record_tool_call` pins no key for a tool's output and the kit's own tools disagree:
# MCP and read_skill record under `preview`, web_search under `results`, the
# make_model_tool convention under `raw`, list_skills under `result`. `replay` read only
# `result`, so THREE of the four shipped families replayed as None — silently, while
# `dataset.py:_action_record` already read the fallback. Two readers of one trace
# disagreeing was the bug.

def test_replay_serves_every_shipped_output_key(tmp_path):
    from rlm_harness import TraceRecorder, load_timeline
    from rlm_harness.replay import RecordedToolProvider
    from rlm_harness.trace import record_tool_call

    path = str(tmp_path / "t.jsonl")
    with TraceRecorder(path, run_id="r"):
        record_tool_call("model_tool", args={}, raw="the model output")
        record_tool_call("web_search", args={}, results=[{"t": "x"}])
        record_tool_call("list_skills", args={}, result="a, b")

    prov = RecordedToolProvider(load_timeline(path, "r"))
    assert prov.replay("model_tool") == "the model output"
    assert prov.replay("web_search") == [{"t": "x"}]
    assert prov.replay("list_skills") == "a, b"


def test_replay_refuses_to_serve_a_truncated_preview(tmp_path):
    """`preview` is deliberately NOT in the fallback: it is a TRUNCATED head of the output,
    so serving it would hand the replay silently-wrong bytes. Fail loudly instead — the same
    posture this class already takes for drift."""
    from rlm_harness import TraceRecorder, load_timeline
    from rlm_harness.replay import RecordedToolProvider
    from rlm_harness.trace import record_tool_call

    path = str(tmp_path / "t.jsonl")
    with TraceRecorder(path, run_id="r"):
        record_tool_call("mcp_thing", args={}, ok=True, preview="truncated head…")

    prov = RecordedToolProvider(load_timeline(path, "r"))
    with pytest.raises(LookupError, match="TRUNCATED"):
        prov.replay("mcp_thing")


def test_replay_matches_the_raw_name_not_the_repl_alias(tmp_path):
    """`payload["tool"]` is the RAW name; a caller holding the sanitised REPL name the model
    typed matches nothing. Pinned so the doc line stays true."""
    from rlm_harness import TraceRecorder, load_timeline
    from rlm_harness.replay import RecordedToolProvider
    from rlm_harness.trace import record_tool_call

    path = str(tmp_path / "t.jsonl")
    with TraceRecorder(path, run_id="r"):
        record_tool_call("get-weather", args={}, raw="sunny", repl_name="get_weather")

    prov = RecordedToolProvider(load_timeline(path, "r"))
    assert prov.replay("get-weather") == "sunny"
    with pytest.raises(LookupError):
        prov.replay("get_weather")


def test_export_actions_carries_repl_name_only_when_it_differs(tmp_path):
    """§4's other half. The MCP mapping is UNRECOVERABLE offline — it depends on the server's
    whole tool list at run time, which never enters the trace — so the exporter must carry it.
    Conditional, mirroring `mcp._repl_alias`: a `null` key on every non-MCP tool record would
    churn every consumer's golden fixtures for nothing."""
    from rlm_harness import TraceRecorder, export_actions, group_by_run, load_events
    from rlm_harness.trace import record_tool_call

    path = str(tmp_path / "t.jsonl")
    with TraceRecorder(path, run_id="r"):
        record_tool_call("get-weather", args={}, raw="sunny", repl_name="get_weather")
        record_tool_call("plain_tool", args={}, raw="ok")

    tools = [r for r in export_actions(group_by_run(load_events(path))) if r["kind"] == "tool"]
    by_name = {r["tool"]: r for r in tools}

    assert by_name["get-weather"]["repl_name"] == "get_weather"   # the join key
    assert by_name["get-weather"]["tool"] == "get-weather"        # identity stays RAW
    assert "repl_name" not in by_name["plain_tool"]               # byte-identical to pre-1.1.0
