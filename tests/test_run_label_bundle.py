import pytest

from rlm_kit.dataset import run_label_bundle


def _runs():
    return {
        "r1": [{"type": "main_step", "step_id": 1, "payload": {}}],
        "r2": [
            {"type": "main_step", "step_id": 1, "payload": {}},
            {"type": "tool_call", "step_id": 2, "payload": {"tool": "t"}},
        ],
    }


def test_run_label_bundle_maps_surfaces_over_runs():
    bundle = run_label_bundle(
        _runs(),
        labels=lambda ev: {"n_events": len(ev)},
        metrics=lambda ev: {"tools": sum(1 for e in ev if e["type"] == "tool_call")},
    )
    assert bundle == {
        "labels": {"r1": {"n_events": 1}, "r2": {"n_events": 2}},
        "metrics": {"r1": {"tools": 0}, "r2": {"tools": 1}},
    }


def test_run_label_bundle_refuses_reward_surface():
    # 'reward' is refused by NAME: the kit exports trajectories, never reward.
    with pytest.raises(ValueError, match="never reward"):
        run_label_bundle(_runs(), reward=lambda ev: 1.0)


def test_run_label_bundle_empty_when_no_surfaces():
    assert run_label_bundle(_runs()) == {}


def test_run_label_bundle_surface_named_runs():
    # `runs` is positional-only (the `/`), so a label surface may itself be named "runs".
    bundle = run_label_bundle(_runs(), runs=lambda ev: {"k": True})
    assert bundle == {"runs": {"r1": {"k": True}, "r2": {"k": True}}}
