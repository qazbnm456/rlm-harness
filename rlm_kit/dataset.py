"""Phase C (part 2) — export recorded runs as Agentic-RL / SFT datasets.

The JSONL trace is the source of truth. This module turns it into training-ready
records, in three shapes:

- ``export_sft_turns`` — per-root-TURN SFT samples (``input = full history`` seeded with the
  run's initial state, ``output = that turn``), the RLM post-training recipe (arXiv 2512.24601).
- ``export_rl``      — per-planner-step ``(state, action, outcome, reward)`` tuples
  (the orchestrator's trajectory).
- ``export_actions`` — EVERY action (planner step, model-as-tool call, sub-LM
  escalation) as a first-class, `kind`-tagged record, so a trainer can split them
  (fine-tune the generator on `kind=="tool"`, the orchestrator on `kind=="planner"`).

``reward`` is a pluggable callable scoring a whole run; its value is attached to
every record of that run (credit assignment is left to the trainer).

Pure stdlib; no dspy import. Reward definitions are intentionally the caller's
responsibility (the deferred Unknown from the plan).
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional

from .trace import (
    EVENT_MAIN_STEP,
    EVENT_RESULT,
    EVENT_RUN_START,
    EVENT_SUB_CALL,
    EVENT_TOOL_CALL,
)

# A reward function scores one run's events -> float.
RewardFn = Callable[[list[dict]], float]

# The three action event types, in the order they should be sequenced (by step_id).
_ACTION_TYPES = (EVENT_MAIN_STEP, EVENT_TOOL_CALL, EVENT_SUB_CALL)


def _main_steps(events: list[dict]) -> list[dict]:
    return [e for e in events if e["type"] == EVENT_MAIN_STEP]


def _action_record(event: dict) -> dict:
    """Normalise one action event into {kind, action, outcome}."""
    t, p = event["type"], event["payload"]
    if t == EVENT_MAIN_STEP:
        return {
            "kind": "planner",
            "action": {"reasoning": p.get("reasoning"), "code": p.get("code")},
            "outcome": p.get("output"),
        }
    if t == EVENT_TOOL_CALL:
        # A tool_call payload may carry its output under any of several keys — record_tool_call pins
        # none, and the kit's tools disagree: model_as_tool/list_skills use "result", read_skill/MCP
        # use "preview", web_search uses "results", and the make_model_tool consumer convention is
        # "raw". Read a fallback so an action record doesn't silently drop a tool's output; "raw" wins
        # first for back-compat with existing traces.
        output = next(
            (p[k] for k in ("raw", "result", "results", "preview") if p.get(k) is not None), None
        )
        return {
            "kind": "tool",
            "tool": p.get("tool"),
            "action": {"input": p.get("args"), "reasoning": p.get("reasoning")},
            "outcome": {"ok": p.get("ok"), "output": output, "errors": p.get("errors")},
        }
    # EVENT_SUB_CALL (escalation to the expensive sub-LM, e.g. gpt-5.5)
    return {
        "kind": "sub",
        "model": p.get("model"),
        "action": {"input": p.get("input")},
        "outcome": {"output": p.get("processed") or p.get("raw"), "error": p.get("error")},
    }


def export_actions(
    runs: dict[str, list[dict]],
    *,
    reward: Optional[RewardFn] = None,
) -> list[dict]:
    """Every action in a run as a first-class RL record, in step order.

    Unlike :func:`export_rl` (planner-trajectory only), this emits one record per
    *action event* — `main_step` (planner), `tool_call` (a model-as-tool generator
    call), and `sub_call` (an escalation to the expensive sub-LM) — each tagged with
    `kind` so a trainer can split them (e.g. fine-tune the generator on `kind=="tool"`
    records, the orchestrator on `kind=="planner"`). `state` is the ordered list of
    prior actions' (kind, outcome). `reward` is the run-level score, attached to every
    record (credit assignment left to the trainer).
    """
    records: list[dict] = []
    for run_id, events in runs.items():
        run_reward = reward(events) if reward is not None else None
        actions = sorted(
            [e for e in events if e["type"] in _ACTION_TYPES],
            key=lambda e: e.get("step_id", 0),
        )
        history: list[dict] = []
        for e in actions:
            rec = _action_record(e)
            records.append(
                {"run_id": run_id, "state": list(history), **rec, "reward": run_reward}
            )
            history.append({"kind": rec["kind"], "outcome": rec["outcome"]})
    return records


def _run_meta(events: list[dict]) -> dict:
    """The ``meta`` recorded at ``run_start`` (the run's initial state), or ``{}``."""
    rs = next((e for e in events if e["type"] == EVENT_RUN_START), None)
    return (rs["payload"].get("meta") or {}) if rs else {}


def export_sft_turns(runs: dict[str, list[dict]]) -> list[dict]:
    """Per-root-turn SFT samples — the RLM post-training recipe (arXiv 2512.24601, App. A).

    The paper fine-tunes by separating *each root RLM turn* (iteration) into its own SFT
    sample: ``input = the full history`` up to that turn, ``output = the output the root LM
    gave at that step``. Unlike a single whole-trajectory record per run, this is one record per
    turn, and the input is complete: the history here is SEEDED with the run's initial state —
    the ``run_start`` ``meta`` (e.g. the prompt /
    source + the task instructions, which the RLM stores as the REPL's starting variables) — so
    the FIRST turn's input is the real starting context, not an empty list. That seed is the
    "first user input" an RLM trajectory otherwise lacks (the prompt lives in a REPL variable,
    not a chat turn).

    Each record is ``{run_id, turn, input: {initial, history}, output: {reasoning, code}}`` —
    format-agnostic. The trainer renders ``initial + history`` into its chat template and masks
    the loss to ``output`` only (the assistant-only-loss multi-turn SFT the paper describes).
    """
    records: list[dict] = []
    for run_id, events in runs.items():
        initial = _run_meta(events)
        history: list[dict] = []
        for i, step in enumerate(_main_steps(events)):
            p = step["payload"]
            records.append(
                {
                    "run_id": run_id,
                    "turn": p.get("turn", i),
                    "input": {"initial": initial, "history": list(history)},
                    "output": {"reasoning": p.get("reasoning"), "code": p.get("code")},
                }
            )
            history.append(
                {
                    "reasoning": p.get("reasoning"),
                    "code": p.get("code"),
                    "output": p.get("output"),
                }
            )
    return records


def export_rl(
    runs: dict[str, list[dict]],
    *,
    reward: Optional[RewardFn] = None,
) -> list[dict]:
    """Produce per-step ``(state, action, outcome, reward)`` records for RL.

    - ``state``   : the reasoning/code/output history accumulated *before* the step.
    - ``action``  : this step's ``code`` (plus any tool calls observed at/after it).
    - ``outcome`` : this step's ``output``.
    - ``reward``  : ``reward(events)`` for the run, or ``None`` if not supplied.
    """
    records: list[dict] = []
    for run_id, events in runs.items():
        steps = _main_steps(events)
        tool_calls = [e for e in events if e["type"] == EVENT_TOOL_CALL]
        run_reward = reward(events) if reward is not None else None

        history: list[dict] = []
        for i, step in enumerate(steps):
            payload = step["payload"]
            records.append(
                {
                    "run_id": run_id,
                    "turn": payload.get("turn", i),
                    "state": list(history),
                    "action": {
                        "code": payload.get("code"),
                        "reasoning": payload.get("reasoning"),
                    },
                    "outcome": payload.get("output"),
                    "reward": run_reward,
                }
            )
            history.append(
                {
                    "reasoning": payload.get("reasoning"),
                    "code": payload.get("code"),
                    "output": payload.get("output"),
                }
            )
        # Attach run-level tool usage for trainers that condition on tools.
        if records and tool_calls:
            records[-1]["tool_calls"] = [e["payload"] for e in tool_calls]
    return records


def final_outputs(events: Iterable[dict]) -> list[Any]:
    """Convenience: pull recorded final outputs (``result`` events) from a run."""
    return [e["payload"].get("output") for e in events if e["type"] == EVENT_RESULT]
