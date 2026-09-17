"""Export recorded runs as Agentic-RL / SFT datasets.

The JSONL trace is the source of truth. This module turns it into training-ready
records, in three shapes:

- ``export_sft_turns`` — per-root-TURN SFT samples (``input = full history`` seeded with the
  run's initial state, ``output = that turn``), the RLM post-training recipe (arXiv 2512.24601).
- ``export_rl``      — per-planner-step ``(state, action, outcome, reward)`` tuples
  (the orchestrator's trajectory).
- ``export_actions`` — EVERY action (planner step, model-as-tool call, sub-LM
  escalation) as a first-class, `kind`-tagged record, so a trainer can split them
  (fine-tune the generator on `kind=="tool"`, the orchestrator on `kind=="planner"`).

``run_label_bundle`` is a companion MAPPER — ``{surface: {run_id: fn(events)}}`` — for per-run
intrinsic LABEL surfaces (validity flags, objective metrics, a rubric's per-criterion facts) that
ride BESIDE the reward-free records; it refuses a ``reward`` surface (the trainer attaches reward).

``reward`` is a pluggable callable scoring a whole run; its value is attached to
every record of that run (credit assignment is left to the trainer).

Pure stdlib; no dspy import. Reward definitions are intentionally the caller's
responsibility (the deferred Unknown from the plan).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from typing import Any

from .trace import (
    EVENT_MAIN_STEP,
    EVENT_RESULT,
    EVENT_RUN_START,
    EVENT_SUB_CALL,
    EVENT_TOOL_CALL,
    payload_cause,
)

# A reward function scores one run's events -> float.
RewardFn = Callable[[list[dict]], float]

# A label function maps one run's events -> a dict of intrinsic FACTS/LABELS (never a score).
LabelFn = Callable[[list[dict]], dict]

logger = logging.getLogger(__name__)

# The three action event types. Sequencing them is `_sequenced_actions`'s job, not
# this tuple's order and — since 1.11.2 — not `step_id`'s.
_ACTION_TYPES = (EVENT_MAIN_STEP, EVENT_TOOL_CALL, EVENT_SUB_CALL)


def _main_steps(events: list[dict]) -> list[dict]:
    return [e for e in events if e["type"] == EVENT_MAIN_STEP]


def _sequenced_actions(events: list[dict]) -> list[dict]:
    """Action events in CAUSAL order — the order they HAPPENED, not the order they were written.

    ``main_step`` events are written in one batch once ``aforward()`` has returned, so every turn's
    ``step_id`` is HIGHER than every live ``tool_call``/``sub_call`` of the same attempt. Sorting
    the three types together by ``step_id`` therefore put every turn after every tool call, and
    ``state`` -- documented as "the ordered list of prior actions" -- was wrong in one direction for
    each kind: a tool record's prior actions contained NO turns at all, and a turn's contained
    EVERY tool call. Systematic, not occasional.

    ``ts`` is the field that places a turn against the live events around it, and that is the only
    thing the guide says it is for. A turn's ``ts`` is stamped when its reasoning was PARSED, so it
    precedes the tool calls that turn's own code then makes -- which is what makes the interleave
    causal rather than merely different. But it is BACKFILLED, and for a turn whose live stamp could
    not be matched it falls back to the flush time, so it is not trustworthy for ordering turns
    against EACH OTHER; ``payload["turn"]`` is. So turns keep ``turn`` order, live events keep
    ``ts`` order, and the two sequences are merged on ``ts``.

    **An unmatched stamp costs more than turn-vs-turn order, and this is the limit of the fix.** A
    flush time is LATER than every live event of the run, so a turn that falls back to one drags
    every subsequent live event in front of itself — and if EVERY turn falls back, the merge
    reproduces the pre-1.11.2 order exactly, wrong ``state`` included. Re-exporting such a corpus
    changes nothing, because the trace holds no signal to interleave on. Reachable in practice, not
    just in theory: `dspy.streamify` captures `settings.callbacks` at construction and so drops
    `_MainStepTimer` (see CHANGELOG 1.6.0), any failure entering `_live_main_timing`'s
    `dspy.context` does the same, and so does any caller of ``record_main_trajectory`` outside
    ``RLMTask.arun``. The ``logger.debug`` below reports the detectable SYMPTOM — the first turn
    not stamped before the run's first live event — because the output of a degraded interleave is
    indistinguishable from a correct one. It is a one-sided hint and not a proof: a retry attempt or
    a host-side ``record_tool_call`` reads the same way with nothing wrong, and the comment there
    names both.

    **Turn order is guaranteed by the pre-sort alone, not by anything in the merge.** A clamp
    holding each turn to no earlier than its predecessor was written here first and removed as
    provably dead: the merge has already consumed every live event up to the previous turn's
    position, so a turn whose backfilled ``ts`` went BACKWARDS emits nothing either way and lands in
    the same place. A mutation deleting the clamp left the whole suite green, which is what exposed
    it -- the surviving guarantee is the ``_turn_key`` sort, and that one does go red when broken.

    A corpus with no timestamps at all (hand-written fixtures, and any trace predating the field)
    falls back to ``step_id`` order, which for such input is the only order there is.
    """
    actions = [e for e in events if e["type"] in _ACTION_TYPES]
    if not any("ts" in e for e in actions):
        return sorted(actions, key=lambda e: e.get("step_id", 0))

    def _turn_key(e: dict) -> tuple:
        turn = e["payload"].get("turn")
        # `turn` absent (a pre-1.x trace, or a fixture) -> fall back to write order among turns.
        return (0, turn) if isinstance(turn, int) and not isinstance(turn, bool) else (
            1, e.get("step_id", 0))

    turns = sorted((e for e in actions if e["type"] == EVENT_MAIN_STEP), key=_turn_key)
    live = sorted(
        (e for e in actions if e["type"] != EVENT_MAIN_STEP),
        key=lambda e: (_ts(e), e.get("step_id", 0)),
    )

    if turns and live and _ts(turns[0]) >= _ts(live[0]):
        # A ONE-SIDED HINT, not a proof, and the difference matters because this is the only thing
        # that speaks. Turn 0 should be stamped BEFORE the run's first live event: a turn is
        # stamped when its reasoning is parsed, and a turn's own code is what produces live
        # events. When it is not, the usual cause is an unmatched stamp that fell back to the
        # flush time. But the first live event need not belong to a RECORDED turn at all, so there
        # are two readings that are not degradation:
        #   * live events from an execution whose turns never reached the trace. `run_with_retry`
        #     re-runs the whole trajectory, and what gets flushed is the last attempt that RETURNED
        #     a prediction -- not the last attempt (see `trace.note_usage`, which documents the
        #     same distinction for usage: a run whose final attempt RAISES keeps an earlier
        #     attempt's turns). So an attempt's live events can sit in the trace with none of its
        #     turns, and the recorded turn 0 legitimately postdates them. Needs `max_retries >= 2`,
        #     or a run that raised mid-`aforward` under a shared recorder.
        #   * a host-side `record_tool_call` before the run. Needs no opt-in.
        # Two healthy `arun`s sharing a recorder and `run_id` do NOT trigger it, which is worth
        # knowing because it is the shape a reader pictures first: `turns` is stable-sorted on
        # `payload["turn"]`, so `turns[0]` is the FIRST run's turn 0 and the second run's is never
        # compared. All three readings were checked by running them, not reasoned about: two
        # healthy runs stay silent, while an orphaned live event and a pre-run host-side call both
        # fire.
        # Chosen over "are ALL the turns late" because ONE unmatched stamp is enough to damage a
        # run, and that version stayed silent on exactly that case. Known miss in the other
        # direction: a LATER turn unmatched while turn 0 is fine degrades the interleave from that
        # point on and is not reported, because a genuinely late turn looks identical.
        logger.debug(
            "main_step turn %r is not stamped before this run's first live event, so the action "
            "order below may be degraded — the usual cause is a turn stamp that was never matched "
            "and fell back to the flush time, and a run whose stamps ALL fell back exports in the "
            "old step_id order with wrong `state`. An earlier retry attempt, or a host-side "
            "tool_call under the same recorder, reads the same way with nothing wrong",
            turns[0]["payload"].get("turn"),
        )

    out: list[dict] = []
    i = 0
    for turn in turns:
        at = _ts(turn)
        while i < len(live) and _ts(live[i]) <= at:
            out.append(live[i])
            i += 1
        out.append(turn)
    out.extend(live[i:])
    return out


def _ts(event: dict) -> float:
    """An event's timestamp as a float, or -inf when it carries none."""
    ts = event.get("ts")
    return float(ts) if isinstance(ts, (int, float)) and not isinstance(ts, bool) else float("-inf")


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
            # The REPL alias, when sanitising changed the name (MCP: the model types
            # `get_weather`, the trace records `get-weather`). CONDITIONAL, mirroring
            # `mcp._repl_alias`: an unconditional key would put `"repl_name": null` on every
            # tool record including non-MCP ones, churning every consumer's golden fixtures for
            # nothing. TOP LEVEL beside `tool` because both are IDENTITY — putting it under
            # `action` would imply the model supplied it. A trainer joining the planner's code
            # (`main_step.payload.code` shows `get_weather(...)`) to this record needs it, and
            # the mapping is unrecoverable offline: it depends on the server's whole tool list
            # at run time, which never enters the trace.
            **({"repl_name": p["repl_name"]} if p.get("repl_name") else {}),
            "action": {"input": p.get("args"), "reasoning": p.get("reasoning")},
            "outcome": {
                "ok": p.get("ok"),
                "output": output,
                "errors": p.get("errors"),
                # WHY it is not ok. `ok` alone cannot tell a validator rejection from an endpoint
                # failure or a circuit break, and this record is what reaches a TRAINER — a dataset
                # that labels infrastructure failures as content declines teaches exactly that. The
                # endpoint string rode nowhere at all before this: it is recorded under `error` (or
                # `endpoint_error`) by consumer convention, and neither was carried, so a downstream
                # reader could not even reconstruct the split by hand. For a tool with no validator
                # this is simply `ok`/`invalid`, which is what `ok` already said.
                "cause": p.get("cause") or payload_cause(p),
                # PRESENCE, not truthiness — the same empty-string trap `payload_cause` carries
                # a note about. `str(exc)` is `''` for the common transport failures, so `or` would
                # skip a present-but-empty `endpoint_error`, fall through to an absent `error`, and
                # emit `null` for a call that really did fail at the endpoint. The record would then
                # say `cause: endpoint` beside `error: null`, which reads as a contradiction to
                # whoever is reading the dataset.
                "error": next(
                    (p[k] for k in ("endpoint_error", "error") if p.get(k) is not None), None
                ),
            },
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
    reward: RewardFn | None = None,
) -> list[dict]:
    """Every action in a run as a first-class RL record, in the order it happened.

    Unlike :func:`export_rl` (planner-trajectory only), this emits one record per
    *action event* — `main_step` (planner), `tool_call` (a model-as-tool generator
    call), and `sub_call` (an escalation to the expensive sub-LM) — each tagged with
    `kind` so a trainer can split them (e.g. fine-tune the generator on `kind=="tool"`
    records, the orchestrator on `kind=="planner"`). `state` is the ordered list of
    prior actions' (kind, outcome). `reward` is the run-level score, attached to every
    record (credit assignment left to the trainer).

    **Records come out in CAUSAL order, which is not the order the trace was written** — see
    `_sequenced_actions`. Until 1.11.2 this sorted by `step_id`, which put every turn after every
    tool call of the same attempt and made `state` systematically wrong for both kinds.
    """
    records: list[dict] = []
    for run_id, events in runs.items():
        run_reward = reward(events) if reward is not None else None
        actions = _sequenced_actions(events)
        history: list[dict] = []
        for e in actions:
            rec = _action_record(e)
            records.append(
                {"run_id": run_id, "state": list(history), **rec, "reward": run_reward}
            )
            history.append({"kind": rec["kind"], "outcome": rec["outcome"]})
    return records


def run_label_bundle(
    runs: dict[str, list[dict]], /, **label_fns: LabelFn
) -> dict[str, dict[str, dict]]:
    """Map named per-run LABEL surfaces over a set of runs: ``{surface: {run_id: fn(events)}}``.

    A companion to the reward-free exporters: each ``label_fns`` entry is a consumer-supplied function
    turning one run's events into a dict of intrinsic labels — validity flags, objective metrics, or a
    rubric's deterministic per-criterion facts (where a criterion is just a dict with ``name`` /
    ``description`` / ``weight`` and an OPAQUE ``category`` string the kit never interprets). These ride
    BESIDE the trajectory records so a downstream trainer reads ONE canonical bundle shape instead of
    each consumer re-deriving it::

        run_label_bundle(runs, labels=run_labels, metrics=run_metrics)
        # -> {"labels": {run_id: {...}}, "metrics": {run_id: {...}}}

    ``reward`` is a REFUSED surface name: rlm-harness produces trajectories, never reward — the trainer
    composes reward from these labels (plus its own credit assignment), so a label fn must emit FACTS,
    not a score. The kit can only refuse the NAME; that a fn returns facts and not a hidden score is the
    same convention-not-enforcement trust model as ``reward=`` on the exporters.
    """
    if "reward" in label_fns:
        raise ValueError(
            "'reward' is not a label surface — rlm-harness exports trajectories, never reward; attach "
            "reward in the trainer, and emit facts (not scores) as labels here."
        )
    return {name: {rid: fn(ev) for rid, ev in runs.items()} for name, fn in label_fns.items()}


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
    reward: RewardFn | None = None,
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
