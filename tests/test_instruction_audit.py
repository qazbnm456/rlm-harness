"""`optimize.audit_instruction`: what a compiled instruction gained and lost.

Every number this guards against came from one measured compile. It raised a held-out set from
0.513 to 0.733, and in the same run it wrote four of ten trainset answers into the instruction and
dropped one capability without anything noticing. `docs/PROMPT.md` has the run.
"""

from __future__ import annotations

import pytest

import rlm_harness
from rlm_harness import optimize

dspy = pytest.importorskip("dspy")

from rlm_harness.testing import ScriptedInterpreter

SUB_LM = {"sub-LM": ["llm_query", "sub-model", "sub-LLM call"]}
BASE = ("Answer from the document. You may call llm_query for a sub-model. "
        "When done, call SUBMIT() with your output.")


def _ex(q: str, answer: str):
    return dspy.Example(q=q, answer=answer).with_inputs("q")


def test_a_trainset_answer_copied_into_the_instruction_is_found():
    """The failure the whole audit exists for: the score rises because the answers are carried."""
    answer = "the schema string is written in TraceRecorder.record, at the event dict literal"
    audit = optimize.audit_instruction(BASE, BASE + " Hint: " + answer,
                                       trainset=[_ex("where?", answer)],
                                       holdout=[_ex("else?", "something entirely different here")])
    assert audit.trainset_answers_in == [0]
    assert audit.holdout_answers_in == []
    assert audit.memorised is True


def test_with_no_holdout_memorised_is_None_because_there_was_no_control():
    """`bool(hits) and not []` cannot tell "the control ran and stayed clean" from "there was no
    control": both return the same answer when everything they measure is working, which is this
    repo's own test for an absent measurement being reported as a real one. The first version of
    this file asserted `True` here."""
    answer = "the schema string is written in TraceRecorder.record, at the event dict literal"
    audit = optimize.audit_instruction(BASE, BASE + " Hint: " + answer,
                                       trainset=[_ex("where?", answer)])
    assert audit.trainset_answers_in == [0], "the hit is still reported"
    assert audit.controlled is False
    assert audit.memorised is None, "a claim was made with no control behind it"


def test_an_extract_only_compile_is_audited_too():
    """The predictor the audit was blind to, and the one most likely to carry a crib sheet.

    GEPA's default `component_selector="round_robin"` moves one component at a time, and
    `RLMTask._apply_compiled_instructions` calls an `extract`-only winner "an ordinary outcome,
    not a corner case" — then applies its text in production. Reading only `generate_action`
    returned `memorised False` with the gold answer verbatim in `extract`.
    """
    answer = "the fingerprint is written by save_program when it is given the task argument"
    base = {"generate_action": BASE, "extract": "Return only the answer span."}
    comp = {"generate_action": BASE, "extract": "Return only the answer span. Hint: " + answer}
    audit = optimize.audit_instruction(base, comp, trainset=[_ex("q", answer)],
                                       holdout=[_ex("h", "an unrelated held-out gold value here")])
    assert audit.predictors == ["generate_action", "extract"]
    assert audit.trainset_answers_in == [0], "an extract-side crib sheet went unseen"
    assert audit.memorised is True


def test_with_no_capability_map_the_empty_list_says_so():
    """`capabilities_lost == []` is produced by construction when no map was given, and by
    measurement when one was. A reader cannot tell those apart from the list alone."""
    unchecked = optimize.audit_instruction(BASE, "Answer. Finish with SUBMIT().")
    assert unchecked.capabilities_lost == [] and unchecked.capabilities_checked is False
    checked = optimize.audit_instruction(BASE, "Answer. Finish with SUBMIT().",
                                         capabilities=SUB_LM)
    assert checked.capabilities_lost == ["sub-LM"] and checked.capabilities_checked is True


def test_the_holdout_is_the_control_that_says_the_matcher_works():
    """Trainset hits alone cannot tell memorisation from a matcher firing on ordinary language.

    The measured run put 4 of 10 trainset answers in and 0 of 6 holdout answers, and it is the
    ZERO that makes the four mean something. If both sides hit, the instrument is over-matching
    and `memorised` says so by refusing to conclude.
    """
    shared = "the answer is written into the event dictionary literal inside record"
    audit = optimize.audit_instruction(BASE, BASE + " " + shared,
                                       trainset=[_ex("a", shared)], holdout=[_ex("b", shared)])
    assert audit.trainset_answers_in == [0], "the trainset answer is there"
    assert audit.holdout_answers_in == [0], "and so is a held-out one, so the match is too loose"
    assert audit.memorised is False, (
        "with both sides hit, a memorisation claim is unsupported and must not be made"
    )


def test_an_answer_the_baseline_already_contained_is_not_a_hit():
    """Otherwise every compile of a task whose own instructions quote an example reads as
    memorising, and the audit cries wolf on its first use."""
    answer = "call SUBMIT() with your output when the citation is complete and verified"
    audit = optimize.audit_instruction(BASE + " " + answer, BASE + " " + answer,
                                       trainset=[_ex("q", answer)])
    assert audit.trainset_answers_in == []


def test_a_short_answer_cannot_be_told_from_a_coincidence_and_is_skipped():
    """A one-word label appears in any prose. Counting it would make the holdout control fire on
    both sides for every task whose answers are short, which is worse than a blind spot that is
    written down."""
    audit = optimize.audit_instruction(BASE, BASE + " yes.", trainset=[_ex("q", "yes")])
    assert audit.trainset_answers_in == []


def test_matching_survives_the_whitespace_dspy_flattens():
    """dspy flattens an instruction with `.replace("\\n", "  ")` before the model reads it, so a
    newline in the recorded answer must not defeat the match."""
    answer = "the record method takes the lock\nthen writes the schema into the event"
    audit = optimize.audit_instruction(BASE, BASE + " " + answer.replace("\n", "  "),
                                       trainset=[_ex("q", answer)])
    assert audit.trainset_answers_in == [0]


def test_a_capability_removed_is_reported_and_a_reworded_one_is_not():
    """The literal-phrase version of this check called five of seven guardrails deleted when one
    had gone. Matching a capability against several wordings is what fixed it, so the test covers
    both directions: a real deletion and a rewrite that only looks like one."""
    rewritten = "Answer from the document. Use llm_query when a sub-model helps. Finish with SUBMIT()."
    kept = optimize.audit_instruction(BASE, rewritten, capabilities=SUB_LM)
    assert kept.capabilities_lost == [] and kept.capabilities_kept == ["sub-LM"]

    deleted = optimize.audit_instruction(BASE, "Answer from the document. Finish with SUBMIT().",
                                         capabilities=SUB_LM)
    assert deleted.capabilities_lost == ["sub-LM"] and deleted.capabilities_kept == []


def test_a_capability_kept_under_a_LATER_wording_is_not_reported_lost():
    """The gate for the thing the design exists for, and the first version of this file did not
    have it: every wording in the earlier test happened to include the FIRST entry in the list, so
    an implementation matching only `wordings[0]` passed. A matcher that checks one phrase is the
    one that called five of seven guardrails deleted when one had gone.
    """
    # `llm_query` is gone; "sub-LLM call", the THIRD wording, is how the rewrite says it.
    rewritten = "Answer from the document. Spend a sub-LLM call when it helps. Finish with SUBMIT()."
    audit = optimize.audit_instruction(BASE, rewritten, capabilities=SUB_LM)
    assert audit.capabilities_lost == [], (
        "a capability the optimizer kept in different words was reported as deleted, which is the "
        "exact failure this check was rebuilt to avoid"
    )
    assert audit.capabilities_kept == ["sub-LM"]


def test_a_capability_the_baseline_never_had_is_neither_lost_nor_kept():
    audit = optimize.audit_instruction("Answer the question.", "Answer it faster.",
                                       capabilities=SUB_LM)
    assert audit.capabilities_lost == [] and audit.capabilities_kept == []


def test_length_is_a_ratio_never_a_character_count():
    """A count measures something a release can edit and goes stale with the next docstring."""
    audit = optimize.audit_instruction("x" * 100, "y" * 40)
    assert audit.length_ratio == pytest.approx(0.40)


def test_compile_task_reports_the_audit_without_being_asked(monkeypatch, tmp_path):
    """A compile whose memorisation nobody looked at is the shape this release is trying to end,
    so the audit is produced by default rather than offered."""
    class _Sig:
        def __init__(self, text): self.instructions = text
    class _P:
        def __init__(self, text): self.signature = _Sig(text)
    class _Prog:
        def __init__(self, text, answer="a"):
            self.generate_action, self.extract = _P(text), _P("extract")
            self.answer = answer
        def named_predictors(self):
            return [("generate_action", self.generate_action), ("extract", self.extract)]
        def __call__(self, **kw):
            return type("R", (), {"answer": self.answer})()

    answer = "the fingerprint is written by save_program when it is given the task argument"
    built = _Prog(BASE)

    class _GEPA:
        def __init__(self, **kw): pass
        def compile(self, program, **kw): return _Prog(BASE + " Hint: " + answer)

    monkeypatch.setattr(dspy, "GEPA", _GEPA)
    rlm_harness.configure(rlm_harness.RLMConfig(
        main_model="openai/x", sub_model="openai/x", api_key="k", interpreter="mock",
        max_retries=1))

    class _T(rlm_harness.RLMTask):
        signature = "q: str -> answer: str"
        output_field = "answer"
        instructions = "Answer."

    task = _T(interpreter=ScriptedInterpreter([]))
    monkeypatch.setattr(type(task), "_build_rlm", lambda self: built)
    result = optimize.compile_task(task, [_ex("q", answer)], lambda e, p, *a, **k: 1.0,
                                   reflection_lm=object(),
                                   holdout=[_ex("h", "a held-out gold value, entirely unrelated")],
                                   capabilities=SUB_LM)
    assert result.instruction_audit is not None
    assert result.instruction_audit.trainset_answers_in == [0]
    assert result.instruction_audit.memorised is True
    assert result.instruction_audit.predictors == ["generate_action", "extract"], (
        "compile_task audited one predictor, so an extract-only compile would read as clean"
    )
    assert result.instruction_audit.capabilities_kept == ["sub-LM"]


def test_a_program_with_no_planner_seam_leaves_the_audit_absent_not_crashed(monkeypatch):
    """A crash here would land AFTER the rollouts were paid for. Absent and visibly so is the
    only acceptable answer."""
    class _Prog:
        def named_predictors(self): return [("something_else", object())]
        def __call__(self, **kw): return type("R", (), {"answer": "a"})()

    class _GEPA:
        def __init__(self, **kw): pass
        def compile(self, program, **kw): return _Prog()

    monkeypatch.setattr(dspy, "GEPA", _GEPA)
    rlm_harness.configure(rlm_harness.RLMConfig(
        main_model="openai/x", sub_model="openai/x", api_key="k", interpreter="mock",
        max_retries=1))

    class _T(rlm_harness.RLMTask):
        signature = "q: str -> answer: str"
        output_field = "answer"
        instructions = "Answer."

    task = _T(interpreter=ScriptedInterpreter([]))
    monkeypatch.setattr(type(task), "_build_rlm", lambda self: _Prog())
    result = optimize.compile_task(task, [_ex("q", "a")], lambda e, p, *a, **k: 1.0,
                                   reflection_lm=object())
    assert result.instruction_audit is None
    assert result.program is not None


def test_a_plain_dict_trainset_still_finds_its_answer():
    """The fallback returned `str(example)`, a repr, which can never appear in an instruction, so
    a trainset of plain dicts reported no hit while the docstring promised the opposite
    direction. The rule `verify_quote`'s false negative was kept for: be wrong loudly."""
    answer = "the fingerprint is written by save_program when it is given the task argument"
    audit = optimize.audit_instruction(BASE, BASE + " Hint: " + answer,
                                       trainset=[{"q": "where?", "answer": answer}],
                                       holdout=[{"q": "x", "answer": "an unrelated gold value"}])
    assert audit.trainset_answers_in == [0]
    assert audit.memorised is True


def test_an_example_with_no_identifiable_gold_is_skipped_not_matched_on_its_repr():
    ex = dspy.Example(q="x", answer="a gold value long enough to pass the floor").with_inputs(
        "q", "answer")
    audit = optimize.audit_instruction(BASE, BASE + " noise", trainset=[ex], holdout=[ex])
    assert audit.trainset_answers_in == [] and audit.holdout_answers_in == []
