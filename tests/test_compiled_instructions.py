"""`RLMTask(compiled_instructions=...)`: run an optimizer's text, keep the kit's guards.

The seam exists because there was no way back. GEPA rewrites
`generate_action.signature.instructions` and `optimize.save_program` persists it, but
`_build_rlm` rebuilt the signature from `self.instructions` on every run, so a compiled program
had nowhere to go inside the kit's own run path. `docs/PROMPT.md` recorded that as open work for
a release.

What it deliberately does NOT do is accept a caller's whole program. Only the TEXT is taken, so
the sandbox guard, the per-pass interpreter factory, tool timing and description normalisation
all still apply: a `program=` seam would route every one of those around the kit.

The first version of this seam read the artifact to decide whether to trust it, and refused any
text that did not still contain the current build's rendered tool block. That rejected the only
artifact the release had actually measured, because an optimizer rewrites the instruction
WHOLESALE and writes its own account of the tools. What a wholesale rewrite destroys cannot be
recovered by reading what it produced, so the question is answered at save time instead: a
fingerprint of the signature and the tools' model-facing docstrings, written by
`save_program(program, path, task=...)` and checked here.
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, ConfigDict, Field, create_model

import rlm_harness
from rlm_harness import EVENT_COMPILED_INSTRUCTIONS, TraceRecorder, load_events

dspy = pytest.importorskip("dspy")

from rlm_harness import optimize
from rlm_harness.testing import ScriptedInterpreter

# What a real GEPA artifact looks like where it matters: the whole string replaced, the kit's
# rendered tool block gone, the tools described in the optimizer's own words. Taken from the
# shape of the run in `docs/PROMPT.md`, whose artifact the first version of this seam refused.
WHOLESALE = (
    "Answer from the document in the REPL.\n\n"
    "`read_file` is available as a REPL global. Its signature is "
    "`read_file(path, start_line=1, end_line=None)`.\n\n"
    "Work in two passes: locate, then quote. Call SUBMIT() with the answer."
)


class _Cite(rlm_harness.RLMTask):
    signature = "q: str -> answer: str"
    output_field = "answer"
    instructions = "Answer the question."


class _Opaque:
    """A type pydantic cannot render into a JSON schema."""


class Out(BaseModel):
    span: str = Field(..., description="Quote the sentence verbatim.")


#: Same class NAME as `Out`, which is what the signature string resolves, and a different field
#: description. A renamed class would fail in dspy's signature parser instead, well before here.
Out2 = create_model("Out", span=(str, Field(..., description="Summarize in your own words.")))


class Unrenderable(BaseModel):
    """`arbitrary_types_allowed` with a field pydantic has no JSON schema for."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    thing: _Opaque


class _Typed(rlm_harness.RLMTask):
    signature = "q: str -> out: Out"
    output_field = "out"
    output_model = Out
    instructions = "Answer."


class _Summarize(rlm_harness.RLMTask):
    signature = "q: str -> summary: str"
    output_field = "summary"
    instructions = "Summarize."


def _configure():
    rlm_harness.configure(rlm_harness.RLMConfig(
        main_model="openai/x", sub_model="openai/x", api_key="k",
        interpreter="mock", max_retries=1,
    ))


def _task(tmp_path, cls=_Cite, **kw):
    _configure()
    from rlm_harness.tools import make_read_file_tool

    return cls(interpreter=ScriptedInterpreter([]),
               tools=[make_read_file_tool(str(tmp_path))], **kw)


def _compile_like_an_optimizer(tmp_path, rewrite, cls=_Cite, name="state.json",
                               fingerprint=True, predictor="generate_action", **kw):
    """Save a built program and rewrite one predictor's text the way an optimizer would.

    A real GEPA run differs only in who writes the string. `fingerprint=False` is what a caller
    gets by forgetting `task=`, which is a state the load path has to refuse.
    """
    task = _task(tmp_path, cls=cls, **kw)
    path = tmp_path / name
    optimize.save_program(task._build_rlm(), str(path), task=task if fingerprint else None)
    state = json.loads(path.read_text())
    state[predictor]["signature"]["instructions"] = rewrite(
        state[predictor]["signature"]["instructions"]
    )
    path.write_text(json.dumps(state))
    return path


def test_without_it_nothing_changes(tmp_path):
    """The default ships what this version of the kit builds."""
    assert _Cite.compiled_instructions is None
    text = _task(tmp_path)._build_rlm().generate_action.signature.instructions
    assert "Answer the question." in text


def test_a_wholesale_rewrite_reaches_the_model(tmp_path):
    """The shape every real optimizer produces, and the one the first guard refused.

    Nothing of the kit's rendered instruction survives in the artifact: not the task's own
    instructions, not dspy's REPL protocol, not the tool block. That is what
    `with_instructions(candidate)` does, and refusing it refuses GEPA.
    """
    path = _compile_like_an_optimizer(tmp_path, lambda s: WHOLESALE)
    text = _task(tmp_path, compiled_instructions=str(path))._build_rlm() \
        .generate_action.signature.instructions
    assert text == WHOLESALE
    assert "Additional tools available" not in text, (
        "the artifact REPLACES the rendered tool documentation; it does not sit beside it"
    )


def test_the_kit_still_builds_the_program(tmp_path):
    """Only the text is loaded, so the program around it is still this task's.

    Asserted on the tools the program holds rather than on the instruction text, because a
    wholesale rewrite removes the rendered tool block: after a load the model knows about the
    tools only through whatever the artifact says, which is why the fingerprint has to hold.
    """
    path = _compile_like_an_optimizer(tmp_path, lambda s: WHOLESALE)
    rlm = _task(tmp_path, compiled_instructions=str(path))._build_rlm()
    doc = next(t.__doc__ for t in rlm.tools.values() if t.name == "read_file")
    assert "``" not in doc, "the tool description was not normalised, so this is not a kit build"
    assert rlm.tools["read_file"].func.__wrapped__, "the timing wrapper is gone"


def test_a_state_without_a_fingerprint_is_refused(tmp_path):
    """`save_program` without `task=` writes a file nothing can validate.

    The refusal names the call that fixes it. "Recompile against this version" was the first
    version's advice and it was a dead end: recompiling produced the same unverifiable shape.
    """
    path = _compile_like_an_optimizer(tmp_path, lambda s: WHOLESALE, fingerprint=False)
    with pytest.raises(ValueError, match="carries no record of what it was compiled against"):
        _task(tmp_path, compiled_instructions=str(path))._build_rlm()


def test_a_state_compiled_against_a_changed_tool_description_is_refused(tmp_path):
    """The hazard that makes this a refusal rather than a warning.

    A tool description is an instruction (`docs/PROMPT.md`), and the artifact carries the
    optimizer's account of the tools as they were. Since it replaces the rendered block rather
    than sitting beside it, a changed description never reaches the model at all: the model is
    told what the tool used to do, with nothing to correct it.
    """
    path = _compile_like_an_optimizer(tmp_path, lambda s: WHOLESALE)
    state = json.loads(path.read_text())
    state["rlm_harness"]["tools"]["read_file"] = "Read a file. OLD DESCRIPTION."
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="description changed=\\['read_file'\\]"):
        _task(tmp_path, compiled_instructions=str(path))._build_rlm()


def test_a_state_compiled_against_a_different_toolset_is_refused(tmp_path):
    path = _compile_like_an_optimizer(tmp_path, lambda s: WHOLESALE)
    state = json.loads(path.read_text())
    state["rlm_harness"]["tools"] = {"grep_files": "Search."}
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match=r"removed=\['grep_files'\], added=\['read_file'\]"):
        _task(tmp_path, compiled_instructions=str(path))._build_rlm()


def test_a_state_compiled_for_another_task_is_refused(tmp_path):
    """Two tasks sharing a toolset is the ordinary shape, so the tool check is not a task check.

    dspy zips a saved signature's fields onto the current ones by position, so without this the
    other task's output field name is what the model is asked for.
    """
    path = _compile_like_an_optimizer(tmp_path, lambda s: WHOLESALE, cls=_Summarize)
    with pytest.raises(ValueError, match="was compiled for the signature"):
        _task(tmp_path, cls=_Cite, compiled_instructions=str(path))._build_rlm()


def test_a_candidate_that_improved_only_extract_is_accepted(tmp_path):
    """dspy's GEPA selects components round-robin by default, so a winner that moved one
    predictor is an ordinary outcome. Reading only the planner called that "changed nothing"."""
    path = _compile_like_an_optimizer(
        tmp_path, lambda s: "Return only the answer span, no preamble.", predictor="extract"
    )
    rlm = _task(tmp_path, compiled_instructions=str(path))._build_rlm()
    assert rlm.extract.signature.instructions == "Return only the answer span, no preamble."
    assert "Answer the question." in rlm.generate_action.signature.instructions, (
        "the planner was not saved as changed, so it should still be what this build ships"
    )


def test_a_load_that_changes_nothing_is_refused(tmp_path):
    """A silent no-op would run the shipped instructions while the caller believes otherwise,
    which is worse than either outcome on its own."""
    path = _compile_like_an_optimizer(tmp_path, lambda s: s)
    with pytest.raises(ValueError, match="changed nothing"):
        _task(tmp_path, compiled_instructions=str(path))._build_rlm()


def test_an_empty_instruction_is_refused(tmp_path):
    """Both other checks pass for it: the text differs from the shipped string, and the
    fingerprint is this build's. An empty prompt is still not a prompt."""
    path = _compile_like_an_optimizer(tmp_path, lambda s: "   \n")
    with pytest.raises(ValueError, match="empty instruction"):
        _task(tmp_path, compiled_instructions=str(path))._build_rlm()


def test_demos_and_a_foreign_lm_in_the_state_never_reach_the_program(tmp_path):
    """Why the kit reads the file instead of calling `rlm.load`.

    dspy's loader restores the whole predictor. A state file could therefore put few-shot
    examples the kit never built in front of the model and replace the planner's model and
    sampling parameters, overriding `RLMConfig`. "Only the TEXT is taken" has to be what the code
    does, and this is the test that makes the sentence checkable.
    """
    path = _compile_like_an_optimizer(tmp_path, lambda s: WHOLESALE)
    state = json.loads(path.read_text())
    state["generate_action"]["demos"] = [{"q": "demo question", "answer": "demo answer"}]
    state["generate_action"]["lm"] = {"model": "openai/somebody-elses-model",
                                      "kwargs": {"temperature": 1.4, "max_tokens": 77}}
    path.write_text(json.dumps(state))

    rlm = _task(tmp_path, compiled_instructions=str(path))._build_rlm()
    assert rlm.generate_action.demos == [], "few-shot demos arrived from a state file"
    assert rlm.generate_action.lm is None, "a state file replaced the planner's LM"
    assert rlm.generate_action.signature.instructions == WHOLESALE, "the text should still load"


def test_a_missing_file_names_the_field_that_asked_for_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="compiled_instructions="):
        _task(tmp_path, compiled_instructions=str(tmp_path / "nope.json"))._build_rlm()


def test_a_file_that_is_not_a_state_names_the_field(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json at all")
    with pytest.raises(ValueError, match="not readable as a dspy JSON state file"):
        _task(tmp_path, compiled_instructions=str(bad))._build_rlm()


def test_the_trace_records_which_predictors_were_applied(tmp_path):
    """Every later reading of a trace is otherwise on a false premise, and nothing else in the
    file would say so. The kit version that compiled it is recorded because a reader asking
    "which build wrote this prompt" has nowhere else to look."""
    path = _compile_like_an_optimizer(tmp_path, lambda s: WHOLESALE)
    trace = tmp_path / "t.jsonl"
    with TraceRecorder(str(trace), run_id="r"):
        _task(tmp_path, compiled_instructions=str(path))._build_rlm()
    events = [e for e in load_events(str(trace)) if e["type"] == EVENT_COMPILED_INSTRUCTIONS]
    assert len(events) == 1, [e["type"] for e in load_events(str(trace))]
    payload = events[0]["payload"]
    assert payload["source"] == str(path)
    assert payload["compiled_by"] == rlm_harness.__version__
    assert list(payload["applied"]) == ["generate_action"]
    assert payload["applied"]["generate_action"]["loaded_chars"] == len(WHOLESALE)


def test_a_kit_version_bump_alone_does_not_refuse(tmp_path):
    """The fingerprint records the version and does not check it: refusing an artifact that is
    still exactly right, because the kit moved, would make every release invalidate every
    artifact."""
    path = _compile_like_an_optimizer(tmp_path, lambda s: WHOLESALE)
    state = json.loads(path.read_text())
    state["rlm_harness"]["kit"] = "0.0.1"
    path.write_text(json.dumps(state))
    rlm = _task(tmp_path, compiled_instructions=str(path))._build_rlm()
    assert rlm.generate_action.signature.instructions == WHOLESALE


def test_the_kwarg_wins_over_the_class_field(tmp_path, monkeypatch):
    """Same order-independence as `tools=`: resolved at build time, so the explicit kwarg wins
    however the subclass is written."""
    path = _compile_like_an_optimizer(tmp_path, lambda s: WHOLESALE)
    monkeypatch.setattr(_Cite, "compiled_instructions", "/nonexistent/state.json")
    text = _task(tmp_path, compiled_instructions=str(path))._build_rlm() \
        .generate_action.signature.instructions
    assert text == WHOLESALE


def test_the_class_field_is_used_when_no_kwarg_is_given(tmp_path, monkeypatch):
    path = _compile_like_an_optimizer(tmp_path, lambda s: WHOLESALE)
    monkeypatch.setattr(_Cite, "compiled_instructions", str(path))
    text = _task(tmp_path)._build_rlm().generate_action.signature.instructions
    assert text == WHOLESALE


def test_an_empty_string_turns_a_class_level_artifact_off(tmp_path, monkeypatch):
    """The A/B a subclass needs to decide whether to keep an artifact at all. `tools=[]` means
    "no tools" for the same reason, and both read `is not None` so the empty value survives."""
    path = _compile_like_an_optimizer(tmp_path, lambda s: WHOLESALE)
    monkeypatch.setattr(_Cite, "compiled_instructions", str(path))
    text = _task(tmp_path, compiled_instructions="")._build_rlm() \
        .generate_action.signature.instructions
    assert "Answer the question." in text, "the class-level artifact could not be turned off"


def test_a_type_error_from_the_load_is_not_read_as_a_budget_rejection(tmp_path):
    """The load runs OUTSIDE the `try` that catches dspy's budget-kwarg TypeError.

    Inside it, the TypeError this path raises when the installed dspy has no instruction seam was
    swallowed: `_build_rlm` logged that dspy had rejected the budget kwargs and set
    `_budget_caps_dropped`, which writes the configured caps into the trace as dropped when they
    were not. The caller still got the right exception from the rebuild, so only the trace and
    the log were wrong, which is the half nobody checks.
    """
    task = _task(tmp_path, compiled_instructions=str(tmp_path / "unused.json"))

    def _no_seam(self, rlm):
        raise TypeError("no instruction seam in this dspy")

    type(task)._apply_compiled_instructions = _no_seam
    try:
        with pytest.raises(TypeError, match="no instruction seam"):
            task._build_rlm()
    finally:
        del type(task)._apply_compiled_instructions
    assert task._budget_caps_dropped is False, (
        "a TypeError from the compiled-instructions load was read as dspy rejecting the budget "
        "kwargs, so the trace would record caps as dropped when they were applied"
    )


def test_a_predictor_block_that_is_not_a_state_is_skipped_not_crashed(tmp_path):
    """Every other malformed input is restated in terms of the field that asked for the file, and
    a predictor whose `signature` is a string used to reach `.get` on it and raise
    AttributeError out of the middle of a build."""
    path = _compile_like_an_optimizer(tmp_path, lambda s: WHOLESALE)
    state = json.loads(path.read_text())
    state["extract"]["signature"] = "not a signature state"
    path.write_text(json.dumps(state))
    rlm = _task(tmp_path, compiled_instructions=str(path))._build_rlm()
    assert rlm.generate_action.signature.instructions == WHOLESALE, "the readable half should load"


def test_a_non_json_path_is_refused_before_anything_is_written(tmp_path):
    """`save_program` raised after `program.save`, so a fingerprint-less file was already on disk
    under the name the caller was about to reuse, and the load path refuses exactly that file."""
    task = _task(tmp_path)
    path = tmp_path / "state.pkl"
    with pytest.raises(ValueError, match="must end with .json"):
        optimize.save_program(task._build_rlm(), str(path), task=task)
    assert not path.exists(), "a state file was written by a call that raised"


def test_a_reworded_output_field_description_is_refused(tmp_path):
    """The output model's JSON schema is interpolated into the planner prompt, measured at 36.9%
    of one real task's instruction, and the signature STRING does not move when a field's
    description is rewritten. Without this, the artifact states an output contract the task no
    longer validates and the model is asked for the old shape."""
    path = _compile_like_an_optimizer(tmp_path, lambda s: WHOLESALE, cls=_Typed)
    try:
        _Typed.output_model = Out2
        with pytest.raises(ValueError, match="different output schema"):
            _task(tmp_path, cls=_Typed, compiled_instructions=str(path))._build_rlm()
    finally:
        _Typed.output_model = Out


def test_a_sandbox_described_differently_is_refused(tmp_path):
    """`execution_instructions` is prompt: one sandbox says code is not executed, another says
    subprocesses are available. An artifact carries its compile-time copy."""
    path = _compile_like_an_optimizer(tmp_path, lambda s: WHOLESALE)
    state = json.loads(path.read_text())
    state["rlm_harness"]["execution_instructions"] = "This REPL can run subprocesses."
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="sandbox described differently"):
        _task(tmp_path, compiled_instructions=str(path))._build_rlm()


def test_a_fingerprint_key_this_build_does_not_write_is_not_compared(tmp_path):
    """Key-wise, both sides. A key added in a later release would otherwise retroactively refuse
    every artifact saved before it, which is a release invalidating work rather than guarding it."""
    path = _compile_like_an_optimizer(tmp_path, lambda s: WHOLESALE)
    state = json.loads(path.read_text())
    state["rlm_harness"]["something_1_20_0_will_check"] = "whatever"
    del state["rlm_harness"]["output_schema"]
    path.write_text(json.dumps(state))
    rlm = _task(tmp_path, compiled_instructions=str(path))._build_rlm()
    assert rlm.generate_action.signature.instructions == WHOLESALE


def test_a_state_saved_from_a_wrapping_module_says_so(tmp_path):
    """dspy prefixes each predictor with the attribute it hangs off, so a state saved from a
    module wrapping the RLM names `rlm.generate_action`. That matched no predictor, `applied`
    stayed empty, and the refusal said every saved instruction was already what the build ships:
    the same misdiagnosis class as reading one predictor and calling it a no-op."""
    path = _compile_like_an_optimizer(tmp_path, lambda s: WHOLESALE)
    state = json.loads(path.read_text())
    for name in ("generate_action", "extract"):
        state[f"rlm.{name}"] = state.pop(name)
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="names none of this build's predictors"):
        _task(tmp_path, compiled_instructions=str(path))._build_rlm()


def test_a_directory_and_a_pickle_are_refused_in_terms_of_the_field(tmp_path):
    """Both died on a bare `IsADirectoryError` / `UnicodeDecodeError` naming neither the file nor
    the field, and the only message that mentions pickles lived where a pickle never arrives."""
    import pickle

    with pytest.raises(ValueError, match="not readable as a dspy JSON state file"):
        _task(tmp_path, compiled_instructions=str(tmp_path))._build_rlm()

    pkl = tmp_path / "state.pkl"
    pkl.write_bytes(pickle.dumps({"generate_action": {}}))
    with pytest.raises(ValueError, match=r"\.pkl"):
        _task(tmp_path, compiled_instructions=str(pkl))._build_rlm()


def test_a_fingerprint_whose_tools_are_the_wrong_shape_is_refused(tmp_path):
    """`TypeError: list indices must be integers` out of a build names neither the file nor the
    field, and in this module `TypeError` means the installed dspy is wrong."""
    path = _compile_like_an_optimizer(tmp_path, lambda s: WHOLESALE)
    state = json.loads(path.read_text())
    state["rlm_harness"]["tools"] = ["read_file"]
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="records its tools as list"):
        _task(tmp_path, compiled_instructions=str(path))._build_rlm()


def test_the_trace_says_whether_the_tasks_own_instructions_moved(tmp_path):
    """Recorded rather than refused, because a typo fix should not invalidate an artifact. A
    caller who GENERATES that prose, a skills manifest being the case the kit ships, needs the
    difference to be legible somewhere."""
    path = _compile_like_an_optimizer(tmp_path, lambda s: WHOLESALE)
    trace = tmp_path / "t.jsonl"
    _Cite.instructions = "Answer the question, citing the file."
    try:
        with TraceRecorder(str(trace), run_id="r"):
            _task(tmp_path, compiled_instructions=str(path))._build_rlm()
    finally:
        _Cite.instructions = "Answer the question."
    event = next(e for e in load_events(str(trace)) if e["type"] == EVENT_COMPILED_INSTRUCTIONS)
    assert event["payload"]["task_instructions_changed"] is True


def test_the_saved_record_carries_every_key_the_load_checks(tmp_path):
    """Otherwise a key-wise check skips the missing ones and protects nothing.

    `save_program` has to pass the built program for `execution_instructions` to be in the file at
    all, and this is the assertion that says so: every test for a given refusal would otherwise
    pass by injecting the key into the state by hand.
    """
    task = _task(tmp_path)
    path = tmp_path / "fresh.json"
    optimize.save_program(task._build_rlm(), str(path), task=task)
    saved = json.loads(path.read_text())[optimize.FINGERPRINT_KEY]
    missing = [k for k in type(task)._CHECKED_FINGERPRINT_KEYS if k not in saved]
    assert not missing, f"save_program wrote no {missing}, so the load would not compare it"
    assert saved["execution_instructions"], "the sandbox description was recorded empty"


def test_a_state_saved_from_a_wrapping_module_is_refused_for_the_right_reason(tmp_path):
    """End to end, which is the scenario the predictor-name message describes.

    The sandbox description can only be read off the program that carries the interpreter factory,
    and a wrapping module does not. Recording the empty read would refuse this state for a sandbox
    that changed from nothing to something, which names the wrong mistake: the caller saved the
    wrong object. So an empty read is not recorded, and the predictor-name check is what answers.
    """
    task = _task(tmp_path)
    inner = task._build_rlm()

    class Wrapper(dspy.Module):
        def __init__(self) -> None:
            super().__init__()
            self.rlm = inner

    wrapper = Wrapper()
    wrapper.rlm.generate_action.signature = \
        wrapper.rlm.generate_action.signature.with_instructions(WHOLESALE)
    path = tmp_path / "wrapped.json"
    optimize.save_program(wrapper, str(path), task=task)

    saved = json.loads(path.read_text())
    assert "execution_instructions" not in saved[optimize.FINGERPRINT_KEY], (
        "an empty read was recorded as the sandbox description"
    )
    assert "rlm.generate_action" in saved, "dspy did not prefix the predictor as expected"

    with pytest.raises(ValueError, match="names none of this build's predictors"):
        _task(tmp_path, compiled_instructions=str(path))._build_rlm()


def test_the_budget_is_recorded_and_reported_but_not_refused(tmp_path):
    """dspy interpolates the caps into the instruction as "You have max N sub-LLM calls", so they
    are a model-facing owner of the string the artifact replaces. Raising a cap must not
    invalidate an artifact, and an artifact compiled at fifty and run under five tells the model a
    budget the kit will cut off, so the difference has to be visible somewhere."""
    path = _compile_like_an_optimizer(tmp_path, lambda s: WHOLESALE)
    budget = json.loads(path.read_text())[optimize.FINGERPRINT_KEY]["budget"]
    # The KIT's names, not the ones the installed dspy accepts: `_dspy_compat.rlm_budget_kwargs`
    # resolves each cap onto the current spelling, and 3.3.x renamed one. Recording the resolved
    # names would read as a changed budget across such a rename with all three numbers identical.
    # No dspy inside the declared bound renames one, so this assertion is red only on the day the
    # distinction starts to matter, which is the day someone moves the bound.
    assert set(budget) == {"max_iterations", "max_llm_calls", "max_output_chars"}, budget

    rlm_harness.configure(rlm_harness.RLMConfig(
        main_model="openai/x", sub_model="openai/x", api_key="k", interpreter="mock",
        max_retries=1, max_llm_calls=5,
    ))
    trace = tmp_path / "t.jsonl"
    from rlm_harness.tools import make_read_file_tool
    with TraceRecorder(str(trace), run_id="r"):
        _Cite(interpreter=ScriptedInterpreter([]), tools=[make_read_file_tool(str(tmp_path))],
              compiled_instructions=str(path))._build_rlm()
    event = next(e for e in load_events(str(trace)) if e["type"] == EVENT_COMPILED_INSTRUCTIONS)
    # The NAMES, because only `max_llm_calls` reaches the string: a boolean would fire for two
    # caps the model is never told about, while every other flag in this payload answers "did what
    # the model saw move".
    assert event["payload"]["budget_changed"] == ["max_llm_calls"], (
        "the trace does not say WHICH cap moved, so a reader cannot tell whether the one the "
        "model is told about is the one that changed"
    )


def test_an_unrenderable_output_model_does_not_break_a_working_task(tmp_path):
    """`arbitrary_types_allowed` with a field pydantic has no schema for raises out of
    `model_json_schema()`, and such a task builds and runs today. Raising here would mean setting
    `compiled_instructions` breaks a task that works, with a pydantic internal naming nothing."""
    class _Odd(rlm_harness.RLMTask):
        signature = "q: str -> a: str"
        output_field = "a"
        output_model = Unrenderable
        instructions = "Answer."

    path = _compile_like_an_optimizer(tmp_path, lambda s: WHOLESALE, cls=_Odd)
    fp = json.loads(path.read_text())[optimize.FINGERPRINT_KEY]
    assert fp["output_schema"] == {"unrenderable": "Unrenderable"}
    rlm = _task(tmp_path, cls=_Odd, compiled_instructions=str(path))._build_rlm()
    assert rlm.generate_action.signature.instructions == WHOLESALE


def test_a_malformed_dspy_metadata_block_does_not_kill_a_traced_run(tmp_path):
    """It crashed AFTER the text was applied and ONLY with a recorder active, so a traced run died
    on a file an untraced run accepted, which reads as "tracing broke the run"."""
    path = _compile_like_an_optimizer(tmp_path, lambda s: WHOLESALE)
    state = json.loads(path.read_text())
    state["metadata"] = "not a metadata block"
    path.write_text(json.dumps(state))
    trace = tmp_path / "t.jsonl"
    with TraceRecorder(str(trace), run_id="r"):
        _task(tmp_path, compiled_instructions=str(path))._build_rlm()
    event = next(e for e in load_events(str(trace)) if e["type"] == EVENT_COMPILED_INSTRUCTIONS)
    assert event["payload"]["compiled_with_dspy"] is None


def test_the_trace_distinguishes_an_absent_record_from_an_unchanged_one(tmp_path):
    """`None`, not `True`: a record that predates a key says nothing about whether the thing
    moved, and this is the one event whose stated purpose is that a reader is otherwise on a false
    premise. The checked keys an old record lacks are listed for the same reason."""
    path = _compile_like_an_optimizer(tmp_path, lambda s: WHOLESALE)
    state = json.loads(path.read_text())
    del state[optimize.FINGERPRINT_KEY]["instructions"]
    del state[optimize.FINGERPRINT_KEY]["output_schema"]
    path.write_text(json.dumps(state))
    trace = tmp_path / "t.jsonl"
    with TraceRecorder(str(trace), run_id="r"):
        _task(tmp_path, compiled_instructions=str(path))._build_rlm()
    payload = next(e for e in load_events(str(trace))
                   if e["type"] == EVENT_COMPILED_INSTRUCTIONS)["payload"]
    assert payload["task_instructions_changed"] is None
    assert payload["unchecked"] == ["output_schema"]


def test_a_checked_key_THIS_BUILD_cannot_produce_is_reported_unchecked(tmp_path):
    """The skip is two-sided and the field read one side.

    A caller-supplied interpreter double carries no `execution_instructions`, which is the
    documented `interpreter=` seam, so the read comes back empty, this build's record omits the
    key, the sandbox gate does not run, and the trace used to say nothing was skipped. A trace
    already written is the one artifact a later release cannot correct.
    """
    class _Double:
        def __init__(self) -> None:
            self.tools: dict = {}

        def start(self) -> None:
            pass

        def execute(self, code: str, variables: dict | None = None) -> str:
            return ""

        def shutdown(self) -> None:
            pass

    path = _compile_like_an_optimizer(tmp_path, lambda s: WHOLESALE)
    assert "execution_instructions" in json.loads(path.read_text())[optimize.FINGERPRINT_KEY]

    trace = tmp_path / "t.jsonl"
    from rlm_harness.tools import make_read_file_tool
    _configure()
    with TraceRecorder(str(trace), run_id="r"):
        _Cite(interpreter=_Double(), tools=[make_read_file_tool(str(tmp_path))],
              compiled_instructions=str(path))._build_rlm()
    payload = next(e for e in load_events(str(trace))
                   if e["type"] == EVENT_COMPILED_INSTRUCTIONS)["payload"]
    assert payload["unchecked"] == ["execution_instructions"], (
        "the sandbox gate did not run and the trace did not say so"
    )
