"""`RLMTask(compiled_instructions=...)`: run an optimizer's text, keep the kit's guards.

The seam exists because there was no way back. GEPA rewrites
`generate_action.signature.instructions` and `optimize.save_program` persists it, but
`_build_rlm` rebuilt the signature from `self.instructions` on every run, so a compiled program
had nowhere to go inside the kit's own run path. `docs/PROMPT.md` recorded that as open work for
a release.

What it deliberately does NOT do is accept a caller's whole program. Only the TEXT is taken, so
the sandbox guard, the per-pass interpreter factory, tool timing and description normalisation
all still apply: a `program=` seam would route every one of those around the kit.
"""

from __future__ import annotations

import json

import pytest

import rlm_harness
from rlm_harness import EVENT_COMPILED_INSTRUCTIONS, TraceRecorder, load_events

dspy = pytest.importorskip("dspy")

from rlm_harness import optimize
from rlm_harness.testing import ScriptedInterpreter


class _Cite(rlm_harness.RLMTask):
    signature = "q: str -> answer: str"
    output_field = "answer"
    instructions = "Answer the question."


def _configure():
    rlm_harness.configure(rlm_harness.RLMConfig(
        main_model="openai/x", sub_model="openai/x", api_key="k",
        interpreter="mock", max_retries=1,
    ))


def _task(tmp_path, **kw):
    _configure()
    from rlm_harness.tools import make_read_file_tool

    return _Cite(interpreter=ScriptedInterpreter([]),
                 tools=[make_read_file_tool(str(tmp_path))], **kw)


def _compile_like_an_optimizer(tmp_path, rewrite):
    """Save a built program, rewrite its planner text the way an optimizer would, and return the
    path. A real GEPA run differs only in who writes the string."""
    built = _task(tmp_path)._build_rlm()
    path = tmp_path / "state.json"
    optimize.save_program(built, str(path))
    state = json.loads(path.read_text())
    state["generate_action"]["signature"]["instructions"] = rewrite(
        state["generate_action"]["signature"]["instructions"]
    )
    path.write_text(json.dumps(state))
    return path


def test_without_it_nothing_changes(tmp_path):
    """The default ships what this version of the kit builds."""
    assert _Cite.compiled_instructions is None
    text = _task(tmp_path)._build_rlm().generate_action.signature.instructions
    assert "Answer the question." in text


def test_an_optimizers_text_reaches_the_model(tmp_path):
    marker = "OPTIMIZED: answer with the file and the line, nothing else."
    path = _compile_like_an_optimizer(tmp_path, lambda s: marker + "\n" + s)
    text = _task(tmp_path, compiled_instructions=str(path))._build_rlm() \
        .generate_action.signature.instructions
    assert text.startswith(marker), text[:120]


def test_the_kit_still_builds_the_program(tmp_path):
    """Only the TEXT is loaded. The tools are still the ones this task resolved, still timed and
    still normalised, because the program was built here rather than handed over."""
    path = _compile_like_an_optimizer(tmp_path, lambda s: "OPTIMIZED\n" + s)
    rlm = _task(tmp_path, compiled_instructions=str(path))._build_rlm()
    text = rlm.generate_action.signature.instructions
    assert "read_file" in text, "the kit's own tool documentation is gone"
    assert "``" not in text.split("Additional tools available")[-1], (
        "the tool block was not normalised, so this is not a kit-built program"
    )


def test_a_state_compiled_against_other_tools_is_refused(tmp_path):
    """The hazard that made this a refusal rather than a warning.

    dspy persists the whole flat string, tool documentation included, so a state compiled against
    an older kit reinstates that kit's descriptions. Past 1.18.0 that means markup and ten-space
    gaps come back; on any upgrade it means the model is told about tools as they used to be. A
    description is an instruction, and a stale one states behaviour the code no longer has.
    """
    path = _compile_like_an_optimizer(
        tmp_path, lambda s: "OPTIMIZED\n\nAdditional tools available:\n- `read_file(path)` - old."
    )
    with pytest.raises(ValueError, match="compiled against different tool documentation"):
        _task(tmp_path, compiled_instructions=str(path))._build_rlm()


def test_a_load_that_changes_nothing_is_refused(tmp_path):
    """A silent no-op would run the shipped instructions while the caller believes otherwise,
    which is worse than either outcome on its own."""
    path = _compile_like_an_optimizer(tmp_path, lambda s: s)
    with pytest.raises(ValueError, match="changed nothing"):
        _task(tmp_path, compiled_instructions=str(path))._build_rlm()


def test_the_run_says_in_its_trace_that_it_was_not_the_shipped_prompt(tmp_path):
    """Every later reading of a trace is otherwise on a false premise, and nothing else in the
    file would say so."""
    path = _compile_like_an_optimizer(tmp_path, lambda s: "OPTIMIZED\n" + s)
    trace = tmp_path / "t.jsonl"
    with TraceRecorder(str(trace), run_id="r"):
        _task(tmp_path, compiled_instructions=str(path))._build_rlm()
    events = [e for e in load_events(str(trace)) if e["type"] == EVENT_COMPILED_INSTRUCTIONS]
    assert len(events) == 1, [e["type"] for e in load_events(str(trace))]
    payload = events[0]["payload"]
    assert payload["source"] == str(path)
    assert payload["loaded_chars"] > payload["shipped_chars"]
    assert payload["tool_block_preserved"] is True


def test_the_kwarg_wins_over_the_class_field(tmp_path, monkeypatch):
    """Same order-independence as `tools=`: resolved at build time, so the explicit kwarg wins
    however the subclass is written."""
    path = _compile_like_an_optimizer(tmp_path, lambda s: "FROM THE KWARG\n" + s)
    monkeypatch.setattr(_Cite, "compiled_instructions", "/nonexistent/state.json")
    text = _task(tmp_path, compiled_instructions=str(path))._build_rlm() \
        .generate_action.signature.instructions
    assert text.startswith("FROM THE KWARG")


def test_the_class_field_is_used_when_no_kwarg_is_given(tmp_path, monkeypatch):
    path = _compile_like_an_optimizer(tmp_path, lambda s: "FROM THE CLASS\n" + s)
    monkeypatch.setattr(_Cite, "compiled_instructions", str(path))
    text = _task(tmp_path)._build_rlm().generate_action.signature.instructions
    assert text.startswith("FROM THE CLASS")
