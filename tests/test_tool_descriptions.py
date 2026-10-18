"""A tool's docstring is PROMPT TEXT, and these tests are the only thing that says so.

dspy takes `dspy.Tool.desc` from a tool's `__doc__`, renders one line per tool in
`RLM._format_tool_docs`, appends the block to `generate_action.signature.instructions`, and the
adapter puts that string in the SYSTEM message. Measured on a two-tool task: the rendered tool-doc
block was 4957 of the instruction string's 7799 characters, 63.6%, inside an 8938-char system
message whose user-message counterpart was 448.

The kit wrote them for Sphinx. dspy flattens with `.replace("\\n", "  ")`, which
turns Python's own indentation into runs of ten spaces and passes ``literals`` through verbatim.
Of the fourteen tools the kit shipped then, THREE had no model-facing docstring at all, so dspy rendered
the literal words "No description"; of the eleven that had one, eight carried Sphinx markup and
ten produced markup or a mid-sentence gap.

These are failure-shaped assertions, not style checks: each one is text a model read.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import sqlite3
import tempfile

import pytest
from pydantic import BaseModel

import rlm_harness.tools as T
from rlm_harness._toolname import model_facing_description, with_model_facing_description
from rlm_harness.memory import MemoryStore

# How dspy renders a description into the prompt: `rlm.py` `_format_tool_docs`.
DSPY_FLATTEN = "  "

RST = re.compile(r"``[^`\n]+``|:[a-zA-Z][a-zA-Z0-9_:+-]*:`[^`\n]+`")
WHITESPACE_RUN = re.compile(r" {4,}")


class _Out(BaseModel):
    x: int = 0


def _build_every_shipped_tool(root: str) -> dict:
    """Every tool the kit hands a model, built with doubles for its injected halves.

    The table is explicit so a NEW factory fails `test_the_table_covers_every_shipped_factory`
    rather than silently escaping the rule.
    """
    def noop(*a, **k):
        return ""

    return {
        "read_file": T.make_read_file_tool(root),
        "grep_files": T.make_grep_files_tool(root, []),
        "write_file": T.make_write_file_tool(root),
        "edit_file": T.make_edit_file_tool(root),
        "extract_archive": T.make_extract_archive_tool(root),
        "fetch_url": T.make_fetch_tool(noop),
        "web_search": T.make_web_search_tool(lambda q: []),
        "run_command": T.make_command_tool(noop),
        "git_clone": T.make_git_clone_tool(root, noop),
        "harness": T.make_harness_tool(noop, noop),
        "model": T.make_model_tool(noop, noop),
        "schema_validator": T.make_schema_validator(_Out),
        "json_schema_validator": T.make_json_schema_validator({"type": "object"}),
        "memory_search": T.make_memory_search_tool(MemoryStore(os.path.join(root, "m.db"))),
        "memory_get": T.make_memory_get_tool(MemoryStore(os.path.join(root, "m.db"))),
        "memory_note": T.make_memory_note_tool(MemoryStore(os.path.join(root, "m.db"))),
        "verify_quote": T.verify_quote,
    }


FACTORIES = sorted(n for n in T.__all__ if n.startswith("make_"))


def test_the_table_covers_every_shipped_factory():
    """A new tool factory must be added to the table above, which is what subjects its docstring
    to the rest of this file. Without this, adding a tool is how the rule gets skipped.

    Asserted by MEMBERSHIP, not by count: `len(built) == len(FACTORIES) + 1` also holds after a
    key is renamed, so the test would have carried less than its name.
    """
    with tempfile.TemporaryDirectory() as root:
        built = _build_every_shipped_tool(root)
    #: The table's key for each factory's tool. Spelled out so a rename fails HERE, naming both
    #: sides, rather than somewhere downstream.
    expected = {
        "make_command_tool": "run_command",
        "make_edit_file_tool": "edit_file",
        "make_extract_archive_tool": "extract_archive",
        "make_fetch_tool": "fetch_url",
        "make_git_clone_tool": "git_clone",
        "make_grep_files_tool": "grep_files",
        "make_harness_tool": "harness",
        "make_json_schema_validator": "json_schema_validator",
        "make_memory_get_tool": "memory_get",
        "make_memory_note_tool": "memory_note",
        "make_memory_search_tool": "memory_search",
        "make_model_tool": "model",
        "make_read_file_tool": "read_file",
        "make_schema_validator": "schema_validator",
        "make_web_search_tool": "web_search",
        "make_write_file_tool": "write_file",
    }
    assert set(expected) == set(FACTORIES), (
        "a tool factory was added or renamed in rlm_harness.tools.__all__: "
        f"{sorted(set(FACTORIES) ^ set(expected))}. Add it to _build_every_shipped_tool and to "
        "this map, so its model-facing description is covered."
    )
    missing = sorted((set(expected.values()) | {"verify_quote"}) - set(built))
    assert not missing, f"the table does not build: {missing}"


@pytest.fixture(scope="module")
def shipped():
    with tempfile.TemporaryDirectory() as root:
        yield _build_every_shipped_tool(root)


def test_every_shipped_tool_has_a_description_at_all(shipped):
    """An empty docstring makes dspy render the literal string "No description" into the system
    message. `web_search` shipped that way: a tool offered to the model with nothing said about
    it."""
    missing = [n for n, t in shipped.items() if not (t.__doc__ or "").strip()]
    assert not missing, f"no model-facing description: {missing}"


def test_no_shipped_description_carries_sphinx_markup(shipped):
    """``start_line`` and :func:`~mod.name` are instructions to a doc build. The model is not a
    doc build; it reads the backticks."""
    leaking = {
        n: RST.findall(model_facing_description(t.__doc__))
        for n, t in shipped.items()
        if RST.search(model_facing_description(t.__doc__))
    }
    assert not leaking, f"Sphinx markup survives normalisation: {leaking}"


def test_no_shipped_description_collapses_into_whitespace_runs(shipped):
    """The defect that started this: dspy replaces each newline with two spaces, so a docstring's
    8-space indentation became a 10-space gap mid-sentence."""
    bad = {}
    for name, tool in shipped.items():
        rendered = model_facing_description(tool.__doc__).replace("\n", DSPY_FLATTEN)
        if WHITESPACE_RUN.search(rendered):
            bad[name] = WHITESPACE_RUN.findall(rendered)[:3]
    assert not bad, f"whitespace runs reach the prompt: {bad}"


def test_the_raw_docstrings_really_are_the_thing_being_fixed(shipped):
    """The CONTROL. Without it every assertion above could pass because `model_facing_description`
    is applied to text that never had the defect, and the three tests would be measuring nothing.

    Asserted as a floor, not an exact count: the point is that the raw docstrings DO carry markup
    today, so normalisation is load-bearing rather than decorative.
    """
    raw_leaks = [
        n for n, t in shipped.items()
        if RST.search(t.__doc__ or "") or WHITESPACE_RUN.search((t.__doc__ or "").replace("\n", DSPY_FLATTEN))
    ]
    assert len(raw_leaks) >= 5, (
        "the raw docstrings no longer carry Sphinx markup, so these tests prove nothing: "
        f"only {raw_leaks} still do. Either the docstrings were rewritten by hand (then relax "
        "this control) or the normaliser is being tested against already-clean input."
    )


def test_normalisation_keeps_what_the_sentence_said(shipped):
    """Not a style pass: the words have to survive. A model that is told less is told wrong."""
    rendered = model_facing_description(shipped["read_file"].__doc__)
    for fragment in ("1-indexed", "relative to the root", "Refused:", "never raises"):
        assert fragment in rendered, f"{fragment!r} was lost: {rendered}"


# --- the seam ----------------------------------------------------------------------------------

def test_the_normaliser_is_actually_wired_into_a_built_task():
    """END TO END, and the one that catches the regression that matters: a correct normaliser that
    nobody calls. Reverting `_build_rlm`'s `with_model_facing_description(...)` reds this."""
    pytest.importorskip("dspy")
    import rlm_harness
    from rlm_harness.testing import ScriptedInterpreter

    with tempfile.TemporaryDirectory() as root:
        read_file = T.make_read_file_tool(root)
        assert "``" in (read_file.__doc__ or ""), "the source docstring must still carry markup"

        class _T(rlm_harness.RLMTask):
            signature = "q: str -> answer: str"
            output_field = "answer"

        rlm_harness.configure(rlm_harness.RLMConfig(
            main_model="openai/x", sub_model="openai/x", api_key="k",
            interpreter="mock", max_retries=1,
        ))
        task = _T(interpreter=ScriptedInterpreter([]), tools=[read_file])
        instructions = task._build_rlm().generate_action.signature.instructions

    assert "read_file" in instructions, "the tool doc did not reach the instructions at all"
    i = instructions.find("- `read_file(")
    tool_block = instructions[i:]
    assert not RST.search(tool_block), f"Sphinx markup reached the prompt: {RST.findall(tool_block)[:3]}"
    assert not WHITESPACE_RUN.search(tool_block), (
        f"whitespace runs reached the prompt: {WHITESPACE_RUN.findall(tool_block)[:3]}"
    )
    assert "1-indexed" in tool_block, "normalisation dropped the sentence's content"


def test_a_consumers_own_function_is_never_mutated():
    """The normaliser writes on the kit's `functools.wraps` wrapper, so a consumer's tool keeps
    the docstring they wrote. Mutating it would change `help()` on an object they own."""
    from rlm_harness.trace import _ensure_tool_timing

    def their_tool(x: str) -> str:
        """Theirs, with ``markup`` they chose."""
        return x

    original = their_tool.__doc__
    wrapped = with_model_facing_description(_ensure_tool_timing(their_tool))
    assert their_tool.__doc__ == original, "the consumer's own function was rewritten"
    assert "``markup``" not in (wrapped.__doc__ or ""), "the wrapper was not normalised"


def test_a_tool_the_timing_seam_skipped_is_left_completely_alone():
    """The guard, measured on an object that really is the CALLER's.

    The subject has to be a shape the timing seam SKIPS. A tool that went through the seam is the
    kit's own wrapper either way, so removing the guard changes nothing and the test proves
    nothing. A callable instance is skipped (not a function or method), arrives here as the
    consumer's object, and has a settable `__doc__`: the exact case the guard exists for.
    """
    import functools

    class Theirs:
        """A callable instance with ``markup`` they chose."""

        def __call__(self, x: str) -> str:
            return x

    theirs = Theirs()
    before = theirs.__doc__
    assert "``markup``" in before
    assert with_model_facing_description(theirs) is theirs
    assert theirs.__doc__ == before, "the consumer's own object was rewritten"

    def base(x: str, y: str) -> str:
        """Base with ``markup``."""
        return x + y

    part = functools.partial(base, "a")
    assert with_model_facing_description(part) is part
    assert base.__doc__ == "Base with ``markup``."


def test_an_explicit_desc_is_normalised_by_copy_not_mutation():
    """How MCP arrives: `mcp.py` builds `dspy.Tool(call, name=…, desc=server_description)`, and
    dspy reads `desc` and ignores `__doc__` there. The text belongs to a remote server, so no
    docstring convention in this kit can reach it; only this seam can."""
    dspy = pytest.importorskip("dspy")

    def call(q: str) -> str:
        return q

    tool = dspy.Tool(call, name="remote", desc="Query the ``index``.\n\n    Returns ``rows``.")
    out = with_model_facing_description(tool)
    assert out is not tool, "the server's Tool object was mutated in place"
    assert out.desc == "Query the `index`. Returns `rows`."
    assert tool.desc.startswith("Query the ``index``."), "the original was modified"
    assert out.name == "remote"


def test_a_description_that_is_already_clean_is_returned_unchanged():
    """No churn: a clean `desc` must not produce a pointless copy, since a copy of a consumer's
    Tool is a different object identity in their own list."""
    dspy = pytest.importorskip("dspy")

    def call(q: str) -> str:
        return q

    tool = dspy.Tool(call, name="clean", desc="Already one clean line.")
    assert with_model_facing_description(tool) is tool


@pytest.mark.parametrize("raw,expected", [
    # A construct the AUTHOR wrapped across two source lines. Both patterns exclude newlines, so
    # these only match because whitespace is collapsed FIRST. Running the substitutions first left
    # the double backticks in the prompt, which no shipped docstring happens to trigger and a
    # consumer's very well might.
    ("a ``very long\n    literal name`` here.", "a `very long literal name` here."),
    ("See :func:`~rlm_harness.tools.fs.\n    make_read_file_tool` now.",
     "See make_read_file_tool now."),
    ("``x``", "`x`"),
    (":func:`~a.b.c`", "c"),
    (":mod:`a.b`", "a.b"),
    ("line one\n        line two", "line one line two"),
    ("", ""),
    ("   ", ""),
    (None, ""),
    (42, ""),
])
def test_the_transform_table(raw, expected):
    assert model_facing_description(raw) == expected


def test_the_indentation_python_put_there_does_not_reach_the_prompt():
    """The defect that started this, as a unit. A docstring's continuation lines carry the source
    indentation; dspy turns each newline into two spaces, so the model read a 10-space gap."""
    raw = """First line.

        Indented continuation."""
    assert "\n        " in raw, "the fixture must carry real source indentation"
    assert WHITESPACE_RUN.search(raw.replace("\n", DSPY_FLATTEN)), "dspy would flatten it badly"
    out = model_facing_description(raw)
    assert out == "First line. Indented continuation."
    assert not WHITESPACE_RUN.search(out)


# --- the baseline ----------------------------------------------------------------------------
#
# Markup and emptiness are shape defects and the tests above catch them. SIZE is the one that came
# back: `verify_quote` shipped 6936 chars of design rationale and `grep_files` 3816 of
# implementation notes: 52.2% and 28.7% of the then-fourteen shipped tools' 13280 characters of
# model-facing text, 81% between them, and no assertion anywhere would have noticed.
#
# Deliberately NOT a hardcoded cap, for two reasons.
#
# An AGGREGATE cap measures a task that never ships: the table above builds every catalogue tool at
# once, where a real task hands the model two to four. A number over the sum of all of them is a
# budget for a fiction.
#
# A PER-TOOL cap is a number someone has to guess for every tool that does not exist yet. A tool
# may genuinely need a long description to be usable, and the only way to argue with a hardcoded
# cap is to edit it upward, which is the failure docs/INVARIANTS.md warns about.
#
# So the gate is a BASELINE, checked in beside this file: whatever the repo ships today, per tool.
# Growth past the tolerance fails and says so by name. Making a description longer on purpose is
# allowed, and costs one line in a reviewed diff. An env var regenerates the file; it does NOT
# relax the rule, because a threshold that can be exported away is silence, not a gate.

BASELINE_PATH = pathlib.Path(__file__).with_name("tool_description_baseline.json")
UPDATE_ENV = "RLM_UPDATE_DESCRIPTION_BASELINE"

#: Growth allowance per tool: the LARGER of a flat margin and a share, so a two-line description
#: can gain a clarifying clause while a long one cannot double quietly.
GROWTH_CHARS = 200
GROWTH_SHARE = 0.25


def _current_lengths(shipped: dict) -> dict[str, int]:
    return {n: len(model_facing_description(t.__doc__)) for n, t in shipped.items()}


def _load_baseline() -> dict[str, int]:
    data = json.loads(BASELINE_PATH.read_text())
    return {k: v for k, v in data.items() if not k.startswith("_")}


def _write_baseline(current: dict[str, int]) -> None:
    payload = {
        "_README": (
            "Model-facing description length per tool, in characters, as rendered into the system "
            "message. Generated: re-run the suite with "
            f"{UPDATE_ENV}=1 after a deliberate change. See tests/test_tool_descriptions.py and "
            "docs/INVARIANTS.md, \"A tool's DOCSTRING is prompt text\"."
        ),
        **{k: current[k] for k in sorted(current)},
    }
    BASELINE_PATH.write_text(json.dumps(payload, indent=2) + "\n")


def _allowance(previous: int) -> int:
    return max(previous + GROWTH_CHARS, int(previous * (1 + GROWTH_SHARE)))


#: What the regeneration changed, so a run with the env var set still PRINTS the growth it
#: absorbed. A skip that says only "rewritten" turns the one loud gate into a silent one.
_regenerated: list[str] = []


def _regenerating() -> bool:
    """Exactly `"1"`, never truthiness. Named so it can be PINNED: inlined in the fixture, a
    revert to `os.environ.get(UPDATE_ENV)` was invisible, and `=0` would then mean on."""
    return os.environ.get(UPDATE_ENV) == "1"


def _drift_threshold(current: int) -> int:
    """How far above reality a recorded number may sit. ONE allowance, not two: named so the
    tightness is pinnable, because the drift test only fires when the baseline is already above
    reality, so the formula can be loosened without reddening anything on real data."""
    return _allowance(current)


@pytest.fixture(scope="session", autouse=True)
def _maybe_regenerate_baseline():
    """Regenerate ONCE per session, before any baseline test reads the file.

    Exact `== "1"`, not truthiness: `RLM_UPDATE_DESCRIPTION_BASELINE=0` is how a person writes
    "off", and `os.environ.get(...)` alone accepts it as "on". docs/INVARIANTS.md states the same
    rule for the kit's own config parsing.

    Session-scoped so all three baseline tests see the same file. Doing it inside the growth test
    left the other two passing only because that one happened to run first.
    """
    if not _regenerating():
        yield
        return
    with tempfile.TemporaryDirectory() as root:
        current = _current_lengths(_build_every_shipped_tool(root))
    try:
        previous = _load_baseline()
    except OSError:
        previous = {}
    for name in sorted(set(current) | set(previous)):
        was, now = previous.get(name), current.get(name)
        if was != now:
            _regenerated.append(f"{name} {was} -> {now}")
    _write_baseline(current)
    if _regenerated:
        print(f"\n{UPDATE_ENV}=1 rewrote {BASELINE_PATH.name}: " + "; ".join(_regenerated))
    yield


def _skip_if_regenerating() -> None:
    if _regenerating():
        changed = "; ".join(_regenerated) or "no change"
        pytest.skip(
            f"{UPDATE_ENV}=1: baseline rewritten, so this gate did not run. It absorbed: "
            f"{changed}. Commit {BASELINE_PATH.name} so the number is in the diff."
        )


def test_no_tool_description_grows_past_its_baseline(shipped):
    """The gate. It names the tool, both numbers, and the two ways out.

    This is the number a consumer pays on every turn of every run, because it sits in the system
    message and is re-sent with each request. It is also the number nobody could see for
    every release up to this one, which is the whole reason this file exists.
    """
    _skip_if_regenerating()
    current = _current_lengths(shipped)
    baseline = _load_baseline()
    grown = {
        name: (baseline[name], n)
        for name, n in current.items()
        if name in baseline and n > _allowance(baseline[name])
    }
    assert not grown, (
        "a tool's model-facing description grew past its baseline: "
        + "; ".join(f"{name} {was} -> {now} chars" for name, (was, now) in sorted(grown.items()))
        + ". That text is in the SYSTEM message, re-sent on every turn of every run. Either move "
        "the developer-facing half to the factory or module docstring (which no model reads), or, "
        "if the model really needs it, accept the cost on the record: re-run with "
        f"{UPDATE_ENV}=1 and commit the changed {BASELINE_PATH.name} so the number is in the diff."
    )


def test_every_shipped_tool_is_in_the_baseline(shipped):
    """A new tool must land in the baseline, which is what puts its prompt cost in front of a
    reviewer the first time. Without this, adding a tool is how the gate gets skipped."""
    current = _current_lengths(shipped)
    baseline = _load_baseline()
    missing = {n: current[n] for n in current if n not in baseline}
    stale = sorted(set(baseline) - set(current))
    assert not missing, (
        f"not in {BASELINE_PATH.name}: "
        + ", ".join(f"{n} ({c} chars)" for n, c in sorted(missing.items()))
        + f". Re-run with {UPDATE_ENV}=1 and commit the file."
    )
    assert not stale, (
        f"{BASELINE_PATH.name} still lists tools the table no longer builds: {stale}. "
        f"Re-run with {UPDATE_ENV}=1 and commit the file."
    )


def test_the_drift_threshold_is_one_allowance_not_two():
    """Pins the FORMULA, not just the arithmetic. The drift test below only fires when a recorded
    number is already above reality, so on a healthy baseline either threshold passes and the
    formula can be loosened invisibly. Two stacked allowances let a baseline sit 56% above the
    truth, and that inflated baseline then licensed growth to roughly double today's size.
    """
    real = 1328
    assert _allowance(real) == 1660, "the arithmetic"
    assert _drift_threshold(real) == _allowance(real), "the drift formula is ONE allowance"
    assert _allowance(_allowance(real)) == 2075 > _drift_threshold(real), "the looser form"


@pytest.mark.parametrize("value,on", [("1", True), ("0", False), ("false", False),
                                      ("", False), ("yes", False), (None, False)])
def test_the_regeneration_probe_is_exact_not_truthy(monkeypatch, value, on):
    """`RLM_UPDATE_DESCRIPTION_BASELINE=0` is how a person writes "off", and
    `os.environ.get(...)` alone accepts it as "on", which would silence the one loud gate. The
    kit pins the analogous `is True` probe for `records_sub_call`; this is its equivalent."""
    if value is None:
        monkeypatch.delenv(UPDATE_ENV, raising=False)
    else:
        monkeypatch.setenv(UPDATE_ENV, value)
    assert _regenerating() is on


def test_the_baseline_has_not_drifted_far_above_reality(shipped):
    """The CONTROL, and the ratchet. A baseline well above what the code ships cannot fail on a
    regression, and looks exactly like a passing gate. Shrinking a description is always allowed
    and never fails the gate above, so this is what keeps the recorded number honest afterwards.
    """
    current = _current_lengths(shipped)
    baseline = _load_baseline()
    slack = {
        name: (baseline[name], current[name])
        for name in current
        if name in baseline and baseline[name] > _drift_threshold(current[name])
    }
    assert not slack, (
        "the baseline sits far above what these tools now ship, so it can no longer catch a "
        "regression: "
        + "; ".join(f"{n} recorded {was}, actually {now}" for n, (was, now) in sorted(slack.items()))
        + f". Good news, but re-run with {UPDATE_ENV}=1 and commit it so the gate tightens."
    )


# --- the producers the table cannot reach -------------------------------------------------------
#
# `_build_every_shipped_tool` covers `rlm_harness.tools.__all__`'s factories. THREE more shipped
# public entry points hand a model a tool without going through one, so the rules above did not
# reach them: progressive-disclosure skills (`list_skills` / `read_skill`) and the sub-LM query
# tool. Measured before this sweep existed: giving `list_skills` a markup docstring, deleting its
# docstring entirely, and making the sub-LM tool's description `None` each left the whole suite
# green, so `No description` and `` ``markup`` `` were still reachable in a prompt on the release
# that claimed to have fixed them.
#
# `tests/test_repl_safety.py::test_other_repl_tool_producers_are_safe` already sweeps exactly
# these for the NAME and SHAPE rules. This is the description half of the same sweep.

def _other_producers(tmp_path):
    from rlm_harness import load_skills_as_tools
    from rlm_harness.sub_lm import model_as_tool

    (tmp_path / "s.md").write_text("---\nname: s\ndescription: d\n---\nbody")
    tools = {
        "query_model[given]": model_as_tool("given", None, description="Ask a sub-model."),
        # The DEFAULT path. `query_model.__doc__ = description or (fallback)`, so a caller that
        # passes nothing relies on the fallback: testing only the `description=` case leaves the
        # fallback unmeasured, and making it `description or None` kept the suite green.
        # Distinct model NAMES, because the tool is named `query_<name>` and dspy refuses a
        # duplicate across one task's tools.
        "query_model[default]": model_as_tool("fallback", None),
    }
    for discovery in ("list", "inject"):
        for tool in load_skills_as_tools(tmp_path, discovery=discovery):
            tools[f"{tool.__name__}[{discovery}]"] = tool
    return tools


def test_the_other_producers_also_describe_their_tools(tmp_path):
    """Non-empty, because the normaliser CANNOT fix emptiness: it returns `""` and the caller
    leaves the tool alone, so dspy renders the literal words `No description`."""
    missing = [n for n, t in _other_producers(tmp_path).items() if not (t.__doc__ or "").strip()]
    assert not missing, f"handed to a model with no description: {missing}"


def test_the_seam_reaches_the_other_producers_too(tmp_path):
    """END TO END, and the only falsifiable thing to assert about these three.

    "No markup in the rendered text" is a property of the normaliser, not of how these docstrings
    are written: putting `` ``markup`` `` into one of them leaves such a test green (correctly, the
    seam cleans it) and disabling the transform leaves it green too (they carry no markup to
    begin with). So a markup assertion here measures nothing. What matters is whether the seam
    REACHES them at all, since none comes from a `rlm_harness.tools` factory: that is what this
    builds a real task to check.
    """
    pytest.importorskip("dspy")
    import rlm_harness
    from rlm_harness.testing import ScriptedInterpreter

    # One tool per REPL NAME: `_other_producers` builds both skills discovery modes, which yield
    # a `read_skill` each, and dspy refuses a duplicate name across one task's tools.
    by_name = {t.__name__: t for t in _other_producers(tmp_path).values()}
    tools = list(by_name.values())
    # Markup planted on purpose, so the assertion has something to clean.
    tools[0].__doc__ = "Ask a ``sub-model``.\n\n        See :func:`~rlm_harness.sub_lm.model_as_tool`."

    class _T(rlm_harness.RLMTask):
        signature = "q: str -> answer: str"
        output_field = "answer"

    rlm_harness.configure(rlm_harness.RLMConfig(
        main_model="openai/x", sub_model="openai/x", api_key="k",
        interpreter="mock", max_retries=1,
    ))
    task = _T(interpreter=ScriptedInterpreter([]), tools=tools)
    instructions = task._build_rlm().generate_action.signature.instructions

    i = instructions.find("\nAdditional tools available")
    assert i > 0, "none of the three producers' tools reached the instructions"
    block = instructions[i:]
    for name in ("query_given", "query_fallback", "list_skills", "read_skill"):
        assert name in block, f"{name} did not reach the prompt at all"
    assert not RST.search(block), f"markup survived the seam: {RST.findall(block)[:3]}"
    assert not WHITESPACE_RUN.search(block), (
        f"whitespace runs survived the seam: {WHITESPACE_RUN.findall(block)[:3]}"
    )


# --- the one description built from an argument --------------------------------------------------

def test_web_searchs_stated_cap_is_the_cap_the_build_enforces():
    """`make_web_search_tool` was the first tool whose description carries a NUMBER, substituted
    from `max_results` at build time; the memory tools below are the others. Nothing asserted that the substitution ran:
    deleting the `.replace(...)` line left `{cap}` in the prompt, and hardcoding a `5` left the
    docstring lying about any other cap, and the whole suite stayed green under both.
    """
    for cap in (1, 3, 5, 25):
        tool = T.make_web_search_tool(lambda q: [], max_results=cap)
        rendered = model_facing_description(tool.__doc__)
        assert "{cap}" not in rendered, "the substitution did not run: `{cap}` reached the prompt"
        assert f"up to {cap} results" in rendered, (
            f"max_results={cap} but the description says: {rendered[:120]}"
        )


def test_memory_searchs_stated_cap_is_the_cap_the_build_enforces(tmp_path):
    """Same rule as `web_search`'s: the number is built from `max_results`, and a model that asks
    for more than it receives was told the truth beforehand."""
    store = MemoryStore(tmp_path / "m.db")
    for n in range(30):
        store.add(f"memory item {n}", provenance="observed", origin="o")
    for cap in (1, 3, 8, 25):
        tool = T.make_memory_search_tool(store, max_results=cap)
        rendered = model_facing_description(tool.__doc__)
        assert "{cap}" not in rendered, "the substitution did not run: `{cap}` reached the prompt"
        assert f"up to {cap} items" in rendered, rendered[:160]
        assert len(json.loads(tool("memory", limit=100))) == cap


def test_the_memory_tools_state_the_text_width_they_cut_at(tmp_path):
    """`memory_search` and `memory_get` both cut each text at `max_chars`; the number a model
    reads is the one the build cuts at."""
    store = MemoryStore(tmp_path / "m.db")
    item = store.add("z" * 100, provenance="observed", origin="o")
    for width in (7, 30, 4000):
        search = T.make_memory_search_tool(store, max_chars=width)
        get = T.make_memory_get_tool(store, max_chars=width)
        for tool in (search, get):
            rendered = model_facing_description(tool.__doc__)
            assert "{width}" not in rendered, "the substitution did not run"
            assert f"cut at {width} characters" in rendered, rendered[:200]
        assert json.loads(search(""))[0]["text"].startswith("z" * min(width, 100))
        assert json.loads(get(item.id))["text"].count("z") == min(width, 100)


def test_memory_notes_stated_cap_is_the_cap_the_build_enforces(tmp_path):
    """The note's length limit is stated in its description and enforced by its gate, and both
    come from `max_chars`."""
    store = MemoryStore(tmp_path / "m.db")
    for cap in (5, 40, 2000):
        tool = T.make_memory_note_tool(store, max_chars=cap)
        rendered = model_facing_description(tool.__doc__)
        assert "{cap}" not in rendered
        assert f"at most {cap} characters" in rendered, rendered[:160]
        assert tool("x" * cap).startswith("Noted: ")
        assert tool("x" * (cap + 1)) == f"Refused: text exceeds {cap} characters."


def test_the_stated_cap_never_promises_fewer_results_than_the_build_returns():
    """`normalise_search_results` appends before checking the cap, so `max_results=0` returns one
    result. The description must not state a number the build will exceed: it is an instruction,
    and a model told "up to 0" that receives one has been misinformed by us, not by the provider.
    """
    for cap in (0, -1):
        tool = T.make_web_search_tool(lambda q: [{"url": "https://example.com"}], max_results=cap)
        stated = model_facing_description(tool.__doc__)
        assert f"up to {max(1, cap)} results" in stated, (
            f"max_results={cap} renders: {stated[:120]}"
        )
        # The falsifiable half is the sentence above; dropping the clamp reds it. The line below
        # is a check on the PREMISE, not a second assertion: the searcher returns one result, so
        # `n <= max(1, cap)` cannot fail for any cap, and saying so beats letting it read like the
        # point of the test.
        returned = tool("q")
        n = len(returned) if isinstance(returned, list) else 0
        assert n == 1, f"the premise moved: max_results={cap} returned {n}, not 1"


# --- the description against the BEHAVIOUR ------------------------------------------------------
#
# The defect this closes: a description is prompt text, so a sentinel string quoted in one is an
# instruction to branch on it, and nothing joined the two halves. The repo already asserts the real
# strings (`tests/test_grounding.py`, `tests/test_tools.py`) and separately quotes them as prompt
# text, and every reverted sentence below left the whole suite green: `verify_quote`'s coordinate
# MATCH shape, its punctuation refusal, and `grep_files`' `Invalid regex` prefix.
#
# Each row carries the prefix the call really returns AND the stable fragment the description has
# to contain. Both are written out rather than derived, because a description legitimately states
# a generic form (`line N`, not `line 2`), and a test that GUESSES which part is stable is a test
# that fails for the wrong reason. Breaking the code reds it; breaking the description reds it too.

def _sentinel_rows(root: str):
    os.makedirs(os.path.join(root, "d"), exist_ok=True)
    with open(os.path.join(root, "a.txt"), "w") as fh:
        fh.write("alpha beta\nseven\n")
    read_file = T.make_read_file_tool(root)
    write_file = T.make_write_file_tool(root)
    extract = T.make_extract_archive_tool(root)
    grep = T.make_grep_files_tool(root, ["a.txt"])
    search = T.make_web_search_tool(lambda q: [])

    def _boom(q):
        raise RuntimeError("provider down")

    failing_search = T.make_web_search_tool(_boom)
    validator = T.make_schema_validator(_Out)

    def _boom_cmd(*a, **k):
        raise RuntimeError("down")

    failing_command = T.make_command_tool(_boom_cmd)
    guarded_command = T.make_command_tool(_boom_cmd, guard=lambda c: "not on this host")
    src = "alpha beta\ngamma delta\n"
    empty_store = MemoryStore(os.path.join(root, "empty.db"))
    store = MemoryStore(os.path.join(root, "mem.db"))
    known = store.add("sleep and memory", provenance="observed", origin="o")
    memory_empty = T.make_memory_search_tool(empty_store)
    retired_store = MemoryStore(os.path.join(root, "retired.db"))
    retired_store.retire(retired_store.add("gone", provenance="said", origin="o").id)
    retired_search = T.make_memory_search_tool(retired_store)
    memory_search = T.make_memory_search_tool(store)
    memory_get = T.make_memory_get_tool(store)
    memory_note = T.make_memory_note_tool(store, max_chars=10)

    class _Locked(MemoryStore):
        def _fail(self, *a, **k):
            raise sqlite3.OperationalError("database is locked")

        search = get = add = revise = count = _fail

    locked = _Locked(os.path.join(root, "locked.db"))
    locked_search = T.make_memory_search_tool(locked)
    locked_get = T.make_memory_get_tool(locked)
    locked_note = T.make_memory_note_tool(locked)
    return [
        ("memory_search reports a store failure", locked_search, lambda: locked_search("q"),
         "Memory error: OperationalError", "Memory error"),
        ("memory_get reports a store failure", locked_get, lambda: locked_get("m1"),
         "Memory error: OperationalError", "Memory error"),
        ("memory_note reports a store failure", locked_note, lambda: locked_note("x"),
         "Memory error: OperationalError", "Memory error"),
        ("memory_search reports an empty store", memory_empty, lambda: memory_empty(""),
         "Memory is empty.", "Memory is empty."),
        ("memory_search reports no match", memory_search, lambda: memory_search("zzzz"),
         "No memories match 'zzzz'", "No memories match"),
        ("memory_search reports an all-retired store", retired_search, lambda: retired_search(""),
         "No current memories.", "No current memories."),
        ("memory_get reports an unknown id", memory_get, lambda: memory_get("m999"),
         "Not found: 'm999'", "Not found"),
        ("memory_note saves", memory_note, lambda: memory_note("a note"), "Noted: m", "Noted"),
        ("memory_note refuses an empty text", memory_note, lambda: memory_note("  "),
         "Refused: text must be non-empty", "Refused"),
        ("memory_note refuses an over-long text", memory_note, lambda: memory_note("x" * 11),
         "Refused: text exceeds 10 characters", "Refused"),
        ("memory_note refuses an unknown id", memory_note,
         lambda: memory_note("fix", revises="m999"), "Refused: no memory with id 'm999'", "Refused"),
        ("memory_note revises a known id", memory_note,
         lambda: memory_note("fix", revises=known.id), "Noted: m", "Noted"),
        ("read_file refuses an escaping path", read_file, lambda: read_file("../x"),
         "Refused:", "Refused:"),
        ("web_search refuses an empty query", search, lambda: search("  "),
         "Refused:", "Refused:"),
        # The second of web_search's two sentinels. Covering only the first is how
        # `"Search error: ..."` sat in the description while the code returned
        # `"Search error for 'q': ..."`.
        ("web_search reports a provider failure", failing_search,
         lambda: failing_search("q"), "Search error for 'q':", "Search error for"),
        ("grep_files reports no match", grep, lambda: grep("zzzz"),
         "No matches for", "No matches for"),
        # Promoted out of _NO_SENTINEL once measured: both return strings with prefixes, and
        # `schema_validator`'s description told the model to call it without saying how to read
        # the answer.
        ("schema_validator accepts", validator, lambda: validator('{"x": 1}'),
         "Validation successful.", "Validation successful."),
        ("schema_validator rejects", validator, lambda: validator('{"x": "no"}'),
         "Validation failed:", "Validation failed:"),
        ("run_command reports a runner failure", failing_command,
         lambda: failing_command("ls"), "Command error for 'ls':", "Command error for"),
        # Its other sentinel, which the per-sentinel coverage test demanded on its first run:
        # the description names `Refused:` and nothing exercised the guard that produces it.
        ("run_command refuses via its guard", guarded_command,
         lambda: guarded_command("ls"), "Refused:", "Refused:"),
        ("write_file refuses an escaping path", write_file, lambda: write_file("../x", "y"),
         "Refused:", "Refused"),
        ("extract_archive refuses an escaping path", extract, lambda: extract("../a.zip", "out"),
         "Refused:", "Refused"),
        ("grep_files rejects a mode", grep, lambda: grep("a", output_mode="nope"),
         "Invalid output_mode", "Invalid output_mode"),
        ("grep_files rejects a pattern", grep, lambda: grep("[("),
         "Invalid regex '[('", "Invalid regex"),
        ("verify_quote matches as text", T.verify_quote,
         lambda: T.verify_quote(src, "gamma delta"), "MATCH: found at line", "MATCH: found at line"),
        ("verify_quote matches by coordinate", T.verify_quote,
         lambda: T.verify_quote(src, "     2\tgamma delta"),
         "MATCH: line 2 verified by line number", "verified by line number"),
        ("verify_quote refuses an empty quote", T.verify_quote,
         lambda: T.verify_quote(src, "   "),
         "MISMATCH: quote must be non-empty", "quote must be non-empty"),
        ("verify_quote refuses a digits-only quote", T.verify_quote,
         lambda: T.verify_quote(src, " 7 "),
         "MISMATCH: quote carries only digits and whitespace",
         "quote carries only digits and whitespace"),
        ("verify_quote reports no match", T.verify_quote,
         lambda: T.verify_quote(src, "zeta"), "MISMATCH: quote not found", "MISMATCH"),
    ]


def test_every_sentinel_a_description_quotes_is_one_the_code_returns(tmp_path):
    """The code half: the prefix really comes back."""
    wrong = []
    for label, _tool, call, prefix, _stem in _sentinel_rows(str(tmp_path)):
        got = call()
        if not (isinstance(got, str) and got.startswith(prefix)):
            wrong.append(f"{label}: expected {prefix!r}, got {str(got)[:70]!r}")
    assert not wrong, "; ".join(wrong)


def test_every_sentinel_in_a_row_is_one_its_description_quotes(tmp_path):
    """The prompt half of a ROW, and the one that was missing. A model is told to branch on these
    strings, so the description has to carry them, in a form it can match on.

    Named for its domain after an earlier name claimed the loop was driven by the code. It is
    driven by `_sentinel_rows`. The loop has three sides and this file asserts two of them:
    quoted-in-a-description to returned-by-the-code, and in-a-row to named-in-the-description.
    The third, returned-by-the-code to quoted-in-a-description, is NOT asserted, and it is the
    one that would surface a tool whose code has a sentinel its description never mentions.
    `test_an_exemption_records_the_sentinel_it_is_blind_to` is as close as this file gets.

    Each row names the fragment to look for, because a description legitimately generalises the
    variable part. What this catches is a description naming a DIFFERENT sentinel than the one the
    code emits, which is the defect that shipped.
    """
    missing = []
    for label, tool, _call, _prefix, stem in _sentinel_rows(str(tmp_path)):
        desc = model_facing_description(tool.__doc__)
        if stem not in desc:
            missing.append(f"{label}: description never mentions {stem!r}")
    assert not missing, "; ".join(missing)


def test_the_default_verify_quote_states_is_the_constant_it_uses():
    """`snippet_chars` is the one number in that description, and the release's own rule is never
    to write a number a constant can change. It is substituted at import from
    `_DEFAULT_SNIPPET_CHARS`.

    What this catches, stated narrowly because the wider claim was false: a substitution that
    stopped running, and a constant that moved while the substitution still ran. What it does NOT
    catch is someone hardcoding the constant's CURRENT value into the docstring and deleting the
    substitution, which leaves both sides agreeing today and drifting later. Closing that needs a
    reload with the constant patched; `make_web_search_tool`'s equivalent has no such hole only
    because a factory argument can be varied in a loop and a module constant cannot.
    """
    from rlm_harness.tools import grounding

    desc = model_facing_description(T.verify_quote.__doc__)
    assert "{snippet_default}" not in desc, "the substitution did not run"
    assert f"default {grounding._DEFAULT_SNIPPET_CHARS}" in desc, desc[-160:]


def test_the_two_match_shapes_are_described_as_they_behave(tmp_path):
    """The specific fact a shortening dropped, pinned.

    A sentinel check is not enough on its own: reverting the sentence about the coordinate match's
    SHAPE back to the false version left the sentinel assertions green, because the sentinel was
    still named. What was wrong was the claim about what the message carries.
    """
    src = "alpha beta\ngamma delta\n"
    literal = T.verify_quote(src, "gamma delta")
    coord = T.verify_quote(src, "     2\tgamma delta")
    desc = model_facing_description(T.verify_quote.__doc__)

    assert "(char " in literal and "Context:" in literal, literal
    assert "(char " not in coord and "Context:" not in coord, coord

    # The description has to say that the second one carries neither.
    assert "no character offset and no snippet" in desc, (
        "the coordinate match returns no offset and no snippet, and the description no longer "
        f"says so: {desc[:200]}"
    )


#: Each row is (the example a description quotes verbatim, a call that must return it as a prefix).
#: Parametrized because the single-tool version did not grow when a second example arrived:
#: `web_search`'s `Search error for 'q': TimeoutError: ...` went unasserted for a round.
def _quoted_examples(root: str):
    import os

    with open(os.path.join(root, "a.txt"), "w") as fh:
        fh.write("x\n")
    grep = T.make_grep_files_tool(root, ["a.txt"])

    def _slow(q):
        raise TimeoutError("slow")

    search = T.make_web_search_tool(_slow)
    return [
        ("grep_files", grep,
         "Invalid regex '[(': unterminated character set at position 2",
         lambda: grep("[(")),
        ("web_search", search,
         "Search error for 'q': TimeoutError: ...",
         lambda: search("q")),
    ]


@pytest.mark.parametrize("which", [0, 1])
def test_the_examples_a_description_quotes_are_prefixes_of_the_real_return(tmp_path, which):
    """A quoted EXAMPLE is a stronger claim than a quoted prefix, and the one `grep_files` got
    wrong. Asserted against the real call, so neither side can drift alone.

    The example is written out here, then asserted to be IN the description and a PREFIX of the
    real return. (An earlier comment claimed the string was pulled out of the description, which
    it was not: a comment claiming a property the code lacks, inside the test added to close that
    class of defect.)
    """
    name, tool, quoted, call = _quoted_examples(str(tmp_path))[which]
    desc = model_facing_description(tool.__doc__)
    assert quoted in desc, f"{name}'s description no longer quotes the example: {desc[:200]}"
    # An example may elide its tail with `...`, which is the honest way to show a shape whose end
    # is a message from somewhere else. Everything BEFORE the ellipsis is a literal claim.
    stable = quoted.split("...")[0]
    assert stable.strip(), f"{name}'s quoted example is all ellipsis: {quoted!r}"
    real = call()
    assert real.startswith(stable), f"{name} returns {real!r}, the description quotes {stable!r}"


# --- the sentinel sweep's own coverage ----------------------------------------------------------
#
# `_sentinel_rows` is a hand-built table, and a hand-built table drifts from the catalogue in
# silence. It did: `web_search`'s description named a `"Search error: ..."` prefix while the code
# returned `"Search error for 'q': ..."`, the same defect as `grep_files`' `Invalid regex`, in a
# docstring this release added, with four tests named "every sentinel" around it. The table
# covered four of the fourteen tools and exercised only one of web_search's two sentinels.
#
# `tests/test_repl_safety.py` already solved this for the NAME rules, with an exemption set that
# argues each entry. Same shape here.

#: Tools whose description quotes no sentinel string, each with the reason. A tool that returns a
#: STRUCT rather than a string has no prefix for a model to branch on.
_NO_SENTINEL = {
    # Return a `ModelToolResult` / `HarnessToolResult`: outcome is read off `ok`, `validated`,
    # `errors`, `endpoint_error` and `circuit_broken`, never parsed out of a prefix.
    "model": "returns a ModelToolResult, not a string",
    "harness": "returns a HarnessToolResult, not a string",
    # Returns a list of violation strings, host-side, with no prefix to branch on.
    "json_schema_validator": "returns a list of violations",
    # Measured per tool, because the reason is what a contributor adding the next entry will
    # reason from. `write_file` and `extract_archive` used to sit here for naming "Refused"
    # without a colon; once the enumerator could SEE that form they left, which is the right
    # direction: an exemption that exists because the detector is blind is not an exemption.
    # These three name no sentinel, and each one's CODE returns one anyway. The reason records
    # what is missing rather than asserting there is nothing to see, because "the rows cover the
    # rest" was the sentence a reader would have relied on and it was false: these are exempted,
    # not covered. `test_an_exemption_records_the_sentinel_it_is_blind_to` keeps each claim true.
    "edit_file": "quotes no prefix; returns Refused: ...",
    "git_clone": "quotes no prefix; returns Refused: ...",
    "fetch_url": "quotes no prefix; returns Fetch error for ...",

}


#: Shapes that match SENTINEL_SHAPE but are labels inside a message rather than something a model
#: branches on, so they need no row of their own.
_NOT_A_BRANCH = {"Context"}


def test_the_sentinel_table_covers_every_tool_that_quotes_one(tmp_path, shipped):
    """Tool-level coverage: nothing in the catalogue is silently outside the sweep."""
    covered = {label.split()[0] for label, *_ in _sentinel_rows(str(tmp_path))}
    uncovered = sorted(set(shipped) - covered - set(_NO_SENTINEL))
    assert not uncovered, (
        f"in the catalogue but not in _sentinel_rows: {uncovered}. Add a row for each sentinel its "
        "description quotes, or list it in _NO_SENTINEL with the reason it quotes none."
    )
    stale = sorted(set(_NO_SENTINEL) - set(shipped))
    assert not stale, f"_NO_SENTINEL lists tools the catalogue no longer builds: {stale}"


def test_every_sentinel_in_a_description_has_a_row_not_just_its_tool(tmp_path, shipped):
    """Per-SENTINEL coverage, which is what the claim in `docs/INVARIANTS.md` and
    `rlm_harness/README.md` actually says.

    Tool-level coverage is not enough, and the gap was not hypothetical: `web_search` named two
    sentinels, one row covered it, and the uncovered one was wrong in the prompt for a whole
    release. The sentinels are ENUMERATED from each rendered description by `SENTINEL_SHAPE`
    rather than listed here, so a new one arrives already needing a row.
    """
    rows = _sentinel_rows(str(tmp_path))
    stems_by_tool: dict[str, list[str]] = {}
    for label, _tool, _call, _prefix, stem in rows:
        stems_by_tool.setdefault(label.split()[0], []).append(stem)

    unrowed = {}
    for name, tool in sorted(shipped.items()):
        if name in _NO_SENTINEL:
            continue
        found = quoted_sentinels(model_facing_description(tool.__doc__)) - _NOT_A_BRANCH
        stems = stems_by_tool.get(name, [])
        for sentinel in sorted(found):
            if not any(st in sentinel or sentinel in st for st in stems):
                unrowed.setdefault(name, []).append(sentinel)
    assert not unrowed, (
        f"quoted in a description with no row to check it against a real call: {unrowed}. "
        "Add a row, or add the shape to _NOT_A_BRANCH if it is a label rather than something to "
        "branch on."
    )


#: What a sentinel LOOKS like, rather than a list of the ones that exist. A list of literals was
#: the first version and it was an exemption list wearing a different hat. A colon-anchored regex
#: was the second, and it was blind to the form this kit actually uses MOST: when the variable
#: part comes BEFORE the colon (`Invalid regex '[(': ...`, `No matches for 'z' (glob '*')`), the
#: only honest way to quote the stable head is to stop before the variable part, which drops the
#: colon. That form is four of the catalogue's quoted sentinels and the regex enumerated none of
#: them.
#:
#: So: find each quoted span in the RENDERED description, and treat it as a sentinel if it opens
#: with a capital and carries a variable marker. The STEM is the part before that marker, which
#: is what a row's `stem` has to match.
_QUOTED = re.compile(r'`([^`\n]{3,120}?)`|"([^"\n]{3,120}?)"')
_VARIABLE = ("...", "'", "<", ":", " - ")   # " - " spaced: a dash doing the colon's job


def quoted_sentinels(description: str) -> set[str]:
    """The sentinel STEMS a description quotes, enumerated rather than listed."""
    found = set()
    for a, b in _QUOTED.findall(description):
        span = (a or b).strip()
        if not (span[:1].isupper() or span[:1].isdigit()):
            continue
        cuts = [span.find(mark) for mark in _VARIABLE if span.find(mark) > 0]
        if not cuts:
            # No variable part at all. Two sub-forms, and the repo uses both:
            #   `"Validation successful."` - a complete sentence, period included
            #   `"Refused"/error string`   - a bare word, which is how `write_file` and
            #                               `extract_archive` quote theirs
            # Requiring the period was the third version of this rule and it had a property the
            # others do not: for a bare word the gate RELEASED ITSELF, because deleting the
            # period stopped the span being a sentinel, so nothing demanded a row any more. One
            # character of prose plus one deleted row and a real sentinel left the sweep.
            # Measured across the catalogue, accepting a short capitalised span adds exactly the
            # two `Refused`s above and no false positive.
            if len(span.split()) <= 3 or (span.endswith(".") and len(span.split()) <= 8):
                found.add(span.rstrip("."))
            continue
        stem = span[: min(cuts)].strip().rstrip(":")
        if stem and (stem[0].isupper() or stem[0].isdigit()) and len(stem.split()) <= 14:
            found.add(stem)
    return found


def test_no_exempted_tool_quietly_grew_a_sentinel(tmp_path, shipped):
    """The exemption set's own control. A tool exempted because it returns a struct, or because its
    description quotes nothing, stops being exempt the moment someone writes a prefix into it.

    Planting a sentinel into one of these descriptions trips it, in every form the repo writes
    one in; `test_the_sentinel_detector_is_a_shape_not_a_list` is the per-form control.
    """
    quoting = {}
    for name in sorted(_NO_SENTINEL):
        found = quoted_sentinels(model_facing_description(shipped[name].__doc__))
        if found:
            quoting[name] = sorted(set(found))
    assert not quoting, (
        f"exempted in _NO_SENTINEL but now quoting a sentinel: {quoting}. Add it to "
        "_sentinel_rows instead, so the string is checked against a real call."
    )


@pytest.mark.parametrize("planted", [
    "Refused: the path escapes the root",      # colon-terminated
    "Clone failed for 'url': exit 1",          # a quoted variable, then a colon
    "Clone failed for ...",                    # the SAME sentinel in the kit's house style
    "No matches for ...",                      # the form four catalogue sentinels use
    "Invalid output_mode ...",                 # with an underscore in the stem
    "Timeout: the server took too long",       # a wording nobody has used yet
    "Validation successful.",                  # a success sentinel, no variable part at all
])
def test_the_sentinel_detector_is_a_shape_not_a_list(planted):
    """The control's own control, with a row for each FORM a sentinel is written in.

    Two earlier versions each missed a form. Seven literal patterns missed everything not on the
    list, including `` ``Refused: ...`` `` in the kit's own markup. A colon-anchored regex then
    missed every sentinel whose variable part comes BEFORE the colon, which is how this repo
    writes four of its own: `No matches for ...`, `Invalid regex ...`, `Invalid output_mode ...`,
    `Clone failed for ...`. Each line above is one of those forms.
    """
    for quoted in (f"``{planted}``", f'"{planted}"', f"`{planted}`"):
        rendered = model_facing_description(f"Does a thing. Returns {quoted} on failure.")
        assert quoted_sentinels(rendered), f"not detected: {rendered}"


# --- a clamp must not rewrite the action ---------------------------------------------------------

@pytest.mark.parametrize("kwargs,raw", [
    ({"glob": None}, None),
    ({"glob": ""}, ""),
    ({"glob": "*.txt"}, "*.txt"),
])
def test_normalising_a_model_argument_does_not_rewrite_what_the_trace_records(tmp_path, kwargs, raw):
    """`grep_files` normalises two model-controlled arguments so its description's promises hold.
    The RAW argument is what the trace records.

    This was a regression once, introduced by the promise-keeping fix itself: `glob = glob or "*"`
    ran before `record_tool_call`, so a turn in which the model wrote `glob=None` was recorded as
    `glob="*"`. docs/INVARIANTS.md makes the JSONL trace the source of truth for replay and RL
    datasets, where a `tool_call`'s `args` IS the action, so that row taught a dataset an action
    the model never took. Before the fix the call raised and no success row was written at all,
    which makes the regression a loud failure traded for a quiet falsehood.
    """
    from rlm_harness import TraceRecorder, load_events

    (tmp_path / "a.txt").write_text("hello\n")
    grep = T.make_grep_files_tool(str(tmp_path), ["a.txt"])
    trace = tmp_path / "t.jsonl"
    with TraceRecorder(str(trace), run_id="r"):
        out = grep("hello", **kwargs)
    assert out == "a.txt:1: hello", out          # the clamp did its job
    calls = [e for e in load_events(str(trace)) if e["type"] == "tool_call"]
    assert calls, "the call was not recorded at all"
    assert calls[-1]["payload"]["args"]["glob"] == raw, (
        f"the model passed glob={raw!r}; the trace records "
        f"{calls[-1]['payload']['args']['glob']!r}"
    )


@pytest.mark.parametrize("kwargs", [
    {"glob": None}, {"glob": ""}, {"max_results": 0}, {"max_results": -1},
    {"max_results": None}, {"max_results": "5"}, {"max_results": 2.5},
])
def test_grep_files_never_raises_on_a_model_controlled_argument(tmp_path, kwargs):
    """Its description promises "Returns a string and never raises", and both arguments are the
    model's to choose. Each shape here raised at some point: `glob=None` and `max_results=None`
    directly, and `max_results="5"` through the clamp that fixed the first one."""
    (tmp_path / "a.txt").write_text("hello\n")
    grep = T.make_grep_files_tool(str(tmp_path), ["a.txt"])
    assert isinstance(grep("hello", **kwargs), str)


# --- the limit of this approach, stated rather than implied -------------------------------------
#
# `quoted_sentinels` reads prose, and prose is not a machine-readable source. Three versions of
# this rule each missed the form that was not yet in view when it was written:
#
#   1. a list of seven literal patterns   missed everything off the list, including the kit's own
#                                         `` ``literal`` `` markup
#   2. a colon-anchored regex             missed every sentinel whose variable part comes BEFORE
#                                         the colon, which is how four catalogue sentinels are
#                                         written
#   3. a marker-or-trailing-period rule   missed a bare capitalised word, which is how two more
#                                         are written, and for that form the gate released itself:
#                                         deleting the period stopped the span being a sentinel,
#                                         so nothing demanded a row any more
#
# Each version was derived from the forms then visible in the repo, which is the ceiling of the
# approach rather than a bug in any one version. **A sentinel written in a form this function does
# not recognise is invisible to the enumeration**, and `docs/INVARIANTS.md` and
# `rlm_harness/README.md` should be read with that in mind: the per-sentinel claim covers the
# forms below, which are the forms the kit uses, and the rows cover the rest by hand.
#
# The way out is to stop reading prose: have each factory DECLARE its sentinels as data and build
# the description from them, so the two cannot disagree. That is a design change rather than a
# test, and it is not in this release.

@pytest.mark.parametrize("form,stem", [
    ("Refused: the path escapes", "Refused"),               # colon
    ("No matches for ...", "No matches for"),               # elided tail
    ("Invalid output_mode ...", "Invalid output_mode"),      # underscore in the stem
    ("Search error for 'q': Timeout", "Search error for"),   # quoted variable, then a colon
    ("Validation successful.", "Validation successful"),     # complete sentence
    ("Refused", "Refused"),                                  # bare capitalised word
    ("Refused - the path escapes", "Refused"),               # a dash doing the colon's job
    ("404: not found", "404"),                               # digit-initial
])
def test_every_form_the_kit_writes_a_sentinel_in_is_enumerated(form, stem):
    """The forms this enumeration covers, as data. A new form belongs here AND in
    `quoted_sentinels`, and until it is in both it is invisible to the sweep."""
    found = quoted_sentinels(model_facing_description(f'Returns "{form}" on failure.'))
    assert stem in found, f"{form!r} enumerated as {sorted(found)}"


def _blind_exemptions(root: str):
    """Each exempted tool built so its ERROR path is reachable. The module fixture builds
    `fetch_url` with a fetcher that succeeds, so it never reaches its own sentinel: an exemption
    has to be checked on the path the exemption is about."""
    def _boom(*a, **k):
        raise RuntimeError("down")

    return {
        "edit_file": ("Refused:", lambda: T.make_edit_file_tool(root)("../esc", "a", "b")),
        "git_clone": ("Refused:", lambda: T.make_git_clone_tool(root, _boom)("u", "../esc")),
        "fetch_url": ("Fetch error for", lambda: T.make_fetch_tool(_boom)("https://example.com")),
    }


def test_an_exemption_records_the_sentinel_it_is_blind_to(tmp_path, shipped):
    """The closest this file gets to the third side of the loop: code to description.

    Three tools are exempted because their descriptions quote no prefix, and all three return
    one. That is a real gap, and it used to be hidden twice over: the exemption reason said only
    that nothing was quoted, and `docs/INVARIANTS.md` said "the rows cover the rest by hand",
    which a reader would have relied on. Each reason now names the sentinel the description omits,
    and this test keeps that naming true, so the gap is recorded rather than invisible.
    """
    wrong = []
    for name, (prefix, call) in sorted(_blind_exemptions(str(tmp_path)).items()):
        assert name in _NO_SENTINEL, f"{name} is no longer exempt; move it to _sentinel_rows"
        got = call()
        if not (isinstance(got, str) and got.startswith(prefix)):
            wrong.append(f"{name}: expected {prefix!r}, got {str(got)[:60]!r}")
        if prefix.rstrip(":") not in _NO_SENTINEL[name]:
            wrong.append(f"{name}'s exemption reason does not name {prefix!r}")
        desc = model_facing_description(shipped[name].__doc__)
        if prefix.rstrip(":") in desc:
            wrong.append(f"{name} now quotes {prefix!r}; give it a row instead of an exemption")
    assert not wrong, "; ".join(wrong)
