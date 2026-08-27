"""Which shipped tools must record `duration_s`, enforced the way REPL safety already is.

`metrics.compute_tool_waste` can only attribute wall-clock to the calls that report it, and a tool
that silently stops reporting is invisible rather than loud. Two factories shipped without it in
1.6.0's first draft — `make_git_clone_tool` (a network clone) and `model_as_tool` (an actual LM
call) — which is precisely the failure mode `tests/test_repl_safety.py`'s `_REPL_FACTORIES` table
exists for: six factories once shipped with no REPL-safety coverage because each author had to
remember. Same cure, same shape: every `make_*` is either listed as outbound-and-timed, or
exempt WITH a written reason, and a new one that is neither fails here at the moment it ships.
"""

import types

import pytest

pytest.importorskip("dspy")

import rlm_harness.tools as tools_pkg
from rlm_harness.tools import make_command_tool, make_fetch_tool, make_web_search_tool
from rlm_harness.trace import TraceRecorder


def _durations(tmp_path, call):
    """Run `call` under a real recorder; return the duration_s of each tool_call it emitted."""
    import json

    p = tmp_path / "t.jsonl"
    with TraceRecorder(str(p), run_id="r"):
        call()
    return [
        (json.loads(x)["payload"].get("tool"), json.loads(x)["payload"].get("duration_s"))
        for x in p.read_text().splitlines()
        if '"tool_call"' in x
    ]


# Factories whose cost is a WAIT on something outside this process. Each MUST record duration_s.
_OUTBOUND = {
    "make_fetch_tool": lambda: make_fetch_tool(lambda url: "body")("https://example.com/x"),
    "make_web_search_tool": lambda: make_web_search_tool(lambda q: [])("a query"),
    "make_command_tool": lambda: make_command_tool(
        lambda cmd: types.SimpleNamespace(
            exit_code=0, stdout="", stderr="", duration_ms=None
        )
    )("ls"),
    "make_git_clone_tool": None,   # built per-test: it needs a root dir. See _build_git_clone.
}


def _build_git_clone(tmp_path):
    """`make_git_clone_tool` needs a root, so it cannot be a zero-arg lambda like the others."""
    from rlm_harness.tools import make_git_clone_tool

    def _runner(argv, **kw):
        return types.SimpleNamespace(exit_code=0, stdout="", stderr="", duration_ms=None)

    tool = make_git_clone_tool(str(tmp_path), _runner)
    return lambda: tool("https://example.com/r.git", "dest")

# ...and every other `make_*`, with the reason it is NOT timed. Named one at a time rather than
# pattern-matched: the guard is only worth having if each exemption has to be argued for.
_NOT_OUTBOUND = {
    # Local filesystem work: sub-millisecond, and their refusal paths never touch anything, so a
    # duration would add noise to the wall-clock attribution rather than signal.
    "make_read_file_tool": "local fs read",
    "make_write_file_tool": "local fs write",
    "make_edit_file_tool": "local fs edit",
    "make_grep_files_tool": "local fs scan",
    "make_extract_archive_tool": "local archive extraction",
    # Host-side validators, never placed in a `tools=[...]` list and doing no I/O at all.
    "make_json_schema_validator": "host-side validator, no I/O",
    "make_schema_validator": "host-side validator, no I/O",
    # Side-effect-free BASES that deliberately record nothing at all — the consumer's own wrapper
    # owns the `record_tool_call`, and therefore owns passing `duration_s`.
    "make_model_tool": "base factory; records nothing (consumer's wrapper does)",
    "make_harness_tool": "base factory; records nothing (consumer's wrapper does)",
}


def test_every_shipped_factory_is_classified():
    """The guard that makes this table self-maintaining: a `make_*` reaching
    `rlm_harness.tools.__all__` with no entry in either table fails HERE, when it ships."""
    shipped = {n for n in tools_pkg.__all__ if n.startswith("make_")}
    unclassified = shipped - set(_OUTBOUND) - set(_NOT_OUTBOUND)
    assert not unclassified, (
        f"shipped but unclassified: {sorted(unclassified)} — add it to _OUTBOUND (and make it "
        f"record duration_s), or to _NOT_OUTBOUND with the reason it does not."
    )


@pytest.mark.parametrize("factory", sorted(_OUTBOUND))
def test_an_outbound_tool_records_its_duration(factory, tmp_path):
    build = _OUTBOUND[factory] or _build_git_clone(tmp_path)
    recorded = _durations(tmp_path, build)
    assert recorded, f"{factory} recorded no tool_call at all"
    for tool, duration in recorded:
        assert duration is not None, f"{factory} recorded {tool!r} with no duration_s"
        assert duration >= 0.0


def test_model_as_tool_records_its_duration(tmp_path):
    """The one model-backed tool the KIT records, so the kit owns its duration. It shipped
    without one; this is the pin."""
    from rlm_harness.sub_lm import model_as_tool

    tool = model_as_tool("openai/gpt-4o-mini", lambda prompt=None, **kw: ["answer"])
    recorded = _durations(tmp_path, lambda: tool("hi"))
    assert recorded and recorded[0][1] is not None
