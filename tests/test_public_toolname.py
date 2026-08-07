"""The public REPL-safety surface (1.1.0) — and the end-to-end path it exists for.

`McpCatalog` hands a consumer the server's RAW tool names and RAW schemas; the consumer
builds its own `dspy.Tool`s. Inside `mcp.py` the kit fixes both halves — the NAME (a
hyphen is the MCP norm and dspy refuses it) and the SIGNATURE (a `**kwargs` wrapper
registers one proxy param literally called `kwargs`). Both were private, so that consumer
had no sanctioned remedy: CLAUDE.md's "consumers EXTEND, they don't fork" invariant bars
reaching into a `_`-private name, and 1.0.2's `assert_repl_safe` detects both problems
while offering no fix.

These tests pin that the promoted functions actually close the path — one half alone
leaves a well-named tool that `assert_repl_safe` still rejects.
"""

from __future__ import annotations

import inspect

import pytest

pytest.importorskip("dspy")

import dspy

from rlm_harness import (
    is_valid_tool_name,
    sanitize_tool_name,
    signature_from_json_schema,
    unique_tool_names,
)
from rlm_harness.testing import assert_repl_safe


def test_the_public_names_are_exported_and_reachable():
    import rlm_harness

    for name in ("is_valid_tool_name", "sanitize_tool_name",
                 "unique_tool_names", "signature_from_json_schema"):
        assert name in rlm_harness.__all__
        assert callable(getattr(rlm_harness, name))


def test_mcpcatalog_consumer_path_end_to_end():
    """The whole reason these are public. A consumer holding raw MCP metadata must be able
    to build a tool that passes BOTH halves of `assert_repl_safe` and that dspy accepts."""
    raw_name = "get-weather"                                    # the MCP naming norm
    raw_schema = {"type": "object",
                  "properties": {"city": {"type": "string"}, "units": {"type": "string"}},
                  "required": ["city"]}

    def call(**kwargs):
        """Fetch the weather."""
        return "sunny"

    call.__signature__ = signature_from_json_schema(raw_schema)
    repl = unique_tool_names([raw_name])[raw_name]
    tool = dspy.Tool(call, name=repl, desc="Fetch the weather.")

    assert_repl_safe(tool)                                       # both halves pass
    rlm = dspy.RLM("q: str -> a: str", tools=[tool])             # and dspy accepts it
    assert list(rlm._user_tools) == ["get_weather"]


def test_name_alone_is_not_enough():
    """Pins WHY `signature_from_json_schema` had to be promoted alongside the name rule:
    fixing only the name leaves a tool the kit's own guard rejects."""
    def call(**kwargs):
        """d"""
    tool = dspy.Tool(call, name=unique_tool_names(["get-weather"])["get-weather"])
    with pytest.raises(AssertionError, match="VAR_KEYWORD"):
        assert_repl_safe(tool)


# ---- signature_from_json_schema ------------------------------------------


def test_required_params_come_first():
    """Not cosmetic: the Deno stub emits `def f(<params>)` in this order, and a no-default
    param after a defaulted one is a SyntaxError that aborts the WHOLE registration."""
    sig = signature_from_json_schema({
        "properties": {"opt": {}, "req": {}, "opt2": {}}, "required": ["req"]})
    params = list(sig.parameters)
    assert params[0] == "req"
    assert sig.parameters["req"].default is inspect.Parameter.empty
    assert all(sig.parameters[p].default is None for p in params[1:])


@pytest.mark.parametrize("schema", [None, {}, {"type": "object"}, {"properties": {}}, "nonsense"])
def test_schemaless_yields_a_zero_arg_signature(schema):
    """Zero params, never `**kwargs` — a no-argument tool must not be left with the shape
    `assert_repl_safe` rejects."""
    assert list(signature_from_json_schema(schema).parameters) == []


def test_required_naming_an_undeclared_property_is_ignored():
    """A `required` entry with no matching property cannot become a parameter; inventing one
    would make the model pass an argument the server never declared."""
    sig = signature_from_json_schema({"properties": {"a": {}}, "required": ["a", "ghost"]})
    assert list(sig.parameters) == ["a"]


def test_non_list_required_does_not_sink_the_tool():
    sig = signature_from_json_schema({"properties": {"id": {}}, "required": "id"})
    assert sig.parameters["id"].default is None      # treated as optional, not an exception


@pytest.mark.parametrize("bad", ["from", "class", "db.query", "2fa"])
def test_unusable_property_name_raises_for_the_caller_to_handle(bad):
    """Deliberately raises rather than sanitising: the proxy forwards the param name to the
    server as a JSON key, so renaming a property would send wrong wire arguments."""
    with pytest.raises((ValueError, TypeError)):
        signature_from_json_schema({"properties": {bad: {}}, "required": [bad]})


# ---- unique_tool_names(taken=) -------------------------------------------


def test_taken_supports_progressive_loading():
    """The McpCatalog case: servers load one at a time, so server B's names must avoid the
    ones server A already registered. Without `taken=` a caller would have to fall back to
    `sanitize_tool_name` and thread the set by hand — what `unique_tool_names` exists to
    make impossible to forget."""
    first = unique_tool_names(["get-weather"])
    assert first["get-weather"] == "get_weather"
    second = unique_tool_names(["get.weather"], taken=first.values())
    assert second["get.weather"] != "get_weather"
    assert is_valid_tool_name(second["get.weather"])


def test_taken_does_not_rewrite_an_untaken_valid_name():
    out = unique_tool_names(["fine_name"], taken=["other"])
    assert out["fine_name"] == "fine_name"          # fixpoint survives `taken=`


def test_sanitize_is_still_a_fixpoint_through_the_public_name():
    for n in ("search_files", "日本語ツール", "café_search"):
        assert sanitize_tool_name(n) == n


def test_taken_beats_an_already_valid_name():
    """The progressive-loading worst case, and the one the other `taken=` test misses: a raw
    name that is ALREADY a valid identifier and is ALSO taken. Server B exposes `search`;
    server A registered `search` a moment ago. Without the taken-check in the reservation
    pass this returns `search` and collides, and dspy rejects a duplicate tool name."""
    out = unique_tool_names(["search"], taken=["search"])
    assert out["search"] != "search"
    assert is_valid_tool_name(out["search"])


def test_taken_beats_a_valid_name_inside_a_batch():
    """Same hazard with the rest of the batch present, so the two-pass reservation cannot
    accidentally re-claim it."""
    out = unique_tool_names(["search", "other"], taken=["search"])
    assert out["search"] != "search" and out["other"] == "other"
    assert len({*out.values(), "search"}) == 3
