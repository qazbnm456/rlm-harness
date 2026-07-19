"""REPL-safety guard: every callable injected into the RLM's REPL must expose EXPLICIT params.

dspy.RLM builds the in-sandbox tool proxy from ``inspect.signature(tool.func)`` (on BOTH the Deno and
container backends), so a ``*args``/``**kwargs`` param — or a required param after a defaulted one —
breaks the model's ability to call the tool (the ``_make_tool`` kwargs bug). This convention was
documented across the ecosystem but never enforced; this test turns it into an invariant so a future
factory can't silently reintroduce the hazard. Pure introspection — no live model, no Deno, no network."""
import inspect
import types

import pytest

pytest.importorskip("dspy")

import dspy  # noqa: E402

from rlm_kit.testing import assert_repl_safe  # noqa: E402


# ---- assert_repl_safe itself ---------------------------------------------

def test_assert_repl_safe_passes_explicit_params():
    def good(url: str, limit: int = 5):  # explicit params, defaults form a tail
        ...
    assert_repl_safe(good)                       # bare callable
    assert_repl_safe(dspy.Tool(good, name="good"))  # and wrapped as a dspy.Tool (checks .func)


def test_assert_repl_safe_rejects_var_keyword():
    def bad(**kwargs):
        ...
    with pytest.raises(AssertionError, match="VAR_KEYWORD"):
        assert_repl_safe(bad)


def test_assert_repl_safe_rejects_var_positional():
    def bad(*args):
        ...
    with pytest.raises(AssertionError, match="VAR_POSITIONAL"):
        assert_repl_safe(bad)


def test_assert_repl_safe_rejects_required_after_default():
    def f(**kw):  # body irrelevant — __signature__ drives inspection
        ...
    f.__signature__ = inspect.Signature([
        inspect.Parameter("a", inspect.Parameter.KEYWORD_ONLY, default=None),
        inspect.Parameter("b", inspect.Parameter.KEYWORD_ONLY),  # required AFTER a defaulted one
    ])
    with pytest.raises(AssertionError, match="required param"):
        assert_repl_safe(f)


# ---- every shipped REPL-tool factory -------------------------------------

def test_all_shipped_repl_factories_are_safe(tmp_path):
    from pydantic import BaseModel

    from rlm_kit.mcp import _make_tool
    from rlm_kit.skills import load_skills_as_tools
    from rlm_kit.sub_lm import model_as_tool
    from rlm_kit.tools import (
        make_command_tool,
        make_fetch_tool,
        make_model_tool,
        make_schema_validator,
        make_web_search_tool,
    )

    class M(BaseModel):
        x: int

    # inner runners/searchers/fetchers are never called at construction — only the RETURNED tool's
    # signature is under test, so their own signatures don't matter.
    tools = {
        "fetch_url": make_fetch_tool(lambda *a, **k: "body"),
        "web_search": make_web_search_tool(lambda *a, **k: []),
        "run_command": make_command_tool(lambda *a, **k: {"exit_code": 0, "stdout": "", "stderr": ""}),
        "model_tool": make_model_tool(lambda spec: "x", lambda raw: types.SimpleNamespace(ok=True, errors=[])),
        "schema_validator": make_schema_validator(M),
        "query_model": model_as_tool("m", None, description="d"),
    }
    for tool in tools.values():
        assert_repl_safe(tool)

    # progressive-disclosure skills: list_skills() + read_skill(name)
    (tmp_path / "s.md").write_text("---\nname: s\ndescription: d\n---\nbody")
    for tool in load_skills_as_tools(tmp_path, discovery="inject"):
        assert_repl_safe(tool)

    # the MCP path — the site that HAD the kwargs bug — via a fake bridge (no live server)
    fake_tool = types.SimpleNamespace(
        name="get_vulnerability", description="d",
        inputSchema={"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
    )
    fake_bridge = types.SimpleNamespace(call=lambda *a, **k: None)
    assert_repl_safe(_make_tool(dspy, fake_bridge, fake_tool, "sc_"))
