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

import dspy

from rlm_harness.testing import assert_repl_safe

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

    from rlm_harness.mcp import _make_tool
    from rlm_harness.skills import load_skills_as_tools
    from rlm_harness.sub_lm import model_as_tool
    from rlm_harness.tools import (
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


# ---- tool NAMES (1.0.2) --------------------------------------------------
#
# The shape checks above were never enough on their own: dspy ALSO validates the tool's
# NAME at `RLM(...)` construction (a Python identifier, not a keyword, unique across the
# task), and a failure aborts registration for EVERY tool — one bad name silently takes
# the rest down with it. Four shipped factories derived a name from data the kit does not
# control and all four were broken; each test below fails on 1.0.1.

from pydantic import create_model

from rlm_harness._dspy_compat import reserved_tool_names
from rlm_harness._toolname import (
    is_valid_tool_name,
    sanitize_tool_name,
    unique_tool_names,
)
from rlm_harness.mcp import _make_tool
from rlm_harness.sub_lm import model_as_tool
from rlm_harness.tools import make_model_tool, make_schema_validator
from rlm_harness.tools.harness import make_harness_tool

_ok = lambda raw: types.SimpleNamespace(ok=True, errors=[])


def test_model_and_harness_tools_do_not_collide():
    """D1. Both factories returned a closure literally named `call`, and dspy raises
    `Duplicate tool name`. CLAUDE.md calls using both together an expected pattern."""
    model_tool = make_model_tool(lambda spec: "x", _ok)
    harness_tool = make_harness_tool(lambda src: types.SimpleNamespace(content="x"), _ok)
    assert model_tool.__name__ != harness_tool.__name__

    rlm = dspy.RLM("q: str -> a: str", tools=[model_tool, harness_tool])
    # BOTH must survive registration — the 1.0.1 bug was a silent drop, not an exception,
    # so asserting on the REGISTERED SET is stronger than asserting an exception was raised.
    assert len(rlm._user_tools) == 2


def test_model_as_tool_accepts_a_real_model_id():
    """D2. `f"query_{name}"` with a real model id produced `query_openai/gpt-4o-mini`,
    which dspy refuses on BOTH versions. The old test passed name="m", masking it."""
    tool = model_as_tool("openai/gpt-4o-mini", None, description="d")
    assert_repl_safe(tool)
    dspy.RLM("q: str -> a: str", tools=[tool])       # must not raise


def test_schema_validator_accepts_a_dynamic_model_name():
    """D4. `validate_{model.__name__}` — a `create_model("bad-name")` carries the hyphen
    through. Dynamic output models are exactly what `RLMTask.output_model` exists for."""
    tool = make_schema_validator(create_model("bad-name", x=(int, 1)))
    assert_repl_safe(tool)
    dspy.RLM("q: str -> a: str", tools=[tool])


@pytest.mark.parametrize("server_name", ["get-weather", "db.query", "2fa", "class", "print"])
def test_mcp_tool_names_are_repl_safe(server_name):
    """D3, the worst of the four: the MCP server's own name went straight to dspy.
    Hyphens and dots are the MCP naming NORM, and one such tool aborted the whole task."""
    # `inputSchema={}` = a zero-arg tool; it makes `_make_tool` stamp a real signature,
    # keeping this test on the NAME rule instead of tripping the shape rule.
    fake = types.SimpleNamespace(name=server_name, description="d", inputSchema={})
    bridge = types.SimpleNamespace(call=lambda *a, **k: None)
    repl = unique_tool_names([server_name])[server_name]
    tool = _make_tool(dspy, bridge, fake, "", repl)

    assert_repl_safe(tool)
    # dspy validates `Tool.name`, NOT `func.__name__` — sanitising only the latter is a
    # placebo that this construction call is here to catch.
    dspy.RLM("q: str -> a: str", tools=[tool])


def test_mcp_keeps_the_raw_name_on_the_wire_and_in_the_trace():
    """The three identities must stay separate: the model calls the sanitised name, the
    SERVER is called with its own name, and the TRACE records the configured name."""
    from rlm_harness import TraceRecorder, load_events

    sent = []
    fake = types.SimpleNamespace(name="get-weather", description="d", inputSchema={})
    bridge = types.SimpleNamespace(
        call=lambda n, a: sent.append(n) or types.SimpleNamespace(isError=False, content=[])
    )
    repl = unique_tool_names(["sc_get-weather"])["sc_get-weather"]
    tool = _make_tool(dspy, bridge, fake, "sc_", repl)

    assert tool.name == "sc_get_weather"          # what the model types
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/t.jsonl"
        with TraceRecorder(path, run_id="r"):
            tool.func()
        ev = [e for e in load_events(path) if e["type"] == "tool_call"]

    assert sent == ["get-weather"]                 # WIRE: server's own name, unprefixed
    assert ev[0]["payload"]["tool"] == "sc_get-weather"     # TRACE: configured name
    assert ev[0]["payload"]["repl_name"] == "sc_get_weather"  # the join key for readers


def test_repl_name_is_absent_when_nothing_was_sanitised():
    """Conditional emission: a tool whose name needed no change must produce a payload
    byte-identical to pre-1.0.2, so consumers' golden fixtures don't churn."""
    from rlm_harness import TraceRecorder, load_events

    fake = types.SimpleNamespace(name="search_files", description="d", inputSchema={})
    bridge = types.SimpleNamespace(
        call=lambda n, a: types.SimpleNamespace(isError=False, content=[])
    )
    tool = _make_tool(dspy, bridge, fake, "", "search_files")

    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/t.jsonl"
        with TraceRecorder(path, run_id="r"):
            tool.func()
        ev = [e for e in load_events(path) if e["type"] == "tool_call"]
    assert "repl_name" not in ev[0]["payload"]


def test_assert_repl_safe_rejects_a_bad_name():
    def f(x: str):
        ...
    assert_repl_safe(dspy.Tool(f, name="fine_name"))
    for bad in ("get-weather", "db.query", "2fa", "class"):
        with pytest.raises(AssertionError, match="valid Python identifier"):
            assert_repl_safe(dspy.Tool(f, name=bad))


def test_assert_repl_safe_reads_the_name_dspy_reads():
    """`dspy.Tool(f, name=…)` overrides `f.__name__`, and dspy validates the OVERRIDE.
    Checking `__name__` would pass a tool dspy refuses — the exact placebo this guards."""
    def sanitised_looking(x: str):
        ...
    tool = dspy.Tool(sanitised_looking, name="get-weather")   # func name fine, Tool.name not
    with pytest.raises(AssertionError, match="valid Python identifier"):
        assert_repl_safe(tool)


def test_assert_repl_safe_rejects_a_reserved_name():
    def f(x: str):
        ...
    for name in sorted(reserved_tool_names()):
        with pytest.raises(AssertionError, match="reserved"):
            assert_repl_safe(dspy.Tool(f, name=name))


# ---- the sanitiser's own contract ----------------------------------------


@pytest.mark.parametrize("name", ["search_files", "naïve_tool", "café_search", "日本語ツール", "_x"])
def test_sanitize_is_a_fixpoint_on_valid_names(name):
    """THE property. `str.isidentifier()` accepts non-ASCII letters and so does dspy, so an
    ASCII-only character class would REWRITE names that work today — collapsing an all-CJK
    name to a bare `_`. That would break a working server in the name of fixing a bug."""
    assert is_valid_tool_name(name)
    assert sanitize_tool_name(name) == name


@pytest.mark.parametrize("raw,expected", [
    ("get-weather", "get_weather"), ("db.query", "db_query"),
    ("2fa", "t_2fa"), ("class", "class_"), ("print", "print_"),
])
def test_sanitize_maps_invalid_names(raw, expected):
    assert sanitize_tool_name(raw) == expected
    assert is_valid_tool_name(sanitize_tool_name(raw))


@pytest.mark.parametrize("raw", ["", "---", "..."])
def test_sanitize_gives_unnameable_input_a_real_stem(raw):
    """Input with nothing usable left must not become a bare `_` — that is the throwaway
    convention in a REPL. Scoped to input that NEEDED sanitising: a name that was already
    `_` stays `_`, because the fixpoint outranks this (rewriting a valid name is the one
    thing the sanitiser must never do)."""
    out = sanitize_tool_name(raw)
    assert is_valid_tool_name(out) and out.strip("_")


@pytest.mark.parametrize("raw", ["_", "__", "___"])
def test_sanitize_leaves_an_already_valid_underscore_name_alone(raw):
    """The counterexample to the rule above, pinned so the stem guard is never hoisted
    above the fixpoint return."""
    assert sanitize_tool_name(raw) == raw


def test_unique_names_reserve_valid_ones_first():
    """Two-pass ordering. One pass lets a sanitised name evict a name that was already
    fine: ['get-weather', 'get_weather'] would rename the SECOND one for no reason."""
    m = unique_tool_names(["get-weather", "get.weather", "get_weather"])
    assert m["get_weather"] == "get_weather"          # untouched — it was always valid
    assert len({*m.values()}) == 3                    # and everything stays distinct
    assert all(is_valid_tool_name(v) for v in m.values())


def test_unique_names_are_stable_and_total():
    raws = ["a-b", "a.b", "a_b", "ok", "class", "日本語"]
    m = unique_tool_names(raws)
    assert set(m) == set(raws)                        # total
    assert m == unique_tool_names(raws)               # deterministic
    assert len({*m.values()}) == len(raws)            # collision-free


def test_reserved_names_are_a_superset_of_the_hardcoded_floor():
    """Union with dspy's live set, never either-or: a stale fallback may only OVER-reject
    (loud, local) — under-rejecting would pass here and raise in a consumer's rollout."""
    assert reserved_tool_names() >= {"llm_query", "llm_query_batched", "SUBMIT", "print"}


# ---- task-level REPL safety (1.1.0) --------------------------------------
#
# `assert_repl_safe` checks ONE tool. Three of dspy's construction-time rules are
# properties of the whole TASK and no per-tool check can see them; each aborts
# registration for every tool. dspy raises on each; the helper catches them STATICALLY, at
# test time, with no runtime and a message that names the rule.

from rlm_harness import RLMConfig
from rlm_harness.testing import assert_task_repl_safe


def _task(signature="q: str -> a: str", tools=()):
    """A duck-typed stand-in. `assert_task_repl_safe` must NOT import RLMTask (task.py
    imports dspy eagerly, and testing.py is documented as importable without it), so it
    duck-types — and this fixture is what pins that it really does."""
    return types.SimpleNamespace(signature=signature, tools=list(tools), output_field="a")


def _named(name):
    def f(x: str):
        """d"""
    f.__name__ = name
    return f


def test_task_level_accepts_a_clean_task():
    assert_task_repl_safe(_task(tools=[_named("alpha"), _named("beta")]))


def test_duplicate_tool_names_rejected():
    """Two tools with one name: dspy keys its tool dict by name, so this is the failure that
    used to end with the model never being told the second tool exists."""
    with pytest.raises(AssertionError, match="duplicate REPL tool name"):
        assert_task_repl_safe(_task(tools=[_named("same"), _named("same")]))


def test_input_field_colliding_with_a_tool_name_rejected():
    with pytest.raises(AssertionError, match="collide with a tool name"):
        assert_task_repl_safe(_task("lookup: str -> a: str", tools=[_named("lookup")]))


def test_input_field_colliding_with_a_sandbox_builtin_rejected():
    with pytest.raises(AssertionError, match="built-in sandbox functions"):
        assert_task_repl_safe(_task("llm_query: str -> a: str"))


@pytest.mark.parametrize("field", ["trajectory", "final_reasoning"])
def test_output_field_colliding_with_rlm_result_metadata_rejected(field):
    with pytest.raises(AssertionError, match="dspy's own RLM Prediction"):
        assert_task_repl_safe(_task(f"q: str -> {field}: str"))


def test_task_level_still_runs_the_per_tool_checks():
    with pytest.raises(AssertionError, match="valid Python identifier"):
        assert_task_repl_safe(_task(tools=[_named("get-weather")]))


def test_task_level_reads_the_INSTANCE_tools_not_the_class():
    """Runtime-assembled tools (the MCP case) live only on an instance — a class-level
    check cannot see them, which is exactly the set these rules bite on."""
    from rlm_harness import RLMTask, configure
    from rlm_harness.testing import ScriptedInterpreter, scripted_lm

    configure(RLMConfig(main_model="d/m", sub_model="d/s"),
              main_lm=scripted_lm([{"reasoning": "x", "code": "y"}]))

    class T(RLMTask):
        signature = "q: str -> a: str"
        output_field = "a"

    assert list(T.tools) == []                       # nothing declared on the class
    inst = T(tools=[_named("dup"), _named("dup")], interpreter=ScriptedInterpreter([]))
    with pytest.raises(AssertionError, match="duplicate REPL tool name"):
        assert_task_repl_safe(inst)


# ---- the signature parser mirrors dspy's ---------------------------------


def test_signature_parser_matches_dspy_on_every_repo_signature():
    """Every signature this repo and its examples actually use must parse."""
    from rlm_harness.testing import _signature_field_names

    for sig, ins, outs in [
        ("q: str -> a: str", ["q"], ["a"]),
        ("context, question -> answer", ["context", "question"], ["answer"]),
        ("document: str -> summary: Summary", ["document"], ["summary"]),
        ("a: str -> b: str, c: int", ["a"], ["b", "c"]),
        ("docs: list[str] -> answer: dict[str, int]", ["docs"], ["answer"]),
    ]:
        assert _signature_field_names(sig) == (ins, outs), sig


def test_signature_parser_rejects_what_dspy_rejects():
    """A keyword field name and a stray arrow are both things dspy itself raises on —
    the helper must not silently no-op on exactly the broken tasks."""
    from rlm_harness.testing import _signature_field_names

    with pytest.raises(AssertionError, match="does not parse"):
        _signature_field_names("class: str -> a: str")
    with pytest.raises(AssertionError, match="exactly one"):
        _signature_field_names("doc: Literal['a->b'] -> a: str")   # a naive split corrupts this
    with pytest.raises(AssertionError, match="must be a str"):
        _signature_field_names(None)


def test_task_level_accepts_a_bare_CLASS():
    """The docstring says a subclass is accepted, so pin it. On a CLASS,
    `getattr(cls, "resolved_tools")` returns the property DESCRIPTOR — truthy and not
    iterable — so a naive `getattr(...) or getattr(...)` raises
    `TypeError: 'property' object is not iterable` on exactly the documented path."""
    from rlm_harness import RLMTask

    class Clean(RLMTask):
        signature = "q: str -> a: str"
        output_field = "a"
        tools = [_named("alpha")]

    assert_task_repl_safe(Clean)                     # must not raise

    class Dup(RLMTask):
        signature = "q: str -> a: str"
        output_field = "a"
        tools = [_named("same"), _named("same")]

    with pytest.raises(AssertionError, match="duplicate REPL tool name"):
        assert_task_repl_safe(Dup)
