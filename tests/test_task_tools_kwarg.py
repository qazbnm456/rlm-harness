"""`RLMTask(tools=…)` — the kwarg the guide has always documented but never had.

Before 1.1.0 `tools` was a ClassVar only, so the guide's own runnable examples

    with mcp_tools(...) as tools:
        finding = MyTask(tools=tools).run(...)

raised `TypeError`. For MCP that was the ONLY documented attach path, and it cannot be a
class-body list because the tools only exist inside the `with` block.
"""

from __future__ import annotations

import pytest

pytest.importorskip("dspy")

import dspy
from dspy.utils.dummies import DummyLM

from rlm_harness import RLMConfig, RLMTask, configure
from rlm_harness.testing import ScriptedInterpreter


@pytest.fixture(autouse=True)
def _configured():
    configure(RLMConfig(main_model="d/m", sub_model="d/s"),
              main_lm=DummyLM([{"reasoning": "x", "code": "y"}], adapter=dspy.JSONAdapter()))


def _tool(name):
    def f(x: str):
        """d"""
    f.__name__ = name
    return f


class Base(RLMTask):
    signature = "q: str -> a: str"
    output_field = "a"


def _mk(cls, **kw):
    return cls(interpreter=ScriptedInterpreter([]), **kw)


def test_kwarg_supplies_tools():
    t = _mk(Base, tools=[_tool("alpha")])
    assert [f.__name__ for f in t.resolved_tools] == ["alpha"]
    assert list(t._build_rlm()._user_tools) == ["alpha"]


def test_default_none_leaves_the_declaration_untouched():
    class Declared(Base):
        tools = [_tool("declared")]

    assert [f.__name__ for f in _mk(Declared).resolved_tools] == ["declared"]


def test_empty_list_is_a_deliberate_no_tools():
    """`tools=[]` must be distinguishable from the `None` default — otherwise there is no way
    to say "this instance gets nothing" for a class that declares some."""
    class Declared(Base):
        tools = [_tool("declared")]

    assert list(_mk(Declared, tools=[]).resolved_tools) == []


def test_kwarg_wins_REGARDLESS_of_where_the_subclass_calls_super():
    """THE reason the override is resolved at build time instead of assigned in __init__.

    Writing `self.tools = tools` inside __init__ makes the winner depend on the subclass's
    ordering: a subclass that assigns AFTER `super().__init__()` — the more idiomatic
    ordering — would silently clobber the caller's explicit kwarg.
    """
    class Before(Base):
        def __init__(self, **kw):
            self.tools = [_tool("subclass")]
            super().__init__(**kw)

    class After(Base):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.tools = [_tool("subclass")]

    for cls in (Before, After):
        got = [f.__name__ for f in _mk(cls, tools=[_tool("kwarg")]).resolved_tools]
        assert got == ["kwarg"], f"{cls.__name__} lost the explicit kwarg"


def test_subclass_assembled_tools_still_work_without_the_kwarg():
    """The pre-1.1.0 idiom (`examples/harness_run.py`) must keep working untouched."""
    class Assembles(Base):
        def __init__(self, **kw):
            self.tools = [_tool("assembled")]
            super().__init__(**kw)

    assert [f.__name__ for f in _mk(Assembles).resolved_tools] == ["assembled"]


def test_instance_tools_do_not_leak_to_the_class_or_to_siblings():
    a = _mk(Base, tools=[_tool("mine")])
    b = _mk(Base)
    assert [f.__name__ for f in a.resolved_tools] == ["mine"]
    assert list(b.resolved_tools) == []
    assert list(Base.tools) == []
    assert list(RLMTask.tools) == []


def test_replaces_never_merges():
    """Merging would make the effective list depend on inheritance depth."""
    class Declared(Base):
        tools = [_tool("declared")]

    got = [f.__name__ for f in _mk(Declared, tools=[_tool("kwarg")]).resolved_tools]
    assert got == ["kwarg"]
