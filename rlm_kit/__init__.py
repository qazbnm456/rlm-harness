"""rlm-kit — a clean, reusable harness for building tasks on DSPy RLMs.

Public surface::

    from rlm_kit import RLMConfig, configure, RLMTask
    from rlm_kit.tools import make_schema_validator, make_fetch_tool, is_safe_url
    # Harness-engineering layer (Phase A/B/C):
    from rlm_kit import intercept_sub_lm, model_as_tool            # sub-LM hook
    from rlm_kit import TraceRecorder, current_recorder, record_tool_call  # tracing
    from rlm_kit import load_skills_as_tools                       # skills-as-tools
    from rlm_kit import load_timeline, export_sft_turns, export_rl  # replay + dataset

``config``, ``trace``, ``sub_lm``, ``skills``, ``replay``, ``dataset`` and the
tools are import-light (no dspy). ``RLMTask`` / ``configure`` pull in dspy lazily
on first attribute access, so ``import rlm_kit`` stays cheap and the dspy-free
modules remain testable in isolation. ``intercept_sub_lm`` imports dspy only
when actually called.
"""

from __future__ import annotations

from ._retry import RLMTaskError
from .config import RLMConfig
from .dataset import export_actions, export_rl, export_sft_turns
from .replay import RecordedToolProvider, load_timeline, reconstruct
from .sandbox import SandboxSecurityError
from .skills import discover_skills, load_skills_as_tools, render_skills_manifest
from .sub_lm import SubLMValidationError, intercept_sub_lm, model_as_tool
from .trace import (
    TraceRecorder,
    current_recorder,
    group_by_run,
    load_events,
    record_tool_call,
)

__all__ = [
    # core
    "RLMConfig",
    "RLMTaskError",
    "SandboxSecurityError",
    "configure",
    "RLMTask",
    # sub-LM hook (Phase A)
    "intercept_sub_lm",
    "SubLMValidationError",
    "model_as_tool",
    "load_skills_as_tools",
    "render_skills_manifest",
    "discover_skills",
    # tracing (Phase B)
    "TraceRecorder",
    "current_recorder",
    "record_tool_call",
    "load_events",
    "group_by_run",
    # replay + dataset (Phase C)
    "load_timeline",
    "reconstruct",
    "RecordedToolProvider",
    "export_sft_turns",
    "export_rl",
    "export_actions",
    # MCP client (optional: rlm-kit[mcp])
    "mcp_tools",
]

__version__ = "0.2.0"


def __getattr__(name: str):  # PEP 562 lazy re-export to defer dspy import
    if name == "configure":
        from .runtime import configure

        return configure
    if name == "RLMTask":
        from .task import RLMTask

        return RLMTask
    if name == "mcp_tools":  # optional MCP client (imports dspy + mcp lazily)
        from .mcp import mcp_tools

        return mcp_tools
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
