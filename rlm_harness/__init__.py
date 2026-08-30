"""rlm-harness — a clean, reusable harness for building tasks on DSPy RLMs.

Public surface::

    from rlm_harness import RLMConfig, configure, RLMTask
    from rlm_harness.tools import make_schema_validator, make_fetch_tool, is_safe_url
    # Harness-engineering layer (Phase A/B/C):
    from rlm_harness import intercept_sub_lm, model_as_tool, get_sub_lm  # sub-LM hook
    from rlm_harness import TraceRecorder, current_recorder, record_tool_call  # tracing
    from rlm_harness import load_skills_as_tools                       # skills-as-tools
    from rlm_harness import load_timeline, export_sft_turns, export_rl  # replay + dataset
    from rlm_harness import atomic_write_text, atomic_write_stream      # atomic filesystem write
    from rlm_harness import compute_run_utilization, RunUtilization    # trace utilization metrics

``config``, ``trace``, ``sub_lm``, ``skills``, ``replay``, ``dataset``, ``atomic``, ``metrics`` and
the tools are import-light (no dspy). ``RLMTask`` / ``configure`` pull in dspy lazily
on first attribute access, so ``import rlm_harness`` stays cheap and the dspy-free
modules remain testable in isolation. ``intercept_sub_lm`` imports dspy only
when actually called.
"""

from __future__ import annotations

from ._retry import RLMTaskError, short_error
from ._toolname import (
    is_valid_tool_name,
    sanitize_tool_name,
    signature_from_json_schema,
    unique_tool_names,
)
from .atomic import atomic_write_stream, atomic_write_text
from .config import RLMConfig
from .dataset import export_actions, export_rl, export_sft_turns, run_label_bundle
from .isolation import run_in_subprocess
from .metrics import (
    RUN_FACT_KEYS,
    RunUtilization,
    ToolWaste,
    compute_run_facts,
    compute_run_facts_by_run,
    compute_run_utilization,
    compute_tool_waste,
    compute_tool_waste_by_run,
    compute_utilization_by_run,
)
from .replay import RecordedToolProvider, load_timeline, reconstruct
from .rubric import (
    Criterion,
    CriterionFact,
    RubricCriteria,
    criteria_facts,
    rubric_from_meta,
    rubric_to_meta,
    validate_rubric,
)
from .sandbox import SandboxCancelled, SandboxSecurityError
from .serving import HarnessPointer, bundle_artifact, parse_artifact_bundle, serve_harness
from .skills import discover_skills, load_skills_as_tools, render_skills_manifest
from .sub_lm import SubLMValidationError, intercept_sub_lm, model_as_tool
from .trace import (
    EVENT_FINAL,
    EVENT_MAIN_STEP,
    EVENT_RESULT,
    EVENT_RUN_END,
    EVENT_RUN_START,
    EVENT_SUB_CALL,
    EVENT_TOOL_CALL,
    TraceRecorder,
    current_recorder,
    group_by_run,
    load_events,
    payload_cause,
    record_tool_call,
)

__all__ = [
    # core
    "RLMConfig",
    "RLMTaskError",
    "SandboxSecurityError",
    "SandboxCancelled",
    "short_error",
    "configure",
    "get_config",
    "RLMTask",
    # sub-LM hook (Phase A)
    "intercept_sub_lm",
    "SubLMValidationError",
    "model_as_tool",
    "get_sub_lm",
    "load_skills_as_tools",
    "render_skills_manifest",
    "discover_skills",
    # tracing (Phase B)
    "TraceRecorder",
    "payload_cause",
    "current_recorder",
    "record_tool_call",
    "load_events",
    "group_by_run",
    # trace/v1 contract constants — read a trace without hardcoding the wire strings
    "EVENT_RUN_START",
    "EVENT_MAIN_STEP",
    "EVENT_SUB_CALL",
    "EVENT_TOOL_CALL",
    "EVENT_FINAL",
    "EVENT_RESULT",
    "EVENT_RUN_END",
    # replay + dataset (Phase C)
    "load_timeline",
    "reconstruct",
    "RecordedToolProvider",
    "export_sft_turns",
    "export_rl",
    "export_actions",
    "run_label_bundle",
    # reward-free rubric primitives (category is an OPAQUE caller-defined label; no taxonomy in the kit)
    "Criterion",
    "RubricCriteria",
    "CriterionFact",
    "rubric_to_meta",
    "rubric_from_meta",
    "validate_rubric",
    "criteria_facts",
    # REPL-safety rules for a tool a CONSUMER builds itself (e.g. from McpCatalog's raw names):
    # the NAME half and the SIGNATURE half. dspy validates both at RLM construction and either
    # failure aborts registration for EVERY tool on the task.
    "is_valid_tool_name",
    "sanitize_tool_name",
    "unique_tool_names",
    "signature_from_json_schema",
    # serving a downstream harness over the make_harness_tool delegation contract (server-side mirror)
    "serve_harness",
    "HarnessPointer",
    "bundle_artifact",
    "parse_artifact_bundle",
    # MCP client (optional: rlm-harness[mcp])
    "mcp_tools",
    "McpConnection",
    "McpCatalog",
    "result_text",
    # Claude subscription LM (optional: rlm-harness[subscription])
    "ClaudeAgentLM",
    "SUBSCRIPTION_PREFIX",
    # atomic filesystem write — never a partial file visible mid-write
    "atomic_write_text",
    "atomic_write_stream",
    # trace utilization metrics — how a run's activity was distributed (reward-free, like rubric)
    "RunUtilization",
    "compute_run_utilization",
    "compute_utilization_by_run",
    "compute_run_facts",
    "compute_run_facts_by_run",
    "RUN_FACT_KEYS",
    "ToolWaste",
    "compute_tool_waste",
    "compute_tool_waste_by_run",
    # safe, isolated-subprocess primitive for a web-facing consumer — a host-level orchestration
    # decision, never invoked by the model, so it lives here rather than under rlm_harness.tools
    "run_in_subprocess",
]

__version__ = "1.8.1"


def __getattr__(name: str):  # PEP 562 lazy re-export to defer dspy import
    if name == "configure":
        from .runtime import configure

        return configure
    if name == "RLMTask":
        from .task import RLMTask

        return RLMTask
    if name == "get_sub_lm":  # the configured base sub-LM, to wrap with intercept_sub_lm
        from .runtime import get_sub_lm

        return get_sub_lm
    if name == "get_config":  # the effective RLMConfig configure() stored
        from .runtime import get_config

        return get_config
    if name in ("mcp_tools", "McpConnection", "McpCatalog", "result_text"):
        # optional MCP client (rlm-harness[mcp]); mcp.py's module top is dspy/mcp-free, the SDK loads on use
        from . import mcp as _mcp

        return getattr(_mcp, name)
    if name == "ClaudeAgentLM":  # optional Claude subscription LM (imports dspy now, the SDK on use)
        from .claude_agent_lm import ClaudeAgentLM

        return ClaudeAgentLM
    if name == "SUBSCRIPTION_PREFIX":  # the claude_agent_lm.py sentinel, joins the same lazy group
        from .claude_agent_lm import SUBSCRIPTION_PREFIX

        return SUBSCRIPTION_PREFIX
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
