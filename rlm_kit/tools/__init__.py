"""Reusable tools that RLM tasks can expose to the model inside the REPL."""

from .command import CommandResult, make_command_tool
from .fetch import (
    is_safe_url,
    make_fetch_tool,
    parse_cidrs,
    resolved_host_is_safe,
)
from .model import ModelToolResult, make_model_tool
from .search import make_web_search_tool, normalise_search_results
from .validation import make_json_schema_validator, make_schema_validator

__all__ = [
    "make_schema_validator",
    "make_json_schema_validator",
    "is_safe_url",
    "resolved_host_is_safe",
    "parse_cidrs",
    "make_fetch_tool",
    "make_web_search_tool",
    "normalise_search_results",
    "make_model_tool",
    "ModelToolResult",
    "make_command_tool",
    "CommandResult",
]
