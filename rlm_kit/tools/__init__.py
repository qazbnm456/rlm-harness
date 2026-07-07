"""Reusable tools that RLM tasks can expose to the model inside the REPL."""

from .fetch import (
    is_safe_url,
    make_fetch_tool,
    parse_cidrs,
    resolved_host_is_safe,
)
from .model import ModelToolResult, make_model_tool
from .search import make_web_search_tool, normalise_search_results
from .validation import make_schema_validator

__all__ = [
    "make_schema_validator",
    "is_safe_url",
    "resolved_host_is_safe",
    "parse_cidrs",
    "make_fetch_tool",
    "make_web_search_tool",
    "normalise_search_results",
    "make_model_tool",
    "ModelToolResult",
]
