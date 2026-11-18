"""Alias of :mod:`rlm_harness.community.claude_agent_lm`, kept for the whole 2.x series.

Importing this path returns the moved module itself (it replaces this entry in ``sys.modules``), so
attribute access, ``monkeypatch`` and ``mock.patch`` through the old path reach the real module.
``__all__`` names the re-exports for type checkers, which cannot see that swap.
"""

import sys

from .community import claude_agent_lm as _moved
from .community.claude_agent_lm import SUBSCRIPTION_PREFIX, ClaudeAgentLM

__all__ = ["ClaudeAgentLM", "SUBSCRIPTION_PREFIX"]

sys.modules[__name__] = _moved
