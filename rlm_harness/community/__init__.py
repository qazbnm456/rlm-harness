"""Language models that run on a vendor's consumer sign-in through the vendor's own tool.

**These routes work as of the dates in ``rlm_harness/community/README.md`` and are not guaranteed to
keep working.** A vendor can change its terms or how it enforces them at any time, and a breach can
suspend the account. This kit is not affiliated with any vendor. Whether a workload fits your plan's
terms is yours to check, and the README quotes what each vendor has said.

``ROUTES`` is the one table ``configure()`` reads to route a model string by its prefix
(``claude-agent-sdk/<id>``) onto the adapter that serves it. This module imports no dspy, so reading
the table costs nothing; an adapter module is imported only when a model string names its prefix.

``ClaudeAgentLM`` is also exported from ``rlm_harness`` and is covered by the kit's SemVer promise.
Every other route is exported from here only and is outside that promise: one can be withdrawn in a
minor release, and its prefix then raises an error that says why.
"""

from __future__ import annotations

import importlib
import logging
import threading
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["EXPLICIT", "NOT_EXPLICIT", "ROUTES", "Route", "route_for"]

#: A vendor has said in writing that this use of a consumer plan is allowed.
EXPLICIT = "explicit"
#: No such statement exists, or the vendor's terms read against this use.
NOT_EXPLICIT = "not-explicit"


@dataclass(frozen=True)
class Route:
    """One model-string prefix and the adapter class that serves it."""

    prefix: str
    module: str
    attr: str
    #: A model id for the error a bare prefix raises, e.g. ``claude-agent-sdk/<example>``.
    example: str
    #: ``EXPLICIT`` or ``NOT_EXPLICIT``. The statement it rests on is quoted, with the date it was
    #: read, in the README's table; this field is only what the code acts on.
    vendor_permission: str


ROUTES: tuple[Route, ...] = (
    Route(
        prefix="claude-agent-sdk/",
        module="rlm_harness.community.claude_agent_lm",
        attr="ClaudeAgentLM",
        example="claude-sonnet-5",
        vendor_permission=EXPLICIT,
    ),
    Route(
        prefix="agy/",
        module="rlm_harness.community.antigravity_lm",
        attr="AntigravityLM",
        example="gemini-3.8-flash-medium",
        vendor_permission=NOT_EXPLICIT,
    ),
)


def route_for(model: str) -> Route | None:
    """The route whose prefix ``model`` starts with, or ``None``."""
    return next((route for route in ROUTES if model.startswith(route.prefix)), None)


def _route(attr: str) -> Route:
    """The route served by the class named ``attr``. Adapters read their own prefix through it."""
    for route in ROUTES:
        if route.attr == attr:
            return route
    raise LookupError(f"no entry in rlm_harness.community.ROUTES serves {attr!r}")


_noted: set[str] = set()
_noted_lock = threading.Lock()


def _note_constructed(attr: str) -> None:
    """Called by every adapter's constructor. Warns once per process for a route whose vendor has
    not explicitly allowed this use; says nothing for one that has."""
    route = _route(attr)
    if route.vendor_permission == EXPLICIT:
        return
    with _noted_lock:
        if attr in _noted:
            return
        _noted.add(attr)
    logger.warning(
        "%s (model prefix %r) runs on a vendor's consumer sign-in, and the vendor has not "
        "explicitly allowed this use. It works today, and a change in the vendor's terms or "
        "enforcement can stop it or put the account at risk. See rlm_harness/community/README.md "
        "before relying on it.",
        attr,
        route.prefix,
    )


def __getattr__(name: str) -> Any:
    """Each route's adapter class, imported on first access so this module stays dspy-free."""
    route = next((r for r in ROUTES if r.attr == name), None)
    if route is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(route.module), name)
