"""Provider-agnostic ``web_search`` building blocks (mirrors ``fetch.py``).

A search tool needs two halves: the PROVIDER (an HTTP call to DuckDuckGo / Tavily /
TinyFish / … plus its API key) and the generic GUARD/NORMALISE step. rlm-harness owns only
the generic half — it picks NO provider. The consuming project supplies a ``searcher``
(``query -> raw results``) and rlm-harness turns the raw results into a safe, capped,
uniform ``[{"title","url","snippet"}]`` list.

Two entry points, matching ``fetch.py``'s ``is_safe_url`` (primitive) + ``make_fetch_tool``
(factory):

- ``normalise_search_results`` — the reusable primitive. Sync, dependency-free. Use it
  inside your own tool, exactly as the fetch tool reuses ``is_safe_url``.
- ``make_web_search_tool`` — a SYNC factory that builds the whole tool for you (parallel to
  ``make_fetch_tool``), ready to hand to ``RLMTask(tools=…)``. Sync because dspy.RLM invokes
  tools synchronously; an async tool there never runs.

Search results are URLs the agent will then fetch, so ``is_safe_url`` is reused here to
drop internal-looking result URLs before they reach the model.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from ..trace import record_tool_call
from .fetch import is_safe_url

# A provider searcher maps a query string to a list of raw result dicts. SYNC — dspy.RLM
# tools must be sync (see make_web_search_tool). Each raw dict should carry at least a
# "url"; "title"/"snippet" are optional.
Searcher = Callable[[str], list]


def normalise_search_results(
    raw: Any, *, max_results: int = 5, drop_unsafe_urls: bool = True
) -> list[dict]:
    """Turn a provider's raw results into a safe, capped, uniform list of
    ``{"title","url","snippet"}`` dicts. Drops entries with no URL and — by default —
    internal-looking URLs (reusing ``is_safe_url``), then caps to ``max_results``. The
    provider-specific shape (mapping the provider's field names onto title/url/snippet)
    is the caller's job; this is the shared guard/normalise step every provider needs."""
    out: list[dict] = []
    for r in raw or []:
        if not isinstance(r, dict):
            continue
        url = str(r.get("url") or "").strip()
        if not url:
            continue
        if drop_unsafe_urls and not is_safe_url(url):
            continue
        out.append(
            {
                "title": str(r.get("title") or "").strip(),
                "url": url,
                "snippet": str(r.get("snippet") or "").strip(),
            }
        )
        if len(out) >= max_results:
            break
    return out


def make_web_search_tool(
    searcher: Searcher, *, max_results: int = 5, drop_unsafe_urls: bool = True
) -> Callable[[str], list[dict] | str]:
    """Wrap a project-supplied (SYNC) ``searcher`` into a sync ``web_search(query)`` tool:
    validates the query, calls the searcher, and returns the normalised result list — or a
    short error string (rather than raising) on an empty query or a searcher failure, so the
    RLM can react to it as text (mirrors ``make_fetch_tool``). Picks NO provider. Sync because
    dspy.RLM invokes tools synchronously; an async tool there returns a coroutine that never runs."""

    def web_search(query: str) -> list[dict] | str:
        # Both ``ok=False`` paths (empty query, searcher error) return an explanatory string,
        # not ``[]`` — an error string is reactable in the REPL where an empty list reads as
        # "searched, found nothing". Symmetric with ``make_fetch_tool``.
        q = (query or "").strip()
        if not q:
            record_tool_call("web_search", args={"query": q}, ok=False, note="empty query")
            return "Refused: empty search query."
        # The outbound call only -- narrower than the task seam's window, so it wins. The
        # empty-query refusal above now carries the seam's ~0 instead of no field at all; see
        # `make_fetch_tool` for why absent stopped being the honest answer.
        t0 = time.perf_counter()
        try:
            results = normalise_search_results(
                searcher(q), max_results=max_results, drop_unsafe_urls=drop_unsafe_urls
            )
        except Exception as exc:
            record_tool_call(
                "web_search", args={"query": q}, ok=False,
                duration_s=time.perf_counter() - t0,
                note=f"error: {type(exc).__name__}",
            )
            return f"Search error for {q!r}: {type(exc).__name__}: {str(exc)[:160]}"
        record_tool_call("web_search", args={"query": q}, ok=True,
                         duration_s=time.perf_counter() - t0, results=results)
        return results

    return web_search
