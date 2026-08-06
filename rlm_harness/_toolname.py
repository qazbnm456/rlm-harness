"""Tool-name rules for the REPL — one derivation, shared by every naming site.

PRIVATE (``_``-prefixed): not part of the public surface, may change without notice.

dspy validates a tool's name when ``RLM`` is constructed: it must be a Python
identifier, must not be a keyword (dspy 3.3.0+), and must be unique across the task's
tools. A name that fails any of those aborts the WHOLE tool registration, so one bad
name takes every other tool down with it.

Four places in this kit derive a tool name from data it does not control, and every one
of them shipped broken (CHANGELOG 1.0.2):

- ``mcp.py`` — the external MCP server's tool name. Hyphens and dots are the MCP naming
  norm (``get-weather``, ``db.query``), and both are hard failures.
- ``sub_lm.py:model_as_tool`` — ``f"query_{model_id}"``; a real model id
  (``openai/gpt-4o-mini``) contains ``/`` and ``.``.
- ``tools/validation.py:make_schema_validator`` — ``f"validate_{model.__name__}"``; a
  ``pydantic.create_model("bad-name")`` carries the hyphen straight through.
- ``tools/model.py`` / ``tools/harness.py`` — both returned a closure literally named
  ``call``, so using the two together silently dropped one on dspy 3.2.x and raised on
  3.3.x. Fixed by naming them distinctly rather than by sanitising.

## The fixpoint rule, and why it is the important one

``sanitize_tool_name`` MUST return a valid name unchanged. This is not a nicety: a naive
``re.sub(r"[^A-Za-z0-9_]", "_", …)`` looks obviously correct and is not, because
``str.isidentifier()`` accepts non-ASCII letters and so does dspy. ``日本語ツール`` and
``café_search`` are valid tool names that build and run today; an ASCII-only sanitiser
rewrites them (and collapses an all-CJK name to a bare ``_``), so "fixing" the hyphen bug
would BREAK a server that currently works. Character validity is therefore tested with
``str.isidentifier()`` itself, never with a character class.
"""

from __future__ import annotations

import keyword
from collections.abc import Iterable, Mapping

#: Stem for a name that cannot begin an identifier (a leading digit, or nothing left).
_STEM = "t_"


def is_valid_tool_name(name: object) -> bool:
    """True if dspy will accept ``name`` as a tool name.

    Mirrors dspy's own rule (``isidentifier()`` and not a keyword) with ONE deliberate
    difference: dspy 3.2.x does not reject keywords, but a keyword name is broken there
    anyway — dspy's Deno runner interpolates the name into ``def <name>(…):``, so
    ``def class(…)`` is a sandbox ``SyntaxError`` that aborts registration. Rejecting it
    on both versions fixes a latent 3.2.x hazard rather than inventing a stricter rule.
    """
    return isinstance(name, str) and name.isidentifier() and not keyword.iskeyword(name)


def _reserved() -> frozenset[str]:
    """dspy's reserved sandbox names, resolved lazily (keeps this module dspy-free)."""
    from ._dspy_compat import reserved_tool_names

    return reserved_tool_names()


def sanitize_tool_name(raw: str, *, taken: Iterable[str] = ()) -> str:
    """Map ``raw`` to a valid, non-reserved, unique Python identifier.

    **Fixpoint:** an already-valid, non-reserved, un-taken name is returned UNCHANGED —
    including a non-ASCII one (see the module docstring; this is the property a character
    class silently violates).

    ``taken`` are names already claimed. Prefer :func:`unique_tool_names` when naming a
    whole set at once: it reserves the valid names FIRST, so a sanitised name can never
    displace one that was already fine.
    """
    claimed = set(taken)
    text = raw if isinstance(raw, str) else str(raw)

    if is_valid_tool_name(text) and text not in _reserved() and text not in claimed:
        return text

    # Per CHARACTER, using isidentifier() rather than a character class: a char is kept
    # when it may appear in that position of an identifier, replaced with `_` otherwise.
    # ("a" + ch) tests the CONTINUATION position, which is the laxer of the two.
    cleaned = "".join(ch if ("a" + ch).isidentifier() else "_" for ch in text)
    if not cleaned or not cleaned[0].isidentifier():
        cleaned = _STEM + cleaned
    # Only reachable for input that needed sanitising at all (the fixpoint returned above):
    # give `"---"` / `""` a real stem rather than a bare `_`, which is the throwaway
    # convention in a REPL. A name that was ALREADY `_` stays `_` — the fixpoint wins, since
    # rewriting a valid name is the one thing this function must never do.
    if not cleaned.strip("_"):
        cleaned = _STEM + cleaned.lstrip("_")
    if keyword.iskeyword(cleaned) or cleaned in _reserved():
        cleaned += "_"

    if cleaned not in claimed:
        return cleaned
    n = 2
    while f"{cleaned}_{n}" in claimed:
        n += 1
    return f"{cleaned}_{n}"


def unique_tool_names(raws: Iterable[str]) -> Mapping[str, str]:
    """Map every name in ``raws`` to a valid, mutually-unique REPL name.

    Owns the collision bookkeeping, so a caller cannot forget to thread a ``taken`` set
    and silently reintroduce a duplicate — which is the exact defect this release fixes.

    **Two passes, and the order is load-bearing.** Every already-valid name is reserved
    FIRST, then the rest are sanitised into what is left. One pass would let a sanitised
    name evict a real one: with ``["get-weather", "get_weather"]`` in that order, a
    single pass renames the second to ``get_weather_2`` even though it needed no change
    at all. Callers keep the RAW name for the wire and the trace; only the REPL-facing
    name comes from here.

    Duplicates in ``raws`` collapse to one entry (the mapping is keyed by raw name); a
    server advertising the same name twice is its own bug, not something to disambiguate.
    """
    ordered = list(dict.fromkeys(raws))
    reserved = _reserved()
    claimed: set[str] = {
        r for r in ordered if is_valid_tool_name(r) and r not in reserved
    }
    mapping: dict[str, str] = {}
    for raw in ordered:
        if raw in claimed and raw not in mapping.values():
            mapping[raw] = raw
            continue
        resolved = sanitize_tool_name(raw, taken=claimed | set(mapping.values()))
        mapping[raw] = resolved
    return mapping
