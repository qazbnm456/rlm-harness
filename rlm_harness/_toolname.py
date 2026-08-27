"""REPL-safety rules for a tool — its NAME and its SIGNATURE, one derivation each.

The module is ``_``-prefixed, but **three of its functions are PUBLIC and SemVer-frozen**
since 1.1.0, re-exported from ``rlm_harness.__all__``: :func:`is_valid_tool_name`,
:func:`sanitize_tool_name` and :func:`unique_tool_names`, plus
:func:`signature_from_json_schema`. Everything else here is private and may change.

They were promoted because a consumer driving :class:`rlm_harness.McpCatalog` gets the
server's RAW tool names and builds its own ``dspy.Tool``s from them — hitting exactly the
defects 1.0.2 fixed inside ``mcp.py``, with no sanctioned remedy, since CLAUDE.md's
"consumers EXTEND, they don't fork" invariant bars reaching into a ``_private`` name.
Both halves are needed: the NAME rule alone leaves that consumer with a valid name on a
``**kwargs`` tool that ``assert_repl_safe`` still rejects.

**What is frozen is the PROPERTIES, not the literal output strings.** Callers may rely on:
the fixpoint (below), that the result is always a valid non-reserved identifier, and that
:func:`unique_tool_names` never collides. They may NOT rely on a specific rewrite —
``t_``, the ``_2`` suffix and the trailing ``_`` are implementation. Note also that the
reserved set is read from the INSTALLED dspy, so ``sanitize_tool_name("print")`` can differ
across dspy versions under an unchanged rlm-harness. Do not persist these names as
long-lived keys; the trace's ``repl_name`` field carries the mapping per run for that.

dspy validates a tool's name when ``RLM`` is constructed: it must be a Python
identifier, must not be a keyword, and must be unique across the task's tools. A name that fails any of those aborts the WHOLE tool registration, so one bad
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
  ``call``, so using the two together made dspy raise ``Duplicate tool name``. Fixed by
  naming them distinctly rather than by sanitising.

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

import inspect
import keyword
from collections.abc import Iterable, Mapping
from typing import Any

#: Stem for a name that cannot begin an identifier (a leading digit, or nothing left).
_STEM = "t_"


def is_valid_tool_name(name: object) -> bool:
    """True if dspy will accept ``name`` as a tool name.

    Mirrors dspy's own rule: ``isidentifier()`` and not a keyword. The keyword half matters
    for a reason worth keeping written down — dspy's Deno runner interpolates the name into
    ``def <name>(…):``, so ``def class(…)`` is a sandbox ``SyntaxError`` that aborts
    registration for EVERY tool on the task, not just the offending one.
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


def unique_tool_names(
    raws: Iterable[str], *, taken: Iterable[str] = ()
) -> Mapping[str, str]:
    """Map every name in ``raws`` to a valid, mutually-unique REPL name.

    Owns the collision bookkeeping, so a caller cannot forget to thread a ``taken`` set
    and silently reintroduce a duplicate — which is the exact defect this release fixes.

    ``taken`` are names already registered on the task from an EARLIER call. This exists for
    the progressive :class:`rlm_harness.McpCatalog` case, where servers load one at a time:
    server B's names must avoid server A's, and without this parameter the caller would have
    to drop back to :func:`sanitize_tool_name` and thread the set by hand — the very thing
    this function exists to make impossible to forget.

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
    already = set(taken)
    claimed: set[str] = {
        r for r in ordered if is_valid_tool_name(r) and r not in reserved and r not in already
    }
    mapping: dict[str, str] = {}
    for raw in ordered:
        if raw in claimed and raw not in mapping.values():
            mapping[raw] = raw
            continue
        resolved = sanitize_tool_name(raw, taken=already | claimed | set(mapping.values()))
        mapping[raw] = resolved
    return mapping


def signature_from_json_schema(schema: Any) -> inspect.Signature:
    """Build the ``inspect.Signature`` to stamp on a ``**kwargs`` tool wrapper, from a JSON
    Schema object (an MCP tool's input schema, or any equivalent).

    **Why a wrapper needs this at all.** ``dspy.RLM`` builds its in-sandbox tool proxy from
    ``inspect.signature(tool.func)`` — NOT from ``dspy.Tool.args`` — on both the Deno and the
    container backend. So a wrapper written as ``def call(**kwargs)`` registers a single proxy
    param literally named ``kwargs``: the model calls ``get_thing(kwargs=…)`` and a strict
    server rejects the unexpected property. Stamping a real signature is the fix, and it is
    the SHAPE half of REPL safety — the NAME half is :func:`sanitize_tool_name`. A consumer
    building tools from :class:`rlm_harness.McpCatalog` needs both; having only the name gives
    a well-named tool that :func:`rlm_harness.testing.assert_repl_safe` still rejects.

    **REQUIRED-FIRST, and it is not cosmetic.** The Deno stub emits ``def f(<params>)`` in
    this order, and a no-default param after a defaulted one is a ``SyntaxError`` that aborts
    the ENTIRE tool registration — every other tool with it. ``KEYWORD_ONLY`` hides that
    host-side, which is exactly why the ordering has to be enforced here rather than trusted.

    Raises ``ValueError``/``TypeError`` when a property name cannot be a Python parameter (a
    keyword like ``from``, or a non-identifier like ``db.query``). **Do not "fix" that by
    sanitising the property name:** the proxy forwards the parameter name to the server as a
    JSON key, so a renamed property sends wrong wire arguments. The honest handling is to let
    the tool keep its ``**kwargs`` shape and know it is degraded — see ``mcp.py``.

    A schema with no properties yields a zero-parameter signature, which is correct and
    strictly better than ``**kwargs`` for a genuinely no-argument tool.
    """
    props = {}
    required: set[str] = set()
    if isinstance(schema, dict):
        raw_props = schema.get("properties")
        if isinstance(raw_props, dict):
            props = raw_props
        req = schema.get("required")
        if isinstance(req, list):
            # Only names that are actually declared properties: a `required` entry with no
            # matching property cannot be given a parameter, and silently inventing one would
            # make the model pass an argument the server never declared.
            required = {r for r in req if r in props}
    ordered = [n for n in props if n in required] + [n for n in props if n not in required]
    return inspect.Signature(
        [
            inspect.Parameter(
                n,
                inspect.Parameter.KEYWORD_ONLY,
                default=(inspect.Parameter.empty if n in required else None),
            )
            for n in ordered
        ]
    )
