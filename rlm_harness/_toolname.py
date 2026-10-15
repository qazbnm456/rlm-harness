"""REPL-safety rules for a tool: its NAME, its SIGNATURE and the DESCRIPTION the model
reads, one derivation each.

The module is ``_``-prefixed, but **three of its functions are PUBLIC and SemVer-frozen**
since 1.1.0, re-exported from ``rlm_harness.__all__``: :func:`is_valid_tool_name`,
:func:`sanitize_tool_name` and :func:`unique_tool_names`, plus
:func:`signature_from_json_schema`. Everything else here is private and may change.

They were promoted because a consumer driving :class:`rlm_harness.McpCatalog` gets the
server's RAW tool names and builds its own ``dspy.Tool``s from them: hitting exactly the
defects 1.0.2 fixed inside ``mcp.py``, with no sanctioned remedy, since docs/INVARIANTS.md's
"consumers EXTEND, they don't fork" invariant bars reaching into a ``_private`` name.
Both halves are needed: the NAME rule alone leaves that consumer with a valid name on a
``**kwargs`` tool that ``assert_repl_safe`` still rejects.

**What is frozen is the PROPERTIES, not the literal output strings.** Callers may rely on:
the fixpoint (below), that the result is always a valid non-reserved identifier, and that
:func:`unique_tool_names` never collides. They may NOT rely on a specific rewrite:
``t_``, the ``_2`` suffix and the trailing ``_`` are implementation. Note also that the
reserved set is read from the INSTALLED dspy, so ``sanitize_tool_name("print")`` can differ
across dspy versions under an unchanged rlm-harness. Do not persist these names as
long-lived keys; the trace's ``repl_name`` field carries the mapping per run for that.

dspy validates a tool's name when ``RLM`` is constructed: it must be a Python
identifier, must not be a keyword, and must be unique across the task's tools. A name that fails any of those aborts the WHOLE tool registration, so one bad
name takes every other tool down with it.

Four places in this kit built a tool name that dspy then refused, and every one of them
shipped broken (CHANGELOG 1.0.2). Three DERIVE the name from data the kit does not control
and are the reason this module exists; the fourth collided on a hardcoded name and is listed
with them because it is the same failure at registration:

- ``mcp.py``: the external MCP server's tool name. Hyphens and dots are the MCP naming
  norm (``get-weather``, ``db.query``), and both are hard failures.
- ``sub_lm.py:model_as_tool``: ``f"query_{model_id}"``; a real model id
  (``openai/gpt-4o-mini``) contains ``/`` and ``.``.
- ``tools/validation.py:make_schema_validator``: ``f"validate_{model.__name__}"``; a
  ``pydantic.create_model("bad-name")`` carries the hyphen straight through.
- ``tools/model.py`` / ``tools/harness.py``: both returned a closure literally named
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
import re
from collections.abc import Iterable, Mapping
from typing import Any

#: Stem for a name that cannot begin an identifier (a leading digit, or nothing left).
_STEM = "t_"


def is_valid_tool_name(name: object) -> bool:
    """True if dspy will accept ``name`` as a tool name.

    Mirrors dspy's own rule: ``isidentifier()`` and not a keyword. The keyword half matters
    for a reason worth keeping written down: dspy's Deno runner interpolates the name into
    ``def <name>(…):``, so ``def class(…)`` is a sandbox ``SyntaxError`` that aborts
    registration for EVERY tool on the task, not just the offending one.
    """
    return isinstance(name, str) and name.isidentifier() and not keyword.iskeyword(name)


def _is_reserved(name: str) -> bool:
    """Would the installed dspy refuse ``name`` because the sandbox owns it?

    A PREDICATE, not membership in a set, because since dspy 3.4.1 the rule is not a set: its
    sandbox ``dspy`` facade owns ``dspy`` plus every ``_dspy``/``__dspy`` prefix. Before this,
    ``sanitize_tool_name("dspy")`` returned ``"dspy"`` unchanged and ``RLM(...)`` then refused
    the whole tool registration, which is the failure mode this module exists to prevent (and
    the module docstring's "one bad name takes every other tool down with it").

    Resolved lazily, like the set it replaced, so this module stays dspy-free at import.
    """
    from ._dspy_compat import is_reserved_tool_name

    return is_reserved_tool_name(name)


def sanitize_tool_name(raw: str, *, taken: Iterable[str] = ()) -> str:
    """Map ``raw`` to a valid, non-reserved, unique Python identifier.

    **Fixpoint:** an already-valid, non-reserved, un-taken name is returned UNCHANGED,
    including a non-ASCII one (see the module docstring; this is the property a character
    class silently violates).

    ``taken`` are names already claimed. Prefer :func:`unique_tool_names` when naming a
    whole set at once: it reserves the valid names FIRST, so a sanitised name can never
    displace one that was already fine.
    """
    claimed = set(taken)
    text = raw if isinstance(raw, str) else str(raw)

    if is_valid_tool_name(text) and not _is_reserved(text) and text not in claimed:
        return text

    # Per CHARACTER, using isidentifier() rather than a character class: a char is kept
    # when it may appear in that position of an identifier, replaced with `_` otherwise.
    # ("a" + ch) tests the CONTINUATION position, which is the laxer of the two.
    cleaned = "".join(ch if ("a" + ch).isidentifier() else "_" for ch in text)
    if not cleaned or not cleaned[0].isidentifier():
        cleaned = _STEM + cleaned
    # Only reachable for input that needed sanitising at all (the fixpoint returned above):
    # give `"---"` / `""` a real stem rather than a bare `_`, which is the throwaway
    # convention in a REPL. A name that was ALREADY `_` stays `_`: the fixpoint wins, since
    # rewriting a valid name is the one thing this function must never do.
    if not cleaned.strip("_"):
        cleaned = _STEM + cleaned.lstrip("_")
    if keyword.iskeyword(cleaned) or _is_reserved(cleaned):
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
    and silently reintroduce a duplicate, which is the exact defect this release fixes.

    ``taken`` are names already registered on the task from an EARLIER call. This exists for
    the progressive :class:`rlm_harness.McpCatalog` case, where servers load one at a time:
    server B's names must avoid server A's, and without this parameter the caller would have
    to drop back to :func:`sanitize_tool_name` and thread the set by hand: the very thing
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
    already = set(taken)
    claimed: set[str] = {
        r for r in ordered if is_valid_tool_name(r) and not _is_reserved(r) and r not in already
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
    ``inspect.signature(tool.func)``, NOT from ``dspy.Tool.args``, on both the Deno and the
    container backend. So a wrapper written as ``def call(**kwargs)`` registers a single proxy
    param literally named ``kwargs``: the model calls ``get_thing(kwargs=…)`` and a strict
    server rejects the unexpected property. Stamping a real signature is the fix, and it is
    the SHAPE half of REPL safety. The NAME half is :func:`sanitize_tool_name`. A consumer
    building tools from :class:`rlm_harness.McpCatalog` needs both; having only the name gives
    a well-named tool that :func:`rlm_harness.testing.assert_repl_safe` still rejects.

    **REQUIRED-FIRST, and it is not cosmetic.** The Deno stub emits ``def f(<params>)`` in
    this order, and a no-default param after a defaulted one is a ``SyntaxError`` that aborts
    the ENTIRE tool registration: every other tool with it. ``KEYWORD_ONLY`` hides that
    host-side, which is exactly why the ordering has to be enforced here rather than trusted.

    Raises ``ValueError``/``TypeError`` when a property name cannot be a Python parameter (a
    keyword like ``from``, or a non-identifier like ``db.query``). **Do not "fix" that by
    sanitising the property name:** the proxy forwards the parameter name to the server as a
    JSON key, so a renamed property sends wrong wire arguments. The honest handling is to let
    the tool keep its ``**kwargs`` shape and know it is degraded: see ``mcp.py``.

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


# --- what the model READS about a tool -------------------------------------------------------
#
# A tool's docstring is PROMPT TEXT, not developer documentation. dspy reads it automatically
# (`dspy.Tool.__init__` takes `desc` from `__doc__`), renders one line per tool in
# `RLM._format_tool_docs`, appends the block to `generate_action.signature.instructions`, and the
# adapter puts that string in the SYSTEM message. Measured on a two-tool task: 8938-char system
# message whose instruction string is 7799, of which the rendered tool-doc block is 4957 (63.6%);
# the user message was 448.
#
# Nobody knew that, so the kit's tool docstrings were written for Sphinx. dspy flattens them with
# `.replace("\n", "  ")`, which turns a docstring's own indentation into runs of ten spaces and
# passes `` `` `` through verbatim, so the model read:
#
#     Read lines ``start_line``..``end_line`` (1-indexed, inclusive; ``end_line=None`` means
#               "to the end of the file") of a file at ``path``
#
# Of the fourteen tools the kit ships, three had no model-facing docstring at all, so dspy
# rendered the literal string "No description" for them; of the eleven that had one, eight
# carried Sphinx markup and ten produced markup or a mid-sentence gap.
#
# This is normalised once, at `RLMTask._build_rlm`, for the same reason `_ensure_tool_timing` is:
# a rule that depends on every author remembering is missing for someone, and a consumer whose
# tools delegate to these factories has no seam of its own. It also covers MCP, where the text
# belongs to a remote server and no docstring convention can reach it at all.

_RST_ROLE = re.compile(r":[a-zA-Z][a-zA-Z0-9_:+-]*:`(~?)([^`\n]+)`")
_RST_LITERAL = re.compile(r"``([^`\n]+)``")


def _role_target(match: re.Match[str]) -> str:
    """``:func:`~a.b.c``` renders as ``c`` and ``:func:`a.b``` as ``a.b``: Sphinx's own rule for
    the tilde, kept because a reader of the prompt should see what a reader of the docs sees.

    Stripped at both ends. Whitespace is collapsed BEFORE this runs, so a target the author
    wrapped across two source lines arrives with a space in it and the tilde's ``rsplit`` hands
    back a leading one.

    Measured, against ``test_the_transform_table``: the OUTER strip and the final re-collapse in
    :func:`model_facing_description` are each independently sufficient, and removing both is the
    only pair that breaks the behaviour. ``match.group(2).strip()`` is redundant in every
    combination, kept because it is where the space actually appears and a reader of this function
    should not have to know about a re-collapse two calls away.
    """
    target = match.group(2).strip()
    tail = target.rsplit(".", 1)[-1] if match.group(1) else target
    return tail.strip()


def model_facing_description(raw: object) -> str:
    """Render a tool's docstring as the one line the model will actually be shown.

    Three transforms, and the ORDER is load-bearing:

    1. Whitespace collapses to single spaces. This is the fix for the defect that started it:
       dspy renders the description on ONE line by replacing each newline with two spaces, so a
       docstring's own 8-space indentation became a 10-space gap mid-sentence. Collapsing cleanly
       loses nothing dspy's own flattening would have kept, and it makes
       :func:`inspect.cleandoc` unnecessary ahead of it.
    2. Sphinx roles become their target. ``:func:`~a.b.c``` is markup for a doc build, and the
       model is not a doc build.
    3. Inline literals lose one backtick layer.

    **Collapsing FIRST is what lets 2 and 3 see a construct the author wrapped across two source
    lines.** Running them before the collapse left ``very long\n    literal name`` untouched in
    the prompt, a defect a consumer hits and the kit's own docstrings happen not to.

    Both patterns still exclude newlines, but by the time they run there are none left, so that
    exclusion no longer decides anything and should not be read as a safeguard. It costs one
    thing, accepted knowingly: two standalone `` `` `` markers on separate lines merge into one
    literal and each loses a backtick. A docstring using a bare double backtick as a token is
    rarer than one wrapping a literal, so this is the better side of the trade.

    Returns ``""`` for a missing or non-string docstring, which is the caller's signal to leave
    the tool alone rather than hand dspy an empty description.
    """
    if not isinstance(raw, str) or not raw.strip():
        return ""
    text = " ".join(raw.split())          # FIRST: see the comment on ordering above
    text = _RST_ROLE.sub(_role_target, text)
    text = _RST_LITERAL.sub(r"`\1`", text)
    return " ".join(text.split())         # again: a substitution can leave a doubled space


def with_model_facing_description(tool: Any) -> Any:
    """Return ``tool`` with its model-facing description normalised, NEVER mutating a caller's
    object.

    Two shapes, because the description arrives by two different doors:

    * An object carrying an explicit ``desc`` (a ``dspy.Tool``, which is how ``mcp.py`` hands a
      server's description over) is replaced by a pydantic copy. dspy reads ``desc`` and ignores
      ``__doc__`` there, so normalising the docstring would do nothing.
    * A plain function or bound method has already been through
      :func:`rlm_harness.trace._ensure_tool_timing`, which returns a ``functools.wraps`` wrapper
      carrying ``__rlm_timed__``. The docstring is set on that wrapper, so the consumer's own
      function is untouched.

    Anything else, a ``functools.partial``, a callable instance, a coroutine function, is returned
    unchanged: the timing seam skipped it too, so there is no kit-owned object to write on and the
    alternative is mutating something the caller owns. The mark is read rather than inferred from
    ``__wrapped__``, which a consumer's own decorator would also set.
    """
    desc = getattr(tool, "desc", None)
    if isinstance(desc, str):
        cleaned = model_facing_description(desc)
        copy = getattr(tool, "model_copy", None)
        if cleaned and cleaned != desc and callable(copy):
            return copy(update={"desc": cleaned})
        return tool

    if not getattr(tool, "__rlm_timed__", False):
        return tool                       # not our wrapper: nothing here is ours to rewrite
    cleaned = model_facing_description(getattr(tool, "__doc__", None))
    if cleaned and cleaned != tool.__doc__:
        try:
            tool.__doc__ = cleaned
        except (AttributeError, TypeError):   # a shape that refuses the assignment
            pass
    return tool
