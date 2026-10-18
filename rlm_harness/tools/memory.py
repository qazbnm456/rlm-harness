"""``make_memory_search_tool`` / ``make_memory_get_tool`` / ``make_memory_note_tool``: the REPL side
of :class:`rlm_harness.memory.MemoryStore`.

Three factories rather than one, so reading and writing are separate capabilities in the way
``make_read_file_tool`` and ``make_write_file_tool`` are: a task that should only consult memory
simply leaves the note tool out of its ``tools=[...]``.

**A model writes only ``inferred``, and that is enforced by SHAPE.** ``memory_note`` takes a text
and an optional id to revise, and nothing else. No argument names a provenance, so no prompt, no
injected page and no confused plan can make a model's words arrive as ``observed`` or ``said``.
The host-side :meth:`~rlm_harness.memory.MemoryStore.add` is where those two are written, by code
that knows how the text arrived.

**Whatever these tools return is untrusted model context**, a prompt-injection surface like
``fetch_url``'s output, and a persistent one: a note written after reading a hostile page is
returned to every later run whose search matches it. Each returned item carries its
``provenance`` so the model, and anyone reading the trace, can tell a note from a source.

**Every call records exactly one ``tool_call``** through
:func:`~rlm_harness.trace.record_tool_call`, failures included, with a ``memory_ids`` field listing
the ids the call returned or wrote. The arguments are the RAW values the model passed, before any
clamp, with one bound: an argument longer than its cap (the note's ``max_chars`` for its text,
``_TRACE_ARG_CHARS`` for anything else; a non-string measured by its JSON form) is recorded cut,
with ``<name>_len`` beside it, so one oversized argument cannot write megabytes into the JSONL. A
sentinel that echoes an argument back cuts it at ``_ECHO_CHARS``. No new trace event type is
involved: a memory call is a tool call.

Each tool returns a string and never raises on a model-controlled argument, whatever its type. A
database failure (a lock held past the store's timeout, a full disk, a damaged file) comes back as
``Memory error: <type>`` and is recorded with ``ok=False``, the way ``read_file`` and
``write_file`` report their I/O errors. All three run host-side, outside the sandbox, like every
other tool in this package.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterable
from typing import Any

from ..memory import MemoryStore, _normalise_tags
from ..trace import current_recorder, record_tool_call
from .fs import _validate_tool_name

_TRUNCATED = " …[truncated]"

#: The longest string argument recorded whole in the trace; a longer one is cut and its length kept.
_TRACE_ARG_CHARS = 1000

#: The longest argument a sentinel echoes back to the model.
_ECHO_CHARS = 200


def _cut(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[:max_chars] + _TRUNCATED


def _as_text(value: Any) -> str:
    """A model-supplied argument as a str: ``None`` becomes empty, anything else its ``str``."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return str(value)
    except Exception:   # an object whose __str__ raises still gets an answer, not a traceback
        return f"<{type(value).__name__}>"


def _as_limit(value: Any, default: int, cap: int) -> int:
    """A model-supplied ``limit`` clamped to ``[1, cap]``; anything unreadable is ``default``."""
    try:
        number = int(value)
    except Exception:   # also an object whose __int__ raises something else entirely
        number = default
    return max(1, min(number, cap))


def _bounded(**limits: tuple[Any, int]) -> tuple[dict, dict]:
    """The trace ``args`` for ``name=(raw value, cap)`` pairs, plus the ``<name>_len`` fields.

    A value is recorded as passed unless it is longer than its cap: a string by its own length,
    anything else (a list or dict a model passed) by the length of its JSON form. An oversized
    value is recorded cut, as text, with its full length in ``<name>_len``. A clamp never changes
    what is recorded; only size does.
    """
    args: dict[str, Any] = {}
    lengths: dict[str, int] = {}
    for key, (value, cap) in limits.items():
        serialisable = True
        if isinstance(value, str):
            text = value
        else:
            try:
                text = json.dumps(value, default=str, ensure_ascii=False)
            except Exception:   # a circular structure, tuple keys, a __str__ that raises
                serialisable = False
                text = _as_text(value)
        if len(text) > cap:
            args[key] = _cut(text, cap)
            lengths[f"{key}_len"] = len(text)
        else:
            # A value the trace's own `json.dumps` would choke on is recorded as its text, or the
            # recorder would raise out of the tool after the work (a note's write) was done.
            args[key] = value if serialisable else text
    return args, lengths


def _echo(text: str) -> str:
    return repr(_cut(text, _ECHO_CHARS))


def make_memory_search_tool(
    store: MemoryStore,
    *,
    name: str = "memory_search",
    max_results: int = 8,
    max_chars: int = 500,
) -> Callable[..., str]:
    """Build a ``memory_search(query, limit)`` tool over ``store``.

    ``max_results`` (default 8) caps what one call returns, whatever ``limit`` the model asks for;
    the description states the cap this build enforces. ``max_chars`` (default 500) cuts each
    item's text, marking the cut with ``…[truncated]``, so one long source cannot fill the
    model's context; ``memory_get`` returns more of one item. Items retired as of the call are not
    returned. ``name`` lets two stores coexist in one task, validated as the file tools validate
    theirs.
    """
    _validate_tool_name(name)
    cap = max(1, int(max_results))
    width = max(1, int(max_chars))

    def memory_search(query: str = "", limit: int = 5) -> str:
        """Search long-term memory for `query`; an empty query lists the newest items. Returns a
        JSON list of up to {cap} items, best match first, each with id, provenance, origin, tags,
        title, title_provenance, text and revises; each text is cut at {width} characters, and
        reading the item by id returns more. provenance is observed (taken from a source), said
        (written by a person) or inferred (written by a model, never evidence on its own). A title
        is a label, not the item: title_provenance says who wrote it, and an inferred title is
        never evidence. Returns "No memories match ..."
        when nothing matches, "Memory is empty." when nothing is stored, "No current
        memories." when everything stored has been retired, or "Memory error: ..." when the store
        cannot be read."""
        args, lengths = _bounded(query=(query, _TRACE_ARG_CHARS), limit=(limit, _TRACE_ARG_CHARS))
        q = _as_text(query).strip()
        try:
            items = store.search(q, limit=_as_limit(limit, 5, cap))
            empty = not items and not q and store.count() == 0
            titles = {i.id: store.labels(i.id) for i in items}
        except sqlite3.Error as exc:
            record_tool_call(name, args=args, ok=False, memory_ids=[],
                             note=f"error: {type(exc).__name__}", **lengths)
            return f"Memory error: {type(exc).__name__}"
        record_tool_call(name, args=args, ok=True, memory_ids=[i.id for i in items], **lengths)
        if not items:
            if q:
                return f"No memories match {_echo(q)}."
            return "Memory is empty." if empty else "No current memories."
        rows = [
            {"id": i.id, "provenance": i.provenance, "origin": i.origin, "tags": list(i.tags),
             "title": titles[i.id].title, "title_provenance": titles[i.id].provenance["title"],
             "text": _cut(i.text, width), "revises": i.revises}
            for i in items
        ]
        return json.dumps(rows, ensure_ascii=False)

    # The description is PROMPT TEXT, so the number in it is built from this build's cap. `.replace`
    # rather than `.format`, which would read any brace in the prose as a field.
    memory_search.__doc__ = (
        (memory_search.__doc__ or "").replace("{cap}", str(cap)).replace("{width}", str(width))
    )
    memory_search.__name__ = name
    memory_search.__qualname__ = name
    return memory_search


def make_memory_get_tool(
    store: MemoryStore,
    *,
    name: str = "memory_get",
    max_chars: int = 4000,
) -> Callable[..., str]:
    """Build a ``memory_get(item_id)`` tool over ``store``.

    Returns one item, retired or not, as JSON with its text cut at ``max_chars`` (default 4000),
    plus ``valid_to`` and ``revised_by``: the ids of the items that revise it, oldest first.
    """
    _validate_tool_name(name)
    width = max(1, int(max_chars))

    def memory_get(item_id: str) -> str:
        """Read one memory by id, retired or not, as JSON with all its fields, its text cut at
        {width} characters, and revised_by, the ids of later items that revise it. Returns
        "Not found: ..." for an unknown id, or "Memory error: ..." when the store cannot be
        read."""
        args, lengths = _bounded(item_id=(item_id, _TRACE_ARG_CHARS))
        key = _as_text(item_id).strip()
        try:
            item = store.get(key)
            revised_by = [] if item is None else [r.id for r in store.revisions(item.id)]
        except sqlite3.Error as exc:
            record_tool_call(name, args=args, ok=False, memory_ids=[],
                             note=f"error: {type(exc).__name__}", **lengths)
            return f"Memory error: {type(exc).__name__}"
        if item is None:
            record_tool_call(name, args=args, ok=False, memory_ids=[], **lengths)
            return f"Not found: {_echo(key)}."
        record_tool_call(name, args=args, ok=True, memory_ids=[item.id], **lengths)
        row = item.to_dict()
        row["text"] = _cut(item.text, width)
        row["revised_by"] = revised_by
        return json.dumps(row, ensure_ascii=False)

    memory_get.__doc__ = (memory_get.__doc__ or "").replace("{width}", str(width))
    memory_get.__name__ = name
    memory_get.__qualname__ = name
    return memory_get


def make_memory_note_tool(
    store: MemoryStore,
    *,
    name: str = "memory_note",
    max_chars: int = 2000,
    tags: Iterable[str] = (),
) -> Callable[..., str]:
    """Build a ``memory_note(text, revises)`` tool: the only way a model writes to ``store``.

    Every note is stored with provenance ``inferred``, ``origin`` ``run:<run_id>`` when a
    :class:`~rlm_harness.trace.TraceRecorder` is active (``model`` otherwise), that run id, and
    the factory's ``tags``. ``revises`` names an existing item the note corrects or extends; the
    original stays exactly as it was. The gate is deterministic and each refusal is a string the
    model can read: an empty text, a text over ``max_chars`` (default 2000), an unknown id, or a
    text the store itself refuses.
    """
    _validate_tool_name(name)
    width = max(1, int(max_chars))
    note_tags = _normalise_tags(tags)   # a bad tag fails here, at build time, not on a model's call

    def memory_note(text: str, revises: str = "") -> str:
        """Save a note to long-term memory, at most {cap} characters, stored as inferred. Pass the
        id of an existing memory as revises to correct or extend it; the original is kept.
        Returns "Noted: ..." with the new id, a "Refused: ..." string (never raises) for an empty
        or over-long text or an unknown id, or "Memory error: ..." when the store cannot be
        written."""
        args, lengths = _bounded(text=(text, width), revises=(revises, _TRACE_ARG_CHARS))
        body = _as_text(text).strip()
        target = _as_text(revises).strip()

        def refuse(note: str, message: str) -> str:
            record_tool_call(name, args=args, ok=False, memory_ids=[], note=note, **lengths)
            return message

        if not body:
            return refuse("empty text", "Refused: text must be non-empty.")
        if len(body) > width:
            return refuse("too long", f"Refused: text exceeds {width} characters.")
        recorder = current_recorder()
        run_id = getattr(recorder, "run_id", None) if recorder is not None else None
        origin = f"run:{run_id}" if run_id else "model"
        try:
            if target:
                item = store.revise(target, body, origin=origin, tags=note_tags, run_id=run_id)
            else:
                item = store.add(body, provenance="inferred", origin=origin, tags=note_tags,
                                 run_id=run_id)
        except KeyError:
            return refuse("unknown id", f"Refused: no memory with id {_echo(target)}.")
        except ValueError as exc:
            # The store refused the text: its own `max_item_chars` is below this tool's cap, or the
            # text cannot be encoded (a lone surrogate raises `UnicodeEncodeError`, a ValueError).
            return refuse("store refused", f"Refused: {_cut(str(exc), _ECHO_CHARS)}")
        except sqlite3.Error as exc:
            return refuse(f"error: {type(exc).__name__}", f"Memory error: {type(exc).__name__}")
        record_tool_call(name, args=args, ok=True, memory_ids=[item.id], **lengths)
        return f"Noted: {item.id}"

    memory_note.__doc__ = (memory_note.__doc__ or "").replace("{cap}", str(width))
    memory_note.__name__ = name
    memory_note.__qualname__ = name
    return memory_note
