"""Reading a memory store to organise it: the pure, model-free half. dspy-free, stdlib only.

What lives here is everything a summarising or organising pass needs that is not a model call, so
it can be tested and reused without dspy: the section map a long item is read through, the
language guard every prompt input passes, and the list of labels already in use that a summary is
shown so related items converge on one label.
"""

from __future__ import annotations

import re
import unicodedata

from .memory import _UNSPACED, MemoryStore
from .memory import _fold as fold

#: Scripts written without spaces between words that the index does NOT split (it only pairs the
#: CJK class in `memory._UNSPACED`): Thai, Lao, Myanmar and Khmer. A name in one of them cannot
#: be matched as a whole word inside running text, so grounding reads it as a substring.
_SPACELESS = re.compile(r"[\u0e00-\u0eff\u1000-\u109f\u1780-\u17ff]")

#: The most lines a section map has, however long the text. Lifted from the long-document
#: summariser this design was taken from; past it, adjacent parts share a line.
MAP_LINES = 160

#: How many of the labels already in use a summary is shown: the most used first, so the listing
#: stays a few hundred words however large the store grows.
KNOWN_TAGS = 80
KNOWN_ENTITIES = 60

#: The shortest chunk a long paragraph is cut into, so a short text is never mapped character by
#: character.
_MIN_CHUNK = 400

#: The longest language name passed into a prompt.
_MAX_LANGUAGE_CHARS = 40

_OPENING_CHARS = 90


def clean_language(raw: object) -> str:
    """A language name safe to splice into an instruction: one line, printable, bounded.

    Not a closed-set check: human language names have no enumerable set, so this bounds the value
    rather than refusing unknown ones. Returns ``""`` for anything empty, read as "no preference".
    """
    if not raw:
        return ""
    collapsed = " ".join(str(raw).split())
    return "".join(c for c in collapsed if c.isprintable())[:_MAX_LANGUAGE_CHARS].strip()


def _paragraphs(text: str) -> list[tuple[int, int]]:
    """``(start, end)`` offsets of the text's paragraphs, split on blank lines."""
    parts: list[tuple[int, int]] = []
    start = None
    offset = 0
    for line in text.splitlines(keepends=True):
        if line.strip():
            if start is None:
                start = offset
        elif start is not None:
            parts.append((start, offset))
            start = None
        offset += len(line)
    if start is not None:
        parts.append((start, offset))
    return parts


def _parts(text: str) -> list[tuple[int, int]]:
    """Paragraphs, with any paragraph longer than one map line's share cut into character chunks.

    Text with no blank lines (single-newline prose, OCR output, a scraped page) is one paragraph,
    so splitting on blank lines alone would map a 90,000-character text to a single line. Each cut
    snaps back to the last newline or space inside the chunk, so a word is never split.
    """
    share = max(_MIN_CHUNK, -(-len(text) // MAP_LINES))
    parts: list[tuple[int, int]] = []
    for start, end in _paragraphs(text):
        while end - start > share:
            cut = start + share
            snap = max(text.rfind("\n", start + 1, cut), text.rfind(" ", start + 1, cut))
            cut = snap + 1 if snap > start else cut
            parts.append((start, cut))
            start = cut
        parts.append((start, end))
    return parts


def section_map(text: str) -> str:
    """A map of a long text for a model that reads it in a REPL: one line per part, at most
    ``MAP_LINES``, each with its character offsets, size and opening words.

    The offsets are what the model slices with (``text[a:b]``); the opening words are for deciding
    where to look, and are the text's own, cut short. An empty text maps to ``""``.
    """
    parts = _parts(text or "")
    if not parts:
        return ""
    per_line = max(1, -(-len(parts) // MAP_LINES))
    lines = [f"{len(text)} characters in {len(parts)} parts."]
    for i in range(0, len(parts), per_line):
        group = parts[i : i + per_line]
        start, end = group[0][0], group[-1][1]
        opening = " ".join(text[start:end].split())[:_OPENING_CHARS]
        lines.append(f"chars {start}-{end} ({end - start} chars): {opening}")
    return "\n".join(lines)


def contains_name(where: str, name: str) -> bool:
    """Whether the folded ``name`` occurs in the folded ``where`` as a whole word.

    A name in a script written without spaces (Chinese, Japanese, Korean, Thai, Lao, Myanmar,
    Khmer) has no word boundaries to respect and is a plain substring; any other name must not
    touch a letter or digit on either side, so ``ai`` is not found inside ``said``.
    """
    if not name:
        return False
    if _UNSPACED.search(name) or _SPACELESS.search(name):
        return name in where
    # The word rule `memory._words` uses (letters, digits AND combining marks), not Python's `\w`,
    # which leaves marks out and so took the Hindi fragment `हिन्द` inside `हिन्दी` for a whole word.
    start = where.find(name)
    while start != -1:
        end = start + len(name)
        before = where[start - 1] if start else ""
        after = where[end] if end < len(where) else ""
        if not _wordlike(before) and not _wordlike(after):
            return True
        start = where.find(name, start + 1)
    return False


def _wordlike(char: str) -> bool:
    return bool(char) and unicodedata.category(char)[0] in "LNM"


def known_listing(store: MemoryStore) -> tuple[str, dict[str, str]]:
    """The labels other items already use, as the text a summariser is shown, and the folded
    ENTITY names among them.

    Shown what is in use, a summary writes the same label when it means the same thing, so two items
    on one subject do not end up as ``databases`` and ``database systems``. The folded entity names
    are what entity grounding accepts besides the item's own text: a summary told to reuse a known
    name may write it in another script than the text uses. They map to the name as it is in use,
    so an entity the model wrote as ``postgresql`` is stored as the known ``PostgreSQL``. A live alias
    of a name in use is accepted too and kept as written (its value is its in-use spelling, or ""
    for "keep the model's"), so readers resolve it and an unmerge undoes it. Aliases are accepted,
    not shown: the text lists each name in use once. Tags are not
    in that map; a tag another item carries is no evidence that this item names it as an entity.
    """
    tags = store.label_counts("tags", limit=KNOWN_TAGS)
    entities = store.label_counts("entities", limit=KNOWN_ENTITIES)
    # Most used first, and the first spelling of a name wins: a plain dict comprehension kept the
    # LAST, so the rarest spelling became the one every new label converged on. The model is shown
    # only that spelling, never two case variants of one name side by side.
    known: dict[str, str] = {}
    for name, _ in entities:
        known.setdefault(fold(name), name)
    lines = []
    if tags:
        lines.append("Tags in use: " + ", ".join(name for name, _ in tags))
    if known:
        lines.append("Entities in use: " + ", ".join(known.values()))
    # An alias grounds too, while some item still names its group (the canonical itself or any of
    # its aliases; an alias left behind after its whole group was purged grounds nothing), and is
    # kept AS WRITTEN: readers resolve it at read time, so an unmerge restores every label written
    # while the merge was live. Writing the canonical instead froze a wrong merge into those labels
    # for good. The most used spelling wins, as above; "" means "keep the model's spelling".
    written: dict[str, str] = {}
    for name in store.entity_names():
        written.setdefault(fold(name), name)
    for alias, canonical in store.aliases().items():
        if fold(canonical) in known:
            known.setdefault(alias, written.get(alias, ""))
    return "\n".join(lines), known
