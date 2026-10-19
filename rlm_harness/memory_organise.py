"""Reading a memory store to organise it: the pure, model-free half. dspy-free, stdlib only.

What lives here is everything a summarising or organising pass needs that is not a model call, so
it can be tested and reused without dspy: the section map a long item is read through, the
language guard every prompt input passes, and the list of labels already in use that a summary is
shown so related items converge on one label. It also holds the local organising reads that need no
model: the best topic for an item (``suggest_topics``) and the bridges between topics
(``topic_graph``).
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .memory import _UNSPACED, MemoryStore
from .memory import _fold as fold
from .memory_vectors import NEIGHBOURS

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


#: How a suggestion is scored, and the least score worth offering. Lifted from the local filing
#: suggestions this design was taken from: a shared entity says more than a shared tag.
ENTITY_WEIGHT = 2
TAG_WEIGHT = 1
SUGGEST_THRESHOLD = 3

#: How many bridges each topic keeps in `topic_graph`. A global cap cut every link of a small topic
#: in the design this was taken from; a per-topic one keeps each topic connected.
BRIDGES_PER_TOPIC = 3

#: How many of the newest unlabelled items one ``suggest_topics`` call compares by similarity. Each
#: costs at most ``NEIGHBOURS + 1`` comparisons against the filed items, from one load of the vectors.
SIMILAR_CANDIDATES = 30


@dataclass(frozen=True)
class TopicSuggestion:
    """The best topic for an item in none, and why: the entities and tags it shares with members,
    or, for a suggestion by similarity, the member it is most like (``similar_to``) and the cosine
    (``similarity``)."""

    item_id: str
    topic_id: str
    score: int
    shared_entities: tuple[str, ...]
    shared_tags: tuple[str, ...]
    similar_to: str | None = None
    similarity: float | None = None


@dataclass(frozen=True)
class TopicGraph:
    """Pairs of live topics whose members name the same entities, and how many bridges each topic
    had beyond the ones kept."""

    bridges: tuple[tuple[str, str, tuple[str, ...]], ...]
    omitted: dict = field(default_factory=dict, hash=False)


def _current_sets(store: MemoryStore) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Each non-retired, labelled item's resolved entities and its tags, by item id, in one read."""
    entities: dict[str, set[str]] = {}
    tags: dict[str, set[str]] = {}
    for item_id, labels in store._labels_by_item().items():
        if labels.entities or labels.tags:
            entities[item_id] = set(labels.entities)
            tags[item_id] = set(labels.tags)
    return entities, tags


def suggest_topics(
    store: MemoryStore,
    *,
    limit: int = 50,
    landing: str | None = None,
    vectors: Any = None,
) -> list[TopicSuggestion]:
    """The best live topic for each item that is in none, strongest first. No model call.

    An item scores against a topic by what its current labels share with the topic's members:
    ``ENTITY_WEIGHT`` per entity (read through the alias table) and ``TAG_WEIGHT`` per tag, and
    only a score of ``SUGGEST_THRESHOLD`` or more is offered. A pair the item was dismissed from,
    or that a person took apart, is never offered. ``landing`` names a catch-all topic: being in it
    alone still counts as being in none, and it is never suggested.

    With ``vectors`` (a :class:`~rlm_harness.memory_vectors.MemoryVectors`), up to
    ``SIMILAR_CANDIDATES`` of the newest items that have no entities or tags yet are also offered
    the topic of the filed item they are most like (a mutual near neighbour above the model's
    threshold). Similarity is the weaker signal, so such a suggestion is scored at
    ``SUGGEST_THRESHOLD`` and sorts after any label match of the same score.
    """
    entities, tags = _current_sets(store)
    topics = [t for t in store.topics() if t.id != landing]
    members = {t.id: {i.id for i in store.topic_items(t.id)} for t in topics}
    blocked = store.dismissed() | store.removed_pairs()
    topic_entities = {tid: set().union(*(entities.get(i, set()) for i in ids)) for tid, ids in members.items()}
    topic_tags = {tid: set().union(*(tags.get(i, set()) for i in ids)) for tid, ids in members.items()}
    filed = {i for ids in members.values() for i in ids}
    out: list[TopicSuggestion] = []
    for item_id in entities:
        if item_id in filed:
            continue
        best: TopicSuggestion | None = None
        for topic in topics:
            if (item_id, topic.id) in blocked:
                continue
            shared = tuple(sorted(entities[item_id] & topic_entities[topic.id]))
            common = tuple(sorted(tags[item_id] & topic_tags[topic.id]))
            score = ENTITY_WEIGHT * len(shared) + TAG_WEIGHT * len(common)
            if score >= SUGGEST_THRESHOLD and (best is None or score > best.score):
                best = TopicSuggestion(item_id, topic.id, score, shared, common)
        if best is not None:
            out.append(best)
    if vectors is not None:
        out.extend(_by_similarity(store, vectors, members, blocked, filed, skip=set(entities)))
    out.sort(key=lambda s: (-s.score, s.similar_to is not None, s.item_id))
    return out[: max(1, int(limit))]


def _by_similarity(store: MemoryStore, vectors: Any, members: dict[str, set[str]],
                   blocked: set[tuple[str, str]], filed: set[str], *, skip: set[str]) -> list[TopicSuggestion]:
    topics_of: dict[str, list[str]] = {}
    for topic_id in sorted(members, key=lambda t: int(t[1:])):
        for item_id in members[topic_id]:
            topics_of.setdefault(item_id, []).append(topic_id)
    candidates = [item_id for item_id in sorted(store._live_ids(), key=lambda i: -int(i[1:]))
                  if item_id not in filed and item_id not in skip][:SIMILAR_CANDIDATES]
    if not candidates or not filed:
        return []
    # One load of the vectors for every candidate, never one per comparison.
    snapshot = vectors._snapshot()
    out: list[TopicSuggestion] = []
    for item_id in candidates:
        for other, score in snapshot.similar(item_id, k=NEIGHBOURS, among=filed):
            topic_id = next((t for t in topics_of.get(other, []) if (item_id, t) not in blocked), None)
            if topic_id is not None:
                out.append(TopicSuggestion(item_id, topic_id, SUGGEST_THRESHOLD, (), (),
                                           similar_to=other, similarity=score))
                break
    return out


def topic_graph(store: MemoryStore, *, per_topic: int = BRIDGES_PER_TOPIC) -> TopicGraph:
    """Pairs of live topics bridged by entities their members both name, strongest first. A bridge
    is kept when it is among the ``per_topic`` strongest of EITHER of its topics, so every topic keeps
    at least its strongest bridges and a busy one may keep more; ``omitted`` counts, per topic, the
    bridges dropped. No model call."""
    entities, _ = _current_sets(store)
    topics = [t.id for t in store.topics()]
    named = {tid: set().union(*(entities.get(i.id, set()) for i in store.topic_items(tid))) for tid in topics}
    pairs: list[tuple[str, str, tuple[str, ...]]] = []
    for index, a in enumerate(topics):
        for b in topics[index + 1:]:
            shared = tuple(sorted(named[a] & named[b]))
            if shared:
                pairs.append((a, b, shared))
    pairs.sort(key=lambda p: (-len(p[2]), p[0], p[1]))
    kept_per: dict[str, int] = {t: 0 for t in topics}
    seen_per: dict[str, int] = {t: 0 for t in topics}
    kept: list[tuple[str, str, tuple[str, ...]]] = []
    cap = max(1, int(per_topic))
    for a, b, shared in pairs:
        seen_per[a] += 1
        seen_per[b] += 1
        # A bridge is kept when it is among the strongest for EITHER end, so no topic loses all
        # its links to busier neighbours.
        if kept_per[a] < cap or kept_per[b] < cap:
            kept.append((a, b, shared))
            kept_per[a] += 1
            kept_per[b] += 1
    omitted = {t: seen_per[t] - kept_per[t] for t in topics if seen_per[t] > kept_per[t]}
    return TopicGraph(tuple(kept), omitted)



# ---- a finished run, as text a model can read ---------------------------------------------------

#: The longest rendering of one run, and the head kept of each source in it. Lifted from the plan's
#: review: a run's sources are what a later run needs, and their heads are enough to choose a quote
#: from; the whole text stays with the host, which checks the quote against it.
RUN_CHARS = 60_000
SOURCE_HEAD = 2_000

#: A tool name the kit gives a model-backed tool (``sub_lm.model_as_tool``): its result is another
#: model's words, so it is never host bytes, whatever the host lists as trusted.
_MODEL_TOOL_PREFIX = "model:"


@dataclass(frozen=True)
class RunSource:
    """One text a fact may quote: a tool result the trace holds, or evidence the host passed.
    ``trusted`` says whether a quote from it may be stored as ``observed``."""

    label: str
    text: str
    trusted: bool
    tool: str = ""


def run_sources(
    events: Iterable[dict],
    *,
    evidence: Mapping[str, Any] | None = None,
    trusted_tools: Iterable[str] = (),
) -> dict[str, RunSource]:
    """The texts in one run that a fact may quote, by label.

    A ``tool_call`` whose payload has a string ``result`` is ``step:<step_id>``; it is trusted only
    when its tool is in ``trusted_tools``, is not a model-backed ``model:`` tool and did not record
    ``ok=False``, because a recorded result may be another model's words and the kit cannot tell; a
    call with no ``step_id`` is left out. A host names a tool only when its result can never contain
    text the model supplied: a tool that echoes its arguments ("no page at <url>") would let a model
    quote its own words back as a source. Each ``evidence`` entry is ``evidence:<key>``: a string is trusted, and
    a ``(text, trusted)`` pair is trusted only when ``trusted is True``. REPL output never appears:
    the model wrote it.
    """
    allowed = {str(name) for name in trusted_tools}
    out: dict[str, RunSource] = {}
    for event in events:
        if not isinstance(event, dict) or event.get("type") != "tool_call":
            continue
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        result, tool = payload.get("result"), str(payload.get("tool", ""))
        # With no step id there is no label to tell two results apart, and one would replace another.
        if isinstance(result, str) and result.strip() and event.get("step_id") is not None:
            label = f"step:{event.get('step_id')}"
            # A failed call's result is usually an error message, the likeliest place a tool echoes
            # the arguments the model wrote; it is never trusted.
            trusted = (tool in allowed and not tool.startswith(_MODEL_TOOL_PREFIX)
                       and payload.get("ok") is not False)
            out[label] = RunSource(label, result, trusted, tool)
    for key, value in (evidence or {}).items():
        text, trusted = value if isinstance(value, tuple) and len(value) == 2 else (value, True)
        if isinstance(text, str) and text.strip():
            label = f"evidence:{key}"
            out[label] = RunSource(label, text, trusted is True)   # "false" is not trust
    return out


def _clip(text: str, chars: int) -> str:
    """``text`` in at most ``chars`` characters, the note saying what was cut included."""
    if len(text) <= chars:
        return text
    keep = max(0, chars - len(f"... [{len(text)} more characters]"))
    note = f"... [{len(text) - keep} more characters]"
    return (text[:keep] + note) if keep else text[:chars]


def _compact(value: Any, chars: int) -> str:
    try:
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = repr(value)
    return _clip(text, chars)


def render_run(
    events: Iterable[dict],
    *,
    evidence: Mapping[str, Any] | None = None,
    trusted_tools: Iterable[str] = (),
    max_chars: int = RUN_CHARS,
) -> str:
    """One finished run as compact text: its meta, the final output, each piece of evidence, and
    each tool call with its arguments and the head of any recorded result. Every source carries the
    label a fact names it by and whether it is trusted (:func:`run_sources`). The model's own REPL
    turns are left out. dspy-free.

    Cut to ``max_chars``, and the tool calls are cut first: the meta, the output and the evidence
    come before them, since the evidence the host passed is usually the only route to ``observed``
    and a run with many tool calls would otherwise push it out of view. Evidence past the budget on
    its own is cut too; the model then cannot see it, though a quote from it is still checked
    against the full text.
    """
    events = [e for e in events if isinstance(e, dict)]
    sources = run_sources(events, evidence=evidence, trusted_tools=trusted_tools)
    run_id = next((e.get("run_id") for e in events if e.get("run_id")), None)
    head: list[str] = [f"# run {run_id or '(unknown)'}"]
    calls: list[str] = []
    for event in events:
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        kind = event.get("type")
        if kind == "run_start" and payload.get("meta"):
            head.append(f"meta: {_compact(payload['meta'], SOURCE_HEAD)}")
        elif kind == "tool_call":
            calls.append(f"\n## tool call {payload.get('tool', '?')} (ok={payload.get('ok')})")
            if payload.get("args"):
                calls.append(f"args: {_compact(payload['args'], SOURCE_HEAD)}")
            label = f"step:{event.get('step_id')}"
            if event.get("step_id") is not None and label in sources:
                source = sources[label]
                calls.append(f"[{label}] ({'trusted' if source.trusted else 'untrusted'}) result:")
                calls.append(_clip(source.text, SOURCE_HEAD))
        elif kind == "result":
            head.append(f"\n## final output\n{_compact(payload.get('output'), SOURCE_HEAD * 2)}")
    for label, source in sources.items():
        if label.startswith("evidence:"):
            head.append(f"\n## evidence\n[{label}] ({'trusted' if source.trusted else 'untrusted'}):")
            head.append(_clip(source.text, SOURCE_HEAD))
    budget = max(1, int(max_chars))
    first = _clip("\n".join(head), budget)
    room = budget - len(first) - 1
    if not calls or room <= 0:
        return first
    return first + "\n" + _clip("\n".join(calls), room)
