"""The model-backed half of organising a memory: label steps, and the pass that runs them.

Store methods never call a model. Everything here does, and only when the host calls a pass.

**Two kinds of model call, chosen by cost.** A short item is labelled by ONE ``dspy.Predict`` call
that reads all of it (:class:`SummariseShort`). A text past :data:`~rlm_harness.memory.SHORT_LIMIT`
is an RLM run instead (:class:`DistillLongItem`): the whole text is a REPL variable and the prompt
carries a section map of it, so the model reads where the map points rather than a prefix. Only
the long path needs the Deno sandbox. ``optimize.compile_task`` compiles :class:`DistillLongItem`;
a ``Predict`` step is compiled with ``dspy.GEPA`` directly.

**The ``run=`` contract.** A pass takes ``run: Callable[[dict], Awaitable[dict]]``. It receives a
JSON-serialisable payload, the step's input fields plus ``"step"`` and ``"run_id"``, and returns a
JSON-serialisable dict of the step's output model. The default, :func:`arun_step`, runs the step
in-process. :func:`run_step` is its synchronous, picklable twin for a child process, and
:func:`subprocess_step_runner` builds the ``run`` a consumer that isolates model calls passes::

    report = await distill_pending(store, run=subprocess_step_runner(timeout_s=300))

Whatever comes back is sanitised by the pass, whichever ``run`` produced it.

**What a label is allowed to say.** Coordinate markers (``[[SRC:...]]``) are stripped. An entity is
kept only when it is grounded: its name occurs in the item as a whole word, or as a substring for a
script written without spaces (for a long item, in the span its quote was found at, the quote itself
checked by ``tools.grounding._find_span``), or it is one of the entity names already in use that the
step was shown. The second case is what lets a text written in one script carry the name other items
use in another.

**Tracing is per item and opt-in.** With ``recorder_for``, each item's call runs inside ``with
recorder_for(item_id):``, so each item is its own run, and the labels it writes carry that run's id.
What the run records depends on the step: a long item's RLM steps are recorded when the step runs in
this process; a short item's single ``Predict`` call records no event, so its run holds only
``run_start`` and ``run_end``. A recorder the caller entered around the whole pass would put every
item in one run; nothing here relies on one.

There is no claim mechanism: two passes running at once may label one item twice, or align one batch
of names twice. That is harmless, since the latest label wins and a second merge of one name is
refused, but it costs a call, so a consumer keeps one pass of each kind at a time.
"""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import json
import logging
import re
import sqlite3
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from .memory import SHORT_LIMIT, MemoryStore
from .memory_organise import clean_language, contains_name, fold, known_listing, section_map
from .runtime import get_config
from .task import RLMTask
from .tools.grounding import _find_span
from .tools.validation import make_schema_validator

logger = logging.getLogger(__name__)

#: The most items one pass takes on, whatever ``limit`` asks for.
MAX_PASS_ITEMS = 500

#: Only the kit's own coordinate markers are stripped, never an arbitrary ``[[...]]``.
_MARKER = re.compile(r"\[\[SRC:[^\]\n]*\]\]")


class ShortLabels(BaseModel):
    """What :class:`SummariseShort` returns."""

    title: str = ""
    summary: str = ""
    tags: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)


class EntityMention(BaseModel):
    """An entity a long item is about, with a quote copied from the text that names it."""

    name: str
    quote: str = ""


class LongLabels(BaseModel):
    """What :class:`DistillLongItem` SUBMITs."""

    title: str = ""
    summary: str = ""
    tags: list[str] = Field(default_factory=list)
    entities: list[EntityMention] = Field(default_factory=list)


_LABEL_RULES = """\
Write:
- title: a short, specific label naming the actual subject, not the kind of text. Never "Article",
  "Document" or "Notes".
- summary: two or three sentences on what this says and why it is worth keeping. Start with the
  subject itself. No preamble in any language ("This document...", "This text describes...",
  "本文介紹"), no closing "In summary", and no dashes; use a comma or a full stop.
- tags: up to 6 lowercase topic labels: broad subjects someone would search by and other items
  could share (databases, security, cooking), not phrases unique to this text.
- entities: up to 6 specific things this is ABOUT: people, organisations, products, places, and
  named concepts such as a method, a standard, a theory or a component (PostgreSQL, the TCP
  handshake). Narrower than a tag, each one something another item could name too. Not every name
  that appears.

Rules:
- Write title and summary in the requested `language`. If none is given, use the text's own.
  Never translate a proper noun that has no established translation.
- `known` lists the tags and entities other items already use. When one of them means what you
  would write, write it exactly as it appears there, so related items connect. Add a new one only
  when none fits, and never use a known one just because it is common. A new tag is written in the
  requested language when one is given.
- Say only what the text supports. If it is too short or too garbled to label, give an empty
  summary rather than inventing one."""

_SHORT_INSTRUCTIONS = (
    "Label one item kept in long-term memory so it can be found again later, by someone who no "
    "longer remembers the words it used.\n\n" + _LABEL_RULES
)

_LONG_INSTRUCTIONS = f"""\
Label ONE long item kept in long-term memory so it can be found again later, by someone who no
longer remembers the words it used.

`text` is the whole item as a REPL variable. `section_map` lists its parts in order, each with its
character offsets (`chars a-b`), its size and its opening words. Use the map to decide where to
read, then read those parts with Python slicing (`text[a:b]`); never print the whole text. The
summary is about the whole item, not only its opening.

{_LABEL_RULES}
- For each entity give `name` and `quote`: at least 20 characters copied VERBATIM from the text,
  naming that entity. An entity you cannot quote is left out.

Before you SUBMIT, pass your answer as a JSON string to `validate_longlabels`, and SUBMIT on a
LATER turn, only after it reports "Validation successful.". Never SUBMIT in the same turn as the
validation call."""


class SummariseShort:
    """Label a short item in one ``dspy.Predict`` call that reads all of it."""

    async def arun(self, *, text: str = "", language: str = "", known: str = "") -> ShortLabels:
        import dspy

        predictor = dspy.Predict(dspy.Signature(
            "text: str, language: str, known: str -> title: str, summary: str, tags: list[str], "
            "entities: list[str]",
            _SHORT_INSTRUCTIONS,
        ))
        result = await predictor.acall(text=(text or "")[:SHORT_LIMIT], language=language or "",
                                       known=known or "")
        return ShortLabels(
            title=str(getattr(result, "title", "") or ""),
            summary=str(getattr(result, "summary", "") or ""),
            tags=[str(t) for t in (getattr(result, "tags", None) or []) if isinstance(t, str)],
            entities=[str(e) for e in (getattr(result, "entities", None) or []) if isinstance(e, str)],
        )


class DistillLongItem(RLMTask):
    """Label one long item from a section map plus the whole text in the REPL.

    Without ``config`` the active one is used with ``max_iterations`` set to :attr:`STEPS`; a
    caller's own ``config`` is used as given, its turn budget included.
    """

    #: A turn budget for one item, lifted from the long-document summariser this was taken from,
    #: where a 230,000-character document took 9 turns.
    STEPS = 15

    # Not `labels`: `dspy.Prediction` inherits a `labels()` method from `dspy.Example`, and an
    # output field of that name reads back as the method, never as the answer.
    signature = "text: str, section_map: str, language: str, known: str -> label_set: LongLabels"
    output_field = "label_set"
    output_model = LongLabels
    instructions = _LONG_INSTRUCTIONS

    def __init__(self, *, config: Any = None, **kw: Any) -> None:
        # A caller's own `tools=` replaces the validator the instructions name; keep one named
        # `validate_longlabels` in it.
        if config is None:
            config = dataclasses.replace(get_config(), max_iterations=self.STEPS)
        kw.setdefault("tools", [make_schema_validator(LongLabels)])
        super().__init__(config=config, **kw)


class EntityMerge(BaseModel):
    """Two names for one thing: ``alias`` is read as ``canonical`` from now on."""

    alias: str
    canonical: str


class EntityMerges(BaseModel):
    """What :class:`AlignEntities` SUBMITs."""

    merges: list[EntityMerge] = Field(default_factory=list)


_ALIGN_INSTRUCTIONS = """\
Decide which of the NEW entity names are the same entity as one already KNOWN, or as another new
name.

`known` is a REPL variable: every entity name already in use, one per line, with how many items
name it. Search it with Python (`.find()`, splitting lines); do not print all of it. `new_names` is
a JSON list of names written by recent labels.

Merge two names ONLY when they name the same thing: a spelling or capitalisation variant, a
translation or transliteration (Tokyo and 東京), an abbreviation and its expansion (k8s and
Kubernetes), or a name with and without a title or suffix. Never merge things that are merely
related (a database and its query language), a part and its whole, a product and its maker, or a
person and their work. When unsure, do not merge: an unmerged pair costs nothing, a wrong merge
hides a real distinction.

For each merge give `alias` (the new name) and `canonical` (the name to keep, usually the known one,
or the more common spelling), each copied EXACTLY as written in `known` or `new_names`. When nothing
should be merged, that is still an answer given from code: `SUBMIT(merges={"merges": []})`.

Before you SUBMIT, pass your answer as a JSON string to `validate_entitymerges`, and SUBMIT on a
LATER turn, only after it reports "Validation successful.". Never SUBMIT in the same turn as the
validation call."""


class AlignEntities(RLMTask):
    """Merge newly written entity names into the ones already known.

    Without ``config`` the active one is used with ``max_iterations`` set to :attr:`STEPS`.
    """

    #: A lookup, a decision and a validated SUBMIT on a later turn: three turns when it goes well;
    #: eight leaves room for a correction.
    STEPS = 8

    signature = "known: str, new_names: str -> merges: EntityMerges"
    output_field = "merges"
    output_model = EntityMerges
    instructions = _ALIGN_INSTRUCTIONS

    def __init__(self, *, config: Any = None, **kw: Any) -> None:
        if config is None:
            config = dataclasses.replace(get_config(), max_iterations=self.STEPS)
        kw.setdefault("tools", [make_schema_validator(EntityMerges)])
        super().__init__(config=config, **kw)


_STEPS: dict[str, tuple[Callable[[], Any], tuple[str, ...]]] = {
    "summarise_short": (SummariseShort, ("text", "language", "known")),
    "distill_long": (DistillLongItem, ("text", "section_map", "language", "known")),
    "align_entities": (AlignEntities, ("known", "new_names")),
}


async def arun_step(payload: dict) -> dict:
    """Run the step ``payload["step"]`` names, in this process, and return its output as a dict."""
    step = payload.get("step")
    if step not in _STEPS:
        raise ValueError(f"unknown memory step {step!r}; known: {sorted(_STEPS)}")
    factory, fields = _STEPS[step]
    result = await factory().arun(**{name: payload.get(name, "") for name in fields})
    return result.model_dump()


def run_step(payload: dict) -> dict:
    """:func:`arun_step` for a child process: synchronous and picklable.

    ``payload["configure_from"]``, when present, is an :class:`~rlm_harness.RLMConfig` or a dict of
    its fields; the child calls ``configure`` with it first, since a spawned process starts
    unconfigured. Must not be called from inside a running event loop.
    """
    settings = payload.get("configure_from")
    if settings is not None:
        from .config import RLMConfig
        from .runtime import configure

        if not isinstance(settings, RLMConfig):
            settings = dict(settings)
            # `dataclasses.asdict` flattens the nested ContainerConfig into a dict, and RLMConfig
            # would keep that dict in a field every reader expects to be a ContainerConfig.
            if isinstance(settings.get("container"), dict):
                from .config import ContainerConfig

                settings["container"] = ContainerConfig(**settings["container"])
            settings = RLMConfig(**settings)
        configure(settings)
    return asyncio.run(arun_step(payload))


def subprocess_step_runner(
    *,
    timeout_s: float | None = 300.0,
    config: Any = None,
    max_memory_mb: int | None = None,
) -> Callable[[dict], Awaitable[dict]]:
    """A ``run=`` that executes each step in a fresh process through :func:`run_in_subprocess`.

    The child starts unconfigured (``run_in_subprocess`` always spawns), so it is configured with
    ``config``, an :class:`~rlm_harness.RLMConfig`, which defaults to the one this process was
    configured with when the runner is built. The object itself crosses the process boundary,
    since it pickles; an LM injected through ``configure(main_lm=...)`` does not, so the child
    builds its LMs from the config's model strings.

    ``run_in_subprocess`` takes a zero-argument, picklable factory and keyword-only options, so
    the step is bound with ``functools.partial`` and the blocking call runs in a thread. A step's
    own exception is re-raised here. Two limits follow from the isolation:

    - Cancelling the awaiting task does not stop the child: it runs, and bills, until it finishes
      or ``timeout_s`` passes. ``should_stop`` is still checked between items.
    - A long item's RLM steps run in the child, so they do not reach a ``recorder_for`` trace,
      which then holds only that item's ``run_start`` and ``run_end``. A short item's single
      ``Predict`` call is never recorded, in this process or another.
    """
    from .isolation import run_in_subprocess

    if config is None:
        try:
            config = get_config()
        except Exception:   # not configured here either: the child gets nothing to configure with
            config = None

    async def run(payload: dict) -> dict:
        sent = dict(payload)
        if config is not None:
            sent["configure_from"] = config
        return await asyncio.to_thread(
            run_in_subprocess, functools.partial(run_step, sent),
            timeout_s=timeout_s, max_memory_mb=max_memory_mb,
        )

    return run


@dataclass
class DistillReport:
    """What one :func:`distill_pending` pass did. ``errors`` maps an item id to its failure."""

    labelled: list[str] = field(default_factory=list)
    #: Labelled items whose summary came back empty: the model judged the text too thin or too
    #: garbled to summarise. They count as labelled too, so they are not asked about again.
    skipped_empty: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    stopped: bool = False


def _clean_text(value: Any) -> str:
    return _MARKER.sub("", value) if isinstance(value, str) else ""


def _clean_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [_clean_text(v) for v in values if isinstance(v, str) and _clean_text(v).strip()]


def _grounded_entities(raw: Any, text: str, known: dict[str, str], *, long: bool) -> list[str]:
    """The entity names worth keeping: named in the item (or in a long item's quoted span), or one
    of the names in use the step was shown, stored in the spelling already in use."""
    kept: list[str] = []
    folded_text = fold(text)
    for entry in raw if isinstance(raw, list) else []:
        if long:
            if not isinstance(entry, dict):
                continue
            name, quote = _clean_text(entry.get("name")), entry.get("quote")
            span = _find_span(text, quote) if isinstance(quote, str) else None
            if span is None:
                continue
            where = fold(text[span[0]:span[1]])
        else:
            name = _clean_text(entry)
            where = folded_text
        key = fold(name)
        if key in known:
            kept.append(known[key] or name)
        elif key and contains_name(where, key):
            kept.append(name)
    return kept


async def distill_pending(
    store: MemoryStore,
    *,
    limit: int = 20,
    language: str | None = None,
    item_ids: Iterable[str] | None = None,
    defer_long: bool = False,
    run: Callable[[dict], Awaitable[dict]] | None = None,
    recorder_for: Callable[[str], Any] | None = None,
    should_stop: Callable[[], bool] | None = None,
    on_progress: Callable[[dict], None] | None = None,
    known_refresh: int = 10,
    max_failures: int = 3,
) -> DistillReport:
    """Label up to ``limit`` items still waiting for a summary (``store.unlabelled``), newest first.

    Nothing runs unless the host calls this; ``limit`` is clamped to ``[1, 500]``. Each item is one
    model call: :class:`SummariseShort` for a text up to ``SHORT_LIMIT`` characters, otherwise
    :class:`DistillLongItem`, which ``defer_long`` skips. A success writes one ``inferred`` label set
    in one transaction; a failure records a failure row and the item stays queued, until it has
    failed ``max_failures`` times (``item_ids`` still reaches it). ``should_stop`` is checked before
    every call and ``on_progress`` receives ``{"item_id", "ok", "error"}`` after every item. A
    database error while writing (a lock held past the store's timeout) fails that item without
    counting toward ``max_failures``, and never ends the pass. The store calls are synchronous and
    run on the event loop; each is milliseconds, but one can wait out a long purge's lock.
    ``language`` passes through ``clean_language``; the labels already in use are re-read every
    ``known_refresh`` items.
    """
    run = run or arun_step
    report = DistillReport()
    lang = clean_language(language)
    items = store.unlabelled(
        limit=max(1, min(int(limit), MAX_PASS_ITEMS)), item_ids=item_ids, max_failures=max_failures,
        max_chars=SHORT_LIMIT if defer_long else None,
    )
    known_text: str = ""
    known_names: dict[str, str] = {}
    for n, item in enumerate(items):
        if should_stop is not None and should_stop():
            report.stopped = True
            break
        if n % max(1, int(known_refresh)) == 0:
            known_text, known_names = known_listing(store)
        long = len(item.text) > SHORT_LIMIT
        payload: dict[str, Any] = {"text": item.text, "language": lang, "known": known_text}
        payload.update({"step": "distill_long", "section_map": section_map(item.text)} if long
                       else {"step": "summarise_short"})
        recorder = recorder_for(item.id) if recorder_for is not None else None
        run_id = getattr(recorder, "run_id", None)
        payload["run_id"] = run_id
        error = ""
        try:
            if recorder is not None:
                with recorder:
                    result = await run(payload)
            else:
                result = await run(payload)
            if not isinstance(result, dict):
                raise TypeError(f"run returned {type(result).__name__}, not a dict")
            summary = _clean_text(result.get("summary"))
            store.label(
                item.id,
                title=_clean_text(result.get("title")),
                summary=summary,
                tags=_clean_list(result.get("tags")),
                entities=_grounded_entities(result.get("entities"), item.text, known_names, long=long),
                provenance="inferred",
                run_id=run_id,
            )
            report.labelled.append(item.id)
            if not summary.strip():
                report.skipped_empty.append(item.id)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"[:2000]
            report.failed.append(item.id)
            report.errors[item.id] = error
            # A database error (a lock held past the timeout, say) is the store's, not the model's,
            # so it is not counted toward `max_failures`; and recording a failure can hit the same
            # lock, which must not end the pass and lose this report.
            if not isinstance(exc, sqlite3.Error):
                try:
                    store.note_distill_failure(item.id, error)
                except KeyError:   # purged while it was being labelled
                    pass
                except sqlite3.Error as note_exc:
                    logger.warning("could not record the distill failure of %s: %s", item.id, note_exc)
        if on_progress is not None:
            on_progress({"item_id": item.id, "ok": not error, "error": error})
    return report


def _dependency_order(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """``pairs`` with every pair whose alias is another pair's canonical moved ahead of that pair.

    For a chained answer (``k8s -> kube``, ``kube -> Kubernetes``) the second link goes first, and
    ``store.merge`` then follows ``kube`` as a live alias when the first is applied. The answer is
    never resolved here, ahead of the store: doing that let a link the store refuses (a name a
    person keeps apart) be skipped over, merging past it. A cycle falls back to the given order.
    """
    by_alias: dict[str, list[int]] = {}
    for index, (alias, _) in enumerate(pairs):
        by_alias.setdefault(fold(alias), []).append(index)
    ordered: list[int] = []
    state: dict[int, str] = {}

    def visit(index: int) -> None:
        if state.get(index):
            return   # done, or on the current path: a cycle, left in the given order
        state[index] = "visiting"
        for before in by_alias.get(fold(pairs[index][1]), []):
            visit(before)
        state[index] = "done"
        ordered.append(index)

    for index in range(len(pairs)):
        visit(index)
    return [pairs[index] for index in ordered]


@dataclass
class AlignReport:
    """What one :func:`align_pending` pass did.

    ``asked`` are the names the model was shown; ``accepted`` the merges recorded, as
    ``(alias, canonical)``; ``refused`` the merges the store turned down, as
    ``(alias, canonical, reason)``. ``error`` is set when the call itself failed, in which case
    nothing was marked seen and the same names are asked about next time. ``stopped`` is true when
    ``should_stop`` ended the pass before the call.
    """

    asked: list[str] = field(default_factory=list)
    accepted: list[tuple[str, str]] = field(default_factory=list)
    refused: list[tuple[str, str, str]] = field(default_factory=list)
    error: str = ""
    stopped: bool = False


async def align_pending(
    store: MemoryStore,
    *,
    limit: int = 100,
    run: Callable[[dict], Awaitable[dict]] | None = None,
    recorder_for: Callable[[str], Any] | None = None,
    should_stop: Callable[[], bool] | None = None,
    on_progress: Callable[[dict], None] | None = None,
) -> AlignReport:
    """Ask once which of the entity names not yet considered mean the same thing as another.

    Nothing runs unless the host calls this. Up to ``limit`` unseen names (clamped to
    ``[1, 500]``) go to ONE :class:`AlignEntities` call, with every name in use and its count in
    the REPL. With no unseen names, no call is made. Each merge it returns goes through
    ``store.merge(..., provenance="inferred", batch=...)``, so a model can only merge a name it was
    asked about and never overrides a person's decision; a refused merge is reported with the
    store's reason. A chained answer is applied link by link, so the store judges every link. The batch is then
    marked seen, merged or not, unless the call itself failed; a name whose merge hit a database
    error stays unseen, so it is asked about again.
    ``recorder_for("align")`` traces the call; the alias rows carry its run id. ``should_stop`` is
    checked before the call and ``on_progress`` receives ``{"batch", "ok", "error"}`` after it.
    """
    run = run or arun_step
    report = AlignReport()
    batch = store.unseen_entities(limit=max(1, min(int(limit), MAX_PASS_ITEMS)))
    if not batch:
        return report
    if should_stop is not None and should_stop():
        report.stopped = True
        return report
    report.asked = list(batch)
    known = "\n".join(f"{name}\t{count}" for name, count in store.entity_names().items())
    recorder = recorder_for("align") if recorder_for is not None else None
    run_id = getattr(recorder, "run_id", None)
    payload = {"step": "align_entities", "known": known, "new_names": json.dumps(batch),
               "run_id": run_id}
    try:
        if recorder is not None:
            with recorder:
                result = await run(payload)
        else:
            result = await run(payload)
        if not isinstance(result, dict):
            raise TypeError(f"run returned {type(result).__name__}, not a dict")
    except Exception as exc:
        report.error = f"{type(exc).__name__}: {exc}"[:2000]
        if on_progress is not None:
            on_progress({"batch": len(batch), "ok": False, "error": report.error})
        return report
    merges = result.get("merges")
    pairs: list[tuple[str, str]] = []
    for entry in merges if isinstance(merges, list) else []:
        alias = entry.get("alias") if isinstance(entry, dict) else None
        canonical = entry.get("canonical") if isinstance(entry, dict) else None
        if not isinstance(alias, str) or not isinstance(canonical, str):
            report.refused.append((str(alias), str(canonical), "not a pair of names"))
            continue
        pairs.append((alias, canonical))
    failed_names: set[str] = set()
    for alias, canonical in _dependency_order(pairs):
        try:
            target = store.merge(alias, canonical, provenance="inferred", batch=batch, run_id=run_id)
            report.accepted.append((alias, target))
        except ValueError as exc:
            report.refused.append((alias, canonical, str(exc)))
        except sqlite3.Error as exc:
            # The store failed, not the answer: this name is asked about again next time.
            failed_names.add(fold(alias))
            report.refused.append((alias, canonical, f"database error: {type(exc).__name__}"))
    try:
        store.mark_seen([name for name in batch if fold(name) not in failed_names])
    except sqlite3.Error as exc:
        logger.warning("could not mark %d aligned names seen: %s", len(batch), exc)
    if on_progress is not None:
        on_progress({"batch": len(batch), "ok": True, "error": ""})
    return report
