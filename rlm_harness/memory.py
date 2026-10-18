"""``MemoryStore``: a built-in, provenance-typed memory that outlives a run. dspy-free, stdlib only.

A run's REPL forgets everything when it ends. This module is the part that does not: an SQLite
file of short texts that a host puts things into and a model reads from, through the tools in
:mod:`rlm_harness.tools.memory`. It needs no extra and no service.

**Every item says where it came from, and that label decides what a reader may do with it.**
``PROVENANCES`` is a closed set of three:

- ``observed``: text the HOST took from a tool output or an external source.
- ``said``: text a person wrote.
- ``inferred``: text a model wrote, which covers a note, a summary and a revision.

The kit cannot verify which of the first two a host-side write really is; the host's own code path
assigns it. What the kit DOES guarantee is the third: the only write path a model has, the note
tool, fixes the provenance to ``inferred`` and takes no argument that could change it. That
matters because a model's own words are the cheapest thing to get wrong and the easiest thing to
plant. A note written after reading an untrusted page is replayed into every later run that
searches for it, which is a PERSISTENT prompt-injection path, and the ``inferred`` label is the
only thing that lets a later reader discount it rather than cite it.

**Nothing written is ever changed or deleted.** Two ways exist to change what a reader sees, and
neither touches the original row:

- :meth:`MemoryStore.revise` adds a NEW item that points at the old one (``revises``).
- :meth:`MemoryStore.retire` sets ``valid_to`` once, from NULL to a time. It is the only UPDATE in
  this module.

SQLite triggers refuse a stray ``UPDATE``, ``DELETE`` or ``INSERT OR REPLACE`` against ``items``,
so a host's own SQL cannot quietly rewrite history either. They are not a security boundary
against someone with the file: ``DROP TRIGGER`` still works. The triggers protect ``items``, the
source of truth. The search index is a contentless FTS5 table, which refuses a plain ``DELETE`` or
``UPDATE``, but FTS5's own ``'delete'`` and ``'delete-all'`` commands and a direct ``INSERT`` can
still change what it matches; that can make search miss an item or match the wrong one, and never
changes an item.

**Chinese, Japanese and Korean are indexed without a dictionary.** That covers kana, the CJK
ideograph blocks including extensions A to G with the marks ``々 〆 〇 〻``, and Hangul syllables
and both jamo blocks. FTS5's ``unicode61`` tokenizer treats a whole unspaced run as one token, so
``記憶`` inside ``睡眠與記憶`` would never match. Each such run is indexed as its overlapping
character pairs AND its single characters, computed in Python because Python's ``sqlite3`` cannot
register an FTS5 tokenizer. A query keeps only the
pairs of a run, or the character itself when the run is one character long, so a two-character
word matches inside a longer run and a one-character query still finds something. Text is NFKC
normalised first, so a full-width ``ＰＤＦ`` and ``PDF`` are one word.

**One connection per operation.** That makes a store safe to share between the thread dspy calls a
tool on and a host thread, with SQLite serialising the writers. Writes take ``BEGIN IMMEDIATE`` so
two writers queue rather than fail halfway. The file runs in WAL mode, which needs shared memory
and is therefore unreliable on a network or VM-shared filesystem; keep the store on a local disk.

**The schema changes only by appending a step to ``_MIGRATIONS``**, counted in
``PRAGMA user_version``. A shipped step is never edited, because a database that already ran it
will not run it again.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import sqlite3
import time
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

#: The closed set of provenance labels, in the order of how much a reader may trust them.
PROVENANCES = ("observed", "said", "inferred")

Provenance = Literal["observed", "said", "inferred"]

#: Weight of a tag hit against a body hit in ``bm25``. A tag says what an item is ABOUT, so a match
#: there outranks a word somewhere in its text.
_TAG_WEIGHT = 3.0

#: The most terms one query searches for. A pasted paragraph would otherwise become hundreds of
#: OR'd terms, each one a posting-list scan.
_MAX_QUERY_TERMS = 64

#: Scripts written without spaces between words: kana, CJK ideographs (extensions A to G and the
#: compatibility block) with the iteration and ideographic-zero marks (々 〆 〇 〻), Hangul syllables
#: and jamo. The katakana range skips U+30A0 and the middle dot U+30FB: inside a run they would
#: leave a lone kana on each side as a term, which matches nearly every Japanese item. Both jamo
#: blocks, because NFKC (applied first) turns a compatibility jamo such as U+3131 into its
#: U+1100-block form.
_UNSPACED = re.compile(
    r"[\u1100-\u11ff\u3005-\u3007\u303b\u3040-\u309f\u30a1-\u30fa\u30fc-\u30ff"
    r"\u3130-\u318f\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af"
    r"\U00020000-\U0003134F]+"
)

_ID = re.compile(r"m([1-9][0-9]{0,18})")

#: SQLite's INTEGER ceiling. A well-formed id above it names no row, and binding it would raise
#: `OverflowError` out of a read that promises to return None.
_MAX_ROWID = 2**63 - 1

# Each step is a tuple of single statements rather than one script: `executescript` COMMITs any
# open transaction before it runs, which would let two stores opening one new file both run a step.
_MIGRATIONS: tuple[tuple[str, ...], ...] = (
    (
        """CREATE TABLE items (
            id         INTEGER PRIMARY KEY AUTOINCREMENT CHECK (id > 0),
            text       TEXT NOT NULL,
            provenance TEXT NOT NULL CHECK (provenance IN ('observed', 'said', 'inferred')),
            origin     TEXT NOT NULL,
            tags_json  TEXT NOT NULL CHECK (json_valid(tags_json) AND json_type(tags_json) = 'array'),
            revises    INTEGER,
            run_id     TEXT,
            created_at REAL NOT NULL,
            valid_from REAL NOT NULL,
            valid_to   REAL,
            meta_json  TEXT NOT NULL CHECK (json_valid(meta_json) AND json_type(meta_json) = 'object')
        )""",
        # AUTOINCREMENT is load-bearing: without it SQLite may hand a new row the rowid of the
        # highest deleted one, and an id cited in an old trace would then name different text.
        "CREATE INDEX items_revises ON items(revises)",
        (
            "CREATE VIRTUAL TABLE items_fts USING fts5("
            "tags, text, content='', tokenize='unicode61 remove_diacritics 2')"
        ),
        # INSERT OR REPLACE deletes the old row without firing a DELETE trigger (recursive_triggers
        # is off by default), so refusing an existing id on INSERT is what closes that route.
        # `NEW.id > 0`: for an auto-assigned id a BEFORE INSERT trigger sees -1, so a guard on
        # `IS NOT NULL` would match a stray row a host once inserted at -1 and refuse every add
        # after it. The column's CHECK keeps such a row out in the first place.
        """CREATE TRIGGER items_no_replace BEFORE INSERT ON items
           WHEN NEW.id > 0 AND EXISTS (SELECT 1 FROM items WHERE id = NEW.id)
           BEGIN SELECT RAISE(ABORT, 'memory items are append-only: this id already exists'); END""",
        # `IS NOT` rather than `!=`: `NULL != 'x'` is NULL, which a WHEN clause reads as false, so a
        # `!=` guard would let a NULL `revises` or `run_id` be filled in after the fact.
        """CREATE TRIGGER items_no_update BEFORE UPDATE ON items
           WHEN NEW.id IS NOT OLD.id OR NEW.text IS NOT OLD.text
             OR NEW.provenance IS NOT OLD.provenance OR NEW.origin IS NOT OLD.origin
             OR NEW.tags_json IS NOT OLD.tags_json OR NEW.revises IS NOT OLD.revises
             OR NEW.run_id IS NOT OLD.run_id OR NEW.created_at IS NOT OLD.created_at
             OR NEW.valid_from IS NOT OLD.valid_from OR NEW.meta_json IS NOT OLD.meta_json
             OR OLD.valid_to IS NOT NULL OR NEW.valid_to IS NULL
           BEGIN
             SELECT RAISE(ABORT, 'memory items are append-only: only an unset valid_to may be set');
           END""",
        """CREATE TRIGGER items_no_delete BEFORE DELETE ON items
           BEGIN SELECT RAISE(ABORT, 'memory items are append-only: rows are never deleted'); END""",
    ),
)

_SCHEMA_VERSION = len(_MIGRATIONS)

#: Run on every open, after any migration. A file migrated by a Python whose SQLite has FTS5 opens
#: without running a step, so without this a SQLite lacking FTS5 would construct cleanly and fail
#: on the first add or search instead.
_FTS_PROBE = "SELECT 1 FROM items_fts LIMIT 0"

#: How long a connection waits on another writer before giving up, in seconds.
_BUSY_TIMEOUT_S = 30.0


class MemoryUnavailable(RuntimeError):
    """This Python's SQLite cannot hold a memory store, because it was built without FTS5."""


@dataclass(frozen=True)
class MemoryItem:
    """One remembered text and where it came from. Immutable, like the row it was read from.

    ``valid_to`` is ``None`` until the item is retired. ``meta`` is excluded from hashing because
    a dict cannot be hashed; two items still compare equal only when every field does.
    """

    id: str
    text: str
    provenance: Provenance
    origin: str
    tags: tuple[str, ...]
    revises: str | None
    run_id: str | None
    created_at: float
    valid_from: float
    valid_to: float | None
    meta: dict = field(default_factory=dict, hash=False)

    @property
    def retired(self) -> bool:
        return self.valid_to is not None

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready dict with every field, ``tags`` as a list."""
        return {
            "id": self.id,
            "text": self.text,
            "provenance": self.provenance,
            "origin": self.origin,
            "tags": list(self.tags),
            "revises": self.revises,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "meta": dict(self.meta),
        }


def _grams(text: str, *, unigrams: bool) -> str:
    """``text`` with every unspaced run replaced by its overlapping character pairs.

    With ``unigrams`` the run's single characters are added too: the INDEX side wants them, so a
    one-character query has something to hit, and the QUERY side does not, so a two-character word
    asks for its pair rather than for either of its halves.
    """
    text = unicodedata.normalize("NFKC", text or "")

    def pairs(match: re.Match[str]) -> str:
        run = match.group(0)
        if len(run) == 1:
            return f" {run} "
        out = [run[i : i + 2] for i in range(len(run) - 1)]
        if unigrams:
            out.extend(run)
        return " " + " ".join(out) + " "

    return _UNSPACED.sub(pairs, text)


def _words(text: str) -> list[str]:
    """Runs of letters, digits and combining marks, which is how ``unicode61`` reads a token.

    Not ``\\w+``: Python's ``\\w`` leaves out combining marks, so it cut ``हिन्दी`` into fragments
    no index row holds, and every Indic word and pointed Hebrew or Arabic word went unfound.
    """
    words: list[str] = []
    current: list[str] = []
    for char in text:
        if unicodedata.category(char)[0] in "LNM":
            current.append(char)
        elif current:
            words.append("".join(current))
            current = []
    if current:
        words.append("".join(current))
    return words


def _query_terms(text: str) -> list[str]:
    """The distinct terms a query searches for, in order, at most ``_MAX_QUERY_TERMS``.

    A lone letter or digit outside the unspaced scripts is dropped; it matches nearly everything
    and says nothing. A query made only of those, or only of punctuation, therefore has no terms at all.
    Each term is quoted as an FTS5 phrase, so the index's own tokenizer reads it the way it read
    the stored text.
    """
    seen: dict[str, None] = {}
    for term in _words(_grams(text, unigrams=False).lower()):
        if len(term) == 1 and not _UNSPACED.match(term):
            continue
        seen.setdefault(term, None)
        if len(seen) >= _MAX_QUERY_TERMS:
            break
    return list(seen)


def _match_expression(terms: list[str]) -> str:
    """An FTS5 query matching ANY of ``terms``, each quoted so no term is read as syntax.

    OR rather than AND: a question carries words no item contains (``what``, ``什麼``), and
    requiring every one would find nothing. ``bm25`` already ranks an item matching more of the
    rarer terms higher.
    """
    return " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)


def _normalise_tags(tags: Iterable[str] | str) -> tuple[str, ...]:
    if isinstance(tags, str):
        tags = (tags,)
    seen: dict[str, None] = {}
    for tag in tags:
        if not isinstance(tag, str):
            raise TypeError(f"a tag must be a str, got {type(tag).__name__}")
        tag = unicodedata.normalize("NFKC", tag).strip()
        if tag:
            seen.setdefault(tag, None)
    return tuple(seen)


def _provenance_set(provenance: str | Iterable[str] | None) -> tuple[str, ...] | None:
    if provenance is None:
        return None
    values = (provenance,) if isinstance(provenance, str) else tuple(provenance)
    for value in values:
        if value not in PROVENANCES:
            raise ValueError(f"provenance must be one of {PROVENANCES}, got {value!r}")
    return values


def _rowid(item_id: Any) -> int | None:
    if not isinstance(item_id, str):
        return None
    match = _ID.fullmatch(item_id.strip())
    if not match:
        return None
    rowid = int(match.group(1))
    return rowid if rowid <= _MAX_ROWID else None


def _timestamp(value: Any, name: str) -> float:
    """A finite float, or ValueError. SQLite stores NaN as NULL, which would read as 'not set'."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


class MemoryStore:
    """A memory that outlives a run, kept in one SQLite file at ``path``.

    ``path`` must be a file: ``":memory:"`` is refused, because a store that opens a connection
    per operation would lose everything between two calls. The parent directory is created.

    ``max_item_chars`` (default ``200_000``) caps one item's text. A longer text raises rather
    than being cut, so nothing is stored that the caller did not mean to store.

    Raises :class:`MemoryUnavailable` at construction when this SQLite has no FTS5.
    """

    def __init__(self, path: str | os.PathLike[str], *, max_item_chars: int = 200_000) -> None:
        if str(path).strip() in ("", ":memory:") or str(path).startswith("file::memory:"):
            raise ValueError("MemoryStore needs a file path; an in-memory database would not persist")
        if max_item_chars < 1:
            raise ValueError("max_item_chars must be at least 1")
        self.path = Path(path)
        self.max_item_chars = max_item_chars
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()

    # ---- connection plumbing -------------------------------------------------------------

    @contextlib.contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # `with sqlite3.connect(...)` commits but never CLOSES, and a handle left open keeps the
        # WAL files locked, which is what breaks temporary-directory cleanup on Windows.
        # `isolation_level=None` is the autocommit spelling that also exists on Python 3.11.
        conn = sqlite3.connect(self.path, timeout=_BUSY_TIMEOUT_S, isolation_level=None)
        try:
            conn.row_factory = sqlite3.Row
            yield conn
        finally:
            conn.close()

    @contextlib.contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                # SQLite may already have rolled back on its own (disk full, I/O error), and a
                # ROLLBACK then raises over the error that matters.
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")

    @staticmethod
    def _enable_wal(conn: sqlite3.Connection) -> None:
        # Switching the journal mode does not wait on SQLite's busy handler: measured, two stores
        # opening one brand-new file at once got `database is locked` here at once, despite the
        # connection's timeout. So this one statement waits for itself, within the same budget.
        deadline = time.monotonic() + _BUSY_TIMEOUT_S
        while True:
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                return
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc) or time.monotonic() >= deadline:
                    raise
                time.sleep(0.02)

    def _migrate(self) -> None:
        with self._connect() as conn:
            try:
                self._enable_wal(conn)
                conn.execute("BEGIN IMMEDIATE")
                try:
                    # Re-read INSIDE the lock: a second store opening the same new file waits here
                    # and then finds the step already done.
                    version = conn.execute("PRAGMA user_version").fetchone()[0]
                    if version > _SCHEMA_VERSION:
                        raise RuntimeError(
                            f"{self.path} was written by a newer rlm-harness (memory schema "
                            f"{version}, this one knows {_SCHEMA_VERSION}); upgrade to open it"
                        )
                    for step in _MIGRATIONS[version:]:
                        for statement in step:
                            conn.execute(statement)
                    if version < _SCHEMA_VERSION:
                        conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
                    conn.execute(_FTS_PROBE)
                except BaseException:
                    if conn.in_transaction:
                        conn.execute("ROLLBACK")
                    raise
                conn.execute("COMMIT")
            except sqlite3.OperationalError as exc:
                if "fts5" in str(exc).lower():
                    raise MemoryUnavailable(
                        f"this SQLite build cannot create the memory index ({exc}); it needs FTS5"
                    ) from exc
                raise

    # ---- writing ------------------------------------------------------------------------

    def _insert(
        self,
        conn: sqlite3.Connection,
        *,
        text: str,
        provenance: str,
        origin: str,
        tags: Iterable[str] | str,
        revises: int | None,
        run_id: str | None,
        valid_from: float | None,
        meta: dict | None,
    ) -> int:
        if provenance not in PROVENANCES:
            raise ValueError(f"provenance must be one of {PROVENANCES}, got {provenance!r}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty str")
        if len(text) > self.max_item_chars:
            raise ValueError(
                f"text is {len(text)} characters, over this store's max_item_chars of "
                f"{self.max_item_chars}"
            )
        if not isinstance(origin, str):
            raise TypeError("origin must be a str")
        clean_tags = _normalise_tags(tags)
        if meta is not None and not isinstance(meta, dict):
            raise TypeError(f"meta must be a dict, got {type(meta).__name__}")
        try:
            # allow_nan=False: `NaN` is not JSON, and the column's json_valid CHECK would refuse it
            # as an IntegrityError rather than the ValueError this method documents.
            meta_json = json.dumps(meta or {}, ensure_ascii=False, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"meta must be JSON-serialisable: {exc}") from exc
        now = time.time()
        cursor = conn.execute(
            "INSERT INTO items (text, provenance, origin, tags_json, revises, run_id, created_at,"
            " valid_from, valid_to, meta_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)",
            (text, provenance, origin, json.dumps(list(clean_tags), ensure_ascii=False), revises,
             run_id, now, now if valid_from is None else _timestamp(valid_from, "valid_from"),
             meta_json),
        )
        rowid = cursor.lastrowid
        conn.execute(
            "INSERT INTO items_fts (rowid, tags, text) VALUES (?, ?, ?)",
            (rowid, _grams(" ".join(clean_tags), unigrams=True), _grams(text, unigrams=True)),
        )
        return rowid

    def add(
        self,
        text: str,
        *,
        provenance: Provenance,
        origin: str,
        tags: Iterable[str] | str = (),
        valid_from: float | None = None,
        run_id: str | None = None,
        meta: dict | None = None,
    ) -> MemoryItem:
        """Remember ``text`` and return the stored item.

        ``provenance`` is required and must be one of :data:`PROVENANCES`; choose it from the code
        path the text arrived by, never from anything the text says about itself. ``origin`` is an
        opaque label of yours (a URL, a file name, ``run:<id>``). ``valid_from`` defaults to now.
        Raises ``ValueError`` for an unknown provenance, an empty text, a text over
        ``max_item_chars`` or a ``meta`` that ``json.dumps`` refuses, and ``TypeError`` for a
        non-str tag or origin or a ``meta`` that is not a dict.
        """
        with self._write() as conn:
            rowid = self._insert(conn, text=text, provenance=provenance, origin=origin, tags=tags,
                                 revises=None, run_id=run_id, valid_from=valid_from, meta=meta)
        return self._get_rowid(rowid)

    def revise(
        self,
        item_id: str,
        text: str,
        *,
        origin: str,
        provenance: Provenance = "inferred",
        tags: Iterable[str] | str = (),
        run_id: str | None = None,
        meta: dict | None = None,
    ) -> MemoryItem:
        """Add a NEW item that revises ``item_id``, leaving the original exactly as it was.

        ``provenance`` defaults to ``inferred``; pass ``said`` when the revision is a person's
        correction. Raises ``KeyError`` for an unknown id, and ``ValueError`` as :meth:`add` does.
        """
        target = _rowid(item_id)
        with self._write() as conn:
            if target is None or conn.execute(
                "SELECT 1 FROM items WHERE id = ?", (target,)
            ).fetchone() is None:
                raise KeyError(item_id)
            rowid = self._insert(conn, text=text, provenance=provenance, origin=origin, tags=tags,
                                 revises=target, run_id=run_id, valid_from=None, meta=meta)
        return self._get_rowid(rowid)

    def retire(self, item_id: str, *, at: float | None = None) -> MemoryItem:
        """Mark ``item_id`` as no longer valid from ``at`` (default now), once.

        The item stays readable through :meth:`get` and a point-in-time :meth:`search`. Raises
        ``KeyError`` for an unknown id, and ``ValueError`` when it is already retired or ``at``
        is not a finite number.
        """
        target = _rowid(item_id)
        when = time.time() if at is None else _timestamp(at, "at")
        with self._write() as conn:
            # One guarded statement, so two racing retires end in this ValueError rather than one
            # of them reaching the trigger.
            cursor = conn.execute(
                "UPDATE items SET valid_to = ? WHERE id = ? AND valid_to IS NULL", (when, target)
            )
            if cursor.rowcount == 0:
                exists = target is not None and conn.execute(
                    "SELECT 1 FROM items WHERE id = ?", (target,)
                ).fetchone()
                if not exists:
                    raise KeyError(item_id)
                raise ValueError(f"{item_id} is already retired")
        return self._get_rowid(target)

    # ---- reading ------------------------------------------------------------------------

    @staticmethod
    def _item(row: sqlite3.Row) -> MemoryItem:
        return MemoryItem(
            id=f"m{row['id']}",
            text=row["text"],
            provenance=row["provenance"],
            origin=row["origin"],
            tags=tuple(json.loads(row["tags_json"])),
            revises=None if row["revises"] is None else f"m{row['revises']}",
            run_id=row["run_id"],
            created_at=row["created_at"],
            valid_from=row["valid_from"],
            valid_to=row["valid_to"],
            meta=json.loads(row["meta_json"]),
        )

    def _get_rowid(self, rowid: int) -> MemoryItem:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM items WHERE id = ?", (rowid,)).fetchone()
        return self._item(row)

    def get(self, item_id: str) -> MemoryItem | None:
        """The item with this id, or ``None``. A malformed id is ``None`` too, never an error."""
        target = _rowid(item_id)
        if target is None:
            return None
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM items WHERE id = ?", (target,)).fetchone()
        return None if row is None else self._item(row)

    def revisions(self, item_id: str) -> list[MemoryItem]:
        """The items that revise ``item_id``, oldest first. Empty for an unknown id."""
        target = _rowid(item_id)
        if target is None:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM items WHERE revises = ? ORDER BY id", (target,)
            ).fetchall()
        return [self._item(row) for row in rows]

    def search(
        self,
        query: str = "",
        *,
        limit: int = 10,
        provenance: str | Iterable[str] | None = None,
        tags: Iterable[str] | str = (),
        at: float | None = None,
        include_retired: bool = False,
    ) -> list[MemoryItem]:
        """Items matching ``query``, best first; with an empty query, the newest first.

        - A non-empty query that reduces to no terms (punctuation, a lone letter or digit) matches
          nothing, rather than falling back to the newest items.
        - ``provenance`` keeps only the given label or labels.
        - ``tags`` keeps only items carrying ALL of them.
        - Items retired as of now are left out unless ``include_retired`` is true. One retired at
          a future time stays visible until then, and one whose ``valid_from`` lies in the future
          is NOT left out either.
        - ``at`` is a point-in-time view instead: only items valid at that instant
          (``valid_from <= at`` and not yet retired at ``at``), whatever ``include_retired`` says.

        ``limit`` is clamped to at least 1. Raises ``ValueError`` for an unknown provenance.
        """
        kinds = _provenance_set(provenance)
        wanted_tags = _normalise_tags(tags)
        limit = max(1, int(limit))
        where: list[str] = []
        params: list[Any] = []
        if kinds is not None:
            where.append(f"items.provenance IN ({', '.join('?' for _ in kinds)})")
            params.extend(kinds)
        for tag in wanted_tags:
            where.append("EXISTS (SELECT 1 FROM json_each(items.tags_json) WHERE value = ?)")
            params.append(tag)
        if at is not None:
            when = _timestamp(at, "at")
            where.append("items.valid_from <= ? AND (items.valid_to IS NULL OR items.valid_to > ?)")
            params.extend([when, when])
        elif not include_retired:
            where.append("(items.valid_to IS NULL OR items.valid_to > ?)")
            params.append(time.time())

        if query and query.strip():
            terms = _query_terms(query)
            if not terms:
                return []
            sql = (
                "SELECT items.* FROM items_fts JOIN items ON items.id = items_fts.rowid "
                "WHERE items_fts MATCH ?"
                + "".join(f" AND {clause}" for clause in where)
                + f" ORDER BY bm25(items_fts, {_TAG_WEIGHT}, 1.0), items.id DESC LIMIT ?"
            )
            params = [_match_expression(terms), *params, limit]
        else:
            sql = (
                "SELECT items.* FROM items"
                + (" WHERE " + " AND ".join(where) if where else "")
                + " ORDER BY items.id DESC LIMIT ?"
            )
            params = [*params, limit]
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._item(row) for row in rows]

    def count(self) -> int:
        """How many items the store holds, retired ones included."""
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
