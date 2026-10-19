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

**What is written is never changed, an id never names different text, and only the host erases.**
Four ways exist to change what a reader sees, and none rewrites a row:

- :meth:`MemoryStore.revise` adds a NEW item that points at the old one (``revises``).
- :meth:`MemoryStore.retire` sets ``valid_to`` once, from NULL to a time. It is the only UPDATE in
  this module.
- :meth:`MemoryStore.label` adds a label row; the latest one shows, and a person's (``said``)
  outranks a model's (``inferred``) whatever the order.
- :meth:`MemoryStore.merge` and :meth:`MemoryStore.unmerge` add alias rows, which every reader
  applies to entity names; a person's decision about a name stands, an unmerge included.

:meth:`MemoryStore.purge` is the one way anything leaves the file, and it is the host's alone: no
tool and no model-facing path reaches it. It exists because a person may want back out what they
put in. An id is never reused, so a purged id in an old trace names nothing rather than something
else, and a ``purged`` record keeps when and why, never what.

SQLite triggers refuse a stray ``UPDATE``, ``DELETE`` or ``INSERT OR REPLACE`` against every table
that holds a row (items, labels, failures, purge records, aliases, seen names), so a host's own SQL cannot quietly rewrite history either; ``purge`` passes them by a
flag row only its own transaction can see. They are not a security boundary against someone with
the file: ``DROP TRIGGER`` still works. The search indexes are contentless FTS5 tables built with
``contentless_delete=1`` (SQLite 3.43 or newer), and a purge merges them afterwards, because a
delete alone leaves the words in the index's segments behind a tombstone. An index row deleted by
other means only makes search miss that item; it never changes the item.

**Chinese, Japanese and Korean are indexed without a dictionary.** That covers kana, the CJK
ideograph blocks including extensions A to G with the marks ``々 〆 〇 〻``, and Hangul syllables
and both jamo blocks. FTS5's ``unicode61`` tokenizer treats a whole unspaced run as one token, so
``索引`` inside ``資料庫與索引`` would never match. Each such run is indexed as its overlapping
character pairs AND its single characters, computed in Python because Python's ``sqlite3`` cannot
register an FTS5 tokenizer. A query keeps only the pairs of a run, or the character itself when the
run is one character long, so a two-character word matches inside a longer run and a one-character
query still finds something. Text is NFKC normalised first, so a full-width ``ＰＤＦ`` and ``PDF``
are one word.

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
import itertools
import json
import logging
import math
import os
import re
import sqlite3
import time
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

#: The closed set of provenance labels, in the order of how much a reader may trust them.
PROVENANCES = ("observed", "said", "inferred")

#: The provenances a DERIVED row (a label) may carry. Nothing derived is observed: a label is either
#: a person's or the host's word (`said`) or a model's (`inferred`).
LABEL_PROVENANCES = ("said", "inferred")

#: The four kinds of label an item can carry.
LABEL_KINDS = ("title", "summary", "tags", "entities")

#: The longest item text summarised in one model call; a longer one is read in the REPL instead.
#: Lifted from the summariser this design was taken from, where it was measured, not guessed.
SHORT_LIMIT = 12_000

#: Per-kind caps a label is trimmed to, never refused for. A list label keeps at most
#: `_LABEL_ITEMS` entries, each cut to its own cap.
_TITLE_CHARS = 80
_SUMMARY_CHARS = 600
_TAG_CHARS = 40
_ENTITY_CHARS = 60
_LABEL_ITEMS = 8

#: The oldest SQLite this store runs on. `contentless_delete=1`, which lets `purge` take a row's
#: tokens out of the search index, arrived in 3.43.
_MIN_SQLITE = (3, 43, 0)

_UNSET: Any = object()

#: Current tag or entity label values of non-retired items with how many items carry each, most
#: used first. Parameters: kind, now.
_COUNT_SQL = (
    "SELECT j.value AS name, COUNT(*) AS n FROM current_labels cl "
    "JOIN items ON items.id = cl.item_id, json_each(cl.value_json) j "
    "WHERE cl.kind = ? AND (items.valid_to IS NULL OR items.valid_to > ?) "
    "GROUP BY j.value ORDER BY n DESC, name"
)

logger = logging.getLogger(__name__)

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
_TOPIC_ID = re.compile(r"t([1-9][0-9]{0,18})")

#: The longest topic name kept; a longer one is cut, never refused.
_TOPIC_NAME_CHARS = 60

#: SQLite's INTEGER ceiling. A well-formed id above it names no row, and binding it would raise
#: `OverflowError` out of a read that promises to return None.
_MAX_ROWID = 2**63 - 1

# Each step is a tuple of single statements rather than one script: `executescript` COMMITs any
# open transaction before it runs, which would let two stores opening one new file both run a step.
# A step may also hold a callable taking the connection, for work SQL cannot do (re-gramming text).
_MIGRATIONS: tuple[tuple[str | Callable[[sqlite3.Connection], None], ...], ...] = (
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
    # Step 2: labels, purge, and an index that can forget. Step 1 is shipped and never edited, so
    # what it created is altered here rather than there.
    (
        # `purge` is the one deliberate path past the delete triggers. It inserts a row here inside
        # its own transaction and removes it before COMMIT, so no other connection ever sees one,
        # and every other DELETE (a host's stray SQL included) is still refused. An ordinary table
        # rather than a TEMP one because a trigger in the main schema may not read the temp schema.
        # A row a host's stray SQL commits here would disarm the triggers, so every open clears it.
        "CREATE TABLE purge_flag (x INTEGER)",
        "DROP TRIGGER items_no_delete",
        """CREATE TRIGGER items_no_delete BEFORE DELETE ON items
           WHEN NOT EXISTS (SELECT 1 FROM purge_flag)
           BEGIN SELECT RAISE(ABORT, 'memory items are append-only: only purge removes a row'); END""",
        # Rebuilt with contentless_delete=1, so a purged item's tokens leave the index too.
        "DROP TABLE items_fts",
        (
            "CREATE VIRTUAL TABLE items_fts USING fts5("
            "tags, text, content='', contentless_delete=1, tokenize='unicode61 remove_diacritics 2')"
        ),
        # Calls the live `_grams`. The index is derived data rebuilt from `items`, so a later change
        # to `_grams` is a reindex question for the whole store, not a reason to freeze this step.
        lambda conn: _reindex_items(conn),
        """CREATE TABLE labels (
            id         INTEGER PRIMARY KEY AUTOINCREMENT CHECK (id > 0),
            item_id    INTEGER NOT NULL,
            kind       TEXT NOT NULL CHECK (kind IN ('title', 'summary', 'tags', 'entities')),
            value_json TEXT CHECK (value_json IS NOT NULL OR provenance = 'said'),
            provenance TEXT NOT NULL CHECK (provenance IN ('said', 'inferred')),
            run_id     TEXT,
            created_at REAL NOT NULL,
            CHECK (value_json IS NULL OR (json_valid(value_json) AND json_type(value_json) =
                   CASE WHEN kind IN ('title', 'summary') THEN 'text' ELSE 'array' END))
        )""",
        "CREATE INDEX labels_key ON labels(item_id, kind, provenance, id)",
        """CREATE TRIGGER labels_no_replace BEFORE INSERT ON labels
           WHEN NEW.id > 0 AND EXISTS (SELECT 1 FROM labels WHERE id = NEW.id)
           BEGIN SELECT RAISE(ABORT, 'memory labels are append-only: this id already exists'); END""",
        # A CHECK cannot hold a subquery, so "an array of strings" is a trigger.
        """CREATE TRIGGER labels_strings_only BEFORE INSERT ON labels
           WHEN NEW.value_json IS NOT NULL AND json_type(NEW.value_json) = 'array'
             AND EXISTS (SELECT 1 FROM json_each(NEW.value_json) WHERE type <> 'text')
           BEGIN SELECT RAISE(ABORT, 'a tags or entities label is an array of strings'); END""",
        """CREATE TRIGGER labels_no_update BEFORE UPDATE ON labels
           BEGIN SELECT RAISE(ABORT, 'memory labels are append-only: a change is a new row'); END""",
        """CREATE TRIGGER labels_no_delete BEFORE DELETE ON labels
           WHEN NOT EXISTS (SELECT 1 FROM purge_flag)
           BEGIN SELECT RAISE(ABORT, 'memory labels are append-only: only purge removes a row'); END""",
        (
            "CREATE VIRTUAL TABLE labels_fts USING fts5("
            "text, content='', contentless_delete=1, tokenize='unicode61 remove_diacritics 2')"
        ),
        """CREATE TABLE distill_failures (
            id         INTEGER PRIMARY KEY AUTOINCREMENT CHECK (id > 0),
            item_id    INTEGER NOT NULL,
            error      TEXT NOT NULL,
            created_at REAL NOT NULL
        )""",
        "CREATE INDEX distill_failures_item ON distill_failures(item_id, id)",
        """CREATE TRIGGER distill_failures_no_replace BEFORE INSERT ON distill_failures
           WHEN NEW.id > 0 AND EXISTS (SELECT 1 FROM distill_failures WHERE id = NEW.id)
           BEGIN SELECT RAISE(ABORT, 'distill failures are append-only: this id already exists'); END""",
        """CREATE TRIGGER distill_failures_no_update BEFORE UPDATE ON distill_failures
           BEGIN SELECT RAISE(ABORT, 'distill failures are append-only'); END""",
        """CREATE TRIGGER distill_failures_no_delete BEFORE DELETE ON distill_failures
           WHEN NOT EXISTS (SELECT 1 FROM purge_flag)
           BEGIN SELECT RAISE(ABORT, 'distill failures are append-only: only purge removes a row'); END""",
        # The record that something was purged: when and why, never what. Not even purge removes it.
        """CREATE TABLE purged (
            id         INTEGER PRIMARY KEY AUTOINCREMENT CHECK (id > 0),
            item_id    INTEGER NOT NULL,
            reason     TEXT NOT NULL,
            created_at REAL NOT NULL
        )""",
        "CREATE INDEX purged_item ON purged(item_id, id)",
        """CREATE TRIGGER purged_no_replace BEFORE INSERT ON purged
           WHEN NEW.id > 0 AND EXISTS (SELECT 1 FROM purged WHERE id = NEW.id)
           BEGIN SELECT RAISE(ABORT, 'purge records are append-only: this id already exists'); END""",
        """CREATE TRIGGER purged_no_update BEFORE UPDATE ON purged
           BEGIN SELECT RAISE(ABORT, 'purge records are append-only'); END""",
        """CREATE TRIGGER purged_no_delete BEFORE DELETE ON purged
           BEGIN SELECT RAISE(ABORT, 'purge records are never deleted'); END""",
        # The label each (item, kind) shows: the latest `said` row when its value is set, otherwise
        # the latest `inferred` row. A `said` row with a NULL value is a person clearing their own
        # label, which lets the model's show through again. Each lookup is one index probe.
        """CREATE VIEW current_labels AS
           SELECT l.* FROM labels l
           WHERE l.id = COALESCE(
               (SELECT CASE WHEN s.value_json IS NOT NULL THEN s.id END FROM labels s
                WHERE s.item_id = l.item_id AND s.kind = l.kind AND s.provenance = 'said'
                ORDER BY s.id DESC LIMIT 1),
               (SELECT i.id FROM labels i
                WHERE i.item_id = l.item_id AND i.kind = l.kind AND i.provenance = 'inferred'
                ORDER BY i.id DESC LIMIT 1))""",
    ),
    # Step 3: names that mean the same thing. Aliases are store-wide vocabulary, not tied to one
    # item, so a purge does not remove them.
    (
        """CREATE TABLE entity_aliases (
            id         INTEGER PRIMARY KEY AUTOINCREMENT CHECK (id > 0),
            alias      TEXT NOT NULL,
            canonical  TEXT,
            provenance TEXT NOT NULL CHECK (provenance IN ('said', 'inferred')),
            run_id     TEXT,
            created_at REAL NOT NULL
        )""",
        "CREATE INDEX entity_aliases_key ON entity_aliases(alias, provenance, id)",
        """CREATE TRIGGER entity_aliases_no_replace BEFORE INSERT ON entity_aliases
           WHEN NEW.id > 0 AND EXISTS (SELECT 1 FROM entity_aliases WHERE id = NEW.id)
           BEGIN SELECT RAISE(ABORT, 'entity aliases are append-only: this id already exists'); END""",
        """CREATE TRIGGER entity_aliases_no_update BEFORE UPDATE ON entity_aliases
           BEGIN SELECT RAISE(ABORT, 'entity aliases are append-only: a change is a new row'); END""",
        """CREATE TRIGGER entity_aliases_no_delete BEFORE DELETE ON entity_aliases
           BEGIN SELECT RAISE(ABORT, 'entity aliases are append-only: unmerge writes a new row'); END""",
        # No UNIQUE on `name`: INSERT OR REPLACE against a UNIQUE column deletes without firing a
        # DELETE trigger. "Insert if absent" is a WHERE NOT EXISTS instead.
        """CREATE TABLE entity_seen (
            id         INTEGER PRIMARY KEY AUTOINCREMENT CHECK (id > 0),
            name       TEXT NOT NULL,
            created_at REAL NOT NULL
        )""",
        "CREATE INDEX entity_seen_name ON entity_seen(name)",
        """CREATE TRIGGER entity_seen_no_replace BEFORE INSERT ON entity_seen
           WHEN NEW.id > 0 AND EXISTS (SELECT 1 FROM entity_seen WHERE id = NEW.id)
           BEGIN SELECT RAISE(ABORT, 'seen names are append-only: this id already exists'); END""",
        """CREATE TRIGGER entity_seen_no_update BEFORE UPDATE ON entity_seen
           BEGIN SELECT RAISE(ABORT, 'seen names are append-only'); END""",
        """CREATE TRIGGER entity_seen_no_delete BEFORE DELETE ON entity_seen
           BEGIN SELECT RAISE(ABORT, 'seen names are append-only'); END""",
        # The alias each name has now. Unlike a label, a `said` row decides whenever one exists,
        # a NULL canonical included: a person's unmerge must stay unmerged, never let a model's
        # earlier merge show through again. Only with no `said` row does the latest `inferred` one
        # decide. A NULL canonical means "not an alias".
        """CREATE VIEW current_aliases AS
           SELECT a.alias, a.canonical, a.provenance FROM entity_aliases a
           WHERE a.canonical IS NOT NULL AND a.id = COALESCE(
               (SELECT MAX(s.id) FROM entity_aliases s
                WHERE s.alias = a.alias AND s.provenance = 'said'),
               (SELECT MAX(i.id) FROM entity_aliases i
                WHERE i.alias = a.alias AND i.provenance = 'inferred'))""",
    ),
    # Step 4: topics, named groups of items. Every table is an event log; what a reader sees is the
    # latest row by the rules in the views below. The rows that name an item (membership,
    # dismissal, leftover) go with it when it is purged; a topic and its names do not.
    (
        """CREATE TABLE topics (
            id         INTEGER PRIMARY KEY AUTOINCREMENT CHECK (id > 0),
            provenance TEXT NOT NULL CHECK (provenance IN ('said', 'inferred')),
            run_id     TEXT,
            created_at REAL NOT NULL
        )""",
        """CREATE TRIGGER topics_no_replace BEFORE INSERT ON topics
           WHEN NEW.id > 0 AND EXISTS (SELECT 1 FROM topics WHERE id = NEW.id)
           BEGIN SELECT RAISE(ABORT, 'topics is append-only: this id already exists'); END""",
        """CREATE TRIGGER topics_no_update BEFORE UPDATE ON topics
           BEGIN SELECT RAISE(ABORT, 'topics is append-only: a change is a new row'); END""",
        """CREATE TRIGGER topics_no_delete BEFORE DELETE ON topics
           BEGIN SELECT RAISE(ABORT, 'topics is append-only'); END""",
        """CREATE TABLE topic_names (
            id         INTEGER PRIMARY KEY AUTOINCREMENT CHECK (id > 0),
            topic_id   INTEGER NOT NULL,
            name       TEXT NOT NULL,
            provenance TEXT NOT NULL CHECK (provenance IN ('said', 'inferred')),
            run_id     TEXT,
            created_at REAL NOT NULL
        )""",
        "CREATE INDEX topic_names_key ON topic_names(topic_id, provenance, id)",
        """CREATE TRIGGER topic_names_no_replace BEFORE INSERT ON topic_names
           WHEN NEW.id > 0 AND EXISTS (SELECT 1 FROM topic_names WHERE id = NEW.id)
           BEGIN SELECT RAISE(ABORT, 'topic_names is append-only: this id already exists'); END""",
        """CREATE TRIGGER topic_names_no_update BEFORE UPDATE ON topic_names
           BEGIN SELECT RAISE(ABORT, 'topic_names is append-only: a change is a new row'); END""",
        """CREATE TRIGGER topic_names_no_delete BEFORE DELETE ON topic_names
           BEGIN SELECT RAISE(ABORT, 'topic_names is append-only'); END""",
        """CREATE TABLE topic_retired (
            id         INTEGER PRIMARY KEY AUTOINCREMENT CHECK (id > 0),
            topic_id   INTEGER NOT NULL,
            retired    INTEGER NOT NULL CHECK (retired IN (0, 1)),
            provenance TEXT NOT NULL CHECK (provenance IN ('said', 'inferred')),
            run_id     TEXT,
            created_at REAL NOT NULL
        )""",
        "CREATE INDEX topic_retired_key ON topic_retired(topic_id, id)",
        """CREATE TRIGGER topic_retired_no_replace BEFORE INSERT ON topic_retired
           WHEN NEW.id > 0 AND EXISTS (SELECT 1 FROM topic_retired WHERE id = NEW.id)
           BEGIN SELECT RAISE(ABORT, 'topic_retired is append-only: this id already exists'); END""",
        """CREATE TRIGGER topic_retired_no_update BEFORE UPDATE ON topic_retired
           BEGIN SELECT RAISE(ABORT, 'topic_retired is append-only: a change is a new row'); END""",
        """CREATE TRIGGER topic_retired_no_delete BEFORE DELETE ON topic_retired
           BEGIN SELECT RAISE(ABORT, 'topic_retired is append-only'); END""",
        """CREATE TABLE topic_members (
            id         INTEGER PRIMARY KEY AUTOINCREMENT CHECK (id > 0),
            item_id    INTEGER NOT NULL,
            topic_id   INTEGER NOT NULL,
            member     INTEGER NOT NULL CHECK (member IN (0, 1)),
            provenance TEXT NOT NULL CHECK (provenance IN ('said', 'inferred')),
            run_id     TEXT,
            created_at REAL NOT NULL
        )""",
        "CREATE INDEX topic_members_key ON topic_members(item_id, topic_id, provenance, id)",
        "CREATE INDEX topic_members_topic ON topic_members(topic_id)",
        """CREATE TRIGGER topic_members_no_replace BEFORE INSERT ON topic_members
           WHEN NEW.id > 0 AND EXISTS (SELECT 1 FROM topic_members WHERE id = NEW.id)
           BEGIN SELECT RAISE(ABORT, 'topic_members is append-only: this id already exists'); END""",
        """CREATE TRIGGER topic_members_no_update BEFORE UPDATE ON topic_members
           BEGIN SELECT RAISE(ABORT, 'topic_members is append-only: a change is a new row'); END""",
        """CREATE TRIGGER topic_members_no_delete BEFORE DELETE ON topic_members
           WHEN NOT EXISTS (SELECT 1 FROM purge_flag)
           BEGIN SELECT RAISE(ABORT, 'topic_members is append-only: only purge removes a row'); END""",
        """CREATE TABLE topic_dismissed (
            id         INTEGER PRIMARY KEY AUTOINCREMENT CHECK (id > 0),
            item_id    INTEGER NOT NULL,
            topic_id   INTEGER NOT NULL,
            created_at REAL NOT NULL
        )""",
        "CREATE INDEX topic_dismissed_key ON topic_dismissed(item_id, topic_id)",
        """CREATE TRIGGER topic_dismissed_no_replace BEFORE INSERT ON topic_dismissed
           WHEN NEW.id > 0 AND EXISTS (SELECT 1 FROM topic_dismissed WHERE id = NEW.id)
           BEGIN SELECT RAISE(ABORT, 'topic_dismissed is append-only: this id already exists'); END""",
        """CREATE TRIGGER topic_dismissed_no_update BEFORE UPDATE ON topic_dismissed
           BEGIN SELECT RAISE(ABORT, 'topic_dismissed is append-only: a change is a new row'); END""",
        """CREATE TRIGGER topic_dismissed_no_delete BEFORE DELETE ON topic_dismissed
           WHEN NOT EXISTS (SELECT 1 FROM purge_flag)
           BEGIN SELECT RAISE(ABORT, 'topic_dismissed is append-only: only purge removes a row'); END""",
        """CREATE TABLE topic_left (
            id         INTEGER PRIMARY KEY AUTOINCREMENT CHECK (id > 0),
            item_id    INTEGER NOT NULL,
            signature  TEXT NOT NULL,
            created_at REAL NOT NULL
        )""",
        "CREATE INDEX topic_left_key ON topic_left(signature, item_id)",
        """CREATE TRIGGER topic_left_no_replace BEFORE INSERT ON topic_left
           WHEN NEW.id > 0 AND EXISTS (SELECT 1 FROM topic_left WHERE id = NEW.id)
           BEGIN SELECT RAISE(ABORT, 'topic_left is append-only: this id already exists'); END""",
        """CREATE TRIGGER topic_left_no_update BEFORE UPDATE ON topic_left
           BEGIN SELECT RAISE(ABORT, 'topic_left is append-only: a change is a new row'); END""",
        """CREATE TRIGGER topic_left_no_delete BEFORE DELETE ON topic_left
           WHEN NOT EXISTS (SELECT 1 FROM purge_flag)
           BEGIN SELECT RAISE(ABORT, 'topic_left is append-only: only purge removes a row'); END""",
        # A topic's name follows the label rule: the latest said name, else the latest inferred one.
        """CREATE VIEW current_topic_names AS
           SELECT n.topic_id, n.name, n.provenance FROM topic_names n
           WHERE n.id = COALESCE(
               (SELECT MAX(s.id) FROM topic_names s
                WHERE s.topic_id = n.topic_id AND s.provenance = 'said'),
               (SELECT MAX(i.id) FROM topic_names i
                WHERE i.topic_id = n.topic_id AND i.provenance = 'inferred'))""",
        # Membership follows the alias rule: a said row decides whenever one exists, a removal (0)
        # included, so a person's "not in this topic" is never undone by a model's placement.
        """CREATE VIEW current_members AS
           SELECT m.item_id, m.topic_id, m.provenance FROM topic_members m
           WHERE m.member = 1 AND m.id = COALESCE(
               (SELECT MAX(s.id) FROM topic_members s WHERE s.item_id = m.item_id
                AND s.topic_id = m.topic_id AND s.provenance = 'said'),
               (SELECT MAX(i.id) FROM topic_members i WHERE i.item_id = m.item_id
                AND i.topic_id = m.topic_id AND i.provenance = 'inferred'))""",
        """CREATE VIEW live_topics AS
           SELECT t.id, t.provenance, t.created_at FROM topics t
           WHERE COALESCE((SELECT r.retired FROM topic_retired r WHERE r.topic_id = t.id
                           ORDER BY r.id DESC LIMIT 1), 0) = 0""",
    ),
)

_SCHEMA_VERSION = len(_MIGRATIONS)

#: Run on every open, after any migration. A file migrated by a Python whose SQLite has FTS5 opens
#: without running a step, so without this a SQLite lacking FTS5 would construct cleanly and fail
#: on the first add or search instead.
_FTS_PROBE = "SELECT 1 FROM items_fts LIMIT 0"

#: How long a connection waits on another writer before giving up, in seconds.
_BUSY_TIMEOUT_S = 30.0

#: The companion file :mod:`rlm_harness.memory_vectors` keeps its vectors in, next to the store.
VECTORS_SUFFIX = ".vectors.db"


class MemoryUnavailable(RuntimeError):
    """This Python's SQLite cannot hold a memory store: it lacks FTS5, or predates 3.43."""


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


@dataclass(frozen=True)
class MemoryLabels:
    """What an item is about: its current title, summary, tags and entities.

    Each kind shows its latest `said` value when one is set, otherwise its latest `inferred` one,
    so a person's label outranks any later summary. ``provenance`` says which, per kind, and is
    ``None`` for a kind with no label at all.
    """

    title: str = ""
    summary: str = ""
    tags: tuple[str, ...] = ()
    entities: tuple[str, ...] = ()
    provenance: dict = field(default_factory=lambda: dict.fromkeys(LABEL_KINDS), hash=False)


@dataclass(frozen=True)
class Topic:
    """A named group of items. ``name_provenance`` says who wrote the name that shows now."""

    id: str
    name: str
    name_provenance: str
    provenance: str
    retired: bool
    created_at: float


@dataclass(frozen=True)
class PurgeRecord:
    """That an item was purged, when and why. It holds nothing of what the item said."""

    item_id: str
    reason: str
    purged_at: float


def vectors_path(store_path: str | os.PathLike[str]) -> Path:
    """Where a store's vectors live: its own path plus ``VECTORS_SUFFIX``."""
    return Path(str(store_path) + VECTORS_SUFFIX)


def _erase_vectors(store_path: Path, rowids: list[int], *, secure: bool) -> None:
    """Delete these items' vectors from the companion file, if there is one.

    Plain SQL on a plain table, so it needs neither the ``memory-vectors`` extra nor an open
    :class:`~rlm_harness.memory_vectors.MemoryVectors`: an embedding can be partly turned back
    into its text, so a purge that left it would not have erased the item. A failure is logged,
    never raised, because the purge itself has committed; the next ``sync`` drops every vector
    whose item is gone.
    """
    path = vectors_path(store_path)
    if not rowids or not path.is_file():
        return
    try:
        conn = sqlite3.connect(path, timeout=_BUSY_TIMEOUT_S, isolation_level=None)
        try:
            conn.execute("PRAGMA secure_delete = ON")
            if conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'vectors'"
                            ).fetchone() is None:
                return
            conn.execute("DELETE FROM vectors WHERE item_id IN (SELECT value FROM json_each(?))",
                         (json.dumps(sorted(rowids)),))
            if secure:
                try:
                    conn.execute("VACUUM")
                except sqlite3.OperationalError as exc:
                    # The delete above has committed; a raise here would read as if it had not.
                    logger.warning("%s: vectors erased, but VACUUM failed (%s); free pages stay until "
                                   "a later VACUUM", path, exc)
            conn.execute("PRAGMA busy_timeout = 0")
            busy, _, _ = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            if busy:
                logger.warning("%s: vectors erased, but another connection kept the WAL checkpoint "
                               "from finishing; the old pages leave the file at the next full "
                               "checkpoint", path)
        finally:
            conn.close()
    except sqlite3.Error as exc:
        logger.warning("%s: purge committed, but its vectors could not be erased (%s); the next "
                       "sync drops them", path, exc)


def _reindex_items(conn: sqlite3.Connection) -> None:
    """Fill a freshly created ``items_fts`` from ``items``: migration step 2's rebuild."""
    for row in conn.execute("SELECT id, tags_json, text FROM items").fetchall():
        conn.execute(
            "INSERT INTO items_fts (rowid, tags, text) VALUES (?, ?, ?)",
            (row[0], _grams(" ".join(json.loads(row[1])), unigrams=True), _grams(row[2], unigrams=True)),
        )


def _fold(text: Any) -> str:
    """How two labels are compared for sameness: NFKC, then casefolded, whitespace collapsed."""
    return " ".join(unicodedata.normalize("NFKC", text if isinstance(text, str) else "").casefold().split())


def _name_group(name: str, aliases: dict[str, str]) -> set[str]:
    """The folded names ``name`` is known by: itself, its canonical, and every alias of that."""
    key = _fold(name)
    canonical = _fold(aliases.get(key, name))
    return {key, canonical} | {alias for alias, target in aliases.items() if _fold(target) == canonical}


def _oneline(value: Any, cap: int) -> str:
    if not isinstance(value, str):
        raise TypeError(f"a label must be a str, got {type(value).__name__}")
    return " ".join(unicodedata.normalize("NFKC", value).split())[:cap].strip()


def _label_list(values: Any, cap: int, *, fold: bool) -> list[str]:
    """NFKC, then (for tags) casefold, then strip, cut and dedupe, keeping order. Trimmed to
    ``_LABEL_ITEMS``; never refused for length."""
    if isinstance(values, str) or not isinstance(values, Iterable):
        raise TypeError("a list label must be a list of str")
    seen: dict[str, None] = {}
    for value in values:
        if not isinstance(value, str):
            raise TypeError(f"a label entry must be a str, got {type(value).__name__}")
        text = unicodedata.normalize("NFKC", value)
        if fold:
            text = text.casefold()
        text = " ".join(text.split())[:cap].strip()
        if text:
            seen.setdefault(text, None)
        if len(seen) >= _LABEL_ITEMS:
            break
    return list(seen)


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


def _topic_rowid(topic_id: Any) -> int | None:
    if not isinstance(topic_id, str):
        return None
    match = _TOPIC_ID.fullmatch(topic_id.strip())
    if not match:
        return None
    rowid = int(match.group(1))
    return rowid if rowid <= _MAX_ROWID else None


def _topic_name(name: Any) -> str:
    if not isinstance(name, str):
        raise TypeError("a topic name must be a str")
    clean = " ".join(unicodedata.normalize("NFKC", name).split())[:_TOPIC_NAME_CHARS].strip()
    if not clean:
        raise ValueError("a topic name must be non-empty")
    return clean


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

    Raises :class:`MemoryUnavailable` at construction when this SQLite has no FTS5 or is older
    than 3.43.
    """

    def __init__(self, path: str | os.PathLike[str], *, max_item_chars: int = 200_000) -> None:
        if str(path).strip() in ("", ":memory:") or str(path).startswith("file::memory:"):
            raise ValueError("MemoryStore needs a file path; an in-memory database would not persist")
        if max_item_chars < 1:
            raise ValueError("max_item_chars must be at least 1")
        if sqlite3.sqlite_version_info < _MIN_SQLITE:
            raise MemoryUnavailable(
                f"this Python's SQLite is {sqlite3.sqlite_version}; a memory store needs 3.43 or "
                "newer, for an index that can forget what purge removes"
            )
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
            # On for EVERY connection, not only purge's: the index merges its segments on ordinary
            # writes too, and a merge without it leaves an old copy of the text as slack in a live
            # page, where a later purge cannot reach it. Measured: a purged word survived on disk
            # until this moved here from the purge path.
            conn.execute("PRAGMA secure_delete = ON")
            # Entity names keep their case, so a filter compares them folded, the same way grounding
            # and the known-name map do.
            conn.create_function("rlm_fold", 1, _fold, deterministic=True)
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
                            if callable(statement):
                                statement(conn)
                            else:
                                conn.execute(statement)
                    if version < _SCHEMA_VERSION:
                        conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
                    conn.execute(_FTS_PROBE)
                    if _SCHEMA_VERSION >= 2:
                        stray = conn.execute("SELECT COUNT(*) FROM purge_flag").fetchone()[0]
                        if stray:
                            logger.warning("%s: cleared %d stray purge_flag row(s) that disarmed "
                                           "the delete triggers", self.path, stray)
                            conn.execute("DELETE FROM purge_flag")
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

    # ---- labels -------------------------------------------------------------------------

    def label(
        self,
        item_id: str,
        *,
        title: Any = _UNSET,
        summary: Any = _UNSET,
        tags: Any = _UNSET,
        entities: Any = _UNSET,
        provenance: str = "said",
        run_id: str | None = None,
    ) -> MemoryLabels:
        """Record what ``item_id`` is about, one row per kind given, and return its current labels.

        An omitted kind is untouched. ``None`` clears a kind, and only a ``said`` call may clear:
        a person removing their own label, so the latest ``inferred`` one shows again. An empty
        value (``""`` or ``[]``) is NOT a clear: it sets an empty label, which a ``said`` call makes
        outrank the model's, and an empty ``said`` summary keeps the item out of the distill queue. Values are
        trimmed to their caps (title 80 characters, summary 600, up to 8 tags of 40, up to 8
        entities of 60), never refused for length; tags are casefolded, entities keep their case.
        Raises ``KeyError`` for an unknown item, ``ValueError`` for a call that names no kind, a
        provenance outside :data:`LABEL_PROVENANCES` or an ``inferred`` clear, and ``TypeError``
        for a value of the wrong type.
        """
        if provenance not in LABEL_PROVENANCES:
            raise ValueError(f"provenance must be one of {LABEL_PROVENANCES}, got {provenance!r}")
        given = {k: v for k, v in (("title", title), ("summary", summary), ("tags", tags),
                                   ("entities", entities)) if v is not _UNSET}
        if not given:
            raise ValueError("label() needs at least one of title, summary, tags or entities")
        rows: list[tuple[str, str | None, str]] = []
        for kind, value in given.items():
            if value is None:
                if provenance != "said":
                    raise ValueError("only a said label may clear a kind")
                rows.append((kind, None, ""))
                continue
            if kind in ("title", "summary"):
                text = _oneline(value, _TITLE_CHARS if kind == "title" else _SUMMARY_CHARS)
                rows.append((kind, json.dumps(text, ensure_ascii=False), text))
            else:
                values = _label_list(value, _TAG_CHARS if kind == "tags" else _ENTITY_CHARS,
                                     fold=kind == "tags")
                rows.append((kind, json.dumps(values, ensure_ascii=False), " ".join(values)))
        target = _rowid(item_id)
        now = time.time()
        with self._write() as conn:
            if target is None or conn.execute(
                "SELECT 1 FROM items WHERE id = ?", (target,)
            ).fetchone() is None:
                raise KeyError(item_id)
            for kind, value_json, text in rows:
                cursor = conn.execute(
                    "INSERT INTO labels (item_id, kind, value_json, provenance, run_id, created_at)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (target, kind, value_json, provenance, run_id, now),
                )
                if text:
                    conn.execute("INSERT INTO labels_fts (rowid, text) VALUES (?, ?)",
                                 (cursor.lastrowid, _grams(text, unigrams=True)))
        return self.labels(item_id)

    def labels(self, item_id: str) -> MemoryLabels:
        """The current labels of ``item_id``: empty for an unlabelled, unknown or malformed id.

        Entities are read through the alias table: a merged name shows as its canonical. The stored
        label is unchanged, so an unmerge shows the name as written again.
        """
        target = _rowid(item_id)
        if target is None:
            return MemoryLabels()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT kind, value_json, provenance FROM current_labels WHERE item_id = ?", (target,)
            ).fetchall()
            aliases = self._readings(conn)
        values: dict[str, Any] = {}
        origin: dict[str, str | None] = dict.fromkeys(LABEL_KINDS)
        for row in rows:
            value = json.loads(row["value_json"])
            if row["kind"] == "entities":
                # Read through the alias table: a merged name shows as the name it was merged into.
                value = list(dict.fromkeys(aliases.get(_fold(v), v) for v in value))
            values[row["kind"]] = tuple(value) if isinstance(value, list) else value
            origin[row["kind"]] = row["provenance"]
        return MemoryLabels(provenance=origin, **values)

    def _labels_by_item(self) -> dict[str, MemoryLabels]:
        """The current labels of every non-retired item that has any, newest item first, read in
        one query and resolved through the alias table once. For the organising reads, which
        otherwise opened a connection per item."""
        out: dict[str, dict[str, Any]] = {}
        with self._connect() as conn:
            aliases = self._readings(conn)
            rows = conn.execute(
                "SELECT cl.item_id, cl.kind, cl.value_json, cl.provenance FROM current_labels cl "
                "JOIN items ON items.id = cl.item_id "
                "WHERE items.valid_to IS NULL OR items.valid_to > ? ORDER BY cl.item_id DESC",
                (time.time(),),
            ).fetchall()
        for row in rows:
            entry = out.setdefault(f"m{row['item_id']}", {"provenance": dict.fromkeys(LABEL_KINDS)})
            value = json.loads(row["value_json"])
            if row["kind"] == "entities":
                value = list(dict.fromkeys(aliases.get(_fold(v), v) for v in value))
            entry[row["kind"]] = tuple(value) if isinstance(value, list) else value
            entry["provenance"][row["kind"]] = row["provenance"]
        return {item_id: MemoryLabels(**values) for item_id, values in out.items()}

    def unlabelled(
        self,
        *,
        limit: int = 20,
        item_ids: Iterable[str] | None = None,
        max_failures: int = 3,
        max_chars: int | None = None,
    ) -> list[MemoryItem]:
        """Items still waiting for a summary, newest first: the queue a distill pass reads.

        It is derived from the labels themselves, so nothing can be stuck in a "being summarised"
        state. An item counts as labelled once a model has answered (an empty summary included, so a
        text too thin to summarise is asked about once) or a summary currently shows; a person who
        sets and then clears their own summary leaves neither, so the item is queued again. Retired
        items are left out. An item that failed ``max_failures`` times is skipped unless
        ``item_ids`` names it; ``item_ids`` limits the queue to those items, and ``max_chars`` to
        texts no longer than that.
        """
        where = ["(items.valid_to IS NULL OR items.valid_to > ?)",
                 # Labelled means a summary shows now: a model's answer (an empty one included) always
                 # does, so it is asked about once. A person who sets and then clears their own
                 # summary leaves none, so the item is queued again.
                 ("NOT EXISTS (SELECT 1 FROM current_labels cl WHERE cl.item_id = items.id "
                  "AND cl.kind = 'summary')")]
        params: list[Any] = [time.time()]
        named: list[int] = []
        if item_ids is not None:
            if isinstance(item_ids, str):
                raise TypeError("item_ids is a list of ids, not a str")
            named = [r for r in (_rowid(i) for i in item_ids) if r is not None]
            if not named:
                return []
            where.append("items.id IN (SELECT value FROM json_each(?))")
            params.append(json.dumps(named))
        else:
            where.append("(SELECT COUNT(*) FROM distill_failures f WHERE f.item_id = items.id) < ?")
            params.append(max(1, int(max_failures)))
        if max_chars is not None:
            where.append("length(items.text) <= ?")
            params.append(int(max_chars))
        sql = f"SELECT items.* FROM items WHERE {' AND '.join(where)} ORDER BY items.id DESC LIMIT ?"
        with self._connect() as conn:
            rows = conn.execute(sql, [*params, max(1, int(limit))]).fetchall()
        return [self._item(row) for row in rows]

    def label_counts(self, kind: str, *, limit: int = 80) -> list[tuple[str, int]]:
        """The most used tags or entities across current labels of non-retired items, with how many
        items carry each.

        Entities are read through the alias table, so a merged group counts as one name, once per
        item. This is the list a summariser is shown so that related items converge on one label.
        """
        if kind not in ("tags", "entities"):
            raise ValueError("label_counts counts 'tags' or 'entities'")
        limit = max(1, int(limit))
        if kind == "tags":
            with self._connect() as conn:
                return [(row["name"], int(row["n"])) for row in conn.execute(
                    _COUNT_SQL + " LIMIT ?", ("tags", time.time(), limit)).fetchall()]
        # Entities are counted by the name they resolve to, once per item: an item labelled with
        # two names of one group carries that entity once.
        totals: Counter[str] = Counter()
        with self._connect() as conn:
            aliases = self._readings(conn)
            for row in conn.execute(
                "SELECT cl.value_json FROM current_labels cl JOIN items ON items.id = cl.item_id "
                "WHERE cl.kind = 'entities' AND (items.valid_to IS NULL OR items.valid_to > ?)",
                (time.time(),),
            ).fetchall():
                totals.update({aliases.get(_fold(n), n) for n in json.loads(row[0])})
        return sorted(totals.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]

    # ---- names that mean the same thing --------------------------------------------------

    @staticmethod
    def _alias_map(conn: sqlite3.Connection) -> dict[str, str]:
        return {row["alias"]: row["canonical"]
                for row in conn.execute("SELECT alias, canonical FROM current_aliases").fetchall()}

    @classmethod
    def _readings(cls, conn: sqlite3.Connection) -> dict[str, str]:
        """What every reader maps a folded name to: each live alias to its canonical, and each case
        or width variant of a canonical to the canonical's own spelling, so one group is one name."""
        aliases = cls._alias_map(conn)
        readings = {_fold(canonical): canonical for canonical in aliases.values()}
        readings.update(aliases)
        return readings

    def aliases(self) -> dict[str, str]:
        """Every live alias, as ``folded alias name -> canonical name``.

        Alias names are kept folded (NFKC, casefolded, whitespace collapsed), so ``Postgres`` and
        ``postgres`` are one alias. The map never holds a chain: a canonical is never an alias.
        """
        with self._connect() as conn:
            return self._alias_map(conn)

    def resolve(self, name: str) -> str:
        """The name ``name`` is read as: its canonical if it is a live alias or a case variant of a
        canonical, else itself."""
        with self._connect() as conn:
            return self._readings(conn).get(_fold(name), name)

    def entity_names(self) -> dict[str, int]:
        """Every entity name in the current labels of non-retired items, as written, with counts.

        These are the names alignment may merge; an alias must be one of them.
        """
        with self._connect() as conn:
            return {row["name"]: int(row["n"]) for row in conn.execute(
                _COUNT_SQL, ("entities", time.time())).fetchall()}

    def merge(
        self,
        alias: str,
        canonical: str,
        *,
        provenance: str = "said",
        batch: Iterable[str] | None = None,
        run_id: str | None = None,
    ) -> str:
        """Read ``alias`` as ``canonical`` from now on, and return the canonical recorded.

        Refused with ``ValueError`` (the message is the reason) unless all of these hold:

        - ``alias`` is an entity name in use (``entity_names``), compared folded, and ``canonical``
          is one too or is already the canonical of a live alias;
        - after following ``canonical`` to its own canonical, the two differ;
        - ``alias`` is not already a canonical (that would leave a chain) or a live alias;
        - an ``inferred`` merge names a ``batch`` holding ``alias``: a model may only merge a name
          it was asked about;
        - an ``inferred`` merge touches no name a person decided about (any ``said`` row, merged or
          unmerged), whether as the alias, the canonical given, or the canonical it resolves to.

        The canonical is recorded in the spelling its group already has, else its most used one.
        ``TypeError`` for a name that is not a str.
        """
        if provenance not in LABEL_PROVENANCES:
            raise ValueError(f"provenance must be one of {LABEL_PROVENANCES}, got {provenance!r}")
        if not isinstance(alias, str) or not isinstance(canonical, str):
            raise TypeError("alias and canonical must be str")
        key = _fold(alias)
        if not key or not _fold(canonical):
            raise ValueError("alias and canonical must be non-empty")
        if provenance == "inferred" and (
            batch is None or key not in {_fold(b) for b in batch if isinstance(b, str)}
        ):
            raise ValueError(f"{alias!r} was not among the names this run was asked about")
        with self._write() as conn:
            names = {row["name"]: int(row["n"]) for row in conn.execute(
                _COUNT_SQL, ("entities", time.time())).fetchall()}
            spellings: dict[str, str] = {}
            for name, _ in sorted(names.items(), key=lambda pair: (-pair[1], pair[0])):
                spellings.setdefault(_fold(name), name)
            aliases = self._alias_map(conn)
            # A live canonical keeps the spelling it was first recorded in, so every alias of one
            # group names it the same way; and it can be merged into even when no item names it
            # any more, since it is still the name the store shows.
            live = {_fold(c): c for c in aliases.values()}
            canon_key = _fold(canonical)
            if key not in spellings:
                raise ValueError("both names must be entity names in use")
            if canon_key in aliases:
                target = aliases[canon_key]
            elif canon_key in live:
                target = live[canon_key]
            elif canon_key in spellings:
                target = spellings[canon_key]
            else:
                raise ValueError("both names must be entity names in use")
            if _fold(target) == key:
                raise ValueError(f"{alias!r} and {canonical!r} are already one name")
            if key in {_fold(c) for c in aliases.values()}:
                raise ValueError(f"{alias!r} is already the canonical of another name")
            if key in aliases:
                raise ValueError(f"{alias!r} is already an alias of {aliases[key]!r}")
            if provenance == "inferred":
                # Every name the merge touches, in either role: checking the alias alone let a model
                # rejoin a pair a person separated by merging it the other way round.
                for name in sorted({key, canon_key, _fold(target)}):
                    if conn.execute("SELECT 1 FROM entity_aliases WHERE alias = ? AND provenance = 'said'",
                                    (name,)).fetchone():
                        raise ValueError(f"a person already decided about {name!r}")
            conn.execute(
                "INSERT INTO entity_aliases (alias, canonical, provenance, run_id, created_at) "
                "VALUES (?, ?, ?, ?, ?)", (key, target, provenance, run_id, time.time()))
        return target

    def unmerge(self, alias: str, *, provenance: str = "said", run_id: str | None = None) -> None:
        """Stop reading ``alias`` as another name.

        A ``said`` unmerge is always recorded, even for a name that is not an alias now: it is a
        person's "keep these apart", and no later ``inferred`` merge can override it. An
        ``inferred`` unmerge is refused (``ValueError``) for a name that is not a live alias or
        that a person already decided about.
        """
        if provenance not in LABEL_PROVENANCES:
            raise ValueError(f"provenance must be one of {LABEL_PROVENANCES}, got {provenance!r}")
        key = _fold(alias) if isinstance(alias, str) else ""
        if not key:
            raise ValueError("alias must be a non-empty str")
        with self._write() as conn:
            if provenance == "inferred":
                if conn.execute("SELECT 1 FROM entity_aliases WHERE alias = ? AND provenance = 'said'",
                                (key,)).fetchone():
                    raise ValueError(f"a person already decided about {alias!r}")
                if key not in self._alias_map(conn):
                    raise ValueError(f"{alias!r} is not an alias")
            conn.execute(
                "INSERT INTO entity_aliases (alias, canonical, provenance, run_id, created_at) "
                "VALUES (?, NULL, ?, ?, ?)", (key, provenance, run_id, time.time()))

    def unseen_entities(self, *, limit: int = 100) -> list[str]:
        """Entity names in use that alignment has not been asked about yet, most used first."""
        with self._connect() as conn:
            seen = {row[0] for row in conn.execute("SELECT name FROM entity_seen").fetchall()}
            rows = conn.execute(_COUNT_SQL, ("entities", time.time())).fetchall()
        fresh = [row["name"] for row in rows if _fold(row["name"]) not in seen]
        return fresh[: max(1, int(limit))]

    def mark_seen(self, names: Iterable[str]) -> None:
        """Record that alignment has considered ``names``, so it is not asked about them again."""
        if isinstance(names, str):
            raise TypeError("mark_seen takes a list of names, not a str")
        keys = sorted({_fold(n) for n in names if isinstance(n, str) and _fold(n)})
        if not keys:
            return
        now = time.time()
        with self._write() as conn:
            for key in keys:
                conn.execute(
                    "INSERT INTO entity_seen (name, created_at) SELECT ?, ? "
                    "WHERE NOT EXISTS (SELECT 1 FROM entity_seen WHERE name = ?)", (key, now, key))

    def entity_graph(self, *, topic: str | None = None, limit: int = 200) -> list[tuple[str, str, int]]:
        """Pairs of entities named together by one item, with how many items do, strongest first.

        Read from the current entity labels of non-retired items, through the alias table, so a
        merged pair is one node. ``topic`` limits it to that topic's members. It is data to draw,
        not a drawing.
        """
        pairs: Counter[tuple[str, str]] = Counter()
        scope, params = "", [time.time()]
        if topic is not None:
            scope = (" AND EXISTS (SELECT 1 FROM current_members cm WHERE cm.item_id = items.id "
                     "AND cm.topic_id = ?)")
            params.append(_topic_rowid(topic) or 0)
        with self._connect() as conn:
            aliases = self._readings(conn)
            for row in conn.execute(
                "SELECT cl.value_json FROM current_labels cl JOIN items ON items.id = cl.item_id "
                "WHERE cl.kind = 'entities' AND (items.valid_to IS NULL OR items.valid_to > ?)" + scope,
                params,
            ).fetchall():
                names = sorted({aliases.get(_fold(n), n) for n in json.loads(row[0])})
                pairs.update(itertools.combinations(names, 2))
        ranked = sorted(pairs.items(), key=lambda pair: (-pair[1], pair[0]))
        return [(a, b, n) for (a, b), n in ranked[: max(1, int(limit))]]

    # ---- topics ---------------------------------------------------------------------------

    @staticmethod
    def _topic_row(conn: sqlite3.Connection, rowid: int) -> Topic | None:
        row = conn.execute(
            "SELECT t.id, t.provenance, t.created_at, n.name, n.provenance AS name_provenance, "
            "COALESCE((SELECT r.retired FROM topic_retired r WHERE r.topic_id = t.id "
            "ORDER BY r.id DESC LIMIT 1), 0) AS retired "
            "FROM topics t JOIN current_topic_names n ON n.topic_id = t.id WHERE t.id = ?",
            (rowid,),
        ).fetchone()
        if row is None:
            return None
        return Topic(f"t{row['id']}", row["name"], row["name_provenance"], row["provenance"],
                     bool(row["retired"]), row["created_at"])

    def _require_topic(self, conn: sqlite3.Connection, topic_id: str) -> int:
        rowid = _topic_rowid(topic_id)
        if rowid is None or conn.execute("SELECT 1 FROM topics WHERE id = ?", (rowid,)).fetchone() is None:
            raise KeyError(topic_id)
        return rowid

    def create_topic(self, name: str, *, provenance: str = "said", run_id: str | None = None) -> Topic:
        """Open a topic named ``name`` (cut to 60 characters) and return it."""
        if provenance not in LABEL_PROVENANCES:
            raise ValueError(f"provenance must be one of {LABEL_PROVENANCES}, got {provenance!r}")
        clean = _topic_name(name)
        now = time.time()
        with self._write() as conn:
            rowid = conn.execute("INSERT INTO topics (provenance, run_id, created_at) VALUES (?, ?, ?)",
                                 (provenance, run_id, now)).lastrowid
            conn.execute("INSERT INTO topic_names (topic_id, name, provenance, run_id, created_at) "
                         "VALUES (?, ?, ?, ?, ?)", (rowid, clean, provenance, run_id, now))
            return self._topic_row(conn, rowid)

    def rename_topic(self, topic_id: str, name: str, *, provenance: str = "said",
                     run_id: str | None = None) -> Topic:
        """Give a topic a new name. A person's (``said``) name outranks any model's, whatever the
        order. Raises ``KeyError`` for an unknown topic."""
        if provenance not in LABEL_PROVENANCES:
            raise ValueError(f"provenance must be one of {LABEL_PROVENANCES}, got {provenance!r}")
        clean = _topic_name(name)
        with self._write() as conn:
            rowid = self._require_topic(conn, topic_id)
            conn.execute("INSERT INTO topic_names (topic_id, name, provenance, run_id, created_at) "
                         "VALUES (?, ?, ?, ?, ?)", (rowid, clean, provenance, run_id, time.time()))
            return self._topic_row(conn, rowid)

    def _set_retired(self, topic_id: str, retired: bool, provenance: str, run_id: str | None) -> Topic:
        if provenance not in LABEL_PROVENANCES:
            raise ValueError(f"provenance must be one of {LABEL_PROVENANCES}, got {provenance!r}")
        with self._write() as conn:
            rowid = self._require_topic(conn, topic_id)
            last = conn.execute("SELECT provenance FROM topic_retired WHERE topic_id = ? "
                                "ORDER BY id DESC LIMIT 1", (rowid,)).fetchone()
            if provenance == "inferred" and last is not None and last[0] == "said":
                # A person retired or restored this topic last; a model does not overturn that.
                raise ValueError(f"a person already decided whether {topic_id} is retired")
            conn.execute("INSERT INTO topic_retired (topic_id, retired, provenance, run_id, created_at) "
                         "VALUES (?, ?, ?, ?, ?)", (rowid, int(retired), provenance, run_id, time.time()))
            return self._topic_row(conn, rowid)

    def retire_topic(self, topic_id: str, *, provenance: str = "said", run_id: str | None = None) -> Topic:
        """Hide a topic from listings and suggestions. Its memberships stay readable. A ``said``
        retirement also keeps its name from being reopened by a model (``said_retired_names``). An
        ``inferred`` retirement is refused (``ValueError``) when a person last retired or restored
        the topic."""
        return self._set_retired(topic_id, True, provenance, run_id)

    def restore_topic(self, topic_id: str) -> Topic:
        """Bring a retired topic back."""
        return self._set_retired(topic_id, False, "said", None)

    def said_retired_names(self) -> set[str]:
        """The folded names of topics a person retired and that are still retired: names a model
        may not open a new topic under. A topic retired by a pass, not a person, is not here."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT n.name FROM topics t JOIN current_topic_names n ON n.topic_id = t.id "
                "JOIN topic_retired r ON r.id = (SELECT MAX(x.id) FROM topic_retired x WHERE x.topic_id = t.id) "
                "WHERE r.retired = 1 AND r.provenance = 'said'").fetchall()
        return {_fold(row[0]) for row in rows}

    def _items_by_ids(self, item_ids: list[str]) -> dict[str, MemoryItem]:
        """The items with these ids that exist, read in one query."""
        rows = [r for r in (_rowid(i) for i in item_ids) if r is not None]
        if not rows:
            return {}
        with self._connect() as conn:
            found = conn.execute("SELECT * FROM items WHERE id IN (SELECT value FROM json_each(?))",
                                 (json.dumps(rows),)).fetchall()
        return {f"m{row['id']}": self._item(row) for row in found}

    def topic(self, topic_id: str) -> Topic | None:
        """The topic with this id, retired or not, or ``None``."""
        rowid = _topic_rowid(topic_id)
        if rowid is None:
            return None
        with self._connect() as conn:
            return self._topic_row(conn, rowid)

    def topics(self, *, include_retired: bool = False) -> list[Topic]:
        """Every topic, oldest first; retired ones only when asked for."""
        with self._connect() as conn:
            ids = [row[0] for row in conn.execute("SELECT id FROM topics ORDER BY id").fetchall()]
            found = [self._topic_row(conn, rowid) for rowid in ids]
        return [t for t in found if t is not None and (include_retired or not t.retired)]

    def _membership(self, item_id: str, topic_id: str, member: bool, provenance: str,
                    run_id: str | None) -> None:
        if provenance not in LABEL_PROVENANCES:
            raise ValueError(f"provenance must be one of {LABEL_PROVENANCES}, got {provenance!r}")
        item = _rowid(item_id)
        with self._write() as conn:
            topic = self._require_topic(conn, topic_id)
            if item is None or conn.execute("SELECT 1 FROM items WHERE id = ?", (item,)).fetchone() is None:
                raise KeyError(item_id)
            if member and self._topic_row(conn, topic).retired:
                raise ValueError(f"{topic_id} is retired")
            if provenance == "inferred" and conn.execute(
                "SELECT 1 FROM topic_members WHERE item_id = ? AND topic_id = ? AND provenance = 'said'",
                (item, topic),
            ).fetchone():
                raise ValueError(f"a person already decided whether {item_id} is in {topic_id}")
            conn.execute(
                "INSERT INTO topic_members (item_id, topic_id, member, provenance, run_id, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)", (item, topic, int(member), provenance, run_id, time.time()))

    def place(self, item_id: str, topic_id: str, *, provenance: str = "said",
              run_id: str | None = None) -> None:
        """Put an item into a topic.

        A person's decision about a pair stands whenever one exists, a removal included: an
        ``inferred`` placement is refused (``ValueError``) for any pair a person placed or removed.
        A retired topic takes no new members. ``KeyError`` for an unknown item or topic.
        """
        self._membership(item_id, topic_id, True, provenance, run_id)

    def unplace(self, item_id: str, topic_id: str, *, provenance: str = "said",
                run_id: str | None = None) -> None:
        """Take an item out of a topic. A ``said`` removal keeps the pair apart for good: no later
        ``inferred`` placement can put it back."""
        self._membership(item_id, topic_id, False, provenance, run_id)

    def topic_items(self, topic_id: str) -> list[MemoryItem]:
        """The items that are members of a topic now, newest first, retired items included (a
        membership outlives an item's retirement). Empty for an unknown id."""
        rowid = _topic_rowid(topic_id)
        if rowid is None:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT items.* FROM current_members cm JOIN items ON items.id = cm.item_id "
                "WHERE cm.topic_id = ? ORDER BY items.id DESC", (rowid,)).fetchall()
        return [self._item(row) for row in rows]

    def item_topics(self, item_id: str) -> list[Topic]:
        """The live topics an item is in now, oldest first."""
        rowid = _rowid(item_id)
        if rowid is None:
            return []
        with self._connect() as conn:
            ids = [row[0] for row in conn.execute(
                "SELECT cm.topic_id FROM current_members cm JOIN live_topics lt ON lt.id = cm.topic_id "
                "WHERE cm.item_id = ? ORDER BY cm.topic_id", (rowid,)).fetchall()]
            return [t for t in (self._topic_row(conn, i) for i in ids) if t is not None]

    def dismiss(self, item_id: str, topic_id: str) -> None:
        """Never suggest this item for this topic again."""
        item = _rowid(item_id)
        with self._write() as conn:
            topic = self._require_topic(conn, topic_id)
            if item is None or conn.execute("SELECT 1 FROM items WHERE id = ?", (item,)).fetchone() is None:
                raise KeyError(item_id)
            conn.execute(
                "INSERT INTO topic_dismissed (item_id, topic_id, created_at) SELECT ?, ?, ? WHERE NOT "
                "EXISTS (SELECT 1 FROM topic_dismissed WHERE item_id = ? AND topic_id = ?)",
                (item, topic, time.time(), item, topic))

    def dismissed(self) -> set[tuple[str, str]]:
        """Every dismissed ``(item_id, topic_id)`` pair."""
        with self._connect() as conn:
            return {(f"m{r[0]}", f"t{r[1]}") for r in
                    conn.execute("SELECT item_id, topic_id FROM topic_dismissed").fetchall()}

    def removed_pairs(self) -> set[tuple[str, str]]:
        """Every ``(item_id, topic_id)`` pair whose latest ``said`` row is a removal: a person took
        it apart, and no model may place it."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT m.item_id, m.topic_id FROM topic_members m WHERE m.provenance = 'said' "
                "AND m.member = 0 AND m.id = (SELECT MAX(s.id) FROM topic_members s "
                "WHERE s.item_id = m.item_id AND s.topic_id = m.topic_id AND s.provenance = 'said')"
            ).fetchall()
        return {(f"m{r[0]}", f"t{r[1]}") for r in rows}

    def record_left(self, item_ids: Iterable[str], signature: str) -> None:
        """Record that organising left these items against the topic set ``signature``, so they
        are not asked about again until the set of live topics changes."""
        if isinstance(item_ids, str):
            raise TypeError("record_left takes a list of ids, not a str")
        rows = [r for r in (_rowid(i) for i in item_ids) if r is not None]
        now = time.time()
        with self._write() as conn:
            for rowid in rows:
                conn.execute(
                    "INSERT INTO topic_left (item_id, signature, created_at) SELECT ?, ?, ? "
                    "WHERE EXISTS (SELECT 1 FROM items WHERE id = ?) AND NOT EXISTS "
                    "(SELECT 1 FROM topic_left WHERE item_id = ? AND signature = ?)",
                    (rowid, signature, now, rowid, rowid, signature))

    def left_items(self, signature: str | None = None) -> list[str]:
        """Items organising left: against ``signature``, or against any topic set when it is
        ``None``, each once."""
        with self._connect() as conn:
            if signature is None:
                rows = conn.execute("SELECT DISTINCT item_id FROM topic_left ORDER BY item_id").fetchall()
            else:
                rows = conn.execute("SELECT item_id FROM topic_left WHERE signature = ? ORDER BY id",
                                    (signature,)).fetchall()
        return [f"m{r[0]}" for r in rows]

    def topic_signature(self) -> str:
        """The set of live topics a decision is made against: a leftover is reconsidered once it
        changes, since a new topic may now take it."""
        with self._connect() as conn:
            return "\u001f".join(f"t{r[0]}" for r in conn.execute("SELECT id FROM live_topics ORDER BY id"))

    def note_distill_failure(self, item_id: str, error: str) -> None:
        """Record that summarising ``item_id`` failed, so a pass can stop retrying it forever."""
        target = _rowid(item_id)
        if target is None:
            raise KeyError(item_id)
        with self._write() as conn:
            if conn.execute("SELECT 1 FROM items WHERE id = ?", (target,)).fetchone() is None:
                raise KeyError(item_id)
            conn.execute("INSERT INTO distill_failures (item_id, error, created_at) VALUES (?, ?, ?)",
                         (target, str(error)[:2000], time.time()))

    def last_distill_error(self, item_id: str) -> tuple[str, float] | None:
        """The latest recorded failure for ``item_id`` as ``(error, when)``, or ``None``."""
        target = _rowid(item_id)
        if target is None:
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT error, created_at FROM distill_failures WHERE item_id = ? ORDER BY id DESC "
                "LIMIT 1", (target,)
            ).fetchone()
        return None if row is None else (row["error"], row["created_at"])

    # ---- removing -----------------------------------------------------------------------

    def purge(
        self,
        item_ids: Iterable[str],
        *,
        reason: str = "",
        include_revisions: bool = False,
        secure: bool = False,
    ) -> list[str]:
        """Erase items for good, and return the ids actually purged.

        This is the host's deliberate exception to append-only, and the only one: no tool and no
        model-facing path reaches it. For each item it deletes the row, its labels, its recorded
        failures, its topic memberships, dismissals and leftovers, its search-index entries and
        its vector in the companion file (``VECTORS_SUFFIX``) when there is one, and writes a
        ``purged`` record holding the time and ``reason``, never the text. Ids are never reused, so an id in an old trace names nothing
        rather than something else.

        **Getting the words out, not only the row.** Deleting from a ``contentless_delete`` index
        only writes a tombstone; the terms stay in its segments until they are merged. So every
        purge merges both indexes (FTS5 ``'optimize'``), which rewrites them whole and costs time
        in proportion to the store's size. Every connection the store opens runs with SQLite's
        ``secure_delete`` on, so space freed by any write, a merge included, is zeroed rather than
        left holding old text. ``secure=True`` then runs ``VACUUM`` to shrink the file. Last, the WAL is
        checkpointed. If another connection is reading at that moment the checkpoint cannot finish,
        a warning is logged, and the old pages leave the main file at the next full checkpoint,
        which SQLite runs when the last connection closes.

        **What it costs others.** The merge holds the write lock for its whole duration, so on a large
        store other writers, a model's ``memory_note`` included, wait up to the 30-second busy
        timeout and past it get ``Memory error``. ``secure_delete`` adds zero-writes to every write
        that frees pages, ordinary adds and labels included. If ``VACUUM`` cannot get the lock, the
        purge has still committed: a warning is logged and the purged ids are returned.

        Aliases and the names alignment has considered are store-wide vocabulary, not tied to an
        item, so a purge leaves them: a name only a purged item used stays there. ``reason`` is kept
        up to 500 characters. ``include_revisions`` also purges every item that
        revises a purged one, transitively; otherwise a revision survives and its ``revises`` names
        an id that no longer exists. Unknown and malformed ids are ignored.
        """
        if isinstance(item_ids, str):
            raise TypeError("purge takes a list of ids; a bare str would be read one character at a time")
        wanted = {r for r in (_rowid(i) for i in item_ids) if r is not None}
        done: list[int] = []
        if wanted:
            with self._write() as conn:
                # One JSON array bound as ONE value: an `IN (?, ?, ...)` list fails past SQLite's
                # 32,766 bound variables, which a host erasing a large selection reaches.
                existing = {row[0] for row in conn.execute(
                    "SELECT id FROM items WHERE id IN (SELECT value FROM json_each(?))",
                    (json.dumps(sorted(wanted)),)).fetchall()}
                if include_revisions:
                    frontier = set(existing)
                    while frontier:
                        found = {row[0] for row in conn.execute(
                            "SELECT id FROM items WHERE revises IN (SELECT value FROM json_each(?))",
                            (json.dumps(sorted(frontier)),)).fetchall()} - existing
                        existing |= found
                        frontier = found
                if existing:
                    conn.execute("INSERT INTO purge_flag (x) VALUES (1)")
                    now = time.time()
                    for rowid in sorted(existing):
                        label_ids = [row[0] for row in conn.execute(
                            "SELECT id FROM labels WHERE item_id = ?", (rowid,)).fetchall()]
                        conn.executemany("DELETE FROM labels_fts WHERE rowid = ?",
                                         [(i,) for i in label_ids])
                        conn.execute("DELETE FROM labels WHERE item_id = ?", (rowid,))
                        conn.execute("DELETE FROM distill_failures WHERE item_id = ?", (rowid,))
                        for table in ("topic_members", "topic_dismissed", "topic_left"):
                            conn.execute(f"DELETE FROM {table} WHERE item_id = ?", (rowid,))
                        conn.execute("DELETE FROM items_fts WHERE rowid = ?", (rowid,))
                        conn.execute("DELETE FROM items WHERE id = ?", (rowid,))
                        conn.execute("INSERT INTO purged (item_id, reason, created_at) VALUES (?, ?, ?)",
                                     (rowid, str(reason)[:500], now))
                        done.append(rowid)
                    conn.execute("DELETE FROM purge_flag")
                    conn.execute("INSERT INTO items_fts (items_fts) VALUES ('optimize')")
                    conn.execute("INSERT INTO labels_fts (labels_fts) VALUES ('optimize')")
        if done:
            with self._connect() as conn:
                # VACUUM before the checkpoint: in WAL mode VACUUM writes into the WAL, so the
                # checkpoint is what carries the rewritten file into the main database.
                if secure:
                    try:
                        conn.execute("VACUUM")
                    except sqlite3.OperationalError as exc:
                        # The purge committed above; a raise here would read as if it had not.
                        logger.warning("%s: purge committed, but VACUUM failed (%s); free pages "
                                       "stay until a later VACUUM", self.path, exc)
                # No waiting: a TRUNCATE checkpoint otherwise sits out the whole busy timeout behind
                # any open reader, and the purge has already committed by now.
                conn.execute("PRAGMA busy_timeout = 0")
                busy, _, _ = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
                if busy:
                    logger.warning("%s: purge committed, but another connection kept the WAL "
                                   "checkpoint from finishing; the old pages leave the main file "
                                   "at the next full checkpoint", self.path)
            _erase_vectors(self.path, done, secure=secure)
        return [f"m{rowid}" for rowid in done]

    def purged(self, item_id: str) -> PurgeRecord | None:
        """The record of ``item_id``'s purge, or ``None`` if it was never purged."""
        target = _rowid(item_id)
        if target is None:
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT reason, created_at FROM purged WHERE item_id = ? ORDER BY id DESC LIMIT 1",
                (target,)).fetchone()
        return None if row is None else PurgeRecord(f"m{target}", row["reason"], row["created_at"])

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

    def _live_heads(self, chars: int) -> list[tuple[str, str]]:
        """``(id, the first ``chars`` characters of its text)`` for every item not retired,
        oldest first: what the vector index embeds, read without loading whole texts."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, substr(text, 1, ?) FROM items WHERE valid_to IS NULL OR valid_to > ? "
                "ORDER BY id", (int(chars), time.time())).fetchall()
        return [(f"m{r[0]}", r[1]) for r in rows]

    def _has_live_text(self, text: str) -> bool:
        """Whether an item not retired already holds exactly this text."""
        with self._connect() as conn:
            return conn.execute("SELECT 1 FROM items WHERE text = ? AND (valid_to IS NULL OR "
                                "valid_to > ?) LIMIT 1", (text, time.time())).fetchone() is not None

    def _live_ids(self) -> set[str]:
        """The ids of every item not retired."""
        with self._connect() as conn:
            rows = conn.execute("SELECT id FROM items WHERE valid_to IS NULL OR valid_to > ?",
                                (time.time(),)).fetchall()
        return {f"m{r[0]}" for r in rows}

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
        entities: Iterable[str] | str = (),
        topic: str | None = None,
        at: float | None = None,
        include_retired: bool = False,
    ) -> list[MemoryItem]:
        """Items matching ``query``, best first; with an empty query, the newest first.

        - A query matches an item's text or one of its CURRENT labels (a superseded title is not
          searched). Ranking is a stated heuristic: an item matching in a label comes before one
          matching only in its text, because a label says what an item is about; within each
          group, by ``bm25`` and then newest first.
        - A non-empty query that reduces to no terms (punctuation, a lone letter or digit) matches
          nothing, rather than falling back to the newest items.
        - ``provenance`` keeps only the given label or labels.
        - ``tags`` keeps only items carrying ALL of them, each found among the item's own tags or
          its current tag labels (compared casefolded there, as tag labels are stored).
        - ``entities`` keeps only items whose current entity labels name ALL of them, compared
          NFKC-normalised and casefolded (``postgresql`` finds ``PostgreSQL``), each matched by
          every name in its alias group.
        - A query that is itself an entity name also searches every other name in its alias group.
        - ``topic`` keeps only the items that are members of that topic now (an unknown id matches
          nothing).
        - Items retired as of now are left out unless ``include_retired`` is true. One retired at
          a future time stays visible until then, and one whose ``valid_from`` lies in the future
          is NOT left out either.
        - ``at`` is a point-in-time view of validity instead: only items valid at that instant
          (``valid_from <= at`` and not yet retired at ``at``), whatever ``include_retired`` says.
          Labels are always the current ones; a label has no validity window.

        ``limit`` is clamped to at least 1. Raises ``ValueError`` for an unknown provenance.
        """
        kinds = _provenance_set(provenance)
        wanted_tags = _normalise_tags(tags)
        wanted_entities = _normalise_tags(entities)
        limit = max(1, int(limit))
        where: list[str] = []
        params: list[Any] = []
        if kinds is not None:
            where.append(f"items.provenance IN ({', '.join('?' for _ in kinds)})")
            params.extend(kinds)
        for tag in wanted_tags:
            where.append(
                "(EXISTS (SELECT 1 FROM json_each(items.tags_json) WHERE value = ?) OR EXISTS ("
                "SELECT 1 FROM current_labels cl, json_each(cl.value_json) j "
                "WHERE cl.item_id = items.id AND cl.kind = 'tags' AND j.value = ?))"
            )
            params.extend([tag, tag.casefold()])
        aliases = self.aliases() if wanted_entities or (query and query.strip()) else {}
        for entity in wanted_entities:
            where.append(
                "EXISTS (SELECT 1 FROM current_labels cl, json_each(cl.value_json) j "
                "WHERE cl.item_id = items.id AND cl.kind = 'entities' "
                "AND rlm_fold(j.value) IN (SELECT value FROM json_each(?)))"
            )
            params.append(json.dumps(sorted(_name_group(entity, aliases))))
        if topic is not None:
            where.append("EXISTS (SELECT 1 FROM current_members cm "
                         "WHERE cm.item_id = items.id AND cm.topic_id = ?)")
            params.append(_topic_rowid(topic) or 0)
        if at is not None:
            when = _timestamp(at, "at")
            where.append("items.valid_from <= ? AND (items.valid_to IS NULL OR items.valid_to > ?)")
            params.extend([when, when])
        elif not include_retired:
            where.append("(items.valid_to IS NULL OR items.valid_to > ?)")
            params.append(time.time())
        filters = "".join(f" AND {clause}" for clause in where)

        if not (query and query.strip()):
            sql = (
                "SELECT items.* FROM items"
                + (" WHERE " + " AND ".join(where) if where else "")
                + " ORDER BY items.id DESC LIMIT ?"
            )
            with self._connect() as conn:
                rows = conn.execute(sql, [*params, limit]).fetchall()
            return [self._item(row) for row in rows]

        terms = _query_terms(query)
        # A query that IS an entity name searches every name it is known by, so `k8s` finds an
        # item that only ever says `Kubernetes`.
        group = _name_group(query, aliases)
        if len(group) > 1:
            for name in sorted(group):
                terms.extend(t for t in _query_terms(name) if t not in terms)
            terms = terms[:_MAX_QUERY_TERMS]
        if not terms:
            return []
        expression = _match_expression(terms)
        # `bm25()` cannot be aggregated directly and a plain subquery is flattened back into the
        # aggregate, so the label hits are MATERIALIZED first, then reduced to each item's best.
        label_sql = (
            "WITH hits AS MATERIALIZED (SELECT rowid AS label_id, bm25(labels_fts) AS score "
            "FROM labels_fts WHERE labels_fts MATCH ?) "
            "SELECT items.*, MIN(hits.score) AS score FROM hits "
            "JOIN current_labels cl ON cl.id = hits.label_id JOIN items ON items.id = cl.item_id "
            "WHERE 1" + filters + " GROUP BY items.id ORDER BY score, items.id DESC LIMIT ?"
        )
        body_sql = (
            "SELECT items.* FROM items_fts JOIN items ON items.id = items_fts.rowid "
            "WHERE items_fts MATCH ?" + filters
            + f" ORDER BY bm25(items_fts, {_TAG_WEIGHT}, 1.0), items.id DESC LIMIT ?"
        )
        with self._connect() as conn:
            labelled = conn.execute(label_sql, [expression, *params, limit]).fetchall()
            seen = {row["id"] for row in labelled}
            # Over-fetched by the label hits, which may repeat here and are dropped.
            body = conn.execute(body_sql, [expression, *params, limit + len(seen)]).fetchall()
        rows = list(labelled) + [row for row in body if row["id"] not in seen]
        return [self._item(row) for row in rows[:limit]]

    def count(self) -> int:
        """How many items the store holds, retired ones included and purged ones not."""
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
