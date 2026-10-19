"""Local similarity for a :class:`~rlm_harness.memory.MemoryStore`: an embedding per item, computed
on this machine, and the items near it. dspy-free; needs the ``memory-vectors`` extra.

Labels need a summary first, and a summary is a model call the host pays for. An embedding is the
free route: the default model, ``multilingual-e5-small`` quantised to int8, runs on the CPU through
``onnxruntime`` in roughly 70 to 100 milliseconds per item (measured at 1,500 characters, English
and Chinese), so an item nobody has summarised can still be linked to what it is like.

**The model is downloaded when the host asks, never bundled and never implicitly.**
:func:`download_model` fetches the files an :class:`EmbeddingModel` names from the fixed addresses
in it, resuming a partial file, and checks each against its size and SHA-256 before moving it into
place, so a changed or truncated file is refused rather than run. :data:`DEFAULT_MODEL` is this
kit's own choice, made for multilingual text; a host passes its own :class:`EmbeddingModel` to use
another, and with it, optionally, any object with an ``embed`` method to run it.

**The vectors are a derived cache in their own file**, ``<store path>.vectors.db``
(:data:`~rlm_harness.memory.VECTORS_SUFFIX`): plain float32 BLOBs, compared with numpy. Every row
can be rebuilt from the store, so the file holds no evidence and is exempt from the store's
append-only triggers, like its search indexes. No SQLite extension is involved, which is what lets
:meth:`~rlm_harness.memory.MemoryStore.purge` erase a purged item's vector with the standard
library alone, whether or not anything here is open. Each row keeps the SHA-256 of the exact text
it embedded and the model it came from, so :meth:`MemoryVectors.sync` re-embeds an item whose title,
summary or text changed, and an item too short to compare is remembered as skipped until its text
changes.

**Similarity is a weak signal, and it is used as one.** Measured on the default model, cosine
similarity sits in a narrow band (0.75 to 0.95) and leans towards text in the same language. So
two items are called similar only at or above a threshold AND when each is among the other's
nearest neighbours, and a missing link means nothing.

**Two texts in Chinese, Japanese or Korean are held to a higher threshold.** Measured on the
default model over eight topics: unrelated English pairs peaked at 0.819, but unrelated Chinese
pairs averaged 0.856 and reached 0.892, above the 0.86 that separates English, while paraphrases
of one Chinese text scored 0.888 to 0.951. In a Chinese store, 0.86 with mutual neighbours drew
14 wrong links beside 8 right ones; 0.90 drew 7 right and none wrong. So ``cjk_threshold``
applies when both items are mostly CJK text (:func:`is_cjk`), and ``similar_threshold`` to every
other pair, a Chinese and an English text included (unrelated ones peaked at 0.817).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .memory import _UNSPACED, MemoryStore, MemoryUnavailable, _erase_vectors, _rowid, vectors_path

logger = logging.getLogger(__name__)

#: How much of an item is embedded: its current title, its summary, and this many characters from
#: the head of its text. The default model reads at most 512 tokens, and a text states its subject
#: early.
TEXT_CHARS = 1500

#: Text shorter than this is not compared at all, counted by :func:`text_weight`. Measured on the
#: default model: an eleven-character note ("untidy notes") scored 0.875 against an unrelated
#: paper, because a vector of almost nothing sits near everything.
MIN_TEXT = 40

#: How much one CJK character counts towards ``MIN_TEXT``: a Chinese character carries about as
#: much as a short English word. Measured on the default model, a Chinese note's links to the
#: long text on its own subject reached the CJK threshold on average from about 14 characters, and
#: no note of 6 to 20 characters crossed it against an unrelated text (at most 0.889), so 40 / 3
#: lets a 14-character note in.
CJK_WEIGHT = 3

#: How many nearest neighbours count when deciding whether two items are mutual neighbours.
NEIGHBOURS = 4

#: How many items one ``sync`` call embeds at most, whatever ``limit`` asks for.
MAX_SYNC = 500

_BATCH = 16
_BUSY_TIMEOUT_S = 30.0


@dataclass(frozen=True)
class ModelFile:
    """One file of a model: where it is fetched from, and the size and SHA-256 it must have."""

    url: str
    size: int
    sha256: str


@dataclass(frozen=True)
class EmbeddingModel:
    """An embedding model pinned to one upstream revision.

    ``files`` maps a local file name to its :class:`ModelFile`; an ONNX model is ``model.onnx`` and
    its tokenizer ``tokenizer.json``. ``prefix`` is put before every text (the default model's
    authors prescribe ``"query: "`` for symmetric similarity between two passages), and
    ``max_tokens`` is where the tokenizer truncates. Two items are similar only at or above
    ``similar_threshold``, or ``cjk_threshold`` when both are mostly CJK text (``None`` uses
    ``similar_threshold`` for those too). Both belong to the model: cosine values are not
    comparable across models.
    """

    name: str
    revision: str
    files: Mapping[str, ModelFile] = field(hash=False)
    dimensions: int
    prefix: str = ""
    max_tokens: int = 512
    similar_threshold: float = 0.86
    cjk_threshold: float | None = None

    @property
    def key(self) -> str:
        """``name@revision``: which model a stored vector came from."""
        return f"{self.name}@{self.revision}"

    @property
    def total_bytes(self) -> int:
        return sum(f.size for f in self.files.values())


_E5_REVISION = "761b726dd34fb83930e26aab4e9ac3899aa1fa78"
_E5_BASE = f"https://huggingface.co/Xenova/multilingual-e5-small/resolve/{_E5_REVISION}"

#: The kit's default: ``multilingual-e5-small``, int8 quantised, about 135 MB. A new revision is a
#: deliberate change here, with new sizes and hashes.
DEFAULT_MODEL = EmbeddingModel(
    name="multilingual-e5-small-int8",
    revision=_E5_REVISION,
    files={
        "model.onnx": ModelFile(
            url=f"{_E5_BASE}/onnx/model_quantized.onnx",
            size=118_308_185,
            sha256="f80102d3f2a1229f387d3c81909990d8945513e347b0eab049f7de3c6f98c193",
        ),
        "tokenizer.json": ModelFile(
            url=f"{_E5_BASE}/tokenizer.json",
            size=17_082_730,
            sha256="0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39",
        ),
    },
    dimensions=384,
    prefix="query: ",
    max_tokens=512,
    similar_threshold=0.86,
    cjk_threshold=0.90,
)


def _cjk_count(text: str) -> int:
    return sum(len(run) for run in _UNSPACED.findall(text))


def text_weight(text: str) -> int:
    """A text's length for ``MIN_TEXT``: its characters, each CJK one counted ``CJK_WEIGHT`` times."""
    return len(text) + (CJK_WEIGHT - 1) * _cjk_count(text)


def is_cjk(text: str) -> bool:
    """Whether at least half of a text's letters are CJK characters (kana, ideographs, Hangul)."""
    cjk = _cjk_count(text)
    return cjk > 0 and 2 * cjk >= sum(1 for ch in text if ch.isalpha())


class Embedder(Protocol):
    """Anything that turns texts into vectors, one per text, in order."""

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class DownloadStopped(Exception):
    """``should_stop`` asked a download to stop. What was fetched so far is kept for a resume."""


# ---- download ---------------------------------------------------------------------------------


def model_installed(model: EmbeddingModel, folder: str | os.PathLike[str]) -> bool:
    """The model's files are all in ``folder`` at their published sizes, and a download of THIS
    revision verified them. Sizes are checked here; hashes were checked when they were fetched."""
    folder = Path(folder)
    try:
        marker = json.loads((folder / "verified.json").read_text("utf-8"))
    except (OSError, ValueError):
        return False
    return (isinstance(marker, dict) and marker.get("revision") == model.revision
            and all((folder / name).is_file() and (folder / name).stat().st_size == spec.size
                    for name, spec in model.files.items()))


def download_model(
    model: EmbeddingModel,
    dest: str | os.PathLike[str],
    *,
    on_progress: Callable[[int, int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    opener: Callable[..., Any] | None = None,
) -> Path:
    """Fetch ``model``'s files into ``dest`` and return the folder. Never called implicitly.

    A partial file is resumed with an HTTP Range request. Each file is checked against its size and
    SHA-256 before it is moved into place; one that does not match is deleted and ``ValueError`` is
    raised, so nothing unverified is ever installed. A file already in place is kept only when it
    still matches its size and SHA-256, and a server that sends more than the published size is cut
    off and refused.
    ``on_progress(done_bytes, total_bytes)`` is called after every chunk, and ``should_stop`` before
    every chunk (raising :class:`DownloadStopped`, with the partial file kept for a resume).
    ``opener`` defaults to ``urllib.request.urlopen``. Only ``https`` addresses are fetched.
    """
    opener = opener or urllib.request.urlopen
    folder = Path(dest)
    folder.mkdir(parents=True, exist_ok=True)
    done = 0
    for name, spec in model.files.items():
        if not spec.url.startswith("https://"):
            raise ValueError(f"{name}: refusing to fetch a model file over anything but https")
        if Path(name).name != name or name in ("", ".", "..", "verified.json"):
            raise ValueError(f"{name!r} is not a plain file name")
        final = folder / name
        if final.is_file() and final.stat().st_size == spec.size and _sha256(final) == spec.sha256:
            done += spec.size
            continue
        part = folder / f"{name}.part"
        have = part.stat().st_size if part.exists() else 0
        if have > spec.size:
            part.unlink()
            have = 0
        if have < spec.size:
            _fetch(spec, part, have, done, model.total_bytes, opener, on_progress, should_stop)
        # A part already at full size (stopped during the last chunk, or during hashing) goes
        # straight to verification: asking for `bytes=<size>-` would be answered 416 forever.
        if part.stat().st_size != spec.size or _sha256(part) != spec.sha256:
            part.unlink(missing_ok=True)
            raise ValueError(f"{name} did not match its published size and checksum; nothing was installed")
        os.replace(part, final)
        done += spec.size
    marker = folder / "verified.json.tmp"
    marker.write_text(json.dumps({"name": model.name, "revision": model.revision}), "utf-8")
    os.replace(marker, folder / "verified.json")
    return folder


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fetch(spec: ModelFile, part: Path, have: int, done: int, total: int, opener, on_progress,
           should_stop) -> None:
    request = urllib.request.Request(spec.url, headers={"User-Agent": "rlm-harness"})
    if have:
        request.add_header("Range", f"bytes={have}-")
    try:
        response = opener(request, timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code != 416 or not have:
            raise
        # The server has nothing past what we hold, so what we hold is not a prefix it recognises:
        # start the file over rather than failing the same way on every attempt.
        part.unlink(missing_ok=True)
        return _fetch(spec, part, 0, done, total, opener, on_progress, should_stop)
    with response:
        if have and getattr(response, "status", 200) != 206:
            have = 0   # the server ignored the range: start this file over
        with part.open("ab" if have else "wb") as out:
            while True:
                if should_stop is not None and should_stop():
                    raise DownloadStopped()
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                have += len(chunk)
                if have > spec.size:
                    break   # longer than published: the size check below refuses it
                if on_progress is not None:
                    on_progress(done + have, total)


# ---- the embedder -----------------------------------------------------------------------------


def _real_numpy():
    """Make ``numpy`` in ``sys.modules`` the real module, and return it.

    dspy installs a lazy proxy for ``numpy`` that loads on first attribute access. ``onnxruntime``'s
    compiled module imports numpy through the C API, which gets the proxy and fails with "import
    numpy failed"; measured in a server that had imported dspy, every embedding failed so. Touching
    one attribute swaps the real module in. Raises :class:`MemoryUnavailable` without numpy.
    """
    try:
        import numpy
    except ImportError as exc:
        raise MemoryUnavailable(
            "local similarity needs numpy; install rlm-harness[memory-vectors]") from exc
    numpy.ndarray  # noqa: B018 - the attribute access IS the point
    return sys.modules["numpy"]


class OnnxEmbedder:
    """An :class:`EmbeddingModel` run on the CPU through ``onnxruntime``: mean-pooled, unit-length
    vectors. The runtime is imported here, so a host that never embeds never loads it."""

    def __init__(self, model: EmbeddingModel, folder: str | os.PathLike[str], *, threads: int = 2) -> None:
        folder = Path(folder)
        if not model_installed(model, folder):
            raise MemoryUnavailable(f"{model.name} is not installed in {folder}; call download_model first")
        np = _real_numpy()
        try:
            import onnxruntime
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise MemoryUnavailable(
                "embedding needs onnxruntime and tokenizers; install rlm-harness[memory-vectors]") from exc
        self._np = np
        self._model = model
        self._tok = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self._tok.enable_truncation(model.max_tokens)
        self._tok.enable_padding()
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = max(1, int(threads))   # a background job, not the foreground
        self._session = onnxruntime.InferenceSession(
            str(folder / "model.onnx"), options, providers=["CPUExecutionProvider"])
        self._inputs = {i.name for i in self._session.get_inputs()}

    def embed(self, texts: list[str]) -> list[list[float]]:
        np = self._np
        encoded = self._tok.encode_batch([f"{self._model.prefix}{t}" for t in texts])
        ids = np.array([e.ids for e in encoded], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._inputs:
            feeds["token_type_ids"] = np.zeros_like(ids)
        hidden = self._session.run(None, feeds)[0]
        weights = mask[..., None].astype(np.float32)
        pooled = (hidden * weights).sum(1) / np.maximum(weights.sum(1), 1e-9)
        pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
        return pooled.astype(np.float32).tolist()


# ---- the index --------------------------------------------------------------------------------

#: The vector file's own schema ladder, counted in its ``PRAGMA user_version``. A shipped step is
#: never edited; a change appends one.
_VECTOR_MIGRATIONS: tuple[str, ...] = (
    # Step 1. `embedding` NULL means considered and skipped (too short); `text_hash` is the SHA-256
    # of the model key and the exact text, so a changed text or a changed model re-embeds; `cjk`
    # says the text is mostly CJK, which picks the threshold a pair is held to.
    """
    CREATE TABLE vectors (
        item_id     INTEGER PRIMARY KEY CHECK (item_id > 0),
        model       TEXT NOT NULL,
        text_hash   TEXT NOT NULL,
        embedding   BLOB,
        cjk         INTEGER NOT NULL DEFAULT 0,
        embedded_at REAL NOT NULL
    );
    """,
)


@dataclass(frozen=True)
class SimilarPair:
    """Two items that are similar enough and each among the other's nearest. ``a < b`` by id."""

    a: str
    b: str
    score: float


@dataclass
class VectorSyncReport:
    """What one :meth:`MemoryVectors.sync` did. ``waiting`` counts items still to embed."""

    embedded: int = 0
    skipped: int = 0
    removed: int = 0
    waiting: int = 0
    stopped: bool = False


class _Snapshot:
    """The vectors of the live items, loaded once, for several comparisons in a row."""

    def __init__(self, np, ids: list[str], matrix, cjk: list[bool], threshold: float,
                 cjk_threshold: float) -> None:
        self.np = np
        self.ids = ids
        self.index = {item_id: i for i, item_id in enumerate(ids)}
        self.matrix = matrix
        self.cjk = cjk
        self.threshold = threshold
        self.cjk_threshold = cjk_threshold

    def _bar(self, a: int, b: int) -> float:
        return self.cjk_threshold if self.cjk[a] and self.cjk[b] else self.threshold

    def _top(self, row: int, pool, k: int) -> list[tuple[int, float]]:
        # Scores against the pool only (`None` is every item); ties go to the lower id, since
        # `ids` is in id order and the sort is stable.
        np = self.np
        if pool is None:
            pool = np.arange(len(self.ids))
            scores = self.matrix @ self.matrix[row]
        else:
            scores = self.matrix[pool] @ self.matrix[row]
        keep = pool != row
        pool, scores = pool[keep], scores[keep]
        order = np.argsort(-scores, kind="stable")[:k]
        return [(int(pool[i]), float(scores[i])) for i in order]

    def similar(self, item_id: str, *, k: int, among: Iterable[str] | None) -> list[tuple[str, float]]:
        np = self.np
        row = self.index.get(item_id)
        if row is None:
            return []
        pool = None
        if among is not None:
            wanted = {self.index[i] for i in among if i in self.index} | {row}
            pool = np.array(sorted(wanted), dtype=np.int64)
        out = []
        for other, score in self._top(row, pool, k):
            # Mutual: the item is among the other's nearest within the same pool, which filters the
            # one-way pull a generic text has on everything.
            if score >= self._bar(row, other) and row in {j for j, _ in self._top(other, pool, k)}:
                out.append((self.ids[other], score))
        return out

    def pairs(self, item_ids: Iterable[str], *, k: int) -> list[SimilarPair]:
        np = self.np
        pool = np.array(sorted({self.index[i] for i in item_ids if i in self.index}), dtype=np.int64)
        if len(pool) < 2:
            return []
        near: dict[int, dict[int, float]] = {}
        sub = self.matrix[pool]
        for start in range(0, len(pool), 512):
            block = sub[start:start + 512] @ sub.T
            for offset, scores in enumerate(block):
                me = start + offset
                scores = scores.copy()
                scores[me] = -np.inf
                order = np.argsort(-scores, kind="stable")[:k]
                near[me] = {int(j): float(scores[j]) for j in order}
        out = []
        for a, others in near.items():
            for b, score in others.items():
                if a < b and score >= self._bar(pool[a], pool[b]) and a in near[b]:
                    out.append(SimilarPair(self.ids[pool[a]], self.ids[pool[b]], score))
        out.sort(key=lambda p: (-p.score, _rowid(p.a), _rowid(p.b)))
        return out


class MemoryVectors:
    """The similarity index of one store, kept in ``<store path>.vectors.db``.

    ``model_dir`` is where :func:`download_model` put the model; the model itself is loaded on the
    first :meth:`sync` that has something to embed. ``embedder`` replaces it with any object that has
    an ``embed(texts) -> vectors`` method (a vector is normalised on the way in, and must have
    ``model.dimensions`` values). Raises :class:`~rlm_harness.memory.MemoryUnavailable` without
    numpy, and ``RuntimeError`` for a vector file written by a newer rlm-harness.
    """

    def __init__(
        self,
        store: MemoryStore,
        model_dir: str | os.PathLike[str] | None = None,
        *,
        model: EmbeddingModel | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        if model_dir is None and embedder is None:
            raise ValueError("MemoryVectors needs a model_dir to load the model from, or an embedder")
        if embedder is not None and model is None:
            # Stamped with the default's key and judged by its threshold, a foreign model's vectors
            # would be compared with e5's and never re-embedded when the embedder changed.
            raise ValueError("an embedder needs model= naming it: its key, dimensions and threshold")
        model = model or DEFAULT_MODEL
        self._np = _real_numpy()
        self.store = store
        self.model = model
        self.model_dir = None if model_dir is None else Path(model_dir)
        self.path = vectors_path(store.path)
        self._embedder = embedder
        self._lock = threading.Lock()
        self._migrate()

    # ---- plumbing ----

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=_BUSY_TIMEOUT_S, isolation_level=None)
        # Freed pages are zeroed, so a vector deleted by a re-embed or a purge does not linger.
        conn.execute("PRAGMA secure_delete = ON")
        return conn

    def _migrate(self) -> None:
        conn = self._connect()
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("BEGIN IMMEDIATE")
            try:
                version = conn.execute("PRAGMA user_version").fetchone()[0]
                if version > len(_VECTOR_MIGRATIONS):
                    raise RuntimeError(f"{self.path} was written by a newer rlm-harness (vector schema "
                                       f"{version}, this one knows {len(_VECTOR_MIGRATIONS)})")
                for step in _VECTOR_MIGRATIONS[version:]:
                    for statement in [s for s in step.split(";") if s.strip()]:
                        conn.execute(statement)
                conn.execute(f"PRAGMA user_version = {len(_VECTOR_MIGRATIONS)}")
                conn.execute("COMMIT")
            except BaseException:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
        finally:
            conn.close()

    def _get_embedder(self) -> Embedder:
        with self._lock:
            if self._embedder is None:
                self._embedder = OnnxEmbedder(self.model, self.model_dir)
            return self._embedder

    def _texts(self) -> list[tuple[str, str]]:
        """``(id, text to embed)`` for every live item: current title, summary, head of the text."""
        labels = self.store._labels_by_item()
        out = []
        for item_id, head in self.store._live_heads(TEXT_CHARS):
            current = labels.get(item_id)
            parts = [current.title, current.summary] if current is not None else []
            out.append((item_id, "\n".join(p for p in [*parts, head or ""] if p).strip()))
        return out

    def _hash(self, text: str) -> str:
        return hashlib.sha256(f"{self.model.key}\n{text}".encode()).hexdigest()

    def _vector(self, values: Any) -> bytes | None:
        np = self._np
        vector = np.asarray(values, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.model.dimensions:
            raise ValueError(f"the embedder returned {vector.shape[0]} values; "
                             f"{self.model.name} has {self.model.dimensions}")
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm == 0.0:
            return None   # nothing to compare: stored as skipped
        return (vector / norm).astype("<f4").tobytes()

    # ---- writing ----

    def sync(self, *, limit: int = 100, should_stop: Callable[[], bool] | None = None) -> VectorSyncReport:
        """Embed live items that have no vector, or whose text or model changed since theirs was made,
        newest first and at most ``limit`` (clamped to ``[1, 500]``); drop the vectors of items that
        are retired or gone. An item whose text weighs under ``MIN_TEXT`` (:func:`text_weight`) is
        recorded as skipped and not read again until its text changes. ``should_stop`` is checked
        before each batch of 16. No model call: the embedding model runs on this machine.
        """
        limit = max(1, min(MAX_SYNC, int(limit)))
        report = VectorSyncReport()
        texts = self._texts()
        live = {item_id for item_id, _ in texts}
        conn = self._connect()
        try:
            stored = {f"m{r[0]}": r[1] for r in conn.execute("SELECT item_id, text_hash FROM vectors")}
            gone = [_rowid(i) for i in stored if i not in live]
            if gone:
                conn.execute("DELETE FROM vectors WHERE item_id IN (SELECT value FROM json_each(?))",
                             (json.dumps(gone),))
                report.removed = len(gone)
        finally:
            conn.close()
        todo = [(i, t, self._hash(t)) for i, t in reversed(texts) if stored.get(i) != self._hash(t)]
        report.waiting = len(todo)
        for start in range(0, min(limit, len(todo)), _BATCH):
            if should_stop is not None and should_stop():
                report.stopped = True
                break
            batch = todo[start:min(start + _BATCH, limit)]
            wanted = [n for n, (_, text, _) in enumerate(batch) if text_weight(text) >= MIN_TEXT]
            vectors = self._get_embedder().embed([batch[n][1] for n in wanted]) if wanted else []
            if len(vectors) != len(wanted):
                raise ValueError(f"the embedder returned {len(vectors)} vectors for {len(wanted)} texts")
            blobs = dict(zip(wanted, (self._vector(v) for v in vectors), strict=True))
            rows = [(_rowid(item_id), self.model.key, digest, blobs.get(n), int(is_cjk(text)), time.time())
                    for n, (item_id, text, digest) in enumerate(batch)]
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                # A skipped row already holding this very text and model was written by a `forget`
                # since `todo` was read; it stands, so a concurrent sync cannot undo a forget.
                kept = [r for r in rows if conn.execute(
                    "INSERT INTO vectors (item_id, model, text_hash, embedding, cjk, embedded_at) "
                    "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(item_id) DO UPDATE SET model = excluded.model, "
                    "text_hash = excluded.text_hash, embedding = excluded.embedding, "
                    "cjk = excluded.cjk, embedded_at = excluded.embedded_at WHERE NOT (vectors.embedding IS NULL AND "
                    "vectors.model = excluded.model AND vectors.text_hash = excluded.text_hash)",
                    r).rowcount == 0]
                conn.execute("COMMIT")
            finally:
                conn.close()
            rows = [r for r in rows if r not in kept]
            report.waiting -= len(kept)
            # Checked AFTER the write: a purge that committed meanwhile either shows here, or erased
            # its vectors after this write landed. Either way no purged item keeps a vector.
            still = self.store._live_ids()
            vanished = [r[0] for r in rows if f"m{r[0]}" not in still]
            if vanished:
                _erase_vectors(self.store.path, vanished, secure=False)
                report.waiting -= len(vanished)
            report.embedded += sum(1 for r in rows if r[3] is not None and r[0] not in vanished)
            report.skipped += sum(1 for r in rows if r[3] is None and r[0] not in vanished)
        report.waiting -= report.embedded + report.skipped
        return report

    def forget(self, item_ids: Iterable[str], *, secure: bool = False) -> None:
        """Erase these items' vectors now and keep them out until their text changes.

        :meth:`~rlm_harness.memory.MemoryStore.purge` already erases what it purges; this is for a
        host that wants a vector gone without the item. A live item is left with a skipped row
        holding its current text hash, so ``sync`` does not embed it again until its title,
        summary or text changes, even if a ``sync`` was already running when it was called."""
        if isinstance(item_ids, str):
            raise TypeError("forget takes a list of ids, not a str")
        wanted = {i for i in item_ids if _rowid(i) is not None}
        _erase_vectors(self.store.path, sorted(_rowid(i) for i in wanted), secure=secure)
        rows = [(_rowid(i), self.model.key, self._hash(t), None, int(is_cjk(t)), time.time())
                for i, t in self._texts() if i in wanted]
        if rows:
            conn = self._connect()
            try:
                conn.executemany("INSERT OR REPLACE INTO vectors (item_id, model, text_hash, embedding, "
                                 "cjk, embedded_at) VALUES (?, ?, ?, ?, ?, ?)", rows)
            finally:
                conn.close()
            # As in `sync`: a purge that committed meanwhile shows here, or erased after this write.
            still = self.store._live_ids()
            vanished = [r[0] for r in rows if f"m{r[0]}" not in still]
            if vanished:
                _erase_vectors(self.store.path, vanished, secure=secure)

    # ---- reading ----

    def count(self) -> int:
        """How many items have a vector from this model."""
        conn = self._connect()
        try:
            return int(conn.execute("SELECT COUNT(*) FROM vectors WHERE model = ? AND embedding IS NOT NULL",
                                    (self.model.key,)).fetchone()[0])
        finally:
            conn.close()

    def _snapshot(self) -> _Snapshot:
        np = self._np
        live = self.store._live_ids()
        conn = self._connect()
        try:
            rows = conn.execute("SELECT item_id, embedding, cjk FROM vectors WHERE model = ? AND "
                                "embedding IS NOT NULL ORDER BY item_id", (self.model.key,)).fetchall()
        finally:
            conn.close()
        rows = [r for r in rows if f"m{r[0]}" in live]
        matrix = (np.frombuffer(b"".join(r[1] for r in rows), dtype="<f4").reshape(len(rows), -1)
                  if rows else np.zeros((0, self.model.dimensions), dtype=np.float32))
        threshold = self.model.similar_threshold
        cjk_threshold = threshold if self.model.cjk_threshold is None else self.model.cjk_threshold
        return _Snapshot(np, [f"m{r[0]}" for r in rows], matrix, [bool(r[2]) for r in rows],
                         threshold, cjk_threshold)

    def similar(self, item_id: str, *, k: int = NEIGHBOURS,
                among: Iterable[str] | None = None) -> list[tuple[str, float]]:
        """The items at or above the pair's threshold that are mutual near neighbours of
        ``item_id``, best first, as ``(id, cosine)``. With ``among``, only those items count, both
        as candidates and as the crowd each side's nearest are counted in. An item with no vector
        yet has no matches; an empty result means nothing."""
        if isinstance(among, str):
            raise TypeError("among takes a list of ids, not a str")
        return self._snapshot().similar(item_id, k=max(1, int(k)), among=among)

    def similar_pairs(self, item_ids: Iterable[str]) -> list[SimilarPair]:
        """Every pair among ``item_ids`` at or above the threshold whose two items are each among
        the other's ``NEIGHBOURS`` nearest within ``item_ids``, strongest first."""
        if isinstance(item_ids, str):
            raise TypeError("similar_pairs takes a list of ids, not a str")
        return self._snapshot().pairs(item_ids, k=NEIGHBOURS)
