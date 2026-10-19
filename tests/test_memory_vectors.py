"""Local similarity: the model download, the vector file, and mutual near neighbours. Offline.

numpy is a dev dependency, so the comparison runs for real; the model runtime (onnxruntime,
tokenizers) is faked through ``sys.modules`` and never installed here.
"""

from __future__ import annotations

import hashlib
import io
import math
import sqlite3
import subprocess
import sys
import types
import urllib.error

import pytest

from rlm_harness.memory import MemoryStore, MemoryUnavailable, vectors_path
from rlm_harness.memory_organise import SUGGEST_THRESHOLD, suggest_topics
from rlm_harness.memory_vectors import (
    CJK_WEIGHT,
    DEFAULT_MODEL,
    MIN_TEXT,
    DownloadStopped,
    EmbeddingModel,
    MemoryVectors,
    ModelFile,
    OnnxEmbedder,
    SimilarPair,
    download_model,
    is_cjk,
    model_installed,
    text_weight,
)

TINY = EmbeddingModel(name="tiny", revision="r1", files={}, dimensions=3, similar_threshold=0.86)


def _unit(*values):
    norm = math.sqrt(sum(v * v for v in values))
    return [v / norm for v in values]


class Table:
    """An embedder that looks each text's keyword up in a table, and counts what it was asked."""

    def __init__(self, table):
        self.table = table
        self.calls: list[list[str]] = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [next(v for k, v in self.table.items() if k in text) for text in texts]


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path / "m.db")


def _add(store, keyword, filler="a text long enough to be compared with its neighbours"):
    return store.add(f"{keyword} {filler}", provenance="observed", origin="o").id


VECTORS = {
    "alpha": _unit(1, 0, 0),
    "beta": _unit(0.95, 0.31, 0),     # nearest to alpha (0.95)
    "gamma": _unit(0.9, 0, 0.43),     # near alpha (0.90), but alpha's nearest is beta
    "delta": _unit(0, 1, 0),          # near nothing
}


def _indexed(store, keywords=("alpha", "beta", "gamma", "delta")):
    ids = {k: _add(store, k) for k in keywords}
    vectors = MemoryVectors(store, model=TINY, embedder=Table(VECTORS))
    vectors.sync()
    return ids, vectors


# ---- the package stays light -------------------------------------------------------------------


def test_importing_the_module_loads_neither_dspy_nor_numpy_nor_the_runtime():
    code = ("import sys, rlm_harness.memory_vectors, rlm_harness.memory_organise; "
            "bad = [m for m in ('dspy', 'numpy', 'onnxruntime', 'tokenizers') if m in sys.modules]; "
            "assert not bad, bad; print('ok')")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr


def test_without_numpy_the_index_is_unavailable(store, monkeypatch):
    monkeypatch.setitem(sys.modules, "numpy", None)
    with pytest.raises(MemoryUnavailable, match="memory-vectors"):
        MemoryVectors(store, model=TINY, embedder=Table(VECTORS))


def test_the_index_needs_a_model_folder_or_an_embedder(store):
    with pytest.raises(ValueError, match="model_dir"):
        MemoryVectors(store, model=TINY)


def test_an_embedder_must_name_its_model(store):
    with pytest.raises(ValueError, match="model="):
        MemoryVectors(store, embedder=Table(VECTORS))


def test_the_default_model_is_pinned():
    assert DEFAULT_MODEL.revision == "761b726dd34fb83930e26aab4e9ac3899aa1fa78"
    assert DEFAULT_MODEL.dimensions == 384 and DEFAULT_MODEL.similar_threshold == 0.86
    assert DEFAULT_MODEL.prefix == "query: " and DEFAULT_MODEL.total_bytes == 118_308_185 + 17_082_730
    assert all(f.url.startswith("https://") and len(f.sha256) == 64 for f in DEFAULT_MODEL.files.values())


# ---- download -----------------------------------------------------------------------------------


def _model_for(payloads: dict[str, bytes]) -> EmbeddingModel:
    return EmbeddingModel(
        name="fake", revision="rev", dimensions=3,
        files={n: ModelFile(f"https://example.test/{n}", len(b), hashlib.sha256(b).hexdigest())
               for n, b in payloads.items()})


class Response(io.BytesIO):
    def __init__(self, data, status=200):
        super().__init__(data)
        self.status = status


def _opener(payloads, *, honour_range=True, seen=None):
    def open_(request, timeout=None):
        name = request.full_url.rsplit("/", 1)[1]
        data = payloads[name]
        rng = request.get_header("Range")
        if seen is not None:
            seen.append((name, rng))
        if rng and honour_range:
            start = int(rng.split("=")[1].rstrip("-"))
            return Response(data[start:], status=206)
        return Response(data)
    return open_


PAYLOADS = {"model.onnx": b"m" * 3_000_000, "tokenizer.json": b'{"t": 1}' * 1000}


def test_a_download_verifies_and_installs_every_file(tmp_path):
    model = _model_for(PAYLOADS)
    progress = []
    folder = download_model(model, tmp_path / "model", opener=_opener(PAYLOADS),
                            on_progress=lambda done, total: progress.append((done, total)))
    assert model_installed(model, folder)
    assert (folder / "model.onnx").read_bytes() == PAYLOADS["model.onnx"]
    assert progress[-1] == (model.total_bytes, model.total_bytes)
    assert not list(folder.glob("*.part"))


def test_a_file_that_does_not_match_its_checksum_is_refused_and_removed(tmp_path):
    model = _model_for(PAYLOADS)
    bad = dict(PAYLOADS, **{"tokenizer.json": b"X" * len(PAYLOADS["tokenizer.json"])})
    with pytest.raises(ValueError, match="checksum"):
        download_model(model, tmp_path / "model", opener=_opener(bad))
    assert not (tmp_path / "model" / "tokenizer.json").exists()
    assert not (tmp_path / "model" / "tokenizer.json.part").exists()
    assert not model_installed(model, tmp_path / "model")


def test_a_download_resumes_a_partial_file(tmp_path):
    model = _model_for(PAYLOADS)
    folder = tmp_path / "model"
    folder.mkdir()
    (folder / "model.onnx.part").write_bytes(PAYLOADS["model.onnx"][:1_000_000])
    seen = []
    download_model(model, folder, opener=_opener(PAYLOADS, seen=seen))
    assert ("model.onnx", "bytes=1000000-") in seen
    assert model_installed(model, folder)


def test_a_server_that_ignores_the_range_starts_the_file_over(tmp_path):
    model = _model_for(PAYLOADS)
    folder = tmp_path / "model"
    folder.mkdir()
    (folder / "model.onnx.part").write_bytes(PAYLOADS["model.onnx"][:1_000_000])
    download_model(model, folder, opener=_opener(PAYLOADS, honour_range=False))
    assert model_installed(model, folder)


def test_a_range_the_server_cannot_satisfy_starts_the_file_over(tmp_path):
    model = _model_for(PAYLOADS)
    folder = tmp_path / "model"
    folder.mkdir()
    (folder / "model.onnx.part").write_bytes(b"z" * 10)
    plain = _opener(PAYLOADS)

    def open_(request, timeout=None):
        if request.get_header("Range"):
            raise urllib.error.HTTPError(request.full_url, 416, "range", {}, None)
        return plain(request, timeout)

    download_model(model, folder, opener=open_)
    assert model_installed(model, folder)


def test_a_stopped_download_keeps_its_partial_file_for_a_resume(tmp_path):
    model = _model_for(PAYLOADS)
    calls = iter([False, True])
    with pytest.raises(DownloadStopped):
        download_model(model, tmp_path / "model", opener=_opener(PAYLOADS), should_stop=lambda: next(calls))
    assert (tmp_path / "model" / "model.onnx.part").stat().st_size == 1 << 20
    assert not model_installed(model, tmp_path / "model")


def test_a_complete_partial_file_is_verified_without_fetching_again(tmp_path):
    model = _model_for(PAYLOADS)
    folder = tmp_path / "model"
    folder.mkdir()
    for name, data in PAYLOADS.items():
        (folder / f"{name}.part").write_bytes(data)
    seen = []
    download_model(model, folder, opener=_opener(PAYLOADS, seen=seen))
    assert seen == [] and model_installed(model, folder)


def test_a_file_in_place_that_no_longer_matches_is_fetched_again(tmp_path):
    model = _model_for(PAYLOADS)
    folder = tmp_path / "model"
    folder.mkdir()
    (folder / "model.onnx").write_bytes(b"x" * len(PAYLOADS["model.onnx"]))
    download_model(model, folder, opener=_opener(PAYLOADS))
    assert (folder / "model.onnx").read_bytes() == PAYLOADS["model.onnx"]


def test_a_server_that_sends_more_than_the_published_size_is_cut_off(tmp_path):
    model = _model_for(PAYLOADS)

    class Endless(Response):
        def read(self, n=-1):
            return b"m" * (1 << 20)   # never ends: without the cut-off this test would not finish

    with pytest.raises(ValueError, match="checksum"):
        download_model(model, tmp_path / "model", opener=lambda request, timeout=None: Endless(b""))
    assert not (tmp_path / "model" / "model.onnx.part").exists()


def test_only_https_and_plain_file_names_are_accepted(tmp_path):
    plain = EmbeddingModel(name="x", revision="r", dimensions=3,
                           files={"m": ModelFile("http://example.test/m", 1, "0" * 64)})
    with pytest.raises(ValueError, match="https"):
        download_model(plain, tmp_path, opener=_opener({}))
    escaping = EmbeddingModel(name="x", revision="r", dimensions=3,
                              files={"../m": ModelFile("https://example.test/m", 1, "0" * 64)})
    with pytest.raises(ValueError, match="plain file name"):
        download_model(escaping, tmp_path, opener=_opener({}))


def test_another_revision_does_not_count_as_installed(tmp_path):
    model = _model_for(PAYLOADS)
    folder = download_model(model, tmp_path / "model", opener=_opener(PAYLOADS))
    other = EmbeddingModel(name="fake", revision="newer", dimensions=3, files=model.files)
    assert not model_installed(other, folder)


# ---- the embedder -------------------------------------------------------------------------------


def _fake_runtime(monkeypatch, *, token_types=False):
    import numpy as np

    seen = {}

    class Encoding:
        def __init__(self, text):
            seen.setdefault("texts", []).append(text)
            self.ids = [1, 2, 3]
            self.attention_mask = [1, 1, 0]

    class Tokenizer:
        @staticmethod
        def from_file(path):
            tok = Tokenizer()
            seen["tokenizer"] = path
            return tok

        def enable_truncation(self, n):
            seen["truncation"] = n

        def enable_padding(self):
            pass

        def encode_batch(self, texts):
            return [Encoding(t) for t in texts]

    class Session:
        def __init__(self, path, options, providers):
            seen["providers"] = providers

        def get_inputs(self):
            names = ["input_ids", "attention_mask"] + (["token_type_ids"] if token_types else [])
            return [types.SimpleNamespace(name=n) for n in names]

        def run(self, _, feeds):
            seen["feeds"] = sorted(feeds)
            batch = feeds["input_ids"].shape[0]
            # Token 3 is padding (mask 0); a mean over the first two gives (3, 4, 0) per text.
            hidden = np.array([[[2.0, 4.0, 0.0], [4.0, 4.0, 0.0], [100.0, 0.0, 0.0]]] * batch)
            return [hidden]

    monkeypatch.setitem(sys.modules, "onnxruntime", types.SimpleNamespace(
        SessionOptions=lambda: types.SimpleNamespace(intra_op_num_threads=0), InferenceSession=Session))
    monkeypatch.setitem(sys.modules, "tokenizers", types.SimpleNamespace(Tokenizer=Tokenizer))
    return seen


def _installed_tiny(tmp_path):
    payloads = {"model.onnx": b"m" * 10, "tokenizer.json": b"{}"}
    model = EmbeddingModel(
        name="tiny", revision="r1", dimensions=3, prefix="query: ", max_tokens=128,
        files={n: ModelFile(f"https://example.test/{n}", len(b), hashlib.sha256(b).hexdigest())
               for n, b in payloads.items()})
    return model, download_model(model, tmp_path / "model", opener=_opener(payloads))


@pytest.mark.parametrize("token_types", [False, True])
def test_the_onnx_embedder_prefixes_pools_over_the_mask_and_normalises(tmp_path, monkeypatch, token_types):
    seen = _fake_runtime(monkeypatch, token_types=token_types)
    model, folder = _installed_tiny(tmp_path)
    vectors = OnnxEmbedder(model, folder).embed(["one", "two"])
    assert seen["texts"] == ["query: one", "query: two"] and seen["truncation"] == 128
    assert seen["providers"] == ["CPUExecutionProvider"]
    assert ("token_type_ids" in seen["feeds"]) is token_types
    assert vectors == [pytest.approx([0.6, 0.8, 0.0])] * 2


def test_the_onnx_embedder_refuses_a_model_that_is_not_installed(tmp_path, monkeypatch):
    _fake_runtime(monkeypatch)
    model, folder = _installed_tiny(tmp_path)
    (folder / "verified.json").unlink()
    with pytest.raises(MemoryUnavailable, match="download_model"):
        OnnxEmbedder(model, folder)


def test_the_onnx_embedder_without_its_runtime_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    model, folder = _installed_tiny(tmp_path)
    with pytest.raises(MemoryUnavailable, match="memory-vectors"):
        OnnxEmbedder(model, folder)


def test_the_index_loads_the_model_only_when_there_is_something_to_embed(store, tmp_path, monkeypatch):
    seen = _fake_runtime(monkeypatch)
    model, folder = _installed_tiny(tmp_path)
    vectors = MemoryVectors(store, folder, model=model)
    assert vectors.sync().embedded == 0 and "tokenizer" not in seen
    _add(store, "alpha")
    assert vectors.sync().embedded == 1 and seen["texts"][0].startswith("query: alpha")


# ---- syncing ------------------------------------------------------------------------------------


def test_sync_embeds_each_item_once_and_skips_short_text(store):
    long_id = _add(store, "alpha")
    short_id = store.add("tiny", provenance="observed", origin="o").id
    table = Table(VECTORS)
    vectors = MemoryVectors(store, model=TINY, embedder=table)
    report = vectors.sync()
    assert (report.embedded, report.skipped, report.waiting) == (1, 1, 0)
    assert vectors.count() == 1 and len(table.calls) == 1 and len(table.calls[0][0]) >= MIN_TEXT
    assert vectors.sync().embedded == 0 and len(table.calls) == 1, "nothing changed, nothing embedded"
    assert vectors.similar(short_id) == [] and long_id


def test_a_changed_title_re_embeds_and_the_title_is_part_of_the_text(store):
    item = _add(store, "alpha")
    table = Table(VECTORS)
    vectors = MemoryVectors(store, model=TINY, embedder=table)
    vectors.sync()
    store.label(item, title="beta", provenance="said")
    assert vectors.sync().embedded == 1 and table.calls[-1][0].startswith("beta\n")


def test_a_short_item_is_read_again_once_its_text_grows(store):
    item = store.add("alpha", provenance="observed", origin="o").id
    vectors = MemoryVectors(store, model=TINY, embedder=Table(VECTORS))
    assert vectors.sync().skipped == 1 and vectors.sync().skipped == 0
    store.label(item, summary="a summary that makes the text long enough to compare", provenance="said")
    assert vectors.sync().embedded == 1


def test_a_new_model_re_embeds_everything_and_old_vectors_are_not_compared(store):
    ids, _ = _indexed(store, ("alpha", "beta"))
    newer = EmbeddingModel(name="tiny", revision="r2", files={}, dimensions=3)
    vectors = MemoryVectors(store, model=newer, embedder=Table(VECTORS))
    assert vectors.count() == 0 and vectors.similar(ids["alpha"]) == []
    assert vectors.sync().embedded == 2 and vectors.count() == 2


def test_sync_is_newest_first_clamped_and_stoppable(store):
    ids = [_add(store, "alpha") for _ in range(40)]
    table = Table(VECTORS)
    vectors = MemoryVectors(store, model=TINY, embedder=table)
    report = vectors.sync(limit=20)
    assert report.embedded == 20 and report.waiting == 20
    assert table.calls[0][0] and len(table.calls[0]) == 16
    stopped = vectors.sync(limit=10_000, should_stop=lambda: True)
    assert stopped.stopped and stopped.embedded == 0
    assert vectors.sync(limit=0).embedded == 1, "a limit under one is one"
    conn = sqlite3.connect(vectors.path)
    first = {f"m{r[0]}" for r in conn.execute("SELECT item_id FROM vectors")}
    conn.close()
    assert ids[-1] in first and ids[0] not in first


def test_a_retired_item_loses_its_vector(store):
    ids, vectors = _indexed(store, ("alpha", "beta"))
    store.retire(ids["beta"])
    assert vectors.similar(ids["alpha"]) == [], "a retired item is never compared, even before a sync"
    assert vectors.sync().removed == 1 and vectors.count() == 1


def test_a_vector_of_the_wrong_size_is_refused_and_a_zero_vector_skipped(store):
    _add(store, "alpha")
    with pytest.raises(ValueError, match="3"):
        MemoryVectors(store, model=TINY, embedder=Table({"alpha": [1.0, 0.0]})).sync()
    report = MemoryVectors(store, model=TINY, embedder=Table({"alpha": [0.0, 0.0, 0.0]})).sync()
    assert report.skipped == 1 and report.embedded == 0


def test_an_embedder_that_returns_too_few_vectors_is_refused(store):
    _add(store, "alpha")

    class Short:
        def embed(self, texts):
            return []

    with pytest.raises(ValueError, match="0 vectors for 1"):
        MemoryVectors(store, model=TINY, embedder=Short()).sync()


def test_a_vector_file_from_a_newer_version_is_refused(store):
    MemoryVectors(store, model=TINY, embedder=Table(VECTORS))
    conn = sqlite3.connect(vectors_path(store.path))
    conn.execute("PRAGMA user_version = 99")
    conn.close()
    with pytest.raises(RuntimeError, match="newer"):
        MemoryVectors(store, model=TINY, embedder=Table(VECTORS))


# ---- erasure ------------------------------------------------------------------------------------

DISTINCT = _unit(0.123456, 0.654321, 0.111111)


def _vector_bytes(store) -> bytes:
    path = vectors_path(store.path)
    return b"".join(p.read_bytes() for p in (path, path.with_name(path.name + "-wal")) if p.exists())


def test_a_purge_erases_the_vector_even_with_no_index_open(store):
    import numpy as np

    keep = _add(store, "alpha")
    gone = _add(store, "distinct")
    MemoryVectors(store, model=TINY, embedder=Table({**VECTORS, "distinct": DISTINCT})).sync()
    blob = np.asarray(DISTINCT, dtype="<f4").tobytes()
    assert blob in _vector_bytes(store)
    assert store.purge([gone]) == [gone]
    assert blob not in _vector_bytes(store), "the purged item's vector is gone from disk"
    conn = sqlite3.connect(vectors_path(store.path))
    assert [f"m{r[0]}" for r in conn.execute("SELECT item_id FROM vectors")] == [keep]
    conn.close()


def test_a_vacuum_that_fails_after_the_erase_is_reported_as_such(store, caplog, monkeypatch):
    from rlm_harness import memory as memory_module

    gone = _add(store, "alpha")
    MemoryVectors(store, model=TINY, embedder=Table(VECTORS)).sync()
    real_connect = sqlite3.connect

    class NoVacuum:
        def __init__(self, conn):
            self._conn = conn

        def execute(self, sql, *args):
            if sql == "VACUUM":
                raise sqlite3.OperationalError("database is locked")
            return self._conn.execute(sql, *args)

        def __getattr__(self, name):
            return getattr(self._conn, name)

    def connect(path, *args, **kwargs):
        conn = real_connect(path, *args, **kwargs)
        return NoVacuum(conn) if str(path).endswith(".vectors.db") else conn

    monkeypatch.setattr(memory_module.sqlite3, "connect", connect)
    with caplog.at_level("WARNING"):
        memory_module._erase_vectors(store.path, [int(gone[1:])], secure=True)
    monkeypatch.undo()
    assert "vectors erased, but VACUUM failed" in caplog.text
    assert "could not be erased" not in caplog.text
    conn = sqlite3.connect(vectors_path(store.path))
    assert conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0] == 0
    conn.close()


def test_a_purge_with_no_vector_file_creates_none(store):
    item = _add(store, "alpha")
    store.purge([item])
    assert not vectors_path(store.path).exists()


def test_forget_drops_a_vector_and_keeps_the_item(store):
    ids, vectors = _indexed(store, ("alpha", "beta"))
    vectors.forget([ids["beta"]])
    assert vectors.count() == 1 and store.get(ids["beta"]) is not None
    assert vectors.sync().embedded == 0 and vectors.count() == 1, "forgotten stays forgotten"
    store.label(ids["beta"], title="beta again", provenance="said")
    assert vectors.sync().embedded == 1, "until its text changes"
    with pytest.raises(TypeError):
        vectors.forget(ids["beta"])


def test_a_sync_already_running_cannot_undo_a_forget(store):
    item = _add(store, "alpha")
    holder: list = []

    class ForgetsMidway:
        def embed(self, texts):
            holder[0].forget([item])
            return [VECTORS["alpha"] for _ in texts]

    vectors = MemoryVectors(store, model=TINY, embedder=ForgetsMidway())
    holder.append(vectors)
    report = vectors.sync()
    assert vectors.count() == 0 and report.embedded == 0 and report.waiting == 0


def test_an_item_purged_while_it_was_being_forgotten_leaves_no_row(store, monkeypatch):
    item = _add(store, "alpha")
    vectors = MemoryVectors(store, model=TINY, embedder=Table(VECTORS))
    vectors.sync()
    read = vectors._texts

    def read_then_purge():
        texts = read()
        store.purge([item])   # lands between forget's read and its write
        return texts

    monkeypatch.setattr(vectors, "_texts", read_then_purge)
    vectors.forget([item])
    conn = sqlite3.connect(vectors_path(store.path))
    assert conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0] == 0, "not even a text hash"
    conn.close()


def test_an_item_purged_while_its_batch_was_embedding_keeps_no_vector(store):
    item = _add(store, "alpha")

    class PurgesMidway:
        def embed(self, texts):
            store.purge([item])
            return [VECTORS["alpha"] for _ in texts]

    vectors = MemoryVectors(store, model=TINY, embedder=PurgesMidway())
    report = vectors.sync()
    assert report.embedded == 0 and report.waiting == 0 and vectors.count() == 0


def test_an_erase_a_reader_holds_back_is_logged(store, caplog):
    gone = _add(store, "alpha")
    MemoryVectors(store, model=TINY, embedder=Table(VECTORS)).sync()
    reader = sqlite3.connect(vectors_path(store.path), isolation_level=None)
    try:
        reader.execute("BEGIN")
        reader.execute("SELECT COUNT(*) FROM vectors").fetchone()
        with caplog.at_level("WARNING"):
            store.purge([gone])
    finally:
        reader.close()
    assert "vectors erased, but another connection" in caplog.text


# ---- similarity ---------------------------------------------------------------------------------


def test_similar_returns_mutual_neighbours_above_the_threshold(store):
    ids, vectors = _indexed(store)
    assert [i for i, _ in vectors.similar(ids["alpha"], k=1)] == [ids["beta"]]
    assert vectors.similar(ids["gamma"], k=1) == [], "alpha is gamma's nearest, but not the other way"
    assert vectors.similar(ids["delta"]) == [], "nothing is near it"
    score = dict(vectors.similar(ids["alpha"], k=2))[ids["beta"]]
    assert score == pytest.approx(0.95, abs=1e-3)


def test_among_limits_the_crowd_each_side_is_counted_in(store):
    ids, vectors = _indexed(store)
    found = vectors.similar(ids["gamma"], k=1, among=[ids["alpha"]])
    assert [i for i, _ in found] == [ids["alpha"]], "without beta in the way, alpha's nearest is gamma"
    with pytest.raises(TypeError):
        vectors.similar(ids["gamma"], among=ids["alpha"])


def test_similar_pairs_are_mutual_strongest_first_and_ordered_by_id(store):
    ids, vectors = _indexed(store)
    pairs = vectors.similar_pairs(list(ids.values()))
    assert all(isinstance(p, SimilarPair) and int(p.a[1:]) < int(p.b[1:]) for p in pairs)
    assert pairs[0] == SimilarPair(ids["alpha"], ids["beta"], pytest.approx(0.95, abs=1e-3))
    assert all(ids["delta"] not in (p.a, p.b) for p in pairs)
    assert [p.score for p in pairs] == sorted((p.score for p in pairs), reverse=True)
    assert vectors.similar_pairs([ids["alpha"]]) == []


# ---- suggestions --------------------------------------------------------------------------------


def test_an_unlabelled_item_is_offered_the_topic_of_the_filed_item_it_is_like(store):
    ids, vectors = _indexed(store, ("alpha", "gamma", "delta"))
    topic = store.create_topic("Letters")
    store.place(ids["alpha"], topic.id)
    found = suggest_topics(store, vectors=vectors)
    assert [(s.item_id, s.topic_id, s.similar_to) for s in found] == [(ids["gamma"], topic.id, ids["alpha"])]
    assert found[0].score == SUGGEST_THRESHOLD and found[0].similarity == pytest.approx(0.902, abs=1e-3)
    assert suggest_topics(store) == [], "without vectors, nothing to go on"


def test_a_label_match_outranks_a_similarity_match_of_the_same_score(store):
    ids, vectors = _indexed(store, ("alpha", "gamma"))
    topic = store.create_topic("Letters")
    store.place(ids["alpha"], topic.id)
    store.label(ids["alpha"], entities=["Thing"], tags=["kind"], provenance="said")
    labelled = _add(store, "delta")
    store.label(labelled, entities=["Thing"], tags=["kind"], provenance="said")
    found = suggest_topics(store, vectors=vectors)
    assert [s.item_id for s in found] == [labelled, ids["gamma"]]
    assert found[0].score == found[1].score == SUGGEST_THRESHOLD


def test_a_similarity_suggestion_respects_dismissals_and_the_landing_topic(store):
    ids, vectors = _indexed(store, ("alpha", "gamma"))
    landing = store.create_topic("Inbox")
    store.place(ids["alpha"], landing.id)
    assert suggest_topics(store, vectors=vectors, landing=landing.id) == []
    topic = store.create_topic("Letters")
    store.place(ids["alpha"], topic.id)
    store.dismiss(ids["gamma"], topic.id)
    assert suggest_topics(store, vectors=vectors, landing=landing.id) == []


def test_the_vector_file_sits_beside_the_store(store):
    _, vectors = _indexed(store, ("alpha",))
    assert vectors.path == vectors_path(store.path) and vectors.path.name == "m.db.vectors.db"
    conn = sqlite3.connect(vectors.path)
    row = conn.execute("SELECT model, length(embedding), text_hash FROM vectors").fetchone()
    conn.close()
    assert row[0] == "tiny@r1" and row[1] == 12 and len(row[2]) == 64


# ---- Chinese, Japanese and Korean ----------------------------------------------------------------

CJK_TINY = EmbeddingModel(name="tiny", revision="r1", files={}, dimensions=3,
                          similar_threshold=0.86, cjk_threshold=0.90)
BETWEEN = {"one": _unit(1, 0, 0), "two": _unit(0.88, 0.475, 0)}   # cosine 0.88: between the two bars


def test_a_cjk_character_weighs_three_and_mixed_text_is_judged_by_its_letters():
    assert CJK_WEIGHT == 3 and text_weight("abc") == 3 and text_weight("咖啡因") == 9
    assert text_weight("咖啡 ok") == 2 * 3 + 3
    assert is_cjk("咖啡因會阻斷腺苷受體") and is_cjk("用 K8s 部署服務時會自動重啟容器")
    assert not is_cjk("Kubernetes restarts failed containers") and not is_cjk("12345 !!")
    assert is_cjk("カフェインはアデノシンを妨げる") and is_cjk("카페인은 아데노신을 막는다")


def test_a_short_chinese_note_is_compared_from_fourteen_characters(store):
    in_ = store.add("一" * 14, provenance="observed", origin="o").id
    out = store.add("二" * 13, provenance="observed", origin="o").id
    vectors = MemoryVectors(store, model=CJK_TINY, embedder=Table({"一": VECTORS["alpha"], "二": VECTORS["beta"]}))
    report = vectors.sync()
    assert (report.embedded, report.skipped) == (1, 1) and vectors.count() == 1
    assert in_ and out


def _pair(store, first, second):
    a = store.add(first, provenance="observed", origin="o").id
    b = store.add(second, provenance="observed", origin="o").id
    vectors = MemoryVectors(store, model=CJK_TINY, embedder=Table(BETWEEN))
    vectors.sync()
    return a, b, vectors


def test_two_chinese_items_are_held_to_the_higher_threshold(store):
    a, b, vectors = _pair(store, "one 咖啡因會阻斷大腦裡的腺苷受體所以讓人清醒", "two 睡眠不足會影響記憶與專注力需要七小時")
    assert vectors.similar(a) == [] and vectors.similar_pairs([a, b]) == []


def test_an_english_pair_and_a_mixed_pair_keep_the_lower_threshold(store, tmp_path):
    a, b, vectors = _pair(store, "one caffeine blocks adenosine receptors in the brain",
                          "two too little sleep hurts memory and focus every day")
    assert [i for i, _ in vectors.similar(a)] == [b] and len(vectors.similar_pairs([a, b])) == 1
    mixed = MemoryStore(tmp_path / "mixed.db")
    c, d, mixed_vectors = _pair(mixed, "one caffeine blocks adenosine receptors in the brain",
                                "two 睡眠不足會影響記憶與專注力需要七小時")
    assert [i for i, _ in mixed_vectors.similar(c)] == [d]


def test_a_model_without_a_cjk_threshold_uses_one_threshold_for_every_pair(store):
    a = store.add("one 咖啡因會阻斷大腦裡的腺苷受體所以讓人清醒", provenance="observed", origin="o").id
    b = store.add("two 睡眠不足會影響記憶與專注力需要七小時", provenance="observed", origin="o").id
    vectors = MemoryVectors(store, model=TINY, embedder=Table(BETWEEN))
    vectors.sync()
    assert [i for i, _ in vectors.similar(a)] == [b]


def test_the_default_model_holds_cjk_pairs_to_its_measured_threshold():
    assert DEFAULT_MODEL.cjk_threshold == 0.90 and DEFAULT_MODEL.similar_threshold == 0.86


def test_similar_pairs_judges_each_pair_by_its_own_items_not_its_place_in_the_store(store):
    # CJK items first, so a subset's positions and the store's rows disagree: a pair judged by the
    # wrong rows would borrow the fillers' flags.
    table = {"F1": _unit(0, 0, 1), "F2": _unit(0, 0.1, 1), **BETWEEN, "uno": BETWEEN["one"], "dos": BETWEEN["two"]}
    for marker in ("F1", "F2"):
        store.add(f"{marker} 這是一段用來佔位的中文文字，內容和其他項目都無關", provenance="observed", origin="o")
    en_a = store.add("one caffeine blocks adenosine receptors in the brain", provenance="observed", origin="o").id
    en_b = store.add("two too little sleep hurts memory and focus every day", provenance="observed", origin="o").id
    zh_a = store.add("uno 咖啡因會阻斷大腦裡的腺苷受體所以讓人清醒", provenance="observed", origin="o").id
    zh_b = store.add("dos 睡眠不足會影響記憶與專注力需要七小時以上", provenance="observed", origin="o").id
    vectors = MemoryVectors(store, model=CJK_TINY, embedder=Table(table))
    vectors.sync()
    assert len(vectors.similar_pairs([en_a, en_b])) == 1, "English: 0.88 clears 0.86"
    assert vectors.similar_pairs([zh_a, zh_b]) == [], "Chinese: 0.88 misses 0.90"
    assert len(vectors.similar_pairs([en_a, zh_b])) == 1, "mixed: 0.86"


def test_an_item_relabelled_into_chinese_is_held_to_the_cjk_threshold(store):
    a = store.add("one caffeine blocks adenosine receptors in the brain", provenance="observed", origin="o").id
    b = store.add("two 睡眠不足會影響記憶與專注力需要七小時以上", provenance="observed", origin="o").id
    vectors = MemoryVectors(store, model=CJK_TINY, embedder=Table(BETWEEN))
    vectors.sync()
    assert [i for i, _ in vectors.similar(a)] == [b], "an English and a Chinese text: 0.86"
    # Title and summary in Chinese now outweigh the English text's 45 letters.
    store.label(a, title="咖啡因會阻斷大腦裡的腺苷受體所以讓人保持清醒不想睡覺",
                summary="下午喝太多咖啡的話晚上就容易睡不著覺，所以最好在中午以前喝完", provenance="said")
    assert vectors.sync().embedded == 1
    conn = sqlite3.connect(vectors.path)
    assert conn.execute("SELECT cjk FROM vectors WHERE item_id = ?", (int(a[1:]),)).fetchone()[0] == 1
    conn.close()
    assert vectors.similar(a) == [], "now both are Chinese: 0.90"
