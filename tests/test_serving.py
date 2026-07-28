"""serve_harness — the server side of the make_harness_tool delegation contract. All offline: a fake
`run` + StringIO streams, no dspy, no Deno, no live harness. Pins the wire, the exit-code split
(infra=1 / ran=0), CWD isolation, env loading, stderr-only diagnostics, and byte-compat with the
client's HarnessInvocation-shaped read."""
import io
import json
import os
import types

import pytest

from rlm_kit import HarnessPointer, serve_harness
from rlm_kit.serving import _default_to_pointer, bundle_artifact, parse_artifact_bundle


@pytest.fixture(autouse=True)
def _restore_cwd():
    # serve_harness chdir's into its isolated workdir (fine in the real -m subprocess, which then exits);
    # in-process tests must restore the CWD so they don't leak into each other.
    cwd = os.getcwd()
    try:
        yield
    finally:
        os.chdir(cwd)


def _streams(spec="LONG SPEC"):
    return io.StringIO(spec), io.StringIO(), io.StringIO()  # stdin, stdout, stderr


# ---- HarnessPointer wire ---------------------------------------------------

def test_pointer_flattens_meta_to_top_level_and_omits_empties():
    p = HarnessPointer(artifact="YAML", run_id="c1", trace_path="t.jsonl",
                       meta={"valid": True, "complete": False})
    obj = json.loads(p.to_json_line())
    assert obj == {"artifact": "YAML", "run_id": "c1", "trace_path": "t.jsonl",
                   "valid": True, "complete": False}          # meta flattened; no reasoning key
    bare = json.loads(HarnessPointer(artifact="X").to_json_line())
    assert bare == {"artifact": "X"}                          # empty optionals omitted


# ---- serve_harness: happy path, isolation, wire ----------------------------

def test_serve_runs_and_emits_the_pointer(tmp_path):
    seen = {}

    def run(source, *, run_id, outdir):
        seen["source"], seen["run_id"], seen["outdir"], seen["cwd"] = source, run_id, outdir, os.getcwd()
        return types.SimpleNamespace(yaml="ARTIFACT")

    stdin, stdout, stderr = _streams("a very long context …")
    rc = serve_harness(
        run, lambda r: HarnessPointer(artifact=r.yaml, run_id="c9", trace_path="tr.jsonl"),
        stdin=stdin, stdout=stdout, stderr=stderr,
        run_kwargs={"outdir": "."}, workdir_base=str(tmp_path / "base"),
    )
    assert rc == 0
    assert seen["source"] == "a very long context …"          # whole stdin reached the harness
    assert seen["outdir"] == "."
    # CWD was isolated into <base>/<run_id>/ (the harness writes traces/ there, not the caller's tree)
    assert seen["cwd"] == os.path.abspath(os.path.join(str(tmp_path / "base"), seen["run_id"]))
    obj = json.loads(stdout.getvalue().strip())
    assert obj == {"artifact": "ARTIFACT", "run_id": "c9", "trace_path": "tr.jsonl"}


def test_serve_defaults_a_uuid_run_id(tmp_path):
    got = {}

    def run(source, *, run_id):
        got["run_id"] = run_id
        return HarnessPointer(artifact="A")

    serve_harness(run, stdin=io.StringIO("x"), stdout=io.StringIO(), stderr=io.StringIO(),
                  workdir_base=str(tmp_path))
    assert got["run_id"].startswith("harness-") and len(got["run_id"]) > 8


# ---- exit-code split: infra failure vs a ran-but-empty artifact ------------

def test_infra_failure_exits_1_and_keeps_stdout_clean(tmp_path):
    def run(source, *, run_id):
        raise RuntimeError("SECRET_HARNESS_NAME planner crashed")

    stdin, stdout, stderr = _streams()
    rc = serve_harness(run, stdin=stdin, stdout=stdout, stderr=stderr, workdir_base=str(tmp_path))
    assert rc == 1                                            # infra → caller retries
    assert stdout.getvalue() == ""                           # NO pointer on a failed run
    assert "SECRET_HARNESS_NAME" in stderr.getvalue()        # the traceback is on STDERR only …
    # … and stdout never carries the harness's identity (the client drops stderr; this keeps it off stdout)
    assert "SECRET_HARNESS_NAME" not in stdout.getvalue()


def test_ran_but_empty_artifact_exits_0(tmp_path):
    # the harness RAN but produced nothing — that's a CONTENT outcome (exit 0); the caller judges it.
    def run(source, *, run_id):
        return HarnessPointer(artifact="")

    stdin, stdout, stderr = _streams()
    rc = serve_harness(run, stdin=stdin, stdout=stdout, stderr=stderr, workdir_base=str(tmp_path))
    assert rc == 0
    assert json.loads(stdout.getvalue().strip()) == {"artifact": ""}


# ---- env loading -----------------------------------------------------------

def test_env_files_load_before_the_run(tmp_path, monkeypatch):
    envf = tmp_path / "harness.env"
    envf.write_text("# roles\nHARNESS_ROLE=planner-x\nEMPTY_IGNORED\n")
    monkeypatch.delenv("HARNESS_ROLE", raising=False)
    seen = {}

    def run(source, *, run_id):
        seen["role"] = os.environ.get("HARNESS_ROLE")
        return HarnessPointer(artifact="A")

    serve_harness(run, stdin=io.StringIO("x"), stdout=io.StringIO(), stderr=io.StringIO(),
                  env_files=[str(envf)], workdir_base=str(tmp_path))
    assert seen["role"] == "planner-x"                        # loaded from the harness's own env file


def test_env_loader_never_invents_an_absent_key(tmp_path, monkeypatch):
    # the file sets exactly its keys — an absent one stays as the parent left it (a subscription parent's
    # unset ANTHROPIC_API_KEY must NOT be invented by loading the harness env).
    envf = tmp_path / "h.env"
    envf.write_text("HARNESS_ROLE=x\n")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    seen = {}

    def run(source, *, run_id):
        seen["has_anthropic"] = "ANTHROPIC_API_KEY" in os.environ
        return HarnessPointer(artifact="A")

    serve_harness(run, stdin=io.StringIO("x"), stdout=io.StringIO(), stderr=io.StringIO(),
                  env_files=[str(envf)], workdir_base=str(tmp_path))
    assert seen["has_anthropic"] is False                     # not invented


# ---- the duck-typed default extractor --------------------------------------

def test_default_to_pointer_reads_flat_attrs():
    p = _default_to_pointer(types.SimpleNamespace(artifact="A", run_id="r", trace_path="t"))
    assert p.artifact == "A" and p.run_id == "r" and p.trace_path == "t"
    assert _default_to_pointer(HarnessPointer(artifact="B")).artifact == "B"   # passthrough
    with pytest.raises(TypeError, match="to_pointer"):
        _default_to_pointer(types.SimpleNamespace(nope=1))   # no .artifact → actionable error


# ---- byte-compat with the CLIENT's read (make_harness_tool consumer) -------

def test_wire_is_readable_by_a_client_read_output(tmp_path):
    # A consumer's read: artifact + run_id/trace_path + top-level meta flags. Prove the exact line
    # serve_harness emits round-trips through that shape (the make_harness_tool client reads meta flags
    # at TOP level, so the flatten is load-bearing).
    def run(source, *, run_id):
        return HarnessPointer(artifact="TMPL", run_id="cx", trace_path="cx.jsonl",
                              meta={"valid": True, "complete": True})

    stdout = io.StringIO()
    serve_harness(run, stdin=io.StringIO("s"), stdout=stdout, stderr=io.StringIO(),
                  workdir_base=str(tmp_path))
    pointer = json.loads(stdout.getvalue().strip())
    # mirror of a consumer's read_output (artifact + link + top-level meta flags)
    assert pointer.get("artifact") == "TMPL"
    assert pointer.get("run_id") == "cx" and pointer.get("trace_path") == "cx.jsonl"
    meta = {k: pointer[k] for k in ("elapsed_s", "child_steps", "valid", "complete") if k in pointer}
    assert meta == {"valid": True, "complete": True}


def test_meta_never_clobbers_the_typed_fields():
    # a stray reserved key inside meta must NOT override the authoritative typed field
    p = HarnessPointer(artifact="REAL", run_id="r1", meta={"artifact": "CLOBBER", "valid": True})
    obj = json.loads(p.to_json_line())
    assert obj["artifact"] == "REAL" and obj["run_id"] == "r1" and obj["valid"] is True


def test_to_pointer_failure_is_a_clean_exit_1_not_a_stdout_leak(tmp_path):
    # a mapping bug AFTER a successful run → could-not-produce-a-pointer (exit 1), stdout clean, the
    # traceback (which could name the harness) on stderr only.
    def run(source, *, run_id):
        return object()

    def bad_to_pointer(_r):
        raise KeyError("SECRET_HARNESS_INTERNAL")

    stdin, stdout, stderr = _streams()
    rc = serve_harness(run, bad_to_pointer, stdin=stdin, stdout=stdout, stderr=stderr,
                       workdir_base=str(tmp_path))
    assert rc == 1 and stdout.getvalue() == ""
    assert "SECRET_HARNESS_INTERNAL" not in stdout.getvalue()


def test_harness_stdout_noise_cannot_corrupt_the_pointer(tmp_path):
    # a harness that prints a banner to stdout during the run must NOT precede/corrupt the pointer —
    # serve_harness redirects the harness's stdout to stderr, so stdout carries ONLY the pointer line.
    def run(source, *, run_id):
        print("HARNESS BANNER: starting", end="")   # partial line, no newline — the worst case
        print("progress 50%")
        return HarnessPointer(artifact="A", run_id="r")

    stdin, stdout, stderr = _streams()
    serve_harness(run, stdin=stdin, stdout=stdout, stderr=stderr, workdir_base=str(tmp_path))
    assert json.loads(stdout.getvalue().strip()) == {"artifact": "A", "run_id": "r"}  # clean, parseable
    assert "BANNER" not in stdout.getvalue() and "BANNER" in stderr.getvalue()        # noise → stderr


# ---- the `python -m rlm_kit.harness_serve <module:run>` entry --------------

def test_harness_serve_entry_resolves_and_serves(tmp_path, monkeypatch, capsys):
    import sys as _sys

    from rlm_kit import harness_serve

    fake = types.ModuleType("fake_harness_mod")
    fake.run = lambda source, *, run_id, **_: types.SimpleNamespace(artifact="FLAT", run_id=run_id)
    monkeypatch.setitem(_sys.modules, "fake_harness_mod", fake)
    monkeypatch.setattr(_sys, "stdin", io.StringIO("spec"))   # feed stdin; stdout via capsys (call-time resolve)

    rc = harness_serve.main(["fake_harness_mod:run", str(tmp_path / "base")])
    assert rc == 0
    assert json.loads(capsys.readouterr().out.strip())["artifact"] == "FLAT"   # duck-typed default extractor


def test_harness_serve_entry_rejects_a_bad_target():
    from rlm_kit import harness_serve
    with pytest.raises(SystemExit, match="expected 'package.module:run'"):
        harness_serve.main(["not_a_spec"])


def test_harness_serve_entry_uses_a_module_to_pointer_and_run_kwargs(tmp_path, monkeypatch, capsys):
    # a harness with a NESTED result exposes its own to_pointer (+ SERVE_RUN_KWARGS) on the module; the
    # -m entry must pick them up rather than fall back to the duck-typed default.
    import sys as _sys

    from rlm_kit import HarnessPointer, harness_serve

    seen = {}

    def _nested_run(source, *, run_id, outdir):
        seen["outdir"] = outdir
        return types.SimpleNamespace(nested=types.SimpleNamespace(yaml="NESTED"))

    fake = types.ModuleType("nested_harness_mod")
    fake.run = _nested_run
    fake.SERVE_RUN_KWARGS = {"outdir": "."}
    fake.to_pointer = lambda r: HarnessPointer(artifact=r.nested.yaml, run_id="n1")
    monkeypatch.setitem(_sys.modules, "nested_harness_mod", fake)
    monkeypatch.setattr(_sys, "stdin", io.StringIO("spec"))

    rc = harness_serve.main(["nested_harness_mod:run", str(tmp_path / "base")])
    assert rc == 0 and seen["outdir"] == "."                  # SERVE_RUN_KWARGS applied
    obj = json.loads(capsys.readouterr().out.strip())
    assert obj == {"artifact": "NESTED", "run_id": "n1"}      # module to_pointer used, not the default


# -- multi-file artifacts: bundle_artifact / parse_artifact_bundle -----------------------------------


def test_a_bundle_round_trips_and_reads_cleanly():
    files = {"README.md": "# Title\n\nprose", "poc.md": "curl http://x/", "changes.diff": "-bad\n+good"}
    bundled = bundle_artifact(files)

    assert bundled.startswith("===== README.md =====\n")
    assert "===== changes.diff =====" in bundled
    assert parse_artifact_bundle(bundled) == files


def test_insertion_order_is_preserved():
    files = {"z.md": "z", "a.md": "a", "m.md": "m"}
    assert list(parse_artifact_bundle(bundle_artifact(files))) == ["z.md", "a.md", "m.md"]


def test_an_empty_mapping_bundles_to_an_empty_artifact():
    # A harness that produced nothing returns an empty artifact and exit 0 — emptiness is the
    # CALLER's judgement to make, not a serving failure.
    assert bundle_artifact({}) == ""
    assert parse_artifact_bundle("") == {}


def test_a_markdown_underline_is_content_not_a_section_break():
    # `=====` alone is a setext heading underline, extremely common in real READMEs. It has no name
    # between the markers, so it is not a header and must not split anything.
    files = {"README.md": "Title\n=====\n\nbody", "x.md": "x"}
    assert parse_artifact_bundle(bundle_artifact(files)) == files


def test_a_file_quoting_a_header_line_escalates_the_marker():
    """The collision case that silently corrupts a bundle if unhandled: a file whose CONTENT holds a
    well-formed header line — a report echoing a child harness's reply, say. Without escalation the
    section would truncate at its own quotation and a phantom file would appear."""
    files = {"report.md": "child said:\n===== poc.md =====\ncurl http://evil/\ndone", "poc.md": "real"}
    bundled = bundle_artifact(files)

    assert bundled.startswith("====== report.md ======\n")   # escalated past the embedded marker
    assert parse_artifact_bundle(bundled) == files           # both files byte-intact
    assert sorted(parse_artifact_bundle(bundled)) == ["poc.md", "report.md"]   # no phantom section


def test_escalation_repeats_until_unambiguous():
    files = {"a.md": "x\n===== q =====\ny\n====== r ======\nz", "b.md": "plain"}
    bundled = bundle_artifact(files)

    assert bundled.startswith("======= a.md =======\n")
    assert parse_artifact_bundle(bundled) == files


def test_text_that_is_not_a_bundle_parses_to_empty():
    # A single-file artifact is the common case and is NOT an error — it simply is not a bundle, and
    # a client that wants the whole thing as context should use the artifact string as-is.
    assert parse_artifact_bundle("just a nuclei template\nid: CVE-2021-1\n") == {}


def test_a_bundle_survives_the_wire():
    files = {"README.md": "# CVE-2021-41773", "changes.diff": "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b"}
    line = HarnessPointer(artifact=bundle_artifact(files), run_id="c1").to_json_line()

    assert parse_artifact_bundle(json.loads(line)["artifact"]) == files
