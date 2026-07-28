"""Serve an rlm-kit harness over the delegation contract — the SERVER-side mirror of
``make_harness_tool`` (``tools/harness.py``).

``make_harness_tool`` is the CLIENT: a parent RLM wraps a downstream harness as a tool and reaches it
through an injected ``call_endpoint`` (a subprocess command, an HTTP URL, …). This module is the missing
SERVER: it turns ANY RLMTask-based harness into a process that SPEAKS that contract, so the operator
points the client's endpoint straight at the harness — no bespoke per-operator glue.

The contract (one JSON line on stdout): ``serve_harness`` reads the caller's long text from STDIN — the
harness binds it to its own long-text RLM input, so the harness Root LM runs its full REPL loop over the
whole context — runs the harness, and prints a :class:`HarnessPointer` as one JSON object line on STDOUT
(the child's artifact + a link to its OWN rollout: run_id / trace_path). The pointer is the ONLY thing on
stdout: the harness's own (Python-level) stdout is redirected to STDERR for the run, and every serve
diagnostic + traceback goes to STDERR with a generic reason — so nothing about the harness leaks into the
parent's trace. Exit code is the infra/content split the client relies on: ``0`` = the harness RAN (the
artifact may be empty/invalid; the caller judges it) · ``1`` = it could not produce a pointer (a run or
mapping failure → the caller retries).

BASE/WRAP split, same as the rest of the kit: rlm-kit owns ALL the generic plumbing (read stdin, run_id,
CWD isolation, the wire schema, exit codes, keeping secrets off stdout). The consuming HARNESS supplies
the one thing the kit cannot know — how to map ITS concrete result object into a :class:`HarnessPointer`
(``to_pointer``) — in a ~5-line ``serve`` module in its OWN repo. The kit names no harness. dspy-free
(stdlib only), so ``import rlm_kit`` stays light and this sits in the dspy-free module set.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import sys
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence, TextIO


@dataclass
class HarnessPointer:
    """The one-JSON-line delegation wire a served harness prints on stdout — the server-side mirror of
    ``tools/harness.HarnessInvocation``. ``make_harness_tool``'s ``read_output`` parses exactly these
    fields back. ``meta`` is flattened to the TOP LEVEL of the JSON object (not nested), so a caller can
    read domain flags (e.g. ``valid``/``complete``) as plain top-level keys."""

    artifact: str                         # the harness's final artifact TEXT (what the caller validates)
    run_id: Optional[str] = None          # the child rollout's own run_id (the parent→child link)
    trace_path: Optional[str] = None      # path to the child's OWN trace (never inlined)
    reasoning: Optional[str] = None        # the child's thinking, if the harness surfaces it
    meta: Optional[dict] = None           # generic extras, flattened top-level: {"valid":…, "complete":…}

    def to_json_line(self) -> str:
        # meta is flattened to TOP level (the caller reads its domain flags as plain keys) — but the
        # authoritative typed fields WIN, so a stray meta key can never clobber artifact/run_id/…
        obj: dict = dict(self.meta) if self.meta else {}
        obj["artifact"] = self.artifact
        if self.run_id:
            obj["run_id"] = self.run_id
        if self.trace_path:
            obj["trace_path"] = self.trace_path
        if self.reasoning:
            obj["reasoning"] = self.reasoning
        return json.dumps(obj)


# The one harness-specific hook: map the harness's concrete result object into a HarnessPointer.
ToPointer = Callable[[Any], HarnessPointer]


# -- multi-file artifacts: one shared convention, so the two sides cannot drift -------------------
#
# ``HarnessPointer.artifact`` is ONE string, which fits a harness whose deliverable is one file (a
# template, a patch, a document). Plenty of harnesses produce a FOLDER instead — a write-up plus a
# PoC plus a diff, or a Dockerfile plus a compose file plus notes — and every such harness/client
# pair otherwise invents its own packing format. That is a silent-failure generator: the two sides
# agree until they don't, and a mismatch degrades into "the child returned junk" rather than
# surfacing as the wiring bug it is. The kit already owns the wire schema; it should own this too.
#
# The format is deliberately plain text, not JSON: the artifact's primary consumer is a Root LM
# reading it in a REPL, and a human debugging the wire is the second. Both read
# ``===== poc.md =====`` far better than an escaped JSON blob — and a text-consuming client (one
# that wants the whole deliverable as context) needs no parser at all.

#: The default section marker. Five ``=`` reads clearly and is rare in prose or code.
_BUNDLE_MARK = "====="
#: Matches ANY well-formed section header, used only to discover a bundle's own marker on parse.
_BUNDLE_HEADER_RE = re.compile(r"^(={5,}) (.+?) \1$")


def _header_re(mark: str) -> re.Pattern:
    return re.compile(rf"{re.escape(mark)} (.+?) {re.escape(mark)}")


def _lines(text: str) -> list[str]:
    """The ONE definition of "a line" both halves of this format use.

    Load-bearing, and the reason it is a named helper rather than an inline `splitlines()`. An
    earlier version scanned for embedded headers with a `re.MULTILINE` regex while the parser split
    with `str.splitlines()` — and those disagree: MULTILINE breaks only on `\\n` and `\\r\\n`, while
    `splitlines()` breaks on ELEVEN separators (`\\r`, `\\x0b`, `\\x0c`, `\\x1c`, `\\x1d`, `\\x1e`,
    `\\x85`, `\\u2028`, `\\u2029` besides). So a header embedded in CRLF text — a PoC quoting an HTTP
    exchange, a Windows-authored file — escaped escalation and was then honoured as a real section
    break on parse: the quoting file truncated at its own quotation and its tail was absorbed into a
    phantom section. The key COUNT, the names and the order all still looked right, so nothing
    downstream could notice. Both sides now split with this function, so they cannot drift again.
    """
    return (text or "").splitlines()


def _has_header_line(mark: str, text: str) -> bool:
    return any(_header_re(mark).fullmatch(line) for line in _lines(text))


def bundle_artifact(files: Mapping[str, str]) -> str:
    """Pack ``{filename: content}`` into ONE ``HarnessPointer.artifact`` string.

    Sections are introduced by ``===== <name> =====`` on its own line. If any file's content already
    contains a line of that exact shape, the marker is ESCALATED (``======``, ``=======``, …) until it
    is unambiguous — the same "choose a boundary that does not occur in the payload" discipline MIME
    uses, so a bundled Markdown file full of ``=====`` rules can never split its own section.

    Insertion order is preserved, and an empty mapping bundles to ``""`` (a harness that produced
    nothing returns an empty artifact and exit 0 — the CALLER judges emptiness, per the contract).

    Round-trips through :func:`parse_artifact_bundle` modulo exactly two documented normalisations,
    both applied HERE so the bundled text and the parsed text always agree:

    * line endings are normalised to ``\\n``. `splitlines()` is the shared notion of a line, and it
      breaks on eleven separators (CRLF, CR, ``\\x0b``, ``\\x0c``, ``\\u2028``, …), so preserving the
      originals would make the two halves disagree about where a header can hide. Normalising at pack
      time is what keeps the format honest; it is stated rather than left to be discovered.
    * leading and trailing blank lines within a file are not significant.

    RAISES ``ValueError`` on a filename that cannot round-trip — one that is empty, or that contains
    a line separator. Such a name would break its own header line and the file would vanish (or
    reappear under a name nobody chose) with no error at all. Names in this API routinely come from
    an LM, so this fails loudly at pack time instead.
    """
    if not files:
        return ""
    for name in files:
        if not name or _lines(name) != [name]:
            raise ValueError(
                f"bundle_artifact: unusable filename {name!r} — a name must be non-empty and hold no "
                "line separator, or its section header cannot be parsed back."
            )
    normalised = {name: "\n".join(_lines(content)).strip("\n") for name, content in files.items()}
    mark = _BUNDLE_MARK
    while any(_has_header_line(mark, content) for content in normalised.values()):
        mark += "="
    return "\n".join(f"{mark} {name} {mark}\n{content}\n" for name, content in normalised.items())


def parse_artifact_bundle(text: str) -> dict[str, str]:
    """Unpack a :func:`bundle_artifact` string back into ``{filename: content}``.

    Only for a client that needs the files SEPARATELY (to write them to disk, or to read one of
    them). A client that just wants the whole deliverable as context should use the artifact string
    as-is — that is the common case and needs nothing from this module.

    The bundle's own marker is discovered from its FIRST header line and then required exactly, so a
    line inside a file that happens to look like a *shorter* header is content, not a section break.
    Text with no header at all yields ``{}`` — an unbundled single-file artifact is not an error
    here, it simply is not a bundle.
    """
    if not text or not text.strip():
        return {}
    lines = _lines(text)
    mark = next((m.group(1) for m in map(_BUNDLE_HEADER_RE.fullmatch, lines) if m), None)
    if mark is None:
        return {}
    header = _header_re(mark)
    files: dict[str, str] = {}
    name: Optional[str] = None
    buf: list[str] = []
    for line in lines:
        found = header.fullmatch(line)
        if found:
            if name is not None:
                files[name] = "\n".join(buf).strip("\n")
            name, buf = found.group(1), []
            continue
        if name is not None:
            buf.append(line)
    if name is not None:
        files[name] = "\n".join(buf).strip("\n")
    return files


def _load_env_files(paths: Sequence[str], stderr: TextIO) -> None:
    """Load ``KEY=VALUE`` lines from each dotenv path into ``os.environ`` (the harness's own roles —
    the kit hardcodes no variable names; the harness names its files). Sets EXACTLY the keys the file
    lists; never invents one (so a subscription parent's unset ANTHROPIC_API_KEY stays unset unless the
    file sets it). A missing file is a logged no-op, not a failure."""
    for path in paths:
        if not os.path.exists(path):
            print(f"[serve_harness] no env file at {path}", file=stderr)
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ[key.strip()] = value.strip()


def _default_to_pointer(result: Any) -> HarnessPointer:
    """Duck-typed fallback for a harness whose ``run()`` already returns a flat, pointer-shaped object
    (``.artifact`` / ``.run_id`` / ``.trace_path``). A harness with a NESTED result (e.g. a template on
    ``.result.template.yaml``) supplies its own ``to_pointer`` instead — this is only the zero-config
    path for the ``python -m rlm_kit.harness_serve`` entry."""
    if isinstance(result, HarnessPointer):
        return result
    artifact = getattr(result, "artifact", None)
    if not isinstance(artifact, str):
        raise TypeError(
            "the harness result has no string `.artifact`; pass an explicit `to_pointer` that maps this "
            "harness's result into a HarnessPointer (see rlm_kit.serve_harness)."
        )
    return HarnessPointer(
        artifact=artifact,
        run_id=getattr(result, "run_id", None),
        trace_path=getattr(result, "trace_path", None),
        reasoning=getattr(result, "reasoning", None),
    )


def serve_harness(
    run: Callable[..., Any],
    to_pointer: ToPointer = _default_to_pointer,
    *,
    stdin: Optional[TextIO] = None,
    stdout: Optional[TextIO] = None,
    stderr: Optional[TextIO] = None,
    run_id: Optional[str] = None,
    run_kwargs: Optional[dict] = None,
    workdir_base: str = "harness-runs",
    isolate_cwd: bool = True,
    env_files: Sequence[str] = (),
) -> int:
    """Run a downstream harness once over the delegation contract; return the process exit code.

    ``run`` is the harness's programmatic entry, called ``run(source, run_id=…, **run_kwargs)`` where
    ``source`` is the long text read from ``stdin`` (bound by the harness to its own RLM input — its
    Root LM's REPL environment). ``to_pointer`` maps the harness's return into a :class:`HarnessPointer`
    (the ONE harness-specific hook; defaults to a duck-typed extractor for a flat result). ``env_files``
    dotenv paths are loaded into ``os.environ`` BEFORE the run (the harness's own roles; the kit names
    none). With ``isolate_cwd`` the run executes in a fresh ``<workdir_base>/<run_id>/`` directory, so a
    harness that writes CWD-relative artifacts (``traces/`` …) never collides with the caller's tree.

    Returns ``0`` when the harness RAN (the pointer's artifact may be empty/invalid — the CALLER judges
    it via its own validator) and ``1`` when the harness FAILED TO RUN (a raise from ``run`` — surfaced
    to the caller as an endpoint error it RETRIES, not a content decline). The pointer line is the ONLY
    thing on ``stdout``: the harness's OWN stdout is redirected to ``stderr`` for the duration of the run
    (so a banner/log the harness prints can't corrupt or precede the pointer), and every serve diagnostic
    + traceback goes to ``stderr`` with a generic reason — so the harness's identity never reaches the
    caller's trace. The streams default to the LIVE ``sys.stdin/stdout/stderr`` resolved at CALL time (so
    a runtime redirection is respected, and tests can inject their own)."""
    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr
    source = stdin.read()
    _load_env_files(env_files, stderr)
    rid = run_id or f"harness-{uuid.uuid4().hex[:12]}"

    try:
        if isolate_cwd:  # a fresh per-run dir — a harness that writes CWD-relative artifacts can't collide
            workdir = os.path.abspath(os.path.join(workdir_base, rid))
            os.makedirs(workdir, exist_ok=True)
            os.chdir(workdir)
        # Redirect the harness's OWN stdout to stderr so ONLY our pointer lands on stdout. Building the
        # pointer (to_pointer) is inside the guard too: a deterministic mapping bug is surfaced as a
        # generic failure, never a half-written stdout line.
        with contextlib.redirect_stdout(stderr):
            result = run(source, run_id=rid, **(run_kwargs or {}))
        line = to_pointer(result).to_json_line()
    except Exception as exc:  # noqa: BLE001 — could not produce a pointer → exit 1 so the caller retries.
        # Generic reason + traceback to STDERR only; never to stdout, never the harness's identity.
        print(f"[serve_harness] harness run failed: {type(exc).__name__}: {exc}", file=stderr)
        import traceback
        traceback.print_exc(file=stderr)
        return 1

    stdout.write("\n" + line + "\n")  # lead with \n so a stray partial harness write can't prefix us
    stdout.flush()
    return 0
