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
import sys
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, TextIO


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
