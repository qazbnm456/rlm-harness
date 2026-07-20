"""Worked example — serve a harness over the make_harness_tool delegation contract (the SERVER side).

Run it directly:

    echo "some long context" | python examples/harness_serve.py [workdir_base]

It prints ONE `HarnessPointer` JSON line on stdout. In production the harness's OWN venv runs this as a
module (`python -m <your_pkg>.serve`), and an upstream rlm-kit consumer points its harness endpoint
config at that command — no bespoke glue project.

To turn a REAL harness into a server, copy this shape into `<your_pkg>/serve.py` and change TWO things:
  1. import your harness's `run` (its RLMTask programmatic entry) instead of `_demo_run`;
  2. write `to_pointer` — the ONE harness-specific hook — mapping YOUR result object into a
     `HarnessPointer`. rlm-kit's `serve_harness` owns everything else (read stdin → the child's RLM
     environment, run_id, CWD isolation, the JSON-pointer wire, exit codes 0=ran/1=infra, and keeping
     the harness's logs + tracebacks OFF stdout).

If your `run()` already returns a FLAT object (`.artifact` / `.run_id` / `.trace_path`), you need NO
file at all — just `python -m rlm_kit.harness_serve <your_pkg.module>:run` uses the duck-typed default.
`to_pointer` is only needed for a NESTED result (navigate to the artifact yourself).
"""

from __future__ import annotations

import sys
import types

from rlm_kit import HarnessPointer, serve_harness


def _demo_run(source: str, *, run_id: str, **_):
    """Stand-in for a real RLMTask harness. A real one runs its Root LM over ``source`` (its RLM
    environment) with its own tools and returns its own result object; here we just echo a fake one."""
    return types.SimpleNamespace(
        artifact=f"# produced from {len(source)} chars of context\nid: demo-artifact\n",
        run_id=run_id,
    )


def to_pointer(result) -> HarnessPointer:
    """The ONE harness-specific hook: map YOUR result object → the wire pointer. For a NESTED result
    (e.g. ``result.report.template.yaml``) navigate to it here; ``run_id``/``trace_path``/``meta`` are
    optional (they link the parent trace to this run's own rollout)."""
    return HarnessPointer(artifact=result.artifact, run_id=result.run_id)


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "harness-runs"
    raise SystemExit(serve_harness(_demo_run, to_pointer, workdir_base=base))
