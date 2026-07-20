"""``python -m rlm_kit.harness_serve <pkg.module:run> [workdir_base]`` — the zero-file way to serve a
harness over the delegation contract (the runnable front-end of :func:`rlm_kit.serve_harness`).

Resolves the harness's ``run`` callable from ``<module:attr>`` and, if the same module exposes a
``to_pointer`` (or ``TO_POINTER``), uses it; otherwise falls back to the duck-typed extractor (for a
harness whose ``run()`` already returns a flat ``.artifact``/``.run_id``/``.trace_path`` object). A
harness with a NESTED result writes a ~5-line ``serve`` module in its OWN repo that calls
``serve_harness(run, to_pointer, …)`` directly — that module, not this one, is what its operator points
at; this ``-m`` entry is the convenience path when no mapping is needed. The kit names no harness: the
target is a runtime argument, exactly like the client's endpoint config.
"""

from __future__ import annotations

import importlib
import sys
from typing import Optional

from .serving import ToPointer, serve_harness


def _resolve(spec: str):
    """``pkg.module:attr`` → the attribute. Errors are actionable (bad spec / missing module/attr)."""
    if ":" not in spec:
        raise SystemExit(f"bad target {spec!r} — expected 'package.module:run'")
    mod_name, _, attr = spec.partition(":")
    try:
        mod = importlib.import_module(mod_name)
    except ModuleNotFoundError as exc:
        raise SystemExit(f"cannot import {mod_name!r}: {exc}") from None
    if not hasattr(mod, attr):
        raise SystemExit(f"{mod_name!r} has no attribute {attr!r}")
    return getattr(mod, attr), mod


def main(argv: Optional[list] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        raise SystemExit(
            "usage: python -m rlm_kit.harness_serve <package.module:run> [workdir_base]\n"
            "  reads the long-text spec on STDIN, runs the harness, prints one JSON pointer on STDOUT."
        )
    run, mod = _resolve(args[0])
    workdir_base = args[1] if len(args) > 1 else "harness-runs"
    to_pointer: Optional[ToPointer] = getattr(mod, "to_pointer", None) or getattr(mod, "TO_POINTER", None)
    run_kwargs = getattr(mod, "SERVE_RUN_KWARGS", None)
    env_files = getattr(mod, "SERVE_ENV_FILES", ())
    kwargs = {"workdir_base": workdir_base, "run_kwargs": run_kwargs, "env_files": env_files}
    if to_pointer is not None:
        return serve_harness(run, to_pointer, **kwargs)
    return serve_harness(run, **kwargs)


if __name__ == "__main__":
    raise SystemExit(main())
