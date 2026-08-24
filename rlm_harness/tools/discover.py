"""``list_candidate_paths`` — a safe, good-default way to compute ``make_read_file_tool``/
``make_grep_files_tool``'s ``candidate_paths``.

``fs.py`` deliberately leaves ``candidate_paths`` REQUIRED, consumer-supplied — no default
directory walk, no built-in ``.gitignore`` handling (the base/wrap split: the kit owns the safety
guard, the consumer decides which files are even candidates). But safely walking a directory tree
— respecting ``.gitignore``, never escaping ``root`` via a symlink, never treating VCS internals
as candidates — is exactly the kind of mechanic that's easy to get subtly wrong and that every
consumer ends up reinventing. This ships a real, safe DEFAULT implementation for exactly that,
while leaving ``candidate_paths`` itself an unchanged plain list: a sophisticated consumer can
still ignore this helper entirely and pass its own.

**A single plain, host-side function — not a REPL tool.** Called by a consumer's own setup code
(e.g. a task's ``__init__``, before wiring ``make_grep_files_tool(root,
candidate_paths=...)``) — the same "public primitive a consumer's own code calls" role
``resolve_within_root`` already plays, not something the model calls mid-trajectory. A future,
separately-reviewable round could wrap this as a ``make_list_files_tool`` REPL tool; explicitly
out of scope here.

**Why ``pathspec`` (optional extra ``gitignore``), not a hand-rolled ``.gitignore`` parser.**
``.gitignore`` syntax is subtler than it looks (negation ``!pattern``, directory-only
``pattern/``, anchoring with a leading ``/``, ``**`` semantics) — a parser that's slightly wrong
either leaks files a consumer explicitly meant to exclude (a real information-disclosure risk:
``.gitignore`` routinely excludes ``.env``, credentials, build secrets) or wrongly excludes real
source files. ``pathspec`` is a small, pure-Python, no-C-extension library implementing git's own
``gitwildmatch`` syntax correctly — the same "don't reinvent a correctness-critical mechanic"
reasoning that chose ``regex`` over hand-rolled backtracking-safe matching for
``make_grep_files_tool``, and ``jsonschema`` over a hand-rolled validator.

Lazily imported, and ONLY when there's an actual pattern to compile (a real ``root/.gitignore``
file exists and ``respect_gitignore=True``, or ``extra_ignore_patterns`` is non-empty) — a caller
with neither never needs the dependency installed. When patterns DO need compiling and
``pathspec`` is missing, this raises a friendly ``ImportError`` — no silent "ignore nothing"
fallback, which would be the exact information-disclosure risk this exists to prevent.

**Scope, stated honestly: root-level ``.gitignore`` only, not per-directory nested ones.** Real
git merges ``.gitignore`` files from every directory in the tree, each scoped to its own
directory — correctly replicating that is a much larger, subtler feature. This ships the single
most common case (one ``.gitignore`` at the repo root) with a stated limitation rather than a
half-correct attempt at full nested semantics. A ``.gitignore`` inside a subdirectory is simply
not read. Likewise, a directory pruned as ignored is never re-visited, so a ``!``-negated
re-include for a path INSIDE it will not be resurfaced — a known, accepted simplification.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from fnmatch import fnmatch

from .fs import resolve_within_root


@dataclass(frozen=True)
class CandidatePaths:
    """The result of one ``list_candidate_paths`` walk — a plain, frozen value object (same
    convention as ``RunUtilization``/``Skill``/``RLMConfig``); never crosses into the REPL
    directly, so there's no dspy JSON-bridging concern the way there would be for a REPL tool's
    own return value."""

    paths: list[str]
    truncated: bool  # True if max_files was hit -- never a silent cutoff


def list_candidate_paths(
    root: str,
    *,
    glob: str = "*",
    respect_gitignore: bool = True,
    extra_ignore_patterns: Sequence[str] = (),
    max_files: int = 5000,
    follow_symlinks: bool = False,
) -> CandidatePaths:
    """Walk ``root`` and return the relative paths of every real, in-bounds file — a safe default
    for building ``make_read_file_tool``/``make_grep_files_tool``'s ``candidate_paths``.

    ``respect_gitignore`` (default ``True``): apply ``root``'s OWN top-level ``.gitignore`` (not
    any nested per-directory ones — see the module docstring). ``extra_ignore_patterns`` (default
    none): additional ``gitwildmatch``-syntax patterns, merged into the same pattern set — the
    escape hatch for a consumer that knows it needs more than one root-level file covers. Needs
    the optional ``pathspec`` package (``pip install "rlm-harness[gitignore]"``) ONLY when there's
    an actual pattern to compile (a real ``.gitignore`` file present, or ``extra_ignore_patterns``
    given) — raises a friendly ``ImportError`` otherwise, no silent fallback to "ignore nothing."

    ``.git`` is ALWAYS excluded, unconditionally, regardless of ``respect_gitignore``/
    ``extra_ignore_patterns`` — both as a directory (pruned from the walk, never descended into)
    and as a plain FILE (git's real submodule gitlink shape: a submodule's working tree contains a
    one-line pointer FILE named ``.git``, not a directory — pruning the directory case alone would
    miss it and leak the pointer into the result).

    ``glob`` (default ``"*"``): an ``fnmatch``-style pattern applied AFTER ignore-filtering,
    against the same relative-path shape ``candidate_paths`` itself expects — same semantics as
    ``make_grep_files_tool``'s own ``glob``, so the result is directly pipeable with zero
    reshaping: ``make_grep_files_tool(root, candidate_paths=list_candidate_paths(root).paths)``.

    ``follow_symlinks`` (default ``False``): whether ``os.walk`` descends into a symlinked
    DIRECTORY. Independent of that flag, every candidate FILE is re-resolved through
    ``resolve_within_root`` before being kept — a symlink FILE pointing outside ``root`` is
    excluded either way, since ``follow_symlinks`` only gates directory descent.

    ``max_files`` (default ``5000``) bounds the walk itself — checked as soon as the cap is
    reached, stopping the walk outright rather than slicing an unbounded result after the fact.
    ``CandidatePaths.truncated`` makes a partial result visible and checkable, never a silent
    cutoff indistinguishable from "that's really every file." Directory and file names are sorted
    at every level before use — ``os.walk``'s own yielded order is filesystem/OS-dependent and
    otherwise unspecified, which would make WHICH files survive a ``max_files`` truncation
    non-reproducible across runs on the same tree; sorting gives a fully deterministic result
    (the same convention this kit's own ``skills.py`` already uses for a directory listing).
    """
    root_real = os.path.realpath(root)
    gitignore_path = os.path.join(root_real, ".gitignore")
    needs_patterns = bool(extra_ignore_patterns) or (
        respect_gitignore and os.path.isfile(gitignore_path)
    )

    spec = None
    if needs_patterns:
        try:
            import pathspec
        except ImportError as exc:
            raise ImportError(
                "list_candidate_paths needs the optional 'pathspec' package to correctly parse "
                ".gitignore syntax (negation, directory-only patterns, anchoring) -- a hand-rolled "
                "parser risks leaking files a consumer meant to exclude. Install it with:  "
                'pip install "rlm-harness[gitignore]"'
            ) from exc
        lines: list[str] = []
        if respect_gitignore and os.path.isfile(gitignore_path):
            with open(gitignore_path, encoding="utf-8") as fh:
                lines.extend(fh.read().splitlines())
        lines.extend(extra_ignore_patterns)
        # "gitignore" (not the older "gitwildmatch" factory name) -- same gitwildmatch syntax and
        # matching behavior, but "gitwildmatch" is deprecated as of pathspec 1.x and emits a
        # DeprecationWarning; "gitignore" is warning-free on both the pinned floor (0.12.0) and
        # current releases, verified empirically against both.
        spec = pathspec.PathSpec.from_lines("gitignore", lines)

    paths: list[str] = []
    truncated = False

    for dirpath, dirnames, filenames in os.walk(root_real, followlinks=follow_symlinks):
        # Sorted in PLACE (not reassigned) -- os.walk reads this exact list object back to decide
        # what to descend into next; a rebind would not propagate. Deterministic order is what
        # makes a max_files truncation reproducible across runs on the same tree.
        dirnames.sort()
        filenames.sort()

        rel_dir = os.path.relpath(dirpath, root_real)
        rel_dir = "" if rel_dir == "." else rel_dir.replace(os.sep, "/")

        if ".git" in dirnames:
            dirnames.remove(".git")  # never descended into, at any depth

        if spec is not None:
            kept_dirs = []
            for d in dirnames:
                candidate_dir = f"{rel_dir}/{d}" if rel_dir else d
                # A trailing slash is REQUIRED for pathspec's gitwildmatch to recognize a
                # directory-only pattern (e.g. "build/") -- without it, such a pattern silently
                # fails to match and the directory is never pruned.
                if not spec.match_file(f"{candidate_dir}/"):
                    kept_dirs.append(d)
            dirnames[:] = kept_dirs

        for filename in filenames:
            if filename == ".git":
                continue  # a submodule's gitlink pointer FILE, not a directory -- never a candidate
            rel_path = f"{rel_dir}/{filename}" if rel_dir else filename
            if spec is not None and spec.match_file(rel_path):
                continue
            if not fnmatch(rel_path, glob):
                continue
            if resolve_within_root(root_real, rel_path) is None:
                continue  # a symlink escaping root -- excluded, not an error
            paths.append(rel_path)
            if len(paths) >= max_files:
                truncated = True
                break
        if truncated:
            break

    return CandidatePaths(paths=paths, truncated=truncated)
