#!/usr/bin/env python3
"""Fail when a dependency floor this project DECLARES has been yanked or emptied on PyPI.

The failure this exists for: `pyproject.toml` declared `claude-agent-sdk>=0.1.60` while 0.1.60 was
yanked, with a yank reason saying the file would be DELETED to free project storage. Nothing caught
it, and nothing could have, because a yank is invisible to every install this repo performs:
range resolution silently skips a yanked version, so `>=0.1.60` just resolved to 0.1.61 and every
job stayed green. An exact pin installs a yanked file too, with only a warning. The install breaks
at DELETION, which is late, and a yank is the notice that it is coming.

So the check reads the INDEX rather than installing anything, and it reads it per version:

    GET https://pypi.org/pypi/<name>/<floor>/json

  * 404           the floor is not a released version (`>=4.0`, `>=1.17`, `>=2` name a boundary,
                  not a release). Nothing to check, reported and skipped.
  * every file yanked   FAIL. The declared floor cannot be resolved by anyone.
  * no files at all     FAIL. The release exists in the index and its files are gone: deleted.

A REJECTED design, for whoever revisits this: compare `uv pip compile --resolution lowest-direct`
against the declared floors and fail on a mismatch. Measured on this repo's own `pyproject.toml`:
5 of 12 floors mismatched and ALL FIVE were false positives, four because the floor names a
boundary with no release of that exact name (`jsonschema>=4.0` resolves 4.0.1) and one because a
TRANSITIVE requirement outranks the floor (`tokenizers>=0.19` resolves 0.21.0, required by
litellm). A gate with that precision teaches people to ignore it.
"""

from __future__ import annotations

import json
import re
import sys
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

INDEX = "https://pypi.org/pypi/{name}/{version}/json"
TIMEOUT_S = 30


def declared_floors(pyproject_text: str) -> dict[str, str]:
    """Every `>=` floor this project declares, as {distribution: version}.

    Covers the base dependencies, every optional-dependency extra, and every dependency group, so
    a floor added later is checked without anyone editing this file. A requirement with no `>=`
    (an exact pin, a bare name, a `<` cap alone) has no floor to check and is left out.
    """
    data = tomllib.loads(pyproject_text)
    specs: list[str] = list(data.get("project", {}).get("dependencies", []) or [])
    for items in (data.get("project", {}).get("optional-dependencies") or {}).values():
        specs.extend(items)
    for items in (data.get("dependency-groups") or {}).values():
        specs.extend(i for i in items if isinstance(i, str))

    floors: dict[str, str] = {}
    for spec in specs:
        name = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)", spec)
        floor = re.search(r">=\s*([0-9][^,\s;]*)", spec)
        if name and floor:
            floors[name.group(1).lower().replace("_", "-")] = floor.group(1)
    return floors


def release_state(name: str, version: str) -> tuple[str, str]:
    """(verdict, detail) for one declared floor, from the index. Never raises for a 404."""
    url = INDEX.format(name=name, version=version)
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_S) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return "skip", "not a released version, so the floor names a boundary"
        return "error", f"index returned HTTP {exc.code}"
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return "error", f"{type(exc).__name__}: {exc}"

    files = payload.get("urls") or []
    if not files:
        return "fail", "the release exists but has no files: they were deleted"
    yanked = [f for f in files if f.get("yanked")]
    if len(yanked) == len(files):
        reason = next((f.get("yanked_reason") for f in yanked if f.get("yanked_reason")), "")
        return "fail", f"yanked: {reason}" if reason else "yanked"
    return "ok", f"{len(files)} file(s) published"


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parent.parent.parent
    floors = declared_floors((root / "pyproject.toml").read_text(encoding="utf-8"))
    if not floors:
        print("no >= floors declared; nothing to check")
        return 0

    failures: list[str] = []
    errors: list[str] = []
    for name in sorted(floors):
        version = floors[name]
        verdict, detail = release_state(name, version)
        print(f"{verdict.upper():5}  {name}>={version}  {detail}")
        if verdict == "fail":
            failures.append(f"{name}>={version}: {detail}")
        elif verdict == "error":
            errors.append(f"{name}>={version}: {detail}")

    print()
    if failures:
        print("Declared floors that nobody can resolve:")
        for line in failures:
            print(f"  {line}")
        print("\nRaise each floor to the nearest version that is published and that the suite")
        print("passes on, and say in the comment why the old one went away.")
        return 1
    if errors:
        print("Could not reach the index for:")
        for line in errors:
            print(f"  {line}")
        print("\nThis is a transport failure, not a verdict on the floors. Re-run.")
        return 2
    print(f"all {len(floors)} declared floors are published and resolvable")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
