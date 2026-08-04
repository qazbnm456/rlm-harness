"""Phase A — expose a directory of Skills to the RLM as tools.

A "Skill" here follows the common convention of a folder containing a ``SKILL.md``
(with optional YAML-ish frontmatter for ``name``/``description``), or a flat
``<name>.md`` file. The main LM decides, inside the REPL, which skill to read —
keeping control flow in the LM's hands (the run stays an RLM, and the decision
lands in the trajectory).

Two DISCOVERY models, per the ``discovery`` arg of :func:`load_skills_as_tools`
(progressive disclosure, Anthropic Agent-Skills style):

- ``"list"`` — surface a ``list_skills`` tool; the LM calls it to see the catalog
  (a discovery round-trip that costs one tool turn). The default; backward-compatible.
- ``"inject"`` — the caller injects the catalog into the system prompt once at
  startup via :func:`render_skills_manifest`, so the LM always knows which skills
  exist without a tool call; only ``read_skill`` (the just-in-time full-body pull)
  is surfaced as a tool.

Pure stdlib; no dspy import.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

from .trace import record_tool_call

_PREVIEW_CHARS = 700   # head of a read skill recorded for inspection (a replay UI shows what was read)


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    path: str

    def read(self) -> str:
        with open(self.path, "r", encoding="utf-8") as fh:
            return fh.read()


def _parse_frontmatter(text: str) -> dict:
    """Extract simple ``key: value`` pairs from a leading ``---`` block."""
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    meta: dict = {}
    for line in text[3:end].splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip().lower()] = value.strip()
    return meta


def discover_skills(skill_dir: str) -> list[Skill]:
    """Find skills under ``skill_dir``: ``*/SKILL.md`` folders and ``*.md`` files."""
    skills: list[Skill] = []
    if not os.path.isdir(skill_dir):
        return skills

    for entry in sorted(os.listdir(skill_dir)):
        full = os.path.join(skill_dir, entry)
        skill_md = os.path.join(full, "SKILL.md")
        if os.path.isdir(full) and os.path.isfile(skill_md):
            meta = _parse_frontmatter(_safe_read(skill_md))
            skills.append(
                Skill(
                    name=meta.get("name", entry),
                    description=meta.get("description", ""),
                    path=skill_md,
                )
            )
        elif os.path.isfile(full) and entry.endswith(".md"):
            meta = _parse_frontmatter(_safe_read(full))
            skills.append(
                Skill(
                    name=meta.get("name", entry[:-3]),
                    description=meta.get("description", ""),
                    path=full,
                )
            )
    return skills


def _safe_read(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read(4096)  # frontmatter only; enough for metadata
    except OSError:
        return ""


def render_skills_manifest(skill_dir: str, *, header: str | None = None) -> str:
    """Render the skill catalog — one ``- name: description`` line per skill — for injection
    into a system prompt. This is the DISCOVERY level of progressive disclosure (Anthropic
    Agent Skills): surface every skill's metadata ONCE at startup so the model always knows
    which skills exist, instead of spending a turn calling ``list_skills`` to find out. Pair it
    with ``load_skills_as_tools(skill_dir, discovery="inject")``, which then omits ``list_skills``
    and keeps only ``read_skill`` (the just-in-time full-body pull).

    Returns ``""`` when the directory holds no skills (nothing to inject). ``header``, when
    given, is prepended on its own line (e.g. an ``<available_skills>`` label).
    """
    skills = discover_skills(skill_dir)
    if not skills:
        return ""
    body = "\n".join(
        f"- {s.name}: {s.description or '(no description)'}" for s in skills
    )
    return f"{header}\n{body}" if header else body


def load_skills_as_tools(skill_dir: str, *, discovery: str = "list") -> list[Callable]:
    """Return the skill tools over ``skill_dir``.

    ``discovery`` selects HOW the model learns which skills exist:

    - ``"list"`` (default): return ``[list_skills, read_skill]``. The model calls ``list_skills``
      to see the catalog — a discovery round-trip that costs one tool turn. Backward-compatible.
    - ``"inject"``: return ``[read_skill]`` ONLY. The caller injects the catalog into the system
      prompt via :func:`render_skills_manifest` (skill metadata auto-surfaced at startup, no
      discovery tool call). ``read_skill`` stays as the just-in-time activation path.

    ``read_skill`` (and ``list_skills`` when present) records a ``tool_call`` event so skill
    access shows up in the trajectory and the RL dataset.
    """
    if discovery not in ("list", "inject"):
        raise ValueError(f"discovery must be 'list' or 'inject', got {discovery!r}")
    skills = {s.name: s for s in discover_skills(skill_dir)}

    def list_skills() -> str:
        """List the available skills and their one-line descriptions."""
        if not skills:
            result = "No skills available."
        else:
            result = "\n".join(
                f"- {s.name}: {s.description or '(no description)'}"
                for s in skills.values()
            )
        record_tool_call("list_skills", args={}, result=result)
        return result

    def read_skill(name: str) -> str:
        """Read the full content of a named skill, by name."""
        skill: Skill | None = skills.get(name)
        # Name the alternatives on a miss. Under discovery="inject" there is no `list_skills` to
        # fall back to, so an error that only says "no such skill" leaves the model with no move.
        known = ", ".join(sorted(skills)) or "(none)"
        result = skill.read() if skill else f"No such skill: {name!r}. Available: {known}."
        # `preview` (a head of the content) is for inspection — a trace reader / replay UI can show
        # WHAT was read, not just how long it was. `result_len` keeps the full size.
        record_tool_call("read_skill", args={"name": name}, result_len=len(result),
                         preview=result[:_PREVIEW_CHARS])
        return result

    # The docstring IS the tool description the model reads, and where the NAMES come from differs
    # by mode — so it is set per mode rather than written once at `def` time. Under "inject" there
    # is no `list_skills` tool to point at; naming it anyway would be a dead instruction, and a
    # NameError if the model took it literally in the REPL. (`read_skill` is a fresh closure per
    # call, so assigning `__doc__` here cannot leak into another caller's tool.)
    if discovery == "inject":
        read_skill.__doc__ = ("Read the full content of a named skill. Names are the ones listed "
                              "in the skills manifest in your instructions.")
        return [read_skill]
    read_skill.__doc__ = "Read the full content of a named skill. Call list_skills first for names."
    return [list_skills, read_skill]
