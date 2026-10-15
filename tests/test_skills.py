import pytest

from rlm_harness.skills import (
    discover_skills,
    load_skills_as_tools,
    render_skills_manifest,
)
from rlm_harness.trace import EVENT_TOOL_CALL, TraceRecorder, load_events


def _make_skill_dir(tmp_path):
    # Folder-style skill with SKILL.md + frontmatter
    sd = tmp_path / "skills"
    (sd / "recon").mkdir(parents=True)
    (sd / "recon" / "SKILL.md").write_text(
        "---\nname: recon\ndescription: Recon helper\n---\nDo recon steps.\n",
        encoding="utf-8",
    )
    # Flat .md skill, no frontmatter
    (sd / "triage.md").write_text("Triage the finding.\n", encoding="utf-8")
    return str(sd)


def test_discover_skills(tmp_path):
    skills = {s.name: s for s in discover_skills(_make_skill_dir(tmp_path))}
    assert set(skills) == {"recon", "triage"}
    assert skills["recon"].description == "Recon helper"
    assert "recon steps" in skills["recon"].read()


def test_discover_missing_dir_returns_empty():
    assert discover_skills("/no/such/dir") == []


def test_skills_tools_list_and_read(tmp_path):
    list_skills, read_skill = load_skills_as_tools(_make_skill_dir(tmp_path))
    listing = list_skills()
    assert "recon" in listing and "triage" in listing
    assert "recon steps" in read_skill("recon")
    assert "No such skill" in read_skill("ghost")


def test_skills_tools_record_events(tmp_path):
    list_skills, read_skill = load_skills_as_tools(_make_skill_dir(tmp_path))
    path = str(tmp_path / "t.jsonl")
    with TraceRecorder(path, run_id="r1"):
        list_skills()
        read_skill("recon")
    tools = [e for e in load_events(path) if e["type"] == EVENT_TOOL_CALL]
    assert [t["payload"]["tool"] for t in tools] == ["list_skills", "read_skill"]


def test_render_skills_manifest(tmp_path):
    sd = _make_skill_dir(tmp_path)
    manifest = render_skills_manifest(sd)
    # one `- name: description` line per skill; missing description falls back to a placeholder
    assert "- recon: Recon helper" in manifest
    assert "- triage: (no description)" in manifest
    # header, when given, is prepended on its own line
    with_header = render_skills_manifest(sd, header="<available_skills>")
    assert with_header.startswith("<available_skills>\n- ")


def test_render_manifest_empty_when_no_skills():
    # nothing to inject → empty string (caller can skip the block)
    assert render_skills_manifest("/no/such/dir") == ""


def test_load_skills_discovery_inject_omits_list_skills(tmp_path):
    # discovery="inject" surfaces ONLY read_skill (the JIT pull); discovery is handled by the
    # injected manifest, so there is no list_skills tool.
    tools = load_skills_as_tools(_make_skill_dir(tmp_path), discovery="inject")
    assert [t.__name__ for t in tools] == ["read_skill"]
    assert "recon steps" in tools[0]("recon")


def test_load_skills_discovery_list_is_default_and_backward_compatible(tmp_path):
    # default stays the 2-tuple [list_skills, read_skill] so existing callers keep working.
    tools = load_skills_as_tools(_make_skill_dir(tmp_path))
    assert [t.__name__ for t in tools] == ["list_skills", "read_skill"]


def test_load_skills_discovery_invalid_raises(tmp_path):
    with pytest.raises(ValueError, match="discovery must be"):
        load_skills_as_tools(_make_skill_dir(tmp_path), discovery="bogus")


def test_inject_mode_does_not_point_at_a_tool_it_does_not_register(tmp_path):
    """The docstring is the tool DESCRIPTION the model reads. Under `discovery="inject"` there is
    no `list_skills` tool, so naming it would send the planner to a name that raises NameError in
    the REPL. The catalog comes from the caller's injected manifest instead."""
    skill_dir = _make_skill_dir(tmp_path)

    (read_skill,) = load_skills_as_tools(skill_dir, discovery="inject")
    assert "list_skills" not in (read_skill.__doc__ or "")
    assert "manifest" in (read_skill.__doc__ or "")

    listed = load_skills_as_tools(skill_dir, discovery="list")
    assert [t.__name__ for t in listed] == ["list_skills", "read_skill"]
    # The default mode keeps its own pointer valid: the tool it names IS registered.
    assert "list_skills" in (listed[1].__doc__ or "")


def test_an_unknown_skill_name_reports_the_available_ones(tmp_path):
    """Under "inject" there is no `list_skills` to recover with, so the miss has to carry them."""
    (read_skill,) = load_skills_as_tools(_make_skill_dir(tmp_path), discovery="inject")

    out = read_skill("gamma")

    assert "No such skill: 'gamma'" in out
    assert "recon, triage" in out


# --- KNOWLEDGE-only, with a gate -----------------------------------------------------------------
#
# `docs/INVARIANTS.md` says `read_skill` "returns markdown TEXT only; it must NOT execute bundled
# scripts or expand linked files (don't add silent exec)". A mutation audit of all twenty
# mechanically-violable invariants found this one alone had no gate: making `read_skill` run a
# subprocess and append its output left the WHOLE SUITE green. Every other rule in that file went
# red under a mutation that violated it.

def _skill_dir(tmp_path, body: str):
    (tmp_path / "s.md").write_text(f"---\nname: s\ndescription: d\n---\n{body}")
    return tmp_path


def test_read_skill_returns_the_file_and_nothing_else(tmp_path):
    """Byte-for-byte, which is what makes "executes nothing" checkable from the outside.

    Asserted as EQUALITY rather than containment: an exec or an expansion appends, and a
    containment check passes while the model reads output no file holds.
    """
    from rlm_harness import load_skills_as_tools

    body = "# S\n\nRun `echo hello` and see [the other file](other.md).\n"
    d = _skill_dir(tmp_path, body)
    (d / "other.md").write_text("LINKED CONTENT THAT MUST NOT APPEAR")
    read_skill = next(t for t in load_skills_as_tools(str(d), discovery="inject")
                      if t.__name__ == "read_skill")

    got = read_skill("s")
    # Equality carries the whole rule. A containment check on the command's own output cannot:
    # whatever marker `echo` would print is already in the line that asks for it.
    assert got == (d / "s.md").read_text(), (
        "read_skill must return the file's text and nothing else: no bundled script runs and no "
        "linked file is expanded, because the sandbox is the only place code runs"
    )
    assert "LINKED CONTENT" not in got, "a linked file was expanded"


def test_read_skill_does_not_run_a_subprocess(tmp_path, monkeypatch):
    """The direct half. The equality check above catches an exec whose output lands in the
    result; this catches one whose output goes anywhere else."""
    import subprocess

    from rlm_harness import load_skills_as_tools

    called: list = []

    def _refuse(*a, **k):
        called.append(a)
        raise AssertionError("read_skill ran a subprocess")

    monkeypatch.setattr(subprocess, "run", _refuse)
    monkeypatch.setattr(subprocess, "Popen", _refuse)
    monkeypatch.setattr(subprocess, "check_output", _refuse)

    d = _skill_dir(tmp_path, "# S\n\nbody\n")
    read_skill = next(t for t in load_skills_as_tools(str(d), discovery="inject")
                      if t.__name__ == "read_skill")
    read_skill("s")
    assert not called
