import pytest

from rlm_kit.skills import (
    discover_skills,
    load_skills_as_tools,
    render_skills_manifest,
)
from rlm_kit.trace import EVENT_TOOL_CALL, TraceRecorder, load_events


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
