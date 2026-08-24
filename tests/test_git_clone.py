"""make_git_clone_tool -- safe git clone with fallback auth. All offline, dspy-free (the
`cloner` is always a stub in these tests; no real git/network activity)."""
import types

import pytest

from rlm_harness.testing import assert_repl_safe, assert_task_repl_safe
from rlm_harness.tools import make_git_clone_tool
from rlm_harness.tools.command import CommandResult
from rlm_harness.trace import EVENT_TOOL_CALL, TraceRecorder, load_events


def _duck_task(signature="q: str -> a: str", tools=()):
    return types.SimpleNamespace(signature=signature, tools=list(tools), output_field="a")


def _calls(results):
    """A stub cloner that returns `results` in order (one per call) and records every call's
    (url, dest, depth, creds) tuple onto `.seen`."""
    seen = []
    it = iter(results)

    def cloner(url, dest, depth, creds):
        seen.append((url, dest, depth, creds))
        return next(it)

    cloner.seen = seen
    return cloner


def test_git_clone_success_on_first_attempt():
    cloner = _calls([CommandResult(exit_code=0)])
    tool = make_git_clone_tool("/tmp/root", cloner)
    result = tool("https://example.com/repo.git", "dest")
    assert result == "Cloned 'https://example.com/repo.git' into 'dest'."
    assert len(cloner.seen) == 1


def test_git_clone_success_trace_has_no_credential_fields_when_unused(tmp_path):
    cloner = _calls([CommandResult(exit_code=0)])
    tool = make_git_clone_tool(str(tmp_path), cloner)
    trace_path = str(tmp_path / "t.jsonl")
    with TraceRecorder(trace_path, run_id="r1"):
        tool("https://example.com/repo.git", "dest")
    tc = [e for e in load_events(trace_path) if e["type"] == EVENT_TOOL_CALL][0]
    assert tc["payload"]["ok"] is True
    assert tc["payload"]["used_fallback_auth"] is False


def test_git_clone_unsafe_url_refused_before_cloner_runs():
    cloner = _calls([CommandResult(exit_code=0)])
    tool = make_git_clone_tool("/tmp/root", cloner)
    result = tool("file:///etc/passwd", "dest")
    assert result.startswith("Refused")
    assert len(cloner.seen) == 0


def test_git_clone_dest_escaping_root_refused_before_cloner_runs(tmp_path):
    cloner = _calls([CommandResult(exit_code=0)])
    tool = make_git_clone_tool(str(tmp_path), cloner)
    result = tool("https://example.com/repo.git", "../../etc")
    assert result.startswith("Refused")
    assert len(cloner.seen) == 0


def test_git_clone_fails_then_succeeds_with_fallback_auth():
    cloner = _calls([CommandResult(exit_code=1, stderr="auth required"), CommandResult(exit_code=0)])
    tool = make_git_clone_tool(
        "/tmp/root", cloner, get_credentials=lambda url: {"secret": "tok3n"}
    )
    result = tool("https://example.com/repo.git", "dest")
    assert result == "Cloned 'https://example.com/repo.git' into 'dest'."
    assert len(cloner.seen) == 2
    assert cloner.seen[0][3] is None            # first attempt: no credentials
    assert cloner.seen[1][3] == {"secret": "tok3n"}  # second attempt: carries credentials


def test_git_clone_redacts_secret_from_stderr_and_return_string(tmp_path):
    cloner = _calls(
        [
            CommandResult(exit_code=1, stderr="auth required"),
            CommandResult(exit_code=1, stderr="fatal: bad credential tok3n-secret"),
        ]
    )
    tool = make_git_clone_tool(
        str(tmp_path), cloner, get_credentials=lambda url: {"secret": "tok3n-secret"}
    )
    trace_path = str(tmp_path / "t.jsonl")
    with TraceRecorder(trace_path, run_id="r1"):
        result = tool("https://example.com/repo.git", "dest")
    assert "tok3n-secret" not in result
    assert "[REDACTED]" in result
    tc = [e for e in load_events(trace_path) if e["type"] == EVENT_TOOL_CALL][0]
    assert "tok3n-secret" not in tc["payload"]["stderr_preview"]


def test_git_clone_redacts_secret_from_stdout_too(tmp_path):
    cloner = _calls(
        [
            CommandResult(exit_code=1, stderr="auth required"),
            CommandResult(exit_code=1, stdout="echoed tok3n-secret back", stderr=""),
        ]
    )
    tool = make_git_clone_tool(
        str(tmp_path), cloner, get_credentials=lambda url: {"secret": "tok3n-secret"}
    )
    trace_path = str(tmp_path / "t.jsonl")
    with TraceRecorder(trace_path, run_id="r1"):
        tool("https://example.com/repo.git", "dest")
    tc = [e for e in load_events(trace_path) if e["type"] == EVENT_TOOL_CALL][0]
    # stdout isn't itself traced verbatim (only stdout_len) -- the secret must not leak via its
    # LENGTH being suspicious either; the real assertion is that redaction ran on stdout before
    # anything derived from it could carry the secret. Confirm by checking the tool's own return
    # string, which is built from the (redacted) stderr_preview, never raw stdout content.
    assert tc["payload"]["stdout_len"] == len("echoed [REDACTED] back")


def test_git_clone_declines_fallback_when_get_credentials_returns_none():
    cloner = _calls([CommandResult(exit_code=1, stderr="nope")])
    tool = make_git_clone_tool("/tmp/root", cloner, get_credentials=lambda url: None)
    result = tool("https://example.com/repo.git", "dest")
    assert "Clone failed" in result
    assert len(cloner.seen) == 1


def test_git_clone_no_credentials_provider_configured_single_attempt():
    cloner = _calls([CommandResult(exit_code=1, stderr="nope")])
    tool = make_git_clone_tool("/tmp/root", cloner)
    result = tool("https://example.com/repo.git", "dest")
    assert "Clone failed" in result
    assert len(cloner.seen) == 1


def test_git_clone_default_depth_passed_through_to_every_attempt():
    cloner = _calls([CommandResult(exit_code=1, stderr="nope"), CommandResult(exit_code=0)])
    tool = make_git_clone_tool(
        "/tmp/root", cloner, get_credentials=lambda url: {"secret": "x"}, default_depth=3
    )
    tool("https://example.com/repo.git", "dest")
    assert cloner.seen[0][2] == 3
    assert cloner.seen[1][2] == 3


def test_git_clone_exception_on_first_attempt_triggers_fallback():
    def cloner(url, dest, depth, creds):
        cloner.calls.append(creds)
        if len(cloner.calls) == 1:
            raise ConnectionError("boom")
        return CommandResult(exit_code=0)

    cloner.calls = []
    tool = make_git_clone_tool(
        "/tmp/root", cloner, get_credentials=lambda url: {"secret": "tok"}
    )
    result = tool("https://example.com/repo.git", "dest")
    assert result == "Cloned 'https://example.com/repo.git' into 'dest'."
    assert len(cloner.calls) == 2
    assert cloner.calls[1] == {"secret": "tok"}


def test_git_clone_exception_on_both_attempts_is_an_error_string_not_raised():
    def cloner(url, dest, depth, creds):
        cloner.calls.append(creds)
        raise ConnectionError("boom")

    cloner.calls = []
    tool = make_git_clone_tool(
        "/tmp/root", cloner, get_credentials=lambda url: {"secret": "tok"}
    )
    result = tool("https://example.com/repo.git", "dest")
    assert "Clone failed" in result
    assert len(cloner.calls) == 2  # never more than two attempts


def test_git_clone_malformed_credentials_missing_secret_key_treated_as_decline():
    cloner = _calls([CommandResult(exit_code=1, stderr="nope")])
    tool = make_git_clone_tool("/tmp/root", cloner, get_credentials=lambda url: {"user": "x"})
    result = tool("https://example.com/repo.git", "dest")
    assert "Clone failed" in result
    assert len(cloner.seen) == 1


def test_git_clone_malformed_credentials_empty_secret_treated_as_decline():
    cloner = _calls([CommandResult(exit_code=1, stderr="nope")])
    tool = make_git_clone_tool("/tmp/root", cloner, get_credentials=lambda url: {"secret": ""})
    result = tool("https://example.com/repo.git", "dest")
    assert "Clone failed" in result
    assert len(cloner.seen) == 1


def test_git_clone_get_credentials_raising_is_treated_as_decline():
    cloner = _calls([CommandResult(exit_code=1, stderr="nope")])

    def boom(url):
        raise RuntimeError("credentials backend down")

    tool = make_git_clone_tool("/tmp/root", cloner, get_credentials=boom)
    result = tool("https://example.com/repo.git", "dest")
    assert "Clone failed" in result
    assert len(cloner.seen) == 1


def test_git_clone_name_override_sets_repl_identity_and_trace_tag(tmp_path):
    cloner = _calls([CommandResult(exit_code=0)])
    tool = make_git_clone_tool(str(tmp_path), cloner, name="clone_deps")
    assert tool.__name__ == "clone_deps"
    trace_path = str(tmp_path / "t.jsonl")
    with TraceRecorder(trace_path, run_id="r1"):
        tool("https://example.com/repo.git", "dest")
    tc = [e for e in load_events(trace_path) if e["type"] == EVENT_TOOL_CALL][0]
    assert tc["payload"]["tool"] == "clone_deps"


def test_git_clone_name_override_fixes_the_real_multi_root_collision():
    cloner_a = _calls([CommandResult(exit_code=0)])
    cloner_b = _calls([CommandResult(exit_code=0)])
    a = make_git_clone_tool("/tmp/source-root", cloner_a)
    b = make_git_clone_tool("/tmp/docs-root", cloner_b)
    with pytest.raises(AssertionError, match="duplicate REPL tool name"):
        assert_task_repl_safe(_duck_task(tools=[a, b]))

    a2 = make_git_clone_tool("/tmp/source-root", cloner_a, name="clone_source")
    b2 = make_git_clone_tool("/tmp/docs-root", cloner_b, name="clone_docs")
    assert_task_repl_safe(_duck_task(tools=[a2, b2]))  # must not raise


def test_git_clone_name_invalid_identifier_raises_value_error():
    cloner = _calls([CommandResult(exit_code=0)])
    with pytest.raises(ValueError, match="not a valid tool name"):
        make_git_clone_tool("/tmp/x", cloner, name="git-clone")


def test_git_clone_name_reserved_by_dspy_raises_value_error():
    from rlm_harness._dspy_compat import reserved_tool_names

    cloner = _calls([CommandResult(exit_code=0)])
    reserved = next(iter(reserved_tool_names()))
    with pytest.raises(ValueError, match="reserved by dspy's sandbox"):
        make_git_clone_tool("/tmp/x", cloner, name=reserved)


def test_git_clone_is_repl_safe():
    cloner = _calls([CommandResult(exit_code=0)])
    assert_repl_safe(make_git_clone_tool("/tmp/root", cloner))
