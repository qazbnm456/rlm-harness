"""``make_git_clone_tool`` — safe git clone with fallback auth, base/wrap, same shape as
``make_fetch_tool``/``make_command_tool`` (not a new pattern).

A task that wants to clone a repository to analyze it has no safe way to do so without this: (1)
SSRF-shaped URL abuse (a ``file://`` URL, an internal git server); (2) a private repo needs
credentials, but git's own default behavior is to hang on an interactive terminal prompt or, worse,
have a caller embed a token directly in the URL where it leaks into logs/traces; (3) being tricked
into cloning an enormous repository; (4) the destination must stay confined to the task's bounded
root, exactly like every other filesystem-touching tool in this kit.

**The kit does NOT shell out to ``git`` itself.** ``command.py``'s own module docstring is explicit
about why: executing a model-adjacent operation host-side needs ISOLATION, and "the kit ships NO
executor and picks NO isolation mechanism" for ``run_command`` — the exact same reasoning applies
here. A ``git clone`` is not meaningfully safer to execute un-isolated than an arbitrary command (a
malicious git server can exploit a client vulnerability; a cloned repo's own hooks can execute code
unless disabled). So ``make_git_clone_tool`` takes a CONSUMER-SUPPLIED, isolated ``cloner`` — the
kit's job is the same "enforce the safe half, never the isolation" role ``make_fetch_tool`` already
plays for its ``fetcher`` and ``make_command_tool`` plays for its ``runner``.

**URL safety reuses ``is_safe_url`` directly** — no reinvented SSRF check. Same caveat as
``fetch.py`` documents for itself: a SYNTACTIC pre-flight only, not a DNS-rebinding-safe check (a
public hostname resolving to a private address at actual connect time) — the wrapper cannot itself
call ``resolved_host_is_safe`` at the right moment, since the real network connection happens
INSIDE the isolated ``cloner``, not here. The ``cloner`` should call ``resolved_host_is_safe``
internally at connect time if that matters for its deployment — the exact same division of
responsibility ``make_fetch_tool`` already documents for its own ``fetcher``.

**Destination confinement reuses ``resolve_within_root`` directly** — ``dest_dir`` is resolved
exactly like ``write_file``'s ``path``; a ``..``-escape or symlink-outside-root is refused before
the cloner ever runs.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from ..trace import record_tool_call
from .command import CommandResult
from .fetch import is_safe_url
from .fs import _validate_tool_name, resolve_within_root

_STDERR_PREVIEW = 500

# (validated url, confined absolute dest path, shallow-clone depth or None, credentials dict or
# None) -> CommandResult. SYNC, isolated -- same contract as command.py's own Runner. Never
# exported from __all__ -- Runner/Guard (command.py) and Fetcher (fetch.py) aren't either; a
# module-level annotation only, matching that precedent.
Cloner = Callable[[str, str, int | None, dict | None], CommandResult]

# (url) -> a credentials dict or None (declines to supply one). A non-None dict MUST include a
# "secret" key (the raw credential string, so the wrapper can redact it from anything model/
# trace-visible) -- a dict missing it, or whose "secret" isn't a non-empty str, is treated exactly
# like a None return (fails closed: no retry attempted, never a crash).
CredentialsProvider = Callable[[str], dict | None]


def _redact(text: str, secret: str | None) -> str:
    if not secret:
        return text
    return text.replace(secret, "[REDACTED]")


def _valid_secret(creds: dict | None) -> str | None:
    if not isinstance(creds, dict):
        return None
    secret = creds.get("secret")
    return secret if isinstance(secret, str) and secret else None


def make_git_clone_tool(
    root: str,
    cloner: Cloner,
    *,
    name: str = "git_clone",
    get_credentials: CredentialsProvider | None = None,
    default_depth: int | None = 1,
) -> Callable[[str, str], str]:
    """Build a ``git_clone``-shaped tool scoped to ``root`` — wired in a task's ``__init__``
    (per-run state, never a classvar).

    ``name`` (default ``"git_clone"``): same rationale and mechanism as
    :func:`rlm_harness.tools.make_read_file_tool`'s ``name`` — ``git_clone`` binds to a ``root``
    at construction time, so a task wanting two differently-scoped clone tools needs the same
    multi-root collision fix every filesystem-mutating factory in this kit already has.

    ``cloner`` — a CONSUMER-SUPPLIED, ISOLATED, sync callable: ``(url, dest_path, depth, creds) ->
    CommandResult``. The kit ships none; see the module docstring for why. ``cloner`` should call
    :func:`rlm_harness.tools.resolved_host_is_safe` internally at connect time if DNS-rebinding
    matters for its deployment — ``is_safe_url`` below is syntactic only.

    ``get_credentials`` (default ``None``): an optional ``(url) -> dict | None`` provider for the
    fallback-auth retry (see ``git_clone`` below). ``default_depth`` (default ``1``): a shallow
    clone by default, passed through to ``cloner`` as a plain argument (the cloner decides how to
    honor it) — the "avoid being tricked into cloning an enormous repository" mitigation; ``None``
    opts out for a caller that explicitly wants full history. Both are factory (operator)
    parameters, never model-controlled, matching ``make_grep_files_tool``'s own
    factory-level-not-call-level configuration precedent.
    """
    _validate_tool_name(name)

    def _attempt(url: str, dest: str, creds: dict | None) -> CommandResult:
        try:
            return cloner(url, dest, default_depth, creds)
        except Exception as exc:
            return CommandResult(exit_code=1, stderr=f"{type(exc).__name__}: {exc}")

    def git_clone(url: str, dest_dir: str) -> str:
        """Clone ``url`` into ``dest_dir`` (relative to the root) via the injected, isolated
        ``cloner``. Refuses (returns a string, never raises) an unsafe URL or a ``dest_dir`` that
        escapes the root, before the cloner ever runs. On a failed first attempt, if a
        credentials provider is configured, retries ONCE with credentials — never more than two
        clone attempts total. A supplied credential's raw secret value is redacted (exact-string
        match only — does not catch a derived/transformed leak such as URL-encoding or a
        truncated echo) from every model/trace-visible string after a credentialed attempt."""
        if not is_safe_url(url):
            record_tool_call(
                name, args={"url": url}, ok=False,
                note="refused: not a permitted external http(s) URL",
            )
            return f"Refused: {url!r} is not a permitted external http(s) URL."
        resolved_dest = resolve_within_root(root, dest_dir)
        if resolved_dest is None:
            record_tool_call(
                name, args={"dest_dir": dest_dir}, ok=False, note="refused: escapes root"
            )
            return f"Refused: {dest_dir!r} is not a path inside this root."

        _t0 = time.perf_counter()   # spans BOTH attempts; see the record below
        result = _attempt(url, resolved_dest, None)
        used_fallback_auth = False
        secret = None
        if result.exit_code != 0 and get_credentials is not None:
            try:
                creds = get_credentials(url)
            except Exception:
                creds = None
            secret = _valid_secret(creds)
            if secret is not None:
                used_fallback_auth = True
                result = _attempt(url, resolved_dest, creds)

        stdout = _redact(result.stdout, secret)
        stderr_preview = _redact(result.stderr[:_STDERR_PREVIEW], secret)
        record_tool_call(
            name,
            args={"url": url, "dest_dir": dest_dir},
            # Covers BOTH attempts when the credentialed fallback ran — that is the honest total
            # this tool cost the turn, and on a large repo it is routinely the slowest call in it.
            duration_s=time.perf_counter() - _t0,
            ok=(result.exit_code == 0),
            exit_code=result.exit_code,
            used_fallback_auth=used_fallback_auth,
            stdout_len=len(stdout),
            stderr_preview=stderr_preview,
        )
        if result.exit_code != 0:
            # stderr_preview is already redacted above; the URL/exit-code portions of this
            # message never carry the secret (the wrapper never embeds one into a URL itself).
            return f"Clone failed for {url!r} (exit {result.exit_code}): {stderr_preview}"
        return f"Cloned {url!r} into {dest_dir!r}."

    git_clone.__name__ = name
    git_clone.__qualname__ = name
    return git_clone
