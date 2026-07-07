"""SSRF-aware fetch tooling for RLM tasks.

Tasks routinely need to pull remote content (web pages, docs, feeds, threat
intel). Handing an LLM-driven REPL an unrestricted fetcher is an SSRF liability:
the model can be steered — by the very untrusted content it is analysing — into
requesting internal services or cloud metadata endpoints.

``is_safe_url`` is a syntactic pre-flight guard (scheme + obvious
private/loopback/metadata targets). ``make_fetch_tool`` wraps a caller-supplied
(SYNC) fetcher with that guard, returning a sync tool ready to hand to
``RLMTask(tools=…)`` — dspy.RLM invokes tools synchronously, so the tool must be sync.

NOTE: a syntactic guard does not stop DNS rebinding (a public hostname that
resolves to a private address). ``resolved_host_is_safe`` is that missing
resolved-address re-check — call it INSIDE your fetcher at connection time (and
on every redirect hop). Its ``allow_nets`` carve-out (build with ``parse_cidrs``)
accommodates a fake-IP proxy / split-DNS VPN that maps public hosts into a
reserved range; it is a re-usable primitive so each consumer's ``direct`` fetcher
shares ONE correct implementation instead of re-deriving it.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from typing import Callable
from urllib.parse import urlparse

from ..trace import record_tool_call

logger = logging.getLogger(__name__)

ALLOWED_SCHEMES = frozenset({"http", "https"})

# Hostnames that must never be fetched regardless of resolution.
_BLOCKED_HOSTNAMES = frozenset(
    {
        "localhost",
        "metadata.google.internal",
    }
)


def _ip_in_blocked_range(ip) -> bool:
    """True if a parsed ``ip_address`` falls in a loopback / private / link-local (incl.
    169.254.169.254 cloud metadata) / reserved / unspecified / multicast range."""
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_unspecified
        or ip.is_multicast
    )


def _hostname_is_blocked_literal_ip(hostname: str) -> bool:
    try:
        ip = ipaddress.ip_address(hostname)
    except ValueError:
        return False
    return _ip_in_blocked_range(ip)


def parse_cidrs(cidrs) -> tuple:
    """Parse an iterable of CIDR strings into networks for ``resolved_host_is_safe(allow_nets=…)``.
    An unparseable entry is warned-and-skipped (never fatal) so a typo can't sink a run."""
    nets = []
    for c in cidrs or ():
        try:
            nets.append(ipaddress.ip_network(c, strict=False))
        except ValueError:
            logger.warning("ignoring invalid allow-CIDR %r", c)
    return tuple(nets)


def _ip_blocked(ip_str: str, allow_nets=()) -> bool:
    """True if ``ip_str`` must be refused. Unparseable → blocked (fail closed). An operator-listed
    ``allow_nets`` range is treated as external (the proxy, not the resolved address, is the real
    endpoint) — everything else falls through to the standard blocked-range check."""
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True
    if any(ip in n for n in allow_nets):
        return False
    return _ip_in_blocked_range(ip)


def resolved_host_is_safe(host: str, port: int, *, allow_nets=()) -> bool:
    """The DNS-rebinding defence: resolve ``host`` and return True only if EVERY resolved address is
    external. Call this INSIDE your fetcher, re-checking each redirect hop — ``is_safe_url`` is
    syntactic and cannot see what a hostname resolves to.

    ``allow_nets`` (from ``parse_cidrs``) carves out operator-trusted ranges: a fake-IP proxy /
    split-DNS VPN (e.g. Clash/Mihomo/Surge default ``198.18.0.0/16``) maps every public host into a
    RESERVED range that would otherwise be refused, starving the model of source. Empty by default =
    full strictness (``is_safe_url`` still refuses localhost/metadata regardless of ``allow_nets``)."""
    if not host:
        return False
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except OSError:
        return False
    return bool(infos) and all(not _ip_blocked(info[4][0], allow_nets) for info in infos)


def is_safe_url(url: str) -> bool:
    """Return True if ``url`` is an http(s) URL not pointing at an obvious
    internal/loopback/metadata target.

    Conservative by design: anything it cannot parse or recognise is rejected.
    """
    if not url or not isinstance(url, str):
        return False
    try:
        parsed = urlparse(url.strip())
    except Exception:
        return False

    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        return False

    hostname = (parsed.hostname or "").lower()
    if not hostname:
        return False
    if hostname in _BLOCKED_HOSTNAMES:
        return False
    if hostname.endswith(".local") or hostname.endswith(".internal"):
        return False
    if _hostname_is_blocked_literal_ip(hostname):
        return False

    return True


# A fetcher maps a URL to its text content. SYNC: dspy.RLM calls tools synchronously
# (its sandbox bridge never awaits — an async tool serialises to a coroutine repr and
# never runs), so both the fetcher and the tool are sync. Wrap an async client yourself.
Fetcher = Callable[[str], str]


def make_fetch_tool(fetcher: Fetcher) -> Callable[[str], str]:
    """Wrap ``fetcher`` with the SSRF guard, returning a SYNC tool for dspy.RLM.

    SYNC because dspy.RLM's interpreter invokes tools synchronously (no await); an
    ``async def`` tool there returns an un-awaited coroutine the model never sees the
    result of, so ``fetcher`` must be sync too.

    The wrapper rejects unsafe URLs before the fetcher ever runs, and turns a fetcher
    error into a short string too (rather than raising), so the RLM can react to either
    as text. Each call records a ``tool_call`` carrying only the outcome (``ok`` +
    ``result_len`` / ``note``), NOT the fetched body — see below.
    """

    def fetch_url(url: str) -> str:
        """Fetch the text content at an http(s) URL. Internal, loopback, and
        cloud-metadata addresses are refused."""
        if not is_safe_url(url):
            record_tool_call(
                "fetch_url", args={"url": url}, ok=False,
                note="refused: not a permitted external http(s) URL",
            )
            return f"Refused: {url!r} is not a permitted external http(s) URL."
        try:
            result = fetcher(url)
        except Exception as exc:  # noqa: BLE001 — surface as text so the RLM can react
            record_tool_call(
                "fetch_url", args={"url": url}, ok=False,
                note=f"error: {type(exc).__name__}",
            )
            return f"Fetch error for {url!r}: {type(exc).__name__}: {str(exc)[:160]}"
        # Record status + size only, NOT the body. In an RLM the fetched text lands in
        # a REPL variable the model slices itself, so the body is bulk content that would
        # bloat the JSONL source-of-truth for no RL value (mirrors ``read_skill`` and
        # a consumer's fetch tool). Replay therefore serves this length, not the bytes.
        record_tool_call("fetch_url", args={"url": url}, ok=True, result_len=len(result))
        return result

    return fetch_url
