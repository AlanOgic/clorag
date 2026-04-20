"""SSRF-safe URL validation for outbound fetches.

Restricts outbound URLs to an allowlisted set of public hostname suffixes
and blocks loopback, private, link-local and reserved IP ranges. Applied
to any code path that fetches a URL supplied (directly or indirectly) by
a request, to prevent attackers from pivoting through the application to
internal services.
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse

DEFAULT_ALLOWED_HOST_SUFFIXES: tuple[str, ...] = (
    "cyanview.com",
    "cyanview.cloud",
)

_BLOCKED_HOST_NAMES = frozenset({"localhost", "localhost.localdomain", "ip6-localhost"})


class UrlNotAllowedError(ValueError):
    """Raised when a URL fails outbound-fetch validation."""


def validate_public_url(
    url: str,
    *,
    allowed_suffixes: tuple[str, ...] = DEFAULT_ALLOWED_HOST_SUFFIXES,
) -> str:
    """Validate a URL is safe to fetch from the server.

    Enforces:
      - scheme in {http, https}
      - hostname present and not a known loopback alias
      - hostname is not a raw IP address (blocks IMDS 169.254.169.254,
        localhost, RFC1918, link-local and reserved ranges)
      - hostname ends with one of ``allowed_suffixes``

    Returns the URL unchanged on success; raises ``UrlNotAllowedError``
    with a descriptive message on failure.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("https", "http"):
        raise UrlNotAllowedError(
            f"URL scheme must be http or https, got {parsed.scheme!r}"
        )
    host = (parsed.hostname or "").lower()
    if not host:
        raise UrlNotAllowedError("URL has no hostname")
    if host in _BLOCKED_HOST_NAMES:
        raise UrlNotAllowedError(f"Hostname {host!r} is blocked")

    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if ip is not None:
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise UrlNotAllowedError(
                f"URL points to a non-public IP address: {host}"
            )
        raise UrlNotAllowedError(
            "Raw IP hostnames are not allowed; use a registered hostname"
        )

    if not any(host == s or host.endswith(f".{s}") for s in allowed_suffixes):
        raise UrlNotAllowedError(
            f"Hostname {host!r} is not in the allowed suffixes"
            f" {allowed_suffixes}"
        )
    return url
