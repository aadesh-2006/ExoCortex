"""
Network security, URL validation, and SSRF defense primitives for ExoCortex web tools.

Provides strict URL sanitization, hostname and DNS resolution checks, private/loopback
IP address blocking, port validation, and safe redirect validation.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from typing import Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

logger = logging.getLogger("exocortex.tools.web.security")

# Security and Resource Limits
DEFAULT_CONNECT_TIMEOUT_SECONDS: float = 5.0
DEFAULT_READ_TIMEOUT_SECONDS: float = 8.0
MAX_DOWNLOAD_BYTES: int = 1024 * 1024  # 1 MB maximum response payload
MAX_EXTRACTED_CHARS: int = 3000        # Maximum extracted text length
MAX_REDIRECTS: int = 3

# Allowed schemes and standard ports
ALLOWED_SCHEMES: Set[str] = {"http", "https"}
ALLOWED_PORTS: Set[int] = {80, 443}

# Blocked hostnames and suffixes
BLOCKED_HOSTNAME_SUFFIXES: Tuple[str, ...] = (
    "localhost",
    ".local",
    ".internal",
    ".lan",
    ".home.arpa",
    ".corp",
    ".onion",
    ".test",
    ".invalid",
    ".example",
)

# Explicit Carrier-Grade NAT (RFC 6598) network
CGNAT_NETWORK = ipaddress.ip_network("100.64.0.0/10")


class URLSecurityError(Exception):
    """Raised when a URL violates security policies or SSRF defenses."""
    pass


def is_ip_blocked(ip_addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """
    Check if an IP address belongs to loopback, private, link-local, multicast,
    unspecified, reserved, or CGNAT ranges.
    """
    if (
        ip_addr.is_loopback
        or ip_addr.is_private
        or ip_addr.is_link_local
        or ip_addr.is_multicast
        or ip_addr.is_unspecified
        or ip_addr.is_reserved
    ):
        return True

    # Carrier-Grade NAT check for IPv4
    if ip_addr.version == 4 and ip_addr in CGNAT_NETWORK:
        return True

    return False


def validate_hostname_syntax(hostname: str) -> None:
    """
    Verify hostname format and check against blocked local/internal domain suffixes.
    """
    if not hostname:
        raise URLSecurityError("URL must contain a valid non-empty hostname.")

    host_lower = hostname.lower().strip(".")

    # Reject localhost or blocked domain suffixes
    if host_lower == "localhost":
        raise URLSecurityError("Access to localhost is prohibited.")

    for suffix in BLOCKED_HOSTNAME_SUFFIXES:
        if suffix.startswith(".") and host_lower.endswith(suffix):
            raise URLSecurityError(f"Access to private/internal domain suffix '{suffix}' is prohibited.")
        elif not suffix.startswith(".") and host_lower == suffix:
            raise URLSecurityError(f"Access to private/internal domain '{suffix}' is prohibited.")


def resolve_and_verify_ip(hostname: str, port: int) -> str:
    """
    Resolve hostname to IP addresses via DNS and verify that all resolved IPs
    are public, non-private, non-loopback addresses.
    
    Returns:
        The first verified public IP address string.
    """
    # Check if the hostname itself is a literal IP address
    is_literal = False
    try:
        literal_ip = ipaddress.ip_address(hostname)
        is_literal = True
    except (ValueError, TypeError):
        is_literal = False

    if is_literal:
        if is_ip_blocked(literal_ip):
            raise URLSecurityError(
                f"Access to private/restricted IP address '{literal_ip}' is prohibited."
            )
        return str(literal_ip)

    try:
        addr_info = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise URLSecurityError(f"DNS resolution failed for hostname '{hostname}': {e}")
    except Exception as e:
        raise URLSecurityError(f"Socket resolution error for hostname '{hostname}': {e}")

    if not addr_info:
        raise URLSecurityError(f"No IP addresses resolved for hostname '{hostname}'.")

    resolved_ips = []
    for entry in addr_info:
        sockaddr = entry[4]
        ip_str = sockaddr[0]
        try:
            ip_obj = ipaddress.ip_address(ip_str)
        except (ValueError, TypeError):
            raise URLSecurityError(f"Invalid IP address resolved from DNS: '{ip_str}'")

        if is_ip_blocked(ip_obj):
            raise URLSecurityError(
                f"Hostname '{hostname}' resolved to prohibited IP address '{ip_str}'."
            )
        resolved_ips.append(ip_str)

    return resolved_ips[0]


def validate_url_security(url: str) -> Tuple[str, int, str]:
    """
    Perform complete security and SSRF validation on a candidate URL.

    Validation steps:
    1. Scheme check: only 'http' and 'https' allowed.
    2. Hostname check: non-empty, reject localhost and internal TLDs.
    3. Port check: only 80 and 443 allowed (explicit or default).
    4. DNS & IP check: resolves IP and blocks loopback, RFC1918, link-local, CGNAT, multicast.

    Returns:
        Tuple of (sanitized_url, port, verified_ip)
    
    Raises:
        URLSecurityError if any validation rule is violated.
    """
    if not url or not isinstance(url, str):
        raise URLSecurityError("URL must be a non-empty string.")

    url_trimmed = url.strip()
    parsed = urlparse(url_trimmed)

    # 1. Scheme Validation
    scheme = (parsed.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        raise URLSecurityError(
            f"Unsupported URL scheme '{scheme}'. Only 'http' and 'https' are permitted."
        )

    # 2. Hostname Validation
    hostname = parsed.hostname
    if not hostname:
        raise URLSecurityError("URL missing valid hostname or network location.")
    validate_hostname_syntax(hostname)

    # 3. Port Validation
    if parsed.port is not None:
        port = parsed.port
        if port not in ALLOWED_PORTS:
            raise URLSecurityError(
                f"Prohibited network port '{port}'. Only standard web ports (80, 443) are allowed."
            )
    else:
        port = 443 if scheme == "https" else 80

    # 4. Resolve and Validate IP
    verified_ip = resolve_and_verify_ip(hostname, port)

    return url_trimmed, port, verified_ip


def validate_redirect_target(original_url: str, redirect_location: str) -> str:
    """
    Validate that a redirect target URL is safe and adheres to all SSRF rules.
    Handles relative redirect paths by resolving against the original URL.
    
    Returns:
        The validated absolute redirect target URL.
    """
    if not redirect_location:
        raise URLSecurityError("Redirect location header is empty.")

    # Resolve relative redirect URLs against the base original URL
    target_url = urljoin(original_url, redirect_location.strip())
    validated_url, _, _ = validate_url_security(target_url)
    return validated_url
