"""
ExoCortex Safe Web Research Suite.

Provides secure, read-only web research tools, URL sanitizers, SSRF defenses, and HTML parsing.
"""

from __future__ import annotations

from exocortex.tools.web.fetch import (
    FetchWebPageTool,
    PinnedHTTPConnection,
    PinnedHTTPSConnection,
    PinnedHTTPHandler,
    PinnedHTTPSHandler,
    build_pinned_opener,
)
from exocortex.tools.web.parser import (
    SafeHTMLTextExtractor,
    extract_readable_text,
)
from exocortex.tools.web.search import (
    DDGSearchResultParser,
    WebSearchTool,
)
from exocortex.tools.web.security import (
    ALLOWED_PORTS,
    ALLOWED_SCHEMES,
    DEFAULT_CONNECT_TIMEOUT_SECONDS,
    DEFAULT_READ_TIMEOUT_SECONDS,
    MAX_DOWNLOAD_BYTES,
    MAX_EXTRACTED_CHARS,
    MAX_REDIRECTS,
    URLSecurityError,
    is_ip_blocked,
    resolve_and_verify_ip,
    validate_hostname_syntax,
    validate_redirect_target,
    validate_url_security,
)

__all__ = [
    "ALLOWED_PORTS",
    "ALLOWED_SCHEMES",
    "DEFAULT_CONNECT_TIMEOUT_SECONDS",
    "DEFAULT_READ_TIMEOUT_SECONDS",
    "MAX_DOWNLOAD_BYTES",
    "MAX_EXTRACTED_CHARS",
    "MAX_REDIRECTS",
    "URLSecurityError",
    "is_ip_blocked",
    "resolve_and_verify_ip",
    "validate_hostname_syntax",
    "validate_redirect_target",
    "validate_url_security",
    "extract_readable_text",
    "SafeHTMLTextExtractor",
    "FetchWebPageTool",
    "PinnedHTTPConnection",
    "PinnedHTTPSConnection",
    "PinnedHTTPHandler",
    "PinnedHTTPSHandler",
    "build_pinned_opener",
    "WebSearchTool",
    "DDGSearchResultParser",
]
