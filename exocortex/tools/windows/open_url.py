"""
Safe URL opening tool for ExoCortex.

Opens validated http/https web URLs in the user's default browser.
Rejects file://, javascript:, data:, and all local/shell URI schemes.
"""

from __future__ import annotations

import logging
import urllib.parse
import webbrowser
from typing import Any, Dict

from exocortex.tools.base import (
    BaseTool,
    PermissionLevel,
    ToolParameter,
    ToolResult,
)

logger = logging.getLogger("exocortex.tools.windows.open_url")

ALLOWED_SCHEMES = {"http", "https"}


class OpenUrlTool(BaseTool):
    """Tool to safely open validated web URLs in the default browser."""

    name = "open_url"
    description = (
        "Opens a web address in the user's default browser. "
        "Strictly restricted to 'http://' and 'https://' URLs."
    )
    permission_level = PermissionLevel.SAFE
    parameters = [
        ToolParameter(
            name="url",
            type="string",
            description="The full HTTP or HTTPS URL to open (e.g. 'https://www.google.com').",
            required=True,
        )
    ]

    def run(self, **kwargs: Any) -> ToolResult:
        raw_url = kwargs.get("url")
        if not raw_url or not isinstance(raw_url, str):
            return ToolResult(
                success=False,
                output="",
                error="Missing or invalid 'url' parameter.",
            )

        clean_url = raw_url.strip()
        if not clean_url:
            return ToolResult(
                success=False,
                output="",
                error="URL cannot be empty.",
            )

        try:
            parsed = urllib.parse.urlparse(clean_url)
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Malformed URL: {str(e)}",
            )

        scheme = (parsed.scheme or "").lower()
        if scheme not in ALLOWED_SCHEMES:
            return ToolResult(
                success=False,
                output="",
                error=(
                    f"Forbidden URL scheme '{scheme}'. "
                    f"Only 'http' and 'https' protocols are permitted. "
                    "File, javascript, data, and shell schemes are rejected for security."
                ),
            )

        if not parsed.netloc:
            return ToolResult(
                success=False,
                output="",
                error=f"Invalid URL format: missing domain or host in '{clean_url}'.",
            )

        try:
            opened = webbrowser.open(clean_url)
            logger.info("Opened URL in default browser: %s", clean_url)
            return ToolResult(
                success=True,
                output=f"Successfully opened URL: {clean_url}",
                data={"url": clean_url, "scheme": scheme, "netloc": parsed.netloc},
            )
        except Exception as e:
            logger.error("Failed to open browser for URL '%s': %s", clean_url, str(e))
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to open URL in browser: {str(e)}",
            )
