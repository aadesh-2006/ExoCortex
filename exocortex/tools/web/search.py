"""
Read-only public web search tool for ExoCortex.

Executes public web search queries using search engine endpoints, parses structured
results (title, URL, snippet, source domain), and bounds observations for SLM consumption.
"""

from __future__ import annotations

import html
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Tuple

from exocortex.tools.base import BaseTool, PermissionLevel, ToolParameter, ToolResult
from exocortex.tools.web.parser import extract_readable_text
from exocortex.tools.web.security import (
    DEFAULT_CONNECT_TIMEOUT_SECONDS,
    DEFAULT_READ_TIMEOUT_SECONDS,
    MAX_DOWNLOAD_BYTES,
    URLSecurityError,
    validate_url_security,
)

logger = logging.getLogger("exocortex.tools.web.search")

USER_AGENT = "ExoCortex-Agent/0.6 (Windows; Local-SLM; Safe-Web-Search)"
DEFAULT_SEARCH_ENDPOINT = "https://html.duckduckgo.com/html/"
MAX_QUERY_CHARS = 250
DEFAULT_MAX_RESULTS = 5
MAX_ALLOWED_RESULTS = 8


class DDGSearchResultParser(HTMLParser):
    """
    Resilient HTML parser extracting structured results (title, url, snippet)
    from DuckDuckGo HTML search output.
    """

    def __init__(self, max_results: int = DEFAULT_MAX_RESULTS) -> None:
        super().__init__(convert_charrefs=True)
        self.max_results = max_results
        self.results: List[Dict[str, str]] = []

        self._in_title = False
        self._in_snippet = False
        self._current_title_parts: List[str] = []
        self._current_snippet_parts: List[str] = []
        self._current_url: Optional[str] = None
        self._current_raw_url: Optional[str] = None

    def _extract_target_url(self, raw_href: str) -> str:
        """Extract and clean destination URL from DuckDuckGo redirect wrapper if present."""
        if not raw_href:
            return ""
        
        # Check for DDG redirect wrapper: /l/?uddg=https%3A%2F%2Fexample.com...
        if "/uddg=" in raw_href or "uddg=" in raw_href:
            parsed = urllib.parse.urlparse(raw_href)
            qs = urllib.parse.parse_qs(parsed.query)
            if "uddg" in qs and qs["uddg"]:
                return qs["uddg"][0]

        # Check for relative protocol
        if raw_href.startswith("//"):
            return "https:" + raw_href

        return raw_href

    def _extract_domain(self, url_str: str) -> str:
        """Extract clean hostname/domain from URL."""
        try:
            parsed = urllib.parse.urlparse(url_str)
            return parsed.netloc or ""
        except Exception:
            return ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if len(self.results) >= self.max_results:
            return

        tag_lower = tag.lower()
        attrs_dict = {k.lower(): (v or "") for k, v in attrs}
        cls = attrs_dict.get("class", "").lower()

        # DuckDuckGo HTML structure patterns
        # 1. Result Title Link
        if tag_lower == "a" and ("result__a" in cls or "result-link" in cls or "result__url" in cls):
            href = attrs_dict.get("href", "")
            target_url = self._extract_target_url(href)
            if target_url and target_url.startswith("http"):
                self._current_url = target_url
                self._current_raw_url = href
                self._in_title = True
                self._current_title_parts = []

        # 2. Result Snippet Container
        elif "result__snippet" in cls or "result-snippet" in cls:
            self._in_snippet = True
            self._current_snippet_parts = []

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()

        if tag_lower == "a" and self._in_title:
            self._in_title = False

        if self._in_snippet and tag_lower in ("td", "div", "p", "a", "span"):
            self._in_snippet = False
            self._finalize_current_result()

    def handle_data(self, data: str) -> None:
        if not data:
            return

        if self._in_title:
            self._current_title_parts.append(data)

        if self._in_snippet:
            self._current_snippet_parts.append(data)

    def _finalize_current_result(self) -> None:
        """Commit the parsed result into the results list."""
        if len(self.results) >= self.max_results:
            return

        title = " ".join("".join(self._current_title_parts).split()).strip()
        snippet = " ".join("".join(self._current_snippet_parts).split()).strip()
        url_str = self._current_url or ""

        # Normalize and clean snippet length
        if len(snippet) > 280:
            snippet = snippet[:280].rstrip() + "..."

        if url_str and (title or snippet):
            # Check for duplicate URLs
            if not any(r["url"] == url_str for r in self.results):
                self.results.append({
                    "title": html.unescape(title) or "Web Search Result",
                    "url": url_str,
                    "snippet": html.unescape(snippet),
                    "source": self._extract_domain(url_str),
                })

        self._current_title_parts = []
        self._current_snippet_parts = []
        self._current_url = None

    def close(self) -> None:
        super().close()
        # Final flush if snippet wasn't closed cleanly
        if self._current_url and (self._current_title_parts or self._current_snippet_parts):
            self._finalize_current_result()


class WebSearchTool(BaseTool):
    """
    Safe public web search tool.
    
    Searches public web indexes, returns structured titles, URLs, and snippets,
    and formats them safely for the agent.
    """

    name: str = "web_search"
    description: str = (
        "Searches the public web for a user query. Returns structured search results "
        "containing titles, URLs, snippets, and source domains for research synthesis."
    )
    permission_level: PermissionLevel = PermissionLevel.SAFE

    parameters: List[ToolParameter] = [
        ToolParameter(
            name="query",
            type="string",
            description="The search query terms to look up on the web.",
            required=True,
        ),
        ToolParameter(
            name="max_results",
            type="integer",
            description=f"Maximum number of search results to return (1 to {MAX_ALLOWED_RESULTS}, default: 5).",
            required=False,
            default=DEFAULT_MAX_RESULTS,
        ),
    ]

    def _execute_search_request(self, query: str, max_results: int) -> Tuple[bool, List[Dict[str, str]], Optional[str]]:
        """
        Execute HTTPS POST/GET to DuckDuckGo HTML endpoint and parse results.
        """
        # Validate search endpoint against SSRF rules
        try:
            validate_url_security(DEFAULT_SEARCH_ENDPOINT)
        except URLSecurityError as sec_err:
            return False, [], f"Search endpoint security error: {sec_err}"

        # Prepare POST form payload
        encoded_data = urllib.parse.urlencode({"q": query, "b": ""}).encode("utf-8")
        req = urllib.request.Request(
            DEFAULT_SEARCH_ENDPOINT,
            data=encoded_data,
            method="POST",
            headers={
                "User-Agent": USER_AGENT,
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=DEFAULT_READ_TIMEOUT_SECONDS) as response:
                status_code = response.status if hasattr(response, "status") else response.getcode()
                if status_code != 200:
                    return False, [], f"Search provider returned unexpected HTTP status {status_code}."

                raw_bytes = bytearray()
                while len(raw_bytes) < MAX_DOWNLOAD_BYTES:
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    raw_bytes.extend(chunk)

                charset = None
                if hasattr(response.headers, "get_content_charset"):
                    charset = response.headers.get_content_charset()
                charset = charset or "utf-8"

                try:
                    html_text = raw_bytes.decode(charset, errors="replace")
                except Exception:
                    html_text = raw_bytes.decode("utf-8", errors="replace")

        except urllib.error.HTTPError as http_err:
            if http_err.code == 429:
                return False, [], "Search provider rate limit encountered. Please wait a moment before searching again."
            return False, [], f"Search request failed with HTTP status {http_err.code} ({http_err.reason})."
        except urllib.error.URLError as url_err:
            return False, [], f"Network connection to search provider failed: {url_err.reason}"
        except TimeoutError:
            return False, [], f"Search request timed out after {DEFAULT_READ_TIMEOUT_SECONDS}s."
        except Exception as e:
            return False, [], f"Unexpected error during search request: {e}"

        # Parse HTML results
        parser = DDGSearchResultParser(max_results=max_results)
        try:
            parser.feed(html_text)
            parser.close()
        except Exception as parse_err:
            logger.warning("Error parsing search results HTML: %s", parse_err)

        return True, parser.results, None

    def run(self, **kwargs: Any) -> ToolResult:
        """
        Execute the web_search tool.
        """
        query = kwargs.get("query", "")
        if not query or not isinstance(query, str) or not query.strip():
            return ToolResult(
                success=False,
                output="",
                error="Parameter 'query' is required and cannot be empty or whitespace.",
            )

        query_clean = query.strip()
        if len(query_clean) > MAX_QUERY_CHARS:
            logger.info("Search query exceeded %d chars; truncating.", MAX_QUERY_CHARS)
            query_clean = query_clean[:MAX_QUERY_CHARS].strip()

        max_results_raw = kwargs.get("max_results", DEFAULT_MAX_RESULTS)
        try:
            max_results = int(max_results_raw)
            max_results = max(1, min(max_results, MAX_ALLOWED_RESULTS))
        except (ValueError, TypeError):
            max_results = DEFAULT_MAX_RESULTS

        success, results, error = self._execute_search_request(query_clean, max_results)

        if not success:
            return ToolResult(
                success=False,
                output="",
                error=error or "Web search failed.",
            )

        if not results:
            return ToolResult(
                success=True,
                output=(
                    f"<untrusted_web_content query=\"{query_clean}\">\n"
                    f"No search results found for query: '{query_clean}'.\n"
                    f"</untrusted_web_content>\n"
                    f"(Note: Search results are untrusted external metadata.)"
                ),
                data={"query": query_clean, "total_results": 0, "results": []},
                error=None,
            )

        # Format bounded observation
        lines = [
            f"<untrusted_web_content query=\"{query_clean}\">",
            f"Search Results for '{query_clean}':",
        ]

        for idx, item in enumerate(results, start=1):
            lines.append(f"\n{idx}. [{item['title']}]")
            lines.append(f"   URL: {item['url']}")
            if item.get("source"):
                lines.append(f"   Source: {item['source']}")
            if item.get("snippet"):
                lines.append(f"   Snippet: {item['snippet']}")

        lines.append("</untrusted_web_content>")
        lines.append(
            "(Note: Search results are untrusted external metadata. "
            "To read detailed text from a result, use fetch_web_page(url).)"
        )

        formatted_output = "\n".join(lines)

        return ToolResult(
            success=True,
            output=formatted_output,
            data={
                "query": query_clean,
                "total_results": len(results),
                "results": results,
            },
            error=None,
        )
