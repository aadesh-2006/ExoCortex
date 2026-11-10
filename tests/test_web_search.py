"""
Unit and security tests for WebSearchTool and DuckDuckGo HTML parser.
"""

import io
import socket
import unittest
import urllib.error
import urllib.request
from unittest.mock import MagicMock, patch

from exocortex.tools.base import PermissionLevel
from exocortex.tools.web.search import DDGSearchResultParser, WebSearchTool


class DummySearchResponse:
    """Mock HTTP response object for search requests."""

    def __init__(self, body: bytes, status: int = 200, headers: dict = None):
        self._body = io.BytesIO(body)
        self.status = status
        self.code = status
        self.headers = headers or {"Content-Type": "text/html; charset=utf-8"}

    def read(self, amt: int = -1) -> bytes:
        return self._body.read(amt)

    def getcode(self) -> int:
        return self.status

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


SAMPLE_DDG_HTML = b"""
<html>
<body>
    <div class="result results_links">
        <a class="result__a" href="/l/?uddg=https%3A%2F%2Fwww.makemytrip.com%2Fflights%2Fvijayawada-pune.html">Vijayawada to Pune Flights | MakeMyTrip</a>
        <a class="result__snippet">Book cheap flights from Vijayawada to Pune starting from &#8377;4,200. Check timetable and flight duration.</a>
    </div>
    <div class="result results_links">
        <a class="result__a" href="/l/?uddg=https%3A%2F%2Fwww.ixigo.com%2Ftrains%2Fvijayawada-pune">Vijayawada to Pune Trains - Schedule &amp; Fares | ixigo</a>
        <a class="result__snippet">Find all trains running between Vijayawada and Pune. Konark Express departs at 23:20.</a>
    </div>
    <div class="result results_links">
        <a class="result__a" href="/l/?uddg=https%3A%2F%2Fwww.goindigo.in%2Fflight-booking%2Fvijayawada-pune">IndiGo Flights: Vijayawada (VGA) to Pune (PNQ)</a>
        <a class="result__snippet">Fly direct with IndiGo. Lowest airfares and daily flight offers on Pune routes.</a>
    </div>
</body>
</html>
"""


class TestWebSearchTool(unittest.TestCase):
    """Test WebSearchTool functionality, HTML parsing, bounds, and error handling."""

    def setUp(self):
        self.tool = WebSearchTool()

    def test_tool_metadata_and_schema(self):
        """Verify tool name, safe permission, and JSON schema."""
        self.assertEqual(self.tool.name, "web_search")
        self.assertEqual(self.tool.permission_level, PermissionLevel.SAFE)
        schema = self.tool.get_schema()
        self.assertEqual(schema["name"], "web_search")
        self.assertIn("query", schema["parameters"]["required"])
        self.assertIn("max_results", schema["parameters"]["properties"])

    def test_valid_search_query_and_multiple_results(self):
        """1 & 2. Valid query extracts multiple structured results."""
        mock_resp = DummySearchResponse(SAMPLE_DDG_HTML, status=200)

        with patch("socket.getaddrinfo", return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("52.142.124.215", 443))]), \
             patch("urllib.request.urlopen", return_value=mock_resp):
            result = self.tool.run(query="flights Vijayawada to Pune", max_results=3)

            self.assertTrue(result.success)
            self.assertEqual(result.data["total_results"], 3)
            self.assertEqual(len(result.data["results"]), 3)

            first = result.data["results"][0]
            self.assertEqual(first["title"], "Vijayawada to Pune Flights | MakeMyTrip")
            self.assertEqual(first["url"], "https://www.makemytrip.com/flights/vijayawada-pune.html")
            self.assertEqual(first["source"], "www.makemytrip.com")
            self.assertIn("₹4,200", first["snippet"])

            self.assertIn("<untrusted_web_content query=\"flights Vijayawada to Pune\">", result.output)
            self.assertIn("1. [Vijayawada to Pune Flights | MakeMyTrip]", result.output)
            self.assertIn("URL: https://www.makemytrip.com/flights/vijayawada-pune.html", result.output)

    def test_max_results_1(self):
        """3. max_results=1 limits results to 1."""
        mock_resp = DummySearchResponse(SAMPLE_DDG_HTML, status=200)

        with patch("socket.getaddrinfo", return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("52.142.124.215", 443))]), \
             patch("urllib.request.urlopen", return_value=mock_resp):
            result = self.tool.run(query="flights", max_results=1)
            self.assertTrue(result.success)
            self.assertEqual(result.data["total_results"], 1)

    def test_max_results_8(self):
        """4. max_results=8 allows up to 8."""
        mock_resp = DummySearchResponse(SAMPLE_DDG_HTML, status=200)

        with patch("socket.getaddrinfo", return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("52.142.124.215", 443))]), \
             patch("urllib.request.urlopen", return_value=mock_resp):
            result = self.tool.run(query="flights", max_results=8)
            self.assertTrue(result.success)
            self.assertLessEqual(result.data["total_results"], 8)

    def test_max_results_bounded_above_maximum(self):
        """5. max_results > 8 is capped to MAX_ALLOWED_RESULTS (8)."""
        mock_resp = DummySearchResponse(SAMPLE_DDG_HTML, status=200)

        with patch("socket.getaddrinfo", return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("52.142.124.215", 443))]), \
             patch("urllib.request.urlopen", return_value=mock_resp):
            result = self.tool.run(query="flights", max_results=100)
            self.assertTrue(result.success)
            self.assertLessEqual(result.data["total_results"], 8)

    def test_empty_query_rejected(self):
        """6. Empty query rejected."""
        result = self.tool.run(query="")
        self.assertFalse(result.success)
        self.assertIn("cannot be empty", result.error)

    def test_whitespace_query_rejected(self):
        """7. Whitespace-only query rejected."""
        result = self.tool.run(query="     \n\t  ")
        self.assertFalse(result.success)
        self.assertIn("cannot be empty", result.error)

    def test_excessively_long_query_handled(self):
        """8. Excessively long query is safely truncated."""
        long_query = "travel " * 100
        mock_resp = DummySearchResponse(SAMPLE_DDG_HTML, status=200)

        with patch("socket.getaddrinfo", return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("52.142.124.215", 443))]), \
             patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            result = self.tool.run(query=long_query)
            self.assertTrue(result.success)
            called_req = mock_urlopen.call_args[0][0]
            self.assertLessEqual(len(called_req.data), 400)

    def test_html_entity_decoding(self):
        """9. HTML entities like &amp;, &#8377;, &quot; are decoded."""
        html_data = b"""
        <div class="result">
            <a class="result__a" href="https://example.com/fare">Air India &amp; SpiceJet Fares</a>
            <a class="result__snippet">Price is &quot;&#8377;3,500&quot; inclusive of all taxes.</a>
        </div>
        """
        mock_resp = DummySearchResponse(html_data, status=200)

        with patch("socket.getaddrinfo", return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("52.142.124.215", 443))]), \
             patch("urllib.request.urlopen", return_value=mock_resp):
            result = self.tool.run(query="test")
            self.assertTrue(result.success)
            item = result.data["results"][0]
            self.assertEqual(item["title"], "Air India & SpiceJet Fares")
            self.assertEqual(item["snippet"], 'Price is "₹3,500" inclusive of all taxes.')

    def test_malformed_search_result_html(self):
        """10. Malformed/incomplete HTML does not crash parser."""
        broken_html = b"<div class='result'><a class='result__a' href='https://broken.com'>Unclosed title"
        mock_resp = DummySearchResponse(broken_html, status=200)

        with patch("socket.getaddrinfo", return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("52.142.124.215", 443))]), \
             patch("urllib.request.urlopen", return_value=mock_resp):
            result = self.tool.run(query="broken")
            self.assertTrue(result.success)
            self.assertEqual(len(result.data["results"]), 1)
            self.assertEqual(result.data["results"][0]["url"], "https://broken.com")

    def test_provider_network_failure(self):
        """11. Network failure returns clean ToolResult."""
        with patch("socket.getaddrinfo", return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("52.142.124.215", 443))]), \
             patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
            result = self.tool.run(query="test query")
            self.assertFalse(result.success)
            self.assertIn("Network connection to search provider failed", result.error)

    def test_timeout_handling(self):
        """12. Timeout returns clean ToolResult."""
        with patch("socket.getaddrinfo", return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("52.142.124.215", 443))]), \
             patch("urllib.request.urlopen", side_effect=TimeoutError("Read timeout")):
            result = self.tool.run(query="timeout query")
            self.assertFalse(result.success)
            self.assertIn("timed out", result.error)

    def test_http_error_and_rate_limiting(self):
        """13. HTTP 429 rate limit and HTTP 500 error return clean ToolResults."""
        # 429 Rate limit
        err_429 = urllib.error.HTTPError("https://html.duckduckgo.com/html/", 429, "Too Many Requests", {}, None)
        with patch("socket.getaddrinfo", return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("52.142.124.215", 443))]), \
             patch("urllib.request.urlopen", side_effect=err_429):
            result = self.tool.run(query="rate limit query")
            self.assertFalse(result.success)
            self.assertIn("rate limit encountered", result.error)

        # 500 Server error
        err_500 = urllib.error.HTTPError("https://html.duckduckgo.com/html/", 500, "Internal Server Error", {}, None)
        with patch("socket.getaddrinfo", return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("52.142.124.215", 443))]), \
             patch("urllib.request.urlopen", side_effect=err_500):
            result = self.tool.run(query="server error query")
            self.assertFalse(result.success)
            self.assertIn("HTTP status 500", result.error)

    def test_zero_results(self):
        """14. Zero search results returns informative output."""
        empty_html = b"<html><body><p>No results found for your query.</p></body></html>"
        mock_resp = DummySearchResponse(empty_html, status=200)

        with patch("socket.getaddrinfo", return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("52.142.124.215", 443))]), \
             patch("urllib.request.urlopen", return_value=mock_resp):
            result = self.tool.run(query="nonexistentquery12345xyz")
            self.assertTrue(result.success)
            self.assertEqual(result.data["total_results"], 0)
            self.assertIn("No search results found", result.output)

    def test_result_urls_preserved_correctly(self):
        """15. Destination URLs in uddg parameters are decoded cleanly."""
        html_uddg = b"""
        <div class="result">
            <a class="result__a" href="/l/?uddg=https%3A%2F%2Fwww.irctc.co.in%2Fnget%2Ftrain-search">IRCTC Official</a>
            <a class="result__snippet">Official Indian Railway catering and tourism portal.</a>
        </div>
        """
        mock_resp = DummySearchResponse(html_uddg, status=200)

        with patch("socket.getaddrinfo", return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("52.142.124.215", 443))]), \
             patch("urllib.request.urlopen", return_value=mock_resp):
            result = self.tool.run(query="irctc")
            self.assertTrue(result.success)
            self.assertEqual(result.data["results"][0]["url"], "https://www.irctc.co.in/nget/train-search")

    def test_no_cookies_or_auth_sent(self):
        """18. Verify no cookies or credentials sent in request."""
        mock_resp = DummySearchResponse(SAMPLE_DDG_HTML, status=200)

        with patch("socket.getaddrinfo", return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("52.142.124.215", 443))]), \
             patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            self.tool.run(query="flight search")
            called_req = mock_urlopen.call_args[0][0]
            self.assertNotIn("Cookie", called_req.headers)
            self.assertNotIn("Authorization", called_req.headers)
            self.assertIn("ExoCortex-Agent", called_req.headers.get("User-agent"))


if __name__ == "__main__":
    unittest.main()
