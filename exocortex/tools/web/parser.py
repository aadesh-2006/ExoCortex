"""
Safe HTML parser and readable-text extractor for ExoCortex web research tools.

Extracts semantic text from HTML documents while stripping scripts, styling, navigation,
headers, footers, and hidden elements without executing JavaScript.
"""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from typing import List, Optional, Set

from exocortex.tools.web.security import MAX_EXTRACTED_CHARS

# HTML tags whose entire subtree content should be completely omitted
IGNORED_TAGS: Set[str] = {
    "script",
    "style",
    "noscript",
    "header",
    "footer",
    "nav",
    "aside",
    "svg",
    "canvas",
    "iframe",
    "object",
    "embed",
    "template",
}

# Block tags that warrant spacing/newlines
BLOCK_TAGS: Set[str] = {
    "p",
    "div",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "tr",
    "br",
    "hr",
    "article",
    "section",
    "blockquote",
    "table",
    "ul",
    "ol",
    "main",
}


class SafeHTMLTextExtractor(HTMLParser):
    """
    Standard-library HTMLParser that safely extracts readable human text
    from HTML without executing scripts or evaluating active content.
    """

    def __init__(self, max_chars: int = MAX_EXTRACTED_CHARS) -> None:
        super().__init__(convert_charrefs=True)
        self.max_chars = max_chars
        self._ignore_depth = 0
        self._text_chunks: List[str] = []
        self._current_length = 0
        self._stopped = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if self._stopped:
            return

        tag_lower = tag.lower()

        # Check for ignored tags
        if tag_lower in IGNORED_TAGS:
            self._ignore_depth += 1
            return

        # Check for hidden elements in attributes
        attrs_dict = {k.lower(): (v or "").lower() for k, v in attrs}
        if "hidden" in attrs_dict or attrs_dict.get("aria-hidden") == "true":
            self._ignore_depth += 1
            return
        style = attrs_dict.get("style", "")
        if "display:none" in style.replace(" ", "") or "visibility:hidden" in style.replace(" ", ""):
            self._ignore_depth += 1
            return

        # Insert boundary spacing for block elements
        if self._ignore_depth == 0 and tag_lower in BLOCK_TAGS:
            self._text_chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self._stopped:
            return

        tag_lower = tag.lower()
        if tag_lower in IGNORED_TAGS:
            if self._ignore_depth > 0:
                self._ignore_depth -= 1
            return

        if self._ignore_depth == 0 and tag_lower in BLOCK_TAGS:
            self._text_chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._stopped or self._ignore_depth > 0:
            return

        if not data:
            return

        self._text_chunks.append(data)
        self._current_length += len(data)

        # Early termination if extracted buffer exceeds 2x max_chars
        if self._current_length > self.max_chars * 2:
            self._stopped = True

    def error(self, message: str) -> None:
        # Ignore parser errors to handle malformed HTML resiliently
        pass

    def get_extracted_text(self) -> str:
        """Combine, normalize whitespace, and hard-truncate extracted text."""
        raw_text = "".join(self._text_chunks)
        
        # Unescape any lingering HTML entities
        unescaped = html.unescape(raw_text)

        # Normalize line breaks and spaces
        # 1. Replace tabs and non-breaking spaces
        cleaned = unescaped.replace("\t", " ").replace("\xa0", " ").replace("\r", "\n")
        
        # 2. Collapse multiple spaces within lines
        cleaned = re.sub(r"[ ]{2,}", " ", cleaned)
        
        # 3. Collapse 3+ consecutive newlines into double newlines
        cleaned = re.sub(r"\n\s*\n\s*\n+", "\n\n", cleaned)
        
        # 4. Strip whitespace from lines and trim
        lines = [line.strip() for line in cleaned.split("\n")]
        result = "\n".join(line for line in lines if line)
        result = result.strip()

        # Hard limit enforcement
        if len(result) > self.max_chars:
            result = result[: self.max_chars].rstrip() + "..."

        return result


def extract_readable_text(html_content: str, max_chars: int = MAX_EXTRACTED_CHARS) -> str:
    """
    Safely extract readable plain text from HTML content.

    Args:
        html_content: Raw HTML string.
        max_chars: Maximum characters to return (defaults to MAX_EXTRACTED_CHARS).

    Returns:
        Clean, normalized plain text representation of the HTML document.
    """
    if not html_content or not isinstance(html_content, str):
        return ""

    extractor = SafeHTMLTextExtractor(max_chars=max_chars)
    try:
        extractor.feed(html_content)
        extractor.close()
    except Exception:
        # Resilient fallback: regex tag strip if parser fails on catastrophic malformed input
        clean = re.sub(r"<script[\s\S]*?</script>", "", html_content, flags=re.IGNORECASE)
        clean = re.sub(r"<style[\s\S]*?</style>", "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"<[^>]+>", " ", clean)
        clean = html.unescape(clean)
        clean = re.sub(r"\s+", " ", clean).strip()
        if len(clean) > max_chars:
            clean = clean[:max_chars].rstrip() + "..."
        return clean

    return extractor.get_extracted_text()
