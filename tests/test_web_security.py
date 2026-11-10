"""
Unit and security tests for Milestone 6 Web Security and URL Validation primitives.
"""

import ipaddress
import socket
import unittest
from unittest.mock import patch

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


class TestWebSecurity(unittest.TestCase):
    """Test web security layer, SSRF defense, and URL validation."""

    def test_https_url_accepted(self):
        """1. https URL accepted."""
        with patch("socket.getaddrinfo", return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]):
            url, port, ip = validate_url_security("https://example.com/search?q=flight")
            self.assertEqual(url, "https://example.com/search?q=flight")
            self.assertEqual(port, 443)
            self.assertEqual(ip, "93.184.216.34")

    def test_http_url_accepted(self):
        """2. http URL accepted."""
        with patch("socket.getaddrinfo", return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80))]):
            url, port, ip = validate_url_security("http://example.com")
            self.assertEqual(url, "http://example.com")
            self.assertEqual(port, 80)
            self.assertEqual(ip, "93.184.216.34")

    def test_file_scheme_rejected(self):
        """3. file:// rejected."""
        with self.assertRaises(URLSecurityError) as ctx:
            validate_url_security("file:///C:/Windows/System32/drivers/etc/hosts")
        self.assertIn("Unsupported URL scheme 'file'", str(ctx.exception))

    def test_ftp_scheme_rejected(self):
        """4. ftp:// rejected."""
        with self.assertRaises(URLSecurityError) as ctx:
            validate_url_security("ftp://ftp.example.com/file.txt")
        self.assertIn("Unsupported URL scheme 'ftp'", str(ctx.exception))

    def test_javascript_scheme_rejected(self):
        """5. javascript: rejected."""
        with self.assertRaises(URLSecurityError) as ctx:
            validate_url_security("javascript:alert(1)")
        self.assertIn("Unsupported URL scheme 'javascript'", str(ctx.exception))

    def test_data_and_ws_schemes_rejected(self):
        """Reject data: and ws: schemes."""
        with self.assertRaises(URLSecurityError):
            validate_url_security("data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==")
        with self.assertRaises(URLSecurityError):
            validate_url_security("ws://example.com/socket")

    def test_localhost_rejected(self):
        """6. localhost rejected."""
        with self.assertRaises(URLSecurityError) as ctx:
            validate_url_security("http://localhost:80/admin")
        self.assertIn("localhost is prohibited", str(ctx.exception))

    def test_loopback_127_0_0_1_rejected(self):
        """7. 127.0.0.1 rejected."""
        with self.assertRaises(URLSecurityError) as ctx:
            validate_url_security("http://127.0.0.1/status")
        self.assertIn("private/restricted IP address", str(ctx.exception))

    def test_rfc1918_192_168_rejected(self):
        """8. 192.168.1.1 rejected."""
        with self.assertRaises(URLSecurityError) as ctx:
            validate_url_security("http://192.168.1.1/router")
        self.assertIn("private/restricted IP address", str(ctx.exception))

    def test_rfc1918_10_0_0_1_rejected(self):
        """9. 10.0.0.1 rejected."""
        with self.assertRaises(URLSecurityError) as ctx:
            validate_url_security("http://10.0.0.1/internal")
        self.assertIn("private/restricted IP address", str(ctx.exception))

    def test_rfc1918_172_16_rejected(self):
        """10. 172.16.x.x private address rejected."""
        with self.assertRaises(URLSecurityError) as ctx:
            validate_url_security("https://172.20.0.5/api")
        self.assertIn("private/restricted IP address", str(ctx.exception))

    def test_link_local_169_254_rejected(self):
        """11. 169.254.169.254 (Cloud metadata service) rejected."""
        with self.assertRaises(URLSecurityError) as ctx:
            validate_url_security("http://169.254.169.254/latest/meta-data/")
        self.assertIn("private/restricted IP address", str(ctx.exception))

    def test_non_standard_port_rejected(self):
        """12. non-standard port rejected."""
        with self.assertRaises(URLSecurityError) as ctx:
            validate_url_security("https://example.com:8080/dashboard")
        self.assertIn("Prohibited network port '8080'", str(ctx.exception))

        with self.assertRaises(URLSecurityError):
            validate_url_security("http://example.com:22/ssh")

    def test_port_80_accepted(self):
        """13. port 80 accepted."""
        with patch("socket.getaddrinfo", return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80))]):
            url, port, _ = validate_url_security("http://example.com:80/path")
            self.assertEqual(port, 80)

    def test_port_443_accepted(self):
        """14. port 443 accepted."""
        with patch("socket.getaddrinfo", return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]):
            url, port, _ = validate_url_security("https://example.com:443/secure")
            self.assertEqual(port, 443)

    def test_redirect_to_private_ip_rejected(self):
        """15. redirect to private IP rejected."""
        with self.assertRaises(URLSecurityError) as ctx:
            validate_redirect_target("https://public-site.com/login", "http://192.168.1.1/admin")
        self.assertIn("private/restricted IP address", str(ctx.exception))

    def test_safe_relative_redirect_accepted(self):
        """Validate relative redirect on public site is safe."""
        with patch("socket.getaddrinfo", return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]):
            target = validate_redirect_target("https://example.com/flights", "/search?date=2026-11-10")
            self.assertEqual(target, "https://example.com/search?date=2026-11-10")

    def test_dns_rebinding_resolution_to_private_ip_blocked(self):
        """Verify that a public-looking domain that resolves to 127.0.0.1 is blocked."""
        with patch("socket.getaddrinfo", return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))]):
            with self.assertRaises(URLSecurityError) as ctx:
                validate_url_security("https://evil-rebind.com")
            self.assertIn("resolved to prohibited IP address '127.0.0.1'", str(ctx.exception))

    def test_blocked_internal_tlds(self):
        """Verify internal/local domain names are rejected."""
        with self.assertRaises(URLSecurityError):
            validate_hostname_syntax("my-server.local")
        with self.assertRaises(URLSecurityError):
            validate_hostname_syntax("database.internal")
        with self.assertRaises(URLSecurityError):
            validate_hostname_syntax("router.lan")
        with self.assertRaises(URLSecurityError):
            validate_hostname_syntax("service.corp")

    def test_cgnat_and_multicast_ip_blocking(self):
        """Verify CGNAT and multicast IP address blocking."""
        cgnat_ip = ipaddress.ip_address("100.64.0.1")
        self.assertTrue(is_ip_blocked(cgnat_ip))

        multicast_ip = ipaddress.ip_address("224.0.0.1")
        self.assertTrue(is_ip_blocked(multicast_ip))

        public_ip = ipaddress.ip_address("8.8.8.8")
        self.assertFalse(is_ip_blocked(public_ip))


if __name__ == "__main__":
    unittest.main()
