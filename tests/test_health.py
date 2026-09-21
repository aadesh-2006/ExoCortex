"""
Unit tests for health diagnostics probe.
"""

import unittest
from exocortex.health import run_health_check, HealthReport


class TestHealth(unittest.TestCase):
    def test_run_health_check(self):
        report = run_health_check()
        self.assertIsInstance(report, HealthReport)
        self.assertIn(report.status, ("healthy", "degraded"))
        self.assertTrue(len(report.version) > 0)
        self.assertIsInstance(report.hardware, dict)
        self.assertIsInstance(report.slm_provider, dict)
        self.assertIsInstance(report.registered_tools, list)
        self.assertTrue(len(report.registered_tools) >= 3)
        self.assertIn("system_info", report.registered_tools)
        self.assertIn("health_check", report.registered_tools)
        self.assertIn("echo", report.registered_tools)

    def test_health_report_to_dict(self):
        report = run_health_check()
        d = report.to_dict()
        self.assertIn("status", d)
        self.assertIn("checks", d)
        self.assertIn("hardware", d)


if __name__ == "__main__":
    unittest.main()
