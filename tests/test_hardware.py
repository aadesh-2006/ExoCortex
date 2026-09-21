"""
Unit tests for hardware and Snapdragon NPU detection.
"""

import unittest
from exocortex.hardware import detect_hardware, detect_execution_providers, HardwareProfile


class TestHardwareDetection(unittest.TestCase):
    def test_detect_hardware_returns_profile(self):
        hw = detect_hardware()
        self.assertIsInstance(hw, HardwareProfile)
        self.assertTrue(len(hw.os_name) > 0)
        self.assertTrue(len(hw.architecture) > 0)
        self.assertIsInstance(hw.is_arm64, bool)
        self.assertIsInstance(hw.is_snapdragon_detected, bool)
        self.assertIsInstance(hw.is_npu_available, bool)
        self.assertIsInstance(hw.available_execution_providers, list)
        self.assertTrue(len(hw.available_execution_providers) > 0)

    def test_hardware_to_dict(self):
        hw = detect_hardware()
        d = hw.to_dict()
        self.assertIn("os_name", d)
        self.assertIn("architecture", d)
        self.assertIn("recommended_execution_provider", d)
        self.assertIn("total_memory_gb", d)


if __name__ == "__main__":
    unittest.main()
