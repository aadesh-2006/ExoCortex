"""
Unit tests for configuration management.
"""

import os
import unittest
from pathlib import Path

from exocortex.config import Settings, get_config


class TestConfig(unittest.TestCase):
    def test_default_settings(self):
        settings = Settings()
        self.assertEqual(settings.env, "development")
        self.assertEqual(settings.slm_provider, "local_slm")
        self.assertEqual(settings.model_name, "qwen2.5-0.5b-instruct")
        self.assertEqual(settings.hardware_target, "auto")
        self.assertTrue(settings.require_confirmation_for_sensitive)
        self.assertEqual(settings.max_plan_steps, 10)

    def test_ensure_data_dir(self):
        settings = Settings(data_dir=Path("./tmp_test_data_dir"))
        d = settings.ensure_data_dir()
        self.assertTrue(d.exists())
        # Clean up
        d.rmdir()

    def test_get_config_singleton(self):
        cfg1 = get_config(reload=True)
        cfg2 = get_config()
        self.assertIs(cfg1, cfg2)


if __name__ == "__main__":
    unittest.main()
