"""
Unit tests for configuration management.
"""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

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
        self.assertEqual(settings.local_server_timeout, 90.0)

    def test_local_server_timeout_parsing_valid(self):
        from exocortex.config import _parse_local_server_timeout
        self.assertEqual(_parse_local_server_timeout("90"), 90.0)
        self.assertEqual(_parse_local_server_timeout("120.5"), 120.5)

    def test_local_server_timeout_parsing_invalid(self):
        from exocortex.config import _parse_local_server_timeout
        invalid_values = ["0", "-1", "-90.0", "abc", "", "nan"]
        for val in invalid_values:
            with self.subTest(val=val):
                with self.assertRaises(ValueError) as ctx:
                    _parse_local_server_timeout(val)
                self.assertIn("must be a positive number of seconds", str(ctx.exception))

    def test_settings_fails_on_invalid_env_timeout(self):
        with unittest.mock.patch.dict(os.environ, {"EXOCORTEX_LOCAL_SERVER_TIMEOUT": "-5"}):
            with self.assertRaises(ValueError):
                Settings()

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
