"""
Unit tests for tool-aware prompt building.
"""

import unittest
from exocortex.slm.prompts import build_system_prompt, build_user_prompt
from exocortex.tools.registry import get_default_tool_registry


class TestPrompts(unittest.TestCase):
    def test_build_system_prompt_contains_tools(self):
        reg = get_default_tool_registry()
        tools = reg.list_tools()
        prompt = build_system_prompt(tools)
        
        self.assertIn("system_info", prompt)
        self.assertIn("health_check", prompt)
        self.assertIn("echo", prompt)
        self.assertIn("OUTPUT JSON SCHEMA:", prompt)
        self.assertIn("thought_summary", prompt)

    def test_build_user_prompt(self):
        up = build_user_prompt("Check my CPU and memory")
        self.assertIn("User Request: Check my CPU and memory", up)


if __name__ == "__main__":
    unittest.main()
