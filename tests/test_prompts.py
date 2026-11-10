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

    def test_build_user_prompt_subgoal_after_web_search(self):
        history = [
            {
                "step": 1,
                "tool": "web_search",
                "arguments": {"query": "Pune Wikipedia"},
                "success": True,
                "observation": "1. [en.wikipedia.org/wiki/Pune] - https://en.wikipedia.org/wiki/Pune",
            }
        ]
        up = build_user_prompt("Find info about Pune", execution_history=history)
        self.assertIn("### CURRENT SUB-GOAL / NEXT ACTION (CHOOSE FROM RETURNED DATA):", up)
        self.assertIn("Search is completed. Select one of the returned URLs from the search results above", up)
        self.assertIn("fetch_web_page", up)

    def test_build_user_prompt_subgoal_after_fetch_web_page(self):
        history = [
            {
                "step": 1,
                "tool": "web_search",
                "arguments": {"query": "Pune Wikipedia"},
                "success": True,
                "observation": "1. [en.wikipedia.org/wiki/Pune] - https://en.wikipedia.org/wiki/Pune",
            },
            {
                "step": 2,
                "tool": "fetch_web_page",
                "arguments": {"url": "https://en.wikipedia.org/wiki/Pune"},
                "success": True,
                "observation": "Pune is a city in Maharashtra, India.",
            },
        ]
        up = build_user_prompt("Find info about Pune", execution_history=history)
        self.assertIn("### CURRENT SUB-GOAL / NEXT ACTION (CHOOSE FROM RETURNED DATA):", up)
        self.assertIn("Webpage content has been fetched. Do NOT call web_search or repeat previous searches.", up)
        self.assertIn("Set 'steps': [] and provide the final summary in 'direct_response'.", up)


if __name__ == "__main__":
    unittest.main()

