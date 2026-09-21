"""
Unit tests for structured response parsing, JSON extraction, and schema validation.
"""

import unittest
from exocortex.schema import AgentDecision, StructuredStep
from exocortex.slm.parser import (
    extract_json_from_text,
    repair_malformed_json,
    parse_and_validate_decision,
)
from exocortex.tools.base import BaseTool, PermissionLevel, ToolParameter, ToolResult
from exocortex.tools.registry import ToolRegistry, get_default_tool_registry


class TestStructuredParser(unittest.TestCase):
    def setUp(self):
        self.registry = get_default_tool_registry()

    def test_extract_json_from_clean_string(self):
        raw = '{"thought_summary": "checking specs", "intent": "hardware_inspection", "steps": [], "requires_confirmation": false}'
        data = extract_json_from_text(raw)
        self.assertIsNotNone(data)
        self.assertEqual(data["intent"], "hardware_inspection")

    def test_extract_json_from_markdown_codeblock(self):
        raw = """Here is the plan:
```json
{
  "thought_summary": "Run system info tool",
  "intent": "hardware_inspection",
  "steps": [
    {
      "tool": "system_info",
      "arguments": {"detail_level": "basic"}
    }
  ],
  "requires_confirmation": false
}
```
Done."""
        data = extract_json_from_text(raw)
        self.assertIsNotNone(data)
        self.assertEqual(len(data["steps"]), 1)
        self.assertEqual(data["steps"][0]["tool"], "system_info")

    def test_repair_malformed_json_trailing_comma(self):
        raw = '{"thought_summary": "test", "intent": "test", "steps": [], "requires_confirmation": false,}'
        repaired = repair_malformed_json(raw)
        self.assertIsNotNone(repaired)
        self.assertEqual(repaired["thought_summary"], "test")

    def test_repair_missing_closing_brace(self):
        raw = '{"thought_summary": "test", "intent": "test", "steps": [], "requires_confirmation": false'
        repaired = repair_malformed_json(raw)
        self.assertIsNotNone(repaired)
        self.assertEqual(repaired["intent"], "test")

    def test_parse_and_validate_valid_decision(self):
        raw = """{
  "thought_summary": "Check system info.",
  "intent": "hardware_inspection",
  "steps": [
    {"tool": "system_info", "arguments": {"detail_level": "basic"}}
  ],
  "requires_confirmation": false
}"""
        decision, err = parse_and_validate_decision(raw, self.registry)
        self.assertIsNone(err)
        self.assertIsNotNone(decision)
        self.assertEqual(decision.intent, "hardware_inspection")
        self.assertEqual(len(decision.steps), 1)

    def test_reject_unknown_tool(self):
        raw = """{
  "thought_summary": "Attempting unapproved action.",
  "intent": "hack_system",
  "steps": [
    {"tool": "non_existent_dangerous_tool", "arguments": {}}
  ],
  "requires_confirmation": false
}"""
        decision, err = parse_and_validate_decision(raw, self.registry)
        self.assertIsNone(decision)
        self.assertIn("Unknown tool requested", err)

    def test_reject_invalid_arguments(self):
        raw = """{
  "thought_summary": "Calling echo without required parameter.",
  "intent": "tool_execution",
  "steps": [
    {"tool": "echo", "arguments": {}}
  ],
  "requires_confirmation": false
}"""
        decision, err = parse_and_validate_decision(raw, self.registry)
        self.assertIsNone(decision)
        self.assertIn("Missing required parameter: 'message'", err)

    def test_auto_elevate_confirmation_for_sensitive_tool(self):
        class ModifyingTool(BaseTool):
            name = "modify_setting"
            description = "Modifies a setting"
            permission_level = PermissionLevel.CONFIRMATION_REQUIRED
            parameters = [ToolParameter(name="key", type="string", description="Key to modify", required=True)]
            def run(self, **kwargs):
                return ToolResult(success=True, output="modified")

        custom_reg = ToolRegistry()
        custom_reg.register(ModifyingTool())

        raw = """{
  "thought_summary": "Modify a setting.",
  "intent": "system_modification",
  "steps": [
    {"tool": "modify_setting", "arguments": {"key": "theme"}}
  ],
  "requires_confirmation": false
}"""
        decision, err = parse_and_validate_decision(raw, custom_reg)
        self.assertIsNone(err)
        self.assertIsNotNone(decision)
        self.assertTrue(decision.requires_confirmation)


if __name__ == "__main__":
    unittest.main()
