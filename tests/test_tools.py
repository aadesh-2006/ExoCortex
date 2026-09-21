"""
Unit tests for tool registry, permissions, and built-in tools.
"""

import unittest
from exocortex.tools.base import BaseTool, PermissionLevel, ToolParameter, ToolResult
from exocortex.tools.builtin import EchoTool, HealthCheckTool, SystemInfoTool
from exocortex.tools.registry import ToolRegistry, get_default_tool_registry


class TestTools(unittest.TestCase):
    def test_echo_tool(self):
        tool = EchoTool()
        self.assertEqual(tool.name, "echo")
        res = tool.run(message="hello world")
        self.assertTrue(res.success)
        self.assertIn("hello world", res.output)

    def test_system_info_tool(self):
        tool = SystemInfoTool()
        self.assertEqual(tool.name, "system_info")
        res = tool.run()
        self.assertTrue(res.success)
        self.assertIn("OS:", res.output)

    def test_tool_registry(self):
        reg = ToolRegistry()
        reg.register(EchoTool())
        
        tool = reg.get_tool("echo")
        self.assertIsNotNone(tool)
        
        schemas = reg.get_schemas()
        self.assertEqual(len(schemas), 1)
        self.assertEqual(schemas[0]["name"], "echo")

        # Execute
        res = reg.execute("echo", {"message": "unit test"})
        self.assertTrue(res.success)
        self.assertIn("unit test", res.output)

    def test_permission_denial(self):
        class SensitiveTool(BaseTool):
            name = "sensitive_action"
            description = "Test sensitive tool"
            permission_level = PermissionLevel.CONFIRMATION_REQUIRED
            def run(self, **kwargs):
                return ToolResult(success=True, output="done")

        # Rejection callback
        reg = ToolRegistry(confirmation_callback=lambda name, args: False)
        reg.register(SensitiveTool())
        res = reg.execute("sensitive_action", {})
        self.assertFalse(res.success)
        self.assertIn("Permission denied", res.error)


if __name__ == "__main__":
    unittest.main()
