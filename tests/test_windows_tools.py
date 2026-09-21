"""
Comprehensive unit tests for ExoCortex safe Windows OS & desktop automation tools.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from exocortex.agent import ExoCortexAgent, AgentRunResult
from exocortex.config import Settings, get_config
from exocortex.slm.mock_provider import MockSLMProvider
from exocortex.tools.base import PermissionLevel
from exocortex.tools.registry import get_default_tool_registry
from exocortex.tools.windows import (
    ALLOWLISTED_APPLICATIONS,
    CreateDirectoryTool,
    LaunchApplicationTool,
    ListDirectoryTool,
    ListProcessesTool,
    OpenUrlTool,
    ReadTextFileTool,
    resolve_safe_workspace_path,
)


class TestLaunchApplicationTool(unittest.TestCase):
    def setUp(self):
        self.tool = LaunchApplicationTool()

    @patch("subprocess.Popen")
    def test_allowed_application_launches(self, mock_popen):
        mock_popen.return_value = MagicMock()

        # Test notepad
        res = self.tool.run(application="notepad")
        self.assertTrue(res.success)
        self.assertIn("notepad", res.output.lower())
        mock_popen.assert_called_with(["notepad.exe"], shell=False)

        # Test calculator
        res = self.tool.run(application="calculator")
        self.assertTrue(res.success)
        mock_popen.assert_called_with(["calc.exe"], shell=False)

        # Test paint
        res = self.tool.run(application="paint")
        self.assertTrue(res.success)
        mock_popen.assert_called_with(["mspaint.exe"], shell=False)

        # Test explorer
        res = self.tool.run(application="explorer")
        self.assertTrue(res.success)
        mock_popen.assert_called_with(["explorer.exe"], shell=False)

    def test_unknown_application_rejected(self):
        res = self.tool.run(application="unknown_app_123")
        self.assertFalse(res.success)
        self.assertIn("not in the safe allowlist", res.error)

    def test_arbitrary_executable_path_rejected(self):
        res = self.tool.run(application="C:\\Windows\\System32\\cmd.exe")
        self.assertFalse(res.success)
        self.assertIn("not in the safe allowlist", res.error)

    def test_empty_application_rejected(self):
        res = self.tool.run(application="")
        self.assertFalse(res.success)
        self.assertIn("empty", res.error)

    def test_permission_is_safe(self):
        self.assertEqual(self.tool.permission_level, PermissionLevel.SAFE)


class TestOpenUrlTool(unittest.TestCase):
    def setUp(self):
        self.tool = OpenUrlTool()

    @patch("webbrowser.open")
    def test_valid_https_url_accepted(self, mock_browser):
        mock_browser.return_value = True
        res = self.tool.run(url="https://www.google.com")
        self.assertTrue(res.success)
        self.assertIn("https://www.google.com", res.output)
        mock_browser.assert_called_once_with("https://www.google.com")

    @patch("webbrowser.open")
    def test_valid_http_url_accepted(self, mock_browser):
        mock_browser.return_value = True
        res = self.tool.run(url="http://example.com/test?q=1")
        self.assertTrue(res.success)
        self.assertIn("http://example.com", res.output)
        mock_browser.assert_called_once_with("http://example.com/test?q=1")

    def test_javascript_scheme_rejected(self):
        res = self.tool.run(url="javascript:alert(1)")
        self.assertFalse(res.success)
        self.assertIn("Forbidden URL scheme", res.error)

    def test_file_scheme_rejected(self):
        res = self.tool.run(url="file:///C:/Windows/System32/calc.exe")
        self.assertFalse(res.success)
        self.assertIn("Forbidden URL scheme", res.error)

    def test_data_scheme_rejected(self):
        res = self.tool.run(url="data:text/html,<h1>Hello</h1>")
        self.assertFalse(res.success)
        self.assertIn("Forbidden URL scheme", res.error)

    def test_malformed_url_rejected(self):
        res = self.tool.run(url="https://")
        self.assertFalse(res.success)
        self.assertIn("Invalid URL format", res.error)

    def test_permission_is_safe(self):
        self.assertEqual(self.tool.permission_level, PermissionLevel.SAFE)


class TestWorkspaceFilesystemTools(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="exocortex_test_ws_")
        self.config = get_config()
        self.original_workspace = self.config.workspace_dir
        self.config.workspace_dir = Path(self.temp_dir)

        self.list_tool = ListDirectoryTool()
        self.read_tool = ReadTextFileTool()
        self.create_tool = CreateDirectoryTool()

    def tearDown(self):
        self.config.workspace_dir = self.original_workspace
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_workspace_creation(self):
        ws_path = self.config.ensure_workspace_dir()
        self.assertTrue(ws_path.exists())
        self.assertTrue(ws_path.is_dir())

    def test_create_directory_inside_workspace(self):
        res = self.create_tool.run(path="subfolder/nested")
        self.assertTrue(res.success)
        created_path = Path(self.temp_dir) / "subfolder" / "nested"
        self.assertTrue(created_path.exists())
        self.assertTrue(created_path.is_dir())

    def test_create_directory_traversal_rejected(self):
        res = self.create_tool.run(path="../../outside_test")
        self.assertFalse(res.success)
        self.assertIn("Security violation", res.error)

    def test_list_directory(self):
        # Create sample files
        (Path(self.temp_dir) / "file1.txt").write_text("Hello 1", encoding="utf-8")
        (Path(self.temp_dir) / "folder1").mkdir()

        res = self.list_tool.run(path="")
        self.assertTrue(res.success)
        self.assertIn("file1.txt", res.output)
        self.assertIn("folder1", res.output)
        self.assertEqual(res.data["count"], 2)

    def test_read_text_file(self):
        sample_file = Path(self.temp_dir) / "notes.txt"
        sample_file.write_text("ExoCortex DBMS Notes", encoding="utf-8")

        res = self.read_tool.run(path="notes.txt")
        self.assertTrue(res.success)
        self.assertEqual(res.output, "ExoCortex DBMS Notes")

    def test_read_nonexistent_file(self):
        res = self.read_tool.run(path="nonexistent.txt")
        self.assertFalse(res.success)
        self.assertIn("File not found", res.error)

    def test_path_traversal_rejected(self):
        res = self.read_tool.run(path="../../../Windows/win.ini")
        self.assertFalse(res.success)
        self.assertIn("Security violation", res.error)

    def test_absolute_path_outside_workspace_rejected(self):
        res = self.read_tool.run(path="C:\\Windows\\System32\\drivers\\etc\\hosts")
        self.assertFalse(res.success)
        self.assertIn("Security violation", res.error)


class TestListProcessesTool(unittest.TestCase):
    def setUp(self):
        self.tool = ListProcessesTool()

    def test_list_processes_structure(self):
        res = self.tool.run(limit=5)
        self.assertTrue(res.success)
        self.assertIn("Running Processes", res.output)
        self.assertIsInstance(res.data["processes"], list)
        self.assertTrue(len(res.data["processes"]) > 0)
        proc = res.data["processes"][0]
        self.assertIn("pid", proc)
        self.assertIn("name", proc)
        self.assertIn("memory_mb", proc)

    def test_list_processes_sort_by_name(self):
        res = self.tool.run(limit=5, sort_by="name")
        self.assertTrue(res.success)
        self.assertEqual(res.data["sort_by"], "name")

    def test_permission_is_safe(self):
        self.assertEqual(self.tool.permission_level, PermissionLevel.SAFE)


class TestDefaultRegistryMilestone3(unittest.TestCase):
    def test_all_milestone3_tools_registered(self):
        reg = get_default_tool_registry()
        tool_names = [t.name for t in reg.list_tools()]

        expected_tools = [
            "echo",
            "system_info",
            "health_check",
            "launch_application",
            "open_url",
            "list_directory",
            "read_text_file",
            "create_directory",
            "list_processes",
        ]

        for expected in expected_tools:
            self.assertIn(expected, tool_names, f"Tool '{expected}' missing from default registry")

    def test_schemas_generated_properly(self):
        reg = get_default_tool_registry()
        schemas = reg.get_schemas()
        schema_names = [s["name"] for s in schemas]
        self.assertIn("launch_application", schema_names)
        self.assertIn("open_url", schema_names)
        self.assertIn("list_directory", schema_names)


class TestAgentDesktopIntegration(unittest.TestCase):
    def setUp(self):
        self.mock_provider = MockSLMProvider()
        self.agent = ExoCortexAgent(slm_provider=self.mock_provider)

    @patch("subprocess.Popen")
    def test_agent_open_notepad(self, mock_popen):
        mock_popen.return_value = MagicMock()
        res = self.agent.run("Open Notepad")
        self.assertTrue(res.success)
        self.assertEqual(res.decision.intent, "application_launch")
        self.assertEqual(res.decision.steps[0].tool, "launch_application")
        self.assertEqual(res.decision.steps[0].arguments["application"], "notepad")

    @patch("webbrowser.open")
    def test_agent_open_google(self, mock_browser):
        mock_browser.return_value = True
        res = self.agent.run("Open Google in my browser")
        self.assertTrue(res.success)
        self.assertEqual(res.decision.intent, "web_navigation")
        self.assertEqual(res.decision.steps[0].tool, "open_url")

    def test_agent_list_processes(self):
        res = self.agent.run("Show me my running processes")
        self.assertTrue(res.success)
        self.assertEqual(res.decision.intent, "process_inspection")
        self.assertEqual(res.decision.steps[0].tool, "list_processes")

    def test_agent_list_workspace(self):
        res = self.agent.run("List the files in my ExoCortex workspace")
        self.assertTrue(res.success)
        self.assertEqual(res.decision.intent, "filesystem_inspection")
        self.assertEqual(res.decision.steps[0].tool, "list_directory")


if __name__ == "__main__":
    unittest.main()
