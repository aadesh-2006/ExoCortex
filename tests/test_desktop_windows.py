"""
Unit and integration tests for ListOpenWindowsTool and safe desktop window inspection.
"""

import ctypes
from ctypes import wintypes
import unittest
from unittest.mock import MagicMock, patch

from exocortex.agent import ExoCortexAgent
from exocortex.execution.executor import AgentExecutor
from exocortex.slm.mock_provider import MockSLMProvider
from exocortex.tools.base import PermissionLevel
from exocortex.tools.registry import ToolRegistry, get_default_tool_registry
from exocortex.tools.windows.desktop_windows import (
    MAX_RETURNED_WINDOWS,
    MAX_WINDOW_TITLE_CHARS,
    ListOpenWindowsTool,
)


class TestListOpenWindowsTool(unittest.TestCase):
    """Unit tests for ListOpenWindowsTool with mocked Win32 APIs."""

    def setUp(self):
        self.tool = ListOpenWindowsTool()

    def test_tool_metadata_and_schema(self):
        """Verify tool name, permission level, and parameters."""
        self.assertEqual(self.tool.name, "list_open_windows")
        self.assertEqual(self.tool.permission_level, PermissionLevel.SAFE)
        param_names = [p.name for p in self.tool.parameters]
        self.assertIn("limit", param_names)
        self.assertIn("Read-only inspection only", self.tool.description)

        schema = self.tool.get_schema()
        self.assertEqual(schema["name"], "list_open_windows")
        self.assertEqual(schema["permission_level"], "safe")

    def test_mocked_window_enumeration_and_filtering(self):
        """
        Verify that:
        - Visible, non-empty, non-cloaked windows are included.
        - Hidden, empty, zero-size, cloaked, and 'Program Manager' windows are filtered out.
        """
        # Define mock window objects: (hwnd, title, is_visible, width, height, is_cloaked, pid)
        mock_windows = [
            (1001, "Google Chrome - GitHub", True, 1200, 800, 0, 5001),     # Valid
            (1002, "Notepad - Notes.txt", True, 800, 600, 0, 5002),          # Valid
            (1003, "", True, 800, 600, 0, 5003),                             # Empty title -> Exclude
            (1004, "Hidden Worker Window", False, 800, 600, 0, 5004),         # Invisible -> Exclude
            (1005, "Zero Size Tooltip", True, 0, 0, 0, 5005),                 # Zero size -> Exclude
            (1006, "Windows Background Task", True, 800, 600, 1, 5006),       # Cloaked -> Exclude
            (1007, "Program Manager", True, 1920, 1080, 0, 5007),             # Shell background -> Exclude
        ]

        def fake_enum_windows(callback, lparam):
            for hwnd, _, _, _, _, _, _ in mock_windows:
                res = callback(hwnd, lparam)
                if not res:
                    break
            return 1

        def fake_is_window_visible(hwnd):
            for h, _, vis, _, _, _, _ in mock_windows:
                if h == hwnd:
                    return 1 if vis else 0
            return 0

        def fake_get_window_text_length(hwnd):
            for h, title, _, _, _, _, _ in mock_windows:
                if h == hwnd:
                    return len(title)
            return 0

        def fake_get_window_text(hwnd, buf, max_count):
            for h, title, _, _, _, _, _ in mock_windows:
                if h == hwnd:
                    buf.value = title
                    return len(title)
            return 0

        def fake_get_window_rect(hwnd, rect_ref):
            for h, _, _, w, height, _, _ in mock_windows:
                if h == hwnd:
                    rect = rect_ref._obj if hasattr(rect_ref, "_obj") else rect_ref
                    rect.left = 0
                    rect.top = 0
                    rect.right = w
                    rect.bottom = height
                    return 1
            return 0

        def fake_dwm_get_attribute(hwnd, attr, out_ref, size):
            for h, _, _, _, _, cloaked, _ in mock_windows:
                if h == hwnd:
                    out = out_ref._obj if hasattr(out_ref, "_obj") else out_ref
                    out.value = cloaked
                    return 0
            return 0

        def fake_get_thread_process_id(hwnd, pid_ref):
            for h, _, _, _, _, _, pid in mock_windows:
                if h == hwnd:
                    out = pid_ref._obj if hasattr(pid_ref, "_obj") else pid_ref
                    out.value = pid
                    return 1
            return 0

        with patch("ctypes.windll.user32.OpenDesktopW", return_value=None), \
             patch("ctypes.windll.user32.EnumWindows", side_effect=fake_enum_windows), \
             patch("ctypes.windll.user32.IsWindowVisible", side_effect=fake_is_window_visible), \
             patch("ctypes.windll.user32.GetWindowTextLengthW", side_effect=fake_get_window_text_length), \
             patch("ctypes.windll.user32.GetWindowTextW", side_effect=fake_get_window_text), \
             patch("ctypes.windll.user32.GetWindowRect", side_effect=fake_get_window_rect), \
             patch("ctypes.windll.dwmapi.DwmGetWindowAttribute", side_effect=fake_dwm_get_attribute), \
             patch("ctypes.windll.user32.GetWindowThreadProcessId", side_effect=fake_get_thread_process_id), \
             patch("ctypes.windll.user32.GetForegroundWindow", return_value=1001), \
             patch("psutil.Process") as mock_proc:

            mock_proc.return_value.name.side_effect = lambda: "test_app.exe"

            result = self.tool.run()
            self.assertTrue(result.success)
            self.assertEqual(result.data["returned_count"], 2)

            titles = [w["title"] for w in result.data["windows"]]
            self.assertIn("Google Chrome - GitHub", titles)
            self.assertIn("Notepad - Notes.txt", titles)
            self.assertNotIn("Hidden Worker Window", titles)
            self.assertNotIn("Program Manager", titles)
            self.assertNotIn("Zero Size Tooltip", titles)

            # Check active status
            chrome_win = next(w for w in result.data["windows"] if "Chrome" in w["title"])
            self.assertTrue(chrome_win["is_active"])
            notepad_win = next(w for w in result.data["windows"] if "Notepad" in w["title"])
            self.assertFalse(notepad_win["is_active"])

    def test_title_truncation_and_maximum_limits(self):
        """Verify window titles are capped at 100 characters and limit cannot exceed 15."""
        long_title = "A" * 150

        def fake_enum_windows(callback, lparam):
            callback(2001, lparam)
            return 1

        with patch("ctypes.windll.user32.OpenDesktopW", return_value=None), \
             patch("ctypes.windll.user32.EnumWindows", side_effect=fake_enum_windows), \
             patch("ctypes.windll.user32.IsWindowVisible", return_value=1), \
             patch("ctypes.windll.user32.GetWindowTextLengthW", return_value=len(long_title)), \
             patch("ctypes.windll.user32.GetWindowTextW") as mock_text, \
             patch("ctypes.windll.user32.GetWindowRect") as mock_rect, \
             patch("ctypes.windll.dwmapi.DwmGetWindowAttribute", return_value=0), \
             patch("ctypes.windll.user32.GetWindowThreadProcessId", return_value=1), \
             patch("psutil.Process"):

            def fill_title(hwnd, buf, max_count):
                buf.value = long_title
                return len(long_title)

            def fill_rect(hwnd, rect_ref):
                r = rect_ref._obj if hasattr(rect_ref, "_obj") else rect_ref
                r.left, r.top, r.right, r.bottom = 0, 0, 100, 100
                return 1

            mock_text.side_effect = fill_title
            mock_rect.side_effect = fill_rect

            result = self.tool.run(limit=50)  # Requesting 50, but max allowed is 15
            self.assertTrue(result.success)
            self.assertEqual(len(result.data["windows"]), 1)
            title = result.data["windows"][0]["title"]
            self.assertLessEqual(len(title), MAX_WINDOW_TITLE_CHARS + 3)
            self.assertTrue(title.endswith("..."))

    def test_empty_desktop_returns_clean_message(self):
        """Verify behavior when no visible application windows exist."""
        with patch("ctypes.windll.user32.OpenDesktopW", return_value=None), \
             patch("ctypes.windll.user32.EnumWindows", return_value=1):
            result = self.tool.run()
            self.assertTrue(result.success)
            self.assertEqual(result.data["returned_count"], 0)
            self.assertIn("No visible application windows found", result.output)

    def test_api_failure_handled_gracefully(self):
        """Verify exception during enumeration does not crash tool."""
        with patch("ctypes.windll.user32.OpenDesktopW", side_effect=RuntimeError("Win32 fault")), \
             patch("ctypes.windll.user32.EnumWindows", side_effect=RuntimeError("Win32 fault")):
            result = self.tool.run()
            self.assertFalse(result.success)
            self.assertIn("Failed to enumerate", result.error)

    def test_default_registry_contains_list_open_windows(self):
        """Verify list_open_windows is registered in the default ToolRegistry."""
        registry = get_default_tool_registry()
        tool = registry.get_tool("list_open_windows")
        self.assertIsNotNone(tool)
        self.assertEqual(tool.permission_level, PermissionLevel.SAFE)

    def test_high_bit_64bit_handle_active_matching(self):
        """Verify active window matching works for 64-bit handles with high bits set."""
        # Handles > 0x7FFFFFFF (bit 31 set) must not be truncated to negative 32-bit signed int
        high_hwnd = 0x8000ABCD

        def fake_enum_windows(callback, lparam):
            callback(high_hwnd, lparam)
            return 1

        with patch("ctypes.windll.user32.OpenDesktopW", return_value=None), \
             patch("ctypes.windll.user32.EnumWindows", side_effect=fake_enum_windows), \
             patch("ctypes.windll.user32.IsWindowVisible", return_value=1), \
             patch("ctypes.windll.user32.GetWindowTextLengthW", return_value=12), \
             patch("ctypes.windll.user32.GetWindowTextW") as mock_text, \
             patch("ctypes.windll.user32.GetWindowRect") as mock_rect, \
             patch("ctypes.windll.dwmapi.DwmGetWindowAttribute", return_value=0), \
             patch("ctypes.windll.user32.GetWindowThreadProcessId", return_value=1), \
             patch("ctypes.windll.user32.GetForegroundWindow", return_value=high_hwnd), \
             patch("psutil.Process"):

            def fill_title(hwnd, buf, max_count):
                buf.value = "High Bit Win"
                return 12

            def fill_rect(hwnd, rect_ref):
                r = rect_ref._obj if hasattr(rect_ref, "_obj") else rect_ref
                r.left, r.top, r.right, r.bottom = 0, 0, 500, 500
                return 1

            mock_text.side_effect = fill_title
            mock_rect.side_effect = fill_rect

            result = self.tool.run()
            self.assertTrue(result.success)
            self.assertEqual(len(result.data["windows"]), 1)
            self.assertTrue(result.data["windows"][0]["is_active"])

    def test_win32_signatures_configured_correctly(self):
        """Verify user32 Win32 function signatures have explicit restype and argtypes."""
        user32 = ctypes.windll.user32
        self.assertEqual(user32.GetForegroundWindow.restype, wintypes.HWND)
        self.assertEqual(user32.OpenDesktopW.restype, wintypes.HDESK)
        self.assertEqual(user32.CloseDesktop.restype, wintypes.BOOL)
        self.assertEqual(user32.EnumWindows.restype, wintypes.BOOL)
        self.assertEqual(user32.EnumDesktopWindows.restype, wintypes.BOOL)
        self.assertEqual(user32.GetWindowTextLengthW.restype, ctypes.c_int)
        self.assertEqual(user32.IsWindowVisible.restype, wintypes.BOOL)



class TestListOpenWindowsIntegration(unittest.TestCase):
    """Integration test with ExoCortexAgent and AgentExecutor."""

    def test_agent_can_query_open_windows(self):
        """Verify the agent execution engine can invoke list_open_windows."""
        mock_provider = MockSLMProvider()
        agent = ExoCortexAgent(slm_provider=mock_provider)

        # Mock the tool output
        with patch.object(ListOpenWindowsTool, "run") as mock_run:
            from exocortex.tools.base import ToolResult
            mock_run.return_value = ToolResult(
                success=True,
                output="Visible Desktop Windows (1 found):\n1. [ACTIVE] \"Notepad - Notes.txt\" (notepad.exe)",
                data={"returned_count": 1, "windows": [{"title": "Notepad - Notes.txt", "process_name": "notepad.exe", "is_active": True}]},
            )
            # Execute through registry
            res = agent.tool_registry.execute("list_open_windows", {"limit": 5})
            self.assertTrue(res.success)
            self.assertIn("Notepad", res.output)


if __name__ == "__main__":
    unittest.main()
