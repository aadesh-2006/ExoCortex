"""
Comprehensive unit and integration tests for InspectWindowUIElementsTool.

Verifies:
- Target window resolution, not-found, and ambiguity handling.
- Bounded Control View traversal, depth limiting, and element count capping.
- Password redaction, offscreen element handling, and absence of value patterns.
- COM apartment initialization and cleanup, including RPC_E_CHANGED_MODE.
- HRESULT error handling and safe interface cleanup.
- Tool registration, SAFE permissions, and agent executor integration.
"""

import ctypes
from ctypes import wintypes
import threading
import unittest
from unittest.mock import MagicMock, call, patch

from exocortex.agent import ExoCortexAgent
from exocortex.slm.mock_provider import MockSLMProvider
from exocortex.tools.base import PermissionLevel
from exocortex.tools.registry import get_default_tool_registry
from exocortex.tools.windows.ui_accessibility import (
    CLSID_CUIAutomation8_STR,
    CLSID_CUIAutomation_STR,
    COINIT_MULTITHREADED,
    DEFAULT_DEPTH_LIMIT,
    DEFAULT_ELEMENTS_LIMIT,
    DEFAULT_UIA_CONNECTION_TIMEOUT_MS,
    DEFAULT_UIA_TRANSACTION_TIMEOUT_MS,
    IID_IUIAutomation2_STR,
    IID_IUIAutomation_STR,
    MAX_DEPTH_LIMIT,
    MAX_ELEMENTS_LIMIT,
    MAX_TEXT_OUTPUT_CHARS,
    RPC_E_CHANGED_MODE,
    S_FALSE,
    S_OK,
    COMSession,
    InspectWindowUIElementsTool,
    UIAUnsupportedPlatformError,
    _safe_release,
    resolve_target_window,
)


class TestTargetWindowResolution(unittest.TestCase):
    """Tests for resolving target windows from candidate desktop windows."""

    def setUp(self):
        self.mock_candidates = [
            {"hwnd": 0x1001, "title": "Notepad - Notes.txt", "process_name": "notepad.exe", "is_active": True},
            {"hwnd": 0x1002, "title": "Google Chrome - GitHub", "process_name": "chrome.exe", "is_active": False},
            {"hwnd": 0x1003, "title": "Google Chrome - Stack Overflow", "process_name": "chrome.exe", "is_active": False},
            {"hwnd": 0x1004, "title": "Calculator", "process_name": "CalculatorApp.exe", "is_active": False},
        ]

    def test_missing_arguments_returns_error(self):
        """Must provide at least target_title or process_name."""
        win, err, meta = resolve_target_window(None, None, self.mock_candidates)
        self.assertIsNone(win)
        self.assertIn("At least one", err)
        self.assertEqual(meta["status"], "invalid_arguments")

        win, err, meta = resolve_target_window("   ", "", self.mock_candidates)
        self.assertIsNone(win)
        self.assertIn("At least one", err)

    def test_single_match_by_title_substring(self):
        """Matches a single window uniquely by title substring."""
        win, err, meta = resolve_target_window(target_title="Notes", candidate_windows=self.mock_candidates)
        self.assertIsNone(err)
        self.assertIsNotNone(win)
        self.assertEqual(win["hwnd"], 0x1001)
        self.assertEqual(win["title"], "Notepad - Notes.txt")
        self.assertEqual(meta["status"], "found")
        self.assertEqual(meta["matched_count"], 1)

    def test_single_match_by_process_name(self):
        """Matches a single window uniquely by process name."""
        win, err, meta = resolve_target_window(process_name="notepad", candidate_windows=self.mock_candidates)
        self.assertIsNone(err)
        self.assertIsNotNone(win)
        self.assertEqual(win["hwnd"], 0x1001)

        win, err, meta = resolve_target_window(process_name="CalculatorApp.exe", candidate_windows=self.mock_candidates)
        self.assertIsNone(err)
        self.assertIsNotNone(win)
        self.assertEqual(win["hwnd"], 0x1004)

    def test_match_by_both_title_and_process(self):
        """Matches when both title and process are specified and narrow down uniquely."""
        win, err, meta = resolve_target_window(
            target_title="GitHub",
            process_name="chrome.exe",
            candidate_windows=self.mock_candidates,
        )
        self.assertIsNone(err)
        self.assertIsNotNone(win)
        self.assertEqual(win["hwnd"], 0x1002)

    def test_not_found_returns_clean_error(self):
        """Returns not_found when no windows match query."""
        win, err, meta = resolve_target_window(
            target_title="Nonexistent Application",
            candidate_windows=self.mock_candidates,
        )
        self.assertIsNone(win)
        self.assertIn("No visible application window found", err)
        self.assertEqual(meta["status"], "not_found")
        self.assertEqual(meta["matched_count"], 0)

    def test_ambiguous_match_returns_explicit_error(self):
        """Multiple matches must return ambiguity error without picking an arbitrary window."""
        win, err, meta = resolve_target_window(
            process_name="chrome",
            candidate_windows=self.mock_candidates,
        )
        self.assertIsNone(win)
        self.assertIn("Ambiguous window match", err)
        self.assertIn("found 2 matching windows", err)
        self.assertEqual(meta["status"], "ambiguous")
        self.assertEqual(meta["matched_count"], 2)
        self.assertEqual(len(meta["matches"]), 2)


class TestCOMSession(unittest.TestCase):
    """Tests for thread COM apartment initialization and cleanup."""

    @patch("ctypes.windll.ole32.CoInitializeEx", return_value=S_OK)
    @patch("ctypes.windll.ole32.CoUninitialize")
    def test_com_init_s_ok_uninitializes(self, mock_uninit, mock_init):
        """When CoInitializeEx returns S_OK, CoUninitialize must be called upon exit with MTA."""
        with COMSession() as session:
            self.assertTrue(session.should_uninitialize)
        mock_init.assert_called_once_with(None, COINIT_MULTITHREADED)
        mock_uninit.assert_called_once()

    @patch("ctypes.windll.ole32.CoInitializeEx", return_value=S_FALSE)
    @patch("ctypes.windll.ole32.CoUninitialize")
    def test_com_init_s_false_uninitializes(self, mock_uninit, mock_init):
        """When CoInitializeEx returns S_FALSE, CoUninitialize must be called upon exit with MTA."""
        with COMSession() as session:
            self.assertTrue(session.should_uninitialize)
        mock_init.assert_called_once_with(None, COINIT_MULTITHREADED)
        mock_uninit.assert_called_once()

    @patch("ctypes.windll.ole32.CoInitializeEx", return_value=RPC_E_CHANGED_MODE)
    @patch("ctypes.windll.ole32.CoUninitialize")
    def test_com_init_rpc_e_changed_mode_does_not_uninitialize(self, mock_uninit, mock_init):
        """When CoInitializeEx returns RPC_E_CHANGED_MODE, it must raise RuntimeError and not uninitialize."""
        with self.assertRaises(RuntimeError) as ctx:
            with COMSession():
                pass
        self.assertIn("RPC_E_CHANGED_MODE", str(ctx.exception))
        mock_uninit.assert_not_called()

    @patch("ctypes.windll.ole32.CoInitializeEx", return_value=ctypes.c_long(0x80004005).value) # E_FAIL
    @patch("ctypes.windll.ole32.CoUninitialize")
    def test_com_init_failure_raises_and_does_not_uninitialize(self, mock_uninit, mock_init):
        """Generic COM error raises RuntimeError and does not call CoUninitialize."""
        with self.assertRaises(RuntimeError) as ctx:
            with COMSession():
                pass
        self.assertIn("CoInitializeEx failed", str(ctx.exception))
        mock_uninit.assert_not_called()

    @patch("ctypes.windll.ole32.CoInitializeEx", return_value=S_OK)
    @patch("ctypes.windll.ole32.CoUninitialize")
    @patch("ctypes.windll.user32.OpenDesktopW")
    @patch("ctypes.windll.user32.SetThreadDesktop")
    @patch("ctypes.windll.user32.CloseDesktop")
    @patch("ctypes.windll.user32.GetUserObjectInformationW")
    def test_desktop_switching_lifecycle(self, mock_uoi, mock_close, mock_set, mock_open, mock_uninit, mock_init):
        """Verify non-default desktop triggers switch, and restores original desktop + closes handle upon exit."""
        mock_open.return_value = 0x9999
        mock_set.return_value = 1

        def fake_uoi(handle, nindex, buf, length, needed):
            if buf:
                buf.value = "custom_desk"
            return 1

        mock_uoi.side_effect = fake_uoi

        with COMSession() as session:
            self.assertTrue(session._switched_desktop)
            self.assertEqual(session._h_desk, 0x9999)
            orig_desk = session._orig_desk

        # On exit, SetThreadDesktop must restore original desktop, and CloseDesktop must close h_desk
        mock_set.assert_has_calls([call(0x9999), call(orig_desk)])
        mock_close.assert_called_once_with(0x9999)

    @patch("ctypes.windll.ole32.CoInitializeEx", return_value=S_OK)
    @patch("ctypes.windll.ole32.CoUninitialize")
    @patch("ctypes.windll.user32.OpenDesktopW")
    @patch("ctypes.windll.user32.SetThreadDesktop")
    @patch("ctypes.windll.user32.CloseDesktop")
    @patch("ctypes.windll.user32.GetUserObjectInformationW")
    def test_already_on_default_desktop_does_not_switch(self, mock_uoi, mock_close, mock_set, mock_open, mock_uninit, mock_init):
        """Verify thread already on Default desktop skips OpenDesktopW and SetThreadDesktop."""
        def fake_uoi(handle, nindex, buf, length, needed):
            needed._obj.value = 16
            if buf:
                buf.value = "Default"
            return 1

        mock_uoi.side_effect = fake_uoi

        with COMSession() as session:
            self.assertFalse(session._switched_desktop)

        mock_open.assert_not_called()
        mock_set.assert_not_called()
        mock_close.assert_not_called()



class TestInspectWindowUIElementsTool(unittest.TestCase):
    """Unit tests for tool schema, permissions, and inspection execution."""

    def setUp(self):
        self.tool = InspectWindowUIElementsTool()

    def test_metadata_and_permission_level(self):
        """Verify tool name, SAFE permission level, and parameters."""
        self.assertEqual(self.tool.name, "inspect_window_ui_elements")
        self.assertEqual(self.tool.permission_level, PermissionLevel.SAFE)
        param_names = [p.name for p in self.tool.parameters]
        self.assertIn("target_title", param_names)
        self.assertIn("process_name", param_names)
        self.assertIn("max_depth", param_names)
        self.assertIn("max_elements", param_names)

        schema = self.tool.get_schema()
        self.assertEqual(schema["name"], "inspect_window_ui_elements")
        self.assertEqual(schema["permission_level"], "safe")

    def test_not_found_window_execution(self):
        """Tool returns clear failure when target window is not found."""
        with patch("exocortex.tools.windows.ui_accessibility.enumerate_desktop_windows", return_value=[]):
            res = self.tool.run(target_title="Nonexistent")
            self.assertFalse(res.success)
            self.assertIn("No visible application window found", res.error)
            self.assertEqual(res.data["status"], "not_found")

    def test_ambiguous_window_execution(self):
        """Tool returns explicit ambiguity failure when multiple windows match."""
        candidates = [
            {"hwnd": 0x1, "title": "Chrome 1", "process_name": "chrome.exe", "is_active": True},
            {"hwnd": 0x2, "title": "Chrome 2", "process_name": "chrome.exe", "is_active": False},
        ]
        with patch("exocortex.tools.windows.ui_accessibility.enumerate_desktop_windows", return_value=candidates):
            res = self.tool.run(process_name="chrome")
            self.assertFalse(res.success)
            self.assertIn("Ambiguous window match", res.error)
            self.assertEqual(res.data["status"], "ambiguous")
            self.assertEqual(res.data["matched_count"], 2)

    def test_password_redaction_and_privacy_safeguards(self):
        """Verify that password fields are fully redacted and value patterns omitted."""
        mock_candidate = {"hwnd": 0x1234, "title": "Login Window", "process_name": "login.exe", "is_active": True}

        # Mock element data directly returned by traversal
        mock_elements = [
            {
                "depth": 0,
                "name": "Login Window",
                "control_type": "window",
                "automation_id": "Win1",
                "class_name": "WindowClass",
                "is_enabled": True,
                "is_keyboard_focusable": True,
                "is_password": False,
                "is_offscreen": False,
            },
            {
                "depth": 1,
                "name": "[REDACTED_PASSWORD]",
                "control_type": "edit",
                "automation_id": "[REDACTED_PASSWORD]",
                "class_name": "PasswordEdit",
                "is_enabled": True,
                "is_keyboard_focusable": True,
                "is_password": True,
                "is_offscreen": False,
            },
            {
                "depth": 1,
                "name": "Sign In",
                "control_type": "button",
                "automation_id": "SubmitBtn",
                "class_name": "Button",
                "is_enabled": True,
                "is_keyboard_focusable": True,
                "is_password": False,
                "is_offscreen": True,
            },
        ]

        with patch("exocortex.tools.windows.ui_accessibility.enumerate_desktop_windows", return_value=[mock_candidate]), \
             patch("exocortex.tools.windows.ui_accessibility.COMSession"), \
             patch.object(self.tool, "_traverse_control_tree", return_value=mock_elements):

            res = self.tool.run(target_title="Login")
            self.assertTrue(res.success)
            self.assertEqual(res.data["total_elements"], 3)

            # Check password element in data
            pw_elem = res.data["elements"][1]
            self.assertTrue(pw_elem["is_password"])
            self.assertEqual(pw_elem["name"], "[REDACTED_PASSWORD]")
            self.assertEqual(pw_elem["automation_id"], "[REDACTED_PASSWORD]")

            # Check text output
            self.assertIn("[REDACTED_PASSWORD]", res.output)
            self.assertIn("password", res.output)
            self.assertIn("offscreen", res.output)

            # Ensure raw HWND is NOT exposed to output or data
            self.assertNotIn("0x1234", res.output)
            for el in res.data["elements"]:
                self.assertNotIn("hwnd", el)
                self.assertNotIn("pointer", el)

    def test_depth_and_element_count_clamping(self):
        """Verify max_depth is clamped to [1..4] and max_elements clamped to [1..50]."""
        mock_candidate = {"hwnd": 0x1234, "title": "Editor", "process_name": "editor.exe", "is_active": True}

        with patch("exocortex.tools.windows.ui_accessibility.enumerate_desktop_windows", return_value=[mock_candidate]), \
             patch("exocortex.tools.windows.ui_accessibility.COMSession"), \
             patch.object(self.tool, "_traverse_control_tree", return_value=[]) as mock_traverse:

            # Request extreme depth 100 and elements 1000
            self.tool.run(target_title="Editor", max_depth=100, max_elements=1000)
            mock_traverse.assert_called_once_with(hwnd=0x1234, max_depth=MAX_DEPTH_LIMIT, max_elements=MAX_ELEMENTS_LIMIT)

            mock_traverse.reset_mock()
            # Request negative / below minimum values
            self.tool.run(target_title="Editor", max_depth=-5, max_elements=0)
            mock_traverse.assert_called_once_with(hwnd=0x1234, max_depth=1, max_elements=1)

    def test_hresult_failure_handled_gracefully(self):
        """HRESULT failure during traversal returns error without crashing."""
        mock_candidate = {"hwnd": 0x1234, "title": "App", "process_name": "app.exe", "is_active": True}

        with patch("exocortex.tools.windows.ui_accessibility.enumerate_desktop_windows", return_value=[mock_candidate]), \
             patch("exocortex.tools.windows.ui_accessibility.COMSession"), \
             patch.object(self.tool, "_traverse_control_tree", side_effect=RuntimeError("ElementFromHandle failed")):

            res = self.tool.run(target_title="App")
            self.assertFalse(res.success)
            self.assertIn("UI Automation inspection failed", res.error)

    def test_output_size_truncation(self):
        """Output text longer than MAX_TEXT_OUTPUT_CHARS is cleanly truncated."""
        mock_candidate = {"hwnd": 0x1234, "title": "Big Window", "process_name": "big.exe", "is_active": True}

        # Generate many elements to exceed MAX_TEXT_OUTPUT_CHARS
        many_elements = [
            {
                "depth": 1,
                "name": f"Long element label with a lot of verbose explanatory information number {i}",
                "control_type": "button",
                "automation_id": f"LongAutomationIdentifierNumber{i}MoreCharactersToFillBuffer",
                "class_name": "VeryLongClassNameThatTakesUpSpaceInTheBuffer",
                "is_enabled": True,
                "is_keyboard_focusable": True,
                "is_password": False,
                "is_offscreen": False,
            }
            for i in range(50)
        ]

        with patch("exocortex.tools.windows.ui_accessibility.enumerate_desktop_windows", return_value=[mock_candidate]), \
             patch("exocortex.tools.windows.ui_accessibility.COMSession"), \
             patch.object(self.tool, "_traverse_control_tree", return_value=many_elements):

            res = self.tool.run(target_title="Big Window")
            self.assertTrue(res.success)
            self.assertLessEqual(len(res.output), MAX_TEXT_OUTPUT_CHARS + 50)
            self.assertIn("[output truncated]", res.output)

    def test_ctypes_vtable_extraction_with_mock_com_object(self):
        """Verify real ctypes vtable dispatch in _extract_element_data."""
        vtbl = (ctypes.c_void_p * 40)()
        oleaut32 = ctypes.windll.oleaut32
        oleaut32.SysAllocString.restype = ctypes.c_void_p
        oleaut32.SysAllocString.argtypes = [ctypes.c_wchar_p]

        def fn_bstr(val):
            def impl(this, out_p):
                out_p.contents.value = oleaut32.SysAllocString(val)
                return 0
            return impl

        def fn_bool(val):
            def impl(this, out_p):
                out_p.contents.value = 1 if val else 0
                return 0
            return impl

        FN_B = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))
        FN_BOOL = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(wintypes.BOOL))
        FN_INT = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int))

        cb_tid = FN_INT(lambda t, o: 0)
        cb_name = FN_B(fn_bstr("Submit Application"))
        cb_type = FN_B(fn_bstr("Button"))
        cb_aid = FN_B(fn_bstr("btn_submit"))
        cb_cname = FN_B(fn_bstr("CustomButtonClass"))
        cb_enabled = FN_BOOL(fn_bool(True))
        cb_focusable = FN_BOOL(fn_bool(True))
        cb_pass = FN_BOOL(fn_bool(False))
        cb_offscreen = FN_BOOL(fn_bool(False))

        vtbl[21] = ctypes.cast(cb_tid, ctypes.c_void_p).value
        vtbl[22] = ctypes.cast(cb_type, ctypes.c_void_p).value
        vtbl[23] = ctypes.cast(cb_name, ctypes.c_void_p).value
        vtbl[27] = ctypes.cast(cb_focusable, ctypes.c_void_p).value
        vtbl[28] = ctypes.cast(cb_enabled, ctypes.c_void_p).value
        vtbl[29] = ctypes.cast(cb_aid, ctypes.c_void_p).value
        vtbl[30] = ctypes.cast(cb_cname, ctypes.c_void_p).value
        vtbl[35] = ctypes.cast(cb_pass, ctypes.c_void_p).value
        vtbl[38] = ctypes.cast(cb_offscreen, ctypes.c_void_p).value

        p_vtbl = ctypes.cast(ctypes.pointer(vtbl), ctypes.c_void_p)
        p_obj = ctypes.pointer(p_vtbl)
        elem_ptr = ctypes.cast(p_obj, ctypes.c_void_p)

        data = self.tool._extract_element_data(elem_ptr, depth=2)
        self.assertEqual(data["depth"], 2)
        self.assertEqual(data["name"], "Submit Application")
        self.assertEqual(data["control_type"], "button")
        self.assertEqual(data["automation_id"], "btn_submit")
        self.assertEqual(data["class_name"], "CustomButtonClass")
        self.assertTrue(data["is_enabled"])
        self.assertTrue(data["is_keyboard_focusable"])
        self.assertFalse(data["is_password"])
        self.assertFalse(data["is_offscreen"])



class TestRegistryAndIntegration(unittest.TestCase):
    """Integration test with ToolRegistry and ExoCortexAgent."""

    def test_registry_has_inspect_window_ui_elements(self):
        """Verify inspect_window_ui_elements is registered in default ToolRegistry."""
        reg = get_default_tool_registry()
        tool = reg.get_tool("inspect_window_ui_elements")
        self.assertIsNotNone(tool)
        self.assertEqual(tool.permission_level, PermissionLevel.SAFE)

    def test_agent_execution_integration(self):
        """Verify the agent can execute inspect_window_ui_elements via ToolRegistry."""
        mock_provider = MockSLMProvider()
        agent = ExoCortexAgent(slm_provider=mock_provider)

        tool = agent.tool_registry.get_tool("inspect_window_ui_elements")
        self.assertIsNotNone(tool)

        mock_candidate = {"hwnd": 0x5555, "title": "Notepad - Task.txt", "process_name": "notepad.exe", "is_active": True}
        mock_elems = [
            {
                "depth": 0,
                "name": "Notepad",
                "control_type": "window",
                "automation_id": "",
                "class_name": "Notepad",
                "is_enabled": True,
                "is_keyboard_focusable": True,
                "is_password": False,
                "is_offscreen": False,
            }
        ]

        with patch("exocortex.tools.windows.ui_accessibility.enumerate_desktop_windows", return_value=[mock_candidate]), \
             patch("exocortex.tools.windows.ui_accessibility.COMSession"), \
             patch.object(InspectWindowUIElementsTool, "_traverse_control_tree", return_value=mock_elems):

            res = agent.tool_registry.execute("inspect_window_ui_elements", {"target_title": "Notepad"})
            self.assertTrue(res.success)
            self.assertIn("Notepad", res.output)
            self.assertEqual(res.data["total_elements"], 1)


class TestSafeRelease(unittest.TestCase):
    """Tests for COM interface pointer safe release and idempotence."""

    def test_safe_release_null_or_none(self):
        """None or null c_void_p do not cause errors."""
        _safe_release(None)
        _safe_release(ctypes.c_void_p())
        _safe_release(ctypes.c_void_p(0))

    def test_safe_release_idempotence(self):
        """_safe_release calls Release exactly once and nulls the pointer to prevent double-free."""
        release_count = 0

        def fake_release(this):
            nonlocal release_count
            release_count += 1
            return 0

        FN_REL = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)
        cb_rel = FN_REL(fake_release)

        vtbl = (ctypes.c_void_p * 3)()
        vtbl[2] = ctypes.cast(cb_rel, ctypes.c_void_p).value
        p_vtbl = ctypes.cast(ctypes.pointer(vtbl), ctypes.c_void_p)
        p_obj = ctypes.pointer(p_vtbl)
        ptr = ctypes.cast(p_obj, ctypes.c_void_p)

        self.assertIsNotNone(ptr.value)
        _safe_release(ptr)
        self.assertEqual(release_count, 1)
        self.assertIsNone(ptr.value)

        # Second release on same pointer object must be a no-op
        _safe_release(ptr)
        self.assertEqual(release_count, 1)


class TestNativeSafetyHardening(unittest.TestCase):
    """Tests for Phase 1 & 2 safety hardening: exception-safe pointer ownership and timeouts."""

    def test_tool_result_contains_timeout_metadata(self):
        """ToolResult.data must expose timeout configuration metadata."""
        tool = InspectWindowUIElementsTool()
        mock_win = {"hwnd": 0x1234, "title": "Test App", "process_name": "app.exe", "is_active": True}
        mock_elems = [
            {
                "depth": 0,
                "name": "Test App",
                "control_type": "window",
                "automation_id": "win1",
                "class_name": "AppClass",
                "is_enabled": True,
                "is_keyboard_focusable": True,
                "is_password": False,
                "is_offscreen": False,
            }
        ]
        mock_meta = {
            "timeout_configured": True,
            "connection_timeout_ms": 2000,
            "transaction_timeout_ms": 2000,
        }

        with patch("exocortex.tools.windows.ui_accessibility.enumerate_desktop_windows", return_value=[mock_win]), \
             patch("exocortex.tools.windows.ui_accessibility.COMSession"), \
             patch.object(tool, "_traverse_control_tree", return_value=(mock_elems, mock_meta)):

            res = tool.run(target_title="Test App")
            self.assertTrue(res.success)
            self.assertTrue(res.data["timeout_configured"])
            self.assertEqual(res.data["connection_timeout_ms"], 2000)
            self.assertEqual(res.data["transaction_timeout_ms"], 2000)
            self.assertEqual(res.data["total_elements"], 1)

    def test_tool_result_empty_elements_contains_timeout_metadata(self):
        """When 0 elements found, timeout metadata is still preserved in data dict."""
        tool = InspectWindowUIElementsTool()
        mock_win = {"hwnd": 0x1234, "title": "Empty App", "process_name": "app.exe", "is_active": True}
        mock_meta = {
            "timeout_configured": True,
            "connection_timeout_ms": 2000,
            "transaction_timeout_ms": 2000,
        }

        with patch("exocortex.tools.windows.ui_accessibility.enumerate_desktop_windows", return_value=[mock_win]), \
             patch("exocortex.tools.windows.ui_accessibility.COMSession"), \
             patch.object(tool, "_traverse_control_tree", return_value=([], mock_meta)):

            res = tool.run(target_title="Empty App")
            self.assertTrue(res.success)
            self.assertTrue(res.data["timeout_configured"])
            self.assertEqual(res.data["connection_timeout_ms"], 2000)
            self.assertEqual(res.data["transaction_timeout_ms"], 2000)
            self.assertEqual(res.data["total_elements"], 0)

    def test_cuiautomation8_timeout_initialization(self):
        """Verify _traverse_control_tree configures timeouts when CUIAutomation8 is available."""
        tool = InspectWindowUIElementsTool()

        put_conn_called = False
        put_tran_called = False
        conn_val = 0
        tran_val = 0

        def fake_put_conn(this, timeout):
            nonlocal put_conn_called, conn_val
            put_conn_called = True
            conn_val = timeout
            return 0

        def fake_put_tran(this, timeout):
            nonlocal put_tran_called, tran_val
            put_tran_called = True
            tran_val = timeout
            return 0

        FN_PUT = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, wintypes.DWORD)
        cb_put_conn = FN_PUT(fake_put_conn)
        cb_put_tran = FN_PUT(fake_put_tran)

        vtbl_auto = (ctypes.c_void_p * 65)()
        vtbl_auto[61] = ctypes.cast(cb_put_conn, ctypes.c_void_p).value
        vtbl_auto[63] = ctypes.cast(cb_put_tran, ctypes.c_void_p).value

        FN_ELEM = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, wintypes.HWND, ctypes.POINTER(ctypes.c_void_p))
        FN_WALK = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))

        cb_elem = FN_ELEM(lambda this, hwnd, out_p: 0x80004005)
        cb_walk = FN_WALK(lambda this, out_p: 0)

        vtbl_auto[6] = ctypes.cast(cb_elem, ctypes.c_void_p).value
        vtbl_auto[14] = ctypes.cast(cb_walk, ctypes.c_void_p).value

        p_vtbl = ctypes.cast(ctypes.pointer(vtbl_auto), ctypes.c_void_p)
        p_obj = ctypes.pointer(p_vtbl)

        def fake_cocreate(rclsid, pUnkOuter, dwClsContext, riid, ppv):
            ppv._obj.value = ctypes.cast(p_obj, ctypes.c_void_p).value
            return 0

        with patch("ctypes.windll.ole32.CoCreateInstance", side_effect=fake_cocreate):
            with self.assertRaises(RuntimeError) as ctx:
                tool._traverse_control_tree(hwnd=0x1234, max_depth=2, max_elements=10)

            self.assertIn("ElementFromHandle failed", str(ctx.exception))
            self.assertTrue(put_conn_called)
            self.assertTrue(put_tran_called)
            self.assertEqual(conn_val, DEFAULT_UIA_CONNECTION_TIMEOUT_MS)
            self.assertEqual(tran_val, DEFAULT_UIA_TRANSACTION_TIMEOUT_MS)

    def test_cuiautomation8_failure_fails_closed_without_fallback(self):
        """When CUIAutomation8 fails, fails closed with UIAUnsupportedPlatformError and does NOT fall back."""
        tool = InspectWindowUIElementsTool()

        cocreate_calls = []

        def fake_cocreate(rclsid, pUnkOuter, dwClsContext, riid, ppv):
            cocreate_calls.append(len(cocreate_calls))
            return 0x80040154  # REGDB_E_CLASSNOTREG

        with patch("ctypes.windll.ole32.CoCreateInstance", side_effect=fake_cocreate):
            with self.assertRaises(UIAUnsupportedPlatformError) as ctx:
                tool._traverse_control_tree(hwnd=0x1234, max_depth=2, max_elements=10)

            self.assertEqual(len(cocreate_calls), 1)
            self.assertIn("Legacy unprotected fallback is disabled for safety", str(ctx.exception))

            # Also verify tool.run returns status="unsupported_platform"
            with patch("exocortex.tools.windows.ui_accessibility.enumerate_desktop_windows", return_value=[{"hwnd": 0x1234, "title": "App", "process_name": "app.exe"}]):
                res = tool.run(process_name="app.exe")
                self.assertFalse(res.success)
                self.assertEqual(res.data.get("status"), "unsupported_platform")

    def test_timeout_setter_failure_fails_closed(self):
        """When put_ConnectionTimeout or put_TransactionTimeout fails, fails closed and releases p_auto."""
        tool = InspectWindowUIElementsTool()

        released_ptrs = []

        def track_release(ptr):
            if ptr and getattr(ptr, "value", None):
                released_ptrs.append(ptr.value)
                ptr.value = None

        vtbl_auto = (ctypes.c_void_p * 65)()
        FN_PUT = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, wintypes.DWORD)
        cb_put_conn = FN_PUT(lambda this, timeout: 0)
        cb_put_tran = FN_PUT(lambda this, timeout: 0x80004005)  # E_FAIL

        vtbl_auto[61] = ctypes.cast(cb_put_conn, ctypes.c_void_p).value
        vtbl_auto[63] = ctypes.cast(cb_put_tran, ctypes.c_void_p).value

        p_vtbl = ctypes.cast(ctypes.pointer(vtbl_auto), ctypes.c_void_p)
        p_obj = ctypes.pointer(p_vtbl)
        auto_ptr_val = ctypes.cast(p_obj, ctypes.c_void_p).value

        def fake_cocreate(rclsid, pUnkOuter, dwClsContext, riid, ppv):
            ppv._obj.value = auto_ptr_val
            return 0

        with patch("ctypes.windll.ole32.CoCreateInstance", side_effect=fake_cocreate), \
             patch("exocortex.tools.windows.ui_accessibility._safe_release", side_effect=track_release):
            with self.assertRaises(UIAUnsupportedPlatformError) as ctx:
                tool._traverse_control_tree(hwnd=0x1234, max_depth=2, max_elements=10)

            self.assertIn("Failed to configure UI Automation timeouts", str(ctx.exception))
            self.assertIn(auto_ptr_val, released_ptrs)

    def test_worker_thread_isolation_keeps_main_thread_unmutated(self):
        """Verify inspection executes inside a dedicated worker thread, keeping main thread untouched."""
        tool = InspectWindowUIElementsTool()
        worker_thread_id = None
        main_thread_id = threading.current_thread().ident

        def fake_worker_task():
            nonlocal worker_thread_id
            worker_thread_id = threading.current_thread().ident
            return ([{"depth": 0, "name": "Isolated"}], {"timeout_configured": True})

        res = tool._execute_in_worker_thread(fake_worker_task)
        self.assertIsNotNone(worker_thread_id)
        self.assertNotEqual(worker_thread_id, main_thread_id)
        self.assertEqual(res[0][0]["name"], "Isolated")

    def test_worker_thread_com_init_failure_prevents_traversal(self):
        """Worker thread COM initialization failure prevents traversal and returns inspection_failed."""
        tool = InspectWindowUIElementsTool()
        mock_win = {"hwnd": 0x1234, "title": "App", "process_name": "app.exe", "is_active": True}

        with patch("exocortex.tools.windows.ui_accessibility.enumerate_desktop_windows", return_value=[mock_win]), \
             patch("ctypes.windll.ole32.CoInitializeEx", return_value=RPC_E_CHANGED_MODE), \
             patch.object(tool, "_traverse_control_tree") as mock_traverse:

            res = tool.run(process_name="app.exe")
            self.assertFalse(res.success)
            self.assertEqual(res.data.get("status"), "inspection_failed")
            self.assertIn("RPC_E_CHANGED_MODE", res.error)
            mock_traverse.assert_not_called()

    def test_exception_in_child_recursion_releases_pointers(self):
        """Verify that when child recursion raises an exception, child and root pointers are released."""
        tool = InspectWindowUIElementsTool()
        released_ptrs = []

        def track_release(ptr):
            if ptr and getattr(ptr, "value", None):
                released_ptrs.append(ptr.value)
                ptr.value = None

        with patch("exocortex.tools.windows.ui_accessibility._safe_release", side_effect=track_release):
            def fake_extract(elem, depth):
                if depth > 0:
                    raise RuntimeError("Simulated child extraction failure")
                return {
                    "depth": depth, "name": "Root", "control_type": "window",
                    "automation_id": "root", "class_name": "RootClass",
                    "is_enabled": True, "is_keyboard_focusable": True,
                    "is_password": False, "is_offscreen": False,
                }

            FN_FC = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))
            FN_SIB = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))

            child_ptr_val = 0xAA01
            cb_fc = FN_FC(lambda this, current, out_p: setattr(out_p.contents, "value", child_ptr_val) or 0)
            cb_sib = FN_SIB(lambda this, current, out_p: setattr(out_p.contents, "value", 0) or 0)

            vtbl_walker = (ctypes.c_void_p * 10)()
            vtbl_walker[4] = ctypes.cast(cb_fc, ctypes.c_void_p).value
            vtbl_walker[6] = ctypes.cast(cb_sib, ctypes.c_void_p).value
            p_obj_w = ctypes.pointer(ctypes.cast(ctypes.pointer(vtbl_walker), ctypes.c_void_p))

            vtbl_auto = (ctypes.c_void_p * 65)()
            FN_PUT = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, wintypes.DWORD)
            cb_put = FN_PUT(lambda this, val: 0)
            vtbl_auto[61] = ctypes.cast(cb_put, ctypes.c_void_p).value
            vtbl_auto[63] = ctypes.cast(cb_put, ctypes.c_void_p).value

            FN_ELEM = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, wintypes.HWND, ctypes.POINTER(ctypes.c_void_p))
            FN_WALK = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))

            root_ptr_val = 0xBB01
            cb_elem = FN_ELEM(lambda this, hwnd, out_p: setattr(out_p.contents, "value", root_ptr_val) or 0)
            cb_walk = FN_WALK(lambda this, out_p: setattr(out_p.contents, "value", ctypes.cast(p_obj_w, ctypes.c_void_p).value) or 0)

            vtbl_auto[6] = ctypes.cast(cb_elem, ctypes.c_void_p).value
            vtbl_auto[14] = ctypes.cast(cb_walk, ctypes.c_void_p).value
            p_obj_a = ctypes.pointer(ctypes.cast(ctypes.pointer(vtbl_auto), ctypes.c_void_p))

            with patch("ctypes.windll.ole32.CoCreateInstance") as mock_cc:
                mock_cc.side_effect = lambda rclsid, pUnk, ctx, riid, ppv: setattr(ppv._obj, "value", ctypes.cast(p_obj_a, ctypes.c_void_p).value) or 0

                with patch.object(tool, "_extract_element_data", side_effect=fake_extract):
                    elems, meta = tool._traverse_control_tree(hwnd=0x1234, max_depth=2, max_elements=10)

                    self.assertEqual(len(elems), 1)
                    self.assertEqual(elems[0]["name"], "Root")
                    self.assertIn(child_ptr_val, released_ptrs)
                    self.assertIn(root_ptr_val, released_ptrs)

    def test_sibling_retrieval_failure_releases_pointers(self):
        """Failure during sibling retrieval releases any populated next_sib pointer."""
        tool = InspectWindowUIElementsTool()
        released_ptrs = []

        def track_release(ptr):
            if ptr and getattr(ptr, "value", None):
                released_ptrs.append(ptr.value)
                ptr.value = None

        with patch("exocortex.tools.windows.ui_accessibility._safe_release", side_effect=track_release):
            child1_val = 0xCC01
            bad_sib_val = 0xCC02

            FN_FC = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))
            FN_SIB = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))

            cb_fc = FN_FC(lambda this, current, out_p: setattr(out_p.contents, "value", child1_val) or 0)
            # get_next_sibling populates next_sib but returns E_FAIL (0x80004005)
            cb_sib = FN_SIB(lambda this, current, out_p: setattr(out_p.contents, "value", bad_sib_val) or 0x80004005)

            vtbl_walker = (ctypes.c_void_p * 10)()
            vtbl_walker[4] = ctypes.cast(cb_fc, ctypes.c_void_p).value
            vtbl_walker[6] = ctypes.cast(cb_sib, ctypes.c_void_p).value
            p_obj_w = ctypes.pointer(ctypes.cast(ctypes.pointer(vtbl_walker), ctypes.c_void_p))

            vtbl_auto = (ctypes.c_void_p * 65)()
            FN_PUT = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, wintypes.DWORD)
            cb_put = FN_PUT(lambda this, val: 0)
            vtbl_auto[61] = ctypes.cast(cb_put, ctypes.c_void_p).value
            vtbl_auto[63] = ctypes.cast(cb_put, ctypes.c_void_p).value

            FN_ELEM = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, wintypes.HWND, ctypes.POINTER(ctypes.c_void_p))
            FN_WALK = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))

            root_ptr_val = 0xCC00
            cb_elem = FN_ELEM(lambda this, hwnd, out_p: setattr(out_p.contents, "value", root_ptr_val) or 0)
            cb_walk = FN_WALK(lambda this, out_p: setattr(out_p.contents, "value", ctypes.cast(p_obj_w, ctypes.c_void_p).value) or 0)

            vtbl_auto[6] = ctypes.cast(cb_elem, ctypes.c_void_p).value
            vtbl_auto[14] = ctypes.cast(cb_walk, ctypes.c_void_p).value
            p_obj_a = ctypes.pointer(ctypes.cast(ctypes.pointer(vtbl_auto), ctypes.c_void_p))

            with patch("ctypes.windll.ole32.CoCreateInstance") as mock_cc:
                mock_cc.side_effect = lambda rclsid, pUnk, ctx, riid, ppv: setattr(ppv._obj, "value", ctypes.cast(p_obj_a, ctypes.c_void_p).value) or 0

                with patch.object(tool, "_extract_element_data", return_value={"depth": 0, "name": "MockElem", "control_type": "button", "automation_id": "", "class_name": "", "is_enabled": True, "is_keyboard_focusable": False, "is_password": False, "is_offscreen": False}):
                    elems, meta = tool._traverse_control_tree(hwnd=0x1234, max_depth=2, max_elements=10)

                # Both child1 and bad_sib were released!
                self.assertIn(child1_val, released_ptrs)
                self.assertIn(bad_sib_val, released_ptrs)
                self.assertIn(root_ptr_val, released_ptrs)

    def test_multiple_siblings_each_released_once(self):
        """Verify multiple siblings (child 1, child 2, child 3) are each released exactly once."""
        tool = InspectWindowUIElementsTool()
        released_ptrs = []

        def track_release(ptr):
            if ptr and getattr(ptr, "value", None):
                released_ptrs.append(ptr.value)
                ptr.value = None

        with patch("exocortex.tools.windows.ui_accessibility._safe_release", side_effect=track_release):
            siblings = [0xDD01, 0xDD02, 0xDD03]

            FN_FC = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))
            FN_SIB = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))

            def fake_get_first_child(this, current, out_p):
                out_p.contents.value = siblings[0]
                return 0

            def fake_get_next_sibling(this, current, out_p):
                if current == siblings[0]:
                    out_p.contents.value = siblings[1]
                elif current == siblings[1]:
                    out_p.contents.value = siblings[2]
                else:
                    out_p.contents.value = 0
                return 0

            vtbl_walker = (ctypes.c_void_p * 10)()
            cb_fc = FN_FC(fake_get_first_child)
            cb_sib = FN_SIB(fake_get_next_sibling)
            vtbl_walker[4] = ctypes.cast(cb_fc, ctypes.c_void_p).value
            vtbl_walker[6] = ctypes.cast(cb_sib, ctypes.c_void_p).value
            p_obj_w = ctypes.pointer(ctypes.cast(ctypes.pointer(vtbl_walker), ctypes.c_void_p))

            vtbl_auto = (ctypes.c_void_p * 65)()
            FN_PUT = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, wintypes.DWORD)
            cb_put = FN_PUT(lambda this, val: 0)
            vtbl_auto[61] = ctypes.cast(cb_put, ctypes.c_void_p).value
            vtbl_auto[63] = ctypes.cast(cb_put, ctypes.c_void_p).value

            FN_ELEM = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, wintypes.HWND, ctypes.POINTER(ctypes.c_void_p))
            FN_WALK = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))

            root_ptr_val = 0xDD00
            cb_elem = FN_ELEM(lambda this, hwnd, out_p: setattr(out_p.contents, "value", root_ptr_val) or 0)
            cb_walk = FN_WALK(lambda this, out_p: setattr(out_p.contents, "value", ctypes.cast(p_obj_w, ctypes.c_void_p).value) or 0)

            vtbl_auto[6] = ctypes.cast(cb_elem, ctypes.c_void_p).value
            vtbl_auto[14] = ctypes.cast(cb_walk, ctypes.c_void_p).value
            p_obj_a = ctypes.pointer(ctypes.cast(ctypes.pointer(vtbl_auto), ctypes.c_void_p))

            with patch("ctypes.windll.ole32.CoCreateInstance") as mock_cc:
                mock_cc.side_effect = lambda rclsid, pUnk, ctx, riid, ppv: setattr(ppv._obj, "value", ctypes.cast(p_obj_a, ctypes.c_void_p).value) or 0

                with patch.object(tool, "_extract_element_data", return_value={"depth": 0, "name": "MockElem", "control_type": "button", "automation_id": "", "class_name": "", "is_enabled": True, "is_keyboard_focusable": False, "is_password": False, "is_offscreen": False}):
                    elems, meta = tool._traverse_control_tree(hwnd=0x1234, max_depth=1, max_elements=10)

                # Root + 3 children collected
                self.assertEqual(len(elems), 4)

                # Verify each sibling is in released_ptrs EXACTLY ONCE
                for sib in siblings:
                    self.assertEqual(released_ptrs.count(sib), 1, f"Pointer {hex(sib)} was not released exactly once!")
                self.assertEqual(released_ptrs.count(root_ptr_val), 1)

    def test_early_termination_at_max_elements_releases_pointer(self):
        """When traversal hits max_elements early, current child pointer is cleanly released."""
        tool = InspectWindowUIElementsTool()
        released_ptrs = []

        def track_release(ptr):
            if ptr and getattr(ptr, "value", None):
                released_ptrs.append(ptr.value)
                ptr.value = None

        with patch("exocortex.tools.windows.ui_accessibility._safe_release", side_effect=track_release):
            child1 = 0xEE01
            child2 = 0xEE02

            FN_FC = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))
            FN_SIB = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))

            cb_fc = FN_FC(lambda this, current, out_p: setattr(out_p.contents, "value", child1) or 0)
            cb_sib = FN_SIB(lambda this, current, out_p: setattr(out_p.contents, "value", child2) or 0)

            vtbl_walker = (ctypes.c_void_p * 10)()
            vtbl_walker[4] = ctypes.cast(cb_fc, ctypes.c_void_p).value
            vtbl_walker[6] = ctypes.cast(cb_sib, ctypes.c_void_p).value
            p_obj_w = ctypes.pointer(ctypes.cast(ctypes.pointer(vtbl_walker), ctypes.c_void_p))

            vtbl_auto = (ctypes.c_void_p * 65)()
            FN_PUT = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, wintypes.DWORD)
            cb_put = FN_PUT(lambda this, val: 0)
            vtbl_auto[61] = ctypes.cast(cb_put, ctypes.c_void_p).value
            vtbl_auto[63] = ctypes.cast(cb_put, ctypes.c_void_p).value

            FN_ELEM = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, wintypes.HWND, ctypes.POINTER(ctypes.c_void_p))
            FN_WALK = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))

            root_ptr_val = 0xEE00
            cb_elem = FN_ELEM(lambda this, hwnd, out_p: setattr(out_p.contents, "value", root_ptr_val) or 0)
            cb_walk = FN_WALK(lambda this, out_p: setattr(out_p.contents, "value", ctypes.cast(p_obj_w, ctypes.c_void_p).value) or 0)

            vtbl_auto[6] = ctypes.cast(cb_elem, ctypes.c_void_p).value
            vtbl_auto[14] = ctypes.cast(cb_walk, ctypes.c_void_p).value
            p_obj_a = ctypes.pointer(ctypes.cast(ctypes.pointer(vtbl_auto), ctypes.c_void_p))

            with patch("ctypes.windll.ole32.CoCreateInstance") as mock_cc:
                mock_cc.side_effect = lambda rclsid, pUnk, ctx, riid, ppv: setattr(ppv._obj, "value", ctypes.cast(p_obj_a, ctypes.c_void_p).value) or 0

                with patch.object(tool, "_extract_element_data", return_value={"depth": 0, "name": "MockElem", "control_type": "button", "automation_id": "", "class_name": "", "is_enabled": True, "is_keyboard_focusable": False, "is_password": False, "is_offscreen": False}):
                    # max_elements=2: root (1) + child1 (2) -> stops before child2
                    elems, meta = tool._traverse_control_tree(hwnd=0x1234, max_depth=1, max_elements=2)

                self.assertEqual(len(elems), 2)
                self.assertIn(child1, released_ptrs)
                self.assertIn(root_ptr_val, released_ptrs)


if __name__ == "__main__":
    unittest.main()
