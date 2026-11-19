"""
Unit tests for Deterministic Fast-Path Router in ExoCortex.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from exocortex.agent import ExoCortexAgent
from exocortex.execution.fast_path import (
    CANONICAL_FAST_PATH_MAP,
    FAST_PATH_ELIGIBLE_TOOLS,
    FAST_PATH_INTENT_MAP,
    FastPathRouter,
    normalize_fast_path_query,
)
from exocortex.execution.state import ActionType, ExecutionStatus
from exocortex.execution.trace import ExecutionTrace
from exocortex.slm.mock_provider import MockSLMProvider
from exocortex.tools.base import BaseTool, PermissionLevel, ToolParameter, ToolResult
from exocortex.tools.registry import ToolRegistry


class DummyMockTool(BaseTool):
    def __init__(
        self,
        name: str,
        permission: PermissionLevel = PermissionLevel.SAFE,
        fail: bool = False,
        output_text: str = "Dummy Output",
        data: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.description = f"Dummy {name}"
        self.permission_level = permission
        self.parameters = []
        self.fail = fail
        self.output_text = output_text
        self.data = data
        self.call_count = 0

    def run(self, **kwargs) -> ToolResult:
        self.call_count += 1
        if self.fail:
            return ToolResult(
                success=False,
                output="",
                error=f"{self.name} simulated failure",
            )
        return ToolResult(
            success=True,
            output=self.output_text,
            data=self.data,
        )


class TestFastPathRouter(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()
        self.windows_tool = DummyMockTool("list_open_windows", output_text="Visible Desktop Windows (1 found): Notepad")
        self.process_tool = DummyMockTool("list_processes", output_text="Running Processes (1 found): notepad.exe")
        self.sysinfo_tool = DummyMockTool("system_info", output_text="OS: Windows 11\nMemory: 4.0 GB free / 16.0 GB total")
        self.health_tool = DummyMockTool("health_check", output_text="ExoCortex System Status: HEALTHY (Version: 0.1.0)")
        self.registry.register(self.windows_tool)
        self.registry.register(self.process_tool)
        self.registry.register(self.sysinfo_tool)
        self.registry.register(self.health_tool)
        self.router = FastPathRouter(tool_registry=self.registry, enabled=True)

    def test_normalization_and_punctuation(self):
        cases = [
            ("  What application windows are open?  ", "what application windows are open"),
            ("list open windows.", "list open windows"),
            ("WHAT WINDOWS ARE OPEN???", "what windows are open"),
            ("  show running processes!  ", "show running processes"),
            ("\"List processes;\"", "list processes"),
            ("What processes are running:", "what processes are running"),
            ("  System   Info?!  ", "system info"),
            ("`Check   system   health.`", "check system health"),
            ("  show   me   open   windows ? ", "show me open windows"),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(normalize_fast_path_query(raw), expected)

    def test_existing_and_newly_approved_canonical_matches(self):
        windows_queries = [
            # Existing
            "list open windows",
            "list windows",
            "what application windows are open",
            "what windows are open",
            "show open windows",
            "show windows",
            "get open windows",
            "get windows",
            # Newly approved (Task 8.8.1)
            "show me open windows",
            "what windows are currently open",
            "what desktop windows are open",
            "list desktop windows",
            "display open windows",
            "get open window list",
            # Variations
            "What application windows are open?",
            "  SHOW ME OPEN WINDOWS!  ",
        ]
        for q in windows_queries:
            with self.subTest(query=q):
                self.assertEqual(self.router.match(q), "list_open_windows")

        process_queries = [
            # Existing
            "list processes",
            "list running processes",
            "show processes",
            "show running processes",
            "get processes",
            "get running processes",
            "what processes are running",
            # Newly approved (Task 8.8.1)
            "display running processes",
            "show process list",
            "show me running processes",
            "what processes are currently running",
            "list active processes",
            "get process list",
            # Variations
            "LIST RUNNING PROCESSES!",
            "  Display   Running   Processes? ",
        ]
        for q in process_queries:
            with self.subTest(query=q):
                self.assertEqual(self.router.match(q), "list_processes")

        sysinfo_queries = [
            "system info",
            "system information",
            "show system info",
            "show system information",
            "get system info",
            "get system information",
            "display system info",
            "display system information",
            "  SYSTEM INFO? ",
            "\"Show system information.\"",
        ]
        for q in sysinfo_queries:
            with self.subTest(query=q):
                self.assertEqual(self.router.match(q), "system_info")

        health_queries = [
            "health check",
            "check system health",
            "run health check",
            "system health check",
            "check exocortex health",
            "exocortex health check",
            "  Health Check!  ",
            "Check   system   health?",
        ]
        for q in health_queries:
            with self.subTest(query=q):
                self.assertEqual(self.router.match(q), "health_check")

    def test_every_map_entry_resolves_to_eligible_tool(self):
        self.assertEqual(
            FAST_PATH_ELIGIBLE_TOOLS,
            {"list_open_windows", "list_processes", "system_info", "health_check"},
        )
        for phrase, tool_name in CANONICAL_FAST_PATH_MAP.items():
            with self.subTest(phrase=phrase):
                self.assertIn(tool_name, FAST_PATH_ELIGIBLE_TOOLS)
                self.assertIn(tool_name, FAST_PATH_INTENT_MAP)
                self.assertEqual(self.router.match(phrase), tool_name)

    def test_unknown_ambiguous_rejected_and_unrelated_fall_through(self):
        non_matching_queries = [
            "Hello who are you",
            "Open Notepad",
            "Search Python docs",
            "What is the weather in Tokyo",
            "Tell me about open source software",
            "The window is broken",
            "Process this file",
            "Check stuff",
            "Fix windows",
            # Rejected candidate phrases (unregistered / ambiguous)
            "what tools are available",
            "list available tools",
            "list directory",
            "list files",
            "show files",
        ]
        for q in non_matching_queries:
            with self.subTest(query=q):
                self.assertIsNone(self.router.match(q))
                self.assertIsNone(self.router.try_execute(q))

    def test_compound_requests_fall_through(self):
        compound_queries = [
            "list open windows and show system info",
            "health check and list processes",
            "show processes then show windows",
            "system info; health check",
            "list windows and close notepad",
        ]
        for q in compound_queries:
            with self.subTest(query=q):
                self.assertIsNone(self.router.match(q))
                self.assertIsNone(self.router.try_execute(q))

    def test_mutating_requests_fall_through(self):
        mutating_queries = [
            "Close all open windows",
            "Kill running process",
            "Terminate notepad.exe",
            "Delete system info",
            "Create directory test",
            "Launch notepad",
            "Open window Notepad",
        ]
        for q in mutating_queries:
            with self.subTest(query=q):
                self.assertIsNone(self.router.match(q))
                self.assertIsNone(self.router.try_execute(q))

    def test_parameterized_requests_fall_through(self):
        # Unsupported parameterized requests that must fall through to SLM
        unsupported = [
            "Inspect the UI elements in the Windows Terminal window.",
            "Inspect window matching 'Visual Studio Code'",
            "Find process with pid 1234",
            "system info full detail",
            "Is some random application running?",
            "Inspect process with high CPU",
        ]
        for q in unsupported:
            with self.subTest(query=q):
                self.assertIsNone(self.router.match(q))
                self.assertIsNone(self.router.try_execute(q))

        # Application-state queries do not match canonical zero-arg router.match()
        app_state_queries = [
            "Is Windows Terminal open?",
            "Is notepad running?",
            "Check if Chrome is open",
        ]
        for q in app_state_queries:
            with self.subTest(query=q):
                self.assertIsNone(self.router.match(q))

    def test_successful_execution_trace_structure(self):
        cases = [
            ("What application windows are open?", "list_open_windows", "window_enumeration", self.windows_tool, "Visible Desktop Windows"),
            ("Show process list", "list_processes", "process_inspection", self.process_tool, "Running Processes"),
            ("Show system information", "system_info", "system_information", self.sysinfo_tool, "Windows 11"),
            ("Check system health", "health_check", "system_health_check", self.health_tool, "HEALTHY"),
        ]
        for query, expected_tool, expected_intent, mock_tool, expected_substr in cases:
            with self.subTest(query=query):
                prev_count = mock_tool.call_count
                trace = self.router.try_execute(query)
                self.assertIsNotNone(trace)
                self.assertEqual(trace.status, ExecutionStatus.COMPLETED)
                self.assertEqual(mock_tool.call_count, prev_count + 1)
                self.assertIn(expected_substr, trace.final_response)
                self.assertIsNone(trace.error_message)
                self.assertEqual(trace.total_steps, 1)
                self.assertEqual(trace.slm_info.get("llm_calls"), 0)

                step = trace.steps[0]
                self.assertEqual(step.tool_name, expected_tool)
                self.assertEqual(step.action_type, ActionType.TOOL_CALL)
                self.assertEqual(step.status, ExecutionStatus.COMPLETED)
                self.assertTrue(step.tool_result.success)
                self.assertEqual(step.decision.intent, expected_intent)
                self.assertFalse(step.decision.requires_confirmation)

    def test_dispatches_via_tool_registry_execute(self):
        with patch.object(self.registry, "execute", wraps=self.registry.execute) as spy_exec:
            trace = self.router.try_execute("system info")
            self.assertIsNotNone(trace)
            self.assertEqual(trace.status, ExecutionStatus.COMPLETED)
            spy_exec.assert_called_once_with("system_info", {})

    def test_tool_failure_does_not_report_success(self):
        registry = ToolRegistry()
        failing_tool = DummyMockTool("list_open_windows", fail=True)
        registry.register(failing_tool)
        router = FastPathRouter(tool_registry=registry, enabled=True)

        trace = router.try_execute("list open windows")
        self.assertIsNotNone(trace)
        self.assertEqual(trace.status, ExecutionStatus.ERROR)
        self.assertIsNone(trace.final_response)
        self.assertIn("simulated failure", trace.error_message)
        self.assertFalse(trace.steps[0].tool_result.success)

    def test_empty_tool_output_does_not_report_success(self):
        for empty_val in ("", "   \n\t "):
            with self.subTest(empty_val=repr(empty_val)):
                registry = ToolRegistry()
                empty_tool = DummyMockTool("system_info", output_text=empty_val)
                registry.register(empty_tool)
                router = FastPathRouter(tool_registry=registry, enabled=True)

                trace = router.try_execute("system info")
                self.assertIsNotNone(trace)
                self.assertEqual(trace.status, ExecutionStatus.ERROR)
                self.assertIsNone(trace.final_response)
                self.assertIn("empty output", trace.error_message)
                self.assertEqual(trace.steps[0].status, ExecutionStatus.ERROR)
                self.assertIsNone(trace.steps[0].decision.direct_response)

    def test_unregistered_or_non_safe_tool_falls_through(self):
        # 1. Non-SAFE CONFIRMATION_REQUIRED tool (callback must not be called/bypassed)
        cb = MagicMock(return_value=True)
        registry = ToolRegistry(confirmation_callback=cb)
        confirm_tool = DummyMockTool("list_processes", permission=PermissionLevel.CONFIRMATION_REQUIRED)
        registry.register(confirm_tool)
        router = FastPathRouter(tool_registry=registry, enabled=True)
        self.assertIsNone(router.try_execute("list processes"))
        cb.assert_not_called()
        self.assertEqual(confirm_tool.call_count, 0)

        # 2. Non-SAFE RESTRICTED tool
        registry_r = ToolRegistry()
        restricted_tool = DummyMockTool("health_check", permission=PermissionLevel.RESTRICTED)
        registry_r.register(restricted_tool)
        router_r = FastPathRouter(tool_registry=registry_r, enabled=True)
        self.assertIsNone(router_r.try_execute("health check"))
        self.assertEqual(restricted_tool.call_count, 0)

        # 3. Tool not in registry
        empty_registry = ToolRegistry()
        router2 = FastPathRouter(tool_registry=empty_registry, enabled=True)
        self.assertIsNone(router2.try_execute("list open windows"))
        self.assertIsNone(router2.try_execute("system info"))
        self.assertIsNone(router2.try_execute("health check"))

    def test_disabled_router_always_returns_none(self):
        disabled_router = FastPathRouter(tool_registry=self.registry, enabled=False)
        for q in ("list open windows", "system info", "health check", "show process list"):
            with self.subTest(q=q):
                self.assertIsNone(disabled_router.match(q))
                self.assertIsNone(disabled_router.try_execute(q))

    def test_agent_integration_fast_path_zero_llm_calls(self):
        mock_slm = MagicMock(wraps=MockSLMProvider())
        agent = ExoCortexAgent(
            slm_provider=mock_slm,
            tool_registry=self.registry,
            enable_fast_path=True,
        )

        # Canonical matches across all 4 eligible tools trigger 0 LLM calls
        canonical_cases = [
            ("What application windows are open?", "window_enumeration", "Visible Desktop Windows"),
            ("Show process list", "process_inspection", "Running Processes"),
            ("System info", "system_information", "Windows 11"),
            ("Check system health", "system_health_check", "HEALTHY"),
        ]
        for query, expected_intent, expected_substr in canonical_cases:
            with self.subTest(query=query):
                res = agent.run(query)
                self.assertTrue(res.success)
                self.assertIn(expected_substr, res.response_text)
                self.assertEqual(res.decision.intent, expected_intent)
                self.assertEqual(res.slm_info.get("provider"), "FastPathRouter")
                self.assertEqual(res.slm_info.get("llm_calls"), 0)

        mock_slm.generate.assert_not_called()

        # Non-fast-path queries fall through to SLM
        for fallthrough_q in ("Hello, who are you?", "What tools are available?", "List windows and system info"):
            with self.subTest(fallthrough_q=fallthrough_q):
                calls_before = mock_slm.generate.call_count
                res_slm = agent.run(fallthrough_q)
                self.assertTrue(res_slm.success)
                self.assertEqual(res_slm.slm_info.get("provider"), "MockSLMProvider")
                self.assertGreater(mock_slm.generate.call_count, calls_before)

    def test_default_registry_eligible_tools_metadata_and_system_info_live(self):
        from exocortex.tools.registry import get_default_tool_registry

        real_registry = get_default_tool_registry()
        for tool_name in FAST_PATH_ELIGIBLE_TOOLS:
            with self.subTest(tool_name=tool_name):
                tool = real_registry.get_tool(tool_name)
                self.assertIsNotNone(tool)
                self.assertEqual(tool.permission_level, PermissionLevel.SAFE)
                valid, err = tool.validate_arguments({})
                self.assertTrue(valid, msg=err)

        router = FastPathRouter(tool_registry=real_registry, enabled=True)
        trace = router.try_execute("show system information")
        self.assertIsNotNone(trace)
        self.assertEqual(trace.status, ExecutionStatus.COMPLETED)
        self.assertIn("OS:", trace.final_response)
        self.assertEqual(trace.slm_info.get("llm_calls"), 0)


class TestApplicationStateRouting(unittest.TestCase):
    """
    Dedicated test suite for M8 Task 8.8.2: Parameterized Application-State Routing.
    
    Verifies:
    - Supported request patterns
    - Approved application aliases
    - Normalization (case, quotes, whitespace, punctuation)
    - Distinguishing window states and process states
    - Negative tests (mutating verbs, compound queries, ambiguous aliases, unknown applications)
    - Permission checks and zero LLM calls
    """

    def setUp(self):
        self.registry = ToolRegistry()
        self.windows_tool = DummyMockTool(
            "list_open_windows",
            output_text="No visible application windows found on the desktop.",
            data={"returned_count": 0, "windows": []},
        )
        self.process_tool = DummyMockTool(
            "list_processes",
            output_text="Top 0 Running Processes",
            data={"total_processes_scanned": 0, "returned_count": 0, "processes": []},
        )
        self.registry.register(self.windows_tool)
        self.registry.register(self.process_tool)
        self.router = FastPathRouter(tool_registry=self.registry, enabled=True)

    def test_every_supported_phrase_pattern(self):
        patterns = [
            ("is notepad running?", "running"),
            ("is notepad open?", "open"),
            ("is notepad active?", "active"),
            ("check if notepad is running", "running"),
            ("check if notepad is open", "open"),
            ("check if notepad is active", "active"),
            ("do i have notepad open?", "open"),
            ("do i have notepad running?", "running"),
            ("can you check if notepad is running", "running"),
            ("can you check if notepad is open", "open"),
            ("please check if notepad is running", "running"),
            ("please check if notepad is open", "open"),
            ("check whether notepad is running", "running"),
            ("check whether notepad is open", "open"),
            ("is notepad currently running", "running"),
            ("is notepad currently open", "open"),
            ("is notepad running right now", "running"),
        ]
        for query, expected_target_state in patterns:
            with self.subTest(query=query):
                res = self.router.match_application_state(query)
                self.assertIsNotNone(res, msg=f"Failed to match pattern: {query}")
                app_meta, state = res
                self.assertEqual(app_meta.display_name, "Notepad")
                self.assertEqual(state, expected_target_state)

    def test_every_approved_application_alias(self):
        approved_cases = [
            ("is notepad open?", "Notepad"),
            ("is windows notepad open?", "Notepad"),
            ("check if chrome is running", "Google Chrome"),
            ("check if google chrome is running", "Google Chrome"),
            ("is firefox open?", "Mozilla Firefox"),
            ("do i have mozilla firefox open?", "Mozilla Firefox"),
            ("is spotify running?", "Spotify"),
            ("is vscode open?", "Visual Studio Code"),
            ("is vs code open?", "Visual Studio Code"),
            ("is visual studio code active?", "Visual Studio Code"),
            ("is windows terminal running?", "Windows Terminal"),
        ]
        for query, expected_display in approved_cases:
            with self.subTest(query=query):
                res = self.router.match_application_state(query)
                self.assertIsNotNone(res, msg=f"Failed to match alias: {query}")
                app_meta, _ = res
                self.assertEqual(app_meta.display_name, expected_display)

    def test_case_and_whitespace_normalization(self):
        cases = [
            ("  Is   NOTEPAD   Open?  ", "Notepad", "open"),
            ("is \"Google Chrome\" RUNNING???", "Google Chrome", "running"),
            ("  CHECK IF   'vs code'   is   Active ", "Visual Studio Code", "active"),
            ("do I have \"Spotify\" open?", "Spotify", "open"),
            ("is 'firefox' active?", "Mozilla Firefox", "active"),
            ("  `check whether Windows Terminal is running.` ", "Windows Terminal", "running"),
        ]
        for raw, expected_display, expected_state in cases:
            with self.subTest(raw=raw):
                res = self.router.match_application_state(raw)
                self.assertIsNotNone(res, msg=f"Failed normalization for: {raw}")
                app_meta, state = res
                self.assertEqual(app_meta.display_name, expected_display)
                self.assertEqual(state, expected_state)

    def test_visible_window_detection_non_active_and_active(self):
        # 1. Non-active visible window
        self.windows_tool.data = {
            "returned_count": 1,
            "windows": [
                {"title": "notes.txt - Notepad", "process_name": "notepad.exe", "is_active": False}
            ],
        }
        self.windows_tool.output_text = "Visible Desktop Windows (1 found):\n1. \"notes.txt - Notepad\" (notepad.exe)"
        self.process_tool.data = {
            "returned_count": 1,
            "processes": [{"name": "notepad.exe", "pid": 1234, "memory_mb": 25.0}],
        }
        self.process_tool.output_text = "Top 1 Running Processes:\n  1234 notepad.exe 25.0 MB"

        # Check 'open'
        trace_open = self.router.try_execute("is notepad open?")
        self.assertIsNotNone(trace_open)
        self.assertEqual(trace_open.status, ExecutionStatus.COMPLETED)
        self.assertEqual(
            trace_open.final_response,
            'Yes, Notepad is open with a visible window: "notes.txt - Notepad".',
        )

        # Check 'running'
        trace_running = self.router.try_execute("is notepad running?")
        self.assertIsNotNone(trace_running)
        self.assertEqual(trace_running.status, ExecutionStatus.COMPLETED)
        self.assertEqual(
            trace_running.final_response,
            'Yes, Notepad is running (PID: 1234).',
        )
        self.assertEqual(len(trace_running.steps), 1)
        self.assertEqual(trace_running.steps[0].tool_name, "list_processes")

        # Check 'active' (when not active foreground window)
        trace_active = self.router.try_execute("is notepad active?")
        self.assertIsNotNone(trace_active)
        self.assertEqual(trace_active.status, ExecutionStatus.COMPLETED)
        self.assertEqual(
            trace_active.final_response,
            'No, Notepad is open with a visible window ("notes.txt - Notepad"), but it is not the active foreground window.',
        )

        # 2. Active visible window
        self.windows_tool.data["windows"][0]["is_active"] = True
        trace_active2 = self.router.try_execute("is notepad active?")
        self.assertIsNotNone(trace_active2)
        self.assertEqual(trace_active2.status, ExecutionStatus.COMPLETED)
        self.assertEqual(
            trace_active2.final_response,
            'Yes, Notepad is currently the active foreground window: "notes.txt - Notepad".',
        )

        trace_open_act = self.router.try_execute("is notepad open?")
        self.assertIsNotNone(trace_open_act)
        self.assertEqual(
            trace_open_act.final_response,
            'Yes, Notepad is open with an active visible window: "notes.txt - Notepad".',
        )

    def test_process_only_background_detection_and_semantic_distinction(self):
        """
        Verify semantic rule:
        For an 'is <app> open?' question, a visible window is the strongest evidence of being open.
        If Chrome has background processes but no visible window, do not claim that a Chrome window is open.
        Report that Chrome is running in the background but no visible Chrome window was found.
        """
        # No visible window
        self.windows_tool.data = {"returned_count": 0, "windows": []}
        self.windows_tool.output_text = "No visible application windows found on the desktop."

        # Background process exists
        self.process_tool.data = {
            "returned_count": 1,
            "processes": [{"name": "chrome.exe", "pid": 4567, "memory_mb": 420.0}],
        }
        self.process_tool.output_text = "Top 1 Running Processes:\n  4567 chrome.exe 420.0 MB"

        # Check 'open' -> reports running in background, no visible window found
        trace_open = self.router.try_execute("is chrome open?")
        self.assertIsNotNone(trace_open)
        self.assertEqual(trace_open.status, ExecutionStatus.COMPLETED)
        self.assertEqual(
            trace_open.final_response,
            "No visible Google Chrome window was found, but Google Chrome is currently running in the background.",
        )

        # Check 'running' -> reports running via process short-circuit
        trace_running = self.router.try_execute("is chrome running?")
        self.assertIsNotNone(trace_running)
        self.assertEqual(trace_running.status, ExecutionStatus.COMPLETED)
        self.assertEqual(trace_running.final_response, "Yes, Google Chrome is running (PID: 4567).")
        self.assertEqual(len(trace_running.steps), 1)
        self.assertEqual(trace_running.steps[0].tool_name, "list_processes")

        # Check 'active' -> reports running in background, not active
        trace_active = self.router.try_execute("is chrome active?")
        self.assertIsNotNone(trace_active)
        self.assertEqual(trace_active.status, ExecutionStatus.COMPLETED)
        self.assertEqual(
            trace_active.final_response,
            "No, Google Chrome is running in the background, but has no visible window and is not active.",
        )

    def test_not_running_detection(self):
        # Neither visible window nor running process
        self.windows_tool.data = {"returned_count": 0, "windows": []}
        self.windows_tool.output_text = "No visible application windows found on the desktop."
        self.process_tool.data = {"returned_count": 0, "processes": []}
        self.process_tool.output_text = "Top 0 Running Processes."

        trace_open = self.router.try_execute("is spotify open?")
        self.assertIsNotNone(trace_open)
        self.assertEqual(trace_open.status, ExecutionStatus.COMPLETED)
        self.assertEqual(
            trace_open.final_response,
            "No, Spotify is not open (no matching visible window or running process was found).",
        )

        trace_running = self.router.try_execute("is spotify running?")
        self.assertIsNotNone(trace_running)
        self.assertEqual(trace_running.status, ExecutionStatus.COMPLETED)
        self.assertEqual(
            trace_running.final_response,
            "No, Spotify is not running (no matching process or window was found).",
        )

        trace_active = self.router.try_execute("is spotify active?")
        self.assertIsNotNone(trace_active)
        self.assertEqual(trace_active.status, ExecutionStatus.COMPLETED)
        self.assertEqual(
            trace_active.final_response,
            "No, Spotify is not active and is not running.",
        )

    def test_tool_execution_failures_do_not_report_success(self):
        # 1. list_open_windows fails
        failing_win_tool = DummyMockTool("list_open_windows", fail=True)
        reg1 = ToolRegistry()
        reg1.register(failing_win_tool)
        reg1.register(self.process_tool)
        router1 = FastPathRouter(tool_registry=reg1, enabled=True)

        trace1 = router1.try_execute("is notepad open?")
        self.assertIsNotNone(trace1)
        self.assertEqual(trace1.status, ExecutionStatus.ERROR)
        self.assertIsNone(trace1.final_response)
        self.assertIn("simulated failure", trace1.error_message)

        # 2. list_processes fails
        failing_proc_tool = DummyMockTool("list_processes", fail=True)
        reg2 = ToolRegistry()
        reg2.register(self.windows_tool)
        reg2.register(failing_proc_tool)
        router2 = FastPathRouter(tool_registry=reg2, enabled=True)

        trace2 = router2.try_execute("is notepad running?")
        self.assertIsNotNone(trace2)
        self.assertEqual(trace2.status, ExecutionStatus.ERROR)
        self.assertIsNone(trace2.final_response)
        self.assertIn("simulated failure", trace2.error_message)

    def test_empty_tool_output_does_not_report_success(self):
        # Empty windows output
        empty_win_tool = DummyMockTool("list_open_windows", output_text="   \n  ")
        reg = ToolRegistry()
        reg.register(empty_win_tool)
        reg.register(self.process_tool)
        router = FastPathRouter(tool_registry=reg, enabled=True)

        trace = router.try_execute("is notepad open?")
        self.assertIsNotNone(trace)
        self.assertEqual(trace.status, ExecutionStatus.ERROR)
        self.assertIsNone(trace.final_response)
        self.assertIn("empty output", trace.error_message)

    def test_unknown_application_names_fall_through(self):
        unknown_queries = [
            "Is some random application running?",
            "Is slack open?",
            "check if vlc is running",
            "do i have excel open?",
            "is python running?",
        ]
        for q in unknown_queries:
            with self.subTest(query=q):
                self.assertIsNone(self.router.match_application_state(q))
                self.assertIsNone(self.router.try_execute(q))

    def test_ambiguous_application_aliases_fall_through(self):
        ambiguous_queries = [
            "is terminal open?",
            "is browser running?",
            "check if web browser is open",
            "is editor active?",
            "is code running?",
            "is app open?",
            "is application running?",
        ]
        for q in ambiguous_queries:
            with self.subTest(query=q):
                self.assertIsNone(self.router.match_application_state(q))
                self.assertIsNone(self.router.try_execute(q))

    def test_compound_and_mutating_requests_rejected(self):
        rejected_queries = [
            "Is Chrome open and close it",
            "Kill Chrome",
            "Open Notepad",
            "Is notepad running and terminate it",
            "Launch Spotify",
            "Is Chrome open; close it",
            "Start Visual Studio Code",
            "is notepad open then close it",
        ]
        for q in rejected_queries:
            with self.subTest(query=q):
                self.assertIsNone(self.router.match_application_state(q))
                self.assertIsNone(self.router.try_execute(q))

    def test_permission_and_confirmation_boundaries(self):
        cb = MagicMock(return_value=True)

        # 1. list_open_windows requires confirmation
        reg1 = ToolRegistry(confirmation_callback=cb)
        confirm_win_tool = DummyMockTool("list_open_windows", permission=PermissionLevel.CONFIRMATION_REQUIRED)
        reg1.register(confirm_win_tool)
        reg1.register(self.process_tool)
        router1 = FastPathRouter(tool_registry=reg1, enabled=True)
        self.assertIsNone(router1.try_execute("is notepad open?"))
        cb.assert_not_called()
        self.assertEqual(confirm_win_tool.call_count, 0)

        # 2. list_processes is RESTRICTED
        reg2 = ToolRegistry()
        reg2.register(self.windows_tool)
        restricted_proc_tool = DummyMockTool("list_processes", permission=PermissionLevel.RESTRICTED)
        reg2.register(restricted_proc_tool)
        router2 = FastPathRouter(tool_registry=reg2, enabled=True)
        self.assertIsNone(router2.try_execute("is notepad running?"))
        self.assertEqual(restricted_proc_tool.call_count, 0)

    def test_agent_integration_application_state_zero_llm_calls(self):
        mock_slm = MagicMock(wraps=MockSLMProvider())
        agent = ExoCortexAgent(
            slm_provider=mock_slm,
            tool_registry=self.registry,
            enable_fast_path=True,
        )

        # Setup mock window
        self.windows_tool.data = {
            "returned_count": 1,
            "windows": [{"title": "ExoCortex - Visual Studio Code", "process_name": "code.exe", "is_active": True}],
        }
        self.windows_tool.output_text = "Visible Desktop Windows (1 found):\n1. [ACTIVE] \"ExoCortex - Visual Studio Code\" (code.exe)"

        res = agent.run("Is VS Code open?")
        self.assertTrue(res.success)
        self.assertIn("Yes, Visual Studio Code is open with an active visible window", res.response_text)
        self.assertEqual(res.slm_info.get("provider"), "FastPathRouter")
        self.assertEqual(res.slm_info.get("llm_calls"), 0)
        mock_slm.generate.assert_not_called()

        # Non-fast-path general conversational query falls through to SLM
        res2 = agent.run("What is the capital of France?")
        self.assertTrue(res2.success)
        self.assertEqual(res2.slm_info.get("provider"), "MockSLMProvider")
        self.assertEqual(mock_slm.generate.call_count, 1)

    def test_window_identity_process_name_overrides_mismatching_title(self):
        """
        Verify identity-first matching:
        A matching title MUST NEVER override a known mismatching process name.
        """
        # 1. Notepad displaying 'chrome_flags.txt' is NOT Chrome
        self.windows_tool.data = {
            "returned_count": 1,
            "windows": [{"title": "chrome_flags.txt - Notepad", "process_name": "notepad.exe", "is_active": False}],
        }
        self.windows_tool.output_text = "Visible Desktop Windows (1 found):\n1. \"chrome_flags.txt - Notepad\" (notepad.exe)"
        self.process_tool.data = {"returned_count": 0, "processes": []}
        self.process_tool.output_text = "Top 0 Running Processes."

        trace_chrome = self.router.try_execute("is chrome open?")
        self.assertIsNotNone(trace_chrome)
        self.assertEqual(trace_chrome.status, ExecutionStatus.COMPLETED)
        self.assertEqual(
            trace_chrome.final_response,
            "No, Google Chrome is not open (no matching visible window or running process was found).",
        )

        # 2. Edge displaying 'How to Install Windows Terminal' is NOT Windows Terminal
        self.windows_tool.data = {
            "returned_count": 1,
            "windows": [{"title": "How to Install Windows Terminal - Microsoft Edge", "process_name": "msedge.exe", "is_active": False}],
        }
        self.windows_tool.output_text = "Visible Desktop Windows (1 found):\n1. \"How to Install Windows Terminal - Microsoft Edge\" (msedge.exe)"
        trace_wt = self.router.try_execute("is windows terminal open?")
        self.assertIsNotNone(trace_wt)
        self.assertEqual(
            trace_wt.final_response,
            "No, Windows Terminal is not open (no matching visible window or running process was found).",
        )

        # 3. Unrelated application displaying '- code' is NOT VS Code
        self.windows_tool.data = {
            "returned_count": 1,
            "windows": [{"title": "morse-code.py - Notepad", "process_name": "notepad.exe", "is_active": False}],
        }
        self.windows_tool.output_text = "Visible Desktop Windows (1 found):\n1. \"morse-code.py - Notepad\" (notepad.exe)"
        trace_vsc = self.router.try_execute("is vs code open?")
        self.assertIsNotNone(trace_vsc)
        self.assertEqual(
            trace_vsc.final_response,
            "No, Visual Studio Code is not open (no matching visible window or running process was found).",
        )

        # 4. Correct executable and valid title is accepted
        self.windows_tool.data = {
            "returned_count": 1,
            "windows": [{"title": "GitHub - Google Chrome", "process_name": "chrome.exe", "is_active": True}],
        }
        self.windows_tool.output_text = "Visible Desktop Windows (1 found):\n1. [ACTIVE] \"GitHub - Google Chrome\" (chrome.exe)"
        trace_real_chrome = self.router.try_execute("is chrome open?")
        self.assertIsNotNone(trace_real_chrome)
        self.assertEqual(
            trace_real_chrome.final_response,
            'Yes, Google Chrome is open with an active visible window: "GitHub - Google Chrome".',
        )

    def test_unknown_process_identity_conservative_fallback(self):
        """
        When process identity is genuinely unknown ('unknown', missing),
        title fallback applies only with conservative application-specific patterns.
        """
        self.process_tool.data = {"returned_count": 0, "processes": []}
        self.process_tool.output_text = "Top 0 Running Processes."

        # 1. Unknown process with generic title 'chrome_flags.txt' -> NOT Chrome
        self.windows_tool.data = {
            "returned_count": 1,
            "windows": [{"title": "chrome_flags.txt", "process_name": "unknown", "is_active": False}],
        }
        trace_generic = self.router.try_execute("is chrome open?")
        self.assertIsNotNone(trace_generic)
        self.assertEqual(
            trace_generic.final_response,
            "No, Google Chrome is not open (no matching visible window or running process was found).",
        )

        # 2. Unknown process with tightened Chrome title '- Google Chrome' -> IS Chrome
        self.windows_tool.data = {
            "returned_count": 1,
            "windows": [{"title": "New Tab - Google Chrome", "process_name": "unknown", "is_active": False}],
        }
        trace_chrome_fallback = self.router.try_execute("is chrome open?")
        self.assertIsNotNone(trace_chrome_fallback)
        self.assertEqual(
            trace_chrome_fallback.final_response,
            'Yes, Google Chrome is open with a visible window: "New Tab - Google Chrome".',
        )

        # 3. Unknown process with tightened Windows Terminal title -> IS Windows Terminal
        self.windows_tool.data = {
            "returned_count": 1,
            "windows": [{"title": "Windows Terminal", "process_name": "unknown", "is_active": False}],
        }
        trace_wt_fallback = self.router.try_execute("is windows terminal open?")
        self.assertIsNotNone(trace_wt_fallback)
        self.assertEqual(
            trace_wt_fallback.final_response,
            'Yes, Windows Terminal is open with a visible window: "Windows Terminal".',
        )

        # 4. Unknown process with bare 'Terminal' title -> NOT Windows Terminal
        self.windows_tool.data = {
            "returned_count": 1,
            "windows": [{"title": "Terminal", "process_name": "unknown", "is_active": False}],
        }
        trace_bare_terminal = self.router.try_execute("is windows terminal open?")
        self.assertIsNotNone(trace_bare_terminal)
        self.assertEqual(
            trace_bare_terminal.final_response,
            "No, Windows Terminal is not open (no matching visible window or running process was found).",
        )

    def test_complete_process_enumeration_below_top_20(self):
        """
        Verify that an application ranked well below the top 20 memory consumers
        is still completely detected by the router when all_processes=True is used.
        """
        # Create 35 dummy high-memory processes
        dummy_procs = [
            {"name": f"heavy_proc_{i}.exe", "pid": 1000 + i, "memory_mb": 500.0 - i}
            for i in range(25)
        ]
        # Append target application at rank #26 (below top 20)
        dummy_procs.append({"name": "notepad.exe", "pid": 9999, "memory_mb": 12.0})

        self.process_tool.data = {
            "total_processes_scanned": len(dummy_procs),
            "returned_count": len(dummy_procs),
            "processes": dummy_procs,
        }
        self.process_tool.output_text = "Processes output..."

        trace = self.router.try_execute("is notepad running?")
        self.assertIsNotNone(trace)
        self.assertEqual(trace.status, ExecutionStatus.COMPLETED)
        self.assertEqual(trace.final_response, "Yes, Notepad is running (PID: 9999).")
        self.assertEqual(len(trace.steps), 1)

    def test_observation_short_circuit_policies(self):
        """
        Verify intent-specific observation short-circuiting:
        - 'open' short-circuits process listing when visible window is found.
        - 'active' short-circuits process listing when visible window is found.
        - 'running' short-circuits window listing when matching process is found.
        - 'running' falls back to window listing when process is not found.
        """
        # 1. 'open' short-circuits process query
        self.windows_tool.call_count = 0
        self.process_tool.call_count = 0
        self.windows_tool.data = {
            "returned_count": 1,
            "windows": [{"title": "notes.txt - Notepad", "process_name": "notepad.exe", "is_active": False}],
        }
        trace_open = self.router.try_execute("is notepad open?")
        self.assertIsNotNone(trace_open)
        self.assertEqual(len(trace_open.steps), 1)
        self.assertEqual(trace_open.steps[0].tool_name, "list_open_windows")
        self.assertEqual(self.windows_tool.call_count, 1)
        self.assertEqual(self.process_tool.call_count, 0)

        # 2. 'active' short-circuits process query
        self.windows_tool.call_count = 0
        self.process_tool.call_count = 0
        trace_active = self.router.try_execute("is notepad active?")
        self.assertIsNotNone(trace_active)
        self.assertEqual(len(trace_active.steps), 1)
        self.assertEqual(trace_active.steps[0].tool_name, "list_open_windows")
        self.assertEqual(self.windows_tool.call_count, 1)
        self.assertEqual(self.process_tool.call_count, 0)

        # 3. 'running' short-circuits windows query when process exists
        self.windows_tool.call_count = 0
        self.process_tool.call_count = 0
        self.process_tool.data = {
            "returned_count": 1,
            "processes": [{"name": "notepad.exe", "pid": 1234, "memory_mb": 15.0}],
        }
        trace_running = self.router.try_execute("is notepad running?")
        self.assertIsNotNone(trace_running)
        self.assertEqual(len(trace_running.steps), 1)
        self.assertEqual(trace_running.steps[0].tool_name, "list_processes")
        self.assertEqual(self.process_tool.call_count, 1)
        self.assertEqual(self.windows_tool.call_count, 0)

        # 4. 'running' falls back to windows when process is absent
        self.windows_tool.call_count = 0
        self.process_tool.call_count = 0
        self.process_tool.data = {"returned_count": 0, "processes": []}
        self.windows_tool.data = {
            "returned_count": 1,
            "windows": [{"title": "notes.txt - Notepad", "process_name": "notepad.exe", "is_active": False}],
        }
        trace_running_fallback = self.router.try_execute("is notepad running?")
        self.assertIsNotNone(trace_running_fallback)
        self.assertEqual(len(trace_running_fallback.steps), 2)
        self.assertEqual(trace_running_fallback.steps[0].tool_name, "list_processes")
        self.assertEqual(trace_running_fallback.steps[1].tool_name, "list_open_windows")
        self.assertEqual(self.process_tool.call_count, 1)
        self.assertEqual(self.windows_tool.call_count, 1)
        self.assertEqual(
            trace_running_fallback.final_response,
            'Yes, Notepad is running with a visible window: "notes.txt - Notepad".',
        )

    def test_list_processes_tool_parameter_compatibility(self):
        """
        Verify ListProcessesTool backwards compatibility and all_processes argument handling.
        """
        from exocortex.tools.windows.processes import ListProcessesTool

        tool = ListProcessesTool()
        schema = tool.get_schema()
        props = schema["parameters"]["properties"]
        self.assertIn("all_processes", props)
        self.assertEqual(props["all_processes"]["type"], "boolean")

        valid, err = tool.validate_arguments({"all_processes": True})
        self.assertTrue(valid, msg=err)

        valid, err = tool.validate_arguments({})
        self.assertTrue(valid, msg=err)


class TestDefensiveParsingAndIdentityHardening(unittest.TestCase):
    """
    Tests for defensive parsing and identity policy hardening in fast_path.py:
    - Malformed structured data (data=None, data not dict, windows/processes not list)
    - Malformed list items ([None], [{}])
    - Identity handling: process_name=None, missing process_name, explicit 'unknown'
    - Known process identity precedence over window title
    - Error trace generation with 0 LLM calls for malformed inputs
    """

    def setUp(self):
        self.registry = ToolRegistry()
        self.windows_tool = DummyMockTool(
            "list_open_windows",
            output_text="Desktop Windows",
            data={"returned_count": 0, "windows": []},
        )
        self.process_tool = DummyMockTool(
            "list_processes",
            output_text="Running Processes",
            data={"total_processes_scanned": 0, "returned_count": 0, "processes": []},
        )
        self.registry.register(self.windows_tool)
        self.registry.register(self.process_tool)
        self.router = FastPathRouter(tool_registry=self.registry, enabled=True)

    def test_malformed_windows_data_none(self):
        self.windows_tool.data = {"windows": None}
        trace = self.router.try_execute("is notepad open?")
        self.assertIsNotNone(trace)
        self.assertEqual(trace.status, ExecutionStatus.ERROR)
        self.assertIsNone(trace.final_response)
        self.assertIn("malformed structured data", trace.error_message)

    def test_malformed_windows_data_unexpected_type(self):
        self.windows_tool.data = {"windows": "unexpected_string"}
        trace = self.router.try_execute("is notepad open?")
        self.assertIsNotNone(trace)
        self.assertEqual(trace.status, ExecutionStatus.ERROR)
        self.assertIsNone(trace.final_response)
        self.assertIn("malformed structured data", trace.error_message)

    def test_malformed_windows_data_item_none(self):
        self.windows_tool.data = {"windows": [None]}
        trace = self.router.try_execute("is notepad open?")
        self.assertIsNotNone(trace)
        self.assertEqual(trace.status, ExecutionStatus.ERROR)
        self.assertIsNone(trace.final_response)
        self.assertIn("malformed structured data", trace.error_message)

    def test_malformed_windows_data_item_empty_dict(self):
        self.windows_tool.data = {"windows": [{}]}
        trace = self.router.try_execute("is notepad open?")
        self.assertIsNotNone(trace)
        self.assertEqual(trace.status, ExecutionStatus.ERROR)
        self.assertIsNone(trace.final_response)
        self.assertIn("malformed structured data", trace.error_message)

    def test_malformed_windows_data_not_dict(self):
        self.windows_tool.data = "not_a_dictionary"
        trace = self.router.try_execute("is notepad open?")
        self.assertIsNotNone(trace)
        self.assertEqual(trace.status, ExecutionStatus.ERROR)
        self.assertIsNone(trace.final_response)
        self.assertIn("malformed structured data", trace.error_message)

    def test_malformed_processes_data_none(self):
        self.process_tool.data = {"processes": None}
        trace = self.router.try_execute("is notepad running?")
        self.assertIsNotNone(trace)
        self.assertEqual(trace.status, ExecutionStatus.ERROR)
        self.assertIsNone(trace.final_response)
        self.assertIn("malformed structured data", trace.error_message)

    def test_malformed_processes_data_item_none(self):
        self.process_tool.data = {"processes": [None]}
        trace = self.router.try_execute("is notepad running?")
        self.assertIsNotNone(trace)
        self.assertEqual(trace.status, ExecutionStatus.ERROR)
        self.assertIsNone(trace.final_response)
        self.assertIn("malformed structured data", trace.error_message)

    def test_malformed_processes_data_item_empty_dict(self):
        self.process_tool.data = {"processes": [{}]}
        trace = self.router.try_execute("is notepad running?")
        self.assertIsNotNone(trace)
        self.assertEqual(trace.status, ExecutionStatus.ERROR)
        self.assertIsNone(trace.final_response)
        self.assertIn("malformed structured data", trace.error_message)

    def test_malformed_processes_data_not_dict(self):
        self.process_tool.data = 42
        trace = self.router.try_execute("is notepad running?")
        self.assertIsNotNone(trace)
        self.assertEqual(trace.status, ExecutionStatus.ERROR)
        self.assertIsNone(trace.final_response)
        self.assertIn("malformed structured data", trace.error_message)

    def test_process_name_none_does_not_convert_to_string_none(self):
        # A window record with process_name=None and title matching Notepad.
        # Under hardened identity policy, process_name=None is NOT explicit "unknown",
        # so it must NOT trigger title fallback and must NOT match Notepad.
        self.windows_tool.data = {
            "returned_count": 1,
            "windows": [{"title": "Untitled - Notepad", "process_name": None, "is_active": True}],
        }
        self.process_tool.data = {"returned_count": 0, "processes": []}
        trace = self.router.try_execute("is notepad open?")
        self.assertIsNotNone(trace)
        self.assertEqual(trace.status, ExecutionStatus.COMPLETED)
        # It must not be reported as open because process_name=None is not eligible for title fallback
        self.assertEqual(
            trace.final_response,
            "No, Notepad is not open (no matching visible window or running process was found).",
        )

    def test_missing_process_name_does_not_fallback(self):
        # Missing process_name field in window record
        self.windows_tool.data = {
            "returned_count": 1,
            "windows": [{"title": "Untitled - Notepad", "is_active": True}],
        }
        self.process_tool.data = {"returned_count": 0, "processes": []}
        trace = self.router.try_execute("is notepad open?")
        self.assertIsNotNone(trace)
        self.assertEqual(trace.status, ExecutionStatus.COMPLETED)
        self.assertEqual(
            trace.final_response,
            "No, Notepad is not open (no matching visible window or running process was found).",
        )

    def test_explicit_unknown_process_name_allows_conservative_fallback(self):
        # Explicit process_name="unknown" IS eligible for conservative title fallback
        self.windows_tool.data = {
            "returned_count": 1,
            "windows": [{"title": "Untitled - Notepad", "process_name": "unknown", "is_active": True}],
        }
        trace = self.router.try_execute("is notepad open?")
        self.assertIsNotNone(trace)
        self.assertEqual(trace.status, ExecutionStatus.COMPLETED)
        self.assertEqual(
            trace.final_response,
            'Yes, Notepad is open with an active visible window: "Untitled - Notepad".',
        )

    def test_known_process_mismatch_never_overridden_by_title(self):
        # Chrome window with title "Untitled - Notepad"
        self.windows_tool.data = {
            "returned_count": 1,
            "windows": [{"title": "Untitled - Notepad", "process_name": "chrome.exe", "is_active": True}],
        }
        self.process_tool.data = {
            "returned_count": 1,
            "processes": [{"name": "chrome.exe", "pid": 5000, "memory_mb": 100.0}],
        }
        # Check if Notepad is open -> must NOT match Notepad
        trace_notepad = self.router.try_execute("is notepad open?")
        self.assertIsNotNone(trace_notepad)
        self.assertEqual(
            trace_notepad.final_response,
            "No, Notepad is not open (no matching visible window or running process was found).",
        )

        # Check if Chrome is open -> MUST match Chrome!
        trace_chrome = self.router.try_execute("is chrome open?")
        self.assertIsNotNone(trace_chrome)
        self.assertEqual(
            trace_chrome.final_response,
            'Yes, Google Chrome is open with an active visible window: "Untitled - Notepad".',
        )

    def test_valid_empty_collections(self):
        self.windows_tool.data = {"returned_count": 0, "windows": []}
        self.process_tool.data = {
            "total_processes_scanned": 0,
            "returned_count": 0,
            "scan_complete": True,
            "access_denied_count": 0,
            "processes": [],
        }
        trace = self.router.try_execute("is notepad running?")
        self.assertIsNotNone(trace)
        self.assertEqual(trace.status, ExecutionStatus.COMPLETED)
        self.assertEqual(
            trace.final_response,
            "No, Notepad is not running (no matching process or window was found).",
        )


class TestProcessScanCompletenessAndDiagnostics(unittest.TestCase):
    """
    Tests for process scan completeness, AccessDenied diagnostics, and router handling:
    - ListProcessesTool access_denied_count, vanished_count, and scan_complete
    - Distinguishing AccessDenied from normal NoSuchProcess races
    - Partial scan does not report confident "not running" when application is absent
    - Positive match confirms running even with partial scan
    - Conclusive visible-window observations are unaffected by partial process scan
    """

    def setUp(self):
        self.registry = ToolRegistry()
        self.windows_tool = DummyMockTool(
            "list_open_windows",
            output_text="Desktop Windows",
            data={"returned_count": 0, "windows": []},
        )
        self.process_tool = DummyMockTool(
            "list_processes",
            output_text="Running Processes",
            data={"total_processes_scanned": 0, "returned_count": 0, "processes": []},
        )
        self.registry.register(self.windows_tool)
        self.registry.register(self.process_tool)
        self.router = FastPathRouter(tool_registry=self.registry, enabled=True)

    @patch("psutil.process_iter")
    def test_list_processes_tool_access_denied_diagnostic(self, mock_iter):
        from exocortex.tools.windows.processes import ListProcessesTool
        import psutil

        p1 = MagicMock()
        p1.pid = 100
        p1.info = {"pid": 100, "name": "user_app.exe", "memory_info": MagicMock(rss=10485760)}

        p2 = MagicMock()
        p2.pid = 4
        # Simulate AccessDenied when accessing info
        type(p2).info = unittest.mock.PropertyMock(side_effect=psutil.AccessDenied(pid=4))

        mock_iter.return_value = [p1, p2]

        tool = ListProcessesTool()
        res = tool.run(all_processes=True)
        self.assertTrue(res.success)
        self.assertEqual(res.data["access_denied_count"], 1)
        self.assertEqual(res.data["vanished_count"], 0)
        self.assertFalse(res.data["scan_complete"])
        self.assertEqual(len(res.data["processes"]), 1)
        self.assertEqual(res.data["processes"][0]["name"], "user_app.exe")

    @patch("psutil.process_iter")
    def test_list_processes_tool_no_such_process_does_not_mark_partial(self, mock_iter):
        from exocortex.tools.windows.processes import ListProcessesTool
        import psutil

        p1 = MagicMock()
        p1.pid = 100
        p1.info = {"pid": 100, "name": "user_app.exe", "memory_info": MagicMock(rss=10485760)}

        p2 = MagicMock()
        p2.pid = 999
        # Simulate transient process exit race (NoSuchProcess)
        type(p2).info = unittest.mock.PropertyMock(side_effect=psutil.NoSuchProcess(pid=999))

        mock_iter.return_value = [p1, p2]

        tool = ListProcessesTool()
        res = tool.run(all_processes=True)
        self.assertTrue(res.success)
        self.assertEqual(res.data["access_denied_count"], 0)
        self.assertEqual(res.data["vanished_count"], 1)
        self.assertTrue(res.data["scan_complete"])
        self.assertEqual(len(res.data["processes"]), 1)

    def test_positive_match_confirms_running_even_with_partial_scan(self):
        # Notepad is present in the accessible portion of a partial scan
        self.process_tool.data = {
            "total_processes_scanned": 150,
            "returned_count": 1,
            "access_denied_count": 5,
            "scan_complete": False,
            "processes": [{"name": "notepad.exe", "pid": 1234, "memory_mb": 18.5}],
        }
        trace = self.router.try_execute("is notepad running?")
        self.assertIsNotNone(trace)
        self.assertEqual(trace.status, ExecutionStatus.COMPLETED)
        self.assertEqual(trace.final_response, "Yes, Notepad is running (PID: 1234).")
        # Short-circuited after single tool step
        self.assertEqual(len(trace.steps), 1)
        self.assertEqual(trace.steps[0].tool_name, "list_processes")

    def test_no_match_with_partial_scan_returns_uncertain_observation(self):
        # Notepad is NOT found, but scan was partial due to 8 inaccessible processes
        self.process_tool.data = {
            "total_processes_scanned": 150,
            "returned_count": 0,
            "access_denied_count": 8,
            "scan_complete": False,
            "processes": [],
        }
        self.windows_tool.data = {"returned_count": 0, "windows": []}

        trace = self.router.try_execute("is notepad running?")
        self.assertIsNotNone(trace)
        self.assertEqual(trace.status, ExecutionStatus.COMPLETED)
        # Must NOT produce a false confident "not running"
        self.assertNotIn("No, Notepad is not running", trace.final_response)
        self.assertIn("Could not confirm whether Notepad is running", trace.final_response)
        self.assertIn("process scan was partial", trace.final_response)
        self.assertIn("8 inaccessible elevated/system processes", trace.final_response)
        # Two-step trace recorded
        self.assertEqual(len(trace.steps), 2)

    def test_no_match_with_complete_scan_returns_conclusive_not_running(self):
        self.process_tool.data = {
            "total_processes_scanned": 200,
            "returned_count": 0,
            "access_denied_count": 0,
            "scan_complete": True,
            "processes": [],
        }
        self.windows_tool.data = {"returned_count": 0, "windows": []}

        trace = self.router.try_execute("is notepad running?")
        self.assertIsNotNone(trace)
        self.assertEqual(trace.status, ExecutionStatus.COMPLETED)
        self.assertEqual(
            trace.final_response,
            "No, Notepad is not running (no matching process or window was found).",
        )

    def test_conclusive_visible_window_unaffected_by_process_partiality(self):
        # Visible window is found -> 'open' and 'active' short-circuit without calling process tool
        self.windows_tool.data = {
            "returned_count": 1,
            "windows": [{"title": "Document - Notepad", "process_name": "notepad.exe", "is_active": True}],
        }
        # Process tool has partial scan, but should never even be called
        self.process_tool.data = {
            "access_denied_count": 10,
            "scan_complete": False,
            "processes": [],
        }
        trace_open = self.router.try_execute("is notepad open?")
        self.assertIsNotNone(trace_open)
        self.assertEqual(trace_open.status, ExecutionStatus.COMPLETED)
        self.assertEqual(
            trace_open.final_response,
            'Yes, Notepad is open with an active visible window: "Document - Notepad".',
        )
        self.assertEqual(len(trace_open.steps), 1)
        self.assertEqual(self.process_tool.call_count, 0)

    def test_active_and_open_partial_scan_background_diagnostic(self):
        # No visible window, and process scan is partial
        self.windows_tool.data = {"returned_count": 0, "windows": []}
        self.process_tool.data = {
            "access_denied_count": 4,
            "scan_complete": False,
            "processes": [],
        }
        trace_active = self.router.try_execute("is notepad active?")
        self.assertIsNotNone(trace_active)
        self.assertEqual(trace_active.status, ExecutionStatus.COMPLETED)
        self.assertIn("No, Notepad is not active (no visible window found)", trace_active.final_response)
        self.assertIn("partial (4 inaccessible processes)", trace_active.final_response)

        trace_open = self.router.try_execute("is notepad open?")
        self.assertIsNotNone(trace_open)
        self.assertEqual(trace_open.status, ExecutionStatus.COMPLETED)
        self.assertIn("No, Notepad is not open (no matching visible window was found)", trace_open.final_response)
        self.assertIn("partial (4 inaccessible processes)", trace_open.final_response)


class TestConservativeTitleFallbackHardening(unittest.TestCase):
    """
    Tests for conservative title fallback hardening:
    - Interior word substring matching MUST be rejected when process identity is unknown:
      * 'Why Notepad is Obsolete - YouTube'
      * 'Download Google Chrome installer - TechBlog'
      * 'How to configure Windows Terminal - Tutorial'
      * 'Cancelling your Spotify subscription - Edge'
    - Exact and title-ending matches MUST be accepted when process identity is unknown:
      * 'Untitled - Notepad'
      * '(4) YouTube - Google Chrome'
      * 'New Tab - Google Chrome'
      * 'Windows Terminal'
    - Positive and negative tests for all supported application aliases
    """

    def setUp(self):
        self.registry = ToolRegistry()
        self.windows_tool = DummyMockTool(
            "list_open_windows",
            output_text="Desktop Windows",
            data={"returned_count": 0, "windows": []},
        )
        self.process_tool = DummyMockTool(
            "list_processes",
            output_text="Running Processes",
            data={"total_processes_scanned": 0, "returned_count": 0, "processes": []},
        )
        self.registry.register(self.windows_tool)
        self.registry.register(self.process_tool)
        self.router = FastPathRouter(tool_registry=self.registry, enabled=True)

    def test_rejection_of_interior_word_substring_matches_when_unknown(self):
        adversarial_cases = [
            ("is notepad open?", "Why Notepad is Obsolete - YouTube"),
            ("is chrome open?", "Download Google Chrome installer - TechBlog"),
            ("is windows terminal open?", "How to configure Windows Terminal - Tutorial"),
            ("is spotify open?", "Cancelling your Spotify subscription - Edge"),
            ("is vscode open?", "Developing in Visual Studio Code - Browser"),
            ("is firefox open?", "Firefox vs Chrome comparison - Medium"),
        ]
        for query, adversarial_title in adversarial_cases:
            with self.subTest(query=query, title=adversarial_title):
                self.windows_tool.data = {
                    "returned_count": 1,
                    "windows": [{"title": adversarial_title, "process_name": "unknown", "is_active": True}],
                }
                self.process_tool.data = {"returned_count": 0, "processes": []}
                trace = self.router.try_execute(query)
                self.assertIsNotNone(trace)
                self.assertEqual(trace.status, ExecutionStatus.COMPLETED)
                # Must be rejected because title only contains the app name as an interior substring
                self.assertIn("not open", trace.final_response)

    def test_acceptance_of_proper_title_endings_and_exact_titles_when_unknown(self):
        genuine_cases = [
            ("is notepad open?", "Untitled - Notepad", "Notepad"),
            ("is chrome open?", "(4) YouTube - Google Chrome", "Google Chrome"),
            ("is chrome open?", "New Tab - Google Chrome", "Google Chrome"),
            ("is windows terminal open?", "Windows Terminal", "Windows Terminal"),
            ("is spotify open?", "Spotify Free", "Spotify"),
            ("is spotify open?", "Spotify", "Spotify"),
            ("is vscode open?", "app.py - Visual Studio Code", "Visual Studio Code"),
            ("is firefox open?", "Mozilla Firefox", "Mozilla Firefox"),
        ]
        for query, genuine_title, app_name in genuine_cases:
            with self.subTest(query=query, title=genuine_title):
                self.windows_tool.data = {
                    "returned_count": 1,
                    "windows": [{"title": genuine_title, "process_name": "unknown", "is_active": True}],
                }
                trace = self.router.try_execute(query)
                self.assertIsNotNone(trace)
                self.assertEqual(trace.status, ExecutionStatus.COMPLETED)
                self.assertIn(f'Yes, {app_name} is open with an active visible window: "{genuine_title}".', trace.final_response)

    def test_all_approved_application_aliases_positive_and_negative(self):
        from exocortex.execution.fast_path import APPROVED_APPLICATION_MAP

        for alias, app_meta in APPROVED_APPLICATION_MAP.items():
            query = f"is {alias} open?"
            # 1. Negative test: title does not match
            self.windows_tool.data = {
                "returned_count": 1,
                "windows": [{"title": "Unrelated Document - Word", "process_name": "unknown", "is_active": True}],
            }
            self.process_tool.data = {"returned_count": 0, "processes": []}
            trace_neg = self.router.try_execute(query)
            self.assertIsNotNone(trace_neg)
            self.assertIn("not open", trace_neg.final_response)

            # 2. Positive test: title matches an approved pattern ending
            pattern = app_meta.window_title_patterns[0]
            matched_title = f"My File {pattern}" if not pattern.startswith("-") and not pattern.startswith("—") else f"My File {pattern}"
            self.windows_tool.data = {
                "returned_count": 1,
                "windows": [{"title": matched_title, "process_name": "unknown", "is_active": True}],
            }
            trace_pos = self.router.try_execute(query)
            self.assertIsNotNone(trace_pos)
            self.assertIn("is open with an active visible window", trace_pos.final_response)


if __name__ == "__main__":
    unittest.main()

