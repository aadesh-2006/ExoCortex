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
    ):
        self.name = name
        self.description = f"Dummy {name}"
        self.permission_level = permission
        self.parameters = []
        self.fail = fail
        self.output_text = output_text
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
        parameterized = [
            "Is Windows Terminal open?",
            "Inspect the UI elements in the Windows Terminal window.",
            "Inspect window matching 'Visual Studio Code'",
            "Find process with pid 1234",
            "Is notepad running?",
            "Check if Chrome is open",
            "system info full detail",
        ]
        for q in parameterized:
            with self.subTest(query=q):
                self.assertIsNone(self.router.match(q))
                self.assertIsNone(self.router.try_execute(q))

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


if __name__ == "__main__":
    unittest.main()
