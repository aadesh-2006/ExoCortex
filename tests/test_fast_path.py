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
        self.registry.register(self.windows_tool)
        self.registry.register(self.process_tool)
        self.router = FastPathRouter(tool_registry=self.registry, enabled=True)

    def test_normalization_and_punctuation(self):
        cases = [
            ("  What application windows are open?  ", "what application windows are open"),
            ("list open windows.", "list open windows"),
            ("WHAT WINDOWS ARE OPEN???", "what windows are open"),
            ("  show running processes!  ", "show running processes"),
            ("\"List processes;\"", "list processes"),
            ("What processes are running:", "what processes are running"),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(normalize_fast_path_query(raw), expected)

    def test_canonical_routing_matches(self):
        windows_queries = [
            "list open windows",
            "list windows",
            "what application windows are open",
            "what windows are open",
            "show open windows",
            "show windows",
            "get open windows",
            "get windows",
            "What application windows are open?",
            "  SHOW OPEN WINDOWS.  ",
        ]
        for q in windows_queries:
            with self.subTest(query=q):
                matched = self.router.match(q)
                self.assertEqual(matched, "list_open_windows")

        process_queries = [
            "list processes",
            "list running processes",
            "show processes",
            "show running processes",
            "get processes",
            "get running processes",
            "what processes are running",
            "LIST RUNNING PROCESSES!",
        ]
        for q in process_queries:
            with self.subTest(query=q):
                matched = self.router.match(q)
                self.assertEqual(matched, "list_processes")

    def test_ambiguous_and_unrelated_fall_through(self):
        non_matching_queries = [
            "Hello who are you",
            "Open Notepad",
            "Search Python docs",
            "What is the weather in Tokyo",
            "Tell me about open source software",
            "Close all open windows",
            "Kill running process",
            "The window is broken",
            "Process this file",
        ]
        for q in non_matching_queries:
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
        ]
        for q in parameterized:
            with self.subTest(query=q):
                self.assertIsNone(self.router.match(q))
                self.assertIsNone(self.router.try_execute(q))

    def test_successful_execution_trace_structure(self):
        trace = self.router.try_execute("What application windows are open?")
        self.assertIsNotNone(trace)
        self.assertEqual(trace.status, ExecutionStatus.COMPLETED)
        self.assertEqual(self.windows_tool.call_count, 1)
        self.assertIn("Visible Desktop Windows", trace.final_response)
        self.assertIsNone(trace.error_message)
        self.assertEqual(trace.total_steps, 1)

        # Check step details
        step = trace.steps[0]
        self.assertEqual(step.tool_name, "list_open_windows")
        self.assertEqual(step.action_type, ActionType.TOOL_CALL)
        self.assertEqual(step.status, ExecutionStatus.COMPLETED)
        self.assertTrue(step.tool_result.success)
        self.assertEqual(step.decision.intent, "window_enumeration")

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

    def test_unregistered_or_non_safe_tool_falls_through(self):
        # 1. Non-SAFE tool
        registry = ToolRegistry()
        restricted_tool = DummyMockTool("list_processes", permission=PermissionLevel.CONFIRMATION_REQUIRED)
        registry.register(restricted_tool)
        router = FastPathRouter(tool_registry=registry, enabled=True)
        self.assertIsNone(router.try_execute("list processes"))

        # 2. Tool not in registry
        empty_registry = ToolRegistry()
        router2 = FastPathRouter(tool_registry=empty_registry, enabled=True)
        self.assertIsNone(router2.try_execute("list open windows"))

    def test_disabled_router_always_returns_none(self):
        disabled_router = FastPathRouter(tool_registry=self.registry, enabled=False)
        self.assertIsNone(disabled_router.match("list open windows"))
        self.assertIsNone(disabled_router.try_execute("list open windows"))

    def test_agent_integration_fast_path(self):
        mock_slm = MockSLMProvider()
        agent = ExoCortexAgent(
            slm_provider=mock_slm,
            tool_registry=self.registry,
            enable_fast_path=True,
        )

        res = agent.run("What application windows are open?")
        self.assertTrue(res.success)
        self.assertEqual(self.windows_tool.call_count, 1)
        self.assertIn("Visible Desktop Windows", res.response_text)
        self.assertEqual(res.decision.intent, "window_enumeration")
        self.assertEqual(res.slm_info.get("provider"), "FastPathRouter")
        self.assertEqual(res.slm_info.get("llm_calls"), 0)

        # Confirm non-fast-path query falls through to SLM
        res2 = agent.run("Hello, who are you?")
        self.assertTrue(res2.success)
        self.assertEqual(res2.slm_info.get("provider"), "MockSLMProvider")


if __name__ == "__main__":
    unittest.main()
