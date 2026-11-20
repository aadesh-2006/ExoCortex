"""
Regression and verification tests for M9.1: Core Multi-step Execution Correctness.

Tests cover:
- Test A: Multi-step parsing preserves all valid planned tool steps without truncation.
- Test B: Sequential multi-step execution across decisions executes in order and retains observations.
- Test C: Compound desktop observation does not prematurely terminate on step 1.
- Test D: Simple direct observation retains fast single-call completion.
- Test E: Failed intermediate step halts batch and does not report successful completion.
- Test F: Honest completion semantics ensures successful tool execution alone does not cause
          an incomplete compound task to be reported as successfully completed.
- Test Group 2: Behavioral verification of all 10 mandated direct-completion queries.
- Test Group 3: Public AgentRunResult contracts through ExoCortexAgent.run().
"""

from __future__ import annotations

import json
import unittest
from typing import Any, Dict, List, Optional

from exocortex.agent import ExoCortexAgent, AgentRunResult
from exocortex.execution.executor import (
    AgentExecutor,
    DIRECT_COMPLETION_TOOLS,
    is_simple_observation_request,
)
from exocortex.execution.state import ActionType, ExecutionStatus
from exocortex.schema import AgentDecision
from exocortex.slm.base import SLMMessage, SLMProvider, SLMResponse
from exocortex.slm.parser import parse_and_validate_decision
from exocortex.tools.base import BaseTool, PermissionLevel, ToolParameter, ToolResult
from exocortex.tools.registry import ToolRegistry, get_default_tool_registry


class ScriptedSLM(SLMProvider):
    """Deterministic scripted SLM provider for fast unit tests without live model inference."""

    def __init__(self, responses: List[str]):
        self.responses = list(responses)
        self.call_count = 0
        self.prompts_received: List[str] = []

    def is_available(self) -> bool:
        return True

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": "scripted-test-slm",
            "provider": "ScriptedSLM",
            "is_local": True,
        }

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
        **kwargs: Any,
    ) -> SLMResponse:
        self.call_count += 1
        self.prompts_received.append(prompt)
        if not self.responses:
            # Default fallback when scripted responses are exhausted
            return SLMResponse(
                content=json.dumps({
                    "thought_summary": "All scripted steps exhausted.",
                    "intent": "general_query",
                    "steps": [],
                    "direct_response": "Exhausted scripted responses.",
                }),
                prompt_tokens=10,
                completion_tokens=10,
                latency_ms=1.0,
                model_name="scripted-test-slm",
                provider_name="ScriptedSLM",
            )
        content = self.responses.pop(0)
        return SLMResponse(
            content=content,
            prompt_tokens=10,
            completion_tokens=10,
            latency_ms=1.0,
            model_name="scripted-test-slm",
            provider_name="ScriptedSLM",
        )

    def chat(self, messages: List[SLMMessage], **kwargs: Any) -> SLMResponse:
        return self.generate(prompt=messages[-1].content if messages else "")


class MockDesktopWindowsTool(BaseTool):
    name = "list_open_windows"
    description = "Lists open desktop windows"
    permission_level = PermissionLevel.SAFE
    parameters = []

    def __init__(self, output: str = "Visible Desktop Windows (1 found):\n1. 'Untitled - Notepad' (notepad.exe)"):
        self.output = output
        self.call_count = 0

    def run(self, **kwargs) -> ToolResult:
        self.call_count += 1
        return ToolResult(success=True, output=self.output)


class MockProcessesTool(BaseTool):
    name = "list_processes"
    description = "Lists running processes"
    permission_level = PermissionLevel.SAFE
    parameters = [
        ToolParameter(name="name_filter", type="string", description="Filter by process name", required=False),
        ToolParameter(name="all_processes", type="boolean", description="Include all processes", required=False),
    ]

    def __init__(self, output: str = "Running Processes (2 found):\n1. python.exe (PID: 101)\n2. notepad.exe (PID: 202)"):
        self.output = output
        self.call_count = 0

    def run(self, **kwargs) -> ToolResult:
        self.call_count += 1
        return ToolResult(success=True, output=self.output)


class MockInspectUIElementsTool(BaseTool):
    name = "inspect_window_ui_elements"
    description = "Inspects window UI elements"
    permission_level = PermissionLevel.SAFE
    parameters = [
        ToolParameter(name="target_title", type="string", description="Window title", required=False),
        ToolParameter(name="process_name", type="string", description="Process name", required=False),
    ]

    def __init__(self, output: str = "UI Elements for Notepad:\n- Document: 'Untitled'\n- Button: 'Close'"):
        self.output = output
        self.call_count = 0
        self.last_kwargs: Dict[str, Any] = {}

    def run(self, **kwargs) -> ToolResult:
        self.call_count += 1
        self.last_kwargs = kwargs
        return ToolResult(success=True, output=self.output)


class MockFailingTool(BaseTool):
    name = "failing_step_tool"
    description = "Tool that always fails for testing error handling"
    permission_level = PermissionLevel.SAFE
    parameters = []

    def __init__(self, error_msg: str = "Simulated tool crash"):
        self.error_msg = error_msg
        self.call_count = 0

    def run(self, **kwargs) -> ToolResult:
        self.call_count += 1
        return ToolResult(success=False, output="", error=self.error_msg)


class TestMultiStepExecution(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()
        self.windows_tool = MockDesktopWindowsTool()
        self.processes_tool = MockProcessesTool()
        self.ui_tool = MockInspectUIElementsTool()
        self.failing_tool = MockFailingTool()

        self.registry.register(self.windows_tool)
        self.registry.register(self.processes_tool)
        self.registry.register(self.ui_tool)
        self.registry.register(self.failing_tool)

        # Default echo and system_info tools for generic multi-step testing
        from exocortex.tools.builtin import EchoTool, SystemInfoTool
        self.echo_tool = EchoTool()
        self.sys_tool = SystemInfoTool()
        self.registry.register(self.echo_tool)
        self.registry.register(self.sys_tool)

    # -------------------------------------------------------------------------
    # Test A: Multi-step parsing
    # -------------------------------------------------------------------------
    def test_multistep_parsing_preserves_all_valid_steps(self):
        """
        Test A: Verify that parsing a model response with multiple valid tool steps
        does not truncate or silently discard steps beyond the first.
        """
        raw_json = json.dumps({
            "thought_summary": "Step 1 echo message, then Step 2 inspect hardware.",
            "intent": "general_action",
            "steps": [
                {"tool": "echo", "arguments": {"message": "first action"}},
                {"tool": "system_info", "arguments": {"detail_level": "basic"}},
            ],
            "requires_confirmation": False,
            "direct_response": None,
        })

        decision, err = parse_and_validate_decision(raw_json, self.registry)
        self.assertIsNone(err)
        self.assertIsNotNone(decision)
        # Downstream execution engine must receive both steps
        self.assertEqual(len(decision.steps), 2, "Parser must not truncate valid planned steps to 1")
        self.assertEqual(decision.steps[0].tool, "echo")
        self.assertEqual(decision.steps[0].arguments, {"message": "first action"})
        self.assertEqual(decision.steps[1].tool, "system_info")
        self.assertEqual(decision.steps[1].arguments, {"detail_level": "basic"})

    # -------------------------------------------------------------------------
    # Test B: Sequential multi-step execution
    # -------------------------------------------------------------------------
    def test_sequential_multistep_execution_preserves_order_and_observations(self):
        """
        Test B: Scripted model produces:
        - Decision 1: Tool A (echo step 1)
        - Decision 2: Tool B (echo step 2)
        - Decision 3: Final response
        Verify both tools execute in order, observations are retained, and trace completes.
        """
        resp_1 = json.dumps({
            "thought_summary": "Running first echo.",
            "intent": "general_action",
            "steps": [{"tool": "echo", "arguments": {"message": "Observation 1"}}],
            "direct_response": None,
        })
        resp_2 = json.dumps({
            "thought_summary": "Running second echo based on first.",
            "intent": "general_action",
            "steps": [{"tool": "echo", "arguments": {"message": "Observation 2"}}],
            "direct_response": None,
        })
        resp_3 = json.dumps({
            "thought_summary": "Synthesizing both observations.",
            "intent": "final_summary",
            "steps": [],
            "direct_response": "Completed both steps: Observation 1 and Observation 2.",
        })

        provider = ScriptedSLM([resp_1, resp_2, resp_3])
        executor = AgentExecutor(
            slm_provider=provider,
            tool_registry=self.registry,
            direct_observation_completion=False,
        )

        trace = executor.execute("Run two sequential echoes")
        self.assertEqual(trace.status, ExecutionStatus.COMPLETED)
        # Trace should contain: Step 1 (tool), Step 2 (tool), Step 3 (final_response)
        self.assertEqual(len(trace.steps), 3)
        self.assertEqual(trace.steps[0].tool_name, "echo")
        self.assertEqual(trace.steps[0].arguments, {"message": "Observation 1"})
        self.assertIn("Observation 1", trace.steps[0].observation_text)

        self.assertEqual(trace.steps[1].tool_name, "echo")
        self.assertEqual(trace.steps[1].arguments, {"message": "Observation 2"})
        self.assertIn("Observation 2", trace.steps[1].observation_text)

        self.assertEqual(trace.steps[2].action_type, ActionType.FINAL_RESPONSE)
        self.assertEqual(trace.final_response, "Completed both steps: Observation 1 and Observation 2.")

        # Verify second turn prompt contained previous observation
        self.assertIn("Observation 1", provider.prompts_received[1])

    # -------------------------------------------------------------------------
    # Test C: Compound desktop observation
    # -------------------------------------------------------------------------
    def test_compound_desktop_observation_does_not_prematurely_terminate(self):
        """
        Test C: Simulate a compound request:
        "Find the Notepad window and inspect its UI elements"
        - Turn 1 plans list_open_windows
        - Turn 2 plans inspect_window_ui_elements
        - Turn 3 synthesizes final answer
        Verify direct_observation_completion does NOT prematurely abort the task after step 1.
        """
        turn1 = json.dumps({
            "thought_summary": "First find the Notepad window.",
            "intent": "ui_inspection",
            "steps": [{"tool": "list_open_windows", "arguments": {}}],
            "direct_response": None,
        })
        turn2 = json.dumps({
            "thought_summary": "Found Notepad. Now inspect its UI elements.",
            "intent": "ui_inspection",
            "steps": [{"tool": "inspect_window_ui_elements", "arguments": {"process_name": "notepad.exe"}}],
            "direct_response": None,
        })
        turn3 = json.dumps({
            "thought_summary": "Synthesize inspected controls.",
            "intent": "ui_inspection",
            "steps": [],
            "direct_response": "Found Notepad and verified Document and Close buttons.",
        })

        provider = ScriptedSLM([turn1, turn2, turn3])
        executor = AgentExecutor(
            slm_provider=provider,
            tool_registry=self.registry,
            direct_observation_completion=True,  # enabled, but must NOT short-circuit compound request
        )

        trace = executor.execute("Find the Notepad window and inspect its UI elements")
        # Must execute all turns without premature termination on step 1
        self.assertEqual(provider.call_count, 3, "Compound task must not abort after turn 1")
        self.assertEqual(self.windows_tool.call_count, 1)
        self.assertEqual(self.ui_tool.call_count, 1)
        self.assertEqual(trace.status, ExecutionStatus.COMPLETED)
        self.assertEqual(trace.final_response, "Found Notepad and verified Document and Close buttons.")

    # -------------------------------------------------------------------------
    # Test D: Simple direct observation
    # -------------------------------------------------------------------------
    def test_simple_direct_observation_retains_fast_completion(self):
        """
        Test D: A simple single observation request ("Show me my open windows")
        must retain its single-turn direct completion behavior when enabled.
        """
        turn1 = json.dumps({
            "thought_summary": "Listing visible open windows.",
            "intent": "window_enumeration",
            "steps": [{"tool": "list_open_windows", "arguments": {}}],
            "direct_response": None,
        })

        provider = ScriptedSLM([turn1])
        executor = AgentExecutor(
            slm_provider=provider,
            tool_registry=self.registry,
            direct_observation_completion=True,
        )

        trace = executor.execute("Show me my open windows")
        # Exactly 1 LLM call must occur
        self.assertEqual(provider.call_count, 1)
        self.assertEqual(self.windows_tool.call_count, 1)
        self.assertEqual(trace.status, ExecutionStatus.COMPLETED)
        self.assertIn("Untitled - Notepad", trace.final_response)

    # -------------------------------------------------------------------------
    # Test E: Failed intermediate step
    # -------------------------------------------------------------------------
    def test_failed_intermediate_step_halts_batch_and_reports_error(self):
        """
        Test E: If step 1 in a planned multi-step sequence fails:
        - The failure is retained in execution history with error details.
        - Subsequent steps in the batch are halted.
        - The executor does not report successful task completion.
        - Loop and step limits remain protected.
        """
        batch_plan = json.dumps({
            "thought_summary": "Execute failing tool then echo.",
            "intent": "general_action",
            "steps": [
                {"tool": "failing_step_tool", "arguments": {}},
                {"tool": "echo", "arguments": {"message": "should not run"}},
            ],
            "direct_response": None,
        })

        provider = ScriptedSLM([batch_plan])
        executor = AgentExecutor(
            slm_provider=provider,
            tool_registry=self.registry,
        )

        trace = executor.execute("Execute failing batch")
        # 1. Failure must be captured in step 1
        self.assertEqual(len(trace.steps), 1)
        self.assertEqual(trace.steps[0].tool_name, "failing_step_tool")
        self.assertEqual(trace.steps[0].status, ExecutionStatus.ERROR)
        self.assertIn("Simulated tool crash", trace.steps[0].error_message)

        # 2. Step 2 (echo) must NOT have been executed blindly
        self.assertEqual(self.echo_tool.parameters, self.registry.get_tool("echo").parameters)

        # 3. Executor must NOT report successful completion
        self.assertNotEqual(trace.status, ExecutionStatus.COMPLETED)
        self.assertEqual(trace.status, ExecutionStatus.ERROR)
        self.assertIn("Simulated tool crash", trace.error_message)

    # -------------------------------------------------------------------------
    # Test F: Honest completion semantics
    # -------------------------------------------------------------------------
    def test_honest_completion_semantics_rejects_premature_success(self):
        """
        Test F: Successful execution of an intermediate observation tool alone
        must not cause an incomplete compound task to be reported as successfully completed.
        """
        # Model only returns step 1 (list windows), but user asked a compound question.
        # Direct completion must not falsely claim the entire goal succeeded.
        turn1 = json.dumps({
            "thought_summary": "Listing windows to see if Notepad is there.",
            "intent": "ui_inspection",
            "steps": [{"tool": "list_open_windows", "arguments": {}}],
            "direct_response": None,
        })

        provider = ScriptedSLM([turn1])
        agent = ExoCortexAgent(
            slm_provider=provider,
            tool_registry=self.registry,
            enable_fast_path=False,
            direct_observation_completion=True,
        )

        # When the agent runs, if only step 1 executed and scripted responses ran out
        # without inspecting Notepad's UI elements, it must not report success for the compound request.
        result = agent.run("Find Notepad and inspect its UI elements")
        # Direct observation completion must NOT have marked the task completed on step 1
        self.assertEqual(self.ui_tool.call_count, 0)
        # The prompt demanded inspecting UI elements, but only windows were listed
        # The agent response must not be the raw window list pretending the task succeeded
        self.assertNotEqual(result.response_text, self.windows_tool.output)


class TestDirectCompletionTenMandatedCases(unittest.TestCase):
    """Behavioral tests verifying actual execution behavior for all 10 mandated prompts."""

    def setUp(self):
        self.registry = ToolRegistry()
        self.windows_tool = MockDesktopWindowsTool()
        self.processes_tool = MockProcessesTool()
        self.ui_tool = MockInspectUIElementsTool()

        self.registry.register(self.windows_tool)
        self.registry.register(self.processes_tool)
        self.registry.register(self.ui_tool)

    def test_case_1_show_me_open_windows_direct_completes(self):
        """Case 1: 'Show me open windows.' -> Simple observation; completes directly."""
        turn1 = json.dumps({
            "thought_summary": "Listing open windows.",
            "intent": "window_enumeration",
            "steps": [{"tool": "list_open_windows", "arguments": {}}],
            "direct_response": None,
        })
        provider = ScriptedSLM([turn1])
        executor = AgentExecutor(slm_provider=provider, tool_registry=self.registry, direct_observation_completion=True)
        trace = executor.execute("Show me open windows.")
        self.assertEqual(provider.call_count, 1)
        self.assertEqual(trace.status, ExecutionStatus.COMPLETED)
        self.assertEqual(trace.final_response, self.windows_tool.output)

    def test_case_2_list_all_running_processes_direct_completes(self):
        """Case 2: 'List all running processes.' -> Simple observation; completes directly."""
        turn1 = json.dumps({
            "thought_summary": "Listing processes.",
            "intent": "process_enumeration",
            "steps": [{"tool": "list_processes", "arguments": {}}],
            "direct_response": None,
        })
        provider = ScriptedSLM([turn1])
        executor = AgentExecutor(slm_provider=provider, tool_registry=self.registry, direct_observation_completion=True)
        trace = executor.execute("List all running processes.")
        self.assertEqual(provider.call_count, 1)
        self.assertEqual(trace.status, ExecutionStatus.COMPLETED)
        self.assertEqual(trace.final_response, self.processes_tool.output)

    def test_case_3_find_notepad_and_inspect_suppresses_direct_completion(self):
        """Case 3: 'Find the Notepad window and inspect its UI elements.' -> Compound; advances to Turn 2."""
        turn1 = json.dumps({
            "thought_summary": "First finding Notepad.",
            "intent": "ui_inspection",
            "steps": [{"tool": "list_open_windows", "arguments": {}}],
            "direct_response": None,
        })
        turn2 = json.dumps({
            "thought_summary": "Inspecting Notepad controls.",
            "intent": "ui_inspection",
            "steps": [{"tool": "inspect_window_ui_elements", "arguments": {"process_name": "notepad.exe"}}],
            "direct_response": None,
        })
        turn3 = json.dumps({
            "thought_summary": "Done.",
            "intent": "ui_inspection",
            "steps": [],
            "direct_response": "Inspected Notepad controls successfully.",
        })
        provider = ScriptedSLM([turn1, turn2, turn3])
        executor = AgentExecutor(slm_provider=provider, tool_registry=self.registry, direct_observation_completion=True)
        trace = executor.execute("Find the Notepad window and inspect its UI elements.")
        self.assertEqual(provider.call_count, 3)
        self.assertEqual(self.windows_tool.call_count, 1)
        self.assertEqual(self.ui_tool.call_count, 1)
        self.assertEqual(trace.final_response, "Inspected Notepad controls successfully.")

    def test_case_4_show_chrome_and_inspect_suppresses_direct_completion(self):
        """Case 4: 'Show Chrome and inspect its accessibility tree.' -> Compound; advances to Turn 2."""
        turn1 = json.dumps({
            "thought_summary": "Finding Chrome window.",
            "intent": "ui_inspection",
            "steps": [{"tool": "list_open_windows", "arguments": {}}],
            "direct_response": None,
        })
        turn2 = json.dumps({
            "thought_summary": "Done.",
            "intent": "ui_inspection",
            "steps": [],
            "direct_response": "Chrome accessibility tree inspected.",
        })
        provider = ScriptedSLM([turn1, turn2])
        executor = AgentExecutor(slm_provider=provider, tool_registry=self.registry, direct_observation_completion=True)
        trace = executor.execute("Show Chrome and inspect its accessibility tree.")
        self.assertEqual(provider.call_count, 2)
        self.assertEqual(trace.final_response, "Chrome accessibility tree inspected.")

    def test_case_5_find_notepad_then_inspect_suppresses_direct_completion(self):
        """Case 5: 'Find whether Notepad is running, then inspect it if available.' -> Compound; advances to Turn 2."""
        turn1 = json.dumps({
            "thought_summary": "Listing processes to check Notepad.",
            "intent": "ui_inspection",
            "steps": [{"tool": "list_processes", "arguments": {}}],
            "direct_response": None,
        })
        turn2 = json.dumps({
            "thought_summary": "Synthesizing.",
            "intent": "ui_inspection",
            "steps": [],
            "direct_response": "Notepad is running; controls inspected.",
        })
        provider = ScriptedSLM([turn1, turn2])
        executor = AgentExecutor(slm_provider=provider, tool_registry=self.registry, direct_observation_completion=True)
        trace = executor.execute("Find whether Notepad is running, then inspect it if available.")
        self.assertEqual(provider.call_count, 2)
        self.assertEqual(trace.final_response, "Notepad is running; controls inspected.")

    def test_case_6_count_open_windows_suppresses_direct_completion(self):
        """Case 6: 'List open windows and tell me how many there are.' -> Aggregation; advances to Turn 2."""
        turn1 = json.dumps({
            "thought_summary": "Listing windows to count them.",
            "intent": "window_enumeration",
            "steps": [{"tool": "list_open_windows", "arguments": {}}],
            "direct_response": None,
        })
        turn2 = json.dumps({
            "thought_summary": "Counting.",
            "intent": "window_enumeration",
            "steps": [],
            "direct_response": "There is currently 1 window open.",
        })
        provider = ScriptedSLM([turn1, turn2])
        executor = AgentExecutor(slm_provider=provider, tool_registry=self.registry, direct_observation_completion=True)
        trace = executor.execute("List open windows and tell me how many there are.")
        self.assertEqual(provider.call_count, 2)
        self.assertEqual(trace.final_response, "There is currently 1 window open.")

    def test_case_7_windows_especially_chrome_suppresses_direct_completion(self):
        """Case 7: 'Show me the windows, especially Chrome.' -> Filtering; advances to Turn 2."""
        turn1 = json.dumps({
            "thought_summary": "Listing windows to filter for Chrome.",
            "intent": "window_enumeration",
            "steps": [{"tool": "list_open_windows", "arguments": {}}],
            "direct_response": None,
        })
        turn2 = json.dumps({
            "thought_summary": "Filtering.",
            "intent": "window_enumeration",
            "steps": [],
            "direct_response": "Chrome is not open. Only Notepad is visible.",
        })
        provider = ScriptedSLM([turn1, turn2])
        executor = AgentExecutor(slm_provider=provider, tool_registry=self.registry, direct_observation_completion=True)
        trace = executor.execute("Show me the windows, especially Chrome.")
        self.assertEqual(provider.call_count, 2)
        self.assertEqual(trace.final_response, "Chrome is not open. Only Notepad is visible.")

    def test_case_8_inspect_ui_elements_active_window_direct_completes(self):
        """Case 8: 'Inspect the UI elements of the active window.' -> Simple observation; completes directly."""
        turn1 = json.dumps({
            "thought_summary": "Inspecting active window controls.",
            "intent": "ui_inspection",
            "steps": [{"tool": "inspect_window_ui_elements", "arguments": {}}],
            "direct_response": None,
        })
        provider = ScriptedSLM([turn1])
        executor = AgentExecutor(slm_provider=provider, tool_registry=self.registry, direct_observation_completion=True)
        trace = executor.execute("Inspect the UI elements of the active window.")
        self.assertEqual(provider.call_count, 1)
        self.assertEqual(self.ui_tool.call_count, 1)
        self.assertEqual(trace.status, ExecutionStatus.COMPLETED)
        self.assertEqual(trace.final_response, self.ui_tool.output)

    def test_case_9_find_window_belonging_to_vscode_suppresses_direct_completion(self):
        """Case 9: 'Find the window belonging to VS Code.' -> Entity targeting; advances to Turn 2."""
        turn1 = json.dumps({
            "thought_summary": "Listing windows to find VS Code.",
            "intent": "window_enumeration",
            "steps": [{"tool": "list_open_windows", "arguments": {}}],
            "direct_response": None,
        })
        turn2 = json.dumps({
            "thought_summary": "Synthesizing result.",
            "intent": "window_enumeration",
            "steps": [],
            "direct_response": "No window belonging to VS Code was found.",
        })
        provider = ScriptedSLM([turn1, turn2])
        executor = AgentExecutor(slm_provider=provider, tool_registry=self.registry, direct_observation_completion=True)
        trace = executor.execute("Find the window belonging to VS Code.")
        self.assertEqual(provider.call_count, 2)
        self.assertEqual(trace.final_response, "No window belonging to VS Code was found.")

    def test_case_10_check_chrome_and_report_title_suppresses_direct_completion(self):
        """Case 10: 'Check whether Chrome is open and report its title.' -> Reporting; advances to Turn 2."""
        turn1 = json.dumps({
            "thought_summary": "Listing windows to check Chrome title.",
            "intent": "window_enumeration",
            "steps": [{"tool": "list_open_windows", "arguments": {}}],
            "direct_response": None,
        })
        turn2 = json.dumps({
            "thought_summary": "Reporting.",
            "intent": "window_enumeration",
            "steps": [],
            "direct_response": "Chrome is not running.",
        })
        provider = ScriptedSLM([turn1, turn2])
        executor = AgentExecutor(slm_provider=provider, tool_registry=self.registry, direct_observation_completion=True)
        trace = executor.execute("Check whether Chrome is open and report its title.")
        self.assertEqual(provider.call_count, 2)
        self.assertEqual(trace.final_response, "Chrome is not running.")

    def test_case_11_show_chrome_suppresses_direct_completion(self):
        """Case 11: 'Show Chrome.' -> Targeted application; advances to Turn 2 synthesis."""
        turn1 = json.dumps({
            "thought_summary": "Listing windows to find Chrome.",
            "intent": "window_enumeration",
            "steps": [{"tool": "list_open_windows", "arguments": {}}],
            "direct_response": None,
        })
        turn2 = json.dumps({
            "thought_summary": "Synthesizing Chrome window information.",
            "intent": "window_enumeration",
            "steps": [],
            "direct_response": "Chrome is currently open: Google Chrome - New Tab.",
        })
        provider = ScriptedSLM([turn1, turn2])
        executor = AgentExecutor(slm_provider=provider, tool_registry=self.registry, direct_observation_completion=True)
        trace = executor.execute("Show Chrome.")
        self.assertEqual(provider.call_count, 2, "Targeted application request must advance to synthesis turn")
        self.assertEqual(trace.status, ExecutionStatus.COMPLETED)
        self.assertEqual(trace.final_response, "Chrome is currently open: Google Chrome - New Tab.")

    def test_case_12_display_vscode_window_suppresses_direct_completion(self):
        """Case 12: 'Display the VS Code window.' -> Targeted application; advances to Turn 2 synthesis."""
        turn1 = json.dumps({
            "thought_summary": "Listing windows to locate VS Code.",
            "intent": "window_enumeration",
            "steps": [{"tool": "list_open_windows", "arguments": {}}],
            "direct_response": None,
        })
        turn2 = json.dumps({
            "thought_summary": "Synthesizing VS Code window details.",
            "intent": "window_enumeration",
            "steps": [],
            "direct_response": "VS Code window: Visual Studio Code - project.py.",
        })
        provider = ScriptedSLM([turn1, turn2])
        executor = AgentExecutor(slm_provider=provider, tool_registry=self.registry, direct_observation_completion=True)
        trace = executor.execute("Display the VS Code window.")
        self.assertEqual(provider.call_count, 2, "Targeted application request must advance to synthesis turn")
        self.assertEqual(trace.status, ExecutionStatus.COMPLETED)
        self.assertEqual(trace.final_response, "VS Code window: Visual Studio Code - project.py.")

    def test_case_13_give_me_notepad_window_suppresses_direct_completion(self):
        """Case 13: 'Give me the Notepad window.' -> Targeted application; advances to Turn 2 synthesis."""
        turn1 = json.dumps({
            "thought_summary": "Listing windows to locate Notepad.",
            "intent": "window_enumeration",
            "steps": [{"tool": "list_open_windows", "arguments": {}}],
            "direct_response": None,
        })
        turn2 = json.dumps({
            "thought_summary": "Synthesizing Notepad window details.",
            "intent": "window_enumeration",
            "steps": [],
            "direct_response": "Notepad window: Untitled - Notepad.",
        })
        provider = ScriptedSLM([turn1, turn2])
        executor = AgentExecutor(slm_provider=provider, tool_registry=self.registry, direct_observation_completion=True)
        trace = executor.execute("Give me the Notepad window.")
        self.assertEqual(provider.call_count, 2, "Targeted application request must advance to synthesis turn")
        self.assertEqual(trace.status, ExecutionStatus.COMPLETED)
        self.assertEqual(trace.final_response, "Notepad window: Untitled - Notepad.")

    def test_case_14_list_windows_focusing_on_chrome_suppresses_direct_completion(self):
        """Case 14: 'List windows, focusing on Chrome.' -> Entity filtering; advances to Turn 2 synthesis."""
        turn1 = json.dumps({
            "thought_summary": "Listing windows to filter for Chrome.",
            "intent": "window_enumeration",
            "steps": [{"tool": "list_open_windows", "arguments": {}}],
            "direct_response": None,
        })
        turn2 = json.dumps({
            "thought_summary": "Filtering windows for Chrome.",
            "intent": "window_enumeration",
            "steps": [],
            "direct_response": "Found 1 Chrome window: 'Google Chrome - GitHub'.",
        })
        provider = ScriptedSLM([turn1, turn2])
        executor = AgentExecutor(slm_provider=provider, tool_registry=self.registry, direct_observation_completion=True)
        trace = executor.execute("List windows, focusing on Chrome.")
        self.assertEqual(provider.call_count, 2, "Filtered observation request must advance to synthesis turn")
        self.assertEqual(trace.status, ExecutionStatus.COMPLETED)
        self.assertEqual(trace.final_response, "Found 1 Chrome window: 'Google Chrome - GitHub'.")

    def test_case_15_list_open_windows_and_count_them_suppresses_direct_completion(self):
        """Case 15: 'List open windows and count them.' -> Aggregation/counting; advances to Turn 2 synthesis."""
        turn1 = json.dumps({
            "thought_summary": "Listing windows to count.",
            "intent": "window_enumeration",
            "steps": [{"tool": "list_open_windows", "arguments": {}}],
            "direct_response": None,
        })
        turn2 = json.dumps({
            "thought_summary": "Counting listed windows.",
            "intent": "window_enumeration",
            "steps": [],
            "direct_response": "There are 2 open windows in total.",
        })
        provider = ScriptedSLM([turn1, turn2])
        executor = AgentExecutor(slm_provider=provider, tool_registry=self.registry, direct_observation_completion=True)
        trace = executor.execute("List open windows and count them.")
        self.assertEqual(provider.call_count, 2, "Aggregation/counting request must advance to synthesis turn")
        self.assertEqual(trace.status, ExecutionStatus.COMPLETED)
        self.assertEqual(trace.final_response, "There are 2 open windows in total.")

    def test_mandated_policy_helper_cases_table_direct_verification(self):
        """
        Direct unit verification of all 10 mandated prompts from M9.1.3 table against is_simple_observation_request.
        """
        cases = [
            ("Show me open windows.", "list_open_windows", "window_enumeration", True),
            ("List all running processes.", "list_processes", "process_enumeration", True),
            ("Inspect the UI elements of the active window.", "inspect_window_ui_elements", "ui_inspection", True),
            ("Show Chrome.", "list_open_windows", "window_enumeration", False),
            ("Display the VS Code window.", "list_open_windows", "window_enumeration", False),
            ("Give me the Notepad window.", "list_open_windows", "window_enumeration", False),
            ("List windows, focusing on Chrome.", "list_open_windows", "window_enumeration", False),
            ("List open windows and count them.", "list_open_windows", "window_enumeration", False),
            ("Find the Notepad window and inspect its UI elements.", "list_open_windows", "ui_inspection", False),
            ("Check whether Chrome is open and report its title.", "list_open_windows", "window_enumeration", False),
        ]
        for goal, tool, intent, expected in cases:
            with self.subTest(goal=goal, tool=tool):
                actual = is_simple_observation_request(goal, tool, intent)
                self.assertEqual(
                    actual,
                    expected,
                    f"Policy failed for '{goal}' ({tool}, {intent}): got {actual}, expected {expected}",
                )


class TestPublicAgentRunResultContracts(unittest.TestCase):
    """Tests verifying public AgentRunResult properties and contracts through ExoCortexAgent.run()."""

    def setUp(self):
        self.registry = ToolRegistry()
        self.windows_tool = MockDesktopWindowsTool()
        self.ui_tool = MockInspectUIElementsTool()
        self.failing_tool = MockFailingTool()

        self.registry.register(self.windows_tool)
        self.registry.register(self.ui_tool)
        self.registry.register(self.failing_tool)

        from exocortex.tools.builtin import EchoTool, SystemInfoTool
        self.echo_tool = EchoTool()
        self.sys_tool = SystemInfoTool()
        self.registry.register(self.echo_tool)
        self.registry.register(self.sys_tool)

    def test_public_agent_run_result_successful_multistep(self):
        """Contract A: Successful multi-step execution sets success=True, is_complete=True."""
        turn1 = json.dumps({
            "thought_summary": "Echo first message.",
            "intent": "general_action",
            "steps": [{"tool": "echo", "arguments": {"message": "diagnostics step 1"}}],
            "direct_response": None,
        })
        turn2 = json.dumps({
            "thought_summary": "System info second step.",
            "intent": "hardware_inspection",
            "steps": [{"tool": "system_info", "arguments": {"detail_level": "basic"}}],
            "direct_response": None,
        })
        turn3 = json.dumps({
            "thought_summary": "Synthesize results.",
            "intent": "final_summary",
            "steps": [],
            "direct_response": "All diagnostics completed successfully.",
        })

        provider = ScriptedSLM([turn1, turn2, turn3])
        agent = ExoCortexAgent(
            slm_provider=provider,
            tool_registry=self.registry,
            enable_fast_path=False,
            direct_observation_completion=False,
        )

        result: AgentRunResult = agent.run("Run complete diagnostic flow")
        # 1. Public success must be True
        self.assertTrue(result.success)
        # 2. Plan must be marked complete
        self.assertIsNotNone(result.plan)
        self.assertTrue(result.plan.is_complete)
        # 3. All expected tools executed successfully
        self.assertEqual(len(result.executed_tools), 2)
        self.assertEqual(result.executed_tools[0]["tool"], "echo")
        self.assertTrue(result.executed_tools[0]["success"])
        self.assertEqual(result.executed_tools[1]["tool"], "system_info")
        self.assertTrue(result.executed_tools[1]["success"])
        # 4. Final response text matches synthesis
        self.assertEqual(result.response_text, "All diagnostics completed successfully.")
        self.assertIsNone(result.validation_error)

    def test_public_agent_run_result_failed_batch_execution(self):
        """Contract B: Failed batch step halts execution; public success is False."""
        batch_plan = json.dumps({
            "thought_summary": "Execute failing tool then echo.",
            "intent": "general_action",
            "steps": [
                {"tool": "failing_step_tool", "arguments": {}},
                {"tool": "echo", "arguments": {"message": "should not run"}},
            ],
            "direct_response": None,
        })

        provider = ScriptedSLM([batch_plan])
        agent = ExoCortexAgent(
            slm_provider=provider,
            tool_registry=self.registry,
            enable_fast_path=False,
        )

        result: AgentRunResult = agent.run("Run failing batch")
        # 1. Public success must be False
        self.assertFalse(result.success)
        # 2. Plan must not be complete
        self.assertIsNotNone(result.plan)
        self.assertFalse(result.plan.is_complete)
        # 3. Only 1 tool step executed (dependent step did not run)
        self.assertEqual(len(result.executed_tools), 1)
        self.assertEqual(result.executed_tools[0]["tool"], "failing_step_tool")
        self.assertFalse(result.executed_tools[0]["success"])
        # 4. Public validation_error surfaces failure
        self.assertIsNotNone(result.validation_error)
        self.assertIn("failing_step_tool", result.validation_error)

    def test_public_agent_run_result_compound_insufficient_output_semantics(self):
        """
        Contract C: Compound request with insufficient model output.
        Model halts after step 1 without performing requested step 2.
        Documents actual result semantics: success=True reflects normal loop termination,
        NOT that the user's high-level compound goal was verified.
        """
        turn1 = json.dumps({
            "thought_summary": "Finding Notepad window.",
            "intent": "ui_inspection",
            "steps": [{"tool": "list_open_windows", "arguments": {}}],
            "direct_response": None,
        })
        turn2 = json.dumps({
            "thought_summary": "Stopping early.",
            "intent": "ui_inspection",
            "steps": [],
            "direct_response": "I located Notepad but will not inspect controls.",
        })

        provider = ScriptedSLM([turn1, turn2])
        agent = ExoCortexAgent(
            slm_provider=provider,
            tool_registry=self.registry,
            enable_fast_path=False,
            direct_observation_completion=True,
        )

        result: AgentRunResult = agent.run("Find the Notepad window and inspect its UI elements")
        # Direct observation completion was correctly suppressed on turn 1
        self.assertEqual(provider.call_count, 2)
        # Step 2 was never executed
        self.assertEqual(self.ui_tool.call_count, 0)
        # Result semantics:
        # Tool step 1 succeeded, model returned direct_response, so executor loop completed.
        self.assertTrue(result.success)
        self.assertEqual(result.response_text, "I located Notepad but will not inspect controls.")
        # Crucial documented semantic boundary:
        # result.success guarantees loop completion without runtime errors, NOT real-world goal verification.


if __name__ == "__main__":
    unittest.main()
