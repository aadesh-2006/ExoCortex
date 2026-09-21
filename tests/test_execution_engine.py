"""
Unit and integration tests for the ExoCortex Milestone 4 Safe Agentic Execution Loop.
"""

import json
import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from exocortex.agent import AgentRunResult, ExoCortexAgent
from exocortex.execution.executor import (
    DEFAULT_MAX_IDENTICAL_ACTIONS,
    DEFAULT_MAX_RESULT_CHARS,
    DEFAULT_MAX_STEPS,
    AgentExecutor,
)
from exocortex.execution.state import (
    ActionHistoryTracker,
    ActionType,
    ExecutionState,
    ExecutionStatus,
    ExecutionStep,
)
from exocortex.execution.trace import ExecutionTrace
from exocortex.schema import AgentDecision, StructuredStep
from exocortex.slm.base import SLMMessage, SLMProvider, SLMResponse
from exocortex.slm.mock_provider import MockSLMProvider
from exocortex.tools.base import BaseTool, PermissionLevel, ToolParameter, ToolResult
from exocortex.tools.builtin import EchoTool, HealthCheckTool, SystemInfoTool
from exocortex.tools.registry import ToolRegistry


class ScriptedSLMProvider(SLMProvider):
    """
    Test helper SLM provider that returns pre-scripted responses in sequence.
    """

    def __init__(self, responses: List[str]):
        self.responses = list(responses)
        self.call_count = 0
        self.history_prompts: List[str] = []

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
        **kwargs: Any,
    ) -> SLMResponse:
        self.history_prompts.append(prompt)
        if self.call_count < len(self.responses):
            resp_content = self.responses[self.call_count]
            self.call_count += 1
        else:
            resp_content = json.dumps({
                "thought_summary": "All scripted steps completed.",
                "intent": "general_query",
                "steps": [],
                "direct_response": "Finished all scripted steps."
            })
        return SLMResponse(
            content=resp_content,
            prompt_tokens=10,
            completion_tokens=20,
            latency_ms=5.0,
            model_name="scripted",
            provider_name="ScriptedSLMProvider",
        )

    def chat(
        self,
        messages: List[SLMMessage],
        temperature: float = 0.1,
        max_tokens: int = 512,
        **kwargs: Any,
    ) -> SLMResponse:
        combined_prompt = "\n".join(f"{m.role}: {m.content}" for m in messages)
        return self.generate(prompt=combined_prompt, temperature=temperature, max_tokens=max_tokens)

    def is_available(self) -> bool:
        return True

    def get_model_info(self) -> Dict[str, Any]:
        return {"provider": "ScriptedSLMProvider", "model_id": "scripted"}


class FailingTool(BaseTool):
    name = "failing_tool"
    description = "A tool that always fails for testing error handling"
    parameters = []
    permission_level = PermissionLevel.SAFE

    def run(self, **kwargs) -> ToolResult:
        return ToolResult(
            success=False,
            output="",
            error="Simulated tool execution failure."
        )


class LongOutputTool(BaseTool):
    name = "long_output_tool"
    description = "A tool that generates huge text output"
    parameters = []
    permission_level = PermissionLevel.SAFE

    def run(self, **kwargs) -> ToolResult:
        return ToolResult(
            success=True,
            output="X" * 5000,
            error=None
        )


class TestExecutionEngine(unittest.TestCase):
    def setUp(self):
        self.mock_provider = MockSLMProvider()
        self.registry = ToolRegistry()
        self.registry.register(EchoTool())
        self.registry.register(HealthCheckTool())
        self.registry.register(SystemInfoTool())

    def test_single_tool_execution(self):
        """Verify single tool execution through executor."""
        executor = AgentExecutor(
            slm_provider=self.mock_provider,
            tool_registry=self.registry,
        )
        trace = executor.execute("Show system hardware specs")
        self.assertEqual(trace.status, ExecutionStatus.COMPLETED)
        self.assertGreater(len(trace.steps), 0)
        tool_steps = [s for s in trace.steps if s.action_type == ActionType.TOOL_CALL]
        self.assertEqual(len(tool_steps), 1)
        self.assertEqual(tool_steps[0].tool_name, "system_info")
        self.assertTrue(tool_steps[0].tool_result.success)
        self.assertIn("OS:", tool_steps[0].tool_result.output)

    def test_multi_step_progression_with_feedback(self):
        """Verify multi-step progression where SLM observes tool output and yields final response."""
        step1_json = json.dumps({
            "thought_summary": "I need to run echo first.",
            "intent": "system_info",
            "steps": [{"tool": "echo", "arguments": {"message": "Step 1 complete"}}],
            "direct_response": None
        })
        step2_json = json.dumps({
            "thought_summary": "Echo completed. Now returning final summary.",
            "intent": "general_query",
            "steps": [],
            "direct_response": "All steps executed successfully."
        })

        scripted_slm = ScriptedSLMProvider([step1_json, step2_json])
        executor = AgentExecutor(
            slm_provider=scripted_slm,
            tool_registry=self.registry,
        )

        trace = executor.execute("Echo something and then finish")
        self.assertEqual(trace.status, ExecutionStatus.COMPLETED)
        self.assertEqual(len(trace.steps), 2)
        self.assertEqual(trace.steps[0].action_type, ActionType.TOOL_CALL)
        self.assertEqual(trace.steps[0].tool_name, "echo")
        self.assertEqual(trace.steps[1].action_type, ActionType.FINAL_RESPONSE)
        self.assertEqual(trace.final_response, "All steps executed successfully.")

        # Verify second prompt contained execution history
        self.assertIn("PREVIOUS ACTIONS AND OBSERVATIONS", scripted_slm.history_prompts[1])
        self.assertIn("Step 1 complete", scripted_slm.history_prompts[1])

    def test_conversational_final_response_no_tools(self):
        """Verify direct conversational answers with no tools."""
        executor = AgentExecutor(
            slm_provider=self.mock_provider,
            tool_registry=self.registry,
        )
        trace = executor.execute("Hello, how are you?")
        self.assertEqual(trace.status, ExecutionStatus.COMPLETED)
        self.assertEqual(len(trace.steps), 1)
        self.assertEqual(trace.steps[0].action_type, ActionType.FINAL_RESPONSE)
        self.assertIn("ExoCortex", trace.final_response)

    def test_max_steps_enforcement(self):
        """Verify execution terminates safely when MAX_STEPS is reached."""
        infinite_step_json = json.dumps({
            "thought_summary": "Repeating echo.",
            "intent": "general_query",
            "steps": [{"tool": "echo", "arguments": {"message": "loop"}}],
            "direct_response": None
        })
        # Use high identical action threshold to test step limit specifically
        scripted_slm = ScriptedSLMProvider([infinite_step_json] * 10)
        executor = AgentExecutor(
            slm_provider=scripted_slm,
            tool_registry=self.registry,
            max_steps=4,
            max_identical_actions=10,
        )
        trace = executor.execute("Run forever")
        self.assertEqual(trace.status, ExecutionStatus.MAX_STEPS_EXCEEDED)
        self.assertLessEqual(len(trace.steps), 4)
        self.assertIn("Maximum step limit", trace.error_message)

    def test_repeated_action_loop_detection(self):
        """Verify loop detection stops execution after MAX_IDENTICAL_ACTIONS = 2."""
        repeated_json = json.dumps({
            "thought_summary": "Repeating identical action.",
            "intent": "general_query",
            "steps": [{"tool": "echo", "arguments": {"message": "duplicate"}}],
            "direct_response": None
        })
        scripted_slm = ScriptedSLMProvider([repeated_json, repeated_json, repeated_json])
        executor = AgentExecutor(
            slm_provider=scripted_slm,
            tool_registry=self.registry,
            max_steps=8,
            max_identical_actions=2,
        )
        trace = executor.execute("Trigger loop")
        self.assertEqual(trace.status, ExecutionStatus.LOOP_DETECTED)
        self.assertIn("Loop detected", trace.error_message)

    def test_tool_failure_handling(self):
        """Verify tool execution failures are captured as error observations."""
        self.registry.register(FailingTool())
        step_json = json.dumps({
            "thought_summary": "Executing failing tool.",
            "intent": "general_query",
            "steps": [{"tool": "failing_tool", "arguments": {}}],
            "direct_response": None
        })
        finish_json = json.dumps({
            "thought_summary": "Handled failure.",
            "intent": "general_query",
            "steps": [],
            "direct_response": "I noticed the tool failed and recovered."
        })
        scripted_slm = ScriptedSLMProvider([step_json, finish_json])
        executor = AgentExecutor(
            slm_provider=scripted_slm,
            tool_registry=self.registry,
        )
        trace = executor.execute("Test failure recovery")
        self.assertEqual(trace.status, ExecutionStatus.COMPLETED)
        self.assertEqual(trace.steps[0].action_type, ActionType.TOOL_CALL)
        self.assertEqual(trace.steps[0].status, ExecutionStatus.ERROR)
        self.assertIn("Simulated tool execution failure", trace.steps[0].observation_text)
        self.assertEqual(trace.final_response, "I noticed the tool failed and recovered.")

    def test_confirmation_required_tool_pauses_execution(self):
        """Verify CONFIRMATION_REQUIRED tools halt execution when no callback is set."""
        class SensitiveTool(BaseTool):
            name = "restricted_action"
            description = "A restricted tool requiring user confirmation"
            parameters = []
            permission_level = PermissionLevel.CONFIRMATION_REQUIRED

            def run(self, **kwargs) -> ToolResult:
                return ToolResult(success=True, output="Executed restricted action", error=None)

        self.registry.register(SensitiveTool())
        step_json = json.dumps({
            "thought_summary": "Executing sensitive action.",
            "intent": "general_query",
            "steps": [{"tool": "restricted_action", "arguments": {}}],
            "direct_response": None
        })
        scripted_slm = ScriptedSLMProvider([step_json])
        executor = AgentExecutor(
            slm_provider=scripted_slm,
            tool_registry=self.registry,
        )
        trace = executor.execute("Run restricted action")
        self.assertEqual(trace.status, ExecutionStatus.CONFIRMATION_REQUIRED)
        self.assertIn("requires explicit user confirmation", trace.error_message)

    def test_invalid_slm_decision_fails_safely(self):
        """Verify malformed non-JSON SLM output is handled safely."""
        scripted_slm = ScriptedSLMProvider(["INVALID_NON_JSON_OUTPUT_1", "INVALID_NON_JSON_OUTPUT_2"])
        executor = AgentExecutor(
            slm_provider=scripted_slm,
            tool_registry=self.registry,
        )
        trace = executor.execute("Do something weird")
        self.assertEqual(trace.status, ExecutionStatus.ERROR)
        self.assertIn("SLM Reasoning Error", trace.error_message)

    def test_tool_result_truncation(self):
        """Verify huge tool output is bounded to max_result_chars."""
        self.registry.register(LongOutputTool())
        step_json = json.dumps({
            "thought_summary": "Executing long output tool.",
            "intent": "general_query",
            "steps": [{"tool": "long_output_tool", "arguments": {}}],
            "direct_response": None
        })
        finish_json = json.dumps({
            "thought_summary": "Done.",
            "intent": "general_query",
            "steps": [],
            "direct_response": "Output received."
        })
        scripted_slm = ScriptedSLMProvider([step_json, finish_json])
        executor = AgentExecutor(
            slm_provider=scripted_slm,
            tool_registry=self.registry,
            max_result_chars=500,
        )
        trace = executor.execute("Run long output")
        self.assertEqual(trace.status, ExecutionStatus.COMPLETED)
        obs = trace.steps[0].observation_text
        self.assertIn("Output bounded to 500 chars", obs)
        self.assertLessEqual(len(obs), 600)

    def test_execution_trace_serialization(self):
        """Verify ExecutionTrace serializes to dictionary and summary text properly."""
        executor = AgentExecutor(
            slm_provider=self.mock_provider,
            tool_registry=self.registry,
        )
        trace = executor.execute("Check system health")
        trace_dict = trace.to_dict()
        self.assertIn("user_goal", trace_dict)
        self.assertIn("steps", trace_dict)
        self.assertIn("status", trace_dict)
        self.assertIn("total_duration_ms", trace_dict)
        self.assertIn("total_steps", trace_dict)

        summary = trace.get_summary()
        self.assertIn("Execution Status", summary)
        self.assertIn("health_check", summary)

    def test_agent_integration_delegation(self):
        """Verify ExoCortexAgent.run() delegates to AgentExecutor seamlessly."""
        agent = ExoCortexAgent(slm_provider=self.mock_provider, tool_registry=self.registry)
        result = agent.run("Show system hardware specs")
        self.assertIsInstance(result, AgentRunResult)
        self.assertTrue(result.success)
        self.assertIsNotNone(result.trace)
        self.assertEqual(result.trace.status, ExecutionStatus.COMPLETED)
        self.assertEqual(len(result.executed_tools), 1)
        self.assertEqual(result.executed_tools[0]["tool"], "system_info")


if __name__ == "__main__":
    unittest.main()
