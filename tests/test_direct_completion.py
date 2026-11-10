"""
Unit tests for Direct Observation Completion Policy in ExoCortex AgentExecutor.
"""

from __future__ import annotations

import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

from exocortex.execution.executor import AgentExecutor, DIRECT_COMPLETION_TOOLS
from exocortex.execution.state import ExecutionStatus
from exocortex.slm.base import SLMMessage, SLMProvider, SLMResponse
from exocortex.tools.base import BaseTool, PermissionLevel, ToolParameter, ToolResult
from exocortex.tools.registry import ToolRegistry


class ControllableMockSLMProvider(SLMProvider):
    """Mock SLM Provider that returns pre-programmed sequential responses and tracks call counts."""

    def __init__(self, responses: List[str]):
        self.responses = list(responses)
        self.call_count = 0
        self.received_prompts: List[str] = []

    def is_available(self) -> bool:
        return True

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": "mock-controllable-slm",
            "provider": "ControllableMockSLMProvider",
            "is_local": True,
        }

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
        **kwargs: Any,
    ) -> SLMResponse:
        self.call_count += 1
        self.received_prompts.append(prompt)
        if not self.responses:
            raise RuntimeError("ControllableMockSLMProvider has no more programmed responses.")
        content = self.responses.pop(0)
        return SLMResponse(
            content=content,
            prompt_tokens=10,
            completion_tokens=10,
            latency_ms=1.0,
            model_name="mock-controllable-slm",
            provider_name="ControllableMockSLMProvider",
        )

    def chat(self, messages: List[SLMMessage], **kwargs: Any) -> SLMResponse:
        return self.generate(prompt=messages[-1].content if messages else "")


class SimpleDummyTool(BaseTool):
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
                error=f"{self.name} failed.",
            )
        return ToolResult(
            success=True,
            output=self.output_text,
        )


class TestDirectObservationCompletion(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()
        self.windows_tool = SimpleDummyTool(
            "list_open_windows",
            output_text="Visible Desktop Windows (2 found):\n1. Notepad\n2. Terminal",
        )
        self.processes_tool = SimpleDummyTool(
            "list_processes",
            output_text="Running Processes (1 found):\n1. python.exe",
        )
        self.uia_tool = SimpleDummyTool(
            "inspect_window_ui_elements",
            output_text="UI Elements for Window 'Notepad':\n- Document: 'Untitled'",
        )
        self.web_tool = SimpleDummyTool(
            "web_search",
            output_text="Search Results: 1. Python docs",
        )
        self.registry.register(self.windows_tool)
        self.registry.register(self.processes_tool)
        self.registry.register(self.uia_tool)
        self.registry.register(self.web_tool)

    def test_single_tool_desktop_completes_in_one_call(self):
        # Program SLM to plan list_open_windows
        plan_response = """```json
{
  "thought_summary": "Checking visible windows.",
  "intent": "window_enumeration",
  "steps": [{"tool": "list_open_windows", "arguments": {}}],
  "requires_confirmation": false,
  "direct_response": null
}
```"""
        provider = ControllableMockSLMProvider([plan_response])
        executor = AgentExecutor(
            slm_provider=provider,
            tool_registry=self.registry,
            direct_observation_completion=True,
        )

        trace = executor.execute("Show me my desktop windows")

        # 1. Exactly 1 LLM call occurred
        self.assertEqual(provider.call_count, 1)
        # 2. Tool was executed
        self.assertEqual(self.windows_tool.call_count, 1)
        # 3. Status is COMPLETED and final response is the tool output
        self.assertEqual(trace.status, ExecutionStatus.COMPLETED)
        self.assertEqual(trace.final_response, self.windows_tool.output_text)
        self.assertEqual(trace.total_steps, 1)

    def test_inspect_window_ui_elements_completes_in_one_call(self):
        plan_response = """```json
{
  "thought_summary": "Inspecting notepad controls.",
  "intent": "ui_inspection",
  "steps": [{"tool": "inspect_window_ui_elements", "arguments": {}}],
  "requires_confirmation": false,
  "direct_response": null
}
```"""
        provider = ControllableMockSLMProvider([plan_response])
        executor = AgentExecutor(
            slm_provider=provider,
            tool_registry=self.registry,
            direct_observation_completion=True,
        )

        trace = executor.execute("Inspect controls in notepad")
        self.assertEqual(provider.call_count, 1)
        self.assertEqual(self.uia_tool.call_count, 1)
        self.assertEqual(trace.status, ExecutionStatus.COMPLETED)
        self.assertEqual(trace.final_response, self.uia_tool.output_text)

    def test_list_processes_completes_in_one_call(self):
        plan_response = """```json
{
  "thought_summary": "Listing running processes.",
  "intent": "process_inspection",
  "steps": [{"tool": "list_processes", "arguments": {}}],
  "requires_confirmation": false,
  "direct_response": null
}
```"""
        provider = ControllableMockSLMProvider([plan_response])
        executor = AgentExecutor(
            slm_provider=provider,
            tool_registry=self.registry,
            direct_observation_completion=True,
        )

        trace = executor.execute("What processes are running")
        self.assertEqual(provider.call_count, 1)
        self.assertEqual(self.processes_tool.call_count, 1)
        self.assertEqual(trace.status, ExecutionStatus.COMPLETED)
        self.assertEqual(trace.final_response, self.processes_tool.output_text)

    def test_disabled_policy_continues_to_synthesis_call(self):
        plan_response = """```json
{
  "thought_summary": "Checking visible windows.",
  "intent": "window_enumeration",
  "steps": [{"tool": "list_open_windows", "arguments": {}}],
  "requires_confirmation": false,
  "direct_response": null
}
```"""
        synthesis_response = """```json
{
  "thought_summary": "Synthesized observation.",
  "intent": "final_summary",
  "steps": [],
  "requires_confirmation": false,
  "direct_response": "Synthesized windows summary."
}
```"""
        provider = ControllableMockSLMProvider([plan_response, synthesis_response])
        executor = AgentExecutor(
            slm_provider=provider,
            tool_registry=self.registry,
            direct_observation_completion=False,  # disabled!
        )

        trace = executor.execute("Show me my desktop windows")

        # Must have made 2 LLM calls when policy is disabled
        self.assertEqual(provider.call_count, 2)
        self.assertEqual(trace.status, ExecutionStatus.COMPLETED)
        self.assertEqual(trace.final_response, "Synthesized windows summary.")

    def test_web_search_does_not_direct_complete(self):
        # Web tools must NEVER direct complete; they require synthesis
        plan_response = """```json
{
  "thought_summary": "Searching web.",
  "intent": "web_search",
  "steps": [{"tool": "web_search", "arguments": {}}],
  "requires_confirmation": false,
  "direct_response": null
}
```"""
        synthesis_response = """```json
{
  "thought_summary": "Synthesizing search.",
  "intent": "final_summary",
  "steps": [],
  "requires_confirmation": false,
  "direct_response": "Here are the web results summarized."
}
```"""
        provider = ControllableMockSLMProvider([plan_response, synthesis_response])
        executor = AgentExecutor(
            slm_provider=provider,
            tool_registry=self.registry,
            direct_observation_completion=True,  # enabled, but tool is web_search
        )

        trace = executor.execute("Search for Python")
        # Must execute second LLM call
        self.assertEqual(provider.call_count, 2)
        self.assertEqual(trace.final_response, "Here are the web results summarized.")

    def test_multi_tool_plan_does_not_direct_complete_early(self):
        # Multi-tool plan in decision: model plans 2 tools
        multi_tool_plan_response = """```json
{
  "thought_summary": "Listing both windows and processes.",
  "intent": "full_inspection",
  "steps": [
    {"tool": "list_open_windows", "arguments": {}},
    {"tool": "list_processes", "arguments": {}}
  ],
  "requires_confirmation": false,
  "direct_response": null
}
```"""
        # Because direct observation completion is NOT triggered for multi-tool plans,
        # the executor must query the SLM again for the follow-up / synthesis
        synthesis_response = """```json
{
  "thought_summary": "Synthesized both.",
  "intent": "final_summary",
  "steps": [],
  "requires_confirmation": false,
  "direct_response": "Both windows and processes inspected."
}
```"""
        provider = ControllableMockSLMProvider([multi_tool_plan_response, synthesis_response])
        executor = AgentExecutor(
            slm_provider=provider,
            tool_registry=self.registry,
            direct_observation_completion=True,
        )

        trace = executor.execute("Inspect everything")
        # Multi-tool plan must not trigger direct completion, so 2 LLM calls are made
        self.assertEqual(provider.call_count, 2)
        self.assertEqual(self.windows_tool.call_count, 1)
        self.assertEqual(trace.final_response, "Both windows and processes inspected.")

    def test_tool_failure_does_not_direct_complete(self):
        failing_tool = SimpleDummyTool("list_open_windows", fail=True)
        reg = ToolRegistry()
        reg.register(failing_tool)

        plan_response = """```json
{
  "thought_summary": "Checking visible windows.",
  "intent": "window_enumeration",
  "steps": [{"tool": "list_open_windows", "arguments": {}}],
  "requires_confirmation": false,
  "direct_response": null
}
```"""
        recovery_response = """```json
{
  "thought_summary": "Handling tool failure.",
  "intent": "final_summary",
  "steps": [],
  "requires_confirmation": false,
  "direct_response": "Apologies, listing windows encountered an error."
}
```"""
        provider = ControllableMockSLMProvider([plan_response, recovery_response])
        executor = AgentExecutor(
            slm_provider=provider,
            tool_registry=reg,
            direct_observation_completion=True,
        )

        trace = executor.execute("Show windows")
        # Direct completion must not trigger on failed tool
        self.assertEqual(provider.call_count, 2)
        self.assertEqual(trace.final_response, "Apologies, listing windows encountered an error.")

    def test_empty_output_does_not_direct_complete(self):
        empty_tool = SimpleDummyTool("list_open_windows", output_text="   ")
        reg = ToolRegistry()
        reg.register(empty_tool)

        plan_response = """```json
{
  "thought_summary": "Checking visible windows.",
  "intent": "window_enumeration",
  "steps": [{"tool": "list_open_windows", "arguments": {}}],
  "requires_confirmation": false,
  "direct_response": null
}
```"""
        synthesis_response = """```json
{
  "thought_summary": "Handling empty output.",
  "intent": "final_summary",
  "steps": [],
  "requires_confirmation": false,
  "direct_response": "No windows were returned."
}
```"""
        provider = ControllableMockSLMProvider([plan_response, synthesis_response])
        executor = AgentExecutor(
            slm_provider=provider,
            tool_registry=reg,
            direct_observation_completion=True,
        )

        trace = executor.execute("Show windows")
        self.assertEqual(provider.call_count, 2)
        self.assertEqual(trace.final_response, "No windows were returned.")


if __name__ == "__main__":
    unittest.main()
