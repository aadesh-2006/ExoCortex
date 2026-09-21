"""
ExoCortex cognitive loop controller and agent orchestrator.

Implements the fundamental architecture:
User -> Local SLM -> Intent Understanding -> Agent Planner -> Tool Selection ->
Computer Interaction -> Observe Result -> Reason Again -> Next Action -> Task Completion.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from exocortex.config import Settings, get_config
from exocortex.intent import IntentAnalysis, IntentEngine
from exocortex.planner import AgentPlanner, ExecutionPlan, StepStatus
from exocortex.slm.base import Role, SLMMessage, SLMProvider, SLMResponse
from exocortex.slm.factory import get_slm_provider
from exocortex.tools.registry import ToolRegistry, get_default_tool_registry

logger = logging.getLogger("exocortex.agent")


@dataclass
class AgentRunResult:
    """Structured result returned upon completion of an agent run."""
    prompt: str
    response_text: str
    success: bool
    plan: ExecutionPlan
    executed_tools: List[Dict[str, Any]] = field(default_factory=list)
    total_latency_ms: float = 0.0
    slm_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "response_text": self.response_text,
            "success": self.success,
            "plan": self.plan.to_dict(),
            "executed_tools": self.executed_tools,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "slm_info": self.slm_info,
        }


class ExoCortexAgent:
    """
    JARVIS-like Autonomous Personal Computer Agent Core.
    
    Orchestrates local SLM reasoning, intent extraction, planning, safe tool execution,
    and iterative result observation on Windows.
    """

    def __init__(
        self,
        config: Optional[Settings] = None,
        slm_provider: Optional[SLMProvider] = None,
        tool_registry: Optional[ToolRegistry] = None,
    ) -> None:
        self.config = config or get_config()
        self.slm_provider = slm_provider or get_slm_provider(config=self.config)
        self.tool_registry = tool_registry or get_default_tool_registry()
        self.intent_engine = IntentEngine(slm_provider=self.slm_provider)
        self.planner = AgentPlanner()
        self.history: List[SLMMessage] = []

    def run(self, user_prompt: str) -> AgentRunResult:
        """
        Execute the complete autonomous cognitive loop for a user prompt.
        """
        start_time = time.perf_counter()
        logger.info("Starting ExoCortex cognitive loop for prompt: '%s'", user_prompt)

        # 1. Record user message in history
        self.history.append(SLMMessage(role="user", content=user_prompt))

        # 2. Intent Understanding
        intent: IntentAnalysis = self.intent_engine.analyze(user_prompt)
        logger.debug("Analyzed intent: %s", intent.primary_category.value)

        # 3. Agent Planner
        plan: ExecutionPlan = self.planner.create_plan(intent)
        logger.debug("Generated execution plan with %d step(s)", plan.total_steps)

        executed_tools_record: List[Dict[str, Any]] = []
        observations: List[str] = []

        # 4. Step-by-step Execution Loop
        while not plan.is_complete:
            step = plan.get_current_step()
            if not step:
                break

            step.status = StepStatus.IN_PROGRESS
            logger.info("Executing plan step %d: %s", step.step_id, step.description)

            if step.tool_name:
                # Tool Selection & Computer Interaction
                tool_res = self.tool_registry.execute(step.tool_name, step.arguments)
                
                tool_record = {
                    "step_id": step.step_id,
                    "tool": step.tool_name,
                    "arguments": step.arguments,
                    "success": tool_res.success,
                    "output": tool_res.output,
                    "error": tool_res.error,
                    "execution_time_ms": tool_res.execution_time_ms,
                }
                executed_tools_record.append(tool_record)

                if tool_res.success:
                    step.status = StepStatus.COMPLETED
                    step.result = tool_res.output
                    observations.append(f"[Step {step.step_id} Output]: {tool_res.output}")
                else:
                    step.status = StepStatus.FAILED
                    step.error = tool_res.error
                    observations.append(f"[Step {step.step_id} Error]: {tool_res.error}")
            else:
                # Direct SLM reasoning step
                slm_res = self.slm_provider.generate(user_prompt)
                step.status = StepStatus.COMPLETED
                step.result = slm_res.content
                observations.append(f"[SLM Reasoning]: {slm_res.content}")

            # Advance plan
            plan.advance()

        # 5. Synthesize Observations & Reason Again for Final Response
        if observations:
            if len(observations) == 1 and not executed_tools_record:
                final_response = observations[0].replace("[SLM Reasoning]: ", "")
            else:
                final_response = "\n\n".join(observations)
        else:
            final_response = "Task processed with no output."

        self.history.append(SLMMessage(role="assistant", content=final_response))
        total_time = (time.perf_counter() - start_time) * 1000

        all_steps_succeeded = all(s.status == StepStatus.COMPLETED for s in plan.steps)

        return AgentRunResult(
            prompt=user_prompt,
            response_text=final_response,
            success=all_steps_succeeded,
            plan=plan,
            executed_tools=executed_tools_record,
            total_latency_ms=total_time,
            slm_info=self.slm_provider.get_model_info(),
        )
