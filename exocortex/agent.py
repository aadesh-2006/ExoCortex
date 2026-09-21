"""
ExoCortex cognitive loop controller and agent orchestrator.

Implements the complete pipeline:
User Prompt -> Local SLM -> Structured Decision -> Tool Selection ->
Execution -> Observation -> Task Completion.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from exocortex.config import Settings, get_config
from exocortex.execution import AgentExecutor, ExecutionStatus, ExecutionTrace
from exocortex.planner import ExecutionPlan, PlanStep, StepStatus
from exocortex.schema import AgentDecision, StructuredStep
from exocortex.slm.base import Role, SLMMessage, SLMProvider, SLMResponse
from exocortex.slm.factory import get_slm_provider
from exocortex.tools.base import PermissionLevel
from exocortex.tools.registry import ToolRegistry, get_default_tool_registry

logger = logging.getLogger("exocortex.agent")


@dataclass
class AgentRunResult:
    """Structured result returned upon completion of an agent run."""
    prompt: str
    response_text: str
    success: bool
    decision: Optional[AgentDecision] = None
    plan: Optional[ExecutionPlan] = None
    executed_tools: List[Dict[str, Any]] = field(default_factory=list)
    total_latency_ms: float = 0.0
    slm_info: Dict[str, Any] = field(default_factory=dict)
    validation_error: Optional[str] = None
    trace: Optional[ExecutionTrace] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "response_text": self.response_text,
            "success": self.success,
            "decision": self.decision.to_dict() if self.decision else None,
            "plan": self.plan.to_dict() if self.plan else None,
            "executed_tools": self.executed_tools,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "slm_info": self.slm_info,
            "validation_error": self.validation_error,
            "trace": self.trace.to_dict() if self.trace else None,
        }


class ExoCortexAgent:
    """
    Local-first autonomous personal computer agent for Windows.
    
    Driven by local SLM structured multi-step execution loop.
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
        self.executor = AgentExecutor(
            slm_provider=self.slm_provider,
            tool_registry=self.tool_registry,
            max_steps=self.config.max_plan_steps,
        )
        self.history: List[SLMMessage] = []

    def run(self, user_prompt: str) -> AgentRunResult:
        """
        Execute the autonomous cognitive loop for a user prompt.
        """
        logger.info("Executing ExoCortex reasoning loop for: '%s'", user_prompt)
        self.history.append(SLMMessage(role="user", content=user_prompt))

        trace = self.executor.execute(user_prompt)

        # Collect executed tool steps for compatibility and reporting
        executed_tools_record: List[Dict[str, Any]] = []
        for s in trace.steps:
            if s.tool_name:
                executed_tools_record.append({
                    "step_id": s.step_number,
                    "tool": s.tool_name,
                    "arguments": s.arguments,
                    "success": s.tool_result.success if s.tool_result else False,
                    "output": s.tool_result.output if s.tool_result else "",
                    "error": s.tool_result.error if s.tool_result else s.error_message,
                    "execution_time_ms": s.duration_ms,
                })

        # Resolve primary decision
        primary_decision = None
        for s in trace.steps:
            if s.decision:
                primary_decision = s.decision
                break

        final_resp = trace.final_response or trace.error_message or "Task completed."
        self.history.append(SLMMessage(role="assistant", content=final_resp))

        is_success = (trace.status == ExecutionStatus.COMPLETED)

        # Construct legacy plan object for backward compatibility
        plan_steps = [
            PlanStep(
                step_id=t["step_id"],
                description=f"Execute tool '{t['tool']}'",
                tool_name=t["tool"],
                arguments=t["arguments"],
                status=StepStatus.COMPLETED if t["success"] else StepStatus.FAILED,
                result=t["output"],
                error=t["error"],
            )
            for t in executed_tools_record
        ]
        plan = ExecutionPlan(goal=user_prompt, steps=plan_steps, is_complete=is_success)

        return AgentRunResult(
            prompt=user_prompt,
            response_text=final_resp,
            success=is_success,
            decision=primary_decision,
            plan=plan,
            executed_tools=executed_tools_record,
            total_latency_ms=trace.total_duration_ms,
            slm_info=self.slm_provider.get_model_info(),
            validation_error=trace.error_message if not is_success else None,
            trace=trace,
        )
