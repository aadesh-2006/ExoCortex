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
from exocortex.planner import ExecutionPlan, PlanStep, StepStatus
from exocortex.schema import AgentDecision, StructuredStep
from exocortex.slm.base import Role, SLMMessage, SLMProvider, SLMResponse
from exocortex.slm.factory import get_slm_provider
from exocortex.slm.parser import parse_and_validate_decision
from exocortex.slm.prompts import build_system_prompt, build_user_prompt
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
        }


class ExoCortexAgent:
    """
    Local-first autonomous personal computer agent for Windows.
    
    Driven by local SLM structured tool-calling reasoning.
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
        self.history: List[SLMMessage] = []

    def run(self, user_prompt: str, max_retries: int = 1) -> AgentRunResult:
        """
        Execute the autonomous cognitive loop for a user prompt.
        """
        start_time = time.perf_counter()
        logger.info("Executing ExoCortex reasoning loop for: '%s'", user_prompt)

        # 1. Dynamically build tool-aware prompt
        available_tools = self.tool_registry.list_tools()
        system_prompt = build_system_prompt(available_tools)
        formatted_user_prompt = build_user_prompt(user_prompt)

        self.history.append(SLMMessage(role="user", content=user_prompt))

        # 2. Local SLM Inference
        slm_res: SLMResponse = self.slm_provider.generate(
            prompt=formatted_user_prompt,
            system_prompt=system_prompt,
            temperature=0.1,
        )

        # 3. Structured Decision Parsing & Validation
        decision, parse_err = parse_and_validate_decision(
            raw_text=slm_res.content,
            tool_registry=self.tool_registry,
        )

        # 4. Retry / Repair Strategy on parsing failure
        if decision is None and max_retries > 0:
            logger.warning("SLM output validation failed: %s. Retrying with explicit formatting prompt...", parse_err)
            repair_prompt = (
                f"Your previous response failed validation with error: {parse_err}\n"
                f"Previous output was:\n{slm_res.content}\n\n"
                "Please output ONLY valid JSON matching the specified schema."
            )
            slm_res = self.slm_provider.generate(
                prompt=repair_prompt,
                system_prompt=system_prompt,
                temperature=0.0,
            )
            decision, parse_err = parse_and_validate_decision(
                raw_text=slm_res.content,
                tool_registry=self.tool_registry,
            )

        # If still failed, return structured failure
        if decision is None:
            total_time = (time.perf_counter() - start_time) * 1000
            error_msg = f"SLM Reasoning Error: {parse_err}"
            return AgentRunResult(
                prompt=user_prompt,
                response_text=error_msg,
                success=False,
                decision=None,
                plan=None,
                total_latency_ms=total_time,
                slm_info=self.slm_provider.get_model_info(),
                validation_error=parse_err,
            )

        # 5. Build Execution Plan from Decision Steps
        plan_steps: List[PlanStep] = []
        for idx, s in enumerate(decision.steps, start=1):
            plan_steps.append(
                PlanStep(
                    step_id=idx,
                    description=f"Execute tool '{s.tool}'",
                    tool_name=s.tool,
                    arguments=s.arguments,
                )
            )
        plan = ExecutionPlan(goal=user_prompt, steps=plan_steps)

        # 6. Execute Steps
        executed_tools_record: List[Dict[str, Any]] = []
        observations: List[str] = []

        if not plan_steps:
            # Direct response without tool execution
            final_response = decision.direct_response or decision.thought_summary
            plan.is_complete = True
        else:
            while not plan.is_complete:
                step = plan.get_current_step()
                if not step:
                    break

                step.status = StepStatus.IN_PROGRESS
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
                    observations.append(f"[{step.tool_name}]: {tool_res.output}")
                else:
                    step.status = StepStatus.FAILED
                    step.error = tool_res.error
                    observations.append(f"[{step.tool_name} Error]: {tool_res.error}")

                plan.advance()

            # Final synthesized response
            summary_part = f"Rationale: {decision.thought_summary}\n\n" if decision.thought_summary else ""
            final_response = summary_part + "\n\n".join(observations)

        self.history.append(SLMMessage(role="assistant", content=final_response))
        total_time = (time.perf_counter() - start_time) * 1000

        all_steps_ok = all(s.status == StepStatus.COMPLETED for s in plan.steps) if plan.steps else True

        return AgentRunResult(
            prompt=user_prompt,
            response_text=final_response,
            success=all_steps_ok,
            decision=decision,
            plan=plan,
            executed_tools=executed_tools_record,
            total_latency_ms=total_time,
            slm_info=self.slm_provider.get_model_info(),
        )
