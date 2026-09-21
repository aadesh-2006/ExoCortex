"""
Safe iterative agent execution engine for ExoCortex.

Orchestrates the multi-step cognitive loop:
User Goal -> SLM Decision -> Validation -> Permission Check -> Tool Execution -> Observation Feedback -> Next Decision / Final Response.

Security Guarantees:
- Every tool execution strictly routes through ToolRegistry.execute().
- Zero eval(), exec(), subprocess, or dynamic imports.
- Hard limits on max steps (MAX_STEPS = 8) and loop detection (MAX_IDENTICAL_ACTIONS = 2).
- Tool results are strictly bounded to 2,000 characters before being fed back to the SLM.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from exocortex.execution.state import (
    ActionType,
    ExecutionState,
    ExecutionStatus,
    ExecutionStep,
)
from exocortex.execution.trace import ExecutionTrace
from exocortex.schema import AgentDecision, StructuredStep
from exocortex.slm.base import SLMMessage, SLMProvider, SLMResponse
from exocortex.slm.parser import parse_and_validate_decision
from exocortex.slm.prompts import build_system_prompt, build_user_prompt
from exocortex.tools.base import PermissionLevel, ToolResult
from exocortex.tools.registry import ToolRegistry, get_default_tool_registry

logger = logging.getLogger("exocortex.execution.executor")

DEFAULT_MAX_STEPS = 8
DEFAULT_MAX_IDENTICAL_ACTIONS = 2
DEFAULT_MAX_RESULT_CHARS = 2000


class AgentExecutor:
    """
    Safe multi-step cognitive loop executor for ExoCortex.
    """

    def __init__(
        self,
        slm_provider: SLMProvider,
        tool_registry: Optional[ToolRegistry] = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        max_identical_actions: int = DEFAULT_MAX_IDENTICAL_ACTIONS,
        max_result_chars: int = DEFAULT_MAX_RESULT_CHARS,
    ) -> None:
        self.slm_provider = slm_provider
        self.tool_registry = tool_registry or get_default_tool_registry()
        self.max_steps = max_steps
        self.max_identical_actions = max_identical_actions
        self.max_result_chars = max_result_chars

    def execute(self, user_goal: str) -> ExecutionTrace:
        """
        Run the safe iterative execution loop for a user request.
        """
        start_time = time.perf_counter()
        logger.info("Starting AgentExecutor loop for goal: '%s'", user_goal)

        state = ExecutionState(
            user_goal=user_goal,
            max_steps=self.max_steps,
            max_identical_actions=self.max_identical_actions,
            max_result_chars=self.max_result_chars,
        )

        trace = ExecutionTrace(
            user_goal=user_goal,
            status=ExecutionStatus.RUNNING,
            start_time=start_time,
            slm_info=self.slm_provider.get_model_info(),
        )

        state.status = ExecutionStatus.RUNNING

        while not state.is_finished():
            # 1. Hard step limit enforcement
            if state.current_step >= state.max_steps:
                logger.warning("Max steps limit (%d) exceeded.", state.max_steps)
                state.status = ExecutionStatus.MAX_STEPS_EXCEEDED
                state.error_message = (
                    f"Execution stopped: Maximum step limit ({state.max_steps}) reached."
                )
                break

            # 2. Build tool-aware system prompt and contextual user prompt
            available_tools = self.tool_registry.list_tools()
            system_prompt = build_system_prompt(available_tools)
            history_summary = state.get_history_summary()
            formatted_user_prompt = build_user_prompt(
                user_query=user_goal,
                execution_history=history_summary if history_summary else None,
            )

            # 3. Query SLM for structured decision
            step_start = time.perf_counter()
            try:
                slm_res: SLMResponse = self.slm_provider.generate(
                    prompt=formatted_user_prompt,
                    system_prompt=system_prompt,
                    temperature=0.1,
                )
            except Exception as slm_ex:
                logger.error("SLM generation encountered error: %s", slm_ex)
                step_duration = (time.perf_counter() - step_start) * 1000
                state.current_step += 1
                state.status = ExecutionStatus.ERROR
                state.error_message = f"SLM Generation Error: {slm_ex}"
                err_step = ExecutionStep(
                    step_number=state.current_step,
                    timestamp=time.time(),
                    decision=None,
                    action_type=ActionType.ERROR,
                    duration_ms=step_duration,
                    status=ExecutionStatus.ERROR,
                    error_message=state.error_message,
                )
                state.steps.append(err_step)
                trace.steps.append(err_step)
                break

            # 4. Parse and strictly validate decision against schema & registry
            decision, parse_err = parse_and_validate_decision(
                raw_text=slm_res.content,
                tool_registry=self.tool_registry,
            )

            # 5. Retry once if output was malformed
            if decision is None:
                logger.warning("SLM output validation failed: %s. Retrying...", parse_err)
                repair_prompt = (
                    f"Your previous response failed validation: {parse_err}\n"
                    "Output ONLY a single valid JSON object matching the required schema: "
                    '{"thought_summary": "...", "intent": "...", "steps": [], "requires_confirmation": false, "direct_response": "..."}'
                )
                try:
                    slm_res = self.slm_provider.generate(
                        prompt=repair_prompt,
                        system_prompt=system_prompt,
                        temperature=0.1,
                    )
                    decision, parse_err = parse_and_validate_decision(
                        raw_text=slm_res.content,
                        tool_registry=self.tool_registry,
                    )
                except Exception as retry_ex:
                    logger.warning("SLM retry generation failed: %s", retry_ex)
                    parse_err = str(retry_ex)

            # If still failed, record error and stop safely
            if decision is None:
                step_duration = (time.perf_counter() - step_start) * 1000
                state.current_step += 1
                state.status = ExecutionStatus.ERROR
                state.error_message = f"SLM Reasoning Error: {parse_err}"
                err_step = ExecutionStep(
                    step_number=state.current_step,
                    timestamp=time.time(),
                    decision=None,
                    action_type=ActionType.ERROR,
                    duration_ms=step_duration,
                    status=ExecutionStatus.ERROR,
                    error_message=state.error_message,
                )
                state.steps.append(err_step)
                trace.steps.append(err_step)
                break

            # 6. Check if SLM decided on final response (no further tool steps)
            if not decision.has_steps:
                step_duration = (time.perf_counter() - step_start) * 1000
                state.current_step += 1
                final_text = decision.direct_response or decision.thought_summary or "Task completed."
                state.status = ExecutionStatus.COMPLETED
                state.final_response = final_text

                final_step = ExecutionStep(
                    step_number=state.current_step,
                    timestamp=time.time(),
                    decision=decision,
                    action_type=ActionType.FINAL_RESPONSE,
                    observation_text=final_text,
                    duration_ms=step_duration,
                    status=ExecutionStatus.COMPLETED,
                )
                state.steps.append(final_step)
                trace.steps.append(final_step)
                logger.info("Agent completed goal with final response: '%s'", final_text)
                break

            # 7. Execute planned tool steps
            for step_item in decision.steps:
                if state.current_step >= state.max_steps:
                    state.status = ExecutionStatus.MAX_STEPS_EXCEEDED
                    state.error_message = (
                        f"Execution stopped: Maximum step limit ({state.max_steps}) reached."
                    )
                    break

                state.current_step += 1
                action_start = time.perf_counter()

                # A. Tool existence check
                tool = self.tool_registry.get_tool(step_item.tool)
                if not tool:
                    err_msg = f"Unknown tool '{step_item.tool}' not found in registry."
                    action_duration = (time.perf_counter() - action_start) * 1000
                    exec_step = ExecutionStep(
                        step_number=state.current_step,
                        timestamp=time.time(),
                        decision=decision,
                        action_type=ActionType.TOOL_CALL,
                        tool_name=step_item.tool,
                        arguments=step_item.arguments,
                        observation_text=f"[{step_item.tool} Error]: {err_msg}",
                        duration_ms=action_duration,
                        status=ExecutionStatus.ERROR,
                        error_message=err_msg,
                    )
                    state.steps.append(exec_step)
                    trace.steps.append(exec_step)
                    continue

                # B. Loop detection check
                is_loop, count = state.loop_tracker.record_action(
                    tool_name=step_item.tool,
                    arguments=step_item.arguments,
                )
                if is_loop:
                    err_msg = (
                        f"Loop detected: Action '{step_item.tool}' with arguments "
                        f"{step_item.arguments} was executed {count} times "
                        f"(max allowed: {state.max_identical_actions})."
                    )
                    logger.warning(err_msg)
                    state.status = ExecutionStatus.LOOP_DETECTED
                    state.error_message = err_msg
                    action_duration = (time.perf_counter() - action_start) * 1000
                    exec_step = ExecutionStep(
                        step_number=state.current_step,
                        timestamp=time.time(),
                        decision=decision,
                        action_type=ActionType.TOOL_CALL,
                        tool_name=step_item.tool,
                        arguments=step_item.arguments,
                        permission_level=tool.permission_level,
                        observation_text=f"[{step_item.tool} Error]: {err_msg}",
                        duration_ms=action_duration,
                        status=ExecutionStatus.LOOP_DETECTED,
                        error_message=err_msg,
                    )
                    state.steps.append(exec_step)
                    trace.steps.append(exec_step)
                    break

                # C. Permission & Confirmation gating
                if tool.permission_level in (
                    PermissionLevel.CONFIRMATION_REQUIRED,
                    PermissionLevel.RESTRICTED,
                ):
                    if not self.tool_registry.confirmation_callback:
                        err_msg = (
                            f"Action '{step_item.tool}' requires explicit user confirmation, "
                            "but no confirmation callback is configured."
                        )
                        logger.warning(err_msg)
                        state.status = ExecutionStatus.CONFIRMATION_REQUIRED
                        state.confirmation_tool = step_item.tool
                        state.confirmation_arguments = step_item.arguments
                        state.error_message = err_msg
                        action_duration = (time.perf_counter() - action_start) * 1000
                        exec_step = ExecutionStep(
                            step_number=state.current_step,
                            timestamp=time.time(),
                            decision=decision,
                            action_type=ActionType.TOOL_CALL,
                            tool_name=step_item.tool,
                            arguments=step_item.arguments,
                            permission_level=tool.permission_level,
                            duration_ms=action_duration,
                            status=ExecutionStatus.CONFIRMATION_REQUIRED,
                            error_message=err_msg,
                        )
                        state.steps.append(exec_step)
                        trace.steps.append(exec_step)
                        break

                # D. Safe execution strictly via ToolRegistry.execute
                tool_res = self.tool_registry.execute(step_item.tool, step_item.arguments)
                action_duration = (time.perf_counter() - action_start) * 1000

                # E. Bounding tool output size before feeding back to SLM
                raw_output = tool_res.output if tool_res.success else (tool_res.error or "Tool failed.")
                if len(raw_output) > state.max_result_chars:
                    bounded_output = (
                        raw_output[: state.max_result_chars]
                        + f"\n... [Output bounded to {state.max_result_chars} chars]"
                    )
                else:
                    bounded_output = raw_output

                observation = (
                    f"[{step_item.tool}]: {bounded_output}"
                    if tool_res.success
                    else f"[{step_item.tool} Error]: {bounded_output}"
                )

                exec_step = ExecutionStep(
                    step_number=state.current_step,
                    timestamp=time.time(),
                    decision=decision,
                    action_type=ActionType.TOOL_CALL,
                    tool_name=step_item.tool,
                    arguments=step_item.arguments,
                    permission_level=tool.permission_level,
                    tool_result=tool_res,
                    observation_text=observation,
                    duration_ms=action_duration,
                    status=ExecutionStatus.RUNNING if tool_res.success else ExecutionStatus.ERROR,
                    error_message=tool_res.error,
                )
                state.steps.append(exec_step)
                trace.steps.append(exec_step)

            # If stopped by confirmation or loop detection, break outer loop
            if state.is_finished():
                break

        # 8. Synthesize final response if completed via multiple tool steps without explicit direct_response
        if state.status == ExecutionStatus.COMPLETED and not state.final_response:
            obs_lines = [s.observation_text for s in state.steps if s.observation_text]
            state.final_response = "\n\n".join(obs_lines)

        # 9. Finalize execution trace
        trace.finalize(
            status=state.status,
            final_response=state.final_response,
            error_message=state.error_message,
        )

        return trace
