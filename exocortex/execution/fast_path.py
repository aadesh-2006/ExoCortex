"""
Deterministic Fast-Path Router for Canonical Desktop Awareness Requests in ExoCortex.

Bypasses heavy SLM inference for unambiguous, zero-argument, read-only desktop requests,
executing strictly through ToolRegistry.execute() and preserving all permission boundaries.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, Optional, Set

from exocortex.execution.state import (
    ActionType,
    ExecutionStatus,
    ExecutionStep,
)
from exocortex.execution.trace import ExecutionTrace
from exocortex.schema import AgentDecision, StructuredStep
from exocortex.tools.base import PermissionLevel
from exocortex.tools.registry import ToolRegistry, get_default_tool_registry

logger = logging.getLogger("exocortex.execution.fast_path")

# Unambiguous canonical zero-argument queries mapped to exact registered tool names.
# Only zero-argument, read-only tools with PermissionLevel.SAFE are eligible.
CANONICAL_FAST_PATH_MAP: Dict[str, str] = {
    # list_open_windows (existing)
    "list open windows": "list_open_windows",
    "list windows": "list_open_windows",
    "what application windows are open": "list_open_windows",
    "what windows are open": "list_open_windows",
    "show open windows": "list_open_windows",
    "show windows": "list_open_windows",
    "get open windows": "list_open_windows",
    "get windows": "list_open_windows",
    # list_open_windows (Task 8.8.1 expanded)
    "show me open windows": "list_open_windows",
    "what windows are currently open": "list_open_windows",
    "what desktop windows are open": "list_open_windows",
    "list desktop windows": "list_open_windows",
    "display open windows": "list_open_windows",
    "get open window list": "list_open_windows",

    # list_processes (existing)
    "list processes": "list_processes",
    "list running processes": "list_processes",
    "show processes": "list_processes",
    "show running processes": "list_processes",
    "get processes": "list_processes",
    "get running processes": "list_processes",
    "what processes are running": "list_processes",
    # list_processes (Task 8.8.1 expanded)
    "display running processes": "list_processes",
    "show process list": "list_processes",
    "show me running processes": "list_processes",
    "what processes are currently running": "list_processes",
    "list active processes": "list_processes",
    "get process list": "list_processes",

    # system_info (Task 8.8.1)
    "system info": "system_info",
    "system information": "system_info",
    "show system info": "system_info",
    "show system information": "system_info",
    "get system info": "system_info",
    "get system information": "system_info",
    "display system info": "system_info",
    "display system information": "system_info",

    # health_check (Task 8.8.1)
    "health check": "health_check",
    "check system health": "health_check",
    "run health check": "health_check",
    "system health check": "health_check",
    "check exocortex health": "health_check",
    "exocortex health check": "health_check",
}

# Explicit set of tools permitted for fast-path execution.
FAST_PATH_ELIGIBLE_TOOLS: Set[str] = {
    "list_open_windows",
    "list_processes",
    "system_info",
    "health_check",
}

# Explicit intent labels for trace decisions produced by canonical fast paths.
FAST_PATH_INTENT_MAP: Dict[str, str] = {
    "list_open_windows": "window_enumeration",
    "list_processes": "process_inspection",
    "system_info": "system_information",
    "health_check": "system_health_check",
}


def normalize_fast_path_query(query: str) -> str:
    """
    Normalize query for exact canonical matching.
    
    Performs case folding, removes leading/trailing harmless punctuation and quotes,
    and collapses internal whitespace. Never performs fuzzy or substring matching.
    """
    if not query:
        return ""
    q = query.strip().lower()
    # Strip leading/trailing punctuation and quotation marks commonly found in queries
    q = re.sub(r"^[\s\"'`]+|[\s\"'`?!.,;:]+$", "", q)
    # Collapse multiple whitespace characters into a single space
    q = re.sub(r"\s+", " ", q).strip()
    return q


class FastPathRouter:
    """
    Fast-path dispatcher for canonical zero-argument desktop awareness requests.
    
    Evaluates incoming user prompts against an exact normalized dictionary of canonical
    phrases. When a match is found, executes the SAFE tool directly via ToolRegistry,
    returning a structured ExecutionTrace with 0 LLM calls.
    
    If the request is ambiguous, parameterized, or not in the canonical map,
    returns None to allow transparent fall-through to the general SLM cognitive loop.
    """

    def __init__(
        self,
        tool_registry: Optional[ToolRegistry] = None,
        enabled: bool = True,
    ) -> None:
        self.tool_registry = tool_registry or get_default_tool_registry()
        self.enabled = enabled

    def match(self, user_query: str) -> Optional[str]:
        """
        Check if the query exactly matches a canonical zero-argument request.
        
        Returns:
            Tool name (str) if an unambiguous match is found, otherwise None.
        """
        if not self.enabled:
            return None

        normalized = normalize_fast_path_query(user_query)
        tool_name = CANONICAL_FAST_PATH_MAP.get(normalized)
        if not tool_name:
            return None

        # Verify eligibility
        if tool_name not in FAST_PATH_ELIGIBLE_TOOLS:
            logger.warning("Matched tool '%s' is not in fast-path eligible tools.", tool_name)
            return None

        return tool_name

    def try_execute(self, user_query: str) -> Optional[ExecutionTrace]:
        """
        Attempt fast-path execution for the given user query.
        
        Returns:
            ExecutionTrace if matched and executed, or None if the request must fall through to the SLM.
        """
        tool_name = self.match(user_query)
        if not tool_name:
            return None

        # Verify tool registration and permission
        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            logger.warning("Fast-path tool '%s' is not registered. Falling through to SLM.", tool_name)
            return None

        if tool.permission_level != PermissionLevel.SAFE:
            logger.warning(
                "Fast-path tool '%s' permission level is %s (must be SAFE). Falling through.",
                tool_name,
                tool.permission_level,
            )
            return None

        logger.info("Executing fast-path route for canonical query: '%s' -> %s()", user_query, tool_name)
        start_time = time.perf_counter()

        # Execute safely strictly via ToolRegistry
        tool_res = self.tool_registry.execute(tool_name, {})
        duration_ms = (time.perf_counter() - start_time) * 1000

        # Construct intent and structured decision
        intent = FAST_PATH_INTENT_MAP.get(tool_name, tool_name)
        has_valid_output = bool(tool_res.output and tool_res.output.strip())
        is_success = bool(tool_res.success and has_valid_output)

        if is_success:
            status = ExecutionStatus.COMPLETED
            error_msg = None
            final_response = tool_res.output
            obs_text = f"[{tool_name}]: {tool_res.output}"
        else:
            status = ExecutionStatus.ERROR
            if not tool_res.success:
                error_msg = tool_res.error or f"Tool '{tool_name}' execution failed."
            else:
                error_msg = f"Tool '{tool_name}' returned empty output."
            final_response = None
            obs_text = f"[{tool_name} Error]: {error_msg}"

        decision = AgentDecision(
            thought_summary=f"Canonical zero-argument fast path executed for '{tool_name}'.",
            intent=intent,
            steps=[StructuredStep(tool=tool_name, arguments={})],
            requires_confirmation=False,
            direct_response=final_response,
        )

        step = ExecutionStep(
            step_number=1,
            timestamp=time.time(),
            decision=decision,
            action_type=ActionType.TOOL_CALL,
            tool_name=tool_name,
            arguments={},
            permission_level=tool.permission_level,
            tool_result=tool_res,
            observation_text=obs_text,
            duration_ms=duration_ms,
            status=status,
            error_message=error_msg,
        )

        trace = ExecutionTrace(
            user_goal=user_query,
            status=status,
            start_time=start_time,
            slm_info={
                "provider": "FastPathRouter",
                "model_name": "deterministic_fast_path",
                "is_local": True,
                "llm_calls": 0,
            },
        )
        trace.steps.append(step)
        trace.finalize(
            status=status,
            final_response=final_response,
            error_message=error_msg,
        )

        return trace
