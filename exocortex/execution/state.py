"""
Execution state models, status enums, and loop detection for ExoCortex.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from exocortex.schema import AgentDecision
from exocortex.tools.base import PermissionLevel, ToolResult


class ExecutionStatus(Enum):
    """Lifecycle status of an agent execution cycle."""
    INITIALIZED = "initialized"
    RUNNING = "running"
    COMPLETED = "completed"
    CONFIRMATION_REQUIRED = "confirmation_required"
    MAX_STEPS_EXCEEDED = "max_steps_exceeded"
    LOOP_DETECTED = "loop_detected"
    ERROR = "error"


class ActionType(Enum):
    """Type of action determined in a step."""
    TOOL_CALL = "tool_call"
    FINAL_RESPONSE = "final_response"
    ERROR = "error"


@dataclass
class ExecutionStep:
    """Detailed record of a single step within an execution cycle."""
    step_number: int
    timestamp: float
    decision: Optional[AgentDecision]
    action_type: ActionType
    tool_name: Optional[str] = None
    arguments: Dict[str, Any] = field(default_factory=dict)
    permission_level: PermissionLevel = PermissionLevel.SAFE
    tool_result: Optional[ToolResult] = None
    observation_text: str = ""
    duration_ms: float = 0.0
    status: ExecutionStatus = ExecutionStatus.RUNNING
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "timestamp": round(self.timestamp, 3),
            "action_type": self.action_type.value,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "permission_level": self.permission_level.value,
            "tool_result": self.tool_result.to_dict() if self.tool_result else None,
            "observation_text": self.observation_text,
            "duration_ms": round(self.duration_ms, 2),
            "status": self.status.value,
            "error_message": self.error_message,
        }


class ActionHistoryTracker:
    """
    Tracks action history and detects repeated identical actions.
    
    Prevents infinite cycles where the SLM invokes the identical tool + arguments repeatedly.
    """

    def __init__(self, max_identical_actions: int = 2) -> None:
        self.max_identical_actions = max_identical_actions
        self._action_counts: Dict[str, int] = {}
        self._action_sequence: List[str] = []

    @staticmethod
    def _action_key(tool_name: str, arguments: Dict[str, Any]) -> str:
        """Create a deterministic hashable key for a tool and its arguments."""
        try:
            sorted_args = json.dumps(arguments, sort_keys=True)
        except Exception:
            sorted_args = str(sorted(arguments.items()))
        return f"{tool_name}:{sorted_args}"

    def record_action(self, tool_name: str, arguments: Dict[str, Any]) -> tuple[bool, int]:
        """
        Record a tool action.
        
        Returns:
            (is_loop_detected, current_count)
        """
        key = self._action_key(tool_name, arguments)
        self._action_sequence.append(key)
        count = self._action_counts.get(key, 0) + 1
        self._action_counts[key] = count

        # If this identical action has occurred more than max_identical_actions times in sequence or total
        is_loop = count > self.max_identical_actions
        return is_loop, count

    def get_action_count(self, tool_name: str, arguments: Dict[str, Any]) -> int:
        key = self._action_key(tool_name, arguments)
        return self._action_counts.get(key, 0)

    def reset(self) -> None:
        self._action_counts.clear()
        self._action_sequence.clear()


@dataclass
class ExecutionState:
    """Runtime state of an agent execution cycle."""
    user_goal: str
    max_steps: int = 8
    max_identical_actions: int = 2
    max_result_chars: int = 2000
    current_step: int = 0
    status: ExecutionStatus = ExecutionStatus.INITIALIZED
    final_response: Optional[str] = None
    confirmation_tool: Optional[str] = None
    confirmation_arguments: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    steps: List[ExecutionStep] = field(default_factory=list)
    loop_tracker: ActionHistoryTracker = field(init=False)

    def __post_init__(self) -> None:
        self.loop_tracker = ActionHistoryTracker(
            max_identical_actions=self.max_identical_actions
        )

    def is_finished(self) -> bool:
        return self.status in (
            ExecutionStatus.COMPLETED,
            ExecutionStatus.CONFIRMATION_REQUIRED,
            ExecutionStatus.MAX_STEPS_EXCEEDED,
            ExecutionStatus.LOOP_DETECTED,
            ExecutionStatus.ERROR,
        )

    def get_history_summary(self) -> List[Dict[str, Any]]:
        """Return a structured list of previous steps and observation snippets for the SLM."""
        history = []
        for step in self.steps:
            if step.tool_name:
                history.append({
                    "step": step.step_number,
                    "tool": step.tool_name,
                    "arguments": step.arguments,
                    "success": step.tool_result.success if step.tool_result else False,
                    "observation": step.observation_text,
                })
        return history
