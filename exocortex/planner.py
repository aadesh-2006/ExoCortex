"""
Agent planner and task decomposition for ExoCortex.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from exocortex.intent import IntentAnalysis, IntentCategory


class StepStatus(Enum):
    """Execution status of an individual plan step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PlanStep:
    """A single discrete step in an execution plan."""
    step_id: int
    description: str
    tool_name: Optional[str] = None
    arguments: Dict[str, Any] = field(default_factory=dict)
    status: StepStatus = StepStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "description": self.description,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
        }


@dataclass
class ExecutionPlan:
    """Multi-step execution plan for accomplishing a goal."""
    goal: str
    steps: List[PlanStep] = field(default_factory=list)
    is_complete: bool = False
    current_step_index: int = 0

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    def get_current_step(self) -> Optional[PlanStep]:
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None

    def advance(self) -> None:
        self.current_step_index += 1
        if self.current_step_index >= len(self.steps):
            self.is_complete = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "total_steps": self.total_steps,
            "current_step_index": self.current_step_index,
            "is_complete": self.is_complete,
            "steps": [s.to_dict() for s in self.steps],
        }


class AgentPlanner:
    """Decomposes user queries and intents into actionable execution plans."""

    def create_plan(self, intent: IntentAnalysis) -> ExecutionPlan:
        """Generate a sequential execution plan based on analyzed intent."""
        steps: List[PlanStep] = []

        if intent.primary_category == IntentCategory.SYSTEM_DIAGNOSTICS:
            steps.append(
                PlanStep(
                    step_id=1,
                    description="Execute ExoCortex health check diagnostics.",
                    tool_name="health_check",
                    arguments={},
                )
            )
        elif intent.primary_category == IntentCategory.HARDWARE_INSPECTION:
            steps.append(
                PlanStep(
                    step_id=1,
                    description="Inspect host system hardware and Qualcomm Snapdragon NPU capabilities.",
                    tool_name="system_info",
                    arguments={"detail_level": "full"},
                )
            )
        elif intent.primary_category == IntentCategory.TOOL_EXECUTION:
            tool = intent.suggested_tools[0] if intent.suggested_tools else "echo"
            steps.append(
                PlanStep(
                    step_id=1,
                    description=f"Execute tool '{tool}'.",
                    tool_name=tool,
                    arguments=intent.extracted_entities or {"message": intent.raw_query},
                )
            )
        else:
            # General reasoning or direct SLM completion
            steps.append(
                PlanStep(
                    step_id=1,
                    description="Evaluate user request via Local SLM reasoning.",
                    tool_name=None,
                    arguments={},
                )
            )

        return ExecutionPlan(goal=intent.raw_query, steps=steps)
