"""
Structured execution trace recorder for ExoCortex.

Provides an auditable, type-safe record of the agent's multi-step decision cycles.
Never logs credentials, tokens, or sensitive environment variables.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from exocortex.execution.state import ExecutionStatus, ExecutionStep


@dataclass
class ExecutionTrace:
    """Complete auditable trace of an agent execution cycle."""
    user_goal: str
    status: ExecutionStatus
    start_time: float
    end_time: float = 0.0
    total_duration_ms: float = 0.0
    total_steps: int = 0
    steps: List[ExecutionStep] = field(default_factory=list)
    final_response: Optional[str] = None
    error_message: Optional[str] = None
    slm_info: Dict[str, Any] = field(default_factory=dict)

    def finalize(
        self,
        status: ExecutionStatus,
        final_response: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Mark trace as finalized and compute duration."""
        self.end_time = time.perf_counter()
        self.total_duration_ms = (self.end_time - self.start_time) * 1000
        self.status = status
        self.final_response = final_response
        self.error_message = error_message
        self.total_steps = len(self.steps)

    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to a sanitized, serializable dictionary."""
        return {
            "user_goal": self.user_goal,
            "status": self.status.value,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "total_steps": self.total_steps,
            "final_response": self.final_response,
            "error_message": self.error_message,
            "slm_info": self.slm_info,
            "steps": [s.to_dict() for s in self.steps],
        }

    def get_summary(self) -> str:
        """Format human-readable execution summary."""
        lines = [
            f"Execution Status: {self.status.value.upper()}",
            f"Total Steps:      {self.total_steps}",
            f"Total Duration:   {self.total_duration_ms:.1f} ms",
        ]
        if self.steps:
            lines.append("\nStep-by-step breakdown:")
            for s in self.steps:
                status_symbol = "[OK]" if (s.tool_result and s.tool_result.success) else (
                    "[DONE]" if s.action_type.value == "final_response" else "[FAIL]"
                )
                if s.tool_name:
                    lines.append(f"  {status_symbol} Step {s.step_number}: {s.tool_name} ({s.duration_ms:.1f} ms)")
                else:
                    lines.append(f"  {status_symbol} Step {s.step_number}: {s.action_type.value}")

        if self.final_response:
            lines.append(f"\nFinal Response:\n{self.final_response}")
        elif self.error_message:
            lines.append(f"\nError:\n{self.error_message}")

        return "\n".join(lines)
