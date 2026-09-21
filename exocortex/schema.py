"""
Strict structured response schema and validation models for ExoCortex.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class StructuredStep(BaseModel):
    """A discrete tool invocation step decided by the SLM."""
    tool: str = Field(description="Name of the registered tool to execute")
    arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary of arguments matching tool schema"
    )

    @field_validator("tool")
    @classmethod
    def validate_tool_name(cls, v: str) -> str:
        v_clean = v.strip()
        if not v_clean:
            raise ValueError("Tool name cannot be empty")
        return v_clean


class AgentDecision(BaseModel):
    """
    Strict structured output produced by the SLM reasoning brain.
    
    Contains a concise rationale (no hidden chain-of-thought), classified intent,
    sequence of tool steps, confirmation flag, and optional direct response.
    """
    thought_summary: str = Field(
        default="Action planned by agent.",
        description="Concise 1-2 sentence action rationale (no raw chain-of-thought)"
    )
    intent: str = Field(
        default="general_action",
        description="High-level classified intent (e.g. system_health, hardware_inspection, general_inquiry)"
    )
    steps: List[StructuredStep] = Field(
        default_factory=list,
        description="List of tool execution steps to perform"
    )
    requires_confirmation: bool = Field(
        default=False,
        description="True if any planned action modifies system state or requires user consent"
    )
    direct_response: Optional[str] = Field(
        default=None,
        description="Direct conversational answer when no tool execution is needed"
    )

    @property
    def has_steps(self) -> bool:
        return len(self.steps) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thought_summary": self.thought_summary,
            "intent": self.intent,
            "steps": [s.model_dump() for s in self.steps],
            "requires_confirmation": self.requires_confirmation,
            "direct_response": self.direct_response,
        }
