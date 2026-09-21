"""
Intent understanding and semantic classification for ExoCortex.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from exocortex.slm.base import Role, SLMMessage, SLMProvider


class IntentCategory(Enum):
    """Categorization of user intentions."""
    SYSTEM_DIAGNOSTICS = "system_diagnostics"
    HARDWARE_INSPECTION = "hardware_inspection"
    TOOL_EXECUTION = "tool_execution"
    GENERAL_REASONING = "general_reasoning"
    FILE_OPERATIONS = "file_operations"
    APP_AUTOMATION = "app_automation"
    UNKNOWN = "unknown"


@dataclass
class IntentAnalysis:
    """Result of intent understanding analysis."""
    raw_query: str
    primary_category: IntentCategory
    confidence: float
    extracted_entities: Dict[str, Any] = field(default_factory=dict)
    suggested_tools: List[str] = field(default_factory=list)
    is_multi_step: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_query": self.raw_query,
            "primary_category": self.primary_category.value,
            "confidence": round(self.confidence, 2),
            "extracted_entities": self.extracted_entities,
            "suggested_tools": self.suggested_tools,
            "is_multi_step": self.is_multi_step,
        }


class IntentEngine:
    """Analyzes natural language queries to determine intent and required actions."""

    def __init__(self, slm_provider: Optional[SLMProvider] = None) -> None:
        self.slm_provider = slm_provider

    def analyze(self, query: str) -> IntentAnalysis:
        """Classify user intent using SLM reasoning or deterministic parser."""
        q_lower = query.lower()

        # Fast heuristic classification
        if any(w in q_lower for w in ("health", "status", "alive", "checkup", "diagnostic")):
            return IntentAnalysis(
                raw_query=query,
                primary_category=IntentCategory.SYSTEM_DIAGNOSTICS,
                confidence=0.95,
                suggested_tools=["health_check"],
                is_multi_step=False,
            )

        if any(w in q_lower for w in ("hardware", "cpu", "npu", "snapdragon", "ram", "memory", "specs")):
            return IntentAnalysis(
                raw_query=query,
                primary_category=IntentCategory.HARDWARE_INSPECTION,
                confidence=0.92,
                suggested_tools=["system_info"],
                is_multi_step=False,
            )

        if "echo" in q_lower or "say" in q_lower:
            return IntentAnalysis(
                raw_query=query,
                primary_category=IntentCategory.TOOL_EXECUTION,
                confidence=0.90,
                suggested_tools=["echo"],
                is_multi_step=False,
            )

        if any(w in q_lower for w in ("open", "launch", "start", "run")):
            return IntentAnalysis(
                raw_query=query,
                primary_category=IntentCategory.APP_AUTOMATION,
                confidence=0.85,
                is_multi_step=True,
            )

        if any(w in q_lower for w in ("find", "search", "organize", "delete", "move", "copy")):
            return IntentAnalysis(
                raw_query=query,
                primary_category=IntentCategory.FILE_OPERATIONS,
                confidence=0.80,
                is_multi_step=True,
            )

        return IntentAnalysis(
            raw_query=query,
            primary_category=IntentCategory.GENERAL_REASONING,
            confidence=0.75,
            suggested_tools=[],
            is_multi_step=False,
        )
