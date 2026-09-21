"""
Base tool interfaces, parameter specifications, and security permission levels.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class PermissionLevel(Enum):
    """Permission level governing tool execution safety."""
    SAFE = "safe"                               # Read-only or harmless actions (no confirmation needed)
    CONFIRMATION_REQUIRED = "confirm_required"  # Modifying actions (files, settings, requires user approval)
    RESTRICTED = "restricted"                   # High-risk actions (system files, registry, destructive commands)


@dataclass
class ToolParameter:
    """Specification of a parameter for a tool."""
    name: str
    type: str
    description: str
    required: bool = True
    default: Optional[Any] = None


@dataclass
class ToolResult:
    """Standardized output structure for tool execution."""
    success: bool
    output: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "data": self.data,
            "error": self.error,
            "execution_time_ms": round(self.execution_time_ms, 2),
        }


class BaseTool(ABC):
    """Abstract base class for all ExoCortex tools."""

    name: str
    description: str
    permission_level: PermissionLevel = PermissionLevel.SAFE
    parameters: List[ToolParameter] = []

    @abstractmethod
    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool logic with validated arguments."""
        pass

    def get_schema(self) -> Dict[str, Any]:
        """Generate JSON Schema describing this tool for SLM function-calling."""
        properties: Dict[str, Any] = {}
        required: List[str] = []

        for p in self.parameters:
            properties[p.name] = {
                "type": p.type,
                "description": p.description,
            }
            if p.default is not None:
                properties[p.name]["default"] = p.default
            if p.required:
                required.append(p.name)

        return {
            "name": self.name,
            "description": self.description,
            "permission_level": self.permission_level.value,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def validate_arguments(self, arguments: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate passed arguments against the tool's parameter specification."""
        for p in self.parameters:
            if p.required and p.name not in arguments:
                return False, f"Missing required parameter: '{p.name}'"
        return True, None
