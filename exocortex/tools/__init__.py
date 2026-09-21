"""
Tool execution framework and permission registry for ExoCortex.
"""

from exocortex.tools.base import (
    BaseTool,
    PermissionLevel,
    ToolParameter,
    ToolResult,
)
from exocortex.tools.registry import ToolRegistry, get_default_tool_registry
from exocortex.tools.builtin import (
    EchoTool,
    HealthCheckTool,
    SystemInfoTool,
)

__all__ = [
    "BaseTool",
    "PermissionLevel",
    "ToolParameter",
    "ToolResult",
    "ToolRegistry",
    "get_default_tool_registry",
    "EchoTool",
    "HealthCheckTool",
    "SystemInfoTool",
]
