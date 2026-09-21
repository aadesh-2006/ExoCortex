"""
Central tool registry and execution dispatcher for ExoCortex.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from exocortex.tools.base import (
    BaseTool,
    PermissionLevel,
    ToolResult,
)


class ToolRegistry:
    """Registry managing available tools, schemas, and safe dispatching."""

    def __init__(
        self,
        confirmation_callback: Optional[Callable[[str, Dict[str, Any]], bool]] = None,
    ) -> None:
        self._tools: Dict[str, BaseTool] = {}
        self.confirmation_callback = confirmation_callback

    def register(self, tool: BaseTool) -> None:
        """Register a new tool instance."""
        if tool.name in self._tools:
            raise ValueError(f"Tool with name '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Retrieve a registered tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[BaseTool]:
        """List all registered tools."""
        return list(self._tools.values())

    def get_schemas(self) -> List[Dict[str, Any]]:
        """Return the function calling schemas for all registered tools."""
        return [tool.get_schema() for tool in self._tools.values()]

    def execute(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        """Execute a tool with argument validation and permission gating."""
        tool = self.get_tool(name)
        if not tool:
            return ToolResult(
                success=False,
                output="",
                error=f"Tool '{name}' not found in registry.",
            )

        # Validate arguments
        valid, err = tool.validate_arguments(arguments)
        if not valid:
            return ToolResult(
                success=False,
                output="",
                error=f"Argument validation error for tool '{name}': {err}",
            )

        # Permission / Confirmation check
        if tool.permission_level in (PermissionLevel.CONFIRMATION_REQUIRED, PermissionLevel.RESTRICTED):
            if self.confirmation_callback:
                allowed = self.confirmation_callback(name, arguments)
                if not allowed:
                    return ToolResult(
                        success=False,
                        output="Operation cancelled by user.",
                        error="Permission denied by user.",
                    )
            else:
                # In non-interactive mode without callback, reject sensitive action by default
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Tool '{name}' requires confirmation, but no confirmation callback was configured.",
                )

        start = time.perf_counter()
        try:
            result = tool.run(**arguments)
            result.execution_time_ms = (time.perf_counter() - start) * 1000
            return result
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return ToolResult(
                success=False,
                output="",
                error=f"Exception during tool '{name}' execution: {str(e)}",
                execution_time_ms=elapsed,
            )


_default_registry: Optional[ToolRegistry] = None


def get_default_tool_registry() -> ToolRegistry:
    """Get the singleton default tool registry pre-loaded with built-in tools."""
    global _default_registry
    if _default_registry is None:
        from exocortex.tools.builtin import EchoTool, HealthCheckTool, SystemInfoTool

        _default_registry = ToolRegistry()
        _default_registry.register(EchoTool())
        _default_registry.register(SystemInfoTool())
        _default_registry.register(HealthCheckTool())
    return _default_registry
