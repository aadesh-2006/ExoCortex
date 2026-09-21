"""
Built-in foundation tools for ExoCortex.
"""

from __future__ import annotations

from typing import Any, Dict

from exocortex.hardware import detect_hardware
from exocortex.tools.base import (
    BaseTool,
    PermissionLevel,
    ToolParameter,
    ToolResult,
)


class EchoTool(BaseTool):
    """Simple echo tool to test parameter passing and tool invocation."""

    name = "echo"
    description = "Echoes back the provided input message. Used for verification."
    permission_level = PermissionLevel.SAFE
    parameters = [
        ToolParameter(
            name="message",
            type="string",
            description="The message string to echo back.",
            required=True,
        )
    ]

    def run(self, **kwargs: Any) -> ToolResult:
        msg = kwargs.get("message", "")
        return ToolResult(
            success=True,
            output=f"Echo: {msg}",
            data={"echoed_text": msg},
        )


class SystemInfoTool(BaseTool):
    """Tool to inspect Windows system hardware, CPU/NPU, and memory."""

    name = "system_info"
    description = "Inspects host system hardware, Snapdragon NPU capabilities, and memory."
    permission_level = PermissionLevel.SAFE
    parameters = [
        ToolParameter(
            name="detail_level",
            type="string",
            description="Detail level: 'basic' or 'full'.",
            required=False,
            default="basic",
        )
    ]

    def run(self, **kwargs: Any) -> ToolResult:
        hw = detect_hardware()
        data = hw.to_dict()

        summary_lines = [
            f"OS: {hw.os_name} {hw.os_release} ({hw.os_version})",
            f"Architecture: {hw.architecture}",
            f"Processor: {hw.processor_name}",
            f"Snapdragon NPU Detected: {'Yes' if hw.is_snapdragon_detected else 'No'}",
            f"Recommended Accelerator: {hw.recommended_execution_provider}",
            f"Memory: {hw.available_memory_gb:.1f} GB free / {hw.total_memory_gb:.1f} GB total",
        ]
        output = "\n".join(summary_lines)

        return ToolResult(
            success=True,
            output=output,
            data=data,
        )


class HealthCheckTool(BaseTool):
    """Tool to run ExoCortex diagnostic probes."""

    name = "health_check"
    description = "Runs comprehensive diagnostics on ExoCortex components and providers."
    permission_level = PermissionLevel.SAFE
    parameters = []

    def run(self, **kwargs: Any) -> ToolResult:
        from exocortex.health import run_health_check

        report = run_health_check()
        status_str = "HEALTHY" if report.status == "healthy" else "DEGRADED"
        summary = f"ExoCortex System Status: {status_str} (Version: {report.version})"

        return ToolResult(
            success=True,
            output=summary,
            data=report.to_dict(),
        )
