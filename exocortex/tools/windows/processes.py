"""
Safe read-only Windows process inspection tool for ExoCortex.

Returns structured telemetry on running desktop processes.
Process termination or manipulation is strictly prohibited in this tool.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import psutil

from exocortex.tools.base import (
    BaseTool,
    PermissionLevel,
    ToolParameter,
    ToolResult,
)

logger = logging.getLogger("exocortex.tools.windows.processes")


class ListProcessesTool(BaseTool):
    """Tool to safely inspect running system processes."""

    name = "list_processes"
    description = (
        "Inspects and lists currently running processes on Windows with PID, name, and memory usage. "
        "Read-only inspection only."
    )
    permission_level = PermissionLevel.SAFE
    parameters = [
        ToolParameter(
            name="limit",
            type="integer",
            description="Maximum number of top processes to return (default: 20, max: 50).",
            required=False,
            default=20,
        ),
        ToolParameter(
            name="sort_by",
            type="string",
            description="Metric to sort by: 'memory' or 'name' (default: 'memory').",
            required=False,
            default="memory",
        ),
    ]

    def run(self, **kwargs: Any) -> ToolResult:
        limit = min(max(int(kwargs.get("limit", 20)), 1), 50)
        sort_by = str(kwargs.get("sort_by", "memory")).lower()

        process_list: List[Dict[str, Any]] = []

        try:
            for p in psutil.process_iter(["pid", "name", "memory_info"]):
                try:
                    info = p.info
                    pid = info.get("pid")
                    name = info.get("name") or "Unknown"
                    mem_info = info.get("memory_info")
                    mem_mb = round(mem_info.rss / (1024 * 1024), 1) if mem_info else 0.0

                    process_list.append({
                        "pid": pid,
                        "name": name,
                        "memory_mb": mem_mb,
                    })
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue

            # Sort
            if sort_by == "name":
                process_list.sort(key=lambda x: x["name"].lower())
            else:
                process_list.sort(key=lambda x: x["memory_mb"], reverse=True)

            top_processes = process_list[:limit]

            # Format summary table
            lines = [f"Top {len(top_processes)} Running Processes (Sorted by {sort_by}):"]
            lines.append(f"  {'PID':<8} {'Process Name':<35} {'Memory (MB)':>12}")
            lines.append("  " + "-" * 57)
            for proc in top_processes:
                lines.append(f"  {proc['pid']:<8} {proc['name']:<35} {proc['memory_mb']:>10.1f} MB")

            output_text = "\n".join(lines)
            return ToolResult(
                success=True,
                output=output_text,
                data={
                    "total_processes_scanned": len(process_list),
                    "returned_count": len(top_processes),
                    "sort_by": sort_by,
                    "processes": top_processes,
                },
            )

        except Exception as e:
            logger.error("Failed to list processes: %s", str(e))
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to inspect processes: {str(e)}",
            )
