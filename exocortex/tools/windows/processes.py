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
        ToolParameter(
            name="all_processes",
            type="boolean",
            description="If True, return all running processes without truncation (default: False).",
            required=False,
            default=False,
        ),
    ]

    def run(self, **kwargs: Any) -> ToolResult:
        all_processes = bool(kwargs.get("all_processes", False))
        limit_val = kwargs.get("limit", 20)
        sort_by = str(kwargs.get("sort_by", "memory")).lower()

        total_pids_scanned = 0
        access_denied_count = 0
        vanished_count = 0
        process_list: List[Dict[str, Any]] = []

        _AD_SENTINEL = object()
        try:
            for p in psutil.process_iter(["pid", "name", "memory_info"], ad_value=_AD_SENTINEL):
                total_pids_scanned += 1
                try:
                    info = p.info
                    if not isinstance(info, dict):
                        continue

                    raw_name = info.get("name")
                    raw_mem = info.get("memory_info")
                    pid = info.get("pid")
                    if pid is None:
                        pid = getattr(p, "pid", None)

                    if raw_name is _AD_SENTINEL:
                        access_denied_count += 1
                        continue

                    name = raw_name or "Unknown"

                    if raw_mem is _AD_SENTINEL:
                        mem_mb = 0.0
                    elif raw_mem:
                        mem_mb = round(raw_mem.rss / (1024 * 1024), 1)
                    else:
                        mem_mb = 0.0

                    process_list.append({
                        "pid": pid,
                        "name": name,
                        "memory_mb": mem_mb,
                    })
                except psutil.AccessDenied:
                    access_denied_count += 1
                except (psutil.NoSuchProcess, psutil.ZombieProcess):
                    vanished_count += 1
                except OSError:
                    continue

            # Sort
            if sort_by == "name":
                process_list.sort(key=lambda x: x["name"].lower())
            else:
                process_list.sort(key=lambda x: x["memory_mb"], reverse=True)

            if all_processes:
                top_processes = process_list
                table_title = f"All {len(top_processes)} Running Processes (Sorted by {sort_by}):"
            else:
                try:
                    limit = min(max(int(limit_val), 1), 50)
                except (ValueError, TypeError):
                    limit = 20
                top_processes = process_list[:limit]
                table_title = f"Top {len(top_processes)} Running Processes (Sorted by {sort_by}):"

            # Format summary table
            lines = [table_title]
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
                    "total_pids_scanned": total_pids_scanned,
                    "access_denied_count": access_denied_count,
                    "vanished_count": vanished_count,
                    "scan_complete": bool(access_denied_count == 0),
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
