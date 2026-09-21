"""
Safe workspace-sandboxed filesystem tools for ExoCortex.

All operations are strictly confined to ~/.exocortex/workspace.
Path traversal (..) and out-of-boundary access are strictly blocked.
Zero destructive (delete/overwrite/format) operations are implemented.
"""

from __future__ import annotations

import datetime
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from exocortex.config import get_config
from exocortex.tools.base import (
    BaseTool,
    PermissionLevel,
    ToolParameter,
    ToolResult,
)

logger = logging.getLogger("exocortex.tools.windows.filesystem")

MAX_READ_BYTES = 1024 * 1024  # 1 MB maximum read size


def resolve_safe_workspace_path(path_str: str) -> Tuple[Optional[Path], Optional[str]]:
    """
    Safely resolve a target path and verify it is strictly within the workspace root.
    
    Returns:
        (resolved_path, error_message): error_message is None if path is valid and contained.
    """
    config = get_config()
    workspace_root = config.ensure_workspace_dir().resolve()

    if not path_str or not path_str.strip():
        return workspace_root, None

    clean_str = path_str.strip().strip("'\"")

    try:
        raw_path = Path(clean_str)
        if raw_path.is_absolute():
            resolved = raw_path.resolve()
        else:
            resolved = (workspace_root / raw_path).resolve()

        # Strict containment verification
        try:
            is_inside = resolved.is_relative_to(workspace_root)
        except AttributeError:
            # Python < 3.9 fallback
            try:
                resolved.relative_to(workspace_root)
                is_inside = True
            except ValueError:
                is_inside = False

        if not is_inside:
            return None, (
                f"Security violation: Path '{path_str}' resolves outside the allowed workspace "
                f"sandbox ({workspace_root}). Directory traversal is strictly blocked."
            )

        return resolved, None

    except Exception as e:
        return None, f"Invalid path syntax '{path_str}': {str(e)}"


class ListDirectoryTool(BaseTool):
    """Tool to safely list files and subdirectories within the ExoCortex workspace."""

    name = "list_directory"
    description = (
        "Lists files and folders within the safe ExoCortex workspace (~/.exocortex/workspace). "
        "Access outside the workspace is blocked."
    )
    permission_level = PermissionLevel.SAFE
    parameters = [
        ToolParameter(
            name="path",
            type="string",
            description="Relative path inside workspace to list (leave empty for workspace root).",
            required=False,
            default="",
        )
    ]

    def run(self, **kwargs: Any) -> ToolResult:
        rel_path = kwargs.get("path", "")
        resolved, err = resolve_safe_workspace_path(rel_path)
        if err or resolved is None:
            return ToolResult(success=False, output="", error=err)

        if not resolved.exists():
            return ToolResult(
                success=False,
                output="",
                error=f"Directory does not exist: '{rel_path}'",
            )

        if not resolved.is_dir():
            return ToolResult(
                success=False,
                output="",
                error=f"Path is not a directory: '{rel_path}'",
            )

        try:
            entries: List[Dict[str, Any]] = []
            lines: List[str] = [f"Contents of workspace folder '{resolved.name or 'workspace'}':"]

            for item in sorted(resolved.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
                stat = item.stat()
                is_dir = item.is_dir()
                mod_time = datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
                size_str = "<DIR>" if is_dir else f"{stat.st_size} bytes"

                entries.append({
                    "name": item.name,
                    "is_directory": is_dir,
                    "size_bytes": stat.st_size if not is_dir else 0,
                    "modified_time": mod_time,
                })
                lines.append(f"  {'[DIR] ' if is_dir else '[FILE]'} {item.name:<30} {size_str:>15}  {mod_time}")

            if len(entries) == 0:
                lines.append("  (Directory is empty)")

            output_text = "\n".join(lines)
            return ToolResult(
                success=True,
                output=output_text,
                data={"path": str(resolved), "count": len(entries), "entries": entries},
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to list directory: {str(e)}",
            )


class ReadTextFileTool(BaseTool):
    """Tool to safely read text files within the ExoCortex workspace."""

    name = "read_text_file"
    description = (
        "Reads the text content of a file located within the safe ExoCortex workspace. "
        "Access outside the workspace is blocked."
    )
    permission_level = PermissionLevel.SAFE
    parameters = [
        ToolParameter(
            name="path",
            type="string",
            description="Relative path of the text file inside the workspace to read.",
            required=True,
        )
    ]

    def run(self, **kwargs: Any) -> ToolResult:
        rel_path = kwargs.get("path")
        if not rel_path or not isinstance(rel_path, str):
            return ToolResult(
                success=False,
                output="",
                error="Missing required 'path' parameter.",
            )

        resolved, err = resolve_safe_workspace_path(rel_path)
        if err or resolved is None:
            return ToolResult(success=False, output="", error=err)

        if not resolved.exists():
            return ToolResult(
                success=False,
                output="",
                error=f"File not found: '{rel_path}' inside workspace.",
            )

        if not resolved.is_file():
            return ToolResult(
                success=False,
                output="",
                error=f"Target path is a directory, not a file: '{rel_path}'",
            )

        try:
            stat = resolved.stat()
            if stat.st_size > MAX_READ_BYTES:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"File exceeds maximum allowed read size of 1 MB ({stat.st_size} bytes).",
                )

            content = resolved.read_text(encoding="utf-8", errors="replace")
            return ToolResult(
                success=True,
                output=content,
                data={"path": str(resolved), "size_bytes": stat.st_size},
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to read file '{rel_path}': {str(e)}",
            )


class CreateDirectoryTool(BaseTool):
    """Tool to safely create a directory inside the ExoCortex workspace."""

    name = "create_directory"
    description = (
        "Creates a new directory inside the safe ExoCortex workspace. "
        "Creation outside the workspace is blocked."
    )
    permission_level = PermissionLevel.SAFE
    parameters = [
        ToolParameter(
            name="path",
            type="string",
            description="Relative path of the directory to create inside the workspace.",
            required=True,
        )
    ]

    def run(self, **kwargs: Any) -> ToolResult:
        rel_path = kwargs.get("path")
        if not rel_path or not isinstance(rel_path, str):
            return ToolResult(
                success=False,
                output="",
                error="Missing required 'path' parameter.",
            )

        clean_path = rel_path.strip().strip("'\"")
        if not clean_path:
            return ToolResult(
                success=False,
                output="",
                error="Directory path cannot be empty.",
            )

        resolved, err = resolve_safe_workspace_path(clean_path)
        if err or resolved is None:
            return ToolResult(success=False, output="", error=err)

        try:
            resolved.mkdir(parents=True, exist_ok=True)
            logger.info("Created directory inside workspace: %s", resolved)
            return ToolResult(
                success=True,
                output=f"Successfully created directory: '{clean_path}' in workspace.",
                data={"path": str(resolved), "relative_path": clean_path},
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to create directory '{clean_path}': {str(e)}",
            )
