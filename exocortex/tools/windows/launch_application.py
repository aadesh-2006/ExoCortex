"""
Safe Windows application launching tool for ExoCortex.

Only explicitly allowlisted applications can be launched.
Arbitrary executable paths, shell expansions, and command strings are strictly forbidden.
"""

from __future__ import annotations

import logging
import subprocess
from typing import Any, Dict, List, Optional

from exocortex.tools.base import (
    BaseTool,
    PermissionLevel,
    ToolParameter,
    ToolResult,
)

logger = logging.getLogger("exocortex.tools.windows.launch_application")

# Explicit mapping of friendly names to system executables
ALLOWLISTED_APPLICATIONS: Dict[str, str] = {
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
    "calc": "calc.exe",
    "paint": "mspaint.exe",
    "mspaint": "mspaint.exe",
    "explorer": "explorer.exe",
    "file explorer": "explorer.exe",
    "windows explorer": "explorer.exe",
}


class LaunchApplicationTool(BaseTool):
    """Tool to safely launch known Windows desktop applications from an explicit allowlist."""

    name = "launch_application"
    description = (
        "Launches a known desktop application from the safe allowlist. "
        "Allowed: notepad, calculator, paint, explorer."
    )
    permission_level = PermissionLevel.SAFE
    parameters = [
        ToolParameter(
            name="application",
            type="string",
            description="Name of the application to launch (e.g. 'notepad', 'calculator', 'paint', 'explorer').",
            required=True,
        )
    ]

    def run(self, **kwargs: Any) -> ToolResult:
        app_raw = kwargs.get("application")
        if app_raw is None or not isinstance(app_raw, str):
            return ToolResult(
                success=False,
                output="",
                error="Missing or invalid 'application' parameter.",
            )

        app_clean = app_raw.strip().lower()
        if not app_clean:
            return ToolResult(
                success=False,
                output="",
                error="Application name cannot be empty.",
            )

        executable = ALLOWLISTED_APPLICATIONS.get(app_clean)
        if not executable:
            allowed_list = sorted(list(set(ALLOWLISTED_APPLICATIONS.keys())))
            return ToolResult(
                success=False,
                output="",
                error=(
                    f"Application '{app_raw}' is not in the safe allowlist. "
                    f"Supported applications: {', '.join(allowed_list)}. "
                    "Arbitrary executable paths and shell execution are not permitted."
                ),
            )

        try:
            # Launch strictly with shell=False, no arguments, and no shell expansion
            subprocess.Popen([executable], shell=False)
            logger.info("Launched application: %s (%s)", app_clean, executable)
            return ToolResult(
                success=True,
                output=f"Successfully launched application: {app_clean} ({executable})",
                data={"application": app_clean, "executable": executable},
            )
        except Exception as e:
            logger.error("Failed to launch application '%s': %s", executable, str(e))
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to launch '{executable}': {str(e)}",
            )
