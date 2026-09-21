"""
Windows system bridge and telemetry utilities for ExoCortex.
"""

from __future__ import annotations

import os
import platform
import subprocess
from typing import Any, Dict, List, Optional


class WindowsSystemBridge:
    """Interface to Windows OS subsystem telemetry and system utilities."""

    @staticmethod
    def is_windows() -> bool:
        return platform.system().lower() == "windows"

    @staticmethod
    def get_windows_version_info() -> Dict[str, Any]:
        return {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "architecture": platform.machine(),
            "windows_edition": platform.win32_edition() if hasattr(platform, "win32_edition") else "Unknown",
        }

    @staticmethod
    def get_environment_info() -> Dict[str, str]:
        """Safely extract safe environment indicators (excluding sensitive values)."""
        safe_keys = [
            "PROCESSOR_ARCHITECTURE",
            "PROCESSOR_IDENTIFIER",
            "NUMBER_OF_PROCESSORS",
            "OS",
            "USERNAME",
            "USERDOMAIN",
        ]
        return {k: os.environ.get(k, "") for k in safe_keys if k in os.environ}
