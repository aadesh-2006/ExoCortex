"""
Health checks and diagnostic probe system for ExoCortex.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import exocortex
from exocortex.config import get_config
from exocortex.hardware import detect_hardware
from exocortex.slm.factory import get_slm_provider
from exocortex.tools.registry import get_default_tool_registry

HealthStatus = Literal["healthy", "degraded", "unhealthy"]


@dataclass
class HealthReport:
    """Consolidated health report across ExoCortex subsystems."""
    status: HealthStatus
    version: str
    python_version: str
    hardware: Dict[str, Any]
    slm_provider: Dict[str, Any]
    registered_tools: List[str]
    checks: Dict[str, bool] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "version": self.version,
            "python_version": self.python_version,
            "hardware": self.hardware,
            "slm_provider": self.slm_provider,
            "registered_tools": self.registered_tools,
            "checks": self.checks,
            "warnings": self.warnings,
        }


def run_health_check() -> HealthReport:
    """
    Run diagnostic probes across all core agent subsystems.
    """
    config = get_config()
    hardware_profile = detect_hardware()
    slm = get_slm_provider(config=config)
    tools = get_default_tool_registry()

    checks: Dict[str, bool] = {}
    warnings: List[str] = []

    # 1. Python check
    py_ok = sys.version_info >= (3, 11)
    checks["python_version_compatible"] = py_ok
    if not py_ok:
        warnings.append(f"Python version {sys.version} is older than recommended 3.11+")

    # 2. Hardware check
    checks["hardware_probed"] = True
    if not hardware_profile.is_snapdragon_detected:
        warnings.append("Qualcomm Snapdragon NPU not detected on current host; running CPU/DirectML emulation mode.")

    # 3. SLM check
    slm_ok = slm.is_available()
    checks["slm_provider_available"] = slm_ok
    if not slm_ok:
        warnings.append(f"Configured SLM provider '{config.slm_provider}' is not currently reachable.")

    # 4. Tool registry check
    tool_list = [t.name for t in tools.list_tools()]
    tools_ok = len(tool_list) > 0
    checks["tools_registered"] = tools_ok

    # Determine overall status
    if not py_ok or not slm_ok or not tools_ok:
        status: HealthStatus = "degraded"
    else:
        status = "healthy"

    return HealthReport(
        status=status,
        version=exocortex.__version__,
        python_version=sys.version.split()[0],
        hardware=hardware_profile.to_dict(),
        slm_provider=slm.get_model_info(),
        registered_tools=tool_list,
        checks=checks,
        warnings=warnings,
    )
