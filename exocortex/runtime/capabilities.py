"""
Hardware capabilities, execution provider enums, and diagnostic representations for ExoCortex.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class CapabilityStatus(Enum):
    """Status of a hardware acceleration capability."""
    SUPPORTED = "supported"        # Supported by architecture/platform
    AVAILABLE = "available"        # Installed and ready in environment
    ACTIVE = "active"              # Currently loaded and executing inference
    UNAVAILABLE = "unavailable"    # Not present or cannot be used


class ExecutionProviderTarget(Enum):
    """Configured target execution provider."""
    AUTO = "auto"
    CPU = "cpu"
    DML = "dml"


@dataclass
class RuntimeProfile:
    """Comprehensive hardware, ONNX runtime, and acceleration profile."""
    # Host System
    os_name: str
    os_version: str
    os_release: str
    architecture: str
    processor_name: str
    is_arm64: bool

    # Memory
    total_memory_gb: float
    available_memory_gb: float

    # ONNX Runtime & Providers
    onnxruntime_version: str
    available_providers: List[str]
    requested_provider: str
    selected_provider: str
    active_provider: str
    fallback_occurred: bool = False
    fallback_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "os_name": self.os_name,
            "os_version": self.os_version,
            "os_release": self.os_release,
            "architecture": self.architecture,
            "processor_name": self.processor_name,
            "is_arm64": self.is_arm64,
            "total_memory_gb": round(self.total_memory_gb, 2),
            "available_memory_gb": round(self.available_memory_gb, 2),
            "onnxruntime_version": self.onnxruntime_version,
            "available_providers": self.available_providers,
            "requested_provider": self.requested_provider,
            "selected_provider": self.selected_provider,
            "active_provider": self.active_provider,
            "fallback_occurred": self.fallback_occurred,
            "fallback_reason": self.fallback_reason,
        }
