"""
ExoCortex Runtime & Hardware Acceleration Dispatch Layer.

Provides hardware detection and deterministic execution provider resolution for ONNX Runtime.
"""

from __future__ import annotations

from exocortex.runtime.capabilities import (
    CapabilityStatus,
    ExecutionProviderTarget,
    RuntimeProfile,
)
from exocortex.runtime.detector import (
    detect_available_providers,
    detect_memory_gb,
    get_onnxruntime_version,
)
from exocortex.runtime.provider import RuntimeDispatcher

__all__ = [
    "CapabilityStatus",
    "ExecutionProviderTarget",
    "RuntimeProfile",
    "RuntimeDispatcher",
    "detect_available_providers",
    "detect_memory_gb",
    "get_onnxruntime_version",
]
