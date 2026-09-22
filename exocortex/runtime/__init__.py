"""
ExoCortex Runtime & Hardware Acceleration Dispatch Layer.

Provides hardware detection, Qualcomm Snapdragon NPU discovery, and deterministic
execution provider resolution for ONNX Runtime.
"""

from __future__ import annotations

from exocortex.runtime.capabilities import (
    CapabilityStatus,
    ExecutionProviderTarget,
    NPUCapability,
    RuntimeProfile,
)
from exocortex.runtime.detector import (
    detect_available_providers,
    detect_memory_gb,
    detect_npu_capability,
    get_onnxruntime_version,
    is_snapdragon_hardware,
)
from exocortex.runtime.provider import RuntimeDispatcher

__all__ = [
    "CapabilityStatus",
    "ExecutionProviderTarget",
    "NPUCapability",
    "RuntimeProfile",
    "RuntimeDispatcher",
    "detect_available_providers",
    "detect_memory_gb",
    "detect_npu_capability",
    "get_onnxruntime_version",
    "is_snapdragon_hardware",
]
