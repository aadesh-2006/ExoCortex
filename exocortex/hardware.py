"""
Hardware detection and Snapdragon NPU acceleration discovery for ExoCortex.

Detects system architecture, Qualcomm Snapdragon NPU readiness, DirectML,
and available ONNX Runtime execution providers.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class HardwareProfile:
    """Hardware profile of the host machine."""

    os_name: str
    os_version: str
    os_release: str
    architecture: str
    processor_name: str
    is_arm64: bool
    is_snapdragon_detected: bool
    is_npu_available: bool
    npu_name: Optional[str]
    available_execution_providers: List[str] = field(default_factory=list)
    recommended_execution_provider: str = "CPUExecutionProvider"
    total_memory_gb: float = 0.0
    available_memory_gb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "os_name": self.os_name,
            "os_version": self.os_version,
            "os_release": self.os_release,
            "architecture": self.architecture,
            "processor_name": self.processor_name,
            "is_arm64": self.is_arm64,
            "is_snapdragon_detected": self.is_snapdragon_detected,
            "is_npu_available": self.is_npu_available,
            "npu_name": self.npu_name,
            "available_execution_providers": self.available_execution_providers,
            "recommended_execution_provider": self.recommended_execution_provider,
            "total_memory_gb": round(self.total_memory_gb, 2),
            "available_memory_gb": round(self.available_memory_gb, 2),
        }


def detect_execution_providers() -> List[str]:
    """Detect execution providers supported by installed ONNX Runtime."""
    try:
        import onnxruntime as ort
        return ort.get_available_providers()
    except ImportError:
        # Fallback list if onnxruntime is not yet installed in the current python env
        return ["CPUExecutionProvider"]


def detect_memory_info() -> tuple[float, float]:
    """Detect total and available RAM in GB."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return mem.total / (1024**3), mem.available / (1024**3)
    except ImportError:
        # Fallback on Windows if psutil is not yet available
        if platform.system() == "Windows":
            try:
                import ctypes
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]
                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
                return stat.ullTotalPhys / (1024**3), stat.ullAvailPhys / (1024**3)
            except Exception:
                pass
        return 0.0, 0.0


def detect_hardware() -> HardwareProfile:
    """Detect system hardware and Snapdragon NPU capabilities."""
    os_name = platform.system()
    os_version = platform.version()
    os_release = platform.release()
    arch = platform.machine().lower()
    proc = platform.processor()

    is_arm64 = "arm" in arch or "aarch64" in arch
    is_snapdragon = (
        "snapdragon" in proc.lower()
        or "qualcomm" in proc.lower()
        or "sc8380" in proc.lower()
        or "x elite" in proc.lower()
        or "x plus" in proc.lower()
    )

    providers = detect_execution_providers()

    # Determine NPU support: strictly require Snapdragon hardware AND QNN Execution Provider
    has_qnn = "QNNExecutionProvider" in providers
    has_dml = "DmlExecutionProvider" in providers

    is_npu_available = is_snapdragon and has_qnn
    npu_name = None
    if is_snapdragon and has_qnn:
        npu_name = "Qualcomm Hexagon NPU (Snapdragon X Series - QNN Active)"
    elif is_snapdragon:
        npu_name = "Qualcomm Hexagon NPU (Requires onnxruntime-qnn)"
    elif has_dml:
        npu_name = "DirectML Neural Accelerator / GPU"
    elif has_qnn:
        npu_name = "QNN Execution Provider (Non-Snapdragon Host)"

    # Select recommended provider
    if has_qnn and is_snapdragon:
        recommended_provider = "QNNExecutionProvider"
    elif has_dml:
        recommended_provider = "DmlExecutionProvider"
    else:
        recommended_provider = "CPUExecutionProvider"

    total_mem, avail_mem = detect_memory_info()

    return HardwareProfile(
        os_name=os_name,
        os_version=os_version,
        os_release=os_release,
        architecture=arch,
        processor_name=proc,
        is_arm64=is_arm64,
        is_snapdragon_detected=is_snapdragon,
        is_npu_available=is_npu_available,
        npu_name=npu_name,
        available_execution_providers=providers,
        recommended_execution_provider=recommended_provider,
        total_memory_gb=total_mem,
        available_memory_gb=avail_mem,
    )
