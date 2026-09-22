"""
Safe hardware and execution provider detector for ExoCortex.

Detects CPU architecture, Qualcomm Snapdragon processors, ONNX Runtime packages,
and QNN Execution Provider presence without external dependencies or risky calls.
"""

from __future__ import annotations

import logging
import os
import platform
from typing import Any, Dict, List, Optional, Tuple

from exocortex.runtime.capabilities import NPUCapability

logger = logging.getLogger("exocortex.runtime.detector")


def get_onnxruntime_version() -> str:
    """Retrieve installed ONNX Runtime version safely."""
    try:
        import onnxruntime as ort
        return getattr(ort, "__version__", "unknown")
    except ImportError:
        return "not installed"


def detect_available_providers() -> List[str]:
    """
    Detect available ONNX Runtime execution providers in the current environment.
    Safe: will not raise exceptions if ONNX is missing or a provider DLL is unavailable.
    """
    try:
        import onnxruntime as ort
        return list(ort.get_available_providers())
    except Exception as e:
        logger.debug("Failed to query ONNX Runtime available providers: %s", e)
        return ["CPUExecutionProvider"]


def is_snapdragon_hardware() -> bool:
    """
    Check if the host processor is a Qualcomm Snapdragon CPU.
    Examines processor string and system architecture.
    """
    proc = platform.processor().lower()
    arch = platform.machine().lower()
    
    # Check processor identifier tokens
    snapdragon_tokens = (
        "snapdragon",
        "qualcomm",
        "sc8380",    # Snapdragon X Elite silicon ID
        "sc8350",    # Snapdragon X Plus
        "x elite",
        "x plus",
        "8cx",
    )
    is_qualcomm_proc = any(token in proc for token in snapdragon_tokens)
    is_arm_arch = ("arm" in arch or "aarch64" in arch)

    return is_qualcomm_proc or (is_arm_arch and "qualcomm" in proc)


def detect_npu_capability(providers: Optional[List[str]] = None) -> NPUCapability:
    """
    Inspect Snapdragon NPU readiness and QNN execution provider support.
    
    Rules:
    - is_supported: True if host is Snapdragon / Qualcomm hardware.
    - is_installed: True if QNNExecutionProvider is present in ONNX Runtime.
    - is_active: Determined when session initializes; default False.
    """
    available_providers = providers if providers is not None else detect_available_providers()
    is_snapdragon = is_snapdragon_hardware()
    has_qnn = "QNNExecutionProvider" in available_providers

    if is_snapdragon and has_qnn:
        status_summary = "READY (Supported & Installed)"
        npu_name = "Qualcomm Hexagon NPU (Snapdragon X Series)"
    elif is_snapdragon and not has_qnn:
        status_summary = "SUPPORTED (QNN Execution Provider not installed)"
        npu_name = "Qualcomm Hexagon NPU (Requires onnxruntime-qnn)"
    elif has_qnn and not is_snapdragon:
        status_summary = "INSTALLED (Non-Snapdragon host; QNN emulation/unavailable)"
        npu_name = "QNN Execution Provider (Non-ARM64 Host)"
    else:
        status_summary = "UNAVAILABLE (Non-Snapdragon host)"
        npu_name = None

    return NPUCapability(
        is_supported=is_snapdragon,
        is_installed=has_qnn,
        is_active=False,
        npu_name=npu_name,
        status_summary=status_summary,
    )


def detect_memory_gb() -> Tuple[float, float]:
    """Return (total_memory_gb, available_memory_gb) safely."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return mem.total / (1024**3), mem.available / (1024**3)
    except ImportError:
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
