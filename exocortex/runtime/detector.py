"""
Safe hardware and execution provider detector for ExoCortex.

Detects CPU architecture, system memory, ONNX Runtime packages,
and available execution providers without external dependencies or risky calls.
"""

from __future__ import annotations

import logging
import os
import platform
from typing import Any, Dict, List, Optional, Tuple

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
