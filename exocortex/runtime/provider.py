"""
Runtime provider resolution and execution provider dispatch for ExoCortex.

Manages deterministic provider selection between DirectML and CPU execution.
"""

from __future__ import annotations

import logging
import platform
from typing import Any, Dict, List, Optional, Tuple

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

logger = logging.getLogger("exocortex.runtime.provider")


class RuntimeDispatcher:
    """
    Dispatcher determining the appropriate execution provider based on configuration
    and verified hardware capabilities.
    """

    @classmethod
    def resolve_provider(
        cls,
        target: str = "auto",
        available_providers: Optional[List[str]] = None,
    ) -> Tuple[str, Optional[str]]:
        """
        Resolve the target configuration into a concrete execution provider name.
        
        Returns:
            (selected_provider_name, warning_or_fallback_reason)
        """
        providers = available_providers if available_providers is not None else detect_available_providers()
        target_norm = (target or "auto").strip().lower()

        if target_norm == "auto":
            # Priority: DirectML (if Windows GPU/accelerator available) -> CPU
            if "DmlExecutionProvider" in providers:
                return "DmlExecutionProvider", None
            else:
                return "CPUExecutionProvider", None

        elif target_norm == "dml":
            if "DmlExecutionProvider" not in providers:
                reason = "DmlExecutionProvider requested but not available in installed ONNX Runtime."
                logger.warning(reason)
                return "DmlExecutionProvider", reason
            return "DmlExecutionProvider", None

        elif target_norm == "cpu":
            return "CPUExecutionProvider", None

        else:
            reason = f"Unknown execution provider target '{target}'. Falling back to CPUExecutionProvider."
            logger.warning(reason)
            return "CPUExecutionProvider", reason

    @classmethod
    def get_runtime_profile(
        cls,
        requested_target: str = "auto",
        active_provider: Optional[str] = None,
        fallback_occurred: bool = False,
        fallback_reason: Optional[str] = None,
        **kwargs: Any,
    ) -> RuntimeProfile:
        """
        Construct a comprehensive RuntimeProfile reflecting actual system and inference state.
        """
        providers = detect_available_providers()
        ort_ver = get_onnxruntime_version()
        selected_ep, select_reason = cls.resolve_provider(requested_target, providers)
        
        act_ep = active_provider or selected_ep
        fb_occurred = fallback_occurred or (select_reason is not None and act_ep != selected_ep)
        fb_reason = fallback_reason or select_reason

        tot_mem, avail_mem = detect_memory_gb()

        return RuntimeProfile(
            os_name=platform.system(),
            os_version=platform.version(),
            os_release=platform.release(),
            architecture=platform.machine(),
            processor_name=platform.processor(),
            is_arm64="arm" in platform.machine().lower() or "aarch64" in platform.machine().lower(),
            total_memory_gb=tot_mem,
            available_memory_gb=avail_mem,
            onnxruntime_version=ort_ver,
            available_providers=providers,
            requested_provider=requested_target,
            selected_provider=selected_ep,
            active_provider=act_ep,
            fallback_occurred=fb_occurred,
            fallback_reason=fb_reason,
        )
