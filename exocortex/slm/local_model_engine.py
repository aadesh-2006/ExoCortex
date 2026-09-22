"""
Local SLM inference engine for ExoCortex.

Executes real on-device neural SLM inference using ONNX Runtime with CPU fallback
and Snapdragon Hexagon NPU readiness.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil

from exocortex.config import get_config
from exocortex.hardware import detect_hardware
from exocortex.slm.base import (
    Role,
    SLMMessage,
    SLMProvider,
    SLMResponse,
    ToolCallRequest,
)
from exocortex.slm.model_manager import (
    DEFAULT_MODEL_NAME,
    check_model_status,
    download_model_assets,
    get_model_directory,
)
from exocortex.slm.onnx_engine import GenerationOutput, ONNXNeuralGenerator

logger = logging.getLogger("exocortex.slm.engine")


@dataclass
class LocalInferenceMetrics:
    """Telemetry and performance metrics for local SLM neural inference."""
    model_name: str
    execution_provider: str
    is_npu_accelerated: bool
    model_load_time_ms: float = 0.0
    prompt_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tokens_per_second: float = 0.0
    ram_usage_mb: float = 0.0
    ram_delta_mb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "execution_provider": self.execution_provider,
            "is_npu_accelerated": self.is_npu_accelerated,
            "model_load_time_ms": round(self.model_load_time_ms, 2),
            "prompt_latency_ms": round(self.prompt_latency_ms, 2),
            "generation_latency_ms": round(self.generation_latency_ms, 2),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "tokens_per_second": round(self.tokens_per_second, 2),
            "ram_usage_mb": round(self.ram_usage_mb, 2),
            "ram_delta_mb": round(self.ram_delta_mb, 2),
        }


class LocalSLMProvider(SLMProvider):
    """
    Real Local SLM Reasoning Engine for Windows.
    
    Primary MVP Model: Qwen2.5-0.5B-Instruct ONNX.
    Execution: Real on-device neural tensor forward passes (Zero cloud APIs).
    Hardware Target: CPU fallback on current host / Snapdragon Hexagon NPU via QNN on Snapdragon Copilot+ PCs.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        model_path: Optional[str] = None,
        hardware_target: str = "auto",
        auto_download: bool = True,
    ) -> None:
        self.model_name = model_name
        self.hardware_target = hardware_target
        self.last_metrics: Optional[LocalInferenceMetrics] = None
        self.generator: Optional[ONNXNeuralGenerator] = None

        # Check model files status
        status = check_model_status(self.model_name)
        if not status.is_ready:
            if auto_download:
                logger.info("Model assets missing. Downloading %s...", self.model_name)
                status = download_model_assets(model_name=self.model_name)
            else:
                raise FileNotFoundError(
                    f"Required model files missing: {status.missing_files}. "
                    f"Please run model downloader or check {status.model_dir}."
                )

        model_file = Path(model_path) if model_path else status.file_paths["model_onnx"]
        tok_file = status.file_paths["tokenizer"]

        # Instantiate real ONNX neural generator with hardware-aware dispatch
        self.generator = ONNXNeuralGenerator(
            model_path=model_file,
            tokenizer_path=tok_file,
            execution_provider=self.hardware_target,
        )

    @property
    def resolved_provider(self) -> str:
        return self.active_provider

    @property
    def active_provider(self) -> str:
        return self.generator.active_provider if self.generator else "CPUExecutionProvider"

    @property
    def is_npu_active(self) -> bool:
        return self.generator.is_npu_active if self.generator else False

    def is_available(self) -> bool:
        """Check if real model and ONNX session are loaded and ready."""
        return self.generator is not None and self.generator.is_loaded

    def get_model_info(self) -> Dict[str, Any]:
        load_time = self.generator.load_time_ms if self.generator else 0.0
        model_size_mb = 0.0
        if self.generator and self.generator.model_path.exists():
            model_size_mb = self.generator.model_path.stat().st_size / (1024 * 1024)

        from exocortex.runtime.provider import RuntimeDispatcher
        runtime_prof = RuntimeDispatcher.get_runtime_profile(
            requested_target=self.hardware_target,
            active_provider=self.active_provider,
            is_npu_active=self.is_npu_active,
            fallback_occurred=self.generator.fallback_occurred if self.generator else False,
            fallback_reason=self.generator.fallback_reason if self.generator else None,
        )

        return {
            "model_name": self.model_name,
            "provider": "LocalSLMProvider (Real ONNX Neural Engine)",
            "requested_provider": self.hardware_target,
            "selected_provider": self.generator.selected_provider if self.generator else "CPUExecutionProvider",
            "execution_provider": self.active_provider,
            "is_npu_accelerated": self.is_npu_active,
            "is_local": True,
            "is_mock": False,
            "model_path": str(self.generator.model_path) if self.generator else None,
            "model_size_mb": round(model_size_mb, 1),
            "quantization": "INT8 (Static / Dynamic Quantized)",
            "inference_backend": "ONNX Runtime",
            "load_time_ms": round(load_time, 2),
            "fallback_occurred": self.generator.fallback_occurred if self.generator else False,
            "fallback_reason": self.generator.fallback_reason if self.generator else None,
            "runtime_profile": runtime_prof.to_dict(),
        }

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 160,
        **kwargs: Any,
    ) -> SLMResponse:
        """
        Execute real on-device neural generation using Qwen2.5-0.5B ONNX.
        """
        if not self.generator:
            raise RuntimeError("Local SLM Generator is not initialized.")

        process = psutil.Process(os.getpid())
        ram_before = process.memory_info().rss / (1024 * 1024)

        # Real neural generation forward pass
        gen_out: GenerationOutput = self.generator.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )

        ram_after = process.memory_info().rss / (1024 * 1024)

        # Record real telemetry metrics
        self.last_metrics = LocalInferenceMetrics(
            model_name=self.model_name,
            execution_provider=self.active_provider,
            is_npu_accelerated=self.is_npu_active,
            model_load_time_ms=self.generator.load_time_ms,
            prompt_latency_ms=gen_out.prompt_latency_ms,
            generation_latency_ms=gen_out.generation_latency_ms,
            total_latency_ms=gen_out.total_latency_ms,
            prompt_tokens=gen_out.prompt_tokens,
            completion_tokens=gen_out.completion_tokens,
            total_tokens=gen_out.total_tokens,
            tokens_per_second=gen_out.tokens_per_second,
            ram_usage_mb=ram_after,
            ram_delta_mb=ram_after - ram_before,
        )

        return SLMResponse(
            content=gen_out.text,
            prompt_tokens=gen_out.prompt_tokens,
            completion_tokens=gen_out.completion_tokens,
            latency_ms=gen_out.total_latency_ms,
            model_name=self.model_name,
            provider_name="LocalSLMProvider",
        )

    def chat(
        self,
        messages: List[SLMMessage],
        temperature: float = 0.1,
        max_tokens: int = 160,
        **kwargs: Any,
    ) -> SLMResponse:
        system_msgs = [m.content for m in messages if m.role == "system"]
        user_msgs = [m.content for m in messages if m.role == "user"]

        sys_prompt = "\n".join(system_msgs) if system_msgs else None
        last_prompt = user_msgs[-1] if user_msgs else (messages[-1].content if messages else "")

        return self.generate(
            prompt=last_prompt,
            system_prompt=sys_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
