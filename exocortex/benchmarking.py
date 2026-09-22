"""
Local SLM inference benchmarking and hardware performance suite for ExoCortex.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import psutil

from exocortex.config import Settings, get_config
from exocortex.slm.base import SLMProvider
from exocortex.slm.factory import get_slm_provider


@dataclass
class BenchmarkResult:
    """Benchmark results from running real local SLM neural inference tests."""
    model_name: str
    benchmark_type: str
    requested_provider: str
    execution_provider: str
    is_npu_accelerated: bool
    model_load_time_ms: float
    iterations: int
    avg_total_latency_ms: float
    avg_prompt_latency_ms: float
    avg_generation_latency_ms: float
    avg_tokens_per_second: float
    total_tokens_generated: int
    peak_ram_mb: float
    fallback_occurred: bool = False
    fallback_reason: Optional[str] = None
    runs: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "benchmark_type": self.benchmark_type,
            "requested_provider": self.requested_provider,
            "execution_provider": self.execution_provider,
            "is_npu_accelerated": self.is_npu_accelerated,
            "model_load_time_ms": round(self.model_load_time_ms, 2),
            "iterations": self.iterations,
            "avg_total_latency_ms": round(self.avg_total_latency_ms, 2),
            "avg_prompt_latency_ms": round(self.avg_prompt_latency_ms, 2),
            "avg_generation_latency_ms": round(self.avg_generation_latency_ms, 2),
            "avg_tokens_per_second": round(self.avg_tokens_per_second, 2),
            "total_tokens_generated": self.total_tokens_generated,
            "peak_ram_mb": round(self.peak_ram_mb, 2),
            "fallback_occurred": self.fallback_occurred,
            "fallback_reason": self.fallback_reason,
            "runs": self.runs,
        }


BENCHMARK_PROMPTS = [
    "Check system hardware information and memory.",
    "Run diagnostics and inspect system health.",
    "Echo 'Qualcomm Snapdragon AI Lab Challenge'.",
]


def run_benchmark(
    provider: Optional[SLMProvider] = None,
    prompts: Optional[List[str]] = None,
) -> BenchmarkResult:
    """
    Run real local inference benchmarking suite across standard test prompts.
    """
    slm = provider or get_slm_provider()
    test_prompts = prompts or BENCHMARK_PROMPTS

    model_info = slm.get_model_info()
    model_name = model_info.get("model_name", "qwen2.5-0.5b-instruct")
    req_provider = model_info.get("requested_provider", "auto")
    exec_provider = model_info.get("execution_provider", "CPUExecutionProvider")
    is_npu = model_info.get("is_npu_accelerated", False)
    fallback_occurred = model_info.get("fallback_occurred", False)
    fallback_reason = model_info.get("fallback_reason")
    load_time = float(model_info.get("load_time_ms", 0.0))

    # Benchmark classification: strictly distinguish CPU vs Snapdragon/QNN measurements
    benchmark_type = "Snapdragon/QNN benchmark" if (is_npu and exec_provider == "QNNExecutionProvider") else "CPU benchmark"

    run_records: List[Dict[str, Any]] = []
    tot_latencies = []
    prompt_latencies = []
    gen_latencies = []
    tps_list = []
    total_tokens = 0

    for i, p in enumerate(test_prompts, start=1):
        resp = slm.generate(p, max_tokens=60)
        metrics = getattr(slm, "last_metrics", None)

        p_lat = metrics.prompt_latency_ms if metrics else 0.0
        g_lat = metrics.generation_latency_ms if metrics else resp.latency_ms
        tps = metrics.tokens_per_second if metrics else (
            (resp.completion_tokens / (resp.latency_ms / 1000.0)) if resp.latency_ms > 0 else 0.0
        )
        mem_mb = metrics.ram_usage_mb if metrics else (psutil.Process().memory_info().rss / (1024 * 1024))

        tot_latencies.append(resp.latency_ms)
        prompt_latencies.append(p_lat)
        gen_latencies.append(g_lat)
        tps_list.append(tps)
        total_tokens += resp.completion_tokens

        run_records.append({
            "run_index": i,
            "prompt": p,
            "total_latency_ms": round(resp.latency_ms, 2),
            "prompt_latency_ms": round(p_lat, 2),
            "generation_latency_ms": round(g_lat, 2),
            "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens,
            "tokens_per_second": round(tps, 2),
            "ram_mb": round(mem_mb, 2),
        })

    avg_tot = sum(tot_latencies) / len(tot_latencies) if tot_latencies else 0.0
    avg_p_lat = sum(prompt_latencies) / len(prompt_latencies) if prompt_latencies else 0.0
    avg_gen = sum(gen_latencies) / len(gen_latencies) if gen_latencies else 0.0
    avg_tps = sum(tps_list) / len(tps_list) if tps_list else 0.0
    peak_ram = max(r["ram_mb"] for r in run_records) if run_records else 0.0

    return BenchmarkResult(
        model_name=model_name,
        benchmark_type=benchmark_type,
        requested_provider=req_provider,
        execution_provider=exec_provider,
        is_npu_accelerated=is_npu,
        model_load_time_ms=load_time,
        iterations=len(test_prompts),
        avg_total_latency_ms=avg_tot,
        avg_prompt_latency_ms=avg_p_lat,
        avg_generation_latency_ms=avg_gen,
        avg_tokens_per_second=avg_tps,
        total_tokens_generated=total_tokens,
        peak_ram_mb=peak_ram,
        fallback_occurred=fallback_occurred,
        fallback_reason=fallback_reason,
        runs=run_records,
    )
