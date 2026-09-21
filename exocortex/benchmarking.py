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
from exocortex.slm.local_model_engine import LocalInferenceMetrics, LocalSLMProvider


@dataclass
class BenchmarkResult:
    """Benchmark results from running local SLM inference tests."""
    model_name: str
    execution_provider: str
    is_npu_accelerated: bool
    iterations: int
    avg_total_latency_ms: float
    avg_generation_latency_ms: float
    avg_tokens_per_second: float
    total_tokens_generated: int
    peak_ram_mb: float
    runs: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "execution_provider": self.execution_provider,
            "is_npu_accelerated": self.is_npu_accelerated,
            "iterations": self.iterations,
            "avg_total_latency_ms": round(self.avg_total_latency_ms, 2),
            "avg_generation_latency_ms": round(self.avg_generation_latency_ms, 2),
            "avg_tokens_per_second": round(self.avg_tokens_per_second, 2),
            "total_tokens_generated": self.total_tokens_generated,
            "peak_ram_mb": round(self.peak_ram_mb, 2),
            "runs": self.runs,
        }


BENCHMARK_PROMPTS = [
    "Check system hardware information and memory.",
    "Run diagnostics and inspect system health.",
    "Echo 'Qualcomm Snapdragon AI Lab Challenge'.",
    "What is ExoCortex and what are its core capabilities?",
]


def run_benchmark(
    provider: Optional[SLMProvider] = None,
    prompts: Optional[List[str]] = None,
) -> BenchmarkResult:
    """Run local inference benchmarking suite across standard prompts."""
    slm = provider or get_slm_provider()
    test_prompts = prompts or BENCHMARK_PROMPTS

    run_records: List[Dict[str, Any]] = []
    tot_latencies = []
    gen_latencies = []
    tps_list = []
    total_tokens = 0

    model_info = slm.get_model_info()
    model_name = model_info.get("model_name", "local-slm")
    exec_provider = model_info.get("execution_provider", "CPUExecutionProvider")
    is_npu = model_info.get("is_npu_accelerated", False)

    for i, p in enumerate(test_prompts, start=1):
        start_mem = psutil.Process().memory_info().rss / (1024 * 1024)
        resp = slm.generate(p)
        end_mem = psutil.Process().memory_info().rss / (1024 * 1024)

        tps = (resp.completion_tokens / (resp.latency_ms / 1000.0)) if resp.latency_ms > 0 else 0.0
        tot_latencies.append(resp.latency_ms)
        gen_latencies.append(resp.latency_ms)
        tps_list.append(tps)
        total_tokens += resp.completion_tokens

        run_records.append({
            "run_index": i,
            "prompt": p,
            "latency_ms": round(resp.latency_ms, 2),
            "completion_tokens": resp.completion_tokens,
            "tokens_per_second": round(tps, 2),
            "ram_mb": round(end_mem, 2),
        })

    avg_tot = sum(tot_latencies) / len(tot_latencies) if tot_latencies else 0.0
    avg_gen = sum(gen_latencies) / len(gen_latencies) if gen_latencies else 0.0
    avg_tps = sum(tps_list) / len(tps_list) if tps_list else 0.0
    peak_ram = max(r["ram_mb"] for r in run_records) if run_records else 0.0

    return BenchmarkResult(
        model_name=model_name,
        execution_provider=exec_provider,
        is_npu_accelerated=is_npu,
        iterations=len(test_prompts),
        avg_total_latency_ms=avg_tot,
        avg_generation_latency_ms=avg_gen,
        avg_tokens_per_second=avg_tps,
        total_tokens_generated=total_tokens,
        peak_ram_mb=peak_ram,
        runs=run_records,
    )
