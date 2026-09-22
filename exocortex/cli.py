"""
Command-line interface (CLI) for ExoCortex.

Entry point for running diagnostics, hardware inspection, local SLM benchmarking,
and executing autonomous agent tasks on Windows.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import List, Optional

import exocortex
from exocortex.agent import ExoCortexAgent
from exocortex.benchmarking import run_benchmark
from exocortex.config import get_config
from exocortex.hardware import detect_hardware
from exocortex.health import run_health_check

try:
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def format_header(title: str) -> str:
    line = "=" * 64
    return f"\n{line}\n  {title}\n{line}"


def cmd_status(args: argparse.Namespace) -> int:
    """Run health check and print system status with runtime dispatch details."""
    report = run_health_check()
    config = get_config()
    slm = report.slm_provider
    hw = report.hardware
    runtime_prof = slm.get("runtime_profile", {})
    
    if args.json:
        data = report.to_dict()
        data["workspace_dir"] = str(config.workspace_dir)
        print(json.dumps(data, indent=2))
        return 0

    print(format_header(f"ExoCortex v{report.version} - System & Runtime Status"))
    status_symbol = "[OK]" if report.status == "healthy" else "[WARN]"
    print(f"Overall Status: {status_symbol} {report.status.upper()}")
    print(f"Python Runtime: {report.python_version}")
    print(f"Workspace Dir:  {config.workspace_dir}")
    
    print("\n[Hardware & Host System]")
    print(f"  OS:                     {hw['os_name']} {hw['os_release']} ({hw['architecture']})")
    print(f"  Processor:              {hw['processor_name']}")
    print(f"  ARM64 Native:           {hw.get('is_arm64', False)}")
    print(f"  Snapdragon CPU:         {'Detected' if hw['is_snapdragon_detected'] else 'Not Detected'}")
    print(f"  Memory (RAM):           {hw['available_memory_gb']:.1f} GB available / {hw['total_memory_gb']:.1f} GB total")

    print("\n[ONNX Runtime & Hardware Acceleration]")
    print(f"  ONNX Runtime Version:   {runtime_prof.get('onnxruntime_version', '1.18.0')}")
    print(f"  Available Providers:    {', '.join(runtime_prof.get('available_providers', hw.get('available_execution_providers', [])))}")
    print(f"  Configured Target:      {config.execution_provider}")
    print(f"  Selected Provider:      {slm.get('selected_provider', slm.get('execution_provider', 'CPUExecutionProvider'))}")
    print(f"  Active Provider:        {slm.get('execution_provider', 'CPUExecutionProvider')}")
    print(f"  NPU Acceleration:       {'ACTIVE' if slm.get('is_npu_accelerated', False) else 'INACTIVE (CPU Fallback)'}")
    if slm.get("fallback_occurred"):
        print(f"  Fallback Notice:        {slm.get('fallback_reason')}")

    print("\n[SLM Reasoning Brain]")
    print(f"  Provider:               {slm.get('provider')}")
    print(f"  Model Name:             {slm.get('model_name')}")
    print(f"  Model Size:             {slm.get('model_size_mb', '~350')} MB")
    print(f"  Quantization:           {slm.get('quantization', 'INT8')}")
    print(f"  Inference Backend:      {slm.get('inference_backend', 'ONNX Runtime')}")
    print(f"  Local Only:             {slm.get('is_local', True)}")

    print("\n[Tool Registry]")
    print(f"  Active Tools:           {', '.join(report.registered_tools)}")

    if report.warnings:
        print("\n[Notices & Warnings]")
        for w in report.warnings:
            print(f"  - {w}")

    print("=" * 64 + "\n")
    return 0


def cmd_tools(args: argparse.Namespace) -> int:
    """List all registered tools with their schemas and security permission levels."""
    from exocortex.tools.registry import get_default_tool_registry

    reg = get_default_tool_registry()
    tools = reg.list_tools()

    if args.json:
        print(json.dumps([t.get_schema() for t in tools], indent=2))
        return 0

    print(format_header("ExoCortex Registered Tools & Security Permissions"))
    print(f"Total Registered Tools: {len(tools)}\n")

    for t in tools:
        perm_label = f"[{t.permission_level.value.upper()}]"
        print(f"• {t.name:<24} {perm_label:<18}")
        print(f"    Description: {t.description}")
        if t.parameters:
            params_str = ", ".join(f"{p.name} ({p.type}{', required' if p.required else ', optional'})" for p in t.parameters)
            print(f"    Parameters:  {params_str}")
        else:
            print(f"    Parameters:  None")
        print()

    print("=" * 64 + "\n")
    return 0


def cmd_hardware(args: argparse.Namespace) -> int:
    """Inspect and display detailed hardware, NPU readiness, and runtime dispatch capabilities."""
    from exocortex.runtime.provider import RuntimeDispatcher
    config = get_config()
    runtime_prof = RuntimeDispatcher.get_runtime_profile(requested_target=config.execution_provider)
    hw = detect_hardware()

    if args.json:
        print(json.dumps(runtime_prof.to_dict(), indent=2))
        return 0

    print(format_header("ExoCortex Hardware & Snapdragon Acceleration Profile"))
    print(f"Host OS:                   {runtime_prof.os_name} {runtime_prof.os_release} (v{runtime_prof.os_version})")
    print(f"Architecture:              {runtime_prof.architecture}")
    print(f"ARM64 Native:              {runtime_prof.is_arm64}")
    print(f"Processor:                 {runtime_prof.processor_name}")
    print(f"Snapdragon Silicon:        {'Detected' if runtime_prof.is_snapdragon_detected else 'Not Detected'}")
    print(f"Total Physical Memory:     {runtime_prof.total_memory_gb:.2f} GB")
    print(f"Available Physical Memory: {runtime_prof.available_memory_gb:.2f} GB")

    print("\n--- ONNX Runtime Dispatch ---")
    print(f"ONNX Runtime Version:      {runtime_prof.onnxruntime_version}")
    print(f"Available Providers:       {', '.join(runtime_prof.available_providers)}")
    print(f"Configured Provider:       {runtime_prof.requested_provider}")
    print(f"Selected Provider:         {runtime_prof.selected_provider}")
    print(f"Active Provider:           {runtime_prof.active_provider}")

    print("\n--- Qualcomm Snapdragon / QNN NPU Readiness ---")
    npu = runtime_prof.npu
    supported_str = "YES" if npu.is_supported else "NO (x86_64 / Intel Host)"
    installed_str = "YES" if npu.is_installed else "NO (QNNExecutionProvider not in ORT)"
    active_str = "YES (NPU Active)" if npu.is_active else "NO (CPU Fallback Active)"

    print(f"Supported by Architecture: {supported_str}")
    print(f"QNN Provider Installed:    {installed_str}")
    print(f"NPU Acceleration Active:   {active_str}")
    print(f"NPU Status Summary:        {npu.status_summary}")

    print("=" * 64 + "\n")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run local SLM inference benchmarks and display performance metrics."""
    print(format_header("ExoCortex Local SLM Inference Benchmark"))
    print("Running on-device inference performance suite...\n")

    res = run_benchmark()

    if args.json:
        print(json.dumps(res.to_dict(), indent=2))
        return 0

    print(f"Benchmark Category:      {res.benchmark_type}")
    print(f"Model Name:              {res.model_name}")
    print(f"Requested Provider:      {res.requested_provider}")
    print(f"Execution Provider:      {res.execution_provider}")
    print(f"NPU Accelerated:         {res.is_npu_accelerated}")
    if res.fallback_occurred:
        print(f"Fallback Reason:         {res.fallback_reason}")
    print(f"Model Load Time:         {res.model_load_time_ms:.1f} ms")
    print(f"Benchmark Iterations:    {res.iterations}")
    print(f"Avg Total Latency:       {res.avg_total_latency_ms:.1f} ms")
    print(f"Avg Prompt (Prefill):    {res.avg_prompt_latency_ms:.1f} ms")
    print(f"Avg Generation Latency:  {res.avg_generation_latency_ms:.1f} ms")
    print(f"Avg Inference Speed:     {res.avg_tokens_per_second:.1f} tokens/sec")
    print(f"Total Tokens Generated:  {res.total_tokens_generated}")
    print(f"Process Peak RAM:        {res.peak_ram_mb:.1f} MB")

    print("\n--- Iteration Breakdown ---")
    for r in res.runs:
        print(f" Run #{r['run_index']}: {r['total_latency_ms']:.1f} ms (Prompt: {r['prompt_latency_ms']:.1f} ms, Gen: {r['generation_latency_ms']:.1f} ms) | {r['completion_tokens']} tokens | {r['tokens_per_second']:.1f} tok/s | RAM: {r['ram_mb']:.1f} MB")
        print(f"   Prompt: '{r['prompt']}'")

    print("=" * 64 + "\n")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Run an autonomous agent task with natural language prompt."""
    prompt = args.prompt
    if not prompt:
        print("Error: Please provide a prompt via positional argument or --prompt.")
        return 1

    agent = ExoCortexAgent()
    result = agent.run(prompt)

    if getattr(args, "json", False):
        print(json.dumps(result.to_dict(), indent=2))
        return 0 if result.success else 1

    print(format_header("ExoCortex Autonomous Agent Execution"))
    print(f"User Prompt: '{prompt}'\n")

    print(f"Execution Status: {'SUCCESS' if result.success else 'FAILED'}")
    print(f"Total Latency:    {result.total_latency_ms:.1f} ms")
    print(f"SLM Engine:       {result.slm_info.get('provider')} ({result.slm_info.get('model_name')})")

    if result.trace and result.trace.steps:
        print("\n--- Multi-Step Execution Trace ---")
        for s in result.trace.steps:
            if s.tool_name:
                status_icon = "[OK]" if (s.tool_result and s.tool_result.success) else "[FAIL]"
                print(f" {status_icon} Step {s.step_number}: {s.tool_name}({s.arguments}) -> {s.observation_text}")
            else:
                status_icon = "[DONE]" if s.status.value == "completed" else "[ERROR]"
                print(f" {status_icon} Step {s.step_number}: {s.action_type.value} -> {s.observation_text or s.error_message}")

    elif result.executed_tools:
        print("\n--- Executed Steps ---")
        for t in result.executed_tools:
            status_icon = "[OK]" if t["success"] else "[FAIL]"
            print(f" {status_icon} Step {t['step_id']}: {t['tool']} -> {t['output'] or t['error']}")

    print("\n--- Final Agent Response ---")
    print(result.response_text)

    if getattr(args, "trace", False) and result.trace:
        print("\n--- Full Debug Trace (JSON) ---")
        print(json.dumps(result.trace.to_dict(), indent=2))

    print("=" * 64 + "\n")
    return 0 if result.success else 1


def cmd_interactive(args: argparse.Namespace) -> int:
    """Run interactive REPL loop with ExoCortex."""
    print(format_header(f"ExoCortex v{exocortex.__version__} Interactive Shell"))
    print("Type your request below. Type 'exit' or 'quit' to stop.\n")

    agent = ExoCortexAgent()

    while True:
        try:
            prompt = input("ExoCortex> ").strip()
            if not prompt:
                continue
            if prompt.lower() in ("exit", "quit", "q"):
                print("Exiting ExoCortex. Goodbye!")
                break

            result = agent.run(prompt)
            print(f"\n[Agent]:\n{result.response_text}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="exocortex",
        description="ExoCortex: Local-First JARVIS-like PC Agent for Windows (Qualcomm Snapdragon Challenge)",
    )
    parser.add_argument(
        "--version", "-v", action="version", version=f"ExoCortex v{exocortex.__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="ExoCortex subcommands")

    # Status / Health / Info command
    status_parser = subparsers.add_parser("status", help="Check system health and status")
    status_parser.add_argument("--json", action="store_true", help="Output health report in JSON format")
    status_parser.set_defaults(func=cmd_status)

    info_parser = subparsers.add_parser("info", help="Alias for status check")
    info_parser.add_argument("--json", action="store_true", help="Output health report in JSON format")
    info_parser.set_defaults(func=cmd_status)

    health_parser = subparsers.add_parser("health", help="Alias for status check")
    health_parser.add_argument("--json", action="store_true", help="Output health report in JSON format")
    health_parser.set_defaults(func=cmd_status)

    # Tools command
    tools_parser = subparsers.add_parser("tools", help="List all registered tools and security permissions")
    tools_parser.add_argument("--json", action="store_true", help="Output registered tools in JSON schema format")
    tools_parser.set_defaults(func=cmd_tools)

    # Hardware command
    hw_parser = subparsers.add_parser("hardware", help="Inspect hardware and Snapdragon acceleration")
    hw_parser.add_argument("--json", action="store_true", help="Output hardware profile in JSON format")
    hw_parser.set_defaults(func=cmd_hardware)

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run local SLM inference benchmarks")
    bench_parser.add_argument("--json", action="store_true", help="Output benchmark results in JSON format")
    bench_parser.set_defaults(func=cmd_benchmark)

    # Run command
    run_parser = subparsers.add_parser("run", help="Run a single natural language task")
    run_parser.add_argument("prompt", nargs="?", default="", help="Natural language prompt")
    run_parser.add_argument("--prompt", "-p", dest="prompt_flag", help="Natural language prompt")
    run_parser.add_argument("--json", action="store_true", help="Output execution result in JSON format")
    run_parser.add_argument("--trace", action="store_true", help="Display full multi-step execution trace")
    run_parser.set_defaults(
        func=lambda args: cmd_run(
            argparse.Namespace(
                prompt=args.prompt_flag or args.prompt,
                json=getattr(args, "json", False),
                trace=getattr(args, "trace", False),
            )
        )
    )

    # Interactive REPL command
    interactive_parser = subparsers.add_parser("interactive", help="Start interactive agent console")
    interactive_parser.set_defaults(func=cmd_interactive)

    args = parser.parse_args(argv)

    if not args.command:
        # Default action is status report
        return cmd_status(argparse.Namespace(json=False))

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
