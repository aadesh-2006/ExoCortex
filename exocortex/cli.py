"""
Command-line interface (CLI) for ExoCortex.

Entry point for running diagnostics, inspecting Snapdragon hardware acceleration,
and invoking autonomous agent tasks.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import List, Optional

import exocortex
from exocortex.agent import ExoCortexAgent
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
    """Run health check and print system status."""
    report = run_health_check()
    
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
        return 0

    print(format_header(f"ExoCortex v{report.version} - System Status"))
    status_symbol = "[OK]" if report.status == "healthy" else "[WARN]"
    print(f"Overall Status: {status_symbol} {report.status.upper()}")
    print(f"Python Runtime: {report.python_version}")
    
    print("\n[Hardware & Acceleration]")
    hw = report.hardware
    print(f"  OS:                 {hw['os_name']} {hw['os_release']} ({hw['architecture']})")
    print(f"  Processor:          {hw['processor_name']}")
    print(f"  Snapdragon NPU:     {'Detected' if hw['is_snapdragon_detected'] else 'Not Detected (Emulation Ready)'}")
    print(f"  Recommended EP:     {hw['recommended_execution_provider']}")
    print(f"  Memory (RAM):       {hw['available_memory_gb']:.1f} GB available / {hw['total_memory_gb']:.1f} GB total")

    print("\n[SLM Inference Provider]")
    slm = report.slm_provider
    print(f"  Provider:           {slm.get('provider')}")
    print(f"  Model Name:         {slm.get('model_name')}")
    print(f"  Local Only:         {slm.get('is_local')}")

    print("\n[Tool Registry]")
    print(f"  Active Tools:       {', '.join(report.registered_tools)}")

    if report.warnings:
        print("\n[Notices & Warnings]")
        for w in report.warnings:
            print(f"  - {w}")

    print("=" * 64 + "\n")
    return 0


def cmd_hardware(args: argparse.Namespace) -> int:
    """Inspect and display detailed hardware and Snapdragon accelerator profile."""
    hw = detect_hardware()
    if args.json:
        print(json.dumps(hw.to_dict(), indent=2))
        return 0

    print(format_header("ExoCortex Hardware & Snapdragon Acceleration Profile"))
    print(f"Host OS:                   {hw.os_name} {hw.os_release} (v{hw.os_version})")
    print(f"Architecture:              {hw.architecture}")
    print(f"ARM64 Native:              {hw.is_arm64}")
    print(f"Processor Name:            {hw.processor_name}")
    print(f"Snapdragon NPU Detected:   {hw.is_snapdragon_detected}")
    print(f"NPU / Accelerator Name:    {hw.npu_name or 'Standard CPU fallback'}")
    print(f"Available ONNX Providers:  {', '.join(hw.available_execution_providers)}")
    print(f"Recommended Provider:      {hw.recommended_execution_provider}")
    print(f"Total Physical Memory:     {hw.total_memory_gb:.2f} GB")
    print(f"Available Physical Memory: {hw.available_memory_gb:.2f} GB")
    print("=" * 64 + "\n")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Run an autonomous agent task with natural language prompt."""
    prompt = args.prompt
    if not prompt:
        print("Error: Please provide a prompt via --prompt or positional argument.")
        return 1

    print(format_header("ExoCortex Autonomous Agent Execution"))
    print(f"User Prompt: '{prompt}'\n")

    agent = ExoCortexAgent()
    result = agent.run(prompt)

    print(f"Execution Status: {'SUCCESS' if result.success else 'FAILED'}")
    print(f"Total Latency:    {result.total_latency_ms:.1f} ms")
    print(f"SLM Engine:       {result.slm_info.get('provider')} ({result.slm_info.get('model_name')})")
    
    print("\n--- Plan Breakdown ---")
    for step in result.plan.steps:
        status_icon = "[OK]" if step.status.value == "completed" else "[FAIL]"
        print(f" {status_icon} Step {step.step_id}: {step.description}")
        if step.tool_name:
            print(f"        Tool: {step.tool_name}")

    print("\n--- Agent Response ---")
    print(result.response_text)
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

    # Status / Health command
    status_parser = subparsers.add_parser("status", help="Check system health and status")
    status_parser.add_argument("--json", action="store_true", help="Output health report in JSON format")
    status_parser.set_defaults(func=cmd_status)

    health_parser = subparsers.add_parser("health", help="Alias for status check")
    health_parser.add_argument("--json", action="store_true", help="Output health report in JSON format")
    health_parser.set_defaults(func=cmd_status)

    # Hardware command
    hw_parser = subparsers.add_parser("hardware", help="Inspect hardware and Snapdragon acceleration")
    hw_parser.add_argument("--json", action="store_true", help="Output hardware profile in JSON format")
    hw_parser.set_defaults(func=cmd_hardware)

    # Run command
    run_parser = subparsers.add_parser("run", help="Run a single natural language task")
    run_parser.add_argument("prompt", nargs="?", default="", help="Natural language prompt")
    run_parser.add_argument("--prompt", "-p", dest="prompt_flag", help="Natural language prompt")
    run_parser.set_defaults(
        func=lambda args: cmd_run(
            argparse.Namespace(prompt=args.prompt_flag or args.prompt)
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
