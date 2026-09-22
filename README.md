# ExoCortex 🧠⚡
**Local-First JARVIS-Like Autonomous Personal Computer Agent for Windows**  
*Built for the Qualcomm Snapdragon AI Lab Build & Present Challenge*

[![Platform](https://img.shields.io/badge/Platform-Windows%2011%20%7C%20ARM64%20%7C%20x64-blue.svg)](https://microsoft.com/windows)
[![Qualcomm](https://img.shields.io/badge/Hardware-Snapdragon%20X%20Elite%20%7C%20NPU-FF6600.svg)](https://www.qualcomm.com/products/mobile/snapdragon)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Milestone%205-Completed-brightgreen.svg)]()

---

## 1. Project Overview

**ExoCortex** is an autonomous, local-first personal computer assistant engineered specifically for **Windows** and optimized for **Qualcomm Snapdragon-powered Copilot+ PCs** (such as HP Snapdragon laptops). 

Unlike cloud-dependent conversational bots, ExoCortex functions as an intelligent cognitive layer residing directly on the user's computer. It receives natural-language user intentions, formulates structured multi-step execution plans, interacts with the Windows OS through strictly typed tools, observes the resulting state, and autonomously drives tasks to completion—all while keeping user data 100% private and on-device.

---

## 2. Problem Statement

Modern AI assistants and computer-use tools suffer from critical drawbacks:
1. **Cloud Latency & Privacy Risks**: Transmitting screen pixels, file hierarchies, and private user communications to external cloud servers introduces severe latency, high operational costs, and catastrophic privacy risks.
2. **Heavyweight LLMs & Battery Drain**: Traditional Large Language Models (70B+) require massive cloud server farms or heavy discrete GPUs that quickly drain laptop battery.
3. **Lack of Hardware-Aware Agent Architectures**: Existing agents treat hardware as a black box rather than leveraging dedicated on-device Neural Processing Units (NPUs) built into modern laptops.
4. **Unsafe Shell & Script Execution**: Agents that execute unrestricted bash, PowerShell, or command strings introduce severe security risks and unintended destructive actions.

---

## 3. Project Vision

ExoCortex is envisioned as a **JARVIS-like copilot for Windows**:
- **Natural Interaction**: Understands conversational, contextual instructions from the user.
- **Autonomous Multi-Step Execution**: Decomposes ambiguous goals into discrete tool actions and system interactions.
- **Safe Desktop Automation**: All OS capabilities are exposed through typed, sandboxed tool definitions—preventing arbitrary command injection.
- **Edge Efficiency**: Powered by Small Language Models (SLMs) accelerated on **Qualcomm Hexagon NPUs**, providing sub-second latency and all-day battery efficiency.

---

## 4. Architecture & Safe Agentic Execution Loop (Milestone 4)

ExoCortex implements a bounded, safe, iterative cognitive loop:

```mermaid
flowchart TD
    User([User Natural Language Goal]) --> CLI[ExoCortex CLI / Interface]
    CLI --> AgentCore[ExoCortexAgent Core]
    AgentCore --> Executor[AgentExecutor Engine]
    
    subgraph Cognitive Loop [Multi-Step Cognitive Execution Loop]
        Executor --> SLM[Local SLM Brain (Qwen2.5-0.5B ONNX)]
        SLM --> StructuredDec[Strict Structured Decision (AgentDecision)]
        StructuredDec --> Validator[Schema & Tool Validator]
        Validator --> LoopCheck{Loop / Step Limits OK?}
        LoopCheck -- Exceeded --> StopLoop[Halt: MAX_STEPS / LOOP_DETECTED]
        LoopCheck -- OK --> PermCheck{Permission & Confirmation Check}
        PermCheck -- Requires Confirm --> PauseConfirm[Pause: CONFIRMATION_REQUIRED]
        PermCheck -- Safe / Approved --> ToolReg[ToolRegistry.execute]
        ToolReg --> WinTools[Safe Windows OS Tool Suite]
        WinTools --> WinOS[Windows OS / Subsystems]
        WinOS --> Observation[Tool Observation (Bounded <= 2,000 Chars)]
        Observation --> HistTracker[Append Step to Execution History]
        HistTracker --> CheckGoal{Has More Steps?}
        CheckGoal -- Yes --> SLM
        CheckGoal -- No / Complete --> FinalResponse[Synthesize Final Response]
    end

    FinalResponse --> Trace[ExecutionTrace Telemetry & Summary]
    Trace --> UserOutput([User Output])

    subgraph Runtime Dispatch [Hardware-Aware Runtime Dispatch Layer]
        SLM --> Dispatcher[RuntimeDispatcher]
        Dispatcher --> HWDetect[Hardware & NPU Detector]
        HWDetect --> QNN{Snapdragon + QNN Available?}
        QNN -- Yes --> QNN_EP[QNNExecutionProvider (Hexagon NPU)]
        QNN -- No / Failure --> CPU_EP[CPUExecutionProvider (Fallback)]
    end
```

---

## 5. Model Selection & Rationale (Milestone 2)

For the core reasoning brain, we evaluated practical open-source SLM candidates:

| Candidate Model | Parameter Count | RAM Footprint | JSON / Tool Calling | License | Assessment for ExoCortex MVP |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen2.5-0.5B-Instruct** | **490M** | **~350 MB** | **Excellent** | **Apache 2.0** | **Selected Primary MVP Model**: Ultra-compact, sub-second CPU latency, zero memory pressure. |
| **Qwen2.5-1.5B-Instruct** | 1.54B | ~900 MB | Exceptional | Apache 2.0 | High-accuracy tier for complex multi-step reasoning. |
| **Phi-3.5-mini-instruct** | 3.82B | ~2.5 GB (INT4) | Very Good | MIT | Supported via ONNX provider for higher-capacity hosts. |
| **Llama-3.2-1B-Instruct** | 1.23B | ~800 MB | Good | Llama Community | Strong baseline; custom license constraints. |

---

## 6. Safe Windows OS & Desktop Automation (Milestone 3)

The SLM **never** executes arbitrary PowerShell, CMD, or shell strings. Every OS capability is encapsulated within a typed `BaseTool` registered in `ToolRegistry`.

```
exocortex/
    execution/
        __init__.py              # Execution engine exports
        executor.py              # Multi-step cognitive loop engine
        state.py                 # ExecutionState & ActionHistoryTracker
        trace.py                 # ExecutionTrace telemetry recorder
    runtime/
        __init__.py              # Runtime dispatch exports
        capabilities.py          # NPU and hardware capability models
        detector.py              # Hardware, Snapdragon, & provider detection
        provider.py              # Deterministic provider resolution dispatcher
    tools/
        windows/
            __init__.py          # Windows tool suite exports
            launch_application.py# Allowlist application launcher
            open_url.py          # Scheme-validated URL browser opener
            filesystem.py        # Workspace-sandboxed file tools
            processes.py         # Read-only process inspector
            system.py            # Host & hardware diagnostics
```

---

## 7. Snapdragon / Hardware Acceleration & Runtime Dispatch (Milestone 5)

ExoCortex is engineered with an intelligent **Hardware-Aware Runtime Dispatch Layer** (`exocortex.runtime`) that deterministically selects the optimal inference backend at runtime.

### Dispatch Rules:
1. **Snapdragon Native Execution**: If running on Qualcomm Snapdragon silicon (e.g. Snapdragon X Elite / X Plus) AND `QNNExecutionProvider` is installed, ONNX Runtime initializes inference on the **Qualcomm Hexagon NPU**.
2. **Safe CPU Fallback**: If running on non-ARM64 / Intel hardware, or if QNN provider initialization fails, the engine logs the failure reason and cleanly falls back to `CPUExecutionProvider`.
3. **No Fake NPU Reporting**: NPU acceleration is strictly reported as `ACTIVE` only when real on-device neural tensor operations are verified on Snapdragon hardware with QNN. On Intel/x64 systems, status clearly reports `NPU: UNAVAILABLE (CPU Fallback Active)`.

### Configuration:
Set via environment variable:
```bash
# Options: auto (default), cpu, qnn, dml
set EXOCORTEX_EXECUTION_PROVIDER=auto
```

---

## 8. Implementation Status: Current vs. Snapdragon Roadmap

> [!IMPORTANT]
> **Hardware Status Note**: The development host is currently running on an **Intel x64 CPU** using `CPUExecutionProvider`. Snapdragon NPU / Hexagon acceleration via `QNNExecutionProvider` is architecturally supported, tested with fallback isolation, and ready for deployment on Qualcomm Copilot+ PCs.

| Feature / Subsystem | Current Status (Milestone 5) | Future Target (Snapdragon NPU) |
| :--- | :--- | :--- |
| **SLM Inference Engine** | ✅ **Active on CPU** via `LocalSLMProvider` (Real ONNX Qwen2.5-0.5B INT8) | 🚀 **Qualcomm Hexagon NPU** via `QNNExecutionProvider` |
| **Runtime Dispatch** | ✅ **Active & Tested**: Hardware-aware `RuntimeDispatcher` with automatic fallback | 🚀 Qualcomm AI Hub INT4 compilation & dynamic NPU context |
| **Agentic Cognitive Loop** | ✅ **Active**: Safe multi-step `AgentExecutor`, loop detection, observation feedback | 🚀 Proactive goal decomposition & background triggers |
| **Tool Calling Pipeline** | ✅ **Active & Tested**: Strict JSON schema, dynamic tool prompting, validation & repair | 🚀 Expanded Windows UI Automation & Vision Tools |
| **Windows Desktop Tools** | ✅ **Active**: App launcher, URL opener, workspace filesystem, process inspector | 🚀 Direct accessibility tree inspection & UI automation |
| **Cloud Dependency** | ❌ **0% (Zero Cloud AI APIs)** | ❌ **0% (100% On-Device)** |
| **Telemetry & Metrics** | ✅ **Active**: Step latency, tokens/sec, RAM, ExecutionTrace JSON export | 🚀 NPU Power Draw (Watts) & Thermal Efficiency |

---

## 9. Milestone Status Tracker

- [x] **Milestone 1 — Foundation**: Modular agent skeleton, configuration manager, hardware discovery, tool sandbox, status probe, Windows CLI.
- [x] **Milestone 2 — Local SLM Brain & Structured Tool Calling**:
  - [x] Evaluation and selection of `Qwen2.5-0.5B-Instruct` as primary MVP model.
  - [x] Real on-device ONNX neural generator with KV caching.
  - [x] Strict structured response format (`AgentDecision`, `StructuredStep`).
  - [x] JSON extractor, validator, and malformed-output repair engine.
  - [x] Dynamic tool-aware prompt generator injecting registered tool schemas.
  - [x] Local inference benchmarking suite (`exocortex benchmark`).
- [x] **Milestone 3 — Safe Windows Desktop & OS Automation**:
  - [x] Safe application launcher (`launch_application`) with strict allowlist.
  - [x] Validated URL opener (`open_url`) supporting `http`/`https`.
  - [x] Workspace-sandboxed filesystem tools (`list_directory`, `read_text_file`, `create_directory`).
  - [x] Read-only Windows process inspector (`list_processes`).
  - [x] Unified tool registry exposing all 9 tools to the SLM.
- [x] **Milestone 4 — Safe Agentic Execution Loop**:
  - [x] Safe multi-step execution loop (`AgentExecutor`) with observation feedback.
  - [x] Hard limit enforcement (`MAX_STEPS = 8`).
  - [x] Loop detection (`MAX_IDENTICAL_ACTIONS = 2`).
  - [x] Context truncation bounding tool observations to 2,000 characters.
  - [x] Structured `ExecutionTrace` with step breakdown and JSON export.
- [x] **Milestone 5 — Snapdragon Hardware Acceleration & Runtime Dispatch**:
  - [x] Created `exocortex.runtime` package (`detector.py`, `provider.py`, `capabilities.py`).
  - [x] Hardware-aware provider resolution (`RuntimeDispatcher`) supporting `auto`, `cpu`, `qnn`, `dml`.
  - [x] Safe exception catching and fallback to `CPUExecutionProvider` upon QNN initialization failure.
  - [x] Accurate NPU status reporting without simulated / fake acceleration.
  - [x] Extended `exocortex status`, `exocortex hardware`, and `exocortex benchmark` with runtime telemetry.
  - [x] 88 passing automated unit tests across 12 test modules.

---

## 10. Quickstart & CLI Commands

### Prerequisites
- Python 3.11 or 3.12
- Windows 11 (ARM64 Snapdragon or x64)

### Running System Status & Hardware Diagnostics
```powershell
# System health, SLM engine, and runtime dispatch status
python -m exocortex.cli status

# Hardware profile and Qualcomm Snapdragon NPU readiness probe
python -m exocortex.cli hardware

# Inspect all registered tools and security permissions
python -m exocortex.cli tools

# Run local SLM inference benchmark
python -m exocortex.cli benchmark
```

### Running Autonomous Multi-Step Desktop Tasks
```powershell
# Run a task with multi-step execution trace
python -m exocortex.cli run --prompt "Open Notepad" --trace

# Run a task and output JSON trace
python -m exocortex.cli run --prompt "Show me my running processes" --json

# Inspect files in the safe workspace (~/.exocortex/workspace)
python -m exocortex.cli run --prompt "List the files in my ExoCortex workspace"

# Create a directory inside workspace
python -m exocortex.cli run --prompt "Create a folder called research inside my workspace"
```

### Running the Test Suite
```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```


