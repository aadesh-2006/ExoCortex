# ExoCortex 🧠⚡
**Local-First JARVIS-Like Autonomous Personal Computer Agent for Windows**  
*Built for the Qualcomm Snapdragon AI Lab Build & Present Challenge*

[![Platform](https://img.shields.io/badge/Platform-Windows%2011%20%7C%20ARM64%20%7C%20x64-blue.svg)](https://microsoft.com/windows)
[![Qualcomm](https://img.shields.io/badge/Hardware-Snapdragon%20X%20Elite%20%7C%20NPU-FF6600.svg)](https://www.qualcomm.com/products/mobile/snapdragon)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Milestone%203-Completed-brightgreen.svg)]()

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

## 4. Architecture & Cognitive Loop

ExoCortex is designed with a decoupled, modular cognitive architecture:

```mermaid
flowchart TD
    User([User Natural Language Prompt]) --> CLI[ExoCortex CLI / Interface]
    CLI --> AgentCore[Agent Cognitive Core]
    
    subgraph Cognitive Loop [Agent Cognitive Reasoning Pipeline]
        AgentCore --> SLM[Local SLM Brain (Qwen2.5-0.5B / ONNX)]
        SLM --> StructuredDec[Strict Structured Decision (AgentDecision Schema)]
        StructuredDec --> Validator[Schema & Tool Validator]
        Validator --> ToolReg[Tool Selection & Permission Engine]
        ToolReg --> WinTools[Safe Windows OS Tool Suite]
        WinTools --> WinOS[Windows OS / Subsystems]
        WinOS --> Observation[Observe Result & Telemetry]
        Observation --> SLM
        Observation --> TaskComplete([Task Completion])
    end

    subgraph Hardware Acceleration [Hardware Acceleration Layer]
        SLM --> HWDetect[Hardware Profile & NPU Detector]
        HWDetect --> QNN[ONNX Runtime QNN (Qualcomm Hexagon NPU - Planned)]
        HWDetect --> DML[DirectML / GPU Neural Accelerator]
        HWDetect --> CPUFallback[CPU Execution Provider (Active Dev Fallback)]
    end
```

### Core Pipeline Flow:
$$\text{User Prompt} \longrightarrow \text{Local SLM Brain} \longrightarrow \text{Structured Decision (JSON)} \longrightarrow \text{Tool Validation} \longrightarrow \text{Execution} \longrightarrow \text{Observation} \longrightarrow \text{Task Completion}$$

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
    tools/
        windows/
            __init__.py              # Windows tool suite exports
            launch_application.py    # Allowlist application launcher
            open_url.py              # Scheme-validated URL browser opener
            filesystem.py            # Workspace-sandboxed file tools
            processes.py             # Read-only process inspector
            system.py                # Host & hardware diagnostics
```

### Registered Windows Tools:
1. **`launch_application`**: Launches known GUI desktop applications from an explicit allowlist (`notepad`, `calculator`, `paint`, `explorer`). Rejects arbitrary executable paths or command strings.
2. **`open_url`**: Opens web addresses in the user's default browser. Strictly restricted to `http://` and `https://` protocols (rejects `file://`, `javascript:`, `data:`, shell URIs).
3. **`list_directory`**: Lists files and subfolders within the safe workspace sandbox (`~/.exocortex/workspace`). Traversal (`..`) is blocked.
4. **`read_text_file`**: Reads text files strictly inside `~/.exocortex/workspace`. Rejects out-of-boundary paths and files exceeding 1 MB.
5. **`create_directory`**: Creates directories safely inside `~/.exocortex/workspace`.
6. **`list_processes`**: Read-only telemetry of active Windows processes (PID, process name, memory RSS). Process termination is strictly prohibited.
7. **`system_info`**: Inspects host hardware, CPU/RAM, and Snapdragon NPU accelerator readiness.
8. **`health_check`**: Comprehensive diagnostic probe of all agent components.
9. **`echo`**: Verification tool for parameter passing and pipeline validation.

### Strict Security Boundaries:
- ❌ **Zero Arbitrary Shell Execution**: No `cmd.exe /c`, `powershell -c`, or shell expansion (`shell=False` everywhere).
- ❌ **Zero Destructive File Operations**: No file deletion, rename, move, or formatting in this milestone.
- ❌ **Zero Process Termination**: Process inspection is 100% read-only.
- ❌ **Zero Unrestricted Filesystem Access**: Sandboxed to `~/.exocortex/workspace`. All paths are resolved and verified against the workspace root.
- ❌ **Zero Dangerous URL Schemes**: `file://`, `javascript:`, `data:`, and local paths are rejected.

---

## 7. Implementation Status: Current vs. Snapdragon Roadmap

> [!IMPORTANT]
> **Hardware Status Note**: The development host is currently running on an **Intel x64 CPU** using `CPUExecutionProvider`. Snapdragon NPU / Hexagon acceleration via `QNNExecutionProvider` is architecturally supported and ready for Qualcomm Copilot+ PCs.

| Feature / Subsystem | Current Status (Milestone 3) | Future Target (Snapdragon NPU) |
| :--- | :--- | :--- |
| **SLM Inference Engine** | ✅ **Active on CPU / DirectML** via `LocalSLMProvider` & `ONNXRuntimeSLMProvider` | 🚀 **Qualcomm Hexagon NPU** via `QNNExecutionProvider` |
| **Tool Calling Pipeline** | ✅ **Active & Tested**: Strict JSON schema, dynamic tool prompting, validation & repair | 🚀 Expanded Windows UI Automation & Vision Tools |
| **Windows Desktop Tools** | ✅ **Active**: App launcher, URL opener, workspace filesystem, process inspector | 🚀 Direct accessibility tree inspection & UI automation |
| **Model Optimization** | ✅ **ONNX Runtime 1.18.0** local graph execution & quantization hooks | 🚀 **Qualcomm AI Hub** compiled INT4/W4A16 weights |
| **Cloud Dependency** | ❌ **0% (Zero Cloud AI APIs)** | ❌ **0% (100% On-Device)** |
| **Telemetry & Metrics** | ✅ **Active**: Model load time, latency (ms), tokens/sec, RAM usage | 🚀 NPU Power Draw (Watts) & Thermal Efficiency |

---

## 8. Milestone Status Tracker

- [x] **Milestone 1 — Foundation**: Modular agent skeleton, configuration manager, hardware discovery, tool sandbox, status probe, Windows CLI.
- [x] **Milestone 2 — Local SLM Brain & Structured Tool Calling**:
  - [x] Evaluation and selection of `Qwen2.5-0.5B-Instruct` as primary MVP model.
  - [x] Real on-device ONNX neural generator with KV caching.
  - [x] Strict structured response format (`AgentDecision`, `StructuredStep`).
  - [x] JSON extractor, validator, and malformed-output repair engine.
  - [x] Dynamic tool-aware prompt generator injecting registered tool schemas.
  - [x] Local inference benchmarking suite (`exocortex benchmark`).
  - [x] 100% passing test suite.
- [x] **Milestone 3 — Safe Windows Desktop & OS Automation**:
  - [x] Safe application launcher (`launch_application`) with strict allowlist.
  - [x] Validated URL opener (`open_url`) supporting `http`/`https`.
  - [x] Workspace-sandboxed filesystem tools (`list_directory`, `read_text_file`, `create_directory`).
  - [x] Read-only Windows process inspector (`list_processes`).
  - [x] Unified tool registry exposing all 9 tools to the SLM.
  - [x] 61 comprehensive automated unit tests.
- [ ] **Milestone 4 — Qualcomm AI Hub Integration & Snapdragon NPU Benchmarking**: QNN Execution Provider compilation and on-device HP Snapdragon benchmarks.

---

## 9. Quickstart & CLI Commands

### Prerequisites
- Python 3.11 or 3.12
- Windows 11 (ARM64 Snapdragon or x64)

### Running System Status & Tools Inspection
```powershell
# System health and hardware status
python -m exocortex.cli status

# Inspect all registered tools and security permissions
python -m exocortex.cli tools

# Run local SLM inference benchmark
python -m exocortex.cli benchmark
```

### Running Autonomous Desktop Tasks
```powershell
# Launch an allowlisted desktop application
python -m exocortex.cli run --prompt "Open Notepad"

# Open a web address in default browser
python -m exocortex.cli run --prompt "Open Google in my browser"

# Inspect files in the safe workspace (~/.exocortex/workspace)
python -m exocortex.cli run --prompt "List the files in my ExoCortex workspace"

# Create a directory inside workspace
python -m exocortex.cli run --prompt "Create a folder called research inside my workspace"

# Inspect active processes
python -m exocortex.cli run --prompt "Show me my running processes"

# Inspect system hardware and memory
python -m exocortex.cli run --prompt "Check my system information"
```

### Running the Test Suite
```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```
