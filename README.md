# ExoCortex 🧠⚡
**Local-First JARVIS-Like Autonomous Personal Computer Agent for Windows**  
*Built for the Qualcomm Snapdragon AI Lab Build & Present Challenge*

[![Platform](https://img.shields.io/badge/Platform-Windows%2011%20%7C%20ARM64%20%7C%20x64-blue.svg)](https://microsoft.com/windows)
[![Qualcomm](https://img.shields.io/badge/Hardware-Snapdragon%20X%20Elite%20%7C%20NPU-FF6600.svg)](https://www.qualcomm.com/products/mobile/snapdragon)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Milestone%202-Completed-brightgreen.svg)]()

---

## 1. Project Overview

**ExoCortex** is an autonomous, local-first personal computer assistant engineered specifically for **Windows** and optimized for **Qualcomm Snapdragon-powered Copilot+ PCs** (such as HP Snapdragon laptops). 

Unlike cloud-dependent conversational bots, ExoCortex functions as an intelligent cognitive layer residing directly on the user's computer. It receives natural-language user intentions, formulates structured multi-step execution plans, interacts with the Windows OS, observes the resulting state, and autonomously drives tasks to completion—all while keeping user data 100% private and on-device.

---

## 2. Problem Statement

Modern AI assistants and computer-use tools suffer from critical drawbacks:
1. **Cloud Latency & Privacy Risks**: Transmitting screen pixels, file hierarchies, and private user communications to external cloud servers introduces severe latency, high operational costs, and catastrophic privacy risks.
2. **Heavyweight LLMs & Battery Drain**: Traditional Large Language Models (70B+) require massive cloud server farms or heavy discrete GPUs that quickly drain laptop battery.
3. **Lack of Hardware-Aware Agent Architectures**: Existing agents treat hardware as a black box rather than leveraging dedicated on-device Neural Processing Units (NPUs) built into modern laptops.

---

## 3. Project Vision

ExoCortex is envisioned as a **JARVIS-like copilot for Windows**:
- **Natural Interaction**: Understands conversational, contextual instructions from the user.
- **Autonomous Multi-Step Execution**: Decomposes ambiguous goals into discrete tool actions and system interactions.
- **Continuous Perception & Adaptation**: Observes desktop feedback, terminal outputs, and application windows, iteratively reasoning until the user's objective is fulfilled.
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
        ToolReg --> Action[Execution Engine]
        Action --> WinOS[Windows OS Subsystems / UI Automation]
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

### Why `Qwen2.5-0.5B-Instruct` was Selected:
1. **Ultra-Low Resource Footprint**: At ~350MB, it runs entirely in physical RAM with zero disk swapping, leaving system memory free for user applications.
2. **Deterministic Structured JSON Adherence**: Excellent instruction-following on strict schema constraints, avoiding markdown hallucinations.
3. **Hardware & Qualcomm Compatibility**: Supported by Qualcomm AI Hub compilation tools and ONNX Runtime execution providers.
4. **Fast Local CPU Fallback**: Generates tokens at high speed on standard developer laptops without requiring discrete Nvidia GPUs or cloud APIs.
5. **Permissive Licensing**: Apache 2.0 allows unrestricted research, benchmarking, and deployment.

---

## 6. Structured Tool-Calling Architecture

The local SLM does not produce arbitrary conversational text when performing computer tasks. It emits strict JSON conforming to the `AgentDecision` schema:

```json
{
  "thought_summary": "Short 1-2 sentence action rationale (no hidden chain-of-thought)",
  "intent": "hardware_inspection",
  "steps": [
    {
      "tool": "system_info",
      "arguments": {
        "detail_level": "full"
      }
    }
  ],
  "requires_confirmation": false,
  "direct_response": null
}
```

### Key Safety & Parsing Features:
- **No Hidden Chain-of-Thought**: SLM generates only a concise, user-auditable `thought_summary`.
- **Dynamic Tool-Aware Prompting**: System prompt dynamically injects schemas, parameter types, and permission levels of all tools currently registered in `ToolRegistry`.
- **Robust JSON Extraction & Repair**: Extracts JSON from codeblocks or raw text; heuristically repairs trailing commas and missing braces.
- **Unknown Tool Rejection**: Strictly rejects hallucinated or unregistered tool requests before execution.
- **Permission Elevation**: Automatically flags `requires_confirmation = true` whenever a tool requires user approval (`CONFIRMATION_REQUIRED` or `RESTRICTED`).

---

## 7. Implementation Status: Current vs. Snapdragon Roadmap

> [!IMPORTANT]
> **Strict Verification Guarantee**: We explicitly distinguish what is currently running on the development machine from what is architected for future Snapdragon hardware.

| Feature / Subsystem | Current Status (Milestone 2) | Future Target (Snapdragon NPU) |
| :--- | :--- | :--- |
| **SLM Inference Engine** | ✅ **Active on CPU / DirectML** via `LocalSLMProvider` & `ONNXRuntimeSLMProvider` | 🚀 **Qualcomm Hexagon NPU** via `QNNExecutionProvider` |
| **Tool Calling Pipeline** | ✅ **Active & Tested**: Strict JSON schema, dynamic tool prompting, validation & repair | 🚀 Expanded Windows UI Automation & Vision Tools |
| **Model Optimization** | ✅ **ONNX Runtime 1.18.0** local graph execution & quantization hooks | 🚀 **Qualcomm AI Hub** compiled INT4/W4A16 weights |
| **Cloud Dependency** | ❌ **0% (Zero Cloud AI APIs)** | ❌ **0% (100% On-Device)** |
| **Telemetry & Metrics** | ✅ **Active**: Model load time, latency (ms), tokens/sec, RAM usage | 🚀 NPU Power Draw (Watts) & Thermal Efficiency |

---

## 8. Milestone Status Tracker

- [x] **Milestone 1 — Foundation**: Modular agent skeleton, configuration manager, hardware discovery, tool sandbox, status probe, Windows CLI.
- [x] **Milestone 2 — Local SLM Brain & Structured Tool Calling**:
  - [x] Evaluation and selection of `Qwen2.5-0.5B-Instruct` as primary MVP model.
  - [x] Local SLM provider (`LocalSLMProvider`) with zero cloud API dependencies.
  - [x] Strict structured response format (`AgentDecision`, `StructuredStep`).
  - [x] JSON extractor, validator, and malformed-output repair engine.
  - [x] Dynamic tool-aware prompt generator injecting registered tool schemas.
  - [x] Local inference benchmarking suite (`exocortex benchmark`).
  - [x] 100% passing test suite (31 automated unit tests).
- [ ] **Milestone 3 — Windows OS Desktop & File System Automation**: Windows UI Automation, safe file operations, application launcher.
- [ ] **Milestone 4 — Qualcomm AI Hub Integration & Snapdragon NPU Benchmarking**: QNN Execution Provider compilation and on-device HP Snapdragon benchmarks.

---

## 9. Quickstart & CLI Commands

### Prerequisites
- Python 3.11 or 3.12
- Windows 11 (ARM64 Snapdragon or x64)

### Running System Status Check
```powershell
python -m exocortex.cli status
```

### Running Local SLM Inference Benchmarks
```powershell
python -m exocortex.cli benchmark
```

### Running Autonomous Tool-Calling Agent
```powershell
# Tool-calling prompt (triggers system_info tool)
python -m exocortex.cli run --prompt "Check my system information"

# Health diagnostics prompt (triggers health_check tool)
python -m exocortex.cli run --prompt "Run diagnostics on system health"

# Conversational prompt (no tool required)
python -m exocortex.cli run --prompt "Hello who are you?"

# Sensitive action prompt (triggers requires_confirmation)
python -m exocortex.cli run --prompt "Delete all files in C drive"
```

### Running the Test Suite
```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```
