# ExoCortex 🧠⚡
**Local-First JARVIS-Like Autonomous Personal Computer Agent for Windows**  
*Built for the Qualcomm Snapdragon AI Lab Build & Present Challenge*

[![Platform](https://img.shields.io/badge/Platform-Windows%2011%20%7C%20ARM64%20%7C%20x64-blue.svg)](https://microsoft.com/windows)
[![Qualcomm](https://img.shields.io/badge/Hardware-Snapdragon%20X%20Elite%20%7C%20NPU-FF6600.svg)](https://www.qualcomm.com/products/mobile/snapdragon)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Milestone%201-Completed-brightgreen.svg)]()

---

## 1. Project Overview

**ExoCortex** is an autonomous, local-first personal computer assistant engineered specifically for **Windows** and optimized for **Qualcomm Snapdragon-powered Copilot+ PCs** (such as HP Snapdragon laptops). 

Unlike cloud-dependent conversational bots, ExoCortex functions as an intelligent cognitive layer residing directly on the user's computer. It receives natural-language user intentions, formulates multi-step execution plans, interacts with the Windows OS, observes the resulting state, and autonomously drives tasks to completion—all while keeping user data 100% private and on-device.

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
        AgentCore --> SLM[SLM Inference Engine]
        SLM --> Intent[Intent Understanding & Entity Parsing]
        Intent --> Planner[Agent Planner & Task Decomposition]
        Planner --> ToolSel[Tool Selection & Permission Validator]
        ToolSel --> Action[Execution Engine]
        Action --> WinOS[Windows OS Subsystems / UI Automation]
        WinOS --> Observation[Observe Result & Telemetry]
        Observation --> SLM
        Observation --> TaskComplete([Task Completion])
    end

    subgraph Hardware Acceleration [Hardware Acceleration Layer]
        SLM --> HWDetect[Hardware Profile & NPU Detector]
        HWDetect --> QNN[ONNX Runtime QNN (Qualcomm Hexagon NPU)]
        HWDetect --> DML[DirectML / GPU Neural Accelerator]
        HWDetect --> LocalSLM[Local SLM Providers: ONNX / GGUF / Ollama / Mock]
    end
```

### Core Pipeline Flow:
$$\text{User Prompt} \longrightarrow \text{SLM Reasoning} \longrightarrow \text{Intent Understanding} \longrightarrow \text{Agent Planner} \longrightarrow \text{Tool Selection} \longrightarrow \text{Computer Interaction} \longrightarrow \text{Observation} \longrightarrow \text{Reason Again} \longrightarrow \text{Task Completion}$$

---

## 5. Core Principles

1. **Local-First AI**: All core reasoning, intent classification, and tool dispatching execute on-device. No cloud AI API keys are required.
2. **SLMs as Primary Reasoning Layer**: Powered by efficient Small Language Models (e.g., *Phi-3.5-mini*, *Llama-3.2-3B*, *Qwen2.5-Coder-3B/7B*).
3. **Data Privacy & User Sovereignty**: Personal files, notes, emails, and desktop interactions never leave the user's machine.
4. **First-Class Snapdragon NPU Strategy**: Designed from inception to leverage **Qualcomm AI Hub** and **ONNX Runtime QNN Execution Provider** on Snapdragon X Elite and Snapdragon X Plus architectures.
5. **Windows-Native Foundation**: Tailored specifically for Windows 11 subsystems, UI Automation, PowerShell, and native APIs.
6. **Safety & Permission Guardrails**: Explicit permission levels (`SAFE`, `CONFIRMATION_REQUIRED`, `RESTRICTED`) to prevent unauthorized destructive operations.

---

## 6. Snapdragon & NPU Strategy

ExoCortex avoids retrofitting hardware acceleration after the fact by treating NPU acceleration as a core architectural layer:

| Component | Target Technology | Purpose |
| :--- | :--- | :--- |
| **Model Optimization** | **Qualcomm AI Hub** | Quantized & compiled SLM models (INT4/W4A16) tailored for Snapdragon X Series. |
| **Inference Engine** | **ONNX Runtime (QNN EP)** | Direct execution on Qualcomm Hexagon NPU with zero CPU/GPU overhead. |
| **Windows Fallback** | **DirectML Execution Provider** | Accelerated execution on Windows AI-capable hardware. |
| **Host Architecture** | **Windows on ARM (ARM64)** | Native ARM64 compilation and execution on Snapdragon-powered HP PCs. |

---

## 7. Planned Capabilities

- [x] **Milestone 1**: Core foundation, hardware detector, SLM abstractions, cognitive loop skeleton, tool registry, diagnostics probe, CLI.
- [ ] **Milestone 2**: Local SLM integration (Phi-3.5-mini / Qwen2.5 via ONNX/GGUF), structured function-calling parser.
- [ ] **Milestone 3**: Windows Desktop & File System Tools (file search, app launcher, process manager, safe PowerShell dispatcher).
- [ ] **Milestone 4**: Qualcomm AI Hub model integration & Snapdragon NPU benchmarking (QNN Execution Provider).
- [ ] **Milestone 5**: Windows UI Automation & Vision-guided computer interaction.
- [ ] **Milestone 6**: Multi-turn proactive autonomous workflows (desktop organization, email drafts, calendar reminders).

---

## 8. Milestone 1 Status — Completed ✅

### Deliverables Accomplished:
- [x] Clean, modular project structure with packaging (`pyproject.toml`, `requirements.txt`).
- [x] Central configuration and environment handling (`exocortex/config.py`, `.env.example`).
- [x] Hardware discovery module detecting Snapdragon NPU, DirectML, QNN EP, and host telemetry (`exocortex/hardware.py`).
- [x] Decoupled SLM Provider interface with Mock, ONNX Runtime, and Local Server adapters (`exocortex/slm/`).
- [x] Safety-gated Tool Registry with permission levels and built-in foundation tools (`exocortex/tools/`).
- [x] Complete cognitive loop orchestrator (`exocortex/agent.py`, `exocortex/intent.py`, `exocortex/planner.py`).
- [x] Health check & system diagnostics probe (`exocortex/health.py`).
- [x] Full-featured CLI with status, hardware, and task execution commands (`exocortex/cli.py`).
- [x] 100% passing unit test suite (`tests/`).

---

## 9. Quickstart Guide

### Prerequisites
- Python 3.11 or 3.12
- Windows 11 (ARM64 Snapdragon or x64)

### Installation
```powershell
# Clone or navigate to the repository
cd C:\Users\Aadesh\.gemini\antigravity\scratch\ExoCortex

# Install dependencies
pip install -r requirements.txt
```

### Running System Health & Diagnostics
```powershell
python -m exocortex.cli status
```

### Inspecting Snapdragon NPU & Hardware Profile
```powershell
python -m exocortex.cli hardware
```

### Running an Autonomous Agent Task
```powershell
python -m exocortex.cli run --prompt "Check system status and hardware capabilities"
```

### Running Automated Tests
```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

---

## 10. Next Milestone (Milestone 2)

**Milestone 2 Target**: Local SLM Model Engine & Structured Function-Calling
1. Integrate local quantized SLM weights (*Phi-3.5-mini-instruct* / *Qwen2.5-Coder*).
2. Wire ONNX Runtime model inference pipeline with tokenizer and generation loop.
3. Implement JSON-schema based function calling parser for local SLM outputs.
4. Add model latency and token-per-second benchmarking utilities.
