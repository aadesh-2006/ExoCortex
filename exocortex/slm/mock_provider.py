"""
Mock / Rule-Based SLM provider for unit tests and zero-dependency verification.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

from exocortex.slm.base import (
    Role,
    SLMMessage,
    SLMProvider,
    SLMResponse,
    ToolCallRequest,
)


class MockSLMProvider(SLMProvider):
    """
    Deterministic SLM Provider for offline testing, CI, and local bootstrapping.
    
    Can parse basic natural language intentions to simulate SLM tool-calling behaviors.
    """

    def __init__(
        self,
        model_name: str = "mock-slm-phi-3.5",
        canned_responses: Optional[Dict[str, str]] = None,
    ) -> None:
        self.model_name = model_name
        self.canned_responses = canned_responses or {}

    def is_available(self) -> bool:
        return True

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "provider": "MockSLMProvider",
            "execution_provider": "Emulated / In-Memory",
            "is_local": True,
            "is_mock": True,
        }

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
        **kwargs: Any,
    ) -> SLMResponse:
        start_time = time.perf_counter()
        
        # Check custom canned responses
        for key, resp in self.canned_responses.items():
            if key.lower() in prompt.lower():
                latency = (time.perf_counter() - start_time) * 1000
                return SLMResponse(
                    content=resp,
                    prompt_tokens=len(prompt.split()),
                    completion_tokens=len(resp.split()),
                    latency_ms=latency,
                    model_name=self.model_name,
                    provider_name="MockSLMProvider",
                )

        # Fallback intelligent rule-based generation
        content, tool_calls = self._simulate_reasoning(prompt)
        latency = (time.perf_counter() - start_time) * 1000
        return SLMResponse(
            content=content,
            tool_calls=tool_calls,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(content.split()),
            latency_ms=latency,
            model_name=self.model_name,
            provider_name="MockSLMProvider",
        )

    def chat(
        self,
        messages: List[SLMMessage],
        temperature: float = 0.2,
        max_tokens: int = 512,
        **kwargs: Any,
    ) -> SLMResponse:
        # Extract last user message or join context
        user_messages = [m.content for m in messages if m.role == "user"]
        prompt = user_messages[-1] if user_messages else (messages[-1].content if messages else "")
        return self.generate(prompt=prompt, temperature=temperature, max_tokens=max_tokens, **kwargs)

    def _simulate_reasoning(self, text: str) -> tuple[str, List[ToolCallRequest]]:
        """Simulate realistic SLM intent parsing, multi-step agent decisions, and tool feedback."""
        lower = text.lower()
        tool_calls: List[ToolCallRequest] = []

        # Handle multi-step execution loop observations
        has_history = "previous actions and observations" in lower

        # Check multi-step task: "Open Notepad and then tell me what processes..." or "Open Notepad and then list..."
        is_notepad_and_process = ("notepad" in lower and ("process" in lower or "running" in lower))
        is_notepad_and_list = ("notepad" in lower and ("list" in lower or "workspace" in lower or "file" in lower))

        if has_history:
            if is_notepad_and_process:
                if "list_processes" in lower:
                    doc = {
                        "thought_summary": "Both Notepad launch and process listing completed.",
                        "intent": "multi_step_completion",
                        "steps": [],
                        "requires_confirmation": False,
                        "direct_response": "Notepad was launched and the list of active running processes was retrieved successfully.",
                    }
                    return json.dumps(doc), []
                elif "launch_application" in lower:
                    doc = {
                        "thought_summary": "Notepad launched. Now inspecting running processes.",
                        "intent": "process_inspection",
                        "steps": [{"tool": "list_processes", "arguments": {}}],
                        "requires_confirmation": False,
                        "direct_response": None,
                    }
                    tool_calls.append(ToolCallRequest(tool_name="list_processes", arguments={}, call_id="call_proc_02"))
                    return json.dumps(doc), tool_calls

            if is_notepad_and_list:
                if "list_directory" in lower:
                    doc = {
                        "thought_summary": "Both Notepad launch and directory listing completed.",
                        "intent": "multi_step_completion",
                        "steps": [],
                        "requires_confirmation": False,
                        "direct_response": "Notepad was launched and workspace files were listed successfully.",
                    }
                    return json.dumps(doc), []
                elif "launch_application" in lower:
                    doc = {
                        "thought_summary": "Notepad launched. Now inspecting workspace files.",
                        "intent": "filesystem_inspection",
                        "steps": [{"tool": "list_directory", "arguments": {"path": ""}}],
                        "requires_confirmation": False,
                        "direct_response": None,
                    }
                    tool_calls.append(ToolCallRequest(tool_name="list_directory", arguments={"path": ""}, call_id="call_ls_02"))
                    return json.dumps(doc), tool_calls

            # For single-step tasks that have already completed an action in history
            if "launch_application" in lower or "open_url" in lower or "list_directory" in lower or "read_text_file" in lower or "create_directory" in lower or "list_processes" in lower or "system_info" in lower or "health_check" in lower:
                doc = {
                    "thought_summary": "Previous action completed successfully. Task finished.",
                    "intent": "task_completion",
                    "steps": [],
                    "requires_confirmation": False,
                    "direct_response": "Task completed successfully based on tool observation.",
                }
                return json.dumps(doc), []

        if "health" in lower or "status" in lower or "diagnostics" in lower:
            doc = {
                "thought_summary": "Checking system health status and component readiness.",
                "intent": "system_health",
                "steps": [{"tool": "health_check", "arguments": {}}],
                "requires_confirmation": False,
                "direct_response": None,
            }
            tool_calls.append(
                ToolCallRequest(tool_name="health_check", arguments={}, call_id="call_health_01")
            )
            return json.dumps(doc), tool_calls

        if "system" in lower or "hardware" in lower or "specs" in lower or "cpu" in lower:
            doc = {
                "thought_summary": "Inspecting host hardware and system specifications.",
                "intent": "hardware_inspection",
                "steps": [{"tool": "system_info", "arguments": {"detail_level": "full"}}],
                "requires_confirmation": False,
                "direct_response": None,
            }
            tool_calls.append(
                ToolCallRequest(
                    tool_name="system_info",
                    arguments={"detail_level": "full"},
                    call_id="call_sysinfo_01",
                )
            )
            return json.dumps(doc), tool_calls

        if "echo" in lower or "say" in lower:
            match = re.search(r'(?:echo|say)\s+["\']?([^"\']+)["\']?', text, re.IGNORECASE)
            msg = match.group(1) if match else text
            doc = {
                "thought_summary": f"Echoing message: {msg}",
                "intent": "echo_message",
                "steps": [{"tool": "echo", "arguments": {"message": msg}}],
                "requires_confirmation": False,
                "direct_response": None,
            }
            tool_calls.append(
                ToolCallRequest(
                    tool_name="echo",
                    arguments={"message": msg},
                    call_id="call_echo_01",
                )
            )
            return json.dumps(doc), tool_calls

        if "notepad" in lower or "calc" in lower or "paint" in lower or "explorer" in lower or "launch" in lower or "open app" in lower:
            app = "notepad"
            if "calc" in lower:
                app = "calculator"
            elif "paint" in lower:
                app = "paint"
            elif "explorer" in lower:
                app = "explorer"
            doc = {
                "thought_summary": f"User wants to launch {app}.",
                "intent": "application_launch",
                "steps": [{"tool": "launch_application", "arguments": {"application": app}}],
                "requires_confirmation": False,
                "direct_response": None,
            }
            tool_calls.append(ToolCallRequest(tool_name="launch_application", arguments={"application": app}, call_id="call_launch_01"))
            return json.dumps(doc), tool_calls

        if "browser" in lower or "http://" in lower or "https://" in lower or "url" in lower or "website" in lower or "google" in lower:
            url_match = re.search(r'https?://[^\s"\']+', text)
            target_url = url_match.group(0) if url_match else "https://www.google.com"
            doc = {
                "thought_summary": f"User wants to open web address: {target_url}.",
                "intent": "web_navigation",
                "steps": [{"tool": "open_url", "arguments": {"url": target_url}}],
                "requires_confirmation": False,
                "direct_response": None,
            }
            tool_calls.append(ToolCallRequest(tool_name="open_url", arguments={"url": target_url}, call_id="call_url_01"))
            return json.dumps(doc), tool_calls

        if "create folder" in lower or "create directory" in lower or "make folder" in lower or "mkdir" in lower:
            match = re.search(r'(?:folder|directory)\s+(?:called\s+|named\s+)?["\']?([^"\']+)["\']?', text, re.IGNORECASE)
            folder_name = match.group(1).strip() if match else "test"
            doc = {
                "thought_summary": f"Creating directory '{folder_name}' in workspace.",
                "intent": "filesystem_create",
                "steps": [{"tool": "create_directory", "arguments": {"path": folder_name}}],
                "requires_confirmation": False,
                "direct_response": None,
            }
            tool_calls.append(ToolCallRequest(tool_name="create_directory", arguments={"path": folder_name}, call_id="call_mkdir_01"))
            return json.dumps(doc), tool_calls

        if "read" in lower and ("file" in lower or ".txt" in lower or "notes" in lower):
            match = re.search(r'read\s+(?:file\s+)?["\']?([a-zA-Z0-9_\-\./\\]+\.[a-zA-Z0-9]+)["\']?', text, re.IGNORECASE)
            file_name = match.group(1) if match else "notes.txt"
            doc = {
                "thought_summary": f"Reading file '{file_name}' from workspace.",
                "intent": "filesystem_read",
                "steps": [{"tool": "read_text_file", "arguments": {"path": file_name}}],
                "requires_confirmation": False,
                "direct_response": None,
            }
            tool_calls.append(ToolCallRequest(tool_name="read_text_file", arguments={"path": file_name}, call_id="call_read_01"))
            return json.dumps(doc), tool_calls

        if "list" in lower and ("file" in lower or "directory" in lower or "workspace" in lower or "folder" in lower):
            doc = {
                "thought_summary": "Listing workspace files and directories.",
                "intent": "filesystem_inspection",
                "steps": [{"tool": "list_directory", "arguments": {"path": ""}}],
                "requires_confirmation": False,
                "direct_response": None,
            }
            tool_calls.append(ToolCallRequest(tool_name="list_directory", arguments={"path": ""}, call_id="call_ls_01"))
            return json.dumps(doc), tool_calls

        if "process" in lower or "tasks" in lower or "running programs" in lower:
            doc = {
                "thought_summary": "Inspecting running processes.",
                "intent": "process_inspection",
                "steps": [{"tool": "list_processes", "arguments": {}}],
                "requires_confirmation": False,
                "direct_response": None,
            }
            tool_calls.append(ToolCallRequest(tool_name="list_processes", arguments={}, call_id="call_proc_01"))
            return json.dumps(doc), tool_calls

        if "hello" in lower or "who are you" in lower or "hi " in lower:
            doc = {
                "thought_summary": "User greeting.",
                "intent": "general_greeting",
                "steps": [],
                "requires_confirmation": False,
                "direct_response": "Hello! I am ExoCortex, your local-first autonomous PC agent for Windows.",
            }
            return json.dumps(doc), []

        doc = {
            "thought_summary": f"General request: {text}",
            "intent": "general_inquiry",
            "steps": [],
            "requires_confirmation": False,
            "direct_response": f"Understood request: '{text}'. ExoCortex reasoning complete.",
        }
        return json.dumps(doc), []
