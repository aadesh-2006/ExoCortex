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
        """Simulate realistic SLM intent parsing and tool invocation."""
        lower = text.lower()
        tool_calls: List[ToolCallRequest] = []

        if "health" in lower or "status" in lower or "diagnostics" in lower:
            tool_calls.append(
                ToolCallRequest(
                    tool_name="health_check",
                    arguments={},
                    call_id="call_health_01",
                )
            )
            return (
                "I will check the system health status and component readiness.",
                tool_calls,
            )

        if "system" in lower or "hardware" in lower or "specs" in lower or "cpu" in lower:
            tool_calls.append(
                ToolCallRequest(
                    tool_name="system_info",
                    arguments={"detail_level": "full"},
                    call_id="call_sysinfo_01",
                )
            )
            return (
                "I will inspect your host hardware and system specifications.",
                tool_calls,
            )

        if "echo" in lower or "say" in lower:
            match = re.search(r'(?:echo|say)\s+["\']?([^"\']+)["\']?', text, re.IGNORECASE)
            msg = match.group(1) if match else text
            tool_calls.append(
                ToolCallRequest(
                    tool_name="echo",
                    arguments={"message": msg},
                    call_id="call_echo_01",
                )
            )
            return (f"Echoing requested message: '{msg}'", tool_calls)

        return (
            f"Understood request: '{text}'. ExoCortex reasoning complete.",
            tool_calls,
        )
