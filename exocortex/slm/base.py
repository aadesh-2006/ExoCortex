"""
Base interfaces and data structures for Small Language Model (SLM) inference.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolCallRequest:
    """Structure representing a requested tool execution from the SLM."""
    tool_name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    call_id: Optional[str] = None


@dataclass
class SLMMessage:
    """A single message in an SLM conversation."""
    role: Role
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCallRequest]] = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "role": self.role,
            "content": self.content,
        }
        if self.name:
            data["name"] = self.name
        if self.tool_calls:
            data["tool_calls"] = [
                {
                    "name": tc.tool_name,
                    "arguments": tc.arguments,
                    "id": tc.call_id,
                }
                for tc in self.tool_calls
            ]
        return data


@dataclass
class SLMResponse:
    """Response returned from an SLM inference call."""
    content: str
    tool_calls: List[ToolCallRequest] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    model_name: str = ""
    provider_name: str = ""
    raw_response: Optional[Any] = None

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class SLMProvider(ABC):
    """Abstract interface for local Small Language Model inference providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
        **kwargs: Any,
    ) -> SLMResponse:
        """Generate text completion from a prompt string."""
        pass

    @abstractmethod
    def chat(
        self,
        messages: List[SLMMessage],
        temperature: float = 0.2,
        max_tokens: int = 512,
        **kwargs: Any,
    ) -> SLMResponse:
        """Conduct a multi-turn chat completion with message history."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider and its required models/dependencies are available."""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return metadata about the current model and execution provider."""
        pass

    def health_check(self) -> Dict[str, Any]:
        """Run a lightweight health check on the SLM provider."""
        available = self.is_available()
        return {
            "provider": self.__class__.__name__,
            "available": available,
            "model_info": self.get_model_info(),
        }
