"""
Local SLM server provider (Ollama / Local OpenAI-compatible server).

Allows connecting to local SLM instances running on Windows without cloud dependencies.
"""

from __future__ import annotations

import json
import math
import socket
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from exocortex.slm.base import (
    SLMMessage,
    SLMProvider,
    SLMResponse,
    ToolCallRequest,
)


class LocalServerSLMError(RuntimeError):
    """Base exception for errors during local SLM server communication."""
    pass


class LocalServerConnectionError(LocalServerSLMError, ConnectionError):
    """Raised when connection to the local SLM server fails."""
    pass


class LocalServerTimeoutError(LocalServerSLMError, TimeoutError):
    """Raised when local SLM server request times out."""
    pass


class LocalServerSLMProvider(SLMProvider):
    """
    SLM Provider communicating with a local server (e.g. Ollama, LM Studio, vLLM).
    
    100% local, no external internet connection required.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model_name: str = "phi-3.5-mini-instruct",
        api_key: str = "not-needed",
        timeout: float = 90.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        try:
            t = float(timeout)
            if not math.isfinite(t) or t <= 0:
                raise ValueError
            self.timeout = t
        except (ValueError, TypeError):
            raise ValueError(
                f"Invalid timeout '{timeout}': must be a positive number of seconds."
            )

    def is_available(self) -> bool:
        """Check if local endpoint is reachable."""
        try:
            url = f"{self.base_url}/models"
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {self.api_key}"})
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "provider": "LocalServerSLMProvider",
            "base_url": self.base_url,
            "timeout": self.timeout,
            "is_local": True,
        }

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
        **kwargs: Any,
    ) -> SLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return self._send_chat_request(messages, temperature, max_tokens)

    def chat(
        self,
        messages: List[SLMMessage],
        temperature: float = 0.2,
        max_tokens: int = 512,
        **kwargs: Any,
    ) -> SLMResponse:
        formatted = [m.to_dict() for m in messages]
        return self._send_chat_request(formatted, temperature, max_tokens)

    def _send_chat_request(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> SLMResponse:
        start_time = time.perf_counter()
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                res_json = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                err_body = e.read().decode("utf-8", errors="replace")
            except Exception:
                err_body = str(e)
            raise LocalServerSLMError(
                f"Local SLM server at {url} returned HTTP {e.code}: {err_body}"
            ) from e
        except urllib.error.URLError as e:
            is_timeout = isinstance(e.reason, (TimeoutError, socket.timeout)) or (
                isinstance(e.reason, str) and "timed out" in e.reason.lower()
            )
            if is_timeout:
                raise LocalServerTimeoutError(
                    f"Local SLM server at {url} timed out after {self.timeout}s: {e.reason}"
                ) from e
            raise LocalServerConnectionError(
                f"Failed to connect to local SLM server at {url}: {e.reason}"
            ) from e
        except (TimeoutError, socket.timeout) as e:
            raise LocalServerTimeoutError(
                f"Local SLM server at {url} timed out after {self.timeout}s: {e}"
            ) from e
        except Exception as e:
            raise LocalServerSLMError(
                f"Unexpected error communicating with local SLM server at {url}: {e}"
            ) from e

        latency = (time.perf_counter() - start_time) * 1000

        choices = res_json.get("choices", [])
        if not choices:
            err_detail = res_json.get("error", res_json)
            raise LocalServerSLMError(
                f"Local SLM server returned response without choices: {err_detail}"
            )

        choice = choices[0]
        msg = choice.get("message", {})
        content = msg.get("content", "") or ""

        # Parse tool calls if returned by local SLM
        tool_calls: List[ToolCallRequest] = []
        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {})
            args = {}
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except Exception:
                pass
            tool_calls.append(
                ToolCallRequest(
                    tool_name=fn.get("name", ""),
                    arguments=args,
                    call_id=tc.get("id"),
                )
            )

        usage = res_json.get("usage", {})
        return SLMResponse(
            content=content,
            tool_calls=tool_calls,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            latency_ms=latency,
            model_name=self.model_name,
            provider_name="LocalServerSLMProvider",
            raw_response=res_json,
        )
