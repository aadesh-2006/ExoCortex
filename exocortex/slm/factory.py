"""
Factory for creating and resolving SLM providers in ExoCortex.
"""

from __future__ import annotations

from typing import Optional

from exocortex.config import Settings, get_config
from exocortex.slm.base import SLMProvider
from exocortex.slm.local_provider import LocalServerSLMProvider
from exocortex.slm.mock_provider import MockSLMProvider
from exocortex.slm.onnx_provider import ONNXRuntimeSLMProvider


def get_slm_provider(
    provider_type: Optional[str] = None,
    config: Optional[Settings] = None,
    **kwargs,
) -> SLMProvider:
    """
    Resolve and initialize an SLM provider instance.
    
    Args:
        provider_type: Type of provider ('mock', 'onnx', 'local_server')
        config: Application settings instance
        **kwargs: Additional parameters passed to the provider
    """
    cfg = config or get_config()
    target_type = (provider_type or cfg.slm_provider).lower()

    if target_type == "mock":
        return MockSLMProvider(
            model_name=kwargs.get("model_name", cfg.model_name),
            canned_responses=kwargs.get("canned_responses"),
        )
    elif target_type == "onnx":
        return ONNXRuntimeSLMProvider(
            model_path=kwargs.get("model_path", cfg.model_path),
            model_name=kwargs.get("model_name", cfg.model_name),
            execution_provider=kwargs.get("hardware_target", cfg.hardware_target),
        )
    elif target_type in ("local_server", "ollama", "local"):
        return LocalServerSLMProvider(
            base_url=kwargs.get("local_server_url", cfg.local_server_url),
            model_name=kwargs.get("model_name", cfg.model_name),
            api_key=kwargs.get("local_server_api_key", cfg.local_server_api_key),
        )
    else:
        # Default fallback to mock provider
        return MockSLMProvider(model_name=cfg.model_name)
