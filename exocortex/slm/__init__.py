"""
SLM (Small Language Model) inference abstractions for ExoCortex.
"""

from exocortex.slm.base import (
    SLMMessage,
    SLMResponse,
    SLMProvider,
    Role,
)
from exocortex.slm.mock_provider import MockSLMProvider
from exocortex.slm.onnx_provider import ONNXRuntimeSLMProvider
from exocortex.slm.local_provider import LocalServerSLMProvider
from exocortex.slm.factory import get_slm_provider

__all__ = [
    "SLMMessage",
    "SLMResponse",
    "SLMProvider",
    "Role",
    "MockSLMProvider",
    "ONNXRuntimeSLMProvider",
    "LocalServerSLMProvider",
    "get_slm_provider",
]
