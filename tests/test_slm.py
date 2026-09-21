"""
Unit tests for SLM provider interface, LocalSLMProvider, and factory.
"""

import unittest
from exocortex.config import Settings
from exocortex.slm.base import SLMMessage, SLMResponse
from exocortex.slm.factory import get_slm_provider
from exocortex.slm.local_model_engine import LocalSLMProvider
from exocortex.slm.local_provider import LocalServerSLMProvider
from exocortex.slm.mock_provider import MockSLMProvider
from exocortex.slm.onnx_provider import ONNXRuntimeSLMProvider


class TestSLMProviders(unittest.TestCase):
    def test_local_slm_provider_generate(self):
        provider = LocalSLMProvider(model_name="qwen2.5-0.5b-instruct")
        self.assertTrue(provider.is_available())
        
        resp = provider.generate("Check system hardware")
        self.assertIsInstance(resp, SLMResponse)
        self.assertTrue(len(resp.content) > 0)
        self.assertEqual(resp.provider_name, "LocalSLMProvider")
        self.assertIsNotNone(provider.last_metrics)
        self.assertTrue(provider.last_metrics.completion_tokens > 0)

    def test_mock_provider_generate(self):
        provider = MockSLMProvider()
        self.assertTrue(provider.is_available())
        
        resp = provider.generate("test prompt")
        self.assertIsInstance(resp, SLMResponse)
        self.assertTrue(len(resp.content) > 0)
        self.assertEqual(resp.provider_name, "MockSLMProvider")

    def test_factory_local_slm(self):
        cfg = Settings(slm_provider="local_slm")
        provider = get_slm_provider(config=cfg)
        self.assertIsInstance(provider, LocalSLMProvider)

    def test_factory_mock(self):
        cfg = Settings(slm_provider="mock")
        provider = get_slm_provider(config=cfg)
        self.assertIsInstance(provider, MockSLMProvider)

    def test_factory_onnx(self):
        cfg = Settings(slm_provider="onnx")
        provider = get_slm_provider(config=cfg)
        self.assertIsInstance(provider, ONNXRuntimeSLMProvider)

    def test_factory_local_server(self):
        cfg = Settings(slm_provider="local_server")
        provider = get_slm_provider(config=cfg)
        self.assertIsInstance(provider, LocalServerSLMProvider)


if __name__ == "__main__":
    unittest.main()
