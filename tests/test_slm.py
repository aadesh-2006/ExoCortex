"""
Unit tests for SLM provider interface and factory.
"""

import unittest
from exocortex.config import Settings
from exocortex.slm.base import SLMMessage, SLMResponse
from exocortex.slm.mock_provider import MockSLMProvider
from exocortex.slm.factory import get_slm_provider
from exocortex.slm.onnx_provider import ONNXRuntimeSLMProvider
from exocortex.slm.local_provider import LocalServerSLMProvider


class TestSLMProviders(unittest.TestCase):
    def test_mock_provider_generate(self):
        provider = MockSLMProvider()
        self.assertTrue(provider.is_available())
        
        resp = provider.generate("test prompt")
        self.assertIsInstance(resp, SLMResponse)
        self.assertTrue(len(resp.content) > 0)
        self.assertEqual(resp.provider_name, "MockSLMProvider")

    def test_mock_provider_chat(self):
        provider = MockSLMProvider()
        messages = [
            SLMMessage(role="system", content="You are ExoCortex."),
            SLMMessage(role="user", content="Hello ExoCortex"),
        ]
        resp = provider.chat(messages)
        self.assertIsInstance(resp, SLMResponse)
        self.assertTrue(len(resp.content) > 0)

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
