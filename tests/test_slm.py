"""
Unit tests for SLM provider interface, LocalSLMProvider, and factory.
"""

import io
import json
import socket
import unittest
import urllib.error
from unittest.mock import MagicMock, patch

from exocortex.config import Settings
from exocortex.slm.base import SLMMessage, SLMResponse
from exocortex.slm.factory import get_slm_provider
from exocortex.slm.local_model_engine import LocalSLMProvider
from exocortex.slm.local_provider import (
    LocalServerConnectionError,
    LocalServerSLMError,
    LocalServerSLMProvider,
    LocalServerTimeoutError,
)
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
        self.assertEqual(provider.timeout, 90.0)

    def test_factory_local_server_with_custom_timeout(self):
        cfg = Settings(slm_provider="ollama", local_server_timeout=45.0)
        provider = get_slm_provider(config=cfg)
        self.assertIsInstance(provider, LocalServerSLMProvider)
        self.assertEqual(provider.timeout, 45.0)

        # Direct kwarg override
        provider2 = get_slm_provider("ollama", timeout=120.0)
        self.assertEqual(provider2.timeout, 120.0)

    def test_model_manager_status_accepts_small_config(self):
        from exocortex.slm.model_manager import check_model_status
        status = check_model_status("qwen2.5-0.5b-instruct")
        self.assertTrue(status.is_ready)
        self.assertTrue(status.files_present.get("config", False))
        self.assertTrue(status.files_present.get("tokenizer", False))
        self.assertTrue(status.files_present.get("model_onnx", False))
        self.assertEqual(len(status.missing_files), 0)


class TestLocalServerSLMProvider(unittest.TestCase):
    """Focused unit tests for LocalServerSLMProvider request/response handling and error behavior."""

    def setUp(self):
        self.provider = LocalServerSLMProvider(
            base_url="http://localhost:11434/v1",
            model_name="qwen3:4b",
            api_key="test-key",
            timeout=90.0,
        )

    def test_init_defaults_and_model_info(self):
        default_provider = LocalServerSLMProvider()
        self.assertEqual(default_provider.base_url, "http://localhost:11434/v1")
        self.assertEqual(default_provider.model_name, "phi-3.5-mini-instruct")
        self.assertEqual(default_provider.timeout, 90.0)

        info = self.provider.get_model_info()
        self.assertEqual(info["model_name"], "qwen3:4b")
        self.assertEqual(info["provider"], "LocalServerSLMProvider")
        self.assertEqual(info["base_url"], "http://localhost:11434/v1")
        self.assertEqual(info["timeout"], 90.0)
        self.assertTrue(info["is_local"])

    def test_init_rejects_non_positive_or_invalid_timeout(self):
        invalid_timeouts = [0, -1, -50.0, "abc", ""]
        for t in invalid_timeouts:
            with self.subTest(timeout=t):
                with self.assertRaises(ValueError) as ctx:
                    LocalServerSLMProvider(timeout=t)
                self.assertIn("must be a positive number of seconds", str(ctx.exception))

    @patch("urllib.request.urlopen")
    def test_is_available_true(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        self.assertTrue(self.provider.is_available())

        # Verify endpoint and health-check timeout (1.5s)
        self.assertEqual(mock_urlopen.call_count, 1)
        req, kwargs = mock_urlopen.call_args
        self.assertEqual(req[0].get_full_url(), "http://localhost:11434/v1/models")
        self.assertEqual(kwargs.get("timeout"), 1.5)

    @patch("urllib.request.urlopen")
    def test_is_available_false_on_connection_failure(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
        self.assertFalse(self.provider.is_available())

    @patch("urllib.request.urlopen")
    def test_generate_request_construction_and_response_parsing(self, mock_urlopen):
        fake_api_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": '```json\n{"thought_summary": "ok", "intent": "answer", "steps": [], "requires_confirmation": false, "direct_response": "Done."}\n```',
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 42,
                "completion_tokens": 18,
            },
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(fake_api_response).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        res = self.provider.generate(
            prompt="Summarize this text",
            system_prompt="You are ExoCortex",
            temperature=0.1,
            max_tokens=256,
        )

        # 1. Verify response parsed cleanly
        self.assertIsInstance(res, SLMResponse)
        self.assertIn("direct_response", res.content)
        self.assertEqual(res.prompt_tokens, 42)
        self.assertEqual(res.completion_tokens, 18)
        self.assertEqual(res.model_name, "qwen3:4b")
        self.assertEqual(res.provider_name, "LocalServerSLMProvider")

        # 2. Verify request details
        req, kwargs = mock_urlopen.call_args
        request_obj = req[0]
        self.assertEqual(request_obj.get_full_url(), "http://localhost:11434/v1/chat/completions")
        self.assertEqual(request_obj.get_method(), "POST")
        self.assertEqual(kwargs.get("timeout"), 90.0)

        # Verify payload sent
        sent_payload = json.loads(request_obj.data.decode("utf-8"))
        self.assertEqual(sent_payload["model"], "qwen3:4b")
        self.assertEqual(sent_payload["temperature"], 0.1)
        self.assertEqual(sent_payload["max_tokens"], 256)
        self.assertEqual(
            sent_payload["messages"],
            [
                {"role": "system", "content": "You are ExoCortex"},
                {"role": "user", "content": "Summarize this text"},
            ],
        )

    @patch("urllib.request.urlopen")
    def test_chat_request_construction(self, mock_urlopen):
        fake_api_response = {
            "choices": [{"message": {"role": "assistant", "content": "Chat reply"}}]
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(fake_api_response).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        messages = [
            SLMMessage(role="system", content="System instruction"),
            SLMMessage(role="user", content="User message"),
        ]
        res = self.provider.chat(messages, temperature=0.3)

        self.assertEqual(res.content, "Chat reply")
        req, _ = mock_urlopen.call_args
        sent_payload = json.loads(req[0].data.decode("utf-8"))
        self.assertEqual(len(sent_payload["messages"]), 2)
        self.assertEqual(sent_payload["messages"][0]["role"], "system")
        self.assertEqual(sent_payload["messages"][1]["content"], "User message")

    @patch("urllib.request.urlopen")
    def test_tool_calls_parsing(self, mock_urlopen):
        fake_api_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "function": {
                                    "name": "web_search",
                                    "arguments": '{"query": "Pune, India"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(fake_api_response).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        res = self.provider.generate("Search Pune")
        self.assertEqual(len(res.tool_calls), 1)
        tc = res.tool_calls[0]
        self.assertEqual(tc.tool_name, "web_search")
        self.assertEqual(tc.arguments, {"query": "Pune, India"})
        self.assertEqual(tc.call_id, "call_123")

    @patch("urllib.request.urlopen")
    def test_http_error_surfaced_as_provider_error(self, mock_urlopen):
        err_fp = io.BytesIO(b'{"error": "model \'qwen3:4b\' not found"}')
        http_err = urllib.error.HTTPError(
            url="http://localhost:11434/v1/chat/completions",
            code=404,
            msg="Not Found",
            hdrs={},
            fp=err_fp,
        )
        mock_urlopen.side_effect = http_err

        with self.assertRaises(LocalServerSLMError) as ctx:
            self.provider.generate("Test prompt")

        self.assertIn("HTTP 404", str(ctx.exception))
        self.assertIn("model 'qwen3:4b' not found", str(ctx.exception))

    @patch("urllib.request.urlopen")
    def test_connection_error_surfaced_as_provider_error(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.URLError(ConnectionRefusedError("Connection refused"))

        with self.assertRaises(LocalServerConnectionError) as ctx:
            self.provider.generate("Test prompt")

        self.assertIn("Failed to connect to local SLM server", str(ctx.exception))
        # Ensure it is also caught as ConnectionError and LocalServerSLMError
        self.assertIsInstance(ctx.exception, ConnectionError)
        self.assertIsInstance(ctx.exception, LocalServerSLMError)

    @patch("urllib.request.urlopen")
    def test_timeout_error_surfaced_as_provider_error(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.URLError(socket.timeout("timed out"))

        with self.assertRaises(LocalServerTimeoutError) as ctx:
            self.provider.generate("Test prompt")

        self.assertIn("timed out after 90.0s", str(ctx.exception))
        # Ensure it is also caught as TimeoutError and LocalServerSLMError
        self.assertIsInstance(ctx.exception, TimeoutError)
        self.assertIsInstance(ctx.exception, LocalServerSLMError)

    @patch("urllib.request.urlopen")
    def test_timeout_error_with_string_reason(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.URLError("The read operation timed out")

        with self.assertRaises(LocalServerTimeoutError) as ctx:
            self.provider.generate("Test prompt")

        self.assertIn("timed out after 90.0s", str(ctx.exception))

    @patch("urllib.request.urlopen")
    def test_empty_choices_raises_provider_error(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"choices": [], "error": {"message": "GPU OOM"}}).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        with self.assertRaises(LocalServerSLMError) as ctx:
            self.provider.generate("Test prompt")

        self.assertIn("without choices", str(ctx.exception))
        self.assertIn("GPU OOM", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
