"""
Unit and integration tests for Hardware Acceleration and Runtime Dispatch.
"""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from exocortex.benchmarking import BenchmarkResult, run_benchmark
from exocortex.config import Settings
from exocortex.runtime.capabilities import CapabilityStatus, RuntimeProfile
from exocortex.runtime.detector import (
    detect_available_providers,
    detect_memory_gb,
    get_onnxruntime_version,
)
from exocortex.runtime.provider import RuntimeDispatcher
from exocortex.slm.base import SLMResponse
from exocortex.slm.mock_provider import MockSLMProvider


class TestRuntimeDetection(unittest.TestCase):
    """Test safe hardware, ORT, and execution provider detection without crashing."""

    def test_onnxruntime_version_detected(self):
        ver = get_onnxruntime_version()
        self.assertIsInstance(ver, str)
        self.assertNotEqual(ver, "")

    def test_detect_available_providers(self):
        providers = detect_available_providers()
        self.assertIsInstance(providers, list)
        self.assertIn("CPUExecutionProvider", providers)

    def test_detect_memory_gb(self):
        total, avail = detect_memory_gb()
        self.assertGreater(total, 0.0)
        self.assertGreater(avail, 0.0)


class TestRuntimeDispatcher(unittest.TestCase):
    """Test deterministic execution provider resolution and fallback handling."""

    def test_auto_selects_cpu_when_only_cpu_available(self):
        ep, reason = RuntimeDispatcher.resolve_provider("auto", ["CPUExecutionProvider"])
        self.assertEqual(ep, "CPUExecutionProvider")
        self.assertIsNone(reason)

    def test_auto_selects_dml_when_available(self):
        ep, reason = RuntimeDispatcher.resolve_provider("auto", ["DmlExecutionProvider", "CPUExecutionProvider"])
        self.assertEqual(ep, "DmlExecutionProvider")
        self.assertIsNone(reason)

    def test_explicit_cpu_target(self):
        ep, reason = RuntimeDispatcher.resolve_provider("cpu", ["DmlExecutionProvider", "CPUExecutionProvider"])
        self.assertEqual(ep, "CPUExecutionProvider")
        self.assertIsNone(reason)

    def test_explicit_dml_when_available(self):
        ep, reason = RuntimeDispatcher.resolve_provider("dml", ["DmlExecutionProvider", "CPUExecutionProvider"])
        self.assertEqual(ep, "DmlExecutionProvider")
        self.assertIsNone(reason)

    def test_explicit_dml_when_missing(self):
        """Verify explicit DML request when DML is missing gives warning reason."""
        ep, reason = RuntimeDispatcher.resolve_provider("dml", ["CPUExecutionProvider"])
        self.assertEqual(ep, "DmlExecutionProvider")
        self.assertIsNotNone(reason)
        self.assertIn("not available", reason)

    def test_invalid_target_falls_back_to_cpu(self):
        ep, reason = RuntimeDispatcher.resolve_provider("invalid_provider_target", ["CPUExecutionProvider"])
        self.assertEqual(ep, "CPUExecutionProvider")
        self.assertIn("Unknown execution provider target", reason)

    def test_runtime_profile_generation(self):
        profile = RuntimeDispatcher.get_runtime_profile(
            requested_target="auto",
            active_provider="CPUExecutionProvider",
        )
        self.assertIsInstance(profile, RuntimeProfile)
        self.assertEqual(profile.active_provider, "CPUExecutionProvider")
        self.assertIn("CPUExecutionProvider", profile.available_providers)
        self.assertTrue(profile.total_memory_gb > 0)
        self.assertTrue(profile.available_memory_gb > 0)


class TestONNXEngineFallback(unittest.TestCase):
    """Test ONNX engine session initialization and fallback safety."""

    @patch("onnxruntime.InferenceSession")
    @patch("tokenizers.Tokenizer.from_file")
    def test_provider_initialization_failure_falls_back_to_cpu(self, mock_tok, mock_sess):
        """
        Verify that if accelerator provider throws an exception during session creation,
        the engine catches it, logs a warning, and initializes with CPUExecutionProvider safely.
        """
        # First call with DML fails; second call with CPU succeeds
        mock_cpu_session = MagicMock()
        mock_cpu_session.get_providers.return_value = ["CPUExecutionProvider"]

        def side_effect(path, sess_options=None, providers=None):
            if providers and "DmlExecutionProvider" in providers:
                raise RuntimeError("Failed to load DirectML: DirectML.dll not found")
            return mock_cpu_session

        mock_sess.side_effect = side_effect

        from exocortex.slm.onnx_engine import ONNXNeuralGenerator
        
        with patch.object(Path, "exists", return_value=True):
            generator = ONNXNeuralGenerator(
                model_path=Path("dummy/model.onnx"),
                tokenizer_path=Path("dummy/tokenizer.json"),
                execution_provider="dml",
            )
            self.assertTrue(generator.fallback_occurred)
            self.assertIn("DirectML", generator.fallback_reason)
            self.assertEqual(generator.active_provider, "CPUExecutionProvider")
            self.assertFalse(generator.is_accelerated)


class TestBenchmarkingLabels(unittest.TestCase):
    """Test benchmark classification and metrics reporting."""

    def test_cpu_benchmark_label(self):
        mock_slm = MockSLMProvider()
        mock_slm.get_model_info = lambda: {
            "model_name": "mock",
            "execution_provider": "CPUExecutionProvider",
            "is_npu_accelerated": False,
        }
        res = run_benchmark(provider=mock_slm, prompts=["Test prompt"])
        self.assertEqual(res.benchmark_type, "CPU benchmark")
        self.assertFalse(res.is_npu_accelerated)
        self.assertEqual(res.iterations, 1)

    def test_benchmark_to_dict(self):
        res = BenchmarkResult(
            model_name="qwen2.5-0.5b-instruct",
            benchmark_type="CPU benchmark",
            requested_provider="auto",
            execution_provider="CPUExecutionProvider",
            is_npu_accelerated=False,
            model_load_time_ms=150.0,
            iterations=3,
            avg_total_latency_ms=45.0,
            avg_prompt_latency_ms=15.0,
            avg_generation_latency_ms=30.0,
            avg_tokens_per_second=25.0,
            total_tokens_generated=75,
            peak_ram_mb=350.0,
        )
        d = res.to_dict()
        self.assertEqual(d["benchmark_type"], "CPU benchmark")
        self.assertEqual(d["execution_provider"], "CPUExecutionProvider")
        self.assertFalse(d["is_npu_accelerated"])


if __name__ == "__main__":
    unittest.main()
