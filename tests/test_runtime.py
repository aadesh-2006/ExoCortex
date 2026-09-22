"""
Unit and integration tests for Milestone 5 Snapdragon Hardware Acceleration and Runtime Dispatch.
"""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from exocortex.benchmarking import BenchmarkResult, run_benchmark
from exocortex.config import Settings
from exocortex.runtime.capabilities import CapabilityStatus, NPUCapability, RuntimeProfile
from exocortex.runtime.detector import (
    detect_available_providers,
    detect_npu_capability,
    get_onnxruntime_version,
    is_snapdragon_hardware,
)
from exocortex.runtime.provider import RuntimeDispatcher
from exocortex.slm.base import SLMResponse
from exocortex.slm.mock_provider import MockSLMProvider


class TestRuntimeDetection(unittest.TestCase):
    """Test safe hardware, ORT, and NPU detection without crashing."""

    def test_onnxruntime_version_detected(self):
        ver = get_onnxruntime_version()
        self.assertIsInstance(ver, str)
        self.assertNotEqual(ver, "")

    def test_detect_available_providers(self):
        providers = detect_available_providers()
        self.assertIsInstance(providers, list)
        self.assertIn("CPUExecutionProvider", providers)

    def test_intel_host_detection(self):
        """Verify Intel development machine does not claim Snapdragon hardware."""
        with patch("platform.processor", return_value="Intel64 Family 6 Model 154 Stepping 3, GenuineIntel"), \
             patch("platform.machine", return_value="AMD64"):
            self.assertFalse(is_snapdragon_hardware())

    def test_snapdragon_host_detection(self):
        """Verify Qualcomm Snapdragon processor is recognized."""
        with patch("platform.processor", return_value="Snapdragon(R) X Elite - X1E80100 - Qualcomm(R) Oryon(TM) CPU"), \
             patch("platform.machine", return_value="ARM64"):
            self.assertTrue(is_snapdragon_hardware())

    def test_npu_capability_intel_machine(self):
        """Verify NPU capability accurately reports UNAVAILABLE on Intel machine."""
        with patch("exocortex.runtime.detector.is_snapdragon_hardware", return_value=False):
            npu = detect_npu_capability(providers=["CPUExecutionProvider"])
            self.assertFalse(npu.is_supported)
            self.assertFalse(npu.is_installed)
            self.assertFalse(npu.is_active)
            self.assertIn("UNAVAILABLE", npu.status_summary)

    def test_npu_capability_snapdragon_without_qnn(self):
        """Verify Snapdragon hardware without QNN EP package is reported as SUPPORTED but not installed."""
        with patch("exocortex.runtime.detector.is_snapdragon_hardware", return_value=True):
            npu = detect_npu_capability(providers=["CPUExecutionProvider"])
            self.assertTrue(npu.is_supported)
            self.assertFalse(npu.is_installed)
            self.assertFalse(npu.is_active)
            self.assertIn("Requires onnxruntime-qnn", npu.npu_name)

    def test_npu_capability_snapdragon_with_qnn(self):
        """Verify Snapdragon hardware with QNN EP package is reported as READY."""
        with patch("exocortex.runtime.detector.is_snapdragon_hardware", return_value=True):
            npu = detect_npu_capability(providers=["QNNExecutionProvider", "CPUExecutionProvider"])
            self.assertTrue(npu.is_supported)
            self.assertTrue(npu.is_installed)
            self.assertIn("READY", npu.status_summary)


class TestRuntimeDispatcher(unittest.TestCase):
    """Test deterministic execution provider resolution and fallback handling."""

    def test_auto_on_intel_selects_cpu(self):
        with patch("exocortex.runtime.provider.is_snapdragon_hardware", return_value=False):
            ep, reason = RuntimeDispatcher.resolve_provider("auto", ["CPUExecutionProvider"])
            self.assertEqual(ep, "CPUExecutionProvider")
            self.assertIsNone(reason)

    def test_auto_on_snapdragon_with_qnn_selects_qnn(self):
        with patch("exocortex.runtime.provider.is_snapdragon_hardware", return_value=True):
            ep, reason = RuntimeDispatcher.resolve_provider("auto", ["QNNExecutionProvider", "CPUExecutionProvider"])
            self.assertEqual(ep, "QNNExecutionProvider")
            self.assertIsNone(reason)

    def test_explicit_cpu_target(self):
        ep, reason = RuntimeDispatcher.resolve_provider("cpu", ["QNNExecutionProvider", "CPUExecutionProvider"])
        self.assertEqual(ep, "CPUExecutionProvider")
        self.assertIsNone(reason)

    def test_explicit_qnn_when_missing(self):
        """Verify explicit QNN request when QNN is missing gives warning reason."""
        ep, reason = RuntimeDispatcher.resolve_provider("qnn", ["CPUExecutionProvider"])
        self.assertEqual(ep, "QNNExecutionProvider")
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
            is_npu_active=False,
        )
        self.assertIsInstance(profile, RuntimeProfile)
        self.assertEqual(profile.active_provider, "CPUExecutionProvider")
        self.assertFalse(profile.npu.is_active)
        self.assertIn("CPUExecutionProvider", profile.available_providers)


class TestONNXEngineFallback(unittest.TestCase):
    """Test ONNX engine session initialization and fallback safety."""

    @patch("onnxruntime.InferenceSession")
    @patch("tokenizers.Tokenizer.from_file")
    def test_qnn_initialization_failure_falls_back_to_cpu(self, mock_tok, mock_sess):
        """
        Verify that if QNN provider throws an exception during session creation,
        the engine catches it, logs a warning, and initializes with CPUExecutionProvider safely.
        """
        # First call with QNN fails; second call with CPU succeeds
        mock_cpu_session = MagicMock()
        mock_cpu_session.get_providers.return_value = ["CPUExecutionProvider"]

        def side_effect(path, sess_options=None, providers=None):
            if providers and "QNNExecutionProvider" in providers:
                raise RuntimeError("Failed to load QNN DLL: QnnHtp.dll not found")
            return mock_cpu_session

        mock_sess.side_effect = side_effect

        from exocortex.slm.onnx_engine import ONNXNeuralGenerator
        
        with patch.object(Path, "exists", return_value=True):
            generator = ONNXNeuralGenerator(
                model_path=Path("dummy/model.onnx"),
                tokenizer_path=Path("dummy/tokenizer.json"),
                execution_provider="qnn",
            )
            self.assertTrue(generator.fallback_occurred)
            self.assertIn("QNN DLL", generator.fallback_reason)
            self.assertEqual(generator.active_provider, "CPUExecutionProvider")
            self.assertFalse(generator.is_npu_active)


class TestBenchmarkingLabels(unittest.TestCase):
    """Test benchmark classification and metrics reporting."""

    def test_cpu_benchmark_label(self):
        mock_slm = MockSLMProvider()
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
