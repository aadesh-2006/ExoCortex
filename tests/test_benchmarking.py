"""
Unit tests for local SLM inference benchmarking suite.
"""

import unittest
from exocortex.benchmarking import run_benchmark, BenchmarkResult
from exocortex.slm.local_model_engine import LocalSLMProvider


class TestBenchmarking(unittest.TestCase):
    def test_run_benchmark(self):
        provider = LocalSLMProvider(model_name="qwen2.5-0.5b-instruct")
        res = run_benchmark(provider=provider, prompts=["Test prompt 1", "Test prompt 2"])
        
        self.assertIsInstance(res, BenchmarkResult)
        self.assertEqual(res.iterations, 2)
        self.assertTrue(res.total_tokens_generated > 0)
        self.assertTrue(res.avg_total_latency_ms >= 0.0)
        self.assertTrue(res.peak_ram_mb > 0.0)
        
        d = res.to_dict()
        self.assertIn("model_name", d)
        self.assertIn("avg_tokens_per_second", d)
        self.assertIn("peak_ram_mb", d)
        self.assertEqual(len(d["runs"]), 2)


if __name__ == "__main__":
    unittest.main()
