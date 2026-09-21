"""
ONNX Runtime neural inference engine for Qwen2.5-0.5B-Instruct.

Performs real autoregressive transformer token generation with KV caching on CPU / Snapdragon NPU.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

logger = logging.getLogger("exocortex.slm.onnx_engine")

# Qwen2.5 special token IDs
EOS_TOKEN_ID = 151645       # <|im_end|>
PAD_TOKEN_ID = 151643       # <|endoftext|>
NUM_LAYERS = 24             # Qwen2.5-0.5B has 24 transformer layers
NUM_KV_HEADS = 2            # 2 key-value heads in GQA
HEAD_DIM = 64               # Head dimension


@dataclass
class GenerationOutput:
    """Output from real ONNX transformer generation."""
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    tokens_per_second: float


class ONNXNeuralGenerator:
    """
    Autoregressive neural generator for Qwen2.5-0.5B-Instruct using ONNX Runtime.
    """

    def __init__(
        self,
        model_path: Path,
        tokenizer_path: Path,
        execution_provider: str = "CPUExecutionProvider",
    ) -> None:
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.execution_provider = execution_provider
        self.session: Optional[ort.InferenceSession] = None
        self.tokenizer: Optional[Tokenizer] = None
        self.is_loaded = False
        self.load_time_ms = 0.0

        self.load()

    def load(self) -> None:
        """Load ONNX inference session and tokenizer into memory."""
        start_time = time.perf_counter()

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at: {self.model_path}. "
                "Run 'python -m exocortex.cli download-model' or ensure model is present."
            )
        if not self.tokenizer_path.exists():
            raise FileNotFoundError(
                f"Tokenizer not found at: {self.tokenizer_path}."
            )

        # 1. Load Tokenizer
        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))

        # 2. Load ONNX Session
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = min(os.cpu_count() or 4, 8)

        providers = [self.execution_provider]
        if self.execution_provider != "CPUExecutionProvider":
            providers.append("CPUExecutionProvider")

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=opts,
            providers=providers,
        )

        self.load_time_ms = (time.perf_counter() - start_time) * 1000
        self.is_loaded = True
        logger.info(
            "Loaded Qwen2.5-0.5B ONNX session in %.1f ms with provider: %s",
            self.load_time_ms,
            self.session.get_providers()[0],
        )

    def format_chatml(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        """Format prompt using Qwen2.5 ChatML template."""
        parts = []
        if system_prompt:
            parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>")
        parts.append(f"<|im_start|>user\n{user_prompt}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 160,
        temperature: float = 0.1,
    ) -> GenerationOutput:
        """
        Execute real autoregressive neural generation with KV cache management.
        """
        if not self.is_loaded or self.session is None or self.tokenizer is None:
            raise RuntimeError("ONNX Neural Generator is not loaded.")

        start_time = time.perf_counter()

        # 1. Tokenize prompt
        full_formatted_prompt = self.format_chatml(prompt, system_prompt)
        encoded = self.tokenizer.encode(full_formatted_prompt)
        prompt_token_ids = encoded.ids
        prompt_tokens = len(prompt_token_ids)

        # 2. Initial Prefill Forward Pass
        input_ids = np.array([prompt_token_ids], dtype=np.int64)
        seq_len = input_ids.shape[1]
        attention_mask = np.ones((1, seq_len), dtype=np.int64)
        position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)

        feeds: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        for i in range(NUM_LAYERS):
            feeds[f"past_key_values.{i}.key"] = np.zeros((1, NUM_KV_HEADS, 0, HEAD_DIM), dtype=np.float32)
            feeds[f"past_key_values.{i}.value"] = np.zeros((1, NUM_KV_HEADS, 0, HEAD_DIM), dtype=np.float32)

        prefill_start = time.perf_counter()
        outputs = self.session.run(None, feeds)
        prefill_time = (time.perf_counter() - prefill_start) * 1000

        # 3. Autoregressive Token Generation Loop
        generated_ids: List[int] = []
        current_seq_len = seq_len
        gen_start = time.perf_counter()

        for _ in range(max_new_tokens):
            logits = outputs[0][0, -1, :]
            
            # Sampling / Greedy argmax
            if temperature > 0.0:
                # Scaled logits
                logits = logits / temperature
                # Subtract max for numerical stability
                exp_logits = np.exp(logits - np.max(logits))
                probs = exp_logits / np.sum(exp_logits)
                next_token = int(np.random.choice(len(probs), p=probs))
            else:
                next_token = int(np.argmax(logits))

            # Stop condition
            if next_token in (EOS_TOKEN_ID, PAD_TOKEN_ID):
                break

            generated_ids.append(next_token)

            # Step feeds with KV caching
            next_input_ids = np.array([[next_token]], dtype=np.int64)
            next_att_mask = np.ones((1, current_seq_len + 1), dtype=np.int64)
            next_pos_ids = np.array([[current_seq_len]], dtype=np.int64)

            step_feeds: Dict[str, Any] = {
                "input_ids": next_input_ids,
                "attention_mask": next_att_mask,
                "position_ids": next_pos_ids,
            }
            for i in range(NUM_LAYERS):
                step_feeds[f"past_key_values.{i}.key"] = outputs[1 + 2 * i]
                step_feeds[f"past_key_values.{i}.value"] = outputs[2 + 2 * i]

            outputs = self.session.run(None, step_feeds)
            current_seq_len += 1

        gen_time = (time.perf_counter() - gen_start) * 1000
        total_time = (time.perf_counter() - start_time) * 1000

        # 4. Decode generated tokens
        decoded_text = self.tokenizer.decode(generated_ids)
        completion_tokens = len(generated_ids)
        tps = (completion_tokens / (gen_time / 1000.0)) if gen_time > 0 else 0.0

        return GenerationOutput(
            text=decoded_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_latency_ms=prefill_time,
            generation_latency_ms=gen_time,
            total_latency_ms=total_time,
            tokens_per_second=tps,
        )
