"""
Configuration management for ExoCortex.

Loads settings from environment variables and optional .env files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


HardwareTarget = Literal["auto", "qnn", "dml", "cpu"]
SLMProviderType = Literal["mock", "onnx", "local_server", "llamacpp"]


@dataclass
class Settings:
    """Central configuration for ExoCortex agent runtime."""

    env: str = field(
        default_factory=lambda: os.getenv("EXOCORTEX_ENV", "development")
    )
    log_level: str = field(
        default_factory=lambda: os.getenv("EXOCORTEX_LOG_LEVEL", "INFO").upper()
    )
    data_dir: Path = field(
        default_factory=lambda: Path(
            os.path.expanduser(os.getenv("EXOCORTEX_DATA_DIR", "~/.exocortex"))
        )
    )

    # SLM Settings
    slm_provider: SLMProviderType = field(
        default_factory=lambda: os.getenv("EXOCORTEX_SLM_PROVIDER", "local_slm")  # type: ignore
    )
    model_name: str = field(
        default_factory=lambda: os.getenv(
            "EXOCORTEX_MODEL_NAME", "qwen2.5-0.5b-instruct"
        )
    )
    model_path: str = field(
        default_factory=lambda: os.getenv("EXOCORTEX_MODEL_PATH", "")
    )

    # Hardware & Snapdragon Acceleration
    hardware_target: HardwareTarget = field(
        default_factory=lambda: os.getenv("EXOCORTEX_HARDWARE_TARGET", "auto")  # type: ignore
    )

    # Local Server Settings
    local_server_url: str = field(
        default_factory=lambda: os.getenv(
            "EXOCORTEX_LOCAL_SERVER_URL", "http://localhost:11434/v1"
        )
    )
    local_server_api_key: str = field(
        default_factory=lambda: os.getenv(
            "EXOCORTEX_LOCAL_SERVER_API_KEY", "not-needed"
        )
    )

    # Security & Guardrails
    require_confirmation_for_sensitive: bool = field(
        default_factory=lambda: os.getenv(
            "EXOCORTEX_REQUIRE_CONFIRMATION_FOR_SENSITIVE", "true"
        ).lower()
        in ("1", "true", "yes", "on")
    )
    max_plan_steps: int = field(
        default_factory=lambda: int(os.getenv("EXOCORTEX_MAX_PLAN_STEPS", "10"))
    )

    def ensure_data_dir(self) -> Path:
        """Ensure that the data directory exists."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir


_config_instance: Settings | None = None


def get_config(reload: bool = False) -> Settings:
    """Retrieve the singleton configuration instance."""
    global _config_instance
    if _config_instance is None or reload:
        _config_instance = Settings()
    return _config_instance
