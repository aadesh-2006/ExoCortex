"""
ExoCortex: Local-First Autonomous Personal Computer Agent for Windows.

Designed for the Qualcomm Snapdragon AI Lab Build & Present Challenge.
"""

__version__ = "0.1.0"
__author__ = "Aadesh Gund"

from exocortex.config import get_config, Settings
from exocortex.agent import ExoCortexAgent
from exocortex.health import run_health_check

__all__ = [
    "__version__",
    "get_config",
    "Settings",
    "ExoCortexAgent",
    "run_health_check",
]
