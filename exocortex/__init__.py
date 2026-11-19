"""
ExoCortex: Hardware-Agnostic, Local-First AI Agent for Windows.

Interprets natural-language requests, reasons through tasks,
and executes them using permission-controlled desktop tools.
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
