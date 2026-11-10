"""
ExoCortex Agentic Execution Engine.

Provides safe multi-step execution loop, state tracking, loop detection,
and auditable execution tracing.
"""

from __future__ import annotations

from exocortex.execution.executor import (
    DEFAULT_MAX_IDENTICAL_ACTIONS,
    DEFAULT_MAX_RESULT_CHARS,
    DEFAULT_MAX_STEPS,
    AgentExecutor,
)
from exocortex.execution.fast_path import (
    CANONICAL_FAST_PATH_MAP,
    FAST_PATH_ELIGIBLE_TOOLS,
    FastPathRouter,
    normalize_fast_path_query,
)
from exocortex.execution.state import (
    ActionHistoryTracker,
    ActionType,
    ExecutionState,
    ExecutionStatus,
    ExecutionStep,
)
from exocortex.execution.trace import ExecutionTrace

__all__ = [
    "AgentExecutor",
    "FastPathRouter",
    "CANONICAL_FAST_PATH_MAP",
    "FAST_PATH_ELIGIBLE_TOOLS",
    "normalize_fast_path_query",
    "ExecutionState",
    "ExecutionStatus",
    "ExecutionStep",
    "ExecutionTrace",
    "ActionType",
    "ActionHistoryTracker",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_MAX_IDENTICAL_ACTIONS",
    "DEFAULT_MAX_RESULT_CHARS",
]
