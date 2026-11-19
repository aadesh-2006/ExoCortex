"""
Deterministic Fast-Path Router for Canonical and Parameterized Desktop Requests in ExoCortex.

Bypasses heavy SLM inference for:
1. Unambiguous, canonical zero-argument desktop requests (e.g., list open windows, list processes).
2. Parameterized application-state inquiries (e.g., is <app> open?, is <app> running?).

Executes strictly through ToolRegistry.execute(), preserving all permission boundaries,
execution traces, and returning structured factual answers without invoking Qwen3:4B.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from exocortex.execution.state import (
    ActionType,
    ExecutionStatus,
    ExecutionStep,
)
from exocortex.execution.trace import ExecutionTrace
from exocortex.schema import AgentDecision, StructuredStep
from exocortex.tools.base import PermissionLevel
from exocortex.tools.registry import ToolRegistry, get_default_tool_registry

logger = logging.getLogger("exocortex.execution.fast_path")

# Unambiguous canonical zero-argument queries mapped to exact registered tool names.
# Only zero-argument, read-only tools with PermissionLevel.SAFE are eligible.
CANONICAL_FAST_PATH_MAP: Dict[str, str] = {
    # list_open_windows (existing)
    "list open windows": "list_open_windows",
    "list windows": "list_open_windows",
    "what application windows are open": "list_open_windows",
    "what windows are open": "list_open_windows",
    "show open windows": "list_open_windows",
    "show windows": "list_open_windows",
    "get open windows": "list_open_windows",
    "get windows": "list_open_windows",
    # list_open_windows (Task 8.8.1 expanded)
    "show me open windows": "list_open_windows",
    "what windows are currently open": "list_open_windows",
    "what desktop windows are open": "list_open_windows",
    "list desktop windows": "list_open_windows",
    "display open windows": "list_open_windows",
    "get open window list": "list_open_windows",

    # list_processes (existing)
    "list processes": "list_processes",
    "list running processes": "list_processes",
    "show processes": "list_processes",
    "show running processes": "list_processes",
    "get processes": "list_processes",
    "get running processes": "list_processes",
    "what processes are running": "list_processes",
    # list_processes (Task 8.8.1 expanded)
    "display running processes": "list_processes",
    "show process list": "list_processes",
    "show me running processes": "list_processes",
    "what processes are currently running": "list_processes",
    "list active processes": "list_processes",
    "get process list": "list_processes",

    # system_info (Task 8.8.1)
    "system info": "system_info",
    "system information": "system_info",
    "show system info": "system_info",
    "show system information": "system_info",
    "get system info": "system_info",
    "get system information": "system_info",
    "display system info": "system_info",
    "display system information": "system_info",

    # health_check (Task 8.8.1)
    "health check": "health_check",
    "check system health": "health_check",
    "run health check": "health_check",
    "system health check": "health_check",
    "check exocortex health": "health_check",
    "exocortex health check": "health_check",
}

# Explicit set of tools permitted for canonical fast-path execution.
FAST_PATH_ELIGIBLE_TOOLS: Set[str] = {
    "list_open_windows",
    "list_processes",
    "system_info",
    "health_check",
}

# Explicit intent labels for trace decisions produced by canonical fast paths.
FAST_PATH_INTENT_MAP: Dict[str, str] = {
    "list_open_windows": "window_enumeration",
    "list_processes": "process_inspection",
    "system_info": "system_information",
    "health_check": "system_health_check",
}


@dataclass(frozen=True)
class ApplicationMetadata:
    """Explicit metadata for recognizing and mapping approved desktop applications."""
    display_name: str
    executable_names: Tuple[str, ...]
    window_title_patterns: Tuple[str, ...]


# Explicit, small alias mapping for commonly recognizable applications.
# Preserves case-insensitive lookup and Windows executable conventions.
APPROVED_APPLICATION_MAP: Dict[str, ApplicationMetadata] = {
    "notepad": ApplicationMetadata(
        display_name="Notepad",
        executable_names=("notepad.exe",),
        window_title_patterns=("notepad",),
    ),
    "windows notepad": ApplicationMetadata(
        display_name="Notepad",
        executable_names=("notepad.exe",),
        window_title_patterns=("notepad",),
    ),
    "chrome": ApplicationMetadata(
        display_name="Google Chrome",
        executable_names=("chrome.exe",),
        window_title_patterns=("google chrome", "chrome"),
    ),
    "google chrome": ApplicationMetadata(
        display_name="Google Chrome",
        executable_names=("chrome.exe",),
        window_title_patterns=("google chrome", "chrome"),
    ),
    "firefox": ApplicationMetadata(
        display_name="Mozilla Firefox",
        executable_names=("firefox.exe",),
        window_title_patterns=("mozilla firefox", "firefox"),
    ),
    "mozilla firefox": ApplicationMetadata(
        display_name="Mozilla Firefox",
        executable_names=("firefox.exe",),
        window_title_patterns=("mozilla firefox", "firefox"),
    ),
    "spotify": ApplicationMetadata(
        display_name="Spotify",
        executable_names=("spotify.exe",),
        window_title_patterns=("spotify",),
    ),
    "vscode": ApplicationMetadata(
        display_name="Visual Studio Code",
        executable_names=("code.exe",),
        window_title_patterns=("visual studio code", "- code"),
    ),
    "vs code": ApplicationMetadata(
        display_name="Visual Studio Code",
        executable_names=("code.exe",),
        window_title_patterns=("visual studio code", "- code"),
    ),
    "visual studio code": ApplicationMetadata(
        display_name="Visual Studio Code",
        executable_names=("code.exe",),
        window_title_patterns=("visual studio code", "- code"),
    ),
    "windows terminal": ApplicationMetadata(
        display_name="Windows Terminal",
        executable_names=("windowsterminal.exe",),
        window_title_patterns=("windows terminal", "terminal"),
    ),
}

# Ambiguous application aliases that must fall through to the SLM.
AMBIGUOUS_APPLICATION_ALIASES: Set[str] = {
    "browser",
    "web browser",
    "terminal",
    "editor",
    "text editor",
    "code editor",
    "code",
    "ide",
    "app",
    "application",
}

# Guard regex rejecting mutating verbs and compound requests.
REJECT_MUTATING_OR_COMPOUND = re.compile(
    r"\b(close|kill|terminate|stop|end|launch|start|restart|quit|exit|and|then|after|also|before)\b|[;&|]",
    re.IGNORECASE,
)

# Anchored regexes for application-state questions.
APPLICATION_STATE_PATTERNS = [
    # "is <app> (currently )?(running|open|active)( right now)?"
    re.compile(
        r"^is\s+(?P<app>.+?)\s+(?:currently\s+)?(?P<target_state>running|open|active)(?:\s+right\s+now)?$",
        re.IGNORECASE,
    ),
    # "(please |can you )?check (if|whether) <app> is (currently )?(running|open|active)( right now)?"
    re.compile(
        r"^(?:please\s+|can\s+you\s+)?check\s+(?:if|whether)\s+(?P<app>.+?)\s+is\s+(?:currently\s+)?(?P<target_state>running|open|active)(?:\s+right\s+now)?$",
        re.IGNORECASE,
    ),
    # "do i have <app> (currently )?(open|running)( right now)?"
    re.compile(
        r"^do\s+i\s+have\s+(?P<app>.+?)\s+(?:currently\s+)?(?P<target_state>open|running)(?:\s+right\s+now)?$",
        re.IGNORECASE,
    ),
]


def normalize_fast_path_query(query: str) -> str:
    """
    Normalize query for exact canonical and parameterized matching.
    
    Performs case folding, removes leading/trailing harmless punctuation and quotes,
    and collapses internal whitespace. Never performs fuzzy or substring matching.
    """
    if not query:
        return ""
    q = query.strip().lower()
    # Strip leading/trailing punctuation and quotation marks commonly found in queries
    q = re.sub(r"^[\s\"'`]+|[\s\"'`?!.,;:]+$", "", q)
    # Collapse multiple whitespace characters into a single space
    q = re.sub(r"\s+", " ", q).strip()
    return q


class FastPathRouter:
    """
    Fast-path dispatcher for canonical zero-argument and application-state desktop requests.
    
    Evaluates incoming user prompts against:
    1. Exact normalized dictionary of canonical zero-argument phrases.
    2. Narrowly defined parameterized application-state inquiries.
    
    When an unambiguous match is found, executes SAFE tools directly via ToolRegistry,
    returning a structured ExecutionTrace with 0 LLM calls.
    
    If the request is ambiguous, mutating, compound, or involves an unrecognized application,
    returns None to allow transparent fall-through to the general SLM cognitive loop.
    """

    def __init__(
        self,
        tool_registry: Optional[ToolRegistry] = None,
        enabled: bool = True,
    ) -> None:
        self.tool_registry = tool_registry or get_default_tool_registry()
        self.enabled = enabled

    def match(self, user_query: str) -> Optional[str]:
        """
        Check if the query exactly matches a canonical zero-argument request.
        
        Returns:
            Tool name (str) if an unambiguous match is found, otherwise None.
        """
        if not self.enabled:
            return None

        normalized = normalize_fast_path_query(user_query)
        tool_name = CANONICAL_FAST_PATH_MAP.get(normalized)
        if not tool_name:
            return None

        # Verify eligibility
        if tool_name not in FAST_PATH_ELIGIBLE_TOOLS:
            logger.warning("Matched tool '%s' is not in fast-path eligible tools.", tool_name)
            return None

        return tool_name

    def match_application_state(
        self, user_query: str
    ) -> Optional[Tuple[ApplicationMetadata, str]]:
        """
        Check if the query matches a parameterized application-state inquiry.

        Returns:
            Tuple of (ApplicationMetadata, target_state) where target_state is 'open',
            'running', or 'active', or None if query does not match or is ambiguous/unknown.
        """
        if not self.enabled:
            return None

        normalized = normalize_fast_path_query(user_query)
        if not normalized:
            return None

        # Reject compound queries or mutating actions
        if REJECT_MUTATING_OR_COMPOUND.search(normalized):
            return None

        target_state: Optional[str] = None
        raw_app: Optional[str] = None

        for pattern in APPLICATION_STATE_PATTERNS:
            match = pattern.match(normalized)
            if match:
                raw_app = match.group("app")
                target_state = match.group("target_state").lower()
                break

        if not raw_app or not target_state:
            return None

        # Clean application entity string of interior quotes and extra whitespace
        app_cleaned = re.sub(r"^[\s\"'`]+|[\s\"'`]+$", "", raw_app)
        app_cleaned = re.sub(r"\s+", " ", app_cleaned).strip().lower()

        # Reject explicitly ambiguous aliases
        if app_cleaned in AMBIGUOUS_APPLICATION_ALIASES:
            logger.info("Application entity '%s' is ambiguous. Falling through to SLM.", app_cleaned)
            return None

        # Look up in approved application mapping
        app_meta = APPROVED_APPLICATION_MAP.get(app_cleaned)
        if not app_meta:
            logger.info("Application entity '%s' is not in approved application map. Falling through to SLM.", app_cleaned)
            return None

        return app_meta, target_state

    def try_execute(self, user_query: str) -> Optional[ExecutionTrace]:
        """
        Attempt fast-path execution for the given user query.
        
        Returns:
            ExecutionTrace if matched and executed, or None if the request must fall through to the SLM.
        """
        if not self.enabled:
            return None

        # 1. Canonical zero-argument fast path
        canonical_tool = self.match(user_query)
        if canonical_tool:
            return self._execute_canonical(user_query, canonical_tool)

        # 2. Parameterized application-state fast path
        app_match = self.match_application_state(user_query)
        if app_match:
            app_meta, target_state = app_match
            return self._execute_application_state(user_query, app_meta, target_state)

        return None

    def _execute_canonical(self, user_query: str, tool_name: str) -> Optional[ExecutionTrace]:
        """Execute a canonical zero-argument tool request directly."""
        # Verify tool registration and permission
        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            logger.warning("Fast-path tool '%s' is not registered. Falling through to SLM.", tool_name)
            return None

        if tool.permission_level != PermissionLevel.SAFE:
            logger.warning(
                "Fast-path tool '%s' permission level is %s (must be SAFE). Falling through.",
                tool_name,
                tool.permission_level,
            )
            return None

        logger.info("Executing fast-path route for canonical query: '%s' -> %s()", user_query, tool_name)
        start_time = time.perf_counter()

        # Execute safely strictly via ToolRegistry
        tool_res = self.tool_registry.execute(tool_name, {})
        duration_ms = (time.perf_counter() - start_time) * 1000

        # Construct intent and structured decision
        intent = FAST_PATH_INTENT_MAP.get(tool_name, tool_name)
        has_valid_output = bool(tool_res.output and tool_res.output.strip())
        is_success = bool(tool_res.success and has_valid_output)

        if is_success:
            status = ExecutionStatus.COMPLETED
            error_msg = None
            final_response = tool_res.output
            obs_text = f"[{tool_name}]: {tool_res.output}"
        else:
            status = ExecutionStatus.ERROR
            if not tool_res.success:
                error_msg = tool_res.error or f"Tool '{tool_name}' execution failed."
            else:
                error_msg = f"Tool '{tool_name}' returned empty output."
            final_response = None
            obs_text = f"[{tool_name} Error]: {error_msg}"

        decision = AgentDecision(
            thought_summary=f"Canonical zero-argument fast path executed for '{tool_name}'.",
            intent=intent,
            steps=[StructuredStep(tool=tool_name, arguments={})],
            requires_confirmation=False,
            direct_response=final_response,
        )

        step = ExecutionStep(
            step_number=1,
            timestamp=time.time(),
            decision=decision,
            action_type=ActionType.TOOL_CALL,
            tool_name=tool_name,
            arguments={},
            permission_level=tool.permission_level,
            tool_result=tool_res,
            observation_text=obs_text,
            duration_ms=duration_ms,
            status=status,
            error_message=error_msg,
        )

        trace = ExecutionTrace(
            user_goal=user_query,
            status=status,
            start_time=start_time,
            slm_info={
                "provider": "FastPathRouter",
                "model_name": "deterministic_fast_path",
                "is_local": True,
                "llm_calls": 0,
            },
        )
        trace.steps.append(step)
        trace.finalize(
            status=status,
            final_response=final_response,
            error_message=error_msg,
        )

        return trace

    def _execute_application_state(
        self,
        user_query: str,
        app_meta: ApplicationMetadata,
        target_state: str,
    ) -> Optional[ExecutionTrace]:
        """
        Execute read-only desktop inspection tools to determine application state.
        
        Distinguishes:
        1. Open with visible window.
        2. Running in background without visible window.
        3. Not running.
        4. Active foreground window.
        """
        # Safety checks: Both tools must be registered and declared SAFE
        tool_windows = self.tool_registry.get_tool("list_open_windows")
        tool_procs = self.tool_registry.get_tool("list_processes")

        if not tool_windows or not tool_procs:
            logger.warning("Required application-state tools are not registered. Falling through to SLM.")
            return None

        if tool_windows.permission_level != PermissionLevel.SAFE or tool_procs.permission_level != PermissionLevel.SAFE:
            logger.warning("Application-state inspection tools must both have SAFE permission level. Falling through.")
            return None

        logger.info(
            "Executing application-state fast-path for '%s' (%s, target: %s)",
            user_query,
            app_meta.display_name,
            target_state,
        )
        total_start = time.perf_counter()

        # Step 1: Execute list_open_windows
        start_windows = time.perf_counter()
        tool_res_windows = self.tool_registry.execute("list_open_windows", {})
        dur_windows = (time.perf_counter() - start_windows) * 1000

        if not tool_res_windows.success:
            err = tool_res_windows.error or "Failed to enumerate desktop windows."
            return self._build_app_state_error_trace(
                user_query, "list_open_windows", tool_windows, tool_res_windows, err, dur_windows
            )

        if not tool_res_windows.output or not tool_res_windows.output.strip():
            err = "Tool 'list_open_windows' returned empty output."
            return self._build_app_state_error_trace(
                user_query, "list_open_windows", tool_windows, tool_res_windows, err, dur_windows
            )

        # Step 2: Execute list_processes
        start_procs = time.perf_counter()
        tool_res_procs = self.tool_registry.execute("list_processes", {})
        dur_procs = (time.perf_counter() - start_procs) * 1000

        if not tool_res_procs.success:
            err = tool_res_procs.error or "Failed to inspect processes."
            return self._build_app_state_error_trace(
                user_query,
                "list_processes",
                tool_procs,
                tool_res_procs,
                err,
                dur_procs,
                prior_tool="list_open_windows",
                prior_result=tool_res_windows,
                prior_tool_obj=tool_windows,
                prior_dur=dur_windows,
            )

        if not tool_res_procs.output or not tool_res_procs.output.strip():
            err = "Tool 'list_processes' returned empty output."
            return self._build_app_state_error_trace(
                user_query,
                "list_processes",
                tool_procs,
                tool_res_procs,
                err,
                dur_procs,
                prior_tool="list_open_windows",
                prior_result=tool_res_windows,
                prior_tool_obj=tool_windows,
                prior_dur=dur_windows,
            )

        # Filter windows
        windows: List[Dict[str, Any]] = []
        if isinstance(tool_res_windows.data, dict):
            windows = tool_res_windows.data.get("windows", [])

        matching_windows: List[Dict[str, Any]] = []
        for w in windows:
            proc_name = str(w.get("process_name", "")).lower()
            title = str(w.get("title", "")).lower()
            exe_matches = any(exe.lower() == proc_name for exe in app_meta.executable_names)
            title_matches = any(pat.lower() in title for pat in app_meta.window_title_patterns)
            if exe_matches or title_matches:
                matching_windows.append(w)

        # Fallback for mock tools that only populate output string
        if not matching_windows and not windows and tool_res_windows.output:
            out_lower = tool_res_windows.output.lower()
            if any(exe.lower() in out_lower for exe in app_meta.executable_names) or any(
                pat.lower() in out_lower for pat in app_meta.window_title_patterns
            ):
                matching_windows.append({
                    "title": app_meta.display_name,
                    "process_name": app_meta.executable_names[0],
                    "is_active": "[active]" in out_lower,
                })

        # Filter processes
        procs: List[Dict[str, Any]] = []
        if isinstance(tool_res_procs.data, dict):
            procs = tool_res_procs.data.get("processes", [])

        matching_processes: List[Dict[str, Any]] = []
        for p in procs:
            p_name = str(p.get("name", "")).lower()
            if any(exe.lower() == p_name for exe in app_meta.executable_names):
                matching_processes.append(p)

        # Fallback for mock tools that only populate output string
        if not matching_processes and not procs and tool_res_procs.output:
            out_lower = tool_res_procs.output.lower()
            if any(exe.lower() in out_lower for exe in app_meta.executable_names):
                matching_processes.append({
                    "name": app_meta.executable_names[0],
                    "pid": None,
                    "memory_mb": None,
                })

        # State determination
        has_visible_window = len(matching_windows) > 0
        has_running_process = len(matching_processes) > 0 or has_visible_window
        is_active_window = any(bool(w.get("is_active")) for w in matching_windows)

        titles = [w.get("title", "").strip() for w in matching_windows if w.get("title")]
        first_title = titles[0] if titles else app_meta.display_name

        # Response construction strictly adhering to observation state
        if target_state == "open":
            if has_visible_window:
                if is_active_window:
                    final_response = f'Yes, {app_meta.display_name} is open with an active visible window: "{first_title}".'
                else:
                    final_response = f'Yes, {app_meta.display_name} is open with a visible window: "{first_title}".'
            elif has_running_process:
                final_response = f"No visible {app_meta.display_name} window was found, but {app_meta.display_name} is currently running in the background."
            else:
                final_response = f"No, {app_meta.display_name} is not open (no matching visible window or running process was found)."

        elif target_state == "running":
            if has_visible_window:
                final_response = f'Yes, {app_meta.display_name} is running with a visible window: "{first_title}".'
            elif has_running_process:
                proc_info = matching_processes[0] if matching_processes else None
                pid_str = f" (PID: {proc_info['pid']})" if proc_info and proc_info.get("pid") is not None else ""
                final_response = f"Yes, {app_meta.display_name} is running in the background{pid_str}. No visible window was found."
            else:
                final_response = f"No, {app_meta.display_name} is not running (no matching process or window was found)."

        elif target_state == "active":
            if is_active_window:
                final_response = f'Yes, {app_meta.display_name} is currently the active foreground window: "{first_title}".'
            elif has_visible_window:
                final_response = f'No, {app_meta.display_name} is open with a visible window ("{first_title}"), but it is not the active foreground window.'
            elif has_running_process:
                final_response = f"No, {app_meta.display_name} is running in the background, but has no visible window and is not active."
            else:
                final_response = f"No, {app_meta.display_name} is not active and is not running."
        else:
            return None

        # Build execution trace with both observation steps
        decision = AgentDecision(
            thought_summary=f"Parameterized fast path executed for '{app_meta.display_name}' application state ({target_state}).",
            intent="application_state_inspection",
            steps=[
                StructuredStep(tool="list_open_windows", arguments={}),
                StructuredStep(tool="list_processes", arguments={}),
            ],
            requires_confirmation=False,
            direct_response=final_response,
        )

        step1 = ExecutionStep(
            step_number=1,
            timestamp=start_windows,
            decision=decision,
            action_type=ActionType.TOOL_CALL,
            tool_name="list_open_windows",
            arguments={},
            permission_level=tool_windows.permission_level,
            tool_result=tool_res_windows,
            observation_text=f"[list_open_windows]: {tool_res_windows.output}",
            duration_ms=dur_windows,
            status=ExecutionStatus.COMPLETED,
            error_message=None,
        )

        step2 = ExecutionStep(
            step_number=2,
            timestamp=start_procs,
            decision=decision,
            action_type=ActionType.TOOL_CALL,
            tool_name="list_processes",
            arguments={},
            permission_level=tool_procs.permission_level,
            tool_result=tool_res_procs,
            observation_text=f"[list_processes]: {tool_res_procs.output}",
            duration_ms=dur_procs,
            status=ExecutionStatus.COMPLETED,
            error_message=None,
        )

        trace = ExecutionTrace(
            user_goal=user_query,
            status=ExecutionStatus.COMPLETED,
            start_time=total_start,
            slm_info={
                "provider": "FastPathRouter",
                "model_name": "deterministic_fast_path",
                "is_local": True,
                "llm_calls": 0,
            },
        )
        trace.steps.extend([step1, step2])
        trace.finalize(
            status=ExecutionStatus.COMPLETED,
            final_response=final_response,
            error_message=None,
        )

        return trace

    def _build_app_state_error_trace(
        self,
        user_query: str,
        failing_tool_name: str,
        failing_tool: Any,
        failing_res: Any,
        error_msg: str,
        failing_dur: float,
        prior_tool: Optional[str] = None,
        prior_result: Optional[Any] = None,
        prior_tool_obj: Optional[Any] = None,
        prior_dur: float = 0.0,
    ) -> ExecutionTrace:
        """Construct an error trace when a required application observation fails."""
        trace = ExecutionTrace(
            user_goal=user_query,
            status=ExecutionStatus.ERROR,
            start_time=time.perf_counter(),
            slm_info={
                "provider": "FastPathRouter",
                "model_name": "deterministic_fast_path",
                "is_local": True,
                "llm_calls": 0,
            },
        )
        decision = AgentDecision(
            thought_summary=f"Application-state inspection failed on '{failing_tool_name}'.",
            intent="application_state_inspection",
            steps=[StructuredStep(tool=failing_tool_name, arguments={})],
            requires_confirmation=False,
            direct_response=None,
        )

        step_num = 1
        if prior_tool and prior_result and prior_tool_obj:
            prior_step = ExecutionStep(
                step_number=1,
                timestamp=time.time(),
                decision=decision,
                action_type=ActionType.TOOL_CALL,
                tool_name=prior_tool,
                arguments={},
                permission_level=prior_tool_obj.permission_level,
                tool_result=prior_result,
                observation_text=f"[{prior_tool}]: {prior_result.output}",
                duration_ms=prior_dur,
                status=ExecutionStatus.COMPLETED,
                error_message=None,
            )
            trace.steps.append(prior_step)
            step_num = 2

        failing_step = ExecutionStep(
            step_number=step_num,
            timestamp=time.time(),
            decision=decision,
            action_type=ActionType.TOOL_CALL,
            tool_name=failing_tool_name,
            arguments={},
            permission_level=failing_tool.permission_level,
            tool_result=failing_res,
            observation_text=f"[{failing_tool_name} Error]: {error_msg}",
            duration_ms=failing_dur,
            status=ExecutionStatus.ERROR,
            error_message=error_msg,
        )
        trace.steps.append(failing_step)
        trace.finalize(
            status=ExecutionStatus.ERROR,
            final_response=None,
            error_message=error_msg,
        )
        return trace
