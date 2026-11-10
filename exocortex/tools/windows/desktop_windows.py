"""
Safe, read-only Windows desktop window inspection tool for ExoCortex.

Enumerates visible, interactive top-level application windows using Win32 APIs
via standard library ctypes, filtering out hidden, cloaked, zero-size, and empty windows.
No window manipulation, clicking, or injection is performed.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import logging
from typing import Any, Dict, List, Optional

import psutil

from exocortex.tools.base import (
    BaseTool,
    PermissionLevel,
    ToolParameter,
    ToolResult,
)

logger = logging.getLogger("exocortex.tools.windows.desktop_windows")

MAX_WINDOW_TITLE_CHARS = 100
MAX_RETURNED_WINDOWS = 15
DEFAULT_WINDOW_LIMIT = 15

# Win32 DWM cloaked window attribute
DWMWA_CLOAKED = 14
DESKTOP_ENUMERATE = 0x0040
DESKTOP_READOBJECTS = 0x0001

WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)


def _configure_win32_signatures() -> None:
    """Explicitly configure ctypes argument and return types for 64-bit safety."""
    user32 = ctypes.windll.user32

    # GetForegroundWindow returns HWND (pointer-sized handle, not signed 32-bit int)
    user32.GetForegroundWindow.argtypes = []
    user32.GetForegroundWindow.restype = wintypes.HWND

    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.restype = wintypes.BOOL

    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextLengthW.restype = ctypes.c_int

    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowTextW.restype = ctypes.c_int

    user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.GetWindowRect.restype = wintypes.BOOL

    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD

    # OpenDesktopW / CloseDesktop / EnumDesktopWindows
    user32.OpenDesktopW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    user32.OpenDesktopW.restype = wintypes.HDESK

    user32.CloseDesktop.argtypes = [wintypes.HDESK]
    user32.CloseDesktop.restype = wintypes.BOOL

    user32.EnumDesktopWindows.argtypes = [wintypes.HDESK, WNDENUMPROC, wintypes.LPARAM]
    user32.EnumDesktopWindows.restype = wintypes.BOOL

    user32.EnumWindows.argtypes = [WNDENUMPROC, wintypes.LPARAM]
    user32.EnumWindows.restype = wintypes.BOOL

    dwmapi = getattr(ctypes.windll, "dwmapi", None)
    if dwmapi and hasattr(dwmapi, "DwmGetWindowAttribute"):
        dwmapi.DwmGetWindowAttribute.argtypes = [
            wintypes.HWND,
            wintypes.DWORD,
            wintypes.LPCVOID,
            wintypes.DWORD,
        ]
        dwmapi.DwmGetWindowAttribute.restype = ctypes.c_long


try:
    _configure_win32_signatures()
except Exception as _sig_err:
    logger.debug("Failed to configure Win32 signatures at module import: %s", _sig_err)


def enumerate_desktop_windows(
    limit: int = DEFAULT_WINDOW_LIMIT,
    include_handles: bool = False,
) -> List[Dict[str, Any]]:
    """
    Enumerate top-level application windows via Win32 APIs.
    Safely filters out hidden, cloaked, zero-size, and background shell windows.

    Args:
        limit: Maximum number of windows to return.
        include_handles: If True, include internal 'hwnd' in the returned dictionary.
    """
    _configure_win32_signatures()
    user32 = ctypes.windll.user32
    dwmapi = getattr(ctypes.windll, "dwmapi", None)

    foreground_hwnd = user32.GetForegroundWindow()
    raw_windows: List[Dict[str, Any]] = []

    def enum_cb(hwnd: int, lparam: int) -> int:
        if len(raw_windows) >= limit * 2:
            # Bounded early exit
            return 1

        try:
            # 1. Visibility check
            if not user32.IsWindowVisible(hwnd):
                return 1

            # 2. Window title check
            length = user32.GetWindowTextLengthW(hwnd)
            if length <= 0:
                return 1

            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            title = buf.value.strip()
            if not title or title == "Program Manager":
                return 1

            # 3. Non-zero dimensions check
            rect = wintypes.RECT()
            if user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                width = rect.right - rect.left
                height = rect.bottom - rect.top
                if width <= 0 or height <= 0:
                    return 1

            # 4. Cloaked window check (Windows 8+ DWM)
            if dwmapi:
                is_cloaked = wintypes.DWORD(0)
                hr = dwmapi.DwmGetWindowAttribute(
                    hwnd,
                    wintypes.DWORD(DWMWA_CLOAKED),
                    ctypes.byref(is_cloaked),
                    ctypes.sizeof(is_cloaked),
                )
                if hr == 0 and is_cloaked.value != 0:
                    return 1

            # 5. Process identification
            pid = wintypes.DWORD(0)
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            proc_name = "unknown"
            if pid.value:
                try:
                    p = psutil.Process(pid.value)
                    proc_name = p.name()
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    proc_name = "unknown"
                except Exception:
                    proc_name = "unknown"

            is_active = bool(foreground_hwnd is not None and hwnd == foreground_hwnd)

            # Truncate title
            clean_title = title
            if len(clean_title) > MAX_WINDOW_TITLE_CHARS:
                clean_title = clean_title[:MAX_WINDOW_TITLE_CHARS].rstrip() + "..."

            win_record: Dict[str, Any] = {
                "title": clean_title,
                "process_name": proc_name,
                "is_active": is_active,
            }
            if include_handles:
                win_record["hwnd"] = hwnd

            raw_windows.append(win_record)
        except Exception as item_err:
            logger.debug("Error processing window %s: %s", hwnd, item_err)

        return 1

    cb = WNDENUMPROC(enum_cb)

    # Attempt to access interactive Default desktop if available (e.g. running in background thread desktop)
    h_desk = None
    try:
        h_desk = user32.OpenDesktopW("Default", 0, False, DESKTOP_ENUMERATE | DESKTOP_READOBJECTS)
    except Exception:
        h_desk = None

    try:
        if h_desk:
            user32.EnumDesktopWindows(h_desk, cb, 0)
        else:
            user32.EnumWindows(cb, 0)
    finally:
        if h_desk:
            try:
                user32.CloseDesktop(h_desk)
            except Exception:
                pass

    # Sort so active/foreground window appears first, then by title
    raw_windows.sort(key=lambda w: (not w["is_active"], w["title"].lower()))
    return raw_windows[:limit]




class ListOpenWindowsTool(BaseTool):
    """Safe, read-only tool to enumerate visible desktop application windows."""

    name: str = "list_open_windows"
    description: str = (
        "Inspects and lists currently visible desktop application windows with window title, "
        "process name, and active foreground status. Read-only inspection only."
    )
    permission_level: PermissionLevel = PermissionLevel.SAFE

    parameters: List[ToolParameter] = [
        ToolParameter(
            name="limit",
            type="integer",
            description=f"Maximum number of windows to return (1 to {MAX_RETURNED_WINDOWS}, default: {DEFAULT_WINDOW_LIMIT}).",
            required=False,
            default=DEFAULT_WINDOW_LIMIT,
        ),
    ]

    def _enumerate_windows(self, limit: int) -> List[Dict[str, Any]]:
        """
        Enumerate top-level application windows via Win32 APIs.
        Safely filters out hidden, cloaked, zero-size, and background shell windows.
        """
        return enumerate_desktop_windows(limit=limit, include_handles=False)


    def run(self, **kwargs: Any) -> ToolResult:
        """Execute open window enumeration."""
        limit_raw = kwargs.get("limit", DEFAULT_WINDOW_LIMIT)
        try:
            limit = int(limit_raw)
            limit = max(1, min(limit, MAX_RETURNED_WINDOWS))
        except (ValueError, TypeError):
            limit = DEFAULT_WINDOW_LIMIT

        try:
            windows = self._enumerate_windows(limit=limit)
        except Exception as e:
            logger.error("Failed to enumerate desktop windows: %s", e)
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to enumerate open desktop windows: {e}",
            )

        if not windows:
            return ToolResult(
                success=True,
                output="No visible application windows found on the desktop.",
                data={"returned_count": 0, "windows": []},
            )

        # Format readable output list
        lines = [f"Visible Desktop Windows ({len(windows)} found):"]
        for idx, win in enumerate(windows, start=1):
            active_tag = " [ACTIVE]" if win["is_active"] else ""
            lines.append(f"{idx}.{active_tag} \"{win['title']}\" ({win['process_name']})")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={
                "returned_count": len(windows),
                "windows": windows,
            },
        )
