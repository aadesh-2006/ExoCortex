"""
Safe, read-only UI Accessibility inspection tool for Windows using UI Automation.

Inspects accessible controls (buttons, inputs, labels, tree items, menus) of a
specified visible application window using the native Windows UI Automation Client COM
interfaces via standard library ctypes. No synthetic clicks, keystrokes, injections,
or accessibility service installations are performed.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import logging
from typing import Any, Dict, List, Optional, Tuple

from exocortex.tools.base import (
    BaseTool,
    PermissionLevel,
    ToolParameter,
    ToolResult,
)
from exocortex.tools.windows.desktop_windows import (
    _configure_win32_signatures,
    enumerate_desktop_windows,
)

logger = logging.getLogger("exocortex.tools.windows.ui_accessibility")

# Traversal & Output Bounds
MAX_DEPTH_LIMIT = 4
DEFAULT_DEPTH_LIMIT = 3
MAX_ELEMENTS_LIMIT = 50
DEFAULT_ELEMENTS_LIMIT = 25
MAX_PROP_STRING_CHARS = 80
MAX_TEXT_OUTPUT_CHARS = 4000

# COM Concurrency Constants
COINIT_APARTMENTTHREADED = 0x2
RPC_E_CHANGED_MODE = 0x80010106
S_OK = 0
S_FALSE = 1
CLSCTX_INPROC_SERVER = 0x1

# COM GUIDs
CLSID_CUIAutomation_STR = "{ff48dba4-60ef-4201-aa87-54103eef594e}"
IID_IUIAutomation_STR = "{30cbe57d-d9d0-452a-ab13-7ac5ac4825ee}"


class GUID(ctypes.Structure):
    """Win32 GUID representation."""

    _fields_ = [
        ("Data1", wintypes.DWORD),
        ("Data2", wintypes.WORD),
        ("Data3", wintypes.WORD),
        ("Data4", ctypes.c_byte * 8),
    ]


# Mapping of standard UIA ControlType IDs to friendly names (fallback when localized name is missing)
CONTROL_TYPE_NAMES: Dict[int, str] = {
    50000: "button",
    50001: "calendar",
    50002: "check box",
    50003: "combo box",
    50004: "edit",
    50005: "hyperlink",
    50006: "image",
    50007: "list item",
    50008: "list",
    50009: "menu",
    50010: "menu bar",
    50011: "menu item",
    50012: "progress bar",
    50013: "radio button",
    50014: "scroll bar",
    50015: "slider",
    50016: "spinner",
    50017: "status bar",
    50018: "tab",
    50019: "tab item",
    50020: "text",
    50021: "tool bar",
    50022: "tool tip",
    50023: "tree",
    50024: "tree item",
    50025: "custom",
    50026: "group",
    50027: "thumb",
    50028: "data grid",
    50029: "data item",
    50030: "document",
    50031: "split button",
    50032: "window",
    50033: "pane",
    50034: "header",
    50035: "header item",
    50036: "table",
    50037: "title bar",
    50038: "separator",
    50039: "semantic zoom",
    50040: "app bar",
}


def _safe_release(ptr: Optional[ctypes.c_void_p]) -> None:
    """Safely invoke IUnknown::Release on a COM interface pointer."""
    if not ptr or not ptr.value:
        return
    try:
        vtbl = ctypes.cast(ptr, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        fn_release = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(vtbl[2])
        fn_release(ptr)
    except Exception as err:
        logger.debug("Error releasing COM pointer %s: %s", ptr, err)


def _read_and_free_bstr(bstr_val: Optional[int]) -> str:
    """Read a BSTR into a Python string and immediately free it via SysFreeString."""
    if not bstr_val:
        return ""
    try:
        return ctypes.wstring_at(bstr_val)
    except Exception:
        return ""
    finally:
        try:
            oleaut32 = ctypes.windll.oleaut32
            oleaut32.SysFreeString.argtypes = [ctypes.c_void_p]
            oleaut32.SysFreeString(ctypes.c_void_p(bstr_val))
        except Exception:
            pass


def resolve_target_window(
    target_title: Optional[str] = None,
    process_name: Optional[str] = None,
    candidate_windows: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Dict[str, Any]]:
    """
    Safely resolve a target desktop window using validated enumeration.

    Returns:
        (matched_window_dict, error_message, metadata)
    """
    clean_title = (target_title or "").strip()
    clean_proc = (process_name or "").strip()

    if not clean_title and not clean_proc:
        return (
            None,
            "At least one of 'target_title' or 'process_name' must be provided.",
            {"status": "invalid_arguments", "matched_count": 0},
        )

    if candidate_windows is None:
        try:
            candidate_windows = enumerate_desktop_windows(limit=100, include_handles=True)
        except Exception as e:
            return (
                None,
                f"Failed to enumerate open windows: {e}",
                {"status": "enumeration_error", "matched_count": 0},
            )

    t_lower = clean_title.lower() if clean_title else ""
    p_lower = clean_proc.lower() if clean_proc else ""
    p_base = p_lower.removesuffix(".exe") if p_lower else ""

    matches: List[Dict[str, Any]] = []
    for win in candidate_windows:
        title_matches = True
        proc_matches = True

        if t_lower:
            win_title = win.get("title", "").lower()
            title_matches = t_lower in win_title

        if p_lower:
            win_proc = win.get("process_name", "").lower()
            win_proc_base = win_proc.removesuffix(".exe")
            proc_matches = (
                p_lower == win_proc
                or p_base == win_proc_base
                or (p_base in win_proc_base and len(p_base) >= 3)
            )

        if title_matches and proc_matches:
            matches.append(win)

    if not matches:
        return (
            None,
            f"No visible application window found matching title='{clean_title}' and process='{clean_proc}'.",
            {"status": "not_found", "matched_count": 0, "matches": []},
        )

    if len(matches) > 1:
        match_summaries = [
            f"\"{m.get('title')}\" ({m.get('process_name')})"
            for m in matches[:5]
        ]
        summary_str = ", ".join(match_summaries)
        if len(matches) > 5:
            summary_str += f", and {len(matches) - 5} more"
        return (
            None,
            f"Ambiguous window match: found {len(matches)} matching windows: {summary_str}. "
            f"Please specify a more specific target_title or process_name.",
            {
                "status": "ambiguous",
                "matched_count": len(matches),
                "matches": [
                    {"title": m.get("title"), "process_name": m.get("process_name")}
                    for m in matches
                ],
            },
        )

    return (matches[0], None, {"status": "found", "matched_count": 1})


class COMSession:
    """Manages thread COM apartment initialization and desktop accessibility."""

    def __init__(self) -> None:
        self.should_uninitialize: bool = False
        self._switched_desktop: bool = False
        self._h_desk: Optional[int] = None
        self._orig_desk: Optional[int] = None

    def __enter__(self) -> COMSession:
        _configure_win32_signatures()
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32

        user32.GetThreadDesktop.restype = wintypes.HDESK
        user32.GetThreadDesktop.argtypes = [wintypes.DWORD]
        kernel32.GetCurrentThreadId.restype = wintypes.DWORD
        kernel32.GetCurrentThreadId.argtypes = []
        user32.GetUserObjectInformationW.restype = wintypes.BOOL
        user32.GetUserObjectInformationW.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ]
        user32.OpenDesktopW.restype = wintypes.HDESK
        user32.OpenDesktopW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        user32.SetThreadDesktop.restype = wintypes.BOOL
        user32.SetThreadDesktop.argtypes = [wintypes.HDESK]
        user32.CloseDesktop.restype = wintypes.BOOL
        user32.CloseDesktop.argtypes = [wintypes.HDESK]

        # Check if the thread is already on the interactive Default desktop
        UOI_NAME = 2
        try:
            tid = kernel32.GetCurrentThreadId()
            self._orig_desk = user32.GetThreadDesktop(tid)
            buf = ctypes.create_unicode_buffer(256)
            needed = wintypes.DWORD(0)
            if user32.GetUserObjectInformationW(self._orig_desk, UOI_NAME, buf, ctypes.sizeof(buf), ctypes.byref(needed)):
                current_desk_name = buf.value.strip().lower()
            else:
                current_desk_name = ""

            if current_desk_name != "default":
                # Background runner or service desktop: safely switch to interactive Default desktop
                h_desk = user32.OpenDesktopW("Default", 0, False, 0x01FF)
                if h_desk:
                    if user32.SetThreadDesktop(h_desk):
                        self._switched_desktop = True
                        self._h_desk = h_desk
                    else:
                        # Could not switch desktop (e.g. thread already has windows or hooks)
                        user32.CloseDesktop(h_desk)
        except Exception as e:
            logger.debug("Desktop inspection/switching skipped or failed: %s", e)

        ole32 = ctypes.windll.ole32
        ole32.CoInitializeEx.argtypes = [ctypes.c_void_p, wintypes.DWORD]
        ole32.CoInitializeEx.restype = ctypes.c_long

        hr = ole32.CoInitializeEx(None, COINIT_APARTMENTTHREADED)
        hr_u = hr & 0xFFFFFFFF

        if hr in (S_OK, S_FALSE):
            self.should_uninitialize = True
        elif hr_u == RPC_E_CHANGED_MODE:
            # Thread is already in MTA mode; do NOT call CoUninitialize
            self.should_uninitialize = False
        else:
            # If CoInitializeEx failed and we switched desktop, restore original desktop immediately
            if self._switched_desktop and self._orig_desk:
                try:
                    user32.SetThreadDesktop(self._orig_desk)
                except Exception:
                    pass
                if self._h_desk:
                    try:
                        user32.CloseDesktop(self._h_desk)
                    except Exception:
                        pass
                self._switched_desktop = False
            raise RuntimeError(f"CoInitializeEx failed with HRESULT 0x{hr_u:08X}")

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        user32 = ctypes.windll.user32

        # 1. Uninitialize COM first while still in the apartment
        if self.should_uninitialize:
            try:
                ctypes.windll.ole32.CoUninitialize()
            except Exception as e:
                logger.debug("CoUninitialize error: %s", e)

        # 2. Restore original thread desktop and close opened desktop handle
        if self._switched_desktop:
            if self._orig_desk:
                try:
                    user32.SetThreadDesktop(self._orig_desk)
                except Exception as e:
                    logger.debug("Failed to restore original thread desktop: %s", e)
            if self._h_desk:
                try:
                    user32.CloseDesktop(self._h_desk)
                except Exception as e:
                    logger.debug("Failed to close desktop handle: %s", e)
            self._switched_desktop = False



class InspectWindowUIElementsTool(BaseTool):
    """Safe, read-only tool to inspect accessible UI elements of an application window."""

    name: str = "inspect_window_ui_elements"
    description: str = (
        "Inspects the accessible UI hierarchy (controls, buttons, inputs, labels) of a "
        "specified visible desktop application window using Windows UI Automation. "
        "Read-only inspection only."
    )
    permission_level: PermissionLevel = PermissionLevel.SAFE

    parameters: List[ToolParameter] = [
        ToolParameter(
            name="target_title",
            type="string",
            description="Substring of the target window title to match.",
            required=False,
        ),
        ToolParameter(
            name="process_name",
            type="string",
            description="Name of the application process to match (e.g. 'notepad.exe', 'code.exe').",
            required=False,
        ),
        ToolParameter(
            name="max_depth",
            type="integer",
            description=f"Maximum child element depth to traverse (1 to {MAX_DEPTH_LIMIT}, default: {DEFAULT_DEPTH_LIMIT}).",
            required=False,
            default=DEFAULT_DEPTH_LIMIT,
        ),
        ToolParameter(
            name="max_elements",
            type="integer",
            description=f"Maximum number of UI elements to return (1 to {MAX_ELEMENTS_LIMIT}, default: {DEFAULT_ELEMENTS_LIMIT}).",
            required=False,
            default=DEFAULT_ELEMENTS_LIMIT,
        ),
    ]

    def _extract_element_data(
        self,
        elem: ctypes.c_void_p,
        depth: int,
    ) -> Dict[str, Any]:
        """Extract privacy-conscious accessibility properties from a UI element."""
        ev = ctypes.cast(elem, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents

        # Vtable method prototypes
        fn_get_type_id = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int))(ev[21])
        fn_get_type_name = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))(ev[22])
        fn_get_name = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))(ev[23])
        fn_get_focusable = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(wintypes.BOOL))(ev[27])
        fn_get_enabled = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(wintypes.BOOL))(ev[28])
        fn_get_auto_id = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))(ev[29])
        fn_get_class_name = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))(ev[30])
        fn_get_is_password = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(wintypes.BOOL))(ev[35])
        fn_get_is_offscreen = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(wintypes.BOOL))(ev[38])

        # 1. Password check (Critical Privacy)
        b_pass = wintypes.BOOL(0)
        fn_get_is_password(elem, ctypes.byref(b_pass))
        is_password = bool(b_pass.value)

        # 2. Control Type
        b_tname = ctypes.c_void_p()
        fn_get_type_name(elem, ctypes.byref(b_tname))
        control_type = _read_and_free_bstr(b_tname.value).strip().lower()
        if not control_type:
            c_tid = ctypes.c_int(0)
            fn_get_type_id(elem, ctypes.byref(c_tid))
            control_type = CONTROL_TYPE_NAMES.get(c_tid.value, "unknown")

        # 3. Name (Redacted if password)
        b_name = ctypes.c_void_p()
        fn_get_name(elem, ctypes.byref(b_name))
        raw_name = _read_and_free_bstr(b_name.value).strip()
        if is_password:
            name = "[REDACTED_PASSWORD]"
        else:
            name = raw_name
            if len(name) > MAX_PROP_STRING_CHARS:
                name = name[:MAX_PROP_STRING_CHARS].rstrip() + "..."

        # 4. Automation ID (Redacted if password)
        b_aid = ctypes.c_void_p()
        fn_get_auto_id(elem, ctypes.byref(b_aid))
        raw_aid = _read_and_free_bstr(b_aid.value).strip()
        if is_password:
            automation_id = "[REDACTED_PASSWORD]"
        else:
            automation_id = raw_aid
            if len(automation_id) > MAX_PROP_STRING_CHARS:
                automation_id = automation_id[:MAX_PROP_STRING_CHARS].rstrip() + "..."

        # 5. Class Name
        b_cname = ctypes.c_void_p()
        fn_get_class_name(elem, ctypes.byref(b_cname))
        class_name = _read_and_free_bstr(b_cname.value).strip()
        if len(class_name) > MAX_PROP_STRING_CHARS:
            class_name = class_name[:MAX_PROP_STRING_CHARS].rstrip() + "..."

        # 6. Boolean Flags
        b_enabled = wintypes.BOOL(1)
        fn_get_enabled(elem, ctypes.byref(b_enabled))
        is_enabled = bool(b_enabled.value)

        b_focus = wintypes.BOOL(0)
        fn_get_focusable(elem, ctypes.byref(b_focus))
        is_focusable = bool(b_focus.value)

        b_offscreen = wintypes.BOOL(0)
        fn_get_is_offscreen(elem, ctypes.byref(b_offscreen))
        is_offscreen = bool(b_offscreen.value)

        # Explicitly omit raw HWND and pointer details to protect abstraction boundary
        return {
            "depth": depth,
            "name": name,
            "control_type": control_type,
            "automation_id": automation_id,
            "class_name": class_name,
            "is_enabled": is_enabled,
            "is_keyboard_focusable": is_focusable,
            "is_password": is_password,
            "is_offscreen": is_offscreen,
        }

    def _traverse_control_tree(
        self,
        hwnd: int,
        max_depth: int,
        max_elements: int,
    ) -> List[Dict[str, Any]]:
        """Perform bounded Control View traversal of the target window element tree."""
        ole32 = ctypes.windll.ole32

        clsid_auto = GUID()
        iid_auto = GUID()
        ole32.CLSIDFromString(CLSID_CUIAutomation_STR, ctypes.byref(clsid_auto))
        ole32.IIDFromString(IID_IUIAutomation_STR, ctypes.byref(iid_auto))

        p_auto = ctypes.c_void_p()
        hr = ole32.CoCreateInstance(
            ctypes.byref(clsid_auto),
            None,
            CLSCTX_INPROC_SERVER,
            ctypes.byref(iid_auto),
            ctypes.byref(p_auto),
        )
        if hr != 0 or not p_auto.value:
            raise RuntimeError(f"Failed to create CUIAutomation instance: HRESULT 0x{hr & 0xFFFFFFFF:08X}")

        root_elem = ctypes.c_void_p()
        walker = ctypes.c_void_p()

        try:
            vtbl_auto = ctypes.cast(p_auto, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
            fn_elem_from_handle = ctypes.WINFUNCTYPE(
                ctypes.c_long, ctypes.c_void_p, wintypes.HWND, ctypes.POINTER(ctypes.c_void_p)
            )(vtbl_auto[6])
            fn_get_walker = ctypes.WINFUNCTYPE(
                ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
            )(vtbl_auto[14])

            hr_root = fn_elem_from_handle(p_auto, hwnd, ctypes.byref(root_elem))
            if hr_root != 0 or not root_elem.value:
                raise RuntimeError(
                    f"ElementFromHandle failed for HWND 0x{hwnd:X}: HRESULT 0x{hr_root & 0xFFFFFFFF:08X}"
                )

            hr_w = fn_get_walker(p_auto, ctypes.byref(walker))
            if hr_w != 0 or not walker.value:
                raise RuntimeError(
                    f"get_ControlViewWalker failed: HRESULT 0x{hr_w & 0xFFFFFFFF:08X}"
                )

            vtbl_walker = ctypes.cast(walker, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
            fn_get_first_child = ctypes.WINFUNCTYPE(
                ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
            )(vtbl_walker[4])
            fn_get_next_sibling = ctypes.WINFUNCTYPE(
                ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
            )(vtbl_walker[6])

            collected: List[Dict[str, Any]] = []

            def traverse_recursive(current: ctypes.c_void_p, depth: int) -> None:
                if len(collected) >= max_elements:
                    return

                elem_data = self._extract_element_data(current, depth)
                collected.append(elem_data)

                if depth >= max_depth or len(collected) >= max_elements:
                    return

                first_child = ctypes.c_void_p()
                hr_fc = fn_get_first_child(walker, current, ctypes.byref(first_child))
                if hr_fc != 0 or not first_child.value:
                    return

                curr_child = first_child
                while curr_child.value:
                    if len(collected) >= max_elements:
                        _safe_release(curr_child)
                        break

                    traverse_recursive(curr_child, depth + 1)

                    next_sib = ctypes.c_void_p()
                    hr_sib = fn_get_next_sibling(walker, curr_child, ctypes.byref(next_sib))
                    _safe_release(curr_child)
                    if hr_sib != 0 or not next_sib.value:
                        break
                    curr_child = next_sib

            traverse_recursive(root_elem, 0)
            return collected

        finally:
            _safe_release(root_elem)
            _safe_release(walker)
            _safe_release(p_auto)

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute read-only UI Accessibility inspection."""
        target_title = kwargs.get("target_title")
        process_name = kwargs.get("process_name")

        # Clamp bounds
        try:
            depth_raw = int(kwargs.get("max_depth", DEFAULT_DEPTH_LIMIT))
            max_depth = max(1, min(depth_raw, MAX_DEPTH_LIMIT))
        except (ValueError, TypeError):
            max_depth = DEFAULT_DEPTH_LIMIT

        try:
            elements_raw = int(kwargs.get("max_elements", DEFAULT_ELEMENTS_LIMIT))
            max_elements = max(1, min(elements_raw, MAX_ELEMENTS_LIMIT))
        except (ValueError, TypeError):
            max_elements = DEFAULT_ELEMENTS_LIMIT

        # 1. Resolve target window
        matched_win, error_msg, meta = resolve_target_window(
            target_title=target_title,
            process_name=process_name,
        )

        if error_msg or not matched_win:
            return ToolResult(
                success=False,
                output="",
                error=error_msg or "Failed to resolve target window.",
                data=meta,
            )

        hwnd = matched_win.get("hwnd")
        if not hwnd:
            return ToolResult(
                success=False,
                output="",
                error=f"No valid HWND available for window '{matched_win.get('title')}'.",
                data={"status": "invalid_handle"},
            )

        # 2. Execute bounded inspection in safe COM apartment session
        try:
            with COMSession():
                elements = self._traverse_control_tree(
                    hwnd=hwnd,
                    max_depth=max_depth,
                    max_elements=max_elements,
                )
        except Exception as e:
            logger.error("UI Automation inspection failed: %s", e)
            return ToolResult(
                success=False,
                output="",
                error=f"UI Automation inspection failed for '{matched_win.get('title')}': {e}",
                data={"status": "inspection_failed"},
            )

        if not elements:
            return ToolResult(
                success=True,
                output=f"No accessible UI elements found for window \"{matched_win.get('title')}\".",
                data={
                    "status": "success",
                    "target_window": {
                        "title": matched_win.get("title"),
                        "process_name": matched_win.get("process_name"),
                    },
                    "total_elements": 0,
                    "elements": [],
                },
            )

        # 3. Format privacy-safe hierarchical textual representation
        lines = [
            f"UI Accessibility Elements for \"{matched_win.get('title')}\" "
            f"({matched_win.get('process_name')}) [{len(elements)} elements found]:"
        ]
        for el in elements:
            indent = "  " * el["depth"]
            status_flags = []
            if el["is_enabled"]:
                status_flags.append("enabled")
            else:
                status_flags.append("disabled")
            if el["is_keyboard_focusable"]:
                status_flags.append("focusable")
            if el["is_password"]:
                status_flags.append("password")
            if el["is_offscreen"]:
                status_flags.append("offscreen")

            flags_str = f" [{', '.join(status_flags)}]" if status_flags else ""
            auto_id_str = f" (id: \"{el['automation_id']}\")" if el["automation_id"] else ""
            name_str = f" \"{el['name']}\"" if el["name"] else ""
            line = f"{indent}- [{el['control_type']}]{name_str}{auto_id_str}{flags_str}"
            lines.append(line)

        output_text = "\n".join(lines)
        if len(output_text) > MAX_TEXT_OUTPUT_CHARS:
            output_text = output_text[:MAX_TEXT_OUTPUT_CHARS].rstrip() + "\n... [output truncated]"

        return ToolResult(
            success=True,
            output=output_text,
            data={
                "status": "success",
                "target_window": {
                    "title": matched_win.get("title"),
                    "process_name": matched_win.get("process_name"),
                },
                "total_elements": len(elements),
                "elements": elements,
            },
        )
