"""Window information collection via win32gui."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import win32con
import win32gui
import win32process

logger = logging.getLogger(__name__)


@dataclass
class WindowInfo:
    """단일 윈도우 정보."""

    hwnd: int
    title: str
    class_name: str
    rect: tuple[int, int, int, int]  # (left, top, right, bottom)
    pid: int


def get_foreground_window() -> WindowInfo:
    """현재 활성 윈도우 정보 반환."""
    hwnd = win32gui.GetForegroundWindow()
    _, pid = win32process.GetWindowThreadProcessId(hwnd)
    return WindowInfo(
        hwnd=hwnd,
        title=win32gui.GetWindowText(hwnd),
        class_name=win32gui.GetClassName(hwnd),
        rect=win32gui.GetWindowRect(hwnd),
        pid=pid,
    )


def list_visible_windows() -> list[WindowInfo]:
    """화면에 보이는 모든 윈도우 목록 반환."""
    windows: list[WindowInfo] = []

    def _callback(hwnd: int, _: object) -> bool:
        if not win32gui.IsWindowVisible(hwnd):
            return True
        title = win32gui.GetWindowText(hwnd)
        if not title:
            return True
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        windows.append(
            WindowInfo(
                hwnd=hwnd,
                title=title,
                class_name=win32gui.GetClassName(hwnd),
                rect=win32gui.GetWindowRect(hwnd),
                pid=pid,
            )
        )
        return True

    win32gui.EnumWindows(_callback, None)
    return windows


def find_window_by_title(keyword: str) -> WindowInfo | None:
    """제목에 keyword가 포함된 첫 번째 가시 윈도우 반환."""
    keyword_lower = keyword.lower()
    for w in list_visible_windows():
        if keyword_lower in w.title.lower():
            return w
    return None


def find_window(
    *,
    hwnd: int | None = None,
    pid: int | None = None,
    class_name: str | None = None,
    title: str | None = None,
) -> WindowInfo | None:
    """Find a visible window that matches a stable signature."""
    title_lower = title.lower() if title else None
    for window in list_visible_windows():
        if hwnd is not None and window.hwnd == hwnd:
            return window
        if pid is not None and window.pid != pid:
            continue
        if class_name is not None and window.class_name != class_name:
            continue
        if title_lower is not None and title_lower not in window.title.lower():
            continue
        return window
    return None


def activate_window(window: WindowInfo) -> bool:
    """Try to bring a visible window to the foreground."""
    try:
        win32gui.ShowWindow(window.hwnd, win32con.SW_RESTORE)
        win32gui.BringWindowToTop(window.hwnd)
        win32gui.SetForegroundWindow(window.hwnd)
    except Exception as exc:
        logger.debug("Failed to activate window %s: %s", window.title, exc)
        return False
    return True


def find_child_window(
    parent_hwnd: int,
    *,
    class_name: str | None = None,
    title: str | None = None,
) -> WindowInfo | None:
    """Find a descendant window under a visible parent by class or title."""
    title_lower = title.lower() if title else None
    matches: list[WindowInfo] = []

    def _callback(hwnd: int, _: object) -> bool:
        if not win32gui.IsWindowVisible(hwnd):
            return True
        child_title = win32gui.GetWindowText(hwnd)
        child_class = win32gui.GetClassName(hwnd)
        if class_name is not None and child_class != class_name:
            return True
        if title_lower is not None and title_lower not in child_title.lower():
            return True
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        matches.append(
            WindowInfo(
                hwnd=hwnd,
                title=child_title,
                class_name=child_class,
                rect=win32gui.GetWindowRect(hwnd),
                pid=pid,
            )
        )
        return False

    try:
        win32gui.EnumChildWindows(parent_hwnd, _callback, None)
    except Exception as exc:
        logger.debug("Failed to enumerate child windows for %s: %s", parent_hwnd, exc)
        return None
    return matches[0] if matches else None
