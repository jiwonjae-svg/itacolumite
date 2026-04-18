"""Window information collection via win32gui."""

from __future__ import annotations

import logging
from dataclasses import dataclass

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
