"""Desktop region helpers for primary-monitor and virtual-desktop modes."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

import win32con

user32 = ctypes.windll.user32

_SM_XVIRTUALSCREEN = getattr(win32con, "SM_XVIRTUALSCREEN", 76)
_SM_YVIRTUALSCREEN = getattr(win32con, "SM_YVIRTUALSCREEN", 77)
_SM_CXVIRTUALSCREEN = getattr(win32con, "SM_CXVIRTUALSCREEN", 78)
_SM_CYVIRTUALSCREEN = getattr(win32con, "SM_CYVIRTUALSCREEN", 79)


@dataclass(frozen=True)
class DesktopRegion:
    """A capture or input region on the Windows desktop."""

    left: int
    top: int
    width: int
    height: int
    target: str

    @property
    def right(self) -> int:
        return self.left + max(self.width - 1, 0)

    @property
    def bottom(self) -> int:
        return self.top + max(self.height - 1, 0)

    @property
    def is_virtual(self) -> bool:
        return self.target == "virtual-desktop"


def get_desktop_region(capture_target: str) -> DesktopRegion:
    """Resolve the configured desktop region for capture and input."""
    normalized = (capture_target or "primary-monitor").strip().casefold()
    if normalized == "virtual-desktop":
        return DesktopRegion(
            left=user32.GetSystemMetrics(_SM_XVIRTUALSCREEN),
            top=user32.GetSystemMetrics(_SM_YVIRTUALSCREEN),
            width=user32.GetSystemMetrics(_SM_CXVIRTUALSCREEN),
            height=user32.GetSystemMetrics(_SM_CYVIRTUALSCREEN),
            target="virtual-desktop",
        )
    return DesktopRegion(
        left=0,
        top=0,
        width=user32.GetSystemMetrics(win32con.SM_CXSCREEN),
        height=user32.GetSystemMetrics(win32con.SM_CYSCREEN),
        target="primary-monitor",
    )