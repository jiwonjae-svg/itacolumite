"""Mouse control via Windows SendInput API."""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import logging
import time

from itacolumite.config.settings import get_settings
from itacolumite.perception.display import DesktopRegion, get_desktop_region

logger = logging.getLogger(__name__)

user32 = ctypes.windll.user32

# --- SendInput 구조체 정의 ---

INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_HWHEEL = 0x1000
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_VIRTUALDESK = 0x4000

WHEEL_DELTA = 120


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.wintypes.LONG),
        ("dy", ctypes.wintypes.LONG),
        ("mouseData", ctypes.wintypes.DWORD),
        ("dwFlags", ctypes.wintypes.DWORD),
        ("time", ctypes.wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class INPUT(ctypes.Structure):
    class _INPUT_UNION(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT)]

    _anonymous_ = ("_union",)
    _fields_ = [
        ("type", ctypes.wintypes.DWORD),
        ("_union", _INPUT_UNION),
    ]


def _send_mouse(flags: int, dx: int = 0, dy: int = 0, data: int = 0) -> None:
    inp = INPUT()
    inp.type = INPUT_MOUSE
    inp.mi.dx = dx
    inp.mi.dy = dy
    inp.mi.mouseData = data
    inp.mi.dwFlags = flags
    inp.mi.time = 0
    inp.mi.dwExtraInfo = ctypes.pointer(ctypes.c_ulong(0))
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))


def _to_abs(x: int, y: int, region: DesktopRegion) -> tuple[int, int]:
    """Desktop pixel 좌표를 SendInput 정규화 좌표(0-65535)로 변환한다.

    화면 범위를 벗어나는 좌표는 경계로 클램프하고 경고를 남긴다.
    """
    clamped_x = max(region.left, min(x, region.right))
    clamped_y = max(region.top, min(y, region.bottom))
    if clamped_x != x or clamped_y != y:
        logger.warning(
            "Coordinate out of bounds: (%d, %d) clamped to (%d, %d) [region (%d,%d) %dx%d]",
            x, y, clamped_x, clamped_y, region.left, region.top, region.width, region.height,
        )
    abs_x = int((clamped_x - region.left) * 65535 / max(region.width - 1, 1))
    abs_y = int((clamped_y - region.top) * 65535 / max(region.height - 1, 1))
    return abs_x, abs_y


def _absolute_flags(region: DesktopRegion, *, include_move: bool) -> int:
    flags = MOUSEEVENTF_ABSOLUTE
    if include_move:
        flags |= MOUSEEVENTF_MOVE
    if region.is_virtual:
        flags |= MOUSEEVENTF_VIRTUALDESK
    return flags


class MouseController:
    """SendInput 기반 마우스 제어."""

    def __init__(self) -> None:
        settings = get_settings()
        self._action_delay = settings.agent.action_delay_ms / 1000.0
        self._region = get_desktop_region(settings.agent.capture_target)

    def click(self, x: int, y: int, button: str = "left") -> None:
        ax, ay = _to_abs(x, y, self._region)
        down, up = _button_flags(button)
        absolute_move = _absolute_flags(self._region, include_move=True)
        absolute_button = _absolute_flags(self._region, include_move=False)
        _send_mouse(absolute_move, ax, ay)
        _send_mouse(down | absolute_button, ax, ay)
        _send_mouse(up | absolute_button, ax, ay)
        time.sleep(self._action_delay)
        logger.debug("click(%d, %d, %s)", x, y, button)

    def double_click(self, x: int, y: int) -> None:
        ax, ay = _to_abs(x, y, self._region)
        absolute_move = _absolute_flags(self._region, include_move=True)
        absolute_button = _absolute_flags(self._region, include_move=False)
        _send_mouse(absolute_move, ax, ay)
        for _ in range(2):
            _send_mouse(MOUSEEVENTF_LEFTDOWN | absolute_button, ax, ay)
            _send_mouse(MOUSEEVENTF_LEFTUP | absolute_button, ax, ay)
            time.sleep(0.05)
        time.sleep(self._action_delay)
        logger.debug("double_click(%d, %d)", x, y)

    def right_click(self, x: int, y: int) -> None:
        self.click(x, y, button="right")

    def move(self, x: int, y: int) -> None:
        ax, ay = _to_abs(x, y, self._region)
        _send_mouse(_absolute_flags(self._region, include_move=True), ax, ay)

    def drag(self, x1: int, y1: int, x2: int, y2: int) -> None:
        ax1, ay1 = _to_abs(x1, y1, self._region)
        ax2, ay2 = _to_abs(x2, y2, self._region)
        absolute_move = _absolute_flags(self._region, include_move=True)
        absolute_button = _absolute_flags(self._region, include_move=False)
        _send_mouse(absolute_move, ax1, ay1)
        _send_mouse(MOUSEEVENTF_LEFTDOWN | absolute_button, ax1, ay1)
        time.sleep(0.1)
        _send_mouse(absolute_move, ax2, ay2)
        time.sleep(0.05)
        _send_mouse(MOUSEEVENTF_LEFTUP | absolute_button, ax2, ay2)
        time.sleep(self._action_delay)
        logger.debug("drag(%d,%d → %d,%d)", x1, y1, x2, y2)

    def scroll(self, x: int, y: int, direction: str = "down", amount: int = 3) -> None:
        ax, ay = _to_abs(x, y, self._region)
        _send_mouse(_absolute_flags(self._region, include_move=True), ax, ay)
        if direction in ("up", "down"):
            delta = WHEEL_DELTA * amount * (1 if direction == "up" else -1)
            _send_mouse(MOUSEEVENTF_WHEEL, data=delta)
        elif direction in ("left", "right"):
            delta = WHEEL_DELTA * amount * (1 if direction == "right" else -1)
            _send_mouse(MOUSEEVENTF_HWHEEL, data=delta)
        time.sleep(self._action_delay)
        logger.debug("scroll(%d, %d, %s, %d)", x, y, direction, amount)


def _button_flags(button: str) -> tuple[int, int]:
    """(down_flag, up_flag) for the given button name."""
    if button == "right":
        return MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP
    if button == "middle":
        return MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP
    return MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP
