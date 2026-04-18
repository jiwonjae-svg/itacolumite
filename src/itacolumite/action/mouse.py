"""Mouse control via Windows SendInput API."""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import logging
import time

from itacolumite.config.settings import get_settings

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


def _to_abs(x: int, y: int) -> tuple[int, int]:
    """주 모니터 pixel 좌표 → SendInput 정규화 좌표(0-65535)."""
    sw = user32.GetSystemMetrics(0)  # SM_CXSCREEN
    sh = user32.GetSystemMetrics(1)  # SM_CYSCREEN
    abs_x = int(x * 65535 / max(sw - 1, 1))
    abs_y = int(y * 65535 / max(sh - 1, 1))
    return abs_x, abs_y


class MouseController:
    """SendInput 기반 마우스 제어."""

    def __init__(self) -> None:
        self._action_delay = get_settings().agent.action_delay_ms / 1000.0

    def click(self, x: int, y: int, button: str = "left") -> None:
        ax, ay = _to_abs(x, y)
        down, up = _button_flags(button)
        _send_mouse(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, ax, ay)
        _send_mouse(down | MOUSEEVENTF_ABSOLUTE, ax, ay)
        _send_mouse(up | MOUSEEVENTF_ABSOLUTE, ax, ay)
        time.sleep(self._action_delay)
        logger.debug("click(%d, %d, %s)", x, y, button)

    def double_click(self, x: int, y: int) -> None:
        ax, ay = _to_abs(x, y)
        _send_mouse(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, ax, ay)
        for _ in range(2):
            _send_mouse(MOUSEEVENTF_LEFTDOWN | MOUSEEVENTF_ABSOLUTE, ax, ay)
            _send_mouse(MOUSEEVENTF_LEFTUP | MOUSEEVENTF_ABSOLUTE, ax, ay)
            time.sleep(0.05)
        time.sleep(self._action_delay)
        logger.debug("double_click(%d, %d)", x, y)

    def right_click(self, x: int, y: int) -> None:
        self.click(x, y, button="right")

    def move(self, x: int, y: int) -> None:
        ax, ay = _to_abs(x, y)
        _send_mouse(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, ax, ay)

    def drag(self, x1: int, y1: int, x2: int, y2: int) -> None:
        ax1, ay1 = _to_abs(x1, y1)
        ax2, ay2 = _to_abs(x2, y2)
        _send_mouse(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, ax1, ay1)
        _send_mouse(MOUSEEVENTF_LEFTDOWN | MOUSEEVENTF_ABSOLUTE, ax1, ay1)
        time.sleep(0.1)
        _send_mouse(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, ax2, ay2)
        time.sleep(0.05)
        _send_mouse(MOUSEEVENTF_LEFTUP | MOUSEEVENTF_ABSOLUTE, ax2, ay2)
        time.sleep(self._action_delay)
        logger.debug("drag(%d,%d → %d,%d)", x1, y1, x2, y2)

    def scroll(self, x: int, y: int, direction: str = "down", amount: int = 3) -> None:
        ax, ay = _to_abs(x, y)
        _send_mouse(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, ax, ay)
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
