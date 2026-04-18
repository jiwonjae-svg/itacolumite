"""Keyboard control via Windows SendInput API (KEYEVENTF_UNICODE)."""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import logging
import time

from itacolumite.config.settings import get_settings

logger = logging.getLogger(__name__)

user32 = ctypes.windll.user32

# --- SendInput 구조체 ---

INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004

# 특수 키 → 가상 키코드 매핑
_VK_MAP: dict[str, int] = {
    "enter": 0x0D, "return": 0x0D,
    "tab": 0x09,
    "escape": 0x1B, "esc": 0x1B,
    "backspace": 0x08,
    "delete": 0x2E, "del": 0x2E,
    "space": 0x20,
    "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
    "home": 0x24, "end": 0x23, "pageup": 0x21, "pagedown": 0x22,
    "insert": 0x2D,
    "f1": 0x70, "f2": 0x71, "f3": 0x72, "f4": 0x73,
    "f5": 0x74, "f6": 0x75, "f7": 0x76, "f8": 0x77,
    "f9": 0x78, "f10": 0x79, "f11": 0x7A, "f12": 0x7B,
    "ctrl": 0xA2, "lctrl": 0xA2, "rctrl": 0xA3,
    "alt": 0xA4, "lalt": 0xA4, "ralt": 0xA5,
    "shift": 0xA0, "lshift": 0xA0, "rshift": 0xA1,
    "win": 0x5B, "lwin": 0x5B, "rwin": 0x5C,
    "capslock": 0x14, "numlock": 0x90, "scrolllock": 0x91,
    "printscreen": 0x2C, "pause": 0x13,
    "apps": 0x5D,  # context menu key
    "grave": 0xC0, "backtick": 0xC0, "`": 0xC0,  # backtick / grave accent
}


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.wintypes.WORD),
        ("wScan", ctypes.wintypes.WORD),
        ("dwFlags", ctypes.wintypes.DWORD),
        ("time", ctypes.wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class INPUT(ctypes.Structure):
    class _INPUT_UNION(ctypes.Union):
        _fields_ = [("ki", KEYBDINPUT)]

    _anonymous_ = ("_union",)
    _fields_ = [
        ("type", ctypes.wintypes.DWORD),
        ("_union", _INPUT_UNION),
    ]


def _send_key(vk: int = 0, scan: int = 0, flags: int = 0) -> None:
    inp = INPUT()
    inp.type = INPUT_KEYBOARD
    inp.ki.wVk = vk
    inp.ki.wScan = scan
    inp.ki.dwFlags = flags
    inp.ki.time = 0
    inp.ki.dwExtraInfo = ctypes.pointer(ctypes.c_ulong(0))
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))


class KeyboardController:
    """SendInput 기반 키보드 제어."""

    def __init__(self) -> None:
        settings = get_settings()
        self._typing_delay = settings.agent.typing_delay_ms / 1000.0
        self._action_delay = settings.agent.action_delay_ms / 1000.0

    def type_text(self, text: str) -> None:
        """유니코드 텍스트 입력 (한글 포함)."""
        for char in text:
            code = ord(char)
            _send_key(scan=code, flags=KEYEVENTF_UNICODE)
            _send_key(scan=code, flags=KEYEVENTF_UNICODE | KEYEVENTF_KEYUP)
            if self._typing_delay > 0:
                time.sleep(self._typing_delay)
        logger.debug("Typed %d chars", len(text))

    def press(self, key: str) -> None:
        """단일 키 누르기 (e.g. 'enter', 'tab', 'f5')."""
        vk = _VK_MAP.get(key.lower())
        if vk is None:
            # 단일 문자면 Unicode로 처리
            if len(key) == 1:
                code = ord(key)
                _send_key(scan=code, flags=KEYEVENTF_UNICODE)
                _send_key(scan=code, flags=KEYEVENTF_UNICODE | KEYEVENTF_KEYUP)
            else:
                logger.warning("Unknown key: %s", key)
                return
        else:
            _send_key(vk=vk)
            _send_key(vk=vk, flags=KEYEVENTF_KEYUP)
        time.sleep(self._action_delay)

    def combo(self, keys: str) -> None:
        """키 조합 (e.g. 'ctrl+s', 'ctrl+shift+i').

        구분자: '+' 또는 '-'
        """
        parts = keys.replace("-", "+").split("+")
        parts = [p.strip() for p in parts if p.strip()]

        modifiers = parts[:-1]
        final_key = parts[-1]

        # 모디파이어 누르기
        for mod in modifiers:
            vk = _VK_MAP.get(mod.lower())
            if vk is not None:
                _send_key(vk=vk)

        # 마지막 키
        final_vk = _VK_MAP.get(final_key.lower())
        if final_vk is not None:
            _send_key(vk=final_vk)
            _send_key(vk=final_vk, flags=KEYEVENTF_KEYUP)
        elif len(final_key) == 1:
            code = ord(final_key)
            _send_key(scan=code, flags=KEYEVENTF_UNICODE)
            _send_key(scan=code, flags=KEYEVENTF_UNICODE | KEYEVENTF_KEYUP)
        else:
            logger.warning("Unknown final key in combo: %s", final_key)

        # 모디파이어 해제 (역순)
        for mod in reversed(modifiers):
            vk = _VK_MAP.get(mod.lower())
            if vk is not None:
                _send_key(vk=vk, flags=KEYEVENTF_KEYUP)

        time.sleep(self._action_delay)
        logger.debug("combo(%s)", keys)

    # --- 편의 메서드 ---

    def enter(self) -> None:
        self.press("enter")

    def escape(self) -> None:
        self.press("escape")

    def tab(self) -> None:
        self.press("tab")

    def backspace(self, count: int = 1) -> None:
        for _ in range(count):
            self.press("backspace")

    def select_all(self) -> None:
        self.combo("ctrl+a")

    def copy(self) -> None:
        self.combo("ctrl+c")

    def paste(self) -> None:
        self.combo("ctrl+v")

    def save(self) -> None:
        self.combo("ctrl+s")

    def undo(self) -> None:
        self.combo("ctrl+z")
