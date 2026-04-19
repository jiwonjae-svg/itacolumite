"""Screen capture via Windows GDI BitBlt API."""

from __future__ import annotations

import ctypes
import io
import logging
import time
from pathlib import Path

import numpy as np
import win32con
import win32gui
import win32ui
from PIL import Image

from itacolumite.config.settings import get_settings

logger = logging.getLogger(__name__)

user32 = ctypes.windll.user32


def enable_dpi_awareness() -> None:
    """SetProcessDPIAware() — 단일 모니터 DPI 보정."""
    user32.SetProcessDPIAware()


class ScreenCapture:
    """Windows GDI BitBlt 기반 전체 화면 캡처."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._screenshot_dir = self._settings.agent_data_dir / "screenshots"
        self._screenshot_dir.mkdir(parents=True, exist_ok=True)
        self._last_screenshot: Image.Image | None = None
        self._screenshot_count = 0

        enable_dpi_awareness()

    def capture(self, *, save: bool = False) -> Image.Image:
        """주 모니터 전체 화면 캡처. save=True면 디스크에도 저장."""
        hdesktop = win32gui.GetDesktopWindow()
        width = user32.GetSystemMetrics(win32con.SM_CXSCREEN)
        height = user32.GetSystemMetrics(win32con.SM_CYSCREEN)

        hwindc = win32gui.GetWindowDC(hdesktop)
        srcdc = win32ui.CreateDCFromHandle(hwindc)
        memdc = srcdc.CreateCompatibleDC()
        bmp = win32ui.CreateBitmap()

        try:
            bmp.CreateCompatibleBitmap(srcdc, width, height)
            memdc.SelectObject(bmp)
            memdc.BitBlt((0, 0), (width, height), srcdc, (0, 0), win32con.SRCCOPY)

            bmp_info = bmp.GetInfo()
            bmp_bits = bmp.GetBitmapBits(True)
            img = Image.frombuffer(
                "RGB",
                (bmp_info["bmWidth"], bmp_info["bmHeight"]),
                bmp_bits,
                "raw",
                "BGRX",
                0,
                1,
            )
        finally:
            memdc.DeleteDC()
            srcdc.DeleteDC()
            win32gui.ReleaseDC(hdesktop, hwindc)
            win32gui.DeleteObject(bmp.GetHandle())

        self._last_screenshot = img
        self._screenshot_count += 1

        if save:
            path = self._screenshot_dir / f"screenshot_{self._screenshot_count:05d}.png"
            img.save(str(path))
            logger.debug("Screenshot saved: %s", path)

        return img

    def capture_after_action(self, *, save: bool = False) -> Image.Image:
        """액션 후 화면 안정 대기 → 캡처."""
        delay = self._settings.agent.screenshot_delay_ms / 1000.0
        time.sleep(delay)
        return self.capture(save=save)

    def capture_bytes(self, fmt: str = "PNG") -> bytes:
        """캡처 후 bytes 반환 (Gemini API 전송용)."""
        img = self.capture()
        buf = io.BytesIO()
        img.save(buf, format=fmt)
        return buf.getvalue()

    @property
    def last_screenshot(self) -> Image.Image | None:
        return self._last_screenshot

    @property
    def screenshot_count(self) -> int:
        return self._screenshot_count

    @staticmethod
    def diff_ratio(img_a: Image.Image, img_b: Image.Image) -> float:
        """두 이미지의 변경 픽셀 비율(0.0 ~ 1.0)을 반환한다.

        크기가 다르면 1.0(완전 다름)을 반환한다.
        """
        if img_a.size != img_b.size:
            return 1.0
        arr_a = np.asarray(img_a)
        arr_b = np.asarray(img_b)
        # 채널별 차이가 있는 픽셀 수를 센다
        changed = np.any(arr_a != arr_b, axis=-1)
        return float(changed.sum()) / max(changed.size, 1)
