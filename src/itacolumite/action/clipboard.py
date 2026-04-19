"""Clipboard operations via win32clipboard."""

from __future__ import annotations

import logging

import win32clipboard
import win32con

logger = logging.getLogger(__name__)


class ClipboardController:
    """Windows 클립보드 읽기/쓰기."""

    def get_text(self) -> str:
        """클립보드 텍스트 읽기."""
        try:
            win32clipboard.OpenClipboard()
            if win32clipboard.IsClipboardFormatAvailable(win32con.CF_UNICODETEXT):
                data = win32clipboard.GetClipboardData(win32con.CF_UNICODETEXT)
                return str(data)
            return ""
        except Exception as e:
            logger.warning("Failed to read clipboard: %s", e)
            return ""
        finally:
            try:
                win32clipboard.CloseClipboard()
            except Exception:
                pass

    def set_text(self, text: str) -> bool:
        """클립보드에 텍스트 쓰기."""
        try:
            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardText(text, win32con.CF_UNICODETEXT)
            return True
        except Exception as e:
            logger.warning("Failed to set clipboard: %s", e)
            return False
        finally:
            try:
                win32clipboard.CloseClipboard()
            except Exception:
                pass
