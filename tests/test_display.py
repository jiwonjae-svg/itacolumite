"""Tests for desktop region selection helpers."""

from types import SimpleNamespace

import win32con

from itacolumite.perception import display


def test_get_desktop_region_primary_monitor(monkeypatch) -> None:
    metrics = {
        win32con.SM_CXSCREEN: 1920,
        win32con.SM_CYSCREEN: 1080,
        getattr(win32con, "SM_XVIRTUALSCREEN", 76): -1920,
        getattr(win32con, "SM_YVIRTUALSCREEN", 77): 0,
        getattr(win32con, "SM_CXVIRTUALSCREEN", 78): 3840,
        getattr(win32con, "SM_CYVIRTUALSCREEN", 79): 1080,
    }
    monkeypatch.setattr(display, "user32", SimpleNamespace(GetSystemMetrics=lambda index: metrics[index]))

    region = display.get_desktop_region("primary-monitor")

    assert region.left == 0
    assert region.top == 0
    assert region.width == 1920
    assert region.height == 1080
    assert region.is_virtual is False


def test_get_desktop_region_virtual_desktop(monkeypatch) -> None:
    metrics = {
        win32con.SM_CXSCREEN: 1920,
        win32con.SM_CYSCREEN: 1080,
        getattr(win32con, "SM_XVIRTUALSCREEN", 76): -1920,
        getattr(win32con, "SM_YVIRTUALSCREEN", 77): -120,
        getattr(win32con, "SM_CXVIRTUALSCREEN", 78): 3840,
        getattr(win32con, "SM_CYVIRTUALSCREEN", 79): 1200,
    }
    monkeypatch.setattr(display, "user32", SimpleNamespace(GetSystemMetrics=lambda index: metrics[index]))

    region = display.get_desktop_region("virtual-desktop")

    assert region.left == -1920
    assert region.top == -120
    assert region.width == 3840
    assert region.height == 1200
    assert region.is_virtual is True