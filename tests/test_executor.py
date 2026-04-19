"""Tests for executor focus guard behavior."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from itacolumite.ai.response_models import ActionParams, AgentAction
from itacolumite.core.executor import ActionExecutor
from itacolumite.perception.window import WindowInfo


def _window_info(*, hwnd: int, title: str, class_name: str = "Chrome_WidgetWin_1", pid: int = 100) -> WindowInfo:
    return WindowInfo(
        hwnd=hwnd,
        title=title,
        class_name=class_name,
        rect=(0, 0, 1920, 1080),
        pid=pid,
    )


def test_focus_guard_allows_same_app_when_title_changes() -> None:
    mouse = MagicMock()
    executor = ActionExecutor(mouse, MagicMock(), MagicMock(), MagicMock())
    executor.set_expected_window(
        _window_info(hwnd=10, title="README.md - itacolumite - Visual Studio Code")
    )

    with patch(
        "itacolumite.core.executor.get_foreground_window",
        return_value=_window_info(
            hwnd=11,
            title="screenshot_00008.png - itacolumite - Visual Studio Code",
        ),
    ):
        result = executor.execute(
            AgentAction(type="mouse_click", params=ActionParams(x=10, y=20))
        )

    assert result.success is True
    mouse.click.assert_called_once_with(10, 20, button="left")


def test_focus_guard_blocks_different_foreground_app() -> None:
    mouse = MagicMock()
    executor = ActionExecutor(mouse, MagicMock(), MagicMock(), MagicMock())
    executor.set_expected_window(
        _window_info(hwnd=10, title="README.md - itacolumite - Visual Studio Code")
    )

    with patch(
        "itacolumite.core.executor.get_foreground_window",
        return_value=_window_info(
            hwnd=20,
            title="Untitled - Notepad",
            class_name="Notepad",
            pid=200,
        ),
    ):
        result = executor.execute(
            AgentAction(type="mouse_click", params=ActionParams(x=10, y=20))
        )

    assert result.success is False
    assert "Focus guard" in result.error
    mouse.click.assert_not_called()


def test_focus_guard_does_not_block_global_keyboard_shortcuts() -> None:
    keyboard = MagicMock()
    executor = ActionExecutor(MagicMock(), keyboard, MagicMock(), MagicMock())
    executor.set_expected_window(
        _window_info(hwnd=10, title="Windows PowerShell")
    )

    with patch(
        "itacolumite.core.executor.get_foreground_window",
        return_value=_window_info(
            hwnd=20,
            title="",
            class_name="Shell_TrayWnd",
            pid=200,
        ),
    ):
        result = executor.execute(
            AgentAction(type="key_combo", params=ActionParams(keys="win+r"))
        )

    assert result.success is True
    keyboard.combo.assert_called_once_with("win+r")


def test_focus_guard_still_blocks_non_global_shortcuts() -> None:
    keyboard = MagicMock()
    executor = ActionExecutor(MagicMock(), keyboard, MagicMock(), MagicMock())
    executor.set_expected_window(
        _window_info(hwnd=10, title="README.md - itacolumite - Visual Studio Code")
    )

    with patch(
        "itacolumite.core.executor.get_foreground_window",
        return_value=_window_info(
            hwnd=20,
            title="Untitled - Notepad",
            class_name="Notepad",
            pid=200,
        ),
    ):
        result = executor.execute(
            AgentAction(type="key_combo", params=ActionParams(keys="ctrl+s"))
        )

    assert result.success is False
    assert "Focus guard" in result.error
    keyboard.combo.assert_not_called()


def test_focus_guard_restores_expected_window_before_mouse_action() -> None:
    mouse = MagicMock()
    executor = ActionExecutor(mouse, MagicMock(), MagicMock(), MagicMock())
    executor.set_expected_window(
        _window_info(hwnd=10, title="*새 텍스트 문서.txt - 메모장", class_name="Notepad", pid=300)
    )

    with (
        patch(
            "itacolumite.core.executor.get_foreground_window",
            side_effect=[
                _window_info(hwnd=20, title="Windows PowerShell", class_name="ConsoleWindowClass", pid=200),
                _window_info(hwnd=10, title="*새 텍스트 문서.txt - 메모장", class_name="Notepad", pid=300),
            ],
        ),
        patch(
            "itacolumite.core.executor.find_window",
            return_value=_window_info(hwnd=10, title="*새 텍스트 문서.txt - 메모장", class_name="Notepad", pid=300),
        ),
        patch("itacolumite.core.executor.activate_window", return_value=True),
    ):
        result = executor.execute(
            AgentAction(type="mouse_click", params=ActionParams(x=100, y=120))
        )

    assert result.success is True
    mouse.click.assert_called_once_with(100, 120, button="left")