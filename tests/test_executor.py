"""Tests for executor focus guard behavior."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from itacolumite.ai.response_models import ActionParams, AgentAction
from itacolumite.core.executor import ActionExecutor
from itacolumite.perception.window import WindowInfo


def _window_info(
    *,
    hwnd: int,
    title: str,
    class_name: str = "Chrome_WidgetWin_1",
    rect: tuple[int, int, int, int] = (0, 0, 1920, 1080),
    pid: int = 100,
) -> WindowInfo:
    return WindowInfo(
        hwnd=hwnd,
        title=title,
        class_name=class_name,
        rect=rect,
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


def test_type_text_uses_clipboard_paste_and_restores_clipboard() -> None:
    keyboard = MagicMock()
    clipboard = MagicMock()
    clipboard.get_text.return_value = "previous"
    clipboard.set_text.side_effect = [True, True]
    executor = ActionExecutor(MagicMock(), keyboard, MagicMock(), clipboard)

    with patch(
        "itacolumite.core.executor.get_foreground_window",
        return_value=_window_info(hwnd=10, title="README.md - itacolumite - Visual Studio Code"),
    ):
        result = executor.execute(
            AgentAction(type="type_text", params=ActionParams(text="안녕하세요"))
        )

    assert result.success is True
    assert result.output == "Pasted 5 chars"
    keyboard.paste.assert_called_once_with()
    keyboard.type_text.assert_not_called()
    assert clipboard.set_text.call_args_list[0].args == ("안녕하세요",)
    assert clipboard.set_text.call_args_list[1].args == ("previous",)


def test_type_text_falls_back_to_sendinput_when_clipboard_write_fails() -> None:
    keyboard = MagicMock()
    clipboard = MagicMock()
    clipboard.get_text.return_value = "previous"
    clipboard.set_text.side_effect = [False, True]
    executor = ActionExecutor(MagicMock(), keyboard, MagicMock(), clipboard)

    with patch(
        "itacolumite.core.executor.get_foreground_window",
        return_value=_window_info(hwnd=10, title="README.md - itacolumite - Visual Studio Code"),
    ):
        result = executor.execute(
            AgentAction(type="type_text", params=ActionParams(text="hello"))
        )

    assert result.success is True
    assert result.output == "Typed 5 chars"
    keyboard.type_text.assert_called_once_with("hello")
    keyboard.paste.assert_not_called()


def test_type_text_pastes_directly_into_notepad_editor() -> None:
    mouse = MagicMock()
    keyboard = MagicMock()
    clipboard = MagicMock()
    clipboard.get_text.return_value = "previous"
    clipboard.set_text.side_effect = [True, True]
    executor = ActionExecutor(mouse, keyboard, MagicMock(), clipboard)

    with (
        patch(
            "itacolumite.core.executor.get_foreground_window",
            return_value=_window_info(
                hwnd=10,
                title="제목 없음 - 메모장",
                class_name="Notepad",
                pid=300,
            ),
        ),
        patch(
            "itacolumite.core.executor.find_child_window",
            return_value=_window_info(
                hwnd=11,
                title="",
                class_name="RichEditD2DPT",
                pid=300,
                rect=(102, 170, 1241, 649),
            ),
        ),
        patch("itacolumite.core.executor.win32gui.SendMessage") as send_message,
    ):
        result = executor.execute(
            AgentAction(type="type_text", params=ActionParams(text="안녕하세요"))
        )

    assert result.success is True
    mouse.click.assert_not_called()
    keyboard.paste.assert_not_called()
    send_message.assert_called_once_with(11, 770, 0, 0)