"""Action executor – bridges Gemini responses to actual Windows actions."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from itacolumite.action.clipboard import ClipboardController
from itacolumite.action.keyboard import KeyboardController
from itacolumite.action.mouse import MouseController
from itacolumite.action.shell import ShellController, ShellRequest
from itacolumite.ai.response_models import AgentAction, ActionParams
from itacolumite.perception.window import get_foreground_window

logger = logging.getLogger(__name__)

# Actions that interact with the foreground UI and need a focus guard
_FOCUS_SENSITIVE_ACTIONS = frozenset({
    "mouse_click", "mouse_double_click", "mouse_move",
    "mouse_drag", "mouse_scroll", "type_text", "key_press", "key_combo",
})


@dataclass
class ExecutionResult:
    """Result of executing an action."""

    success: bool
    action_type: str
    output: str = ""
    error: str = ""
    task_complete: bool = False
    task_result: str = ""


class ActionExecutor:
    """Executes agent actions by dispatching to the appropriate controller."""

    def __init__(
        self,
        mouse: MouseController,
        keyboard: KeyboardController,
        shell: ShellController,
        clipboard: ClipboardController,
    ) -> None:
        self._mouse = mouse
        self._keyboard = keyboard
        self._shell = shell
        self._clipboard = clipboard
        self._expected_window: str | None = None

    def set_expected_window(self, title_keyword: str | None) -> None:
        """Set the expected foreground window keyword (or None to disable the guard)."""
        self._expected_window = title_keyword

    def _check_focus(self, action_type: str) -> ExecutionResult | None:
        """Return an error result when the foreground window is unexpected, else None."""
        if self._expected_window is None:
            return None
        if action_type not in _FOCUS_SENSITIVE_ACTIONS:
            return None
        try:
            winfo = get_foreground_window()
            if self._expected_window.lower() not in winfo.title.lower():
                msg = (
                    f"Focus guard: expected '{self._expected_window}' but "
                    f"foreground is '{winfo.title}'"
                )
                logger.warning(msg)
                return ExecutionResult(success=False, action_type=action_type, error=msg)
        except Exception as e:
            logger.debug("Focus guard check failed: %s", e)
        return None

    def execute(self, action: AgentAction) -> ExecutionResult:
        """Execute a single agent action. Returns the result."""
        action_type = action.type.lower()
        params = action.params

        # Focus guard: abort UI actions when the wrong window is active
        focus_err = self._check_focus(action_type)
        if focus_err is not None:
            return focus_err

        try:
            handler = self._get_handler(action_type)
            if handler is None:
                return ExecutionResult(
                    success=False,
                    action_type=action_type,
                    error=f"Unknown action type: {action_type}",
                )
            return handler(params)
        except Exception as e:
            logger.error("Action execution failed: %s – %s", action_type, e)
            return ExecutionResult(
                success=False,
                action_type=action_type,
                error=str(e),
            )

    def _get_handler(self, action_type: str):
        """Map action type string to handler method."""
        handlers = {
            "mouse_click": self._handle_mouse_click,
            "mouse_double_click": self._handle_mouse_double_click,
            "mouse_move": self._handle_mouse_move,
            "mouse_drag": self._handle_mouse_drag,
            "mouse_scroll": self._handle_mouse_scroll,
            "type_text": self._handle_type_text,
            "key_press": self._handle_key_press,
            "key_combo": self._handle_key_combo,
            "shell_exec": self._handle_shell_exec,
            "wait": self._handle_wait,
            "task_complete": self._handle_task_complete,
        }
        return handlers.get(action_type)

    # ── Mouse handlers ───────────────────────────────────────

    def _handle_mouse_click(self, p: ActionParams) -> ExecutionResult:
        if p.x is None or p.y is None:
            return ExecutionResult(False, "mouse_click", error="Missing x or y")
        self._mouse.click(p.x, p.y, button=p.button or "left")
        return ExecutionResult(True, "mouse_click", output=f"Clicked ({p.x}, {p.y})")

    def _handle_mouse_double_click(self, p: ActionParams) -> ExecutionResult:
        if p.x is None or p.y is None:
            return ExecutionResult(False, "mouse_double_click", error="Missing x or y")
        self._mouse.double_click(p.x, p.y)
        return ExecutionResult(True, "mouse_double_click", output=f"Double-clicked ({p.x}, {p.y})")

    def _handle_mouse_move(self, p: ActionParams) -> ExecutionResult:
        if p.x is None or p.y is None:
            return ExecutionResult(False, "mouse_move", error="Missing x or y")
        self._mouse.move(p.x, p.y)
        return ExecutionResult(True, "mouse_move", output=f"Moved to ({p.x}, {p.y})")

    def _handle_mouse_drag(self, p: ActionParams) -> ExecutionResult:
        if p.x1 is None or p.y1 is None or p.x2 is None or p.y2 is None:
            return ExecutionResult(False, "mouse_drag", error="Missing coordinates")
        self._mouse.drag(p.x1, p.y1, p.x2, p.y2)
        return ExecutionResult(True, "mouse_drag", output=f"Dragged ({p.x1},{p.y1}) → ({p.x2},{p.y2})")

    def _handle_mouse_scroll(self, p: ActionParams) -> ExecutionResult:
        x = p.x or 960
        y = p.y or 540
        direction = p.direction or "down"
        amount = p.amount or 3
        self._mouse.scroll(x, y, direction=direction, amount=amount)
        return ExecutionResult(True, "mouse_scroll", output=f"Scrolled {direction} {amount} at ({x},{y})")

    # ── Keyboard handlers ────────────────────────────────────

    def _handle_type_text(self, p: ActionParams) -> ExecutionResult:
        if not p.text:
            return ExecutionResult(False, "type_text", error="Missing text")
        self._keyboard.type_text(p.text)
        return ExecutionResult(True, "type_text", output=f"Typed {len(p.text)} chars")

    def _handle_key_press(self, p: ActionParams) -> ExecutionResult:
        key = p.key
        if not key:
            return ExecutionResult(False, "key_press", error="Missing key")
        self._keyboard.press(key)
        return ExecutionResult(True, "key_press", output=f"Pressed {key}")

    def _handle_key_combo(self, p: ActionParams) -> ExecutionResult:
        keys = p.keys
        if not keys:
            return ExecutionResult(False, "key_combo", error="Missing keys")
        self._keyboard.combo(keys)
        return ExecutionResult(True, "key_combo", output=f"Combo {keys}")

    # ── Shell handler ────────────────────────────────────────

    def _handle_shell_exec(self, p: ActionParams) -> ExecutionResult:
        if not p.program:
            return ExecutionResult(False, "shell_exec", error="Missing program")
        request = ShellRequest(
            program=p.program,
            args=p.args if p.args else [],
            cwd=p.cwd,
            timeout=p.timeout if p.timeout else 30,
        )
        result = self._shell.execute(request)
        if not result.success and result.exit_code == -1:
            return ExecutionResult(False, "shell_exec", error=result.error)
        return ExecutionResult(
            success=result.success,
            action_type="shell_exec",
            output=result.output,
            error=result.error,
        )

    # ── Utility handlers ─────────────────────────────────────

    def _handle_wait(self, p: ActionParams) -> ExecutionResult:
        seconds = p.seconds or 1.0
        seconds = min(seconds, 10.0)  # Cap at 10 seconds
        time.sleep(seconds)
        return ExecutionResult(True, "wait", output=f"Waited {seconds}s")

    def _handle_task_complete(self, p: ActionParams) -> ExecutionResult:
        result = p.result or "Task completed"
        return ExecutionResult(
            True,
            "task_complete",
            output=result,
            task_complete=True,
            task_result=result,
        )
