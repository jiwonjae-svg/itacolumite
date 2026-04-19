"""Agent core loop tests – _detect_loop, _process_control_commands, _execute_step."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from queue import Queue
from unittest.mock import MagicMock, patch

import pytest

from itacolumite.core.agent import (
    Agent,
    AgentState,
    StepSnapshot,
    _MAX_IDENTICAL_ACTIONS,
    _LOOP_HISTORY_SIZE,
)
from itacolumite.core.executor import ExecutionResult
from itacolumite.interface.control_server import ControlCommand, ControlMessage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_snapshot(action: str = "mouse_click", params_hash: int = 42, obs: str = "ok") -> StepSnapshot:
    return StepSnapshot(action_type=action, params_hash=params_hash, observation_prefix=obs)


def _make_agent() -> Agent:
    """Create an Agent with all heavy dependencies mocked out."""
    with (
        patch("itacolumite.core.agent.ScreenCapture"),
        patch("itacolumite.core.agent.StateCollector"),
        patch("itacolumite.core.agent.MouseController"),
        patch("itacolumite.core.agent.KeyboardController"),
        patch("itacolumite.core.agent.ShellController"),
        patch("itacolumite.core.agent.ClipboardController"),
        patch("itacolumite.core.agent.GeminiClient"),
        patch("itacolumite.core.agent.Memory"),
        patch("itacolumite.core.agent.ControlServer") as mock_cs,
    ):
        mock_cs.return_value.queue = Queue()
        agent = Agent()
    return agent


# ---------------------------------------------------------------------------
# _detect_loop
# ---------------------------------------------------------------------------

class TestDetectLoop:
    def test_no_loop_when_history_short(self) -> None:
        agent = _make_agent()
        agent._recent_actions = deque(maxlen=_LOOP_HISTORY_SIZE)
        # Fewer than _MAX_IDENTICAL_ACTIONS items → never a loop
        for _ in range(_MAX_IDENTICAL_ACTIONS - 1):
            agent._recent_actions.append(_make_snapshot())
        assert agent._detect_loop(_make_snapshot()) is False

    def test_detects_loop_with_identical_actions(self) -> None:
        agent = _make_agent()
        agent._recent_actions = deque(maxlen=_LOOP_HISTORY_SIZE)
        for _ in range(_MAX_IDENTICAL_ACTIONS):
            agent._recent_actions.append(_make_snapshot())
        assert agent._detect_loop(_make_snapshot()) is True

    def test_no_loop_with_varied_actions(self) -> None:
        agent = _make_agent()
        agent._recent_actions = deque(maxlen=_LOOP_HISTORY_SIZE)
        for i in range(_MAX_IDENTICAL_ACTIONS):
            agent._recent_actions.append(_make_snapshot(params_hash=i))
        assert agent._detect_loop(_make_snapshot(params_hash=99)) is False

    def test_no_loop_different_action_types(self) -> None:
        agent = _make_agent()
        agent._recent_actions = deque(maxlen=_LOOP_HISTORY_SIZE)
        for i in range(_MAX_IDENTICAL_ACTIONS):
            agent._recent_actions.append(_make_snapshot(action=f"action_{i}"))
        assert agent._detect_loop(_make_snapshot(action="action_new")) is False


# ---------------------------------------------------------------------------
# _process_control_commands
# ---------------------------------------------------------------------------

class TestProcessControlCommands:
    def test_pause_sets_flag(self) -> None:
        agent = _make_agent()
        agent._control_queue.put(ControlMessage(command=ControlCommand.PAUSE))
        agent._process_control_commands()
        assert agent._paused is True
        assert agent._agent_state.paused is True

    def test_resume_clears_flag(self) -> None:
        agent = _make_agent()
        agent._paused = True
        agent._agent_state.paused = True
        agent._control_queue.put(ControlMessage(command=ControlCommand.RESUME))
        agent._process_control_commands()
        assert agent._paused is False
        assert agent._agent_state.paused is False

    def test_stop_clears_running(self) -> None:
        agent = _make_agent()
        agent._running = True
        agent._control_queue.put(ControlMessage(command=ControlCommand.STOP))
        agent._process_control_commands()
        assert agent._running is False

    def test_send_appends_user_message(self) -> None:
        agent = _make_agent()
        agent._control_queue.put(ControlMessage(command=ControlCommand.SEND, payload="hello"))
        agent._process_control_commands()
        assert agent._user_messages == ["hello"]

    def test_multiple_commands_processed(self) -> None:
        agent = _make_agent()
        agent._control_queue.put(ControlMessage(command=ControlCommand.PAUSE))
        agent._control_queue.put(ControlMessage(command=ControlCommand.SEND, payload="msg"))
        agent._control_queue.put(ControlMessage(command=ControlCommand.RESUME))
        agent._process_control_commands()
        assert agent._paused is False  # resume after pause
        assert agent._user_messages == ["msg"]


# ---------------------------------------------------------------------------
# _execute_step (integration-level with mocked Gemini + executor)
# ---------------------------------------------------------------------------

class TestExecuteStep:
    def test_successful_step_resets_failure_counter(self) -> None:
        agent = _make_agent()
        agent._consecutive_failures = 3
        agent._memory.next_step = MagicMock(return_value=1)
        agent._screen.capture_bytes = MagicMock(return_value=b"png-data")
        agent._state_collector.collect = MagicMock(return_value=MagicMock(
            cwd="/test", foreground_window="VS Code", processes="", git_status=None, extra={},
        ))
        agent._gemini.generate_with_image = MagicMock(return_value='{"observation":"ok","reasoning":"r","plan":[],"next_action":{"type":"wait","params":{"seconds":1}},"confidence":0.8}')
        agent._executor.execute = MagicMock(return_value=ExecutionResult(success=True, action_type="wait", output="Waited 1s"))
        agent._screen.capture_after_action = MagicMock(return_value=None)
        agent._screen.screenshot_count = 1

        result = agent._execute_step("test task")
        assert result.success is True
        assert agent._consecutive_failures == 0

    def test_failed_step_increments_failure_counter(self) -> None:
        agent = _make_agent()
        agent._consecutive_failures = 0
        agent._memory.next_step = MagicMock(return_value=1)
        agent._screen.capture_bytes = MagicMock(return_value=b"png-data")
        agent._state_collector.collect = MagicMock(return_value=MagicMock(
            cwd="/test", foreground_window="VS Code", processes="", git_status=None, extra={},
        ))
        agent._gemini.generate_with_image = MagicMock(return_value='{"observation":"ok","reasoning":"r","plan":[],"next_action":{"type":"mouse_click","params":{"x":10,"y":20}},"confidence":0.5}')
        agent._executor.execute = MagicMock(return_value=ExecutionResult(success=False, action_type="mouse_click", error="click failed"))
        agent._screen.capture_after_action = MagicMock(return_value=None)
        agent._screen.screenshot_count = 1

        result = agent._execute_step("test task")
        assert result.success is False
        assert agent._consecutive_failures == 1

    def test_parse_error_records_failure(self) -> None:
        agent = _make_agent()
        agent._memory.next_step = MagicMock(return_value=1)
        agent._memory.record_action = MagicMock()
        agent._screen.capture_bytes = MagicMock(return_value=b"png-data")
        agent._state_collector.collect = MagicMock(return_value=MagicMock(
            cwd="/test", foreground_window="VS Code", processes="", git_status=None, extra={},
        ))
        agent._gemini.generate_with_image = MagicMock(return_value="NOT-JSON")

        result = agent._execute_step("test task")
        assert result.success is False
        assert result.action_type == "parse_error"
        assert agent._consecutive_failures == 1
