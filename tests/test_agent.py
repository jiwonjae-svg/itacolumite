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
from itacolumite.perception.screen import CaptureContext


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
        patch("itacolumite.core.agent.GeminiGroundingExtractor") as mock_grounding_extractor,
        patch("itacolumite.core.agent.OmniParserRunner") as mock_omniparser_runner,
        patch("itacolumite.core.agent.save_grounding_capture_image"),
        patch("itacolumite.core.agent.write_grounding_provider_payload"),
        patch("itacolumite.core.agent.Memory"),
        patch("itacolumite.core.agent.GroundingTelemetryLogger"),
        patch("itacolumite.core.agent.ControlServer") as mock_cs,
    ):
        mock_cs.return_value.queue = Queue()
        mock_grounding_extractor.return_value.extract_provider_payload.return_value = {
            "provider": "gemini_ocr",
            "items": [],
        }
        mock_omniparser_runner.from_settings.return_value.extract_provider_payload.return_value = {
            "provider": "omniparser",
            "items": [],
        }
        agent = Agent()
    return agent


def _capture_context() -> CaptureContext:
    return CaptureContext(
        screen_width=1920,
        screen_height=1080,
        capture_width=1920,
        capture_height=1080,
        timestamp=0.0,
    )


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
        agent._memory.get_recent_history = MagicMock(return_value=[])
        agent._screen.capture_bytes_with_context = MagicMock(return_value=(b"png-data", _capture_context()))
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
        agent._memory.get_recent_history = MagicMock(return_value=[])
        agent._screen.capture_bytes_with_context = MagicMock(return_value=(b"png-data", _capture_context()))
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
        agent._screen.capture_bytes_with_context = MagicMock(return_value=(b"png-data", _capture_context()))
        agent._state_collector.collect = MagicMock(return_value=MagicMock(
            cwd="/test", foreground_window="VS Code", processes="", git_status=None, extra={},
        ))
        agent._gemini.generate_with_image = MagicMock(return_value="NOT-JSON")

        result = agent._execute_step("test task")
        assert result.success is False
        assert result.action_type == "parse_error"
        assert agent._consecutive_failures == 1

    def test_coordinate_validation_blocks_executor(self) -> None:
        agent = _make_agent()
        agent._memory.next_step = MagicMock(return_value=1)
        agent._memory.get_recent_history = MagicMock(return_value=[])
        agent._memory.record_action = MagicMock()
        agent._executor.execute = MagicMock()
        agent._screen.capture_bytes_with_context = MagicMock(return_value=(b"png-data", _capture_context()))
        agent._state_collector.collect = MagicMock(return_value=MagicMock(
            cwd="/test", foreground_window="VS Code", processes="", git_status=None, extra={},
        ))
        agent._gemini.generate_with_image = MagicMock(return_value='{"observation":"ok","reasoning":"r","plan":[],"next_action":{"type":"mouse_click","params":{"center_norm":[0.4,0.5]}},"confidence":0.9}')

        result = agent._execute_step("test task")

        assert result.success is False
        assert "coordinate validation blocked" in result.error
        agent._executor.execute.assert_not_called()

    def test_normalized_coordinates_are_converted_before_execution(self) -> None:
        agent = _make_agent()
        agent._memory.next_step = MagicMock(return_value=1)
        agent._memory.get_recent_history = MagicMock(return_value=[])
        agent._screen.capture_bytes_with_context = MagicMock(return_value=(b"png-data", _capture_context()))
        agent._state_collector.collect = MagicMock(return_value=MagicMock(
            cwd="/test", foreground_window="VS Code", processes="", git_status=None, extra={},
        ))
        agent._gemini.generate_with_image = MagicMock(return_value='{"observation":"ok","reasoning":"r","plan":[],"next_action":{"type":"mouse_click","params":{"target_description":"Search","bbox_norm":[0.45,0.25,0.55,0.35],"center_norm":[0.5,0.3]}},"confidence":0.9}')
        agent._executor.execute = MagicMock(return_value=ExecutionResult(success=True, action_type="mouse_click", output="clicked"))
        agent._screen.capture_after_action = MagicMock(return_value=None)
        agent._screen.screenshot_count = 1

        result = agent._execute_step("test task")

        assert result.success is True
        executed_action = agent._executor.execute.call_args.args[0]
        assert executed_action.params.x == 960
        assert executed_action.params.y == 324

    def test_drag_grounding_is_converted_before_execution(self) -> None:
        agent = _make_agent()
        agent._memory.next_step = MagicMock(return_value=1)
        agent._memory.get_recent_history = MagicMock(return_value=[])
        agent._screen.capture_bytes_with_context = MagicMock(return_value=(b"png-data", _capture_context()))
        agent._state_collector.collect = MagicMock(return_value=MagicMock(
            cwd="/test", foreground_window="VS Code", processes="", git_status=None, extra={},
        ))
        agent._gemini.generate_with_image = MagicMock(return_value='{"observation":"ok","reasoning":"r","plan":[],"next_action":{"type":"mouse_drag","params":{"start_target_description":"Task card","start_bbox_norm":[0.18,0.22,0.28,0.34],"start_center_norm":[0.23,0.28],"end_target_description":"Done column","end_bbox_norm":[0.72,0.18,0.90,0.82],"end_center_norm":[0.81,0.50]}},"confidence":0.9}')
        agent._executor.execute = MagicMock(return_value=ExecutionResult(success=True, action_type="mouse_drag", output="dragged"))
        agent._screen.capture_after_action = MagicMock(return_value=None)
        agent._screen.screenshot_count = 1

        result = agent._execute_step("test task")

        assert result.success is True
        executed_action = agent._executor.execute.call_args.args[0]
        assert executed_action.params.x1 == 441
        assert executed_action.params.y1 == 302
        assert executed_action.params.x2 == 1554
        assert executed_action.params.y2 == 540

    def test_simple_notepad_typing_adds_verification_hint_instead_of_auto_complete(self) -> None:
        agent = _make_agent()
        agent._memory.next_step = MagicMock(return_value=1)
        agent._memory.get_recent_history = MagicMock(return_value=[])
        agent._screen.capture_bytes_with_context = MagicMock(return_value=(b"png-data", _capture_context()))
        agent._state_collector.collect = MagicMock(return_value=MagicMock(
            cwd="/test", foreground_window="Notepad", processes="", git_status=None, extra={},
        ))
        agent._gemini.generate_with_image = MagicMock(return_value='{"observation":"ok","reasoning":"r","plan":[],"next_action":{"type":"type_text","params":{"text":"안녕하세요"}},"confidence":1.0}')
        agent._executor.execute = MagicMock(return_value=ExecutionResult(success=True, action_type="type_text", output="Typed 5 chars"))
        agent._screen.capture_after_action = MagicMock(return_value=None)
        agent._screen.screenshot_count = 1

        result = agent._execute_step("메모장을 열고 안녕하세요 라고 써봐")

        assert result.success is True
        assert result.task_complete is False
        assert any("verify that the requested text is visibly present" in message for message in agent._user_messages)

    def test_does_not_auto_complete_coding_typing_task(self) -> None:
        agent = _make_agent()
        agent._memory.next_step = MagicMock(return_value=1)
        agent._memory.get_recent_history = MagicMock(return_value=[])
        agent._screen.capture_bytes_with_context = MagicMock(return_value=(b"png-data", _capture_context()))
        agent._state_collector.collect = MagicMock(return_value=MagicMock(
            cwd="/test", foreground_window="VS Code", processes="", git_status=None, extra={},
        ))
        agent._gemini.generate_with_image = MagicMock(return_value='{"observation":"ok","reasoning":"r","plan":[],"next_action":{"type":"type_text","params":{"text":"hello"}},"confidence":1.0}')
        agent._executor.execute = MagicMock(return_value=ExecutionResult(success=True, action_type="type_text", output="Typed 5 chars"))
        agent._screen.capture_after_action = MagicMock(return_value=None)
        agent._screen.screenshot_count = 1

        result = agent._execute_step("VS Code에서 Copilot Chat에 hello 를 입력해")

        assert result.success is True
        assert result.task_complete is False
        assert agent._user_messages == []
