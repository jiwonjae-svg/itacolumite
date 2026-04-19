"""Tests for response model parsing."""

import pytest

from itacolumite.ai.response_models import AgentResponse, parse_agent_response


class TestParseAgentResponse:
    """Test parsing of Gemini JSON responses."""

    def test_valid_response(self) -> None:
        raw = '''
        {
            "observation": "Desktop is visible with terminal open",
            "reasoning": "Need to open VS Code to start coding",
            "plan": ["Open VS Code", "Open Copilot Chat"],
            "next_action": {
                "type": "key_combo",
                "params": {"keys": "ctrl-shift-p"}
            },
            "confidence": 0.85
        }
        '''
        resp = parse_agent_response(raw)
        assert resp.observation == "Desktop is visible with terminal open"
        assert resp.next_action.type == "key_combo"
        assert resp.next_action.params.keys == "ctrl-shift-p"
        assert resp.confidence == 0.85

    def test_markdown_fences(self) -> None:
        raw = '''```json
        {
            "observation": "test",
            "reasoning": "test",
            "next_action": {"type": "wait", "params": {"seconds": 1}},
            "confidence": 0.5
        }
        ```'''
        resp = parse_agent_response(raw)
        assert resp.next_action.type == "wait"
        assert resp.next_action.params.seconds == 1

    def test_minimal_response(self) -> None:
        raw = '{"next_action": {"type": "task_complete", "params": {"result": "done"}}}'
        resp = parse_agent_response(raw)
        assert resp.next_action.type == "task_complete"
        assert resp.next_action.params.result == "done"
        assert resp.confidence == 0.5  # default

    def test_mouse_click_response(self) -> None:
        raw = '''
        {
            "observation": "VS Code is open",
            "reasoning": "Click on the terminal button",
            "next_action": {
                "type": "mouse_click",
                "params": {"x": 500, "y": 300, "button": "left"}
            },
            "confidence": 0.9
        }
        '''
        resp = parse_agent_response(raw)
        assert resp.next_action.params.x == 500
        assert resp.next_action.params.y == 300
        assert resp.next_action.params.button == "left"

    def test_normalized_grounding_response(self) -> None:
        raw = '''
        {
            "observation": "Browser window is visible",
            "reasoning": "Click the search button",
            "next_action": {
                "type": "mouse_click",
                "params": {
                    "target_description": "Search button",
                    "bbox_norm": [0.75, 0.12, 0.82, 0.18],
                    "center_norm": [0.785, 0.15]
                }
            },
            "confidence": 0.88
        }
        '''
        resp = parse_agent_response(raw)
        assert resp.next_action.params.target_description == "Search button"
        assert resp.next_action.params.bbox_norm == [0.75, 0.12, 0.82, 0.18]
        assert resp.next_action.params.center_norm == [0.785, 0.15]

    def test_drag_grounding_response(self) -> None:
        raw = '''
        {
            "observation": "A kanban card and destination lane are visible",
            "reasoning": "Drag the task card into Done",
            "next_action": {
                "type": "mouse_drag",
                "params": {
                    "start_target_description": "Task card 'Write tests'",
                    "start_bbox_norm": [0.18, 0.22, 0.28, 0.34],
                    "start_center_norm": [0.23, 0.28],
                    "end_target_description": "Done column drop area",
                    "end_bbox_norm": [0.72, 0.18, 0.90, 0.82],
                    "end_center_norm": [0.81, 0.50]
                }
            },
            "confidence": 0.86
        }
        '''
        resp = parse_agent_response(raw)
        assert resp.next_action.type == "mouse_drag"
        assert resp.next_action.params.start_target_description == "Task card 'Write tests'"
        assert resp.next_action.params.start_center_norm == [0.23, 0.28]
        assert resp.next_action.params.end_target_description == "Done column drop area"
        assert resp.next_action.params.end_bbox_norm == [0.72, 0.18, 0.90, 0.82]

    def test_invalid_center_norm_length(self) -> None:
        raw = '''
        {
            "next_action": {
                "type": "mouse_click",
                "params": {"center_norm": [0.5, 0.2, 0.1]}
            }
        }
        '''
        with pytest.raises(ValueError, match="center_norm"):
            parse_agent_response(raw)

    def test_invalid_drag_center_norm_length(self) -> None:
        raw = '''
        {
            "next_action": {
                "type": "mouse_drag",
                "params": {"start_center_norm": [0.5, 0.2, 0.1]}
            }
        }
        '''
        with pytest.raises(ValueError, match="center_norm"):
            parse_agent_response(raw)

    def test_invalid_json(self) -> None:
        with pytest.raises(ValueError, match="Invalid JSON"):
            parse_agent_response("not json at all")

    def test_shell_exec_response(self) -> None:
        raw = '''
        {
            "observation": "Terminal is open",
            "reasoning": "Run tests",
            "next_action": {
                "type": "shell_exec",
                "params": {"program": "pytest", "args": ["-v", "tests/"]}
            },
            "confidence": 0.95
        }
        '''
        resp = parse_agent_response(raw)
        assert resp.next_action.type == "shell_exec"
        assert resp.next_action.params.program == "pytest"
        assert resp.next_action.params.args == ["-v", "tests/"]
