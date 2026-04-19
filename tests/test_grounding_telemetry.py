"""Tests for grounding telemetry persistence."""

import json

from itacolumite.ai.response_models import AgentAction, ActionParams
from itacolumite.core.coordinate_validation import ValidationResult
from itacolumite.core.grounding_telemetry import GroundingTelemetryLogger


def test_grounding_telemetry_logger_writes_jsonl(tmp_path) -> None:
    logger = GroundingTelemetryLogger(tmp_path)
    action = AgentAction(
        type="mouse_click",
        params=ActionParams(
            target_description="Search button",
            bbox_norm=[0.45, 0.25, 0.55, 0.35],
            center_norm=[0.5, 0.3],
            x=960,
            y=324,
        ),
    )
    validation = ValidationResult(
        approved=True,
        action_type="mouse_click",
        pixel_point=(960, 324),
        pixel_bbox=(864, 270, 1056, 378),
        score=0.82,
        reasons=["external_provider_support"],
        provider_assessments=[{"provider": "file_grounding", "score_delta": 0.12}],
    )

    logger.record_validation(
        task_id="task-123",
        step=4,
        model="gemini-2.5-pro",
        confidence=0.92,
        action=action,
        validation=validation,
    )
    logger.record_outcome(
        task_id="task-123",
        step=4,
        action_type="mouse_click",
        result="success",
        success=True,
        diff_ratio=0.013,
        validation=validation,
    )

    lines = logger.events_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    validation_event = json.loads(lines[0])
    outcome_event = json.loads(lines[1])
    assert validation_event["event_type"] == "validation"
    assert validation_event["task_id"] == "task-123"
    assert validation_event["provider_assessments"][0]["provider"] == "file_grounding"
    assert outcome_event["event_type"] == "outcome"
    assert outcome_event["diff_ratio"] == 0.013