"""Tests for coordinate validation and normalized grounding conversion."""

import json

from PIL import Image, ImageDraw

from itacolumite.ai.response_models import AgentAction, ActionParams
from itacolumite.core.coordinate_validation import (
    ValidationConfig,
    normalized_bbox_to_screen_bbox,
    normalized_point_to_screen,
    pixel_to_normalized_point,
    validate_action_coordinates,
)
from itacolumite.core.grounding_providers import FileBackedGroundingProvider, LocalCropGroundingProvider
from itacolumite.core.memory import ActionRecord
from itacolumite.perception.screen import CaptureContext


def _capture_context() -> CaptureContext:
    return CaptureContext(
        screen_width=1920,
        screen_height=1080,
        capture_width=1920,
        capture_height=1080,
        timestamp=0.0,
    )


def _config() -> ValidationConfig:
    return ValidationConfig(
        require_bbox=True,
        min_confidence=0.45,
        edge_margin_px=8,
        min_bbox_size_px=12,
        max_repeat_failures=2,
    )


def test_normalized_point_to_screen() -> None:
    assert normalized_point_to_screen([0.5, 0.25], _capture_context()) == (960, 270)


def test_normalized_point_to_screen_with_virtual_desktop_origin() -> None:
    context = CaptureContext(
        screen_width=3840,
        screen_height=1080,
        capture_width=3840,
        capture_height=1080,
        timestamp=0.0,
        screen_left=-1920,
        screen_top=0,
        capture_target="virtual-desktop",
    )

    assert normalized_point_to_screen([0.0, 0.0], context) == (-1920, 0)
    assert normalized_point_to_screen([0.5, 0.25], context) == (0, 270)


def test_normalized_bbox_to_screen_bbox() -> None:
    assert normalized_bbox_to_screen_bbox([0.1, 0.2, 0.3, 0.4], _capture_context()) == (
        192,
        216,
        576,
        432,
    )


def test_validator_approves_valid_grounding() -> None:
    action = AgentAction(
        type="mouse_click",
        params=ActionParams(
            target_description="Search button",
            bbox_norm=[0.45, 0.25, 0.55, 0.35],
            center_norm=[0.5, 0.3],
        ),
    )

    result = validate_action_coordinates(
        action,
        confidence=0.9,
        capture_context=_capture_context(),
        recent_records=[],
        config=_config(),
    )

    assert result.approved is True
    assert result.pixel_point == (960, 324)


def test_validator_rejects_missing_bbox_when_required() -> None:
    action = AgentAction(type="mouse_click", params=ActionParams(center_norm=[0.5, 0.2]))

    result = validate_action_coordinates(
        action,
        confidence=0.9,
        capture_context=_capture_context(),
        recent_records=[],
        config=_config(),
    )

    assert result.approved is False
    assert "missing_bbox" in result.reasons


def test_validator_rejects_repeat_failure_hotspot() -> None:
    action = AgentAction(
        type="mouse_click",
        params=ActionParams(
            target_description="Search button",
            bbox_norm=[0.45, 0.25, 0.55, 0.35],
            center_norm=[0.5, 0.3],
        ),
    )
    recent = [
        ActionRecord(
            step=1,
            timestamp="2026-04-19T00:00:00",
            action_type="mouse_click",
            params={"x": 960, "y": 324},
            observation="",
            reasoning="",
            confidence=0.2,
            result="failure",
        ),
        ActionRecord(
            step=2,
            timestamp="2026-04-19T00:00:01",
            action_type="mouse_click",
            params={"x": 962, "y": 326},
            observation="",
            reasoning="",
            confidence=0.2,
            result="blocked",
        ),
    ]

    result = validate_action_coordinates(
        action,
        confidence=0.9,
        capture_context=_capture_context(),
        recent_records=recent,
        config=_config(),
    )

    assert result.approved is False
    assert "repeat_failure_hotspot" in result.reasons


def test_validator_allows_legacy_pixel_fallback() -> None:
    action = AgentAction(type="mouse_click", params=ActionParams(x=500, y=300))

    result = validate_action_coordinates(
        action,
        confidence=0.9,
        capture_context=_capture_context(),
        recent_records=[],
        config=_config(),
    )

    assert result.approved is True
    assert result.pixel_point == (500, 300)


def test_validator_allows_virtual_desktop_pixel_fallback() -> None:
    action = AgentAction(type="mouse_click", params=ActionParams(x=-100, y=300))
    capture_context = CaptureContext(
        screen_width=3840,
        screen_height=1080,
        capture_width=3840,
        capture_height=1080,
        timestamp=0.0,
        screen_left=-1920,
        screen_top=0,
        capture_target="virtual-desktop",
    )

    result = validate_action_coordinates(
        action,
        confidence=0.9,
        capture_context=capture_context,
        recent_records=[],
        config=_config(),
    )

    assert result.approved is True
    assert result.pixel_point == (-100, 300)


def test_drag_validator_approves_start_and_end_grounding() -> None:
    action = AgentAction(
        type="mouse_drag",
        params=ActionParams(
            start_target_description="Task card",
            start_bbox_norm=[0.18, 0.22, 0.28, 0.34],
            start_center_norm=[0.23, 0.28],
            end_target_description="Done column",
            end_bbox_norm=[0.72, 0.18, 0.90, 0.82],
            end_center_norm=[0.81, 0.50],
        ),
    )

    result = validate_action_coordinates(
        action,
        confidence=0.9,
        capture_context=_capture_context(),
        recent_records=[],
        config=_config(),
    )

    assert result.approved is True
    assert result.pixel_point == normalized_point_to_screen([0.23, 0.28], _capture_context())
    assert result.pixel_point_end == normalized_point_to_screen([0.81, 0.50], _capture_context())


def test_drag_validator_rejects_missing_end_bbox_when_required() -> None:
    action = AgentAction(
        type="mouse_drag",
        params=ActionParams(
            start_target_description="Task card",
            start_bbox_norm=[0.18, 0.22, 0.28, 0.34],
            start_center_norm=[0.23, 0.28],
            end_center_norm=[0.81, 0.50],
        ),
    )

    result = validate_action_coordinates(
        action,
        confidence=0.9,
        capture_context=_capture_context(),
        recent_records=[],
        config=_config(),
    )

    assert result.approved is False
    assert "end_missing_bbox" in result.reasons


def test_local_crop_recheck_blocks_flat_regions() -> None:
    action = AgentAction(
        type="mouse_click",
        params=ActionParams(
            target_description="Search button",
            bbox_norm=[0.45, 0.25, 0.55, 0.35],
            center_norm=[0.5, 0.3],
        ),
    )
    screenshot = Image.new("RGB", (1920, 1080), color="white")
    provider = LocalCropGroundingProvider(
        crop_padding_px=4,
        min_crop_stddev=12.0,
        min_edge_density=0.015,
        max_bbox_area_ratio=0.35,
    )

    result = validate_action_coordinates(
        action,
        confidence=0.9,
        capture_context=_capture_context(),
        recent_records=[],
        config=_config(),
        screenshot=screenshot,
        providers=[provider],
    )

    assert result.approved is False
    assert "flat_local_crop" in result.reasons


def test_local_crop_recheck_uses_capture_relative_bbox_for_virtual_desktop() -> None:
    capture_context = CaptureContext(
        screen_width=3840,
        screen_height=1080,
        capture_width=3840,
        capture_height=1080,
        timestamp=0.0,
        screen_left=-1920,
        screen_top=0,
        capture_target="virtual-desktop",
    )
    screenshot = Image.new("RGB", (3840, 1080), color="gray")
    draw = ImageDraw.Draw(screenshot)
    for offset in range(0, 201, 8):
        color = "white" if (offset // 8) % 2 == 0 else "black"
        draw.rectangle((320 + offset, 100, 323 + offset, 220), fill=color)

    left_top = pixel_to_normalized_point(-1600, 100, capture_context)
    right_bottom = pixel_to_normalized_point(-1400, 220, capture_context)
    center = pixel_to_normalized_point(-1500, 160, capture_context)
    action = AgentAction(
        type="mouse_click",
        params=ActionParams(
            target_description="Secondary monitor target",
            bbox_norm=[left_top[0], left_top[1], right_bottom[0], right_bottom[1]],
            center_norm=center,
        ),
    )
    provider = LocalCropGroundingProvider(
        crop_padding_px=4,
        min_crop_stddev=12.0,
        min_edge_density=0.015,
        max_bbox_area_ratio=0.35,
    )

    result = validate_action_coordinates(
        action,
        confidence=0.9,
        capture_context=capture_context,
        recent_records=[],
        config=_config(),
        screenshot=screenshot,
        providers=[provider],
    )

    assert result.approved is True
    assert "invalid_local_crop" not in result.reasons
    assert "local_crop_support" in result.reasons


def test_file_provider_supports_matching_external_detection(tmp_path) -> None:
    action = AgentAction(
        type="mouse_click",
        params=ActionParams(
            target_description="Search button",
            bbox_norm=[0.45, 0.25, 0.55, 0.35],
            center_norm=[0.5, 0.3],
        ),
    )
    provider_dir = tmp_path / "grounding" / "providers"
    provider_dir.mkdir(parents=True)
    payload = {
        "provider": "ocr",
        "anchors": [
            {
                "text": "Search button",
                "bbox_norm": [0.45, 0.25, 0.55, 0.35],
                "score": 0.97,
            }
        ],
    }
    (provider_dir / "ocr.json").write_text(json.dumps(payload), encoding="utf-8")
    provider = FileBackedGroundingProvider(
        provider_dir=provider_dir,
        match_iou_threshold=0.3,
    )

    result = validate_action_coordinates(
        action,
        confidence=0.9,
        capture_context=_capture_context(),
        recent_records=[],
        config=_config(),
        screenshot=Image.new("RGB", (1920, 1080), color="gray"),
        providers=[provider],
    )

    assert result.approved is True
    assert "external_provider_support" in result.reasons
    assert result.provider_assessments[0]["provider"] == "file_grounding"


def test_file_provider_refines_pixel_point_from_external_bbox(tmp_path) -> None:
    action = AgentAction(
        type="mouse_click",
        params=ActionParams(
            target_description="Search button",
            bbox_norm=[0.40, 0.20, 0.60, 0.40],
            center_norm=[0.50, 0.30],
        ),
    )
    provider_dir = tmp_path / "grounding" / "providers"
    provider_dir.mkdir(parents=True)
    payload = {
        "provider": "ocr",
        "anchors": [
            {
                "text": "Search button",
                "bbox_norm": [0.50, 0.27, 0.56, 0.33],
                "center_norm": [0.53, 0.30],
                "score": 0.99,
            }
        ],
    }
    (provider_dir / "ocr.json").write_text(json.dumps(payload), encoding="utf-8")
    provider = FileBackedGroundingProvider(
        provider_dir=provider_dir,
        match_iou_threshold=0.3,
    )

    result = validate_action_coordinates(
        action,
        confidence=0.9,
        capture_context=_capture_context(),
        recent_records=[],
        config=_config(),
        screenshot=Image.new("RGB", (1920, 1080), color="gray"),
        providers=[provider],
    )

    assert result.approved is True
    assert result.pixel_point == normalized_point_to_screen([0.53, 0.30], _capture_context())
    assert result.provider_assessments[0]["metadata"]["applied_refinement"] is True


def test_file_provider_ignores_single_character_substring_false_positive(tmp_path) -> None:
    action = AgentAction(
        type="mouse_click",
        params=ActionParams(
            target_description="The File menu at the top left of the VS Code window",
            bbox_norm=[0.01, 0.0, 0.025, 0.027],
            center_norm=[0.018, 0.014],
        ),
    )
    provider_dir = tmp_path / "grounding" / "providers"
    provider_dir.mkdir(parents=True)
    payload = {
        "provider": "ocr",
        "anchors": [
            {
                "text": "a",
                "bbox_norm": [0.70, 0.70, 0.75, 0.75],
                "center_norm": [0.725, 0.725],
                "score": 0.99,
            }
        ],
    }
    (provider_dir / "ocr.json").write_text(json.dumps(payload), encoding="utf-8")
    provider = FileBackedGroundingProvider(
        provider_dir=provider_dir,
        match_iou_threshold=0.3,
    )

    result = validate_action_coordinates(
        action,
        confidence=0.95,
        capture_context=_capture_context(),
        recent_records=[],
        config=_config(),
        screenshot=Image.new("RGB", (1920, 1080), color="gray"),
        providers=[provider],
    )

    assert result.approved is True
    assert "external_provider_conflict" not in result.reasons
    assert result.provider_assessments == []


def test_file_provider_refines_drag_end_pixel_point_from_external_bbox(tmp_path) -> None:
    action = AgentAction(
        type="mouse_drag",
        params=ActionParams(
            start_target_description="Task card",
            start_bbox_norm=[0.18, 0.22, 0.28, 0.34],
            start_center_norm=[0.23, 0.28],
            end_target_description="Done column",
            end_bbox_norm=[0.65, 0.18, 0.88, 0.82],
            end_center_norm=[0.76, 0.50],
        ),
    )
    provider_dir = tmp_path / "grounding" / "providers"
    provider_dir.mkdir(parents=True)
    payload = {
        "provider": "ocr",
        "anchors": [
            {
                "text": "Done column",
                "bbox_norm": [0.72, 0.22, 0.86, 0.78],
                "center_norm": [0.79, 0.50],
                "score": 0.99,
            }
        ],
    }
    (provider_dir / "ocr.json").write_text(json.dumps(payload), encoding="utf-8")
    provider = FileBackedGroundingProvider(
        provider_dir=provider_dir,
        match_iou_threshold=0.3,
    )

    result = validate_action_coordinates(
        action,
        confidence=0.9,
        capture_context=_capture_context(),
        recent_records=[],
        config=_config(),
        screenshot=Image.new("RGB", (1920, 1080), color="gray"),
        providers=[provider],
    )

    assert result.approved is True
    assert result.pixel_point_end == normalized_point_to_screen([0.79, 0.50], _capture_context())
    assert result.provider_assessments[0]["metadata"]["target_role"] == "end"