"""Coordinate validation and conversion helpers for screen-grounded actions."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isnan
from typing import Any, Iterable

from PIL import Image

from itacolumite.ai.response_models import ActionParams, AgentAction
from itacolumite.core.grounding_providers import GroundingProvider, GroundingProviderContext
from itacolumite.core.memory import ActionRecord
from itacolumite.perception.screen import CaptureContext

_VALIDATED_POINTER_ACTIONS = frozenset({
    "mouse_click",
    "mouse_double_click",
    "mouse_move",
    "mouse_drag",
    "mouse_scroll",
})
_REQUIRED_COORDINATE_ACTIONS = frozenset({"mouse_click", "mouse_double_click", "mouse_move", "mouse_drag"})
_MIN_APPROVAL_SCORE = 0.5


@dataclass(frozen=True)
class ValidationConfig:
    """Runtime configuration for coordinate validation."""

    require_bbox: bool = True
    min_confidence: float = 0.45
    edge_margin_px: int = 8
    min_bbox_size_px: int = 12
    max_repeat_failures: int = 2


@dataclass
class ValidationResult:
    """Outcome of validating a proposed grounded action."""

    approved: bool
    action_type: str
    pixel_point: tuple[int, int] | None = None
    pixel_bbox: tuple[int, int, int, int] | None = None
    pixel_point_end: tuple[int, int] | None = None
    pixel_bbox_end: tuple[int, int, int, int] | None = None
    score: float = 1.0
    reasons: list[str] = field(default_factory=list)
    retry_hint: str | None = None
    repeat_failures: int = 0
    provider_assessments: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _TargetValidation:
    approved: bool
    pixel_point: tuple[int, int] | None = None
    pixel_bbox: tuple[int, int, int, int] | None = None
    score: float = 1.0
    reasons: list[str] = field(default_factory=list)
    provider_assessments: list[dict[str, Any]] = field(default_factory=list)


def needs_coordinate_validation(action_type: str) -> bool:
    """Return True when the action should pass through the grounding validator."""
    return action_type.lower() in _VALIDATED_POINTER_ACTIONS


def validate_action_coordinates(
    action: AgentAction,
    *,
    confidence: float,
    capture_context: CaptureContext,
    recent_records: Iterable[ActionRecord],
    config: ValidationConfig,
    screenshot: Image.Image | None = None,
    task: str = "",
    providers: Iterable[GroundingProvider] = (),
) -> ValidationResult:
    """Validate and score model-proposed pointing coordinates before execution."""
    action_type = action.type.lower()
    if not needs_coordinate_validation(action_type):
        return ValidationResult(approved=True, action_type=action_type)

    if action_type == "mouse_drag":
        return _validate_drag_action(
            action,
            confidence=confidence,
            capture_context=capture_context,
            recent_records=recent_records,
            config=config,
            screenshot=screenshot,
            task=task,
            providers=providers,
        )

    return _validate_single_pointer_action(
        action,
        confidence=confidence,
        capture_context=capture_context,
        recent_records=recent_records,
        config=config,
        screenshot=screenshot,
        task=task,
        providers=providers,
    )


def normalized_point_to_screen(
    center_norm: list[float],
    capture_context: CaptureContext,
) -> tuple[int, int]:
    """Convert a normalized point to screen pixels via capture-space pixels."""
    capture_x = round(center_norm[0] * max(capture_context.capture_width - 1, 1))
    capture_y = round(center_norm[1] * max(capture_context.capture_height - 1, 1))
    screen_x = capture_context.screen_left + round(
        capture_x * capture_context.screen_width / max(capture_context.capture_width, 1)
    )
    screen_y = capture_context.screen_top + round(
        capture_y * capture_context.screen_height / max(capture_context.capture_height, 1)
    )
    screen_x = min(max(screen_x, capture_context.screen_left), capture_context.screen_left + max(capture_context.screen_width - 1, 0))
    screen_y = min(max(screen_y, capture_context.screen_top), capture_context.screen_top + max(capture_context.screen_height - 1, 0))
    return screen_x, screen_y


def normalized_bbox_to_screen_bbox(
    bbox_norm: list[float],
    capture_context: CaptureContext,
) -> tuple[int, int, int, int]:
    """Convert a normalized bbox to screen pixel coordinates."""
    x1, y1 = normalized_point_to_screen([bbox_norm[0], bbox_norm[1]], capture_context)
    x2, y2 = normalized_point_to_screen([bbox_norm[2], bbox_norm[3]], capture_context)
    return x1, y1, x2, y2


def pixel_to_normalized_point(
    x: int,
    y: int,
    capture_context: CaptureContext,
) -> list[float]:
    """Backfill normalized coordinates from legacy absolute pixels."""
    local_x = x - capture_context.screen_left
    local_y = y - capture_context.screen_top
    capture_x = local_x * capture_context.capture_width / max(capture_context.screen_width, 1)
    capture_y = local_y * capture_context.capture_height / max(capture_context.screen_height, 1)
    return [
        min(max(capture_x / max(capture_context.capture_width - 1, 1), 0.0), 1.0),
        min(max(capture_y / max(capture_context.capture_height - 1, 1), 0.0), 1.0),
    ]


def build_retry_hint(reasons: list[str]) -> str:
    """Build a short retry hint that can be appended to the next prompt."""
    is_drag = any(reason.startswith("start_") or reason.startswith("end_") for reason in reasons)

    if is_drag and _has_reason(reasons, "repeat_failure_hotspot"):
        return "Do not retry the same drag path. Choose a different source or destination target."
    if is_drag and _has_reason(reasons, "missing_bbox"):
        return "For mouse_drag, return start_bbox_norm/start_center_norm and end_bbox_norm/end_center_norm."
    if is_drag and _has_reason(reasons, "missing_target_coordinates"):
        return "For mouse_drag, return both a grounded drag source and a grounded drag destination."
    if is_drag and (_has_reason(reasons, "center_norm_out_of_range") or _has_reason(reasons, "bbox_norm_out_of_range")):
        return "For mouse_drag, normalize both start and end coordinates to the 0.0-1.0 screenshot range."
    if is_drag and _has_reason(reasons, "low_confidence"):
        return "Do not guess the drag source or destination. Return clearer visible start and end targets."
    if is_drag and _has_reason(reasons, "external_provider_conflict"):
        return "The proposed drag path conflicts with OCR or detection evidence. Re-ground both drag endpoints."
    if is_drag and _has_reason(reasons, "near_screen_edge"):
        return "The drag source or destination is too close to the screen edge. Return tighter visible target boxes."
    if is_drag and (
        _has_reason(reasons, "low_crop_contrast")
        or _has_reason(reasons, "low_crop_structure")
        or _has_reason(reasons, "flat_local_crop")
    ):
        return "The proposed drag targets look visually weak. Return tighter boxes around the real drag handle and drop target."

    if _has_reason(reasons, "bbox_too_small"):
        return "Return a larger visible target region with bbox_norm and center_norm."
    if _has_reason(reasons, "bbox_too_large"):
        return "Return a tighter bbox around the actual clickable control, not a broad region."
    if _has_reason(reasons, "repeat_failure_hotspot"):
        return "Do not retry the same area. Pick a different visible target or alternate action."
    if _has_reason(reasons, "missing_bbox"):
        return "Return both bbox_norm and center_norm for the intended target."
    if _has_reason(reasons, "center_norm_out_of_range") or _has_reason(reasons, "bbox_norm_out_of_range"):
        return "Return normalized coordinates in the 0.0-1.0 range relative to the screenshot."
    if _has_reason(reasons, "low_confidence"):
        return "Do not guess. Describe the ambiguity and return a clearer target proposal."
    if (
        _has_reason(reasons, "low_crop_contrast")
        or _has_reason(reasons, "low_crop_structure")
        or _has_reason(reasons, "flat_local_crop")
    ):
        return "The proposed bbox looks visually weak. Return a tighter, more distinctive target region."
    if _has_reason(reasons, "external_provider_conflict"):
        return "The proposed target conflicts with OCR or detection evidence. Re-evaluate the visible control."
    if _has_reason(reasons, "near_screen_edge"):
        return "The target is too close to the screen edge. Return a tighter visible bbox around the actual control."
    return "Return a clearer visible target with normalized bbox_norm and center_norm."


def _validate_single_pointer_action(
    action: AgentAction,
    *,
    confidence: float,
    capture_context: CaptureContext,
    recent_records: Iterable[ActionRecord],
    config: ValidationConfig,
    screenshot: Image.Image | None,
    task: str,
    providers: Iterable[GroundingProvider],
) -> ValidationResult:
    params = action.params
    result = _validate_pointer_target(
        action_type=action.type.lower(),
        center_norm=params.center_norm,
        bbox_norm=params.bbox_norm,
        target_description=(params.target_description or "").strip(),
        legacy_point=(params.x, params.y) if params.x is not None and params.y is not None else None,
        capture_context=capture_context,
        config=config,
        confidence=confidence,
        screenshot=screenshot,
        task=task,
        providers=providers,
    )
    if not result.approved:
        return _blocked(
            action.type.lower(),
            _dedupe(result.reasons),
            score=max(result.score, 0.0),
            pixel_point=result.pixel_point,
            pixel_bbox=result.pixel_bbox,
            provider_assessments=result.provider_assessments,
        )

    score = result.score
    reasons = list(result.reasons)
    repeat_radius = max(config.edge_margin_px, config.min_bbox_size_px)
    repeat_failures = _count_repeat_failures(
        recent_records=recent_records,
        action_type=action.type.lower(),
        pixel_point=result.pixel_point,
        radius_px=repeat_radius,
    )
    if repeat_failures >= config.max_repeat_failures:
        reasons.append("repeat_failure_hotspot")
        return _blocked(
            action.type.lower(),
            _dedupe(reasons),
            score=max(score - 0.25, 0.0),
            repeat_failures=repeat_failures,
            pixel_point=result.pixel_point,
            pixel_bbox=result.pixel_bbox,
            provider_assessments=result.provider_assessments,
        )
    if repeat_failures > 0:
        score -= 0.25
        reasons.append("repeat_failure_history")

    score = max(score, 0.0)
    approved = score >= _MIN_APPROVAL_SCORE
    reasons = _dedupe(reasons)
    return ValidationResult(
        approved=approved,
        action_type=action.type.lower(),
        pixel_point=result.pixel_point,
        pixel_bbox=result.pixel_bbox,
        score=score,
        reasons=reasons,
        retry_hint=None if approved else build_retry_hint(reasons),
        repeat_failures=repeat_failures,
        provider_assessments=result.provider_assessments,
    )


def _validate_drag_action(
    action: AgentAction,
    *,
    confidence: float,
    capture_context: CaptureContext,
    recent_records: Iterable[ActionRecord],
    config: ValidationConfig,
    screenshot: Image.Image | None,
    task: str,
    providers: Iterable[GroundingProvider],
) -> ValidationResult:
    params = action.params
    fallback_description = (params.target_description or "").strip()
    start_result = _validate_pointer_target(
        action_type=action.type.lower(),
        center_norm=params.start_center_norm,
        bbox_norm=params.start_bbox_norm,
        target_description=(params.start_target_description or fallback_description).strip(),
        legacy_point=(params.x1, params.y1) if params.x1 is not None and params.y1 is not None else None,
        capture_context=capture_context,
        config=config,
        confidence=confidence,
        screenshot=screenshot,
        task=task,
        providers=providers,
        role="start",
        apply_confidence_penalty=True,
    )
    end_result = _validate_pointer_target(
        action_type=action.type.lower(),
        center_norm=params.end_center_norm,
        bbox_norm=params.end_bbox_norm,
        target_description=(params.end_target_description or fallback_description).strip(),
        legacy_point=(params.x2, params.y2) if params.x2 is not None and params.y2 is not None else None,
        capture_context=capture_context,
        config=config,
        confidence=confidence,
        screenshot=screenshot,
        task=task,
        providers=providers,
        role="end",
        apply_confidence_penalty=False,
    )

    reasons = _dedupe(start_result.reasons + end_result.reasons)
    provider_assessments = start_result.provider_assessments + end_result.provider_assessments

    if not start_result.approved or not end_result.approved:
        return _blocked(
            action.type.lower(),
            reasons,
            score=max(min(start_result.score, end_result.score), 0.0),
            pixel_point=start_result.pixel_point,
            pixel_bbox=start_result.pixel_bbox,
            pixel_point_end=end_result.pixel_point,
            pixel_bbox_end=end_result.pixel_bbox,
            provider_assessments=provider_assessments,
        )

    score = min(start_result.score, end_result.score)
    repeat_radius = max(config.edge_margin_px, config.min_bbox_size_px)
    repeat_failures = _count_repeat_drag_failures(
        recent_records=recent_records,
        start_point=start_result.pixel_point,
        end_point=end_result.pixel_point,
        radius_px=repeat_radius,
    )
    if repeat_failures >= config.max_repeat_failures:
        reasons.append("repeat_failure_hotspot")
        return _blocked(
            action.type.lower(),
            _dedupe(reasons),
            score=max(score - 0.25, 0.0),
            repeat_failures=repeat_failures,
            pixel_point=start_result.pixel_point,
            pixel_bbox=start_result.pixel_bbox,
            pixel_point_end=end_result.pixel_point,
            pixel_bbox_end=end_result.pixel_bbox,
            provider_assessments=provider_assessments,
        )
    if repeat_failures > 0:
        score -= 0.25
        reasons.append("repeat_failure_history")

    score = max(score, 0.0)
    approved = score >= _MIN_APPROVAL_SCORE
    reasons = _dedupe(reasons)
    return ValidationResult(
        approved=approved,
        action_type=action.type.lower(),
        pixel_point=start_result.pixel_point,
        pixel_bbox=start_result.pixel_bbox,
        pixel_point_end=end_result.pixel_point,
        pixel_bbox_end=end_result.pixel_bbox,
        score=score,
        reasons=reasons,
        retry_hint=None if approved else build_retry_hint(reasons),
        repeat_failures=repeat_failures,
        provider_assessments=provider_assessments,
    )


def _validate_pointer_target(
    *,
    action_type: str,
    center_norm: list[float] | None,
    bbox_norm: list[float] | None,
    target_description: str,
    legacy_point: tuple[int, int] | None,
    capture_context: CaptureContext,
    config: ValidationConfig,
    confidence: float,
    screenshot: Image.Image | None,
    task: str,
    providers: Iterable[GroundingProvider],
    role: str | None = None,
    apply_confidence_penalty: bool = True,
) -> _TargetValidation:
    reasons: list[str] = []
    score = 1.0
    pixel_bbox: tuple[int, int, int, int] | None = None
    working_center = list(center_norm) if center_norm is not None else None
    working_bbox = list(bbox_norm) if bbox_norm is not None else None
    using_legacy_pixels = False
    provider_assessments: list[dict[str, Any]] = []

    if working_bbox is not None:
        if not _is_valid_normalized_sequence(working_bbox, expected_len=4):
            reasons.append(_reason(role, "bbox_norm_out_of_range"))
            return _TargetValidation(approved=False, score=0.0, reasons=reasons)
        if working_bbox[0] > working_bbox[2] or working_bbox[1] > working_bbox[3]:
            reasons.append(_reason(role, "bbox_inverted"))
            return _TargetValidation(approved=False, score=0.0, reasons=reasons)
        pixel_bbox = normalized_bbox_to_screen_bbox(working_bbox, capture_context)
        if _bbox_size(pixel_bbox)[0] < config.min_bbox_size_px or _bbox_size(pixel_bbox)[1] < config.min_bbox_size_px:
            reasons.append(_reason(role, "bbox_too_small"))
            return _TargetValidation(approved=False, score=0.0, reasons=reasons, pixel_bbox=pixel_bbox)
        if working_center is None:
            working_center = _center_from_bbox(working_bbox)
            reasons.append(_reason(role, "center_derived_from_bbox"))
    elif working_center is not None and config.require_bbox:
        reasons.append(_reason(role, "missing_bbox"))
        return _TargetValidation(approved=False, score=0.0, reasons=reasons)

    if working_center is None:
        if legacy_point is not None:
            working_center = pixel_to_normalized_point(legacy_point[0], legacy_point[1], capture_context)
            using_legacy_pixels = True
            score -= 0.10
            reasons.append(_reason(role, "legacy_pixel_coordinates"))
        elif action_type in _REQUIRED_COORDINATE_ACTIONS:
            reasons.append(_reason(role, "missing_target_coordinates"))
            return _TargetValidation(approved=False, score=0.0, reasons=reasons)
        else:
            reasons.append(_reason(role, "using_executor_default_position"))
            return _TargetValidation(approved=True, score=0.9, reasons=reasons)

    if working_center is None or not _is_valid_normalized_sequence(working_center, expected_len=2):
        reasons.append(_reason(role, "center_norm_out_of_range"))
        return _TargetValidation(approved=False, score=0.0, reasons=reasons, pixel_bbox=pixel_bbox)

    pixel_point = normalized_point_to_screen(working_center, capture_context)
    if not _is_point_in_bounds(pixel_point, capture_context):
        reasons.append(_reason(role, "pixel_point_out_of_bounds"))
        return _TargetValidation(approved=False, score=0.0, reasons=reasons, pixel_point=pixel_point, pixel_bbox=pixel_bbox)

    if not using_legacy_pixels and working_bbox is None:
        score -= 0.10
        reasons.append(_reason(role, "missing_bbox"))

    if not target_description:
        score -= 0.05
        reasons.append(_reason(role, "missing_target_description"))

    if apply_confidence_penalty and confidence < config.min_confidence:
        score -= 0.20
        reasons.append(_reason(role, "low_confidence"))

    if _is_near_screen_edge(pixel_point, capture_context, config.edge_margin_px):
        score -= 0.15
        reasons.append(_reason(role, "near_screen_edge"))

    provider_context = GroundingProviderContext(
        task=task,
        screenshot=screenshot,
        capture_context=capture_context,
        pixel_point=pixel_point,
        pixel_bbox=pixel_bbox,
        confidence=confidence,
    )
    provider_blocking_reasons: list[str] = []
    provider_action = AgentAction(
        type=action_type,
        params=ActionParams(
            target_description=target_description,
            center_norm=working_center,
            bbox_norm=working_bbox,
        ),
    )
    for provider in providers:
        assessment = provider.evaluate(provider_action, provider_context)
        if assessment is None:
            continue
        if role is not None:
            assessment.reasons = [_reason(role, reason) for reason in assessment.reasons]
            assessment.blocking_reasons = [_reason(role, reason) for reason in assessment.blocking_reasons]
            assessment.metadata["target_role"] = role
        refined_context = _refine_context_from_assessment(assessment, provider_context)
        if refined_context is not None:
            provider_context = refined_context
            pixel_point = refined_context.pixel_point
            pixel_bbox = refined_context.pixel_bbox
            assessment.metadata["applied_refinement"] = True
            assessment.metadata["refined_pixel_point"] = list(pixel_point) if pixel_point is not None else None
            assessment.metadata["refined_pixel_bbox"] = list(pixel_bbox) if pixel_bbox is not None else None
        provider_assessments.append(assessment.to_dict())
        score += assessment.score_delta
        reasons.extend(assessment.reasons)
        if assessment.blocking_reasons:
            provider_blocking_reasons.extend(assessment.blocking_reasons)

    if provider_blocking_reasons and not _has_reason(reasons, "external_provider_support"):
        reasons.extend(provider_blocking_reasons)
        return _TargetValidation(
            approved=False,
            pixel_point=pixel_point,
            pixel_bbox=pixel_bbox,
            score=max(score, 0.0),
            reasons=_dedupe(reasons),
            provider_assessments=provider_assessments,
        )

    score = max(score, 0.0)
    reasons = _dedupe(reasons)
    return _TargetValidation(
        approved=score >= _MIN_APPROVAL_SCORE,
        pixel_point=pixel_point,
        pixel_bbox=pixel_bbox,
        score=score,
        reasons=reasons,
        provider_assessments=provider_assessments,
    )


def _blocked(
    action_type: str,
    reasons: list[str],
    *,
    score: float,
    repeat_failures: int = 0,
    pixel_point: tuple[int, int] | None = None,
    pixel_bbox: tuple[int, int, int, int] | None = None,
    pixel_point_end: tuple[int, int] | None = None,
    pixel_bbox_end: tuple[int, int, int, int] | None = None,
    provider_assessments: list[dict[str, Any]] | None = None,
) -> ValidationResult:
    return ValidationResult(
        approved=False,
        action_type=action_type,
        pixel_point=pixel_point,
        pixel_bbox=pixel_bbox,
        pixel_point_end=pixel_point_end,
        pixel_bbox_end=pixel_bbox_end,
        score=score,
        reasons=reasons,
        retry_hint=build_retry_hint(reasons),
        repeat_failures=repeat_failures,
        provider_assessments=provider_assessments or [],
    )


def _count_repeat_failures(
    *,
    recent_records: Iterable[ActionRecord],
    action_type: str,
    pixel_point: tuple[int, int] | None,
    radius_px: int,
) -> int:
    if pixel_point is None:
        return 0

    count = 0
    for record in recent_records:
        if record.action_type != action_type or record.result not in {"failure", "blocked"}:
            continue
        prior_x = record.params.get("x")
        prior_y = record.params.get("y")
        if prior_x is None or prior_y is None:
            continue
        if abs(int(prior_x) - pixel_point[0]) <= radius_px and abs(int(prior_y) - pixel_point[1]) <= radius_px:
            count += 1
    return count


def _count_repeat_drag_failures(
    *,
    recent_records: Iterable[ActionRecord],
    start_point: tuple[int, int] | None,
    end_point: tuple[int, int] | None,
    radius_px: int,
) -> int:
    if start_point is None or end_point is None:
        return 0

    count = 0
    for record in recent_records:
        if record.action_type != "mouse_drag" or record.result not in {"failure", "blocked"}:
            continue
        prior_x1 = record.params.get("x1")
        prior_y1 = record.params.get("y1")
        prior_x2 = record.params.get("x2")
        prior_y2 = record.params.get("y2")
        if None in {prior_x1, prior_y1, prior_x2, prior_y2}:
            continue
        if (
            abs(int(prior_x1) - start_point[0]) <= radius_px
            and abs(int(prior_y1) - start_point[1]) <= radius_px
            and abs(int(prior_x2) - end_point[0]) <= radius_px
            and abs(int(prior_y2) - end_point[1]) <= radius_px
        ):
            count += 1
    return count


def _is_valid_normalized_sequence(values: list[float], *, expected_len: int) -> bool:
    if len(values) != expected_len:
        return False
    for value in values:
        if isinstance(value, bool):
            return False
        numeric = float(value)
        if isnan(numeric) or numeric < 0.0 or numeric > 1.0:
            return False
    return True


def _center_from_bbox(bbox_norm: list[float]) -> list[float]:
    return [
        (bbox_norm[0] + bbox_norm[2]) / 2.0,
        (bbox_norm[1] + bbox_norm[3]) / 2.0,
    ]


def _is_point_in_bounds(point: tuple[int, int], capture_context: CaptureContext) -> bool:
    return (
        capture_context.screen_left <= point[0] < capture_context.screen_left + capture_context.screen_width
        and capture_context.screen_top <= point[1] < capture_context.screen_top + capture_context.screen_height
    )


def _is_near_screen_edge(
    point: tuple[int, int],
    capture_context: CaptureContext,
    margin_px: int,
) -> bool:
    max_x = capture_context.screen_left + max(capture_context.screen_width - 1, 0)
    max_y = capture_context.screen_top + max(capture_context.screen_height - 1, 0)
    return (
        point[0] <= capture_context.screen_left + margin_px
        or point[1] <= capture_context.screen_top + margin_px
        or point[0] >= max_x - margin_px
        or point[1] >= max_y - margin_px
    )


def _bbox_size(pixel_bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    return pixel_bbox[2] - pixel_bbox[0], pixel_bbox[3] - pixel_bbox[1]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _reason(role: str | None, reason: str) -> str:
    return f"{role}_{reason}" if role else reason


def _has_reason(reasons: list[str], reason: str) -> bool:
    return any(
        candidate == reason or candidate.endswith(f"_{reason}")
        for candidate in reasons
    )


def _refine_context_from_assessment(
    assessment: dict[str, Any] | Any,
    provider_context: GroundingProviderContext,
) -> GroundingProviderContext | None:
    metadata = getattr(assessment, "metadata", None)
    if not isinstance(metadata, dict):
        return None

    suggested_bbox = metadata.get("suggested_bbox_norm")
    suggested_center = metadata.get("suggested_center_norm")
    if suggested_bbox is not None and not _is_valid_normalized_sequence(suggested_bbox, expected_len=4):
        suggested_bbox = None
    if suggested_center is not None and not _is_valid_normalized_sequence(suggested_center, expected_len=2):
        suggested_center = None

    if suggested_bbox is None and suggested_center is None:
        return None

    refined_bbox: tuple[int, int, int, int] | None = None
    if suggested_bbox is not None:
        refined_bbox = normalized_bbox_to_screen_bbox(suggested_bbox, provider_context.capture_context)

    if suggested_center is None and suggested_bbox is not None:
        suggested_center = _center_from_bbox(suggested_bbox)
    refined_point = (
        normalized_point_to_screen(suggested_center, provider_context.capture_context)
        if suggested_center is not None
        else provider_context.pixel_point
    )

    return GroundingProviderContext(
        task=provider_context.task,
        screenshot=provider_context.screenshot,
        capture_context=provider_context.capture_context,
        pixel_point=refined_point,
        pixel_bbox=refined_bbox or provider_context.pixel_bbox,
        confidence=provider_context.confidence,
    )