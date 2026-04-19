"""Grounding evidence providers for local and external re-validation."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

from itacolumite.ai.response_models import AgentAction
from itacolumite.config.settings import Settings
from itacolumite.perception.screen import CaptureContext

logger = logging.getLogger(__name__)

_EXTERNAL_PROVIDER_SNAP_DISTANCE = 0.08
_EXTERNAL_PROVIDER_CONFLICT_DISTANCE = 0.20


@dataclass(frozen=True)
class GroundingProviderContext:
    """Runtime context supplied to grounding evidence providers."""

    task: str
    screenshot: Image.Image | None
    capture_context: CaptureContext
    pixel_point: tuple[int, int] | None
    pixel_bbox: tuple[int, int, int, int] | None
    confidence: float


@dataclass
class GroundingProviderAssessment:
    """Score adjustments and metadata produced by a grounding provider."""

    provider: str
    score_delta: float = 0.0
    reasons: list[str] = field(default_factory=list)
    blocking_reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GroundingProvider(Protocol):
    """Protocol for provider-specific grounding checks."""

    name: str

    def evaluate(
        self,
        action: AgentAction,
        context: GroundingProviderContext,
    ) -> GroundingProviderAssessment | None:
        """Return score adjustments for the grounded action proposal."""


class LocalCropGroundingProvider:
    """Re-score a proposal by inspecting the proposed bbox crop locally."""

    name = "local_crop"

    def __init__(
        self,
        *,
        crop_padding_px: int,
        min_crop_stddev: float,
        min_edge_density: float,
        max_bbox_area_ratio: float,
    ) -> None:
        self._crop_padding_px = crop_padding_px
        self._min_crop_stddev = min_crop_stddev
        self._min_edge_density = min_edge_density
        self._max_bbox_area_ratio = max_bbox_area_ratio

    def evaluate(
        self,
        action: AgentAction,
        context: GroundingProviderContext,
    ) -> GroundingProviderAssessment | None:
        if context.pixel_bbox is None:
            return None
        if not isinstance(context.screenshot, Image.Image):
            return None

        left, top, right, bottom = _expand_bbox(
            _screen_bbox_to_capture_bbox(context.pixel_bbox, context.capture_context),
            width=context.screenshot.width,
            height=context.screenshot.height,
            padding=self._crop_padding_px,
        )
        if right <= left or bottom <= top:
            return GroundingProviderAssessment(
                provider=self.name,
                score_delta=-0.15,
                reasons=["invalid_local_crop"],
            )

        crop = context.screenshot.crop((left, top, right + 1, bottom + 1)).convert("L")
        gray = np.asarray(crop, dtype=np.float32)
        if gray.size == 0:
            return GroundingProviderAssessment(
                provider=self.name,
                score_delta=-0.15,
                reasons=["empty_local_crop"],
            )

        stddev = float(gray.std())
        grad_x = np.diff(gray, axis=1)
        grad_y = np.diff(gray, axis=0)
        edge_mask_x = np.abs(grad_x) > 18.0 if grad_x.size else np.asarray([], dtype=bool)
        edge_mask_y = np.abs(grad_y) > 18.0 if grad_y.size else np.asarray([], dtype=bool)
        edge_pixels = int(edge_mask_x.sum()) + int(edge_mask_y.sum())
        edge_denominator = max(edge_mask_x.size + edge_mask_y.size, 1)
        edge_density = float(edge_pixels) / edge_denominator
        crop_area = crop.width * crop.height
        screen_area = max(context.capture_context.screen_width * context.capture_context.screen_height, 1)
        bbox_area_ratio = float(crop_area) / screen_area

        assessment = GroundingProviderAssessment(
            provider=self.name,
            metadata={
                "crop_stddev": round(stddev, 4),
                "edge_density": round(edge_density, 6),
                "bbox_area_ratio": round(bbox_area_ratio, 6),
                "crop_size": [crop.width, crop.height],
            },
        )
        if bbox_area_ratio > self._max_bbox_area_ratio:
            assessment.score_delta -= 0.20
            assessment.reasons.append("bbox_too_large")
        if stddev < self._min_crop_stddev:
            assessment.score_delta -= 0.15
            assessment.reasons.append("low_crop_contrast")
        if edge_density < self._min_edge_density:
            assessment.score_delta -= 0.15
            assessment.reasons.append("low_crop_structure")
        if stddev < self._min_crop_stddev * 0.5 and edge_density < self._min_edge_density * 0.5:
            assessment.score_delta -= 0.15
            assessment.reasons.append("flat_local_crop")
            assessment.blocking_reasons.append("flat_local_crop")
        if not assessment.reasons and stddev >= self._min_crop_stddev * 1.25 and edge_density >= self._min_edge_density * 1.25:
            assessment.score_delta += 0.05
            assessment.reasons.append("local_crop_support")
        return assessment


class FileBackedGroundingProvider:
    """Read OCR/OmniParser-style detections from JSON files and compare them locally."""

    name = "file_grounding"

    def __init__(self, *, provider_dir: Path, match_iou_threshold: float) -> None:
        self._provider_dir = provider_dir
        self._match_iou_threshold = match_iou_threshold

    def evaluate(
        self,
        action: AgentAction,
        context: GroundingProviderContext,
    ) -> GroundingProviderAssessment | None:
        target_description = (action.params.target_description or "").strip()
        if not target_description:
            return None

        provider_items = list(self._load_provider_items(context.capture_context))
        if not provider_items:
            return None

        target_tokens = _tokenize(target_description)
        best_match: dict[str, Any] | None = None
        best_similarity = 0.0
        for item in provider_items:
            label = item["label"]
            candidate_tokens = _tokenize(label)
            similarity = _token_overlap_ratio(target_tokens, candidate_tokens)
            if target_description.casefold() in label.casefold() or label.casefold() in target_description.casefold():
                similarity = max(similarity, 1.0)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = item

        if best_match is None or best_similarity <= 0.0:
            return None

        assessment = GroundingProviderAssessment(
            provider=self.name,
            metadata={
                "source_file": best_match["source_file"],
                "provider": best_match["provider"],
                "matched_label": best_match["label"],
                "token_similarity": round(best_similarity, 4),
            },
        )

        action_bbox_norm = action.params.bbox_norm
        match_bbox_norm = best_match.get("bbox_norm")
        action_center_norm = action.params.center_norm
        match_center_norm = best_match.get("center_norm")
        if action_bbox_norm is not None and match_bbox_norm is not None:
            iou = _bbox_iou(action_bbox_norm, match_bbox_norm)
            action_center = action_center_norm or _center_from_bbox(action_bbox_norm)
            match_center = match_center_norm or _center_from_bbox(match_bbox_norm)
            distance = _center_distance(action_center, match_center)
            assessment.metadata["bbox_iou"] = round(iou, 4)
            assessment.metadata["center_distance"] = round(distance, 4)
            if iou >= self._match_iou_threshold:
                assessment.score_delta += 0.12
                assessment.reasons.append("external_provider_support")
                _attach_refinement_metadata(assessment, match_bbox_norm, match_center)
            elif distance <= _EXTERNAL_PROVIDER_SNAP_DISTANCE and best_similarity >= 0.5:
                assessment.score_delta += 0.10
                assessment.reasons.extend(["external_provider_support", "external_provider_refined_target"])
                _attach_refinement_metadata(assessment, match_bbox_norm, match_center)
            elif distance >= _EXTERNAL_PROVIDER_CONFLICT_DISTANCE:
                assessment.score_delta -= 0.20
                assessment.reasons.append("external_provider_conflict")
        elif action_center_norm is not None and match_center_norm is not None:
            distance = _center_distance(action_center_norm, match_center_norm)
            assessment.metadata["center_distance"] = round(distance, 4)
            if distance <= _EXTERNAL_PROVIDER_SNAP_DISTANCE:
                assessment.score_delta += 0.10
                assessment.reasons.extend(["external_provider_support", "external_provider_refined_target"])
                if match_bbox_norm is not None:
                    _attach_refinement_metadata(assessment, match_bbox_norm, match_center_norm)
            elif distance >= _EXTERNAL_PROVIDER_CONFLICT_DISTANCE:
                assessment.score_delta -= 0.20
                assessment.reasons.append("external_provider_conflict")
        return assessment if assessment.reasons else None

    def _load_provider_items(
        self,
        capture_context: CaptureContext,
    ) -> list[dict[str, Any]]:
        if not self._provider_dir.exists():
            return []

        items: list[dict[str, Any]] = []
        for file_path in sorted(self._provider_dir.glob("*.json"), reverse=True):
            try:
                payload = json.loads(file_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Failed to read grounding provider file %s: %s", file_path, exc)
                continue

            provider_name = str(payload.get("provider") or file_path.stem)
            raw_items = payload.get("items") or payload.get("detections") or payload.get("anchors") or payload.get("results") or []
            for raw_item in raw_items:
                if not isinstance(raw_item, dict):
                    continue
                label = str(raw_item.get("label") or raw_item.get("text") or raw_item.get("name") or raw_item.get("target") or "").strip()
                if not label:
                    continue
                bbox_norm = _coerce_bbox_norm(raw_item, capture_context)
                center_norm = _coerce_center_norm(raw_item, capture_context, bbox_norm)
                items.append({
                    "provider": provider_name,
                    "label": label,
                    "bbox_norm": bbox_norm,
                    "center_norm": center_norm,
                    "source_file": str(file_path),
                })
        return items


def build_default_grounding_providers(settings: Settings) -> list[GroundingProvider]:
    """Build the default provider chain for grounding re-validation."""
    providers: list[GroundingProvider] = []
    grounding = settings.grounding
    if grounding.grounding_enable_external_providers:
        providers.append(
            FileBackedGroundingProvider(
                provider_dir=settings.agent_data_dir / grounding.grounding_provider_inputs_subdir,
                match_iou_threshold=grounding.grounding_provider_match_iou_threshold,
            )
        )
    if grounding.grounding_enable_local_crop_recheck:
        providers.append(
            LocalCropGroundingProvider(
                crop_padding_px=grounding.grounding_local_crop_padding_px,
                min_crop_stddev=grounding.grounding_min_crop_stddev,
                min_edge_density=grounding.grounding_min_edge_density,
                max_bbox_area_ratio=grounding.grounding_max_bbox_area_ratio,
            )
        )
    return providers


def _expand_bbox(
    bbox: tuple[int, int, int, int],
    *,
    width: int,
    height: int,
    padding: int,
) -> tuple[int, int, int, int]:
    left = max(bbox[0] - padding, 0)
    top = max(bbox[1] - padding, 0)
    right = min(bbox[2] + padding, width - 1)
    bottom = min(bbox[3] + padding, height - 1)
    return left, top, right, bottom


def _screen_bbox_to_capture_bbox(
    bbox: tuple[int, int, int, int],
    capture_context: CaptureContext,
) -> tuple[int, int, int, int]:
    return (
        bbox[0] - capture_context.screen_left,
        bbox[1] - capture_context.screen_top,
        bbox[2] - capture_context.screen_left,
        bbox[3] - capture_context.screen_top,
    )


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[0-9A-Za-z가-힣]+", text.casefold()))


def _token_overlap_ratio(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return float(len(left & right)) / float(max(len(left), len(right), 1))


def _bbox_iou(left: list[float], right: list[float]) -> float:
    left_x1, left_y1, left_x2, left_y2 = left
    right_x1, right_y1, right_x2, right_y2 = right
    inter_x1 = max(left_x1, right_x1)
    inter_y1 = max(left_y1, right_y1)
    inter_x2 = min(left_x2, right_x2)
    inter_y2 = min(left_y2, right_y2)
    inter_width = max(inter_x2 - inter_x1, 0.0)
    inter_height = max(inter_y2 - inter_y1, 0.0)
    inter_area = inter_width * inter_height
    if inter_area <= 0.0:
        return 0.0
    left_area = max(left_x2 - left_x1, 0.0) * max(left_y2 - left_y1, 0.0)
    right_area = max(right_x2 - right_x1, 0.0) * max(right_y2 - right_y1, 0.0)
    union_area = left_area + right_area - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def _center_distance(left: list[float], right: list[float]) -> float:
    dx = left[0] - right[0]
    dy = left[1] - right[1]
    return float((dx * dx + dy * dy) ** 0.5)


def _center_from_bbox(bbox_norm: list[float]) -> list[float]:
    return [
        (bbox_norm[0] + bbox_norm[2]) / 2.0,
        (bbox_norm[1] + bbox_norm[3]) / 2.0,
    ]


def _attach_refinement_metadata(
    assessment: GroundingProviderAssessment,
    bbox_norm: list[float] | None,
    center_norm: list[float] | None,
) -> None:
    if bbox_norm is not None:
        assessment.metadata["suggested_bbox_norm"] = [round(value, 6) for value in bbox_norm]
    if center_norm is not None:
        assessment.metadata["suggested_center_norm"] = [round(value, 6) for value in center_norm]


def _coerce_bbox_norm(
    item: dict[str, Any],
    capture_context: CaptureContext,
) -> list[float] | None:
    raw_bbox = item.get("bbox_norm") or item.get("bbox") or item.get("bbox_px")
    if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
        return None
    if any(float(value) < 0.0 or float(value) > 1.0 for value in raw_bbox):
        return [
            min(max((float(raw_bbox[0]) - capture_context.screen_left) / max(capture_context.screen_width - 1, 1), 0.0), 1.0),
            min(max((float(raw_bbox[1]) - capture_context.screen_top) / max(capture_context.screen_height - 1, 1), 0.0), 1.0),
            min(max((float(raw_bbox[2]) - capture_context.screen_left) / max(capture_context.screen_width - 1, 1), 0.0), 1.0),
            min(max((float(raw_bbox[3]) - capture_context.screen_top) / max(capture_context.screen_height - 1, 1), 0.0), 1.0),
        ]
    return [float(value) for value in raw_bbox]


def _coerce_center_norm(
    item: dict[str, Any],
    capture_context: CaptureContext,
    bbox_norm: list[float] | None,
) -> list[float] | None:
    raw_center = item.get("center_norm") or item.get("center") or item.get("center_px")
    if isinstance(raw_center, list) and len(raw_center) == 2:
        if any(float(value) < 0.0 or float(value) > 1.0 for value in raw_center):
            return [
                min(max((float(raw_center[0]) - capture_context.screen_left) / max(capture_context.screen_width - 1, 1), 0.0), 1.0),
                min(max((float(raw_center[1]) - capture_context.screen_top) / max(capture_context.screen_height - 1, 1), 0.0), 1.0),
            ]
        return [float(value) for value in raw_center]
    if bbox_norm is not None:
        return [
            (bbox_norm[0] + bbox_norm[2]) / 2.0,
            (bbox_norm[1] + bbox_norm[3]) / 2.0,
        ]
    return None