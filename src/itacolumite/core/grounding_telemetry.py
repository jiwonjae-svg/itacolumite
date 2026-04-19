"""Structured telemetry logging for grounding validation tuning."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from itacolumite.ai.response_models import AgentAction
from itacolumite.config.settings import get_settings
from itacolumite.core.coordinate_validation import ValidationResult


class GroundingTelemetryLogger:
    """Append grounding validation and outcome events as JSONL."""

    def __init__(self, agent_data_dir: Path | None = None) -> None:
        root = agent_data_dir or get_settings().agent_data_dir
        self._grounding_dir = root / "grounding"
        self._grounding_dir.mkdir(parents=True, exist_ok=True)
        self._events_path = self._grounding_dir / "validation_events.jsonl"

    @property
    def events_path(self) -> Path:
        return self._events_path

    def record_validation(
        self,
        *,
        task_id: str,
        step: int,
        model: str,
        confidence: float,
        action: AgentAction,
        validation: ValidationResult,
    ) -> None:
        self._append({
            "event_type": "validation",
            "timestamp": datetime.now().isoformat(),
            "task_id": task_id,
            "step": step,
            "model": model,
            "confidence": confidence,
            "action_type": action.type,
            "action_params": action.params.model_dump(exclude_none=True),
            "approved": validation.approved,
            "score": validation.score,
            "reasons": validation.reasons,
            "pixel_point": list(validation.pixel_point) if validation.pixel_point is not None else None,
            "pixel_bbox": list(validation.pixel_bbox) if validation.pixel_bbox is not None else None,
            "pixel_point_end": list(validation.pixel_point_end) if validation.pixel_point_end is not None else None,
            "pixel_bbox_end": list(validation.pixel_bbox_end) if validation.pixel_bbox_end is not None else None,
            "provider_assessments": validation.provider_assessments,
            "retry_hint": validation.retry_hint,
            "repeat_failures": validation.repeat_failures,
        })

    def record_outcome(
        self,
        *,
        task_id: str,
        step: int,
        action_type: str,
        result: str,
        success: bool,
        diff_ratio: float | None,
        validation: ValidationResult | None,
    ) -> None:
        self._append({
            "event_type": "outcome",
            "timestamp": datetime.now().isoformat(),
            "task_id": task_id,
            "step": step,
            "action_type": action_type,
            "result": result,
            "success": success,
            "diff_ratio": diff_ratio,
            "score": validation.score if validation is not None else None,
            "reasons": validation.reasons if validation is not None else [],
            "provider_assessments": validation.provider_assessments if validation is not None else [],
        })

    def _append(self, payload: dict[str, Any]) -> None:
        with self._events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")