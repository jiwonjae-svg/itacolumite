"""Run OmniParser and convert its output into grounding provider payloads."""

from __future__ import annotations

import json
import shlex
import subprocess
import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from itacolumite.config.settings import Settings
from itacolumite.perception.screen import CaptureContext


class OmniParserRunner:
    """Execute a local OmniParser command and normalize its output."""

    def __init__(
        self,
        *,
        command: str,
        args_template: str,
        timeout_sec: int,
        workdir: str | None,
    ) -> None:
        self._command = _strip_wrapping_quotes(command.strip())
        self._args_template = args_template
        self._timeout_sec = timeout_sec
        self._workdir = workdir

    @classmethod
    def from_settings(cls, settings: Settings) -> OmniParserRunner:
        grounding = settings.grounding
        return cls(
            command=grounding.grounding_omniparser_command,
            args_template=grounding.grounding_omniparser_args,
            timeout_sec=grounding.grounding_omniparser_timeout_sec,
            workdir=grounding.grounding_omniparser_workdir,
        )

    @property
    def is_configured(self) -> bool:
        return bool(self._command)

    def extract_provider_payload(
        self,
        *,
        image_path: Path,
        capture_context: CaptureContext,
    ) -> dict[str, Any]:
        if not self.is_configured:
            raise ValueError("GROUNDING_OMNIPARSER_COMMAND is not configured")

        with tempfile.TemporaryDirectory(prefix="itacolumite_omniparser_") as temp_dir:
            output_path = Path(temp_dir) / "omniparser_output.json"
            command = self._build_command(
                image_path=image_path,
                output_path=output_path,
                capture_context=capture_context,
            )
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self._timeout_sec,
                cwd=self._workdir or None,
                check=False,
            )
            if completed.returncode != 0:
                stderr = (completed.stderr or "").strip()
                raise RuntimeError(
                    f"OmniParser command failed with exit code {completed.returncode}: {stderr or (completed.stdout or '').strip() or 'no output'}"
                )

            raw_output = output_path.read_text(encoding="utf-8") if output_path.exists() else (completed.stdout or "").strip()
            if not raw_output:
                raise RuntimeError("OmniParser command produced no JSON output")

        return build_omniparser_provider_payload(
            raw_output,
            capture_context=capture_context,
            source_image_path=image_path,
            command=command,
        )

    def _build_command(
        self,
        *,
        image_path: Path,
        output_path: Path,
        capture_context: CaptureContext,
    ) -> list[str]:
        command_args = self._args_template.format(
            image_path=str(image_path),
            output_path=str(output_path),
            screen_left=capture_context.screen_left,
            screen_top=capture_context.screen_top,
            screen_width=capture_context.screen_width,
            screen_height=capture_context.screen_height,
            capture_width=capture_context.capture_width,
            capture_height=capture_context.capture_height,
        )
        args = [_strip_wrapping_quotes(token) for token in shlex.split(command_args, posix=False)] if command_args.strip() else []
        return [self._command, *args]


def build_omniparser_provider_payload(
    raw_output: str | dict[str, Any] | list[Any],
    *,
    capture_context: CaptureContext,
    source_image_path: Path | None = None,
    command: list[str] | None = None,
) -> dict[str, Any]:
    """Normalize OmniParser JSON into the shared grounding provider format."""
    parsed = _coerce_json_payload(raw_output)
    items: list[dict[str, Any]] = []
    for raw_item in _extract_items(parsed):
        if not isinstance(raw_item, dict):
            continue

        label = _coerce_label(raw_item)
        if not label:
            continue
        bbox_norm = _coerce_bbox_norm(raw_item, capture_context)
        if bbox_norm is None:
            continue
        center_norm = _coerce_center_norm(raw_item, capture_context, bbox_norm)
        score = _coerce_score(raw_item)
        item: dict[str, Any] = {
            "label": label,
            "text": label,
            "bbox_norm": bbox_norm,
            "center_norm": center_norm,
            "score": score,
        }
        item_type = str(raw_item.get("type") or raw_item.get("category") or raw_item.get("kind") or "").strip()
        if item_type:
            item["type"] = item_type
        items.append(item)

    items.sort(key=lambda item: (-float(item.get("score") or 0.0), item["label"].casefold()))
    payload: dict[str, Any] = {
        "provider": "omniparser",
        "generated_at": datetime.now().isoformat(),
        "capture_context": asdict(capture_context),
        "items": items,
    }
    if source_image_path is not None:
        payload["source_image"] = str(source_image_path)
    if command is not None:
        payload["command"] = command
    return payload


def _coerce_json_payload(raw_output: str | dict[str, Any] | list[Any]) -> dict[str, Any] | list[Any]:
    if isinstance(raw_output, (dict, list)):
        return raw_output
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid OmniParser JSON: {exc}") from exc


def _extract_items(payload: dict[str, Any] | list[Any]) -> list[Any]:
    if isinstance(payload, list):
        return payload

    for key in ("parsed_content_list", "items", "detections", "results", "elements"):
        value = payload.get(key)
        if isinstance(value, list):
            return value

    data = payload.get("data")
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return _extract_items(data)
    return []


def _coerce_label(item: dict[str, Any]) -> str:
    label = str(
        item.get("label")
        or item.get("text")
        or item.get("content")
        or item.get("description")
        or item.get("name")
        or ""
    ).strip()
    if label:
        return label

    return str(item.get("type") or item.get("category") or item.get("kind") or "").strip()


def _coerce_bbox_norm(item: dict[str, Any], capture_context: CaptureContext) -> list[float] | None:
    raw_bbox = item.get("bbox_norm") or item.get("bbox") or item.get("box") or item.get("bbox_px")
    if isinstance(raw_bbox, dict):
        raw_bbox = [
            _first_present(raw_bbox, "x1", "left"),
            _first_present(raw_bbox, "y1", "top"),
            _first_present(raw_bbox, "x2", "right"),
            _first_present(raw_bbox, "y2", "bottom"),
        ]
    if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
        return None

    values = [float(value) for value in raw_bbox]
    if any(value < 0.0 or value > 1.0 for value in values):
        values = [
            (values[0] - capture_context.screen_left) / max(capture_context.screen_width - 1, 1),
            (values[1] - capture_context.screen_top) / max(capture_context.screen_height - 1, 1),
            (values[2] - capture_context.screen_left) / max(capture_context.screen_width - 1, 1),
            (values[3] - capture_context.screen_top) / max(capture_context.screen_height - 1, 1),
        ]

    x1, y1, x2, y2 = values
    left, right = sorted((min(max(x1, 0.0), 1.0), min(max(x2, 0.0), 1.0)))
    top, bottom = sorted((min(max(y1, 0.0), 1.0), min(max(y2, 0.0), 1.0)))
    return [left, top, right, bottom]


def _coerce_center_norm(
    item: dict[str, Any],
    capture_context: CaptureContext,
    bbox_norm: list[float],
) -> list[float]:
    raw_center = item.get("center_norm") or item.get("center") or item.get("center_px")
    if isinstance(raw_center, dict):
        raw_center = [raw_center.get("x"), raw_center.get("y")]
    if isinstance(raw_center, list) and len(raw_center) == 2:
        values = [float(value) for value in raw_center]
        if any(value < 0.0 or value > 1.0 for value in values):
            values = [
                (values[0] - capture_context.screen_left) / max(capture_context.screen_width - 1, 1),
                (values[1] - capture_context.screen_top) / max(capture_context.screen_height - 1, 1),
            ]
        return [min(max(values[0], 0.0), 1.0), min(max(values[1], 0.0), 1.0)]

    return [
        round((bbox_norm[0] + bbox_norm[2]) / 2.0, 6),
        round((bbox_norm[1] + bbox_norm[3]) / 2.0, 6),
    ]


def _coerce_score(item: dict[str, Any]) -> float:
    raw_score = item.get("score") or item.get("confidence") or item.get("prob") or 0.5
    numeric = float(raw_score)
    return min(max(numeric, 0.0), 1.0)


def _strip_wrapping_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None