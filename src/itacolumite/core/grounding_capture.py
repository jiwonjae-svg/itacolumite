"""Capture and generate grounding provider files from Gemini OCR-style extraction."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from itacolumite.ai.gemini_client import GeminiClient
from itacolumite.perception.screen import CaptureContext

_GROUNDING_OCR_SYSTEM_INSTRUCTION = """You extract visible UI text anchors from a Windows screenshot.
Return valid JSON only with this shape:
{
  "items": [
    {
      "text": "Visible UI label",
      "bbox_norm": [x_min, y_min, x_max, y_max],
      "center_norm": [x, y],
      "confidence": 0.0
    }
  ]
}

Rules:
- Use normalized coordinates in the 0.0-1.0 range relative to the screenshot.
- Keep bbox_norm tight around the actual visible text or control label.
- Return only clearly visible, actionable UI text anchors.
- Omit uncertain guesses.
- Do not wrap the JSON in markdown fences.
"""


@dataclass(frozen=True)
class GroundingTextAnchor:
    """A single OCR-style text anchor extracted from a screenshot."""

    text: str
    bbox_norm: list[float]
    center_norm: list[float]
    confidence: float

    def to_provider_item(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "label": self.text,
            "bbox_norm": self.bbox_norm,
            "center_norm": self.center_norm,
            "score": round(self.confidence, 4),
        }


class GeminiGroundingExtractor:
    """Generate OCR-style provider payloads using Gemini vision."""

    def __init__(self, client: GeminiClient | None = None) -> None:
        self._client = client or GeminiClient()

    def extract_provider_payload(
        self,
        *,
        image_bytes: bytes,
        capture_context: CaptureContext,
        use_pro: bool,
        max_items: int,
        source_image_path: Path | None = None,
    ) -> dict[str, Any]:
        raw_response = self._client.generate_with_image(
            text_prompt=build_grounding_ocr_prompt(capture_context, max_items=max_items),
            image_bytes=image_bytes,
            system_instruction=_GROUNDING_OCR_SYSTEM_INSTRUCTION,
            use_pro=use_pro,
        )
        anchors = parse_grounding_ocr_response(raw_response, max_items=max_items)
        payload: dict[str, Any] = {
            "provider": "gemini_ocr",
            "generated_at": datetime.now().isoformat(),
            "capture_context": asdict(capture_context),
            "items": [anchor.to_provider_item() for anchor in anchors],
        }
        if source_image_path is not None:
            payload["source_image"] = str(source_image_path)
        return payload


def build_grounding_ocr_prompt(capture_context: CaptureContext, *, max_items: int) -> str:
    """Build the prompt used for OCR-style provider extraction."""
    return (
        "Extract visible UI text anchors from this screenshot for downstream grounding validation.\n"
        f"Screenshot size: {capture_context.capture_width}x{capture_context.capture_height}.\n"
        f"Return at most {max_items} items.\n"
        "Include only text that is visible enough to target reliably."
    )


def parse_grounding_ocr_response(raw_response: str, *, max_items: int) -> list[GroundingTextAnchor]:
    """Parse a Gemini OCR response into normalized provider items."""
    payload_text = _strip_markdown_fences(raw_response)
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        repaired = _repair_common_json_escapes(payload_text)
        if repaired != payload_text:
            try:
                payload = json.loads(repaired)
            except json.JSONDecodeError as repaired_exc:
                raise ValueError(f"Invalid grounding OCR JSON: {repaired_exc}") from repaired_exc
        else:
            raise ValueError(f"Invalid grounding OCR JSON: {exc}") from exc

    raw_items: Any
    if isinstance(payload, list):
        raw_items = payload
    elif isinstance(payload, dict):
        raw_items = payload.get("items") or payload.get("anchors") or payload.get("detections") or []
    else:
        raw_items = []

    anchors: list[GroundingTextAnchor] = []
    seen: set[tuple[str, tuple[float, float, float, float]]] = set()
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            continue
        text = str(raw_item.get("text") or raw_item.get("label") or raw_item.get("name") or "").strip()
        if not text:
            continue

        bbox_norm = _coerce_bbox_norm(raw_item.get("bbox_norm") or raw_item.get("bbox"))
        if bbox_norm is None:
            continue
        center_norm = _coerce_center_norm(raw_item.get("center_norm") or raw_item.get("center"), bbox_norm)
        confidence = _clamp_unit(raw_item.get("confidence") or raw_item.get("score") or 0.5)

        dedupe_key = (text.casefold(), tuple(round(value, 4) for value in bbox_norm))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        anchors.append(
            GroundingTextAnchor(
                text=text,
                bbox_norm=bbox_norm,
                center_norm=center_norm,
                confidence=confidence,
            )
        )

    anchors.sort(key=lambda item: (-item.confidence, item.text.casefold()))
    return anchors[:max_items]


def save_grounding_capture_image(
    agent_data_dir: Path,
    image_bytes: bytes,
    *,
    timestamp: float,
) -> Path:
    """Persist the screenshot that produced a provider file."""
    capture_dir = agent_data_dir / "grounding" / "captures"
    capture_dir.mkdir(parents=True, exist_ok=True)
    timestamp_str = datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S_%f")
    output_path = capture_dir / f"grounding_capture_{timestamp_str}.png"
    output_path.write_bytes(image_bytes)
    return output_path


def write_grounding_provider_payload(
    provider_dir: Path,
    payload: dict[str, Any],
    *,
    output_name: str,
) -> Path:
    """Write a provider payload JSON file to disk."""
    provider_dir.mkdir(parents=True, exist_ok=True)
    file_name = output_name if output_name.endswith(".json") else f"{output_name}.json"
    output_path = provider_dir / file_name
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def _strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _repair_common_json_escapes(text: str) -> str:
    """Escape invalid backslashes so slightly malformed model JSON still parses."""
    return re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', text)


def _coerce_bbox_norm(raw_bbox: Any) -> list[float] | None:
    if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
        return None
    values = [_clamp_unit(value) for value in raw_bbox]
    x1, y1, x2, y2 = values
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    return [left, top, right, bottom]


def _coerce_center_norm(raw_center: Any, bbox_norm: list[float]) -> list[float]:
    if isinstance(raw_center, list) and len(raw_center) == 2:
        return [_clamp_unit(raw_center[0]), _clamp_unit(raw_center[1])]
    return [
        round((bbox_norm[0] + bbox_norm[2]) / 2.0, 6),
        round((bbox_norm[1] + bbox_norm[3]) / 2.0, 6),
    ]


def _clamp_unit(value: Any) -> float:
    numeric = float(value)
    return min(max(numeric, 0.0), 1.0)