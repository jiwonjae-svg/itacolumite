"""Tests for Gemini grounding provider extraction helpers."""

from __future__ import annotations

import json

from itacolumite.core.grounding_capture import (
    GroundingTextAnchor,
    parse_grounding_ocr_response,
    write_grounding_provider_payload,
)


def test_parse_grounding_ocr_response_accepts_json_fences() -> None:
    raw = """
    ```json
    {
      "items": [
        {
          "text": "Search",
          "bbox_norm": [0.6, 0.2, 0.4, 0.1],
          "center_norm": [0.5, 0.15],
          "confidence": 0.91
        },
        {
          "text": "Search",
          "bbox_norm": [0.4, 0.1, 0.6, 0.2],
          "confidence": 0.88
        }
      ]
    }
    ```
    """

    anchors = parse_grounding_ocr_response(raw, max_items=10)

    assert len(anchors) == 1
    assert anchors[0].text == "Search"
    assert anchors[0].bbox_norm == [0.4, 0.1, 0.6, 0.2]
    assert anchors[0].center_norm == [0.5, 0.15]


def test_write_grounding_provider_payload_writes_json(tmp_path) -> None:
    payload = {
        "provider": "gemini_ocr",
        "items": [
            GroundingTextAnchor(
                text="Open",
                bbox_norm=[0.1, 0.1, 0.2, 0.2],
                center_norm=[0.15, 0.15],
                confidence=0.8,
            ).to_provider_item()
        ],
    }

    output_path = write_grounding_provider_payload(
        tmp_path,
        payload,
        output_name="latest_provider",
    )

    assert output_path.name == "latest_provider.json"
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["provider"] == "gemini_ocr"
    assert saved["items"][0]["text"] == "Open"