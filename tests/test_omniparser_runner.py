"""Tests for OmniParser native runner integration."""

from __future__ import annotations

import json
import sys

import pytest

from itacolumite.core.omniparser_runner import OmniParserRunner, build_omniparser_provider_payload
from itacolumite.perception.screen import CaptureContext


def _capture_context() -> CaptureContext:
    return CaptureContext(
        screen_width=3840,
        screen_height=1080,
        capture_width=3840,
        capture_height=1080,
        timestamp=0.0,
        screen_left=-1920,
        screen_top=0,
        capture_target="virtual-desktop",
    )


def test_build_omniparser_provider_payload_normalizes_absolute_pixels() -> None:
    payload = {
        "parsed_content_list": [
            {
                "text": "Search",
                "bbox": [-1600, 100, -1400, 200],
                "confidence": 0.93,
                "type": "button",
            }
        ]
    }

    result = build_omniparser_provider_payload(
        payload,
        capture_context=_capture_context(),
    )

    assert result["provider"] == "omniparser"
    assert result["items"][0]["label"] == "Search"
    assert result["items"][0]["bbox_norm"] == pytest.approx([0.0834, 0.0927, 0.1355, 0.1854], abs=1e-4)


def test_build_omniparser_provider_payload_accepts_dict_bbox_with_zero_origin() -> None:
    payload = {
        "parsed_content_list": [
            {
                "text": "Origin item",
                "bbox": {"x1": 0, "y1": 0, "x2": 100, "y2": 50},
                "confidence": 0.93,
            }
        ]
    }

    result = build_omniparser_provider_payload(
        payload,
        capture_context=CaptureContext(
            screen_width=1920,
            screen_height=1080,
            capture_width=1920,
            capture_height=1080,
            timestamp=0.0,
        ),
    )

    assert result["items"][0]["label"] == "Origin item"
    assert result["items"][0]["bbox_norm"] == pytest.approx([0.0, 0.0, 0.0521, 0.0463], abs=1e-4)


def test_omniparser_runner_executes_command_and_reads_output(tmp_path) -> None:
    script_path = tmp_path / "fake_omniparser.py"
    script_path.write_text(
        """
import json
import sys
from pathlib import Path

output_index = sys.argv.index('--output') + 1
output_path = Path(sys.argv[output_index])
payload = {
    'parsed_content_list': [
        {
            'text': 'Open File',
            'bbox': [100, 80, 260, 160],
            'confidence': 0.88,
        }
    ]
}
output_path.write_text(json.dumps(payload), encoding='utf-8')
""".strip(),
        encoding="utf-8",
    )
    image_path = tmp_path / "capture.png"
    image_path.write_bytes(b"not-a-real-png")
    runner = OmniParserRunner(
        command=sys.executable,
        args_template=f'"{script_path}" --image "{{image_path}}" --output "{{output_path}}"',
        timeout_sec=10,
        workdir=None,
    )

    payload = runner.extract_provider_payload(
        image_path=image_path,
        capture_context=CaptureContext(
            screen_width=1920,
            screen_height=1080,
            capture_width=1920,
            capture_height=1080,
            timestamp=0.0,
        ),
    )

    assert payload["provider"] == "omniparser"
    assert payload["items"][0]["label"] == "Open File"
    assert payload["items"][0]["bbox_norm"] == pytest.approx([0.0521, 0.0741, 0.1355, 0.1483], abs=1e-4)