"""Tests for grounding telemetry report summarization."""

from __future__ import annotations

import json

from itacolumite.core.grounding_report import (
    load_grounding_events,
    render_grounding_report_html,
    summarize_grounding_events,
)


def test_summarize_grounding_events_counts_metrics(tmp_path) -> None:
    events_path = tmp_path / "validation_events.jsonl"
    events_path.write_text(
        "\n".join([
            json.dumps({
                "event_type": "validation",
                "task_id": "task-1",
                "step": 1,
                "action_type": "mouse_click",
                "approved": True,
                "score": 0.81,
                "pixel_point": [320, 160],
                "reasons": ["external_provider_support"],
                "provider_assessments": [{"provider": "file_grounding"}],
            }),
            json.dumps({
                "event_type": "validation",
                "task_id": "task-1",
                "step": 2,
                "action_type": "mouse_click",
                "approved": False,
                "score": 0.31,
                "pixel_point": [352, 192],
                "reasons": ["flat_local_crop"],
                "provider_assessments": [{"provider": "local_crop"}],
            }),
            json.dumps({
                "event_type": "outcome",
                "task_id": "task-1",
                "step": 1,
                "action_type": "mouse_click",
                "success": True,
                "diff_ratio": 0.02,
            }),
            json.dumps({
                "event_type": "outcome",
                "task_id": "task-1",
                "step": 2,
                "action_type": "mouse_click",
                "success": False,
                "diff_ratio": 0.001,
            }),
        ]),
        encoding="utf-8",
    )

    summary = summarize_grounding_events(load_grounding_events(events_path), events_path=events_path)

    assert summary.total_events == 4
    assert summary.total_validations == 2
    assert summary.approved_validations == 1
    assert summary.total_outcomes == 2
    assert summary.successful_outcomes == 1
    assert summary.reason_counts[0][0] in {"external_provider_support", "flat_local_crop"}
    assert summary.provider_counts == [("file_grounding", 1), ("local_crop", 1)]
    assert summary.failure_hotspots[0].failures == 1
    assert summary.success_hotspots[0].successes == 1


def test_render_grounding_report_html_contains_key_sections(tmp_path) -> None:
    events_path = tmp_path / "validation_events.jsonl"
    summary = summarize_grounding_events([], events_path=events_path)

    html = render_grounding_report_html(summary)

    assert "Grounding Telemetry Report" in html
    assert str(events_path) in html
    assert "Score Distribution" in html
    assert "Hotspot Map" in html