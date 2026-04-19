"""Summaries and HTML reports for grounding validation telemetry."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from statistics import fmean
from typing import Any

_HOTSPOT_GRID_SIZE = 64


@dataclass(frozen=True)
class GroundingHotspot:
    """Aggregated success/failure statistics for a screen region."""

    x: int
    y: int
    total: int
    approved: int
    blocked: int
    successes: int
    failures: int
    average_score: float | None
    average_diff_ratio: float | None


@dataclass(frozen=True)
class GroundingTelemetrySummary:
    """Aggregated metrics for grounding validation telemetry."""

    events_path: Path
    total_events: int
    total_validations: int
    approved_validations: int
    blocked_validations: int
    total_outcomes: int
    successful_outcomes: int
    approval_rate: float
    success_rate: float | None
    average_score: float | None
    average_diff_ratio: float | None
    action_counts: list[tuple[str, int]]
    reason_counts: list[tuple[str, int]]
    provider_counts: list[tuple[str, int]]
    score_buckets: list[tuple[str, int]]
    failure_hotspots: list[GroundingHotspot]
    success_hotspots: list[GroundingHotspot]
    hotspot_canvas_size: tuple[int, int]


def load_grounding_events(events_path: Path) -> list[dict[str, Any]]:
    """Load grounding events from a JSONL file."""
    if not events_path.exists():
        raise FileNotFoundError(f"Grounding events file not found: {events_path}")

    events: list[dict[str, Any]] = []
    for raw_line in events_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            events.append(parsed)
    return events


def summarize_grounding_events(
    events: list[dict[str, Any]],
    *,
    events_path: Path,
) -> GroundingTelemetrySummary:
    """Aggregate key grounding telemetry metrics."""
    validations = [event for event in events if event.get("event_type") == "validation"]
    outcomes = [event for event in events if event.get("event_type") == "outcome"]

    action_counter: Counter[str] = Counter()
    reason_counter: Counter[str] = Counter()
    provider_counter: Counter[str] = Counter()
    score_bucket_counter: Counter[str] = Counter()
    validation_scores: list[float] = []
    diff_ratios: list[float] = []
    outcome_index: dict[tuple[str, int, str], dict[str, Any]] = {}
    max_hotspot_x = 1
    max_hotspot_y = 1

    for outcome in outcomes:
        task_id = str(outcome.get("task_id") or "")
        step = int(outcome.get("step") or -1)
        action_type = str(outcome.get("action_type") or "unknown")
        outcome_index[(task_id, step, action_type)] = outcome

    hotspot_buckets: dict[tuple[int, int], dict[str, Any]] = {}

    for validation in validations:
        action_counter[str(validation.get("action_type") or "unknown")] += 1

        for reason in validation.get("reasons") or []:
            reason_counter[str(reason)] += 1

        for provider in validation.get("provider_assessments") or []:
            if isinstance(provider, dict):
                provider_counter[str(provider.get("provider") or "unknown")] += 1

        score = validation.get("score")
        if isinstance(score, (int, float)):
            numeric_score = float(score)
            validation_scores.append(numeric_score)
            score_bucket_counter[_score_bucket_label(numeric_score)] += 1

        pixel_point = validation.get("pixel_point")
        if isinstance(pixel_point, list) and len(pixel_point) == 2:
            point_x = int(pixel_point[0])
            point_y = int(pixel_point[1])
            bucket = (
                (point_x // _HOTSPOT_GRID_SIZE) * _HOTSPOT_GRID_SIZE,
                (point_y // _HOTSPOT_GRID_SIZE) * _HOTSPOT_GRID_SIZE,
            )
            hotspot = hotspot_buckets.setdefault(
                bucket,
                {
                    "x": bucket[0],
                    "y": bucket[1],
                    "total": 0,
                    "approved": 0,
                    "blocked": 0,
                    "successes": 0,
                    "failures": 0,
                    "scores": [],
                    "diff_ratios": [],
                },
            )
            hotspot["total"] += 1
            if bool(validation.get("approved")):
                hotspot["approved"] += 1
            else:
                hotspot["blocked"] += 1
            if isinstance(score, (int, float)):
                hotspot["scores"].append(float(score))

            task_id = str(validation.get("task_id") or "")
            step = int(validation.get("step") or -1)
            action_type = str(validation.get("action_type") or "unknown")
            outcome = outcome_index.get((task_id, step, action_type))
            if outcome is not None:
                if bool(outcome.get("success")):
                    hotspot["successes"] += 1
                else:
                    hotspot["failures"] += 1
                diff_ratio = outcome.get("diff_ratio")
                if isinstance(diff_ratio, (int, float)):
                    hotspot["diff_ratios"].append(float(diff_ratio))

            max_hotspot_x = max(max_hotspot_x, point_x)
            max_hotspot_y = max(max_hotspot_y, point_y)

    for outcome in outcomes:
        diff_ratio = outcome.get("diff_ratio")
        if isinstance(diff_ratio, (int, float)):
            diff_ratios.append(float(diff_ratio))

    approved_validations = sum(1 for event in validations if bool(event.get("approved")))
    blocked_validations = len(validations) - approved_validations
    successful_outcomes = sum(1 for event in outcomes if bool(event.get("success")))
    hotspots = _build_hotspots(hotspot_buckets)

    return GroundingTelemetrySummary(
        events_path=events_path,
        total_events=len(events),
        total_validations=len(validations),
        approved_validations=approved_validations,
        blocked_validations=blocked_validations,
        total_outcomes=len(outcomes),
        successful_outcomes=successful_outcomes,
        approval_rate=(approved_validations / len(validations)) if validations else 0.0,
        success_rate=(successful_outcomes / len(outcomes)) if outcomes else None,
        average_score=fmean(validation_scores) if validation_scores else None,
        average_diff_ratio=fmean(diff_ratios) if diff_ratios else None,
        action_counts=_top_counts(action_counter),
        reason_counts=_top_counts(reason_counter),
        provider_counts=_top_counts(provider_counter),
        score_buckets=_sorted_bucket_counts(score_bucket_counter),
        failure_hotspots=sorted(hotspots, key=lambda item: (-item.failures, -item.blocked, -item.total, item.x, item.y))[:8],
        success_hotspots=sorted(hotspots, key=lambda item: (-item.successes, -item.total, item.x, item.y))[:8],
        hotspot_canvas_size=(max_hotspot_x, max_hotspot_y),
    )


def render_grounding_report_html(summary: GroundingTelemetrySummary) -> str:
    """Render a self-contained HTML report for grounding telemetry."""
    generated_at = datetime.now().isoformat()
    metrics = [
        ("Events", str(summary.total_events)),
        (
            "Validations",
            f"{summary.total_validations} total / {summary.approved_validations} approved / {summary.blocked_validations} blocked",
        ),
        ("Approval rate", f"{summary.approval_rate:.1%}"),
        ("Outcomes", f"{summary.total_outcomes} total / {summary.successful_outcomes} success"),
        (
            "Outcome success rate",
            f"{summary.success_rate:.1%}" if summary.success_rate is not None else "n/a",
        ),
        (
            "Average score",
            f"{summary.average_score:.3f}" if summary.average_score is not None else "n/a",
        ),
        (
            "Average diff ratio",
            f"{summary.average_diff_ratio:.4f}" if summary.average_diff_ratio is not None else "n/a",
        ),
    ]

    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>Grounding Telemetry Report</title>
  <style>
    :root {{
      --bg: #f6f2ea;
      --panel: #fffdf8;
      --ink: #1d1b18;
      --muted: #746a5c;
      --line: #d8cdbd;
      --accent: #0c7c59;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Georgia, \"Times New Roman\", serif; background: linear-gradient(180deg, #efe7db 0%, var(--bg) 100%); color: var(--ink); }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 32px 20px 48px; }}
    h1, h2 {{ margin: 0 0 12px; line-height: 1.1; }}
    p {{ margin: 0 0 10px; color: var(--muted); }}
    .hero {{ background: var(--panel); border: 1px solid var(--line); border-radius: 20px; padding: 24px; box-shadow: 0 18px 40px rgba(0,0,0,0.05); }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; margin-top: 20px; }}
    .card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 16px; padding: 18px; }}
    .metric-label {{ font-size: 0.92rem; color: var(--muted); margin-bottom: 6px; }}
    .metric-value {{ font-size: 1.2rem; font-weight: 700; color: var(--ink); }}
    .section-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; margin-top: 20px; }}
    .bar-row + .bar-row {{ margin-top: 12px; }}
    .bar-meta {{ display: flex; justify-content: space-between; gap: 12px; margin-bottom: 6px; font-size: 0.94rem; }}
    .bar-track {{ background: #f1e8db; border-radius: 999px; overflow: hidden; height: 10px; }}
    .bar-fill {{ background: linear-gradient(90deg, #2d9c74 0%, var(--accent) 100%); height: 10px; border-radius: 999px; }}
    .empty {{ color: var(--muted); font-style: italic; }}
    code {{ background: #f1e8db; border-radius: 6px; padding: 2px 6px; }}
  </style>
</head>
<body>
  <main>
    <section class=\"hero\">
      <h1>Grounding Telemetry Report</h1>
      <p>Source: <code>{escape(str(summary.events_path))}</code></p>
      <p>Generated at: {escape(generated_at)}</p>
      <div class=\"grid\">
        {''.join(_render_metric_card(label, value) for label, value in metrics)}
      </div>
    </section>
    <section class=\"section-grid\">
      <div class=\"card\">
        <h2>Top Reasons</h2>
        {_render_bar_rows(summary.reason_counts)}
      </div>
      <div class=\"card\">
        <h2>Actions</h2>
        {_render_bar_rows(summary.action_counts)}
      </div>
      <div class=\"card\">
        <h2>Providers</h2>
        {_render_bar_rows(summary.provider_counts)}
      </div>
      <div class=\"card\">
        <h2>Score Distribution</h2>
        {_render_bar_rows(summary.score_buckets)}
      </div>
            <div class="card">
                <h2>Hotspot Map</h2>
                {_render_hotspot_svg(summary.failure_hotspots + [item for item in summary.success_hotspots if item not in summary.failure_hotspots], summary.hotspot_canvas_size)}
            </div>
            <div class="card">
                <h2>Failure Hotspots</h2>
                {_render_hotspot_rows(summary.failure_hotspots, mode="failure")}
            </div>
            <div class="card">
                <h2>Success Hotspots</h2>
                {_render_hotspot_rows(summary.success_hotspots, mode="success")}
            </div>
    </section>
  </main>
</body>
</html>
"""


def write_grounding_report(output_path: Path, html: str) -> Path:
    """Write the rendered grounding report to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def _render_metric_card(label: str, value: str) -> str:
    return (
        '<div class="card">'
        f'<div class="metric-label">{escape(label)}</div>'
        f'<div class="metric-value">{escape(value)}</div>'
        '</div>'
    )


def _render_bar_rows(rows: list[tuple[str, int]]) -> str:
    if not rows:
        return '<p class="empty">No data recorded yet.</p>'

    max_count = max(count for _label, count in rows) or 1
    parts: list[str] = []
    for label, count in rows:
        width = 10.0 + (90.0 * count / max_count)
        parts.append(
            '<div class="bar-row">'
            f'<div class="bar-meta"><span>{escape(label)}</span><span>{count}</span></div>'
            '<div class="bar-track">'
            f'<div class="bar-fill" style="width: {width:.1f}%"></div>'
            '</div>'
            '</div>'
        )
    return ''.join(parts)


def _render_hotspot_rows(rows: list[GroundingHotspot], *, mode: str) -> str:
    if not rows:
        return '<p class="empty">No hotspot data recorded yet.</p>'

    parts: list[str] = []
    for hotspot in rows:
        count = hotspot.failures + hotspot.blocked if mode == "failure" else hotspot.successes
        parts.append(
            '<div class="bar-row">'
            f'<div class="bar-meta"><span>({hotspot.x}, {hotspot.y})</span><span>{count}</span></div>'
            f'<div class="metric-label">total={hotspot.total} approved={hotspot.approved} blocked={hotspot.blocked} success={hotspot.successes} failure={hotspot.failures}</div>'
            '</div>'
        )
    return ''.join(parts)


def _render_hotspot_svg(rows: list[GroundingHotspot], canvas_size: tuple[int, int]) -> str:
    if not rows:
        return '<p class="empty">No hotspot points available yet.</p>'

    max_x = max(canvas_size[0], 1)
    max_y = max(canvas_size[1], 1)
    svg_width = 560
    svg_height = 320
    max_total = max(hotspot.total for hotspot in rows) or 1
    circles: list[str] = []
    for hotspot in rows:
        x_pos = 16 + (hotspot.x / max_x) * (svg_width - 32)
        y_pos = 16 + (hotspot.y / max_y) * (svg_height - 32)
        radius = 5 + (hotspot.total / max_total) * 12
        fail_ratio = (hotspot.failures + hotspot.blocked) / max(hotspot.total, 1)
        fill = "#c0392b" if fail_ratio >= 0.5 else "#0c7c59"
        circles.append(
            f'<circle cx="{x_pos:.1f}" cy="{y_pos:.1f}" r="{radius:.1f}" fill="{fill}" fill-opacity="0.68">'
            f'<title>({hotspot.x}, {hotspot.y}) total={hotspot.total} success={hotspot.successes} failure={hotspot.failures} blocked={hotspot.blocked}</title>'
            '</circle>'
        )
    return (
        f'<svg viewBox="0 0 {svg_width} {svg_height}" width="100%" height="320" role="img" aria-label="Grounding hotspots">'
        '<rect x="0" y="0" width="100%" height="100%" rx="18" fill="#f7efe4"></rect>'
        '<rect x="16" y="16" width="528" height="288" rx="14" fill="#fffdf8" stroke="#d8cdbd"></rect>'
        + ''.join(circles)
        + '</svg>'
    )


def _top_counts(counter: Counter[str], limit: int = 10) -> list[tuple[str, int]]:
    items = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return items[:limit]


def _sorted_bucket_counts(counter: Counter[str]) -> list[tuple[str, int]]:
    return sorted(counter.items(), key=lambda item: float(item[0].split("-", maxsplit=1)[0]))


def _score_bucket_label(score: float) -> str:
    bounded = min(max(score, 0.0), 1.0)
    bucket_index = min(int(bounded * 10), 9)
    start = bucket_index / 10.0
    end = start + 0.1
    return f"{start:.1f}-{end:.1f}"


def _build_hotspots(hotspot_buckets: dict[tuple[int, int], dict[str, Any]]) -> list[GroundingHotspot]:
    hotspots: list[GroundingHotspot] = []
    for bucket in hotspot_buckets.values():
        scores = bucket.get("scores") or []
        diff_ratios = bucket.get("diff_ratios") or []
        hotspots.append(
            GroundingHotspot(
                x=int(bucket["x"]),
                y=int(bucket["y"]),
                total=int(bucket["total"]),
                approved=int(bucket["approved"]),
                blocked=int(bucket["blocked"]),
                successes=int(bucket["successes"]),
                failures=int(bucket["failures"]),
                average_score=fmean(scores) if scores else None,
                average_diff_ratio=fmean(diff_ratios) if diff_ratios else None,
            )
        )
    return hotspots