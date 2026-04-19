"""CLI validation tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import click
from click.testing import CliRunner
import pytest

from itacolumite.interface import cli as cli_module


def test_ensure_gemini_ready_rejects_placeholder_key() -> None:
    """Task execution should stop immediately when .env still contains the template key."""
    with pytest.raises(click.ClickException, match="GEMINI_API_KEY is not configured"):
        cli_module._ensure_gemini_ready("your_api_key_here")


def test_ensure_gemini_ready_rejects_failed_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gemini connection failures should surface as a user-friendly CLI error."""

    class FailingGeminiClient:
        def validate_api_access(self) -> None:
            raise RuntimeError("403 forbidden")

    monkeypatch.setattr(cli_module, "GeminiClient", FailingGeminiClient)

    with pytest.raises(click.ClickException, match="Gemini API validation failed: 403 forbidden"):
        cli_module._ensure_gemini_ready("valid-test-key")


def test_task_command_fails_before_agent_start_when_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The task command should abort before importing or starting the agent when the key is missing."""
    fake_settings = SimpleNamespace(
        gemini=SimpleNamespace(gemini_api_key="your_api_key_here"),
        agent=SimpleNamespace(
            agent_log_level="INFO",
            control_pipe_name="itacolumite-control",
        ),
    )

    monkeypatch.setattr(cli_module, "get_settings", lambda: fake_settings)
    monkeypatch.setattr(cli_module, "setup_logging", lambda _level: None)

    runner = CliRunner()
    result = runner.invoke(cli_module.cli, ["task", "demo"])

    assert result.exit_code == 1
    assert "GEMINI_API_KEY is not configured" in result.output


def test_grounding_extract_text_command_reports_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_settings = SimpleNamespace(
        gemini=SimpleNamespace(gemini_api_key="valid-test-key"),
        grounding=SimpleNamespace(),
        agent=SimpleNamespace(agent_log_level="INFO"),
        agent_data_dir=tmp_path,
    )
    provider_path = tmp_path / "grounding" / "providers" / "gemini_ocr_latest.json"
    screenshot_path = tmp_path / "grounding" / "captures" / "capture.png"

    monkeypatch.setattr(cli_module, "get_settings", lambda: fake_settings)
    monkeypatch.setattr(cli_module, "setup_logging", lambda _level: None)
    monkeypatch.setattr(cli_module, "_ensure_gemini_ready", lambda _api_key=None: None)
    monkeypatch.setattr(
        cli_module,
        "_capture_grounding_text_provider",
        lambda _output_name, use_pro: (provider_path, 7, screenshot_path),
    )

    runner = CliRunner()
    result = runner.invoke(cli_module.cli, ["grounding", "extract-text"])

    assert result.exit_code == 0
    assert str(provider_path) in result.output
    assert "Anchors: 7" in result.output


def test_grounding_run_omniparser_command_reports_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_settings = SimpleNamespace(
        gemini=SimpleNamespace(gemini_api_key="valid-test-key"),
        grounding=SimpleNamespace(),
        agent=SimpleNamespace(agent_log_level="INFO"),
        agent_data_dir=tmp_path,
    )
    provider_path = tmp_path / "grounding" / "providers" / "omniparser_latest.json"
    screenshot_path = tmp_path / "grounding" / "captures" / "capture.png"

    monkeypatch.setattr(cli_module, "get_settings", lambda: fake_settings)
    monkeypatch.setattr(cli_module, "setup_logging", lambda _level: None)
    monkeypatch.setattr(
        cli_module,
        "_capture_omniparser_provider",
        lambda _output_name: (provider_path, 11, screenshot_path),
    )

    runner = CliRunner()
    result = runner.invoke(cli_module.cli, ["grounding", "run-omniparser"])

    assert result.exit_code == 0
    assert str(provider_path) in result.output
    assert "Anchors: 11" in result.output


def test_grounding_report_command_reports_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_settings = SimpleNamespace(
        gemini=SimpleNamespace(gemini_api_key="valid-test-key"),
        grounding=SimpleNamespace(grounding_reports_subdir="grounding/reports"),
        agent=SimpleNamespace(agent_log_level="INFO"),
        agent_data_dir=tmp_path,
    )
    report_path = tmp_path / "grounding" / "reports" / "grounding_report.html"
    summary = SimpleNamespace(
        total_events=4,
        total_validations=2,
        approved_validations=1,
        blocked_validations=1,
        approval_rate=0.5,
        total_outcomes=2,
        successful_outcomes=1,
        success_rate=0.5,
        average_score=0.62,
        average_diff_ratio=0.014,
        reason_counts=[("external_provider_support", 1)],
    )

    monkeypatch.setattr(cli_module, "get_settings", lambda: fake_settings)
    monkeypatch.setattr(cli_module, "setup_logging", lambda _level: None)
    monkeypatch.setattr(cli_module, "_build_grounding_report", lambda _events, _output: (report_path, summary))

    runner = CliRunner()
    result = runner.invoke(cli_module.cli, ["grounding", "report"])

    assert result.exit_code == 0
    assert str(report_path) in result.output
    assert "Grounding Telemetry" in result.output