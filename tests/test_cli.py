"""CLI validation tests."""

from __future__ import annotations

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