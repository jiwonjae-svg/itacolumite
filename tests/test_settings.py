"""Tests for the settings module."""

from itacolumite.config.settings import Settings, get_settings


def test_default_settings() -> None:
    """Settings should have sensible defaults."""
    s = Settings()
    assert s.gemini.gemini_model_fast == "gemini-2.5-flash"
    assert s.gemini.gemini_model_pro == "gemini-2.5-pro"
    assert s.gemini.gemini_auto_upgrade is True
    assert s.gemini.gemini_auto_upgrade_threshold == 0.3lash"
    assert s.gemini.gemini_model_pro == "gemini-2.5-pro"
    assert s.gemini.gemini_auto_upgrade is True
    assert s.gemini.gemini_auto_upgrade_threshold == 0.3
    assert s.agent.agent_max_steps == 200
    assert s.agent.shell_executable == "powershell.exe"
    assert s.agent.control_pipe_name == "itacolumite-control"


def test_dpi_awareness_default() -> None:
    """Agent should have DPI awareness enabled by default."""
    s = Settings()
    assert s.agent.dpi_awareness == "system-aware"


def test_capture_target_default() -> None:
    """Capture target should default to primary monitor."""
    s = Settings()
    assert s.agent.capture_target == "primary-monitor"
