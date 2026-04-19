"""Tests for the settings module."""

from itacolumite.config.settings import AgentSettings, GeminiSettings, GroundingSettings, Settings


def test_default_settings() -> None:
    """Settings classes should declare sensible defaults."""
    assert GeminiSettings.model_fields["gemini_model_fast"].default == "gemini-2.5-flash"
    assert GeminiSettings.model_fields["gemini_model_pro"].default == "gemini-2.5-pro"
    assert GeminiSettings.model_fields["gemini_use_pro_for_vision"].default is True
    assert GeminiSettings.model_fields["gemini_auto_upgrade"].default is True
    assert GeminiSettings.model_fields["gemini_auto_upgrade_threshold"].default == 0.3
    assert AgentSettings.model_fields["agent_max_steps"].default == 200
    assert AgentSettings.model_fields["shell_executable"].default == "powershell.exe"
    assert AgentSettings.model_fields["control_pipe_name"].default == "itacolumite-control"
    assert GroundingSettings.model_fields["grounding_use_normalized_coords"].default is True
    assert GroundingSettings.model_fields["grounding_require_bbox"].default is True
    assert GroundingSettings.model_fields["grounding_min_confidence"].default == 0.45
    assert GroundingSettings.model_fields["grounding_enable_local_crop_recheck"].default is True
    assert GroundingSettings.model_fields["grounding_min_crop_stddev"].default == 12.0
    assert GroundingSettings.model_fields["grounding_enable_external_providers"].default is True
    assert GroundingSettings.model_fields["grounding_provider_inputs_subdir"].default == "grounding/providers"
    assert GroundingSettings.model_fields["grounding_enable_gemini_ocr_provider"].default is True
    assert GroundingSettings.model_fields["grounding_auto_refresh_gemini_ocr"].default is True
    assert GroundingSettings.model_fields["grounding_gemini_ocr_output_name"].default == "gemini_ocr_latest.json"
    assert GroundingSettings.model_fields["grounding_enable_omniparser_runner"].default is False
    assert GroundingSettings.model_fields["grounding_auto_refresh_omniparser"].default is False
    assert GroundingSettings.model_fields["grounding_omniparser_command"].default == ""
    assert GroundingSettings.model_fields["grounding_omniparser_timeout_sec"].default == 60
    assert GroundingSettings.model_fields["grounding_omniparser_output_name"].default == "omniparser_latest.json"
    assert GroundingSettings.model_fields["grounding_reports_subdir"].default == "grounding/reports"


def test_dpi_awareness_default() -> None:
    """Agent should have DPI awareness enabled by default."""
    s = Settings(_env_file=None)
    assert s.agent.dpi_awareness == "system-aware"


def test_capture_target_default() -> None:
    """Capture target should default to primary monitor."""
    s = Settings(_env_file=None)
    assert s.agent.capture_target == "primary-monitor"


def test_capture_target_reads_agent_prefixed_env(monkeypatch) -> None:
    """AGENT_CAPTURE_TARGET should configure the capture mode used by the agent."""
    monkeypatch.delenv("CAPTURE_TARGET", raising=False)
    monkeypatch.setenv("AGENT_CAPTURE_TARGET", "virtual-desktop")

    s = Settings(_env_file=None)

    assert s.agent.capture_target == "virtual-desktop"
