"""Global configuration loaded from environment and .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # src/itacolumite/config -> project root


class GeminiSettings(BaseSettings):
    """Gemini API configuration."""

    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    gemini_api_key: str = "your_api_key_here"
    gemini_model_fast: str = "gemini-2.5-flash"
    gemini_model_pro: str = "gemini-2.5-pro"
    gemini_use_pro_for_vision: bool = True
    gemini_auto_upgrade: bool = True
    gemini_auto_upgrade_threshold: float = 0.3
    gemini_temperature: float = 0.3
    gemini_max_tokens: int = 4096


class GroundingSettings(BaseSettings):
    """Screen grounding and coordinate validation configuration."""

    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    grounding_use_normalized_coords: bool = True
    grounding_require_bbox: bool = True
    grounding_min_confidence: float = 0.45
    grounding_edge_margin_px: int = 8
    grounding_min_bbox_size_px: int = 12
    grounding_max_repeat_failures: int = 2
    grounding_post_click_diff_threshold: float = 0.002
    grounding_enable_local_crop_recheck: bool = True
    grounding_local_crop_padding_px: int = 4
    grounding_min_crop_stddev: float = 12.0
    grounding_min_edge_density: float = 0.015
    grounding_max_bbox_area_ratio: float = 0.35
    grounding_enable_external_providers: bool = True
    grounding_provider_inputs_subdir: str = "grounding/providers"
    grounding_provider_match_iou_threshold: float = 0.3
    grounding_enable_gemini_ocr_provider: bool = True
    grounding_auto_refresh_gemini_ocr: bool = True
    grounding_gemini_ocr_output_name: str = "gemini_ocr_latest.json"
    grounding_gemini_ocr_max_items: int = 40
    grounding_enable_omniparser_runner: bool = False
    grounding_auto_refresh_omniparser: bool = False
    grounding_omniparser_command: str = ""
    grounding_omniparser_args: str = ""
    grounding_omniparser_workdir: str | None = None
    grounding_omniparser_timeout_sec: int = 60
    grounding_omniparser_output_name: str = "omniparser_latest.json"
    grounding_reports_subdir: str = "grounding/reports"


class AgentSettings(BaseSettings):
    """Agent behaviour configuration."""

    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    agent_max_steps: int = 200
    agent_log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # Timing (ms)
    typing_delay_ms: int = 50
    action_delay_ms: int = 200
    screenshot_delay_ms: int = 500

    # Display
    dpi_awareness: str = "system-aware"
    capture_target: Literal["primary-monitor", "virtual-desktop"] = Field(
        default="primary-monitor",
        validation_alias=AliasChoices("AGENT_CAPTURE_TARGET", "CAPTURE_TARGET"),
    )

    # Runtime
    shell_executable: str = "powershell.exe"
    control_pipe_name: str = "itacolumite-control"


class Settings(BaseSettings):
    """Unified top-level settings object."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gemini: GeminiSettings = Field(default_factory=GeminiSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    grounding: GroundingSettings = Field(default_factory=GroundingSettings)

    @property
    def project_root(self) -> Path:
        return _PROJECT_ROOT

    @property
    def agent_data_dir(self) -> Path:
        return _PROJECT_ROOT / "agent-data"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings instance."""
    return Settings()
