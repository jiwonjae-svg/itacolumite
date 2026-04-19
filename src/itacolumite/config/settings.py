"""Global configuration loaded from environment and .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # src/itacolumite/config -> project root


class GeminiSettings(BaseSettings):
    """Gemini API configuration."""

    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    gemini_api_key: str = "your_api_key_here"
    gemini_model_fast: str = "gemini-2.5-flash"
    gemini_model_pro: str = "gemini-2.5-pro"
    gemini_auto_upgrade: bool = True
    gemini_auto_upgrade_threshold: float = 0.3
    gemini_temperature: float = 0.3
    gemini_max_tokens: int = 4096


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
    capture_target: str = "primary-monitor"

    # Runtime
    shell_executable: str = "powershell.exe"
    control_pipe_name: str = "itacolumite-control"


class Settings(BaseSettings):
    """Unified top-level settings object."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gemini: GeminiSettings = Field(default_factory=GeminiSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)

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
