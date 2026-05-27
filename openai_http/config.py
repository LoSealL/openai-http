"""
Configuration module using pydantic-settings.

Supports both TOML config files and environment variables.
Environment variables take precedence over config file values.

Usage:
    from openai_http.config import get_settings
    settings = get_settings()
    print(settings.host)  # "0.0.0.0"
"""

import tomllib
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerSettings(BaseSettings):
    """Server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000


class AuthSettings(BaseSettings):
    """Authentication configuration."""
    enabled: bool = False
    api_keys: list[str] = Field(default_factory=list)


class QueueSettings(BaseSettings):
    """Request queue configuration."""
    depth: int = 32  # Max requests in queue before returning 429


class ObservabilitySettings(BaseSettings):
    """Observability configuration."""
    log_level: str = "info"
    log_format: str = "json"  # "json" or "text"
    metrics_enabled: bool = True
    metrics_port: int = 9464


class Settings(BaseSettings):
    """
    Main application settings.

    Merges configuration from:
    1. Default values
    2. config.toml file (if exists)
    3. Environment variables (highest priority)

    Environment variables use OPENAI_HTTP__ prefix with __ as nested delimiter.
    Examples:
        OPENAI_HTTP__SERVER__PORT=9000
        OPENAI_HTTP__AUTH__ENABLED=true
        OPENAI_HTTP__BACKEND__DEVICE=cuda
    """

    model_config = SettingsConfigDict(
        env_prefix="OPENAI_HTTP__",
        env_nested_delimiter="__",
        env_file=".env",
        extra="allow",
    )

    server: ServerSettings = ServerSettings()
    auth: AuthSettings = AuthSettings()
    queue: QueueSettings = QueueSettings()
    observability: ObservabilitySettings = ObservabilitySettings()

    @classmethod
    def from_toml(cls, config_path: str = "config.toml") -> "Settings":
        """
        Load settings from TOML file with environment variable override.

        Args:
            config_path: Path to TOML config file (default: "config.toml")

        Returns:
            Settings instance with values from TOML + env var overrides
        """
        config_file = Path(config_path)

        if config_file.exists():
            with open(config_file, "rb") as f:
                toml_data = tomllib.load(f)
        else:
            toml_data = {}

        # pydantic-settings automatically merges env vars (highest priority)
        return cls(**toml_data)


_settings: Optional[Settings] = None


def get_settings(config_path: str = "config.toml") -> Settings:
    """
    Get application settings (singleton).

    Loads from TOML file + environment variables on first call,
    returns cached instance on subsequent calls.

    Args:
        config_path: Path to TOML config file

    Returns:
        Settings instance
    """
    global _settings
    if _settings is None:
        _settings = Settings.from_toml(config_path)
    return _settings


def reset_settings() -> None:
    """Reset settings cache (useful for testing)."""
    global _settings
    _settings = None
