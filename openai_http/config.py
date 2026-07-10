"""
Copyright (C) 2026 The OPENAI-HTTP Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

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
from typing import ClassVar, Self

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
    """Request queue configuration.

    Controls the maximum number of requests that can wait in the queue
    before new requests receive a 429 Too Many Requests response.
    """

    depth: int = 32


class ObservabilitySettings(BaseSettings):
    """Observability configuration.

    Attributes:
        log_level: Logging verbosity (``"debug"``, ``"info"``,
            ``"warning"``, ``"error"``, ``"critical"``).
        log_format: Output format — ``"json"`` for structured JSON or
            ``"text"`` for plain text logs.
    """

    log_level: str = "info"
    log_format: str = "json"


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

    _singleton: ClassVar[Self | None] = None

    @classmethod
    def from_toml(cls, config_path: str = "config.toml") -> Self:
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

        return cls(**toml_data)

    @classmethod
    def get_settings(cls, config_path: str = "config.toml") -> Self:
        """
        Get application settings (singleton).

        Loads from TOML file + environment variables on first call,
        returns cached instance on subsequent calls.

        Args:
            config_path: Path to TOML config file

        Returns:
            Settings instance
        """
        if cls._singleton is None:
            cls._singleton = cls.from_toml(config_path)
        return cls._singleton

    @classmethod
    def reset_settings(cls) -> None:
        """Reset settings cache (useful for testing)."""
        cls._singleton = None


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
    return Settings.get_settings(config_path)


def reset_settings() -> None:
    """Reset settings cache (useful for testing)."""
    Settings.reset_settings()
