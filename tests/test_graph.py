"""Tests for the LangGraph workflow."""

from __future__ import annotations

from src.config import Settings


def test_settings_defaults() -> None:
    """Settings can be instantiated without environment variables."""
    settings = Settings()
    assert settings.model_name
    assert settings.temperature >= 0
