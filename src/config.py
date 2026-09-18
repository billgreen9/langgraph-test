"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    """Global application settings."""

    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", ""))
    model_name: str = field(default_factory=lambda: os.getenv("MODEL_NAME", "gpt-4o-mini"))
    temperature: float = field(default_factory=lambda: float(os.getenv("TEMPERATURE", "0.7")))

    def get_llm_kwargs(self) -> dict[str, Any]:
        """Return kwargs for instantiating a ChatOpenAI model."""
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "temperature": self.temperature,
        }
        if self.openai_base_url:
            kwargs["base_url"] = self.openai_base_url
        if self.openai_api_key:
            kwargs["api_key"] = self.openai_api_key
        return kwargs


settings = Settings()
