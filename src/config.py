"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# 项目根目录（src 的上一级）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 默认 PostgreSQL DSN：本机 localhost:5432，库 graph-test，密码 123456
DEFAULT_PG_DSN = "postgresql://postgres:123456@localhost:5432/graph-test"


@dataclass
class Settings:
    """Global application settings."""

    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", ""))
    model_name: str = field(default_factory=lambda: os.getenv("MODEL_NAME", "gpt-4o-mini"))
    temperature: float = field(default_factory=lambda: float(os.getenv("TEMPERATURE", "0.7")))

    # PostgreSQL 连接串
    pg_dsn: str = field(default_factory=lambda: os.getenv("POSTGRES_DSN", DEFAULT_PG_DSN))

    # skills 多级目录根路径（后台线程从这里加载一级 skills）
    skills_dir: Path = field(
        default_factory=lambda: Path(os.getenv("SKILLS_DIR", str(PROJECT_ROOT / "skills")))
    )

    # 后台线程扫描 skills 目录的间隔（秒）
    skill_scan_interval: float = field(
        default_factory=lambda: float(os.getenv("SKILL_SCAN_INTERVAL", "10"))
    )

    # 连接池大小
    db_pool_min_size: int = field(
        default_factory=lambda: int(os.getenv("DB_POOL_MIN_SIZE", "1"))
    )
    db_pool_max_size: int = field(
        default_factory=lambda: int(os.getenv("DB_POOL_MAX_SIZE", "10"))
    )

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
