"""数据库表模型（Pydantic），与表一一对应。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ChatRecord(BaseModel):
    """chat_record：一条聊天消息（纯记录，不含任何执行态）。"""

    chat_id: str
    session_id: str
    user_id: str | None = None
    role: str = "user"  # user / assistant
    content: str
    created_at: datetime | None = None


class ChatTask(BaseModel):
    """chat_task：由路由从消息拆出的执行任务。"""

    task_id: str
    session_id: str
    title: str = ""
    content: str = ""
    entry_skill_id: str = ""
    arguments: dict[str, Any] = {}
    # collecting(信息不全,等后续消息) / pending(就绪待执行)
    # / running / paused / completed / failed
    status: str = "pending"
    pause_requested: bool = False
    output: str | None = None
    error: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class IntentMathRow(BaseModel):
    """intent_math：一条可被召回的用户说法。"""

    id: int | None = None
    msg: str
    score_limit: float = 0.45
    skill_id: str | None = None
    skill_level: int = 0
    answer: str = ""
    forbid: int = 0
    source: str = "seed"
    seed: str = ""
    enabled: bool = True
    msg_embedding: list[float] | None = None


class SkillRow(BaseModel):
    """skill_registry：一行技能元数据（后台扫描线程维护）。"""

    skill_id: str
    name: str
    description: str
    skill_type: str  # category / atomic / dynamic / react
    level: int
    parent_id: str | None
    fs_path: str
    keywords: list[str] = []
    has_children: bool = False
    manifest: dict[str, Any] = {}
    prefetched: bool = False
