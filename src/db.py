"""PostgreSQL 数据访问层。

提供：
- 同步连接池（后台扫描线程 / CLI 使用）
- 业务表建表语句：chat_records（用户聊天记录）、skill_registry（技能注册表）
- 聊天记录与技能注册表的 CRUD
- 运行控制（按 chat_id 启动/暂停/恢复）
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from pydantic import BaseModel

from .config import settings


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------- 数据模型 ----------
class ChatRecord(BaseModel):
    chat_id: str
    user_id: str | None = None
    content: str
    status: str = "pending"  # pending/running/paused/completed/failed
    pause_requested: bool = False
    response: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class SkillRow(BaseModel):
    skill_id: str
    name: str
    description: str
    skill_type: str  # category / atomic / dynamic
    level: int
    parent_id: str | None
    fs_path: str
    keywords: list[str] = []
    has_children: bool = False
    manifest: dict[str, Any] = {}
    prefetched: bool = False


# ---------- 建表 DDL ----------
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS chat_records (
    id              BIGSERIAL PRIMARY KEY,
    chat_id         VARCHAR(64) UNIQUE NOT NULL,
    user_id         VARCHAR(64),
    content         TEXT NOT NULL,
    status          VARCHAR(20) NOT NULL DEFAULT 'pending',
    pause_requested BOOLEAN NOT NULL DEFAULT FALSE,
    response        TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS skill_registry (
    skill_id     VARCHAR(255) PRIMARY KEY,
    name         VARCHAR(255) NOT NULL,
    description  TEXT NOT NULL DEFAULT '',
    skill_type   VARCHAR(20) NOT NULL,
    level        INTEGER NOT NULL,
    parent_id    VARCHAR(255),
    fs_path      TEXT NOT NULL,
    keywords     JSONB NOT NULL DEFAULT '[]'::jsonb,
    has_children BOOLEAN NOT NULL DEFAULT FALSE,
    prefetched   BOOLEAN NOT NULL DEFAULT FALSE,
    manifest     JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_skill_registry_level ON skill_registry(level);
CREATE INDEX IF NOT EXISTS idx_skill_registry_parent ON skill_registry(parent_id);
"""


class Database:
    """同步 PostgreSQL 访问封装（线程安全，连接池可被后台线程共享）。"""

    def __init__(self, dsn: str | None = None) -> None:
        self.dsn = dsn or settings.pg_dsn
        self._pool: ConnectionPool | None = None
        self._lock = threading.Lock()

    # ---------- 连接池生命周期 ----------
    def connect(self) -> None:
        with self._lock:
            if self._pool is None:
                self._pool = ConnectionPool(
                    conninfo=self.dsn,
                    min_size=settings.db_pool_min_size,
                    max_size=settings.db_pool_max_size,
                    kwargs={"row_factory": dict_row},
                    open=True,
                )

    def close(self) -> None:
        with self._lock:
            if self._pool is not None:
                self._pool.close()
                self._pool = None

    def __enter__(self) -> Database:  # noqa: PYI034
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def pool(self) -> ConnectionPool:
        if self._pool is None:
            raise RuntimeError("Database 尚未连接，请先调用 connect()")
        return self._pool

    # ---------- schema ----------
    def init_schema(self) -> None:
        with self.pool.connection() as conn:
            conn.execute(_SCHEMA_SQL)
            conn.commit()

    # ---------- chat_records ----------
    def upsert_chat(self, chat: ChatRecord) -> None:
        sql = """
            INSERT INTO chat_records (chat_id, user_id, content, status, pause_requested, response, updated_at)
            VALUES (%(chat_id)s, %(user_id)s, %(content)s, %(status)s, %(pause_requested)s, %(response)s, %(ts)s)
            ON CONFLICT (chat_id) DO UPDATE SET
                user_id         = EXCLUDED.user_id,
                content         = EXCLUDED.content,
                status          = EXCLUDED.status,
                pause_requested = EXCLUDED.pause_requested,
                response        = EXCLUDED.response,
                updated_at      = EXCLUDED.updated_at
        """
        with self.pool.connection() as conn:
            conn.execute(
                sql,
                {
                    "chat_id": chat.chat_id,
                    "user_id": chat.user_id,
                    "content": chat.content,
                    "status": chat.status,
                    "pause_requested": chat.pause_requested,
                    "response": chat.response,
                    "ts": _utcnow(),
                },
            )
            conn.commit()

    def get_chat(self, chat_id: str) -> ChatRecord | None:
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT * FROM chat_records WHERE chat_id = %s", (chat_id,)
            ).fetchone()
        return ChatRecord(**dict(row)) if row else None

    def list_chats(self, limit: int = 50) -> list[ChatRecord]:
        with self.pool.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM chat_records ORDER BY id DESC LIMIT %s", (limit,)
            ).fetchall()
        return [ChatRecord(**dict(r)) for r in rows]

    def get_pending_chat(self) -> ChatRecord | None:
        """读取一条待处理（pending）的聊天记录。"""
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT * FROM chat_records WHERE status = 'pending' ORDER BY id ASC LIMIT 1"
            ).fetchone()
        return ChatRecord(**dict(row)) if row else None

    def update_chat_status(
        self,
        chat_id: str,
        status: str,
        response: str | None = None,
        pause_requested: bool | None = None,
    ) -> None:
        sets = ["status = %s", "updated_at = %s"]
        params: list[Any] = [status, _utcnow()]
        if response is not None:
            sets.append("response = %s")
            params.append(response)
        if pause_requested is not None:
            sets.append("pause_requested = %s")
            params.append(pause_requested)
        params.append(chat_id)
        with self.pool.connection() as conn:
            conn.execute(
                f"UPDATE chat_records SET {', '.join(sets)} WHERE chat_id = %s", params
            )
            conn.commit()

    def request_pause(self, chat_id: str) -> None:
        """请求暂停：置位 pause_requested，运行中的 graph 会在步骤边界挂起。"""
        self.update_chat_status(chat_id, "paused", pause_requested=True)

    def request_resume(self, chat_id: str) -> None:
        """解除暂停标记，恢复执行时由 graph 重新置为 running。"""
        with self.pool.connection() as conn:
            conn.execute(
                "UPDATE chat_records SET pause_requested = FALSE, status = 'running', updated_at = %s "
                "WHERE chat_id = %s",
                (_utcnow(), chat_id),
            )
            conn.commit()

    # ---------- skill_registry ----------
    def upsert_skill(self, skill: SkillRow) -> None:
        sql = """
            INSERT INTO skill_registry
                (skill_id, name, description, skill_type, level, parent_id, fs_path,
                 keywords, has_children, prefetched, manifest, updated_at)
            VALUES
                (%(skill_id)s, %(name)s, %(description)s, %(skill_type)s, %(level)s,
                 %(parent_id)s, %(fs_path)s, %(keywords)s, %(has_children)s,
                 %(prefetched)s, %(manifest)s, %(ts)s)
            ON CONFLICT (skill_id) DO UPDATE SET
                name         = EXCLUDED.name,
                description  = EXCLUDED.description,
                skill_type   = EXCLUDED.skill_type,
                level        = EXCLUDED.level,
                parent_id    = EXCLUDED.parent_id,
                fs_path      = EXCLUDED.fs_path,
                keywords     = EXCLUDED.keywords,
                has_children = EXCLUDED.has_children,
                prefetched   = EXCLUDED.prefetched,
                manifest     = EXCLUDED.manifest,
                updated_at   = EXCLUDED.updated_at
        """
        import json

        with self.pool.connection() as conn:
            conn.execute(
                sql,
                {
                    "skill_id": skill.skill_id,
                    "name": skill.name,
                    "description": skill.description,
                    "skill_type": skill.skill_type,
                    "level": skill.level,
                    "parent_id": skill.parent_id,
                    "fs_path": skill.fs_path,
                    "keywords": json.dumps(skill.keywords, ensure_ascii=False),
                    "has_children": skill.has_children,
                    "prefetched": skill.prefetched,
                    "manifest": json.dumps(skill.manifest, ensure_ascii=False),
                    "ts": _utcnow(),
                },
            )
            conn.commit()

    def get_level1_skills(self) -> list[SkillRow]:
        """只读取一级 skills（后台线程预加载的那一层）。"""
        with self.pool.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM skill_registry WHERE level = 1 ORDER BY skill_id"
            ).fetchall()
        return [_row_to_skill(r) for r in rows]

    def get_skill(self, skill_id: str) -> SkillRow | None:
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT * FROM skill_registry WHERE skill_id = %s", (skill_id,)
            ).fetchone()
        return _row_to_skill(row) if row else None

    def delete_skill(self, skill_id: str) -> None:
        with self.pool.connection() as conn:
            conn.execute("DELETE FROM skill_registry WHERE skill_id = %s", (skill_id,))
            conn.commit()

    def prune_level1(self, valid_ids: set[str]) -> None:
        """删除磁盘上已不存在的一级 skill（保留更深层级的渐进式缓存）。"""
        with self.pool.connection() as conn:
            conn.execute(
                "DELETE FROM skill_registry WHERE level = 1 AND skill_id <> ALL(%s)",
                (list(valid_ids),),
            )
            conn.commit()


def _row_to_skill(row: dict[str, Any]) -> SkillRow:
    import json

    data = dict(row)
    if isinstance(data.get("keywords"), str):
        data["keywords"] = json.loads(data["keywords"])
    if isinstance(data.get("manifest"), str):
        data["manifest"] = json.loads(data["manifest"])
    return SkillRow(**data)


# 模块级单例，便于后台线程与主程序共享
db = Database()
