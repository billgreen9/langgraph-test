"""PostgreSQL 数据访问层。

提供：
- 同步连接池（后台扫描线程 / CLI 使用）
- 业务表：
  * chat_record：纯聊天消息时间线（user / assistant）
  * chat_task：路由拆出的执行任务（执行态/暂停/产物的唯一载体）
  * chat_task_message：任务与消息的多对多关联
  * skill_registry：技能注册表（后台扫描线程维护）
- 上述各表的 CRUD 与运行控制（按 task_id 启动/暂停/恢复）
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
    """一条聊天消息（纯记录，不含任何执行态）。"""

    chat_id: str
    session_id: str
    user_id: str | None = None
    role: str = "user"  # user / assistant
    content: str
    created_at: datetime | None = None


class ChatTask(BaseModel):
    """由路由从消息拆出的执行任务。"""

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


class SkillRow(BaseModel):
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


# ---------- 建表 DDL ----------
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS chat_record (
    id          BIGSERIAL PRIMARY KEY,
    chat_id     VARCHAR(64) UNIQUE NOT NULL,
    session_id  VARCHAR(64) NOT NULL,
    user_id     VARCHAR(64),
    role        VARCHAR(16) NOT NULL DEFAULT 'user',
    content     TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_chat_record_session ON chat_record(session_id, created_at);

CREATE TABLE IF NOT EXISTS chat_task (
    id              BIGSERIAL PRIMARY KEY,
    task_id         VARCHAR(64) UNIQUE NOT NULL,
    session_id      VARCHAR(64) NOT NULL,
    title           VARCHAR(255) NOT NULL DEFAULT '',
    content         TEXT NOT NULL DEFAULT '',
    entry_skill_id  VARCHAR(255) NOT NULL,
    arguments       JSONB NOT NULL DEFAULT '{}'::jsonb,
    status          VARCHAR(20) NOT NULL DEFAULT 'collecting',
    pause_requested BOOLEAN NOT NULL DEFAULT FALSE,
    output          TEXT,
    error           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_chat_task_session ON chat_task(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_task_status  ON chat_task(status);

CREATE TABLE IF NOT EXISTS chat_task_message (
    id         BIGSERIAL PRIMARY KEY,
    task_id    VARCHAR(64) NOT NULL REFERENCES chat_task(task_id),
    chat_id    VARCHAR(64) NOT NULL REFERENCES chat_record(chat_id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (task_id, chat_id)
);
CREATE INDEX IF NOT EXISTS idx_ctm_chat ON chat_task_message(chat_id);
CREATE INDEX IF NOT EXISTS idx_ctm_task ON chat_task_message(task_id);

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

# 旧版单体 chat_records 表（含 status/response/pause_requested 列）的特征列
_LEGACY_CHECK_SQL = """
SELECT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'chat_records' AND column_name = 'status'
) AS is_legacy
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
            # 旧版单体表无迁移价值（均为测试数据），检测到直接丢弃重建
            legacy = conn.execute(_LEGACY_CHECK_SQL).fetchone()
            if legacy and legacy["is_legacy"]:
                conn.execute("DROP TABLE IF EXISTS chat_records CASCADE")
            conn.execute(_SCHEMA_SQL)
            conn.commit()

    # ---------- chat_record ----------
    def insert_message(self, chat: ChatRecord) -> ChatRecord:
        with self.pool.connection() as conn:
            row = conn.execute(
                """
                INSERT INTO chat_record (chat_id, session_id, user_id, role, content)
                VALUES (%(chat_id)s, %(session_id)s, %(user_id)s, %(role)s, %(content)s)
                RETURNING created_at
                """,
                {
                    "chat_id": chat.chat_id,
                    "session_id": chat.session_id,
                    "user_id": chat.user_id,
                    "role": chat.role,
                    "content": chat.content,
                },
            ).fetchone()
            conn.commit()
        chat.created_at = row["created_at"]
        return chat

    def get_message(self, chat_id: str) -> ChatRecord | None:
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT * FROM chat_record WHERE chat_id = %s", (chat_id,)
            ).fetchone()
        return _row_to_message(row) if row else None

    def list_messages(self, session_id: str, limit: int = 100) -> list[ChatRecord]:
        with self.pool.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM chat_record WHERE session_id = %s "
                "ORDER BY id ASC LIMIT %s",
                (session_id, limit),
            ).fetchall()
        return [_row_to_message(r) for r in rows]

    def list_recent_messages(self, limit: int = 50) -> list[ChatRecord]:
        with self.pool.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM chat_record ORDER BY id DESC LIMIT %s", (limit,)
            ).fetchall()
        return [_row_to_message(r) for r in rows]

    # ---------- chat_task ----------
    def insert_task(self, task: ChatTask) -> ChatTask:
        import json

        with self.pool.connection() as conn:
            row = conn.execute(
                """
                INSERT INTO chat_task
                    (task_id, session_id, title, content, entry_skill_id, arguments,
                     status, pause_requested)
                VALUES
                    (%(task_id)s, %(session_id)s, %(title)s, %(content)s,
                     %(entry_skill_id)s, %(arguments)s, %(status)s, FALSE)
                RETURNING created_at, updated_at
                """,
                {
                    "task_id": task.task_id,
                    "session_id": task.session_id,
                    "title": task.title,
                    "content": task.content,
                    "entry_skill_id": task.entry_skill_id,
                    "arguments": json.dumps(task.arguments, ensure_ascii=False),
                    "status": task.status,
                },
            ).fetchone()
            conn.commit()
        task.created_at = row["created_at"]
        task.updated_at = row["updated_at"]
        return task

    def get_task(self, task_id: str) -> ChatTask | None:
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT * FROM chat_task WHERE task_id = %s", (task_id,)
            ).fetchone()
        return _row_to_task(row) if row else None

    def list_tasks(
        self, session_id: str | None = None, status: str | None = None,
        limit: int = 100,
    ) -> list[ChatTask]:
        sql = "SELECT * FROM chat_task WHERE 1=1"
        params: list[Any] = []
        if session_id is not None:
            sql += " AND session_id = %s"
            params.append(session_id)
        if status is not None:
            sql += " AND status = %s"
            params.append(status)
        sql += " ORDER BY id ASC LIMIT %s"
        params.append(limit)
        with self.pool.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_task(r) for r in rows]

    def list_tasks_for_message(self, chat_id: str) -> list[ChatTask]:
        """与某条消息关联的全部任务（多对多）。"""
        with self.pool.connection() as conn:
            rows = conn.execute(
                """
                SELECT t.* FROM chat_task t
                JOIN chat_task_message m ON m.task_id = t.task_id
                WHERE m.chat_id = %s
                ORDER BY t.id ASC
                """,
                (chat_id,),
            ).fetchall()
        return [_row_to_task(r) for r in rows]

    def update_task(
        self,
        task_id: str,
        *,
        status: str | None = None,
        output: str | None = None,
        error: str | None = None,
        content: str | None = None,
        title: str | None = None,
        pause_requested: bool | None = None,
        arguments: dict[str, Any] | None = None,
    ) -> None:
        import json

        sets = ["updated_at = %s"]
        params: list[Any] = [_utcnow()]
        if status is not None:
            sets.append("status = %s")
            params.append(status)
        if output is not None:
            sets.append("output = %s")
            params.append(output)
        if error is not None:
            sets.append("error = %s")
            params.append(error)
        if content is not None:
            sets.append("content = %s")
            params.append(content)
        if title is not None:
            sets.append("title = %s")
            params.append(title)
        if pause_requested is not None:
            sets.append("pause_requested = %s")
            params.append(pause_requested)
        if arguments is not None:
            sets.append("arguments = %s")
            params.append(json.dumps(arguments, ensure_ascii=False))
        params.append(task_id)
        with self.pool.connection() as conn:
            conn.execute(
                f"UPDATE chat_task SET {', '.join(sets)} WHERE task_id = %s", params
            )
            conn.commit()

    def request_task_pause(self, task_id: str) -> None:
        """请求暂停：置位 pause_requested，task graph 在下一个步骤边界挂起。"""
        self.update_task(task_id, status="paused", pause_requested=True)

    def request_task_resume(self, task_id: str) -> None:
        """解除暂停标记，恢复执行时由 task graph 重新置为 running。"""
        with self.pool.connection() as conn:
            conn.execute(
                "UPDATE chat_task SET pause_requested = FALSE, status = 'running', "
                "updated_at = %s WHERE task_id = %s",
                (_utcnow(), task_id),
            )
            conn.commit()

    # ---------- chat_task_message ----------
    def link_task_message(self, task_id: str, chat_id: str) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                """
                INSERT INTO chat_task_message (task_id, chat_id)
                VALUES (%s, %s)
                ON CONFLICT (task_id, chat_id) DO NOTHING
                """,
                (task_id, chat_id),
            )
            conn.commit()

    def list_message_ids_for_task(self, task_id: str) -> list[str]:
        with self.pool.connection() as conn:
            rows = conn.execute(
                "SELECT chat_id FROM chat_task_message WHERE task_id = %s ORDER BY id",
                (task_id,),
            ).fetchall()
        return [r["chat_id"] for r in rows]

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


def _row_to_message(row: dict[str, Any]) -> ChatRecord:
    return ChatRecord(
        chat_id=row["chat_id"],
        session_id=row["session_id"],
        user_id=row.get("user_id"),
        role=row["role"],
        content=row["content"],
        created_at=row.get("created_at"),
    )


def _row_to_task(row: dict[str, Any]) -> ChatTask:
    import json

    arguments = row.get("arguments") or {}
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    return ChatTask(
        task_id=row["task_id"],
        session_id=row["session_id"],
        title=row.get("title") or "",
        content=row.get("content") or "",
        entry_skill_id=row["entry_skill_id"],
        arguments=arguments,
        status=row["status"],
        pause_requested=row["pause_requested"],
        output=row.get("output"),
        error=row.get("error"),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


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
