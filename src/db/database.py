"""Database：连接池生命周期、建表与按表仓储的组合根。

仓储划分（每张表一个业务模块）：
- :class:`ChatRecordRepo`      → chat_record        纯聊天消息时间线
- :class:`ChatTaskRepo`        → chat_task          执行任务（运行控制在此）
- :class:`ChatTaskMessageRepo` → chat_task_message  任务-消息多对多关联
- :class:`SkillRegistryRepo`   → skill_registry     技能注册表

模块级单例 ``db`` 供后台线程与主程序共享。
"""

from __future__ import annotations

import threading

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .base import TableRepo
from .chat_record import ChatRecordRepo
from .chat_task import ChatTaskRepo
from .chat_task_message import ChatTaskMessageRepo
from .schema import LEGACY_CHECK_SQL, SCHEMA_SQL
from .skill_registry import SkillRegistryRepo
from ..config import settings

__all__ = [
    "Database",
    "TableRepo",
    "ChatRecordRepo",
    "ChatTaskRepo",
    "ChatTaskMessageRepo",
    "SkillRegistryRepo",
    "db",
]


class Database:
    """同步 PostgreSQL 访问封装（线程安全，连接池可被后台线程共享）。"""

    def __init__(self, dsn: str | None = None) -> None:
        self.dsn = dsn or settings.pg_dsn
        self._pool: ConnectionPool | None = None
        self._lock = threading.Lock()
        # 按表组合仓储；仓储通过 Database.pool 取用连接
        self.messages = ChatRecordRepo(self)
        self.tasks = ChatTaskRepo(self)
        self.task_messages = ChatTaskMessageRepo(self)
        self.skills = SkillRegistryRepo(self)

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
            legacy = conn.execute(LEGACY_CHECK_SQL).fetchone()
            if legacy and legacy["is_legacy"]:
                conn.execute("DROP TABLE IF EXISTS chat_records CASCADE")
            conn.execute(SCHEMA_SQL)
            conn.commit()


db = Database()
