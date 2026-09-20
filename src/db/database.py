"""Database：连接池、建表、按表仓储组合根。"""

from __future__ import annotations

import logging
import threading

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .base import TableRepo
from .chat_record import ChatRecordRepo
from .chat_task import ChatTaskRepo
from .chat_task_message import ChatTaskMessageRepo
from .intent_config import IntentConfigRepo
from .intent_math import IntentMathRepo
from .schema import INTENT_SCHEMA_SQL, INTENT_VECTOR_INDEX_SQL, LEGACY_CHECK_SQL, SCHEMA_SQL
from .skill_registry import SkillRegistryRepo
from ..config import settings

logger = logging.getLogger(__name__)

__all__ = [
    "Database",
    "TableRepo",
    "ChatRecordRepo",
    "ChatTaskRepo",
    "ChatTaskMessageRepo",
    "SkillRegistryRepo",
    "IntentMathRepo",
    "IntentConfigRepo",
    "db",
]


class Database:
    def __init__(self, dsn: str | None = None) -> None:
        self.dsn = dsn or settings.pg_dsn
        self._pool: ConnectionPool | None = None
        self._lock = threading.Lock()
        self.messages = ChatRecordRepo(self)
        self.tasks = ChatTaskRepo(self)
        self.task_messages = ChatTaskMessageRepo(self)
        self.skills = SkillRegistryRepo(self)
        self.intents = IntentMathRepo(self)
        self.intent_config = IntentConfigRepo(self)

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

    def init_schema(self) -> None:
        with self.pool.connection() as conn:
            legacy = conn.execute(LEGACY_CHECK_SQL).fetchone()
            if legacy and legacy["is_legacy"]:
                conn.execute("DROP TABLE IF EXISTS chat_records CASCADE")
            conn.execute(SCHEMA_SQL)
            try:
                conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            except Exception:
                logger.warning("无法启用 pgvector，语义召回将跳过")
            try:
                conn.execute(INTENT_SCHEMA_SQL)
            except Exception:
                logger.warning("vector 类型不可用，intent_math.msg_embedding 降级为 float8[]")
                conn.execute(
                    INTENT_SCHEMA_SQL.replace(
                        "msg_embedding  vector", "msg_embedding  float8[]"
                    )
                )
            conn.execute(
                "ALTER TABLE intent_math ADD COLUMN IF NOT EXISTS skill_level "
                "INTEGER NOT NULL DEFAULT 0"
            )
            try:
                conn.execute(INTENT_VECTOR_INDEX_SQL)
            except Exception:
                logger.warning("intent_math 向量索引创建失败")
            conn.commit()


db = Database()
