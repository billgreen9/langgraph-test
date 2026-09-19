"""仓储基类：按表划分的仓储共享所属 Database 的连接池。"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from psycopg_pool import ConnectionPool

    from .database import Database


class TableRepo:
    """单表仓储基类。

    子类只关注本表的 SQL 与行转换；连接池生命周期统一由
    :class:`~src.db.database.Database` 管理，仓储通过 pool 属性取用。
    """

    def __init__(self, database: "Database") -> None:
        self._database = database

    @property
    def pool(self) -> "ConnectionPool":
        return self._database.pool
