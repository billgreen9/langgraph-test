"""intent_config：分层阈值。已有 key 不被 yaml 覆盖。"""

from __future__ import annotations

from .base import TableRepo
from .models import utcnow


class IntentConfigRepo(TableRepo):
    def get_all(self) -> dict[str, str]:
        with self.pool.connection() as conn:
            rows = conn.execute("SELECT key, value FROM intent_config").fetchall()
        return {r["key"]: str(r["value"]) for r in rows}

    def insert_missing(self, values: dict[str, str]) -> None:
        sql = """
            INSERT INTO intent_config (key, value, updated_at)
            VALUES (%s, %s, %s)
            ON CONFLICT (key) DO NOTHING
        """
        ts = utcnow()
        with self.pool.connection() as conn:
            for key, value in values.items():
                conn.execute(sql, (key, value, ts))
            conn.commit()
