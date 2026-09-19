"""chat_task_message 表仓储：任务与消息的多对多关联。

本表是 RouterGraph / TaskGraph / 聚合层之间的唯一关联媒介：
- 任务 -> 消息（回溯一条任务来自哪些消息，供 P2 跨消息归并使用）
- 消息 -> 任务（执行前取出某条消息拆出的全部任务）
"""

from __future__ import annotations

from .base import TableRepo
from .chat_task import row_to_task
from .models import ChatTask


class ChatTaskMessageRepo(TableRepo):
    def link(self, task_id: str, chat_id: str) -> None:
        """建立任务-消息关联（幂等，重复关联忽略）。"""
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

    def list_message_ids(self, task_id: str) -> list[str]:
        """一个任务关联的全部消息 id（按建立顺序）。"""
        with self.pool.connection() as conn:
            rows = conn.execute(
                "SELECT chat_id FROM chat_task_message WHERE task_id = %s ORDER BY id",
                (task_id,),
            ).fetchall()
        return [r["chat_id"] for r in rows]

    def list_tasks_for_message(self, chat_id: str) -> list[ChatTask]:
        """一条消息关联的全部任务（按任务创建顺序）。"""
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
        return [row_to_task(r) for r in rows]
