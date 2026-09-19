"""chat_record 表仓储：纯聊天消息时间线（user / assistant）的增查。"""

from __future__ import annotations

from typing import Any

from .base import TableRepo
from .models import ChatRecord


def row_to_message(row: dict[str, Any]) -> ChatRecord:
    return ChatRecord(
        chat_id=row["chat_id"],
        session_id=row["session_id"],
        user_id=row.get("user_id"),
        role=row["role"],
        content=row["content"],
        created_at=row.get("created_at"),
    )


class ChatRecordRepo(TableRepo):
    def insert(self, chat: ChatRecord) -> ChatRecord:
        """写入一条消息，回填 created_at 后返回同一对象。"""
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

    def get(self, chat_id: str) -> ChatRecord | None:
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT * FROM chat_record WHERE chat_id = %s", (chat_id,)
            ).fetchone()
        return row_to_message(row) if row else None

    def list_by_session(self, session_id: str, limit: int = 100) -> list[ChatRecord]:
        """按会话列出消息（时间正序）。"""
        with self.pool.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM chat_record WHERE session_id = %s "
                "ORDER BY id ASC LIMIT %s",
                (session_id, limit),
            ).fetchall()
        return [row_to_message(r) for r in rows]

    def list_recent(self, limit: int = 50) -> list[ChatRecord]:
        """最近消息（时间倒序，用于 CLI 展示）。"""
        with self.pool.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM chat_record ORDER BY id DESC LIMIT %s", (limit,)
            ).fetchall()
        return [row_to_message(r) for r in rows]
