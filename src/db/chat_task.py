"""chat_task 表仓储：执行任务的增查改与运行控制（暂停/恢复）。"""

from __future__ import annotations

import json
from typing import Any

from .base import TableRepo
from .models import ChatTask, utcnow


def row_to_task(row: dict[str, Any]) -> ChatTask:
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


class ChatTaskRepo(TableRepo):
    def insert(self, task: ChatTask) -> ChatTask:
        """写入一个任务，回填 created_at/updated_at 后返回同一对象。"""
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

    def get(self, task_id: str) -> ChatTask | None:
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT * FROM chat_task WHERE task_id = %s", (task_id,)
            ).fetchone()
        return row_to_task(row) if row else None

    def list(
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
        return [row_to_task(r) for r in rows]

    def update(
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
        """按需更新任务字段（未提供的字段不动）。"""
        sets = ["updated_at = %s"]
        params: list[Any] = [utcnow()]
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

    # ---------- 运行控制 ----------
    def request_pause(self, task_id: str) -> None:
        """请求暂停：置位 pause_requested，TaskGraph 在下一个步骤边界挂起。"""
        self.update(task_id, status="paused", pause_requested=True)

    def request_resume(self, task_id: str) -> None:
        """解除暂停标记；恢复执行时由 TaskGraph 重新置为 running。"""
        with self.pool.connection() as conn:
            conn.execute(
                "UPDATE chat_task SET pause_requested = FALSE, status = 'running', "
                "updated_at = %s WHERE task_id = %s",
                (utcnow(), task_id),
            )
            conn.commit()
