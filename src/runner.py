"""编排层：RouterGraph → TaskGraph 并发执行 → 结果聚合。

与下游的唯一交互媒介是 chat_task 表（task_id）：
- Router 产出 pending 任务；
- 每个任务在独立 TaskGraph 中执行（thread_id=task_id），互不共享状态；
- 全部终态后聚合产物，以 assistant 消息写回 chat_record。

P1：一条消息对应一个任务；gather 结构已就位，P2 多任务可直接并发。
"""

from __future__ import annotations

import asyncio
import logging
import uuid

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .db import ChatRecord, ChatTask, db
from .skill_runtime.graph import (
    Runtime,
    build_graph,
    get_memory,
    resume_task,
    run_config,
    start_task,
)
from .skill_runtime.loader import SkillLoader
from .skill_runtime.router_graph import build_router_graph

logger = logging.getLogger(__name__)


async def _synthesize(
    rt: Runtime, user_content: str, tasks: list[ChatTask]
) -> str:
    """多任务产物综合为一条回复；LLM 不可用时按任务标题顺序拼接（含失败标注）。"""
    blocks: list[str] = []
    for t in tasks:
        if t.status == "completed":
            blocks.append(f"【{t.title or t.entry_skill_id}】\n{t.output or ''}")
        else:
            blocks.append(
                f"【{t.title or t.entry_skill_id}】（执行失败：{t.error or '未知错误'}）"
            )
    fallback = "\n\n".join(blocks) if blocks else "（任务执行未产生输出）"
    if len(tasks) <= 1 or rt.force_keyword:
        return fallback

    prompt = (
        "以下是为用户请求并行执行的若干任务及其结果，请用中文综合成一条连贯回复，"
        "失败任务需简要说明。直接给结论，不要罗列任务名。\n"
        f"用户请求：{user_content}\n任务结果：\n{fallback}"
    )
    try:
        resp = await rt.llm.ainvoke(
            [SystemMessage(content=prompt), HumanMessage(content=user_content)],
            config={"run_name": "synthesize_final_answer", "tags": ["aggregate"]},
        )
        text = resp.content if isinstance(resp, AIMessage) else str(resp)
        if text and text.strip():
            return text.strip()
    except Exception:
        logger.exception("聚合综合失败，使用任务产物拼接兜底")
    return fallback


async def process_message(
    chat_id: str, *, loader: SkillLoader | None = None
) -> ChatRecord:
    """处理一条 user 消息：路由 → 执行任务 → 聚合写 assistant 消息。"""
    loader = loader or SkillLoader()
    async with Runtime(loader=loader) as rt:
        # 1. 路由：一条消息 -> pending 任务（P1 恰好 1 个）
        router = build_router_graph(rt)
        routed = await router.ainvoke(
            {"chat_id": chat_id},
            run_config(
                rt,
                chat_id,
                run_name=f"router_{chat_id}",
                tags=["router"],
                metadata={"chat_id": chat_id},
            ),
        )

        if routed.get("mode") == "reply":
            from .skill_runtime.match_config import UNKNOWN_ANSWER

            message = await asyncio.to_thread(db.messages.get, chat_id)
            assistant = ChatRecord(
                chat_id=f"chat-{uuid.uuid4().hex[:8]}",
                session_id=message.session_id if message else "default",
                user_id=message.user_id if message else None,
                role="assistant",
                content=routed.get("reply_text") or UNKNOWN_ANSWER,
            )
            await asyncio.to_thread(db.messages.insert, assistant)
            logger.info("消息 %s 直接回复 assistant=%s", chat_id, assistant.chat_id)
            return assistant

        # 2. 执行：与该消息关联的 pending 任务，各自独立 TaskGraph
        tasks = await asyncio.to_thread(
            db.task_messages.list_tasks_for_message, chat_id
        )
        pending = [t for t in tasks if t.status == "pending"]
        if not pending:
            raise RuntimeError(f"消息 {chat_id} 没有可执行的 pending 任务")

        app = build_graph(rt)
        await asyncio.gather(
            *[start_task(rt, app, t.task_id) for t in pending]
        )

        # 3. 回读终态并聚合
        final_tasks = [
            await asyncio.to_thread(db.tasks.get, t.task_id) for t in pending
        ]
        final_tasks = [t for t in final_tasks if t is not None]
        message = await asyncio.to_thread(db.messages.get, chat_id)
        answer = await _synthesize(rt, message.content if message else chat_id,
                                   final_tasks)

        # 4. assistant 消息落库，并与本批任务建立多对多关联
        assistant = ChatRecord(
            chat_id=f"chat-{uuid.uuid4().hex[:8]}",
            session_id=message.session_id if message else final_tasks[0].session_id,
            user_id=message.user_id if message else None,
            role="assistant",
            content=answer,
        )
        await asyncio.to_thread(db.messages.insert, assistant)
        for t in final_tasks:
            await asyncio.to_thread(db.task_messages.link, t.task_id, assistant.chat_id)
        logger.info("消息 %s 处理完成，assistant=%s，任务 %d 个",
                    chat_id, assistant.chat_id, len(final_tasks))
        return assistant


async def resume_one_task(task_id: str, *, loader: SkillLoader | None = None) -> dict:
    """恢复一个被暂停的任务（P3 将扩展为 chat 级广播恢复）。"""
    loader = loader or SkillLoader()
    async with Runtime(loader=loader) as rt:
        app = build_graph(rt)
        await asyncio.to_thread(db.tasks.request_resume, task_id)
        return await resume_task(rt, app, task_id)


async def show_task_memory(
    task_id: str, *, loader: SkillLoader | None = None
) -> dict | None:
    loader = loader or SkillLoader()
    async with Runtime(loader=loader) as rt:
        app = build_graph(rt)
        return await get_memory(rt, app, task_id)
