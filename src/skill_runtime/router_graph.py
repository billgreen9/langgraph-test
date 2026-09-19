"""意图路由 LangGraph（RouterGraph）。

职责单一：读取一条 user 消息（chat_record）→ 用 skills/intents/*.md 意图技能
识别意图 → 生成一个 pending 的 chat_task（entry_skill_id 指向一级业务技能）
并写入消息-任务关联。本图不执行任何业务技能。

P1：一条消息固定产出一个任务；多意图拆分与跨消息归并在 P2 引入。

    START → route_message → END

checkpoint thread_id = chat_id（消息 id）。
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from ..db import ChatTask, db
from .graph import Runtime, _runtime  # 复用同一运行时（LLM/连接池/加载器）
from .intent_loader import IntentLoader, IntentSpec

logger = logging.getLogger(__name__)


class RouterState(TypedDict, total=False):
    chat_id: str
    task_id: str
    entry_skill_id: str
    intent_id: str


class LlmIntentChoice(BaseModel):
    intent_id: str
    reason: str = ""


# ---------- 意图匹配 ----------
def _intent_score(spec: IntentSpec, text: str) -> int:
    low = text.lower()
    return sum(low.count(k.lower()) for k in spec.keywords if k)


def _fallback_intent(intents: list[IntentSpec], text: str) -> IntentSpec:
    scored = [(s, _intent_score(s, text)) for s in intents]
    best = max((score for _, score in scored), default=0)
    if best == 0:
        chat = next((s for s, _ in scored if s.intent_id == "chat"), None)
        return chat or intents[0]
    positives = sorted(
        ((s, score) for s, score in scored if score > 0),
        key=lambda item: (-item[1], item[0].order),
    )
    return positives[0][0]


def match_intent(
    intents: list[IntentSpec], text: str, rt: Runtime | None = None
) -> IntentSpec:
    """关键词命中优先；无命中时让 LLM 基于意图说明裁决；失败则兜底。"""
    if not intents:
        raise ValueError("skills/intents 下没有任何意图技能")
    if len(intents) == 1:
        return intents[0]
    if any(_intent_score(s, text) > 0 for s in intents):
        return _fallback_intent(intents, text)
    if rt is not None and not rt.force_keyword:
        listing = "\n".join(
            f"- {s.intent_id}：{s.name}。{s.body}" for s in intents
        )
        prompt = (
            "你是意图路由器。根据用户消息，从候选意图中选择唯一最合适的一个，"
            "只输出 intent_id，不要执行任务。\n候选意图：\n"
            f"{listing}\n用户消息：{text}"
        )
        try:
            choice = rt.llm.with_structured_output(LlmIntentChoice).invoke(
                [SystemMessage(content=prompt), HumanMessage(content=text)]
            )
            selected = next(
                (s for s in intents if s.intent_id == choice.intent_id), None
            )
            if selected is not None:
                logger.info("LLM 意图选择 %s（%s）", selected.intent_id, choice.reason)
                return selected
        except Exception:
            logger.exception("LLM 意图路由失败，使用关键词/默认兜底")
    return _fallback_intent(intents, text)


# ---------- 图节点 ----------
async def route_message_node(
    state: RouterState, config: RunnableConfig
) -> dict[str, Any]:
    """读取消息 → 匹配意图 → 落一个 pending 任务并关联消息。"""
    rt = _runtime(config)
    chat_id = state["chat_id"]

    message = await asyncio.to_thread(db.get_message, chat_id)
    if message is None:
        raise ValueError(f"聊天消息不存在：{chat_id}")

    intents = IntentLoader().load_all()
    # 入口技能闭环校验：每个 intent.entry_skill 必须是存在的一级业务技能
    level1 = await rt.level1_candidates()
    valid_ids = {m.skill_id for m in level1}
    intents = [s for s in intents if s.entry_skill in valid_ids]
    if not intents:
        raise ValueError(
            f"没有 entry_skill 命中一级业务技能 {sorted(valid_ids)} 的有效意图"
        )

    spec = match_intent(intents, message.content, rt)
    task = ChatTask(
        task_id=f"task-{uuid.uuid4().hex[:8]}",
        session_id=message.session_id,
        title=spec.name,
        content=message.content,
        entry_skill_id=spec.entry_skill,
        status="pending",
    )
    await asyncio.to_thread(db.insert_task, task)
    await asyncio.to_thread(db.link_task_message, task.task_id, chat_id)
    logger.info("[router] chat_id=%s 意图=%s -> task_id=%s entry=%s",
                chat_id, spec.intent_id, task.task_id, spec.entry_skill)
    return {
        "chat_id": chat_id,
        "task_id": task.task_id,
        "intent_id": spec.intent_id,
        "entry_skill_id": spec.entry_skill,
    }


def build_router_graph(rt: Runtime):
    g = StateGraph(RouterState)
    g.add_node("route_message", route_message_node)
    g.add_edge(START, "route_message")
    g.add_edge("route_message", END)
    return g.compile(checkpointer=rt.saver)
